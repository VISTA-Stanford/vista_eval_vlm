"""OpenAI models on the Stanford Health Care AI Hub (Azure OpenAI deployments), behind the adapter
interface run_bq drives. Plan: planning/vista_eval_vlm/vlm-radiology-progression-codex.md.

  * one chat-completions call per row to
    ``https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/<model>/chat/completions``,
    header ``api-key``; the key comes from ``AI_HUB_API_KEY`` in the environment, else from the
    config's ``aihub.key_file`` (a ``AI_HUB_API_KEY=...`` line; default ``~/.config/vista/aihub.env``);
  * the CT slices go inline as JPEG data URLs, before the text;
  * the answer is forced onto the task's options by a strict JSON schema (``response_format``);
    an answer off the list or an empty one is an error, not a guess; no logprobs (Astra refuses them);
  * the hub's firewall answers an over-size request with an HTML "Request Rejected" page and HTTP
    200 (from phil-sllm-01 anything over ~48 KB, 10-15): that comes back as an error naming the body
    size and is not retried; Azure itself takes at most 50 images per request, checked before sending;
  * every readable answer is cached under a hash of the exact request; 429 / 5xx / connection
    errors retry with backoff; measured usage is returned for every row.

No weights: ``load()`` only reads the key, so it runs on a CPU box.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import BaseVLMAdapter
from .codex_exec import encode_image

DEFAULT_BASE = "https://aihubapi.stanfordhealthcare.org/azure-openai/deployments"
DEFAULT_API_VERSION = "2025-04-01-preview"
MAX_IMAGES = 50  # Azure: "maximum allowed: 50" (probed 10-15)
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
USAGE_KEYS = ("prompt_tokens", "cached_prompt_tokens", "cache_write_tokens", "output_tokens", "thinking_tokens",
              "total_tokens", "image_count", "request_bytes", "seconds")


def build_request(question, encoded_images, choices=None, max_output_tokens=4096, reasoning_effort=None,
                  image_detail=None, mime="image/jpeg"):
    """The chat-completions body for one row. Pure (no network): unit-tested."""
    content = []
    for b in encoded_images or []:
        url = {"url": f"data:{mime};base64," + base64.b64encode(b).decode("ascii")}
        if image_detail:
            url["detail"] = image_detail
        content.append({"type": "image_url", "image_url": url})
    content.append({"type": "text", "text": str(question)})
    body = {"messages": [{"role": "user", "content": content}], "max_completion_tokens": int(max_output_tokens)}
    if choices:
        # A list forces one `answer`; a dict {field: options} (a panel) forces one field per question.
        fields = choices if isinstance(choices, dict) else {"answer": choices}
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": list(fields),
            "properties": {k: {"type": "string", "enum": [str(c) for c in v]} for k, v in fields.items()}}}}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def build_parts_request(parts, choices=None, max_output_tokens=4096, reasoning_effort=None, image_detail=None,
                        cache_breakpoint=False, cache_key=None):
    """Chat-completions body from ordered parts ({"type": "text", "text"} / {"type": "image", "data", "mime"}),
    so slice labels can sit between the images (the eval set's recipes). Same forcing as ``build_request``.

    ``cache_breakpoint`` marks the last image as the end of the reusable prefix (GPT-5.6+ prompt-cache breakpoints),
    so a scan's other questions read its images from the cache whatever their question text; ``cache_key`` sets
    ``prompt_cache_key`` so those calls are routed to the same cache (Azure prompt-caching guide, 2026-08)."""
    body = build_request("", [], choices, max_output_tokens, reasoning_effort, image_detail)
    content = []
    for p in parts:
        if p["type"] == "text":
            content.append({"type": "text", "text": str(p["text"])})
        else:
            url = {"url": f"data:{p.get('mime', 'image/jpeg')};base64," + base64.b64encode(p["data"]).decode("ascii")}
            if image_detail:
                url["detail"] = image_detail
            content.append({"type": "image_url", "image_url": url})
    if cache_breakpoint:
        last = max((i for i, c in enumerate(content) if c["type"] == "image_url"), default=None)
        if last is not None:
            content[last]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    body["messages"] = [{"role": "user", "content": content}]
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def scan_cache_key(parts) -> str:
    """A routing key shared by every call on the same images: a hash of the image bytes (no identifiers)."""
    h = hashlib.sha256()
    for p in parts:
        if p["type"] == "image":
            h.update(p["data"])
    return "scan-" + h.hexdigest()[:24]


def parse_response(status, content_type, text, choices=None):
    """(answer, usage, error, retryable) from one HTTP reply."""
    usage = {k: None for k in USAGE_KEYS if k not in ("image_count", "request_bytes")}
    if "Request Rejected" in (text or "") or "text/html" in (content_type or ""):
        return None, usage, "AI Hub gateway rejected the request (its firewall page, not the model)", False
    if status != 200:
        return None, usage, f"HTTP {status}: {(text or '')[:300]}", status in _RETRY_STATUS
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None, usage, f"reply is not JSON: {(text or '')[:200]!r}", False
    u = body.get("usage") or {}
    usage = {"prompt_tokens": u.get("prompt_tokens"),
             "cached_prompt_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
             "cache_write_tokens": (u.get("prompt_tokens_details") or {}).get("cache_write_tokens"),
             "output_tokens": u.get("completion_tokens"),
             "thinking_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
             "total_tokens": u.get("total_tokens")}
    ch = (body.get("choices") or [{}])[0]
    msg = (ch.get("message") or {}).get("content") or ""
    if not msg.strip():
        return None, usage, f"empty answer (finish_reason={ch.get('finish_reason')})", False
    try:
        parsed = json.loads(msg) if choices else None
        answer = msg.strip() if not choices else (parsed if isinstance(choices, dict) else str(parsed.get("answer", "")).strip())
    except (json.JSONDecodeError, AttributeError):
        return None, usage, f"answer is not the schema's JSON: {msg[:100]!r}", False
    if isinstance(choices, dict):
        # Every question answered, each on its own options; kept as one JSON string, split per task at scoring.
        if not isinstance(answer, dict):
            return None, usage, f"answer is not one object: {msg[:100]!r}", False
        bad = {k: answer.get(k) for k, opts in choices.items() if str(answer.get(k, "")).strip() not in [str(o) for o in opts]}
        if bad:
            return None, usage, f"{len(bad)} answers missing or off their options: {json.dumps(bad)[:150]}", False
        return json.dumps({k: str(answer[k]).strip() for k in choices}, sort_keys=True), usage, None, False
    if choices and answer not in [str(c) for c in choices]:
        return None, usage, f"answer {answer[:100]!r} not in options", False
    return answer, usage, None, False


def read_key(key_file):
    key = os.environ.get("AI_HUB_API_KEY")
    if key:
        return key.strip()
    p = Path(os.path.expanduser(key_file))
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip().startswith("AI_HUB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"no AI Hub key: set AI_HUB_API_KEY or put AI_HUB_API_KEY=... in {key_file}")


class AIHubOpenAIAdapter(BaseVLMAdapter):
    config_block = "aihub"
    context_window = 272_000
    input_token_limit = 272_000
    # Measured 10-15: ~308 prompt tokens per 512 px image at default detail.
    image_token_estimate = 310

    def __init__(self, model_name, device="auto", cache_dir=None):
        super().__init__(model_name, device, cache_dir)
        self.base_url = DEFAULT_BASE
        self.api_version = DEFAULT_API_VERSION
        self.key_file = "~/.config/vista/aihub.env"
        self.reasoning_effort = None
        self.image_detail = None
        self.cache_breakpoint = False  # recipe setting: explicit prompt-cache breakpoint after the last image
        self.prompt_cache_key = False  # recipe setting: route a scan's calls together (key = hash of its images)
        self.jpeg_quality = 95
        self.concurrency = 1
        self.max_retries = 6
        self.timeout_s = 600
        self.max_total_tokens = None
        self.tokens_used = 0
        self.api_calls = 0
        self._key = None
        self._lock = threading.Lock()
        self.response_cache = Path(cache_dir or ".") / "aihub_openai_responses"

    def configure(self, opts):
        opts = opts or {}
        for k in ("base_url", "api_version", "key_file", "reasoning_effort", "image_detail"):
            setattr(self, k, opts.get(k, getattr(self, k)))
        for k in ("jpeg_quality", "concurrency", "max_retries", "timeout_s"):
            setattr(self, k, int(opts.get(k, getattr(self, k))))
        self.max_total_tokens = opts.get("max_total_tokens", self.max_total_tokens)
        for k in ("cache_breakpoint", "prompt_cache_key"):
            setattr(self, k, bool(opts.get(k, getattr(self, k))))

    def _parts_body(self, msg, max_new_tokens, choices):
        return build_parts_request(msg["parts"], choices, max_new_tokens, self.reasoning_effort, self.image_detail,
                                   cache_breakpoint=self.cache_breakpoint,
                                   cache_key=scan_cache_key(msg["parts"]) if self.prompt_cache_key else None)

    def load(self):
        self._key = read_key(self.key_file)
        self.response_cache.mkdir(parents=True, exist_ok=True)
        return self, None

    def count_text_tokens(self, text):
        return len(str(text)) // 4

    def create_template(self, item):
        images = item.get("image")
        if images is not None and not isinstance(images, list):
            images = [images]
        return {"question": item.get("question", ""), "images": [im for im in (images or []) if im is not None]}

    def prepare_inputs(self, messages, processor, model):
        return list(messages)

    def stack_inputs(self, input_list, model):
        return input_list

    def _url(self):
        return f"{self.base_url}/{self.model_name}/chat/completions?api-version={self.api_version}"

    def _post(self, data):
        import requests
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._lock:
                    self.api_calls += 1
                r = requests.post(self._url(), data=data, timeout=self.timeout_s,
                                  headers={"api-key": self._key, "Content-Type": "application/json"})
                return r.status_code, r.headers.get("content-type", ""), r.text
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(120, 2 ** attempt) + random.random())
        return None, "", f"gave up after {self.max_retries + 1} tries: {last}"

    def _one(self, msg, max_new_tokens, choices):
        if "parts" in msg:  # eval-set message: ordered parts, images already encoded by the pack
            encoded = [p["data"] for p in msg["parts"] if p["type"] == "image"]
        else:
            encoded = None
        n_img = len(encoded) if encoded is not None else len(msg["images"])
        if n_img > MAX_IMAGES:
            return {"text": "", "error": f"{n_img} images; Azure takes at most {MAX_IMAGES} per request"}
        if encoded is not None:
            body = self._parts_body(msg, max_new_tokens, choices)
        else:
            encoded = [encode_image(im, "jpeg", self.jpeg_quality) for im in msg["images"]]
            body = build_request(msg["question"], encoded, choices, max_new_tokens, self.reasoning_effort, self.image_detail)
        data = json.dumps(body)
        key = hashlib.sha256((self.model_name + "\n" + data).encode()).hexdigest()
        path = self.response_cache / key[:2] / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            return {"text": rec["answer"], "cumulative_logprob": None, "log_probs": None,
                    "usage": {k: rec["usage"].get(k) for k in USAGE_KEYS}, "cached": True,
                    "request_key": key, "raw": rec.get("raw")}
        with self._lock:
            if self.max_total_tokens is not None and self.tokens_used >= int(self.max_total_tokens):
                return {"text": "", "error": f"token cap {self.max_total_tokens} reached ({self.tokens_used} used)",
                        "request_key": key}
        answer, usage, err, text = None, {}, None, None
        t0 = time.monotonic()  # wall time for this scan, retries included
        for attempt in range(self.max_retries + 1):
            status, ctype, text = self._post(data)
            if status is None:
                err = text
                break
            answer, usage, err, retryable = parse_response(status, ctype, text, choices)
            if err is None or not retryable:
                break
            time.sleep(min(120, 2 ** attempt) + random.random())
        usage = dict(usage or {})
        usage["image_count"] = len(encoded)
        usage["request_bytes"] = len(data)
        usage["seconds"] = round(time.monotonic() - t0, 1)
        usage = {k: usage.get(k) for k in USAGE_KEYS}
        with self._lock:
            self.tokens_used += int(usage.get("total_tokens") or 0)
        try:  # the full reply body, kept for re-reading later (eval-set attempt log and the cache)
            raw = json.loads(text) if text else None
        except (json.JSONDecodeError, TypeError):
            raw = (text or "")[:20000]
        if err:
            if "rejected" in err:
                err += f" — request was {len(data):,} bytes"
            return {"text": "", "error": err, "usage": usage, "cached": False, "request_key": key, "raw": raw}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"answer": answer, "usage": usage, "model": self.model_name, "raw": raw}))
        tmp.replace(path)
        return {"text": answer, "cumulative_logprob": None, "log_probs": None, "usage": usage, "cached": False,
                "request_key": key, "raw": raw}

    def is_answered(self, msg, max_new_tokens, choices):
        """True when this eval-set call's answer is already in the disk cache (same key as ``_one``)."""
        body = self._parts_body(msg, max_new_tokens, choices)
        key = hashlib.sha256((self.model_name + "\n" + json.dumps(body)).encode()).hexdigest()
        return (self.response_cache / key[:2] / f"{key}.json").exists()

    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        msgs = inputs if isinstance(inputs, list) else [inputs]
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            return list(ex.map(lambda m: self._one(m, max_new_tokens, constrained_choices), msgs))
