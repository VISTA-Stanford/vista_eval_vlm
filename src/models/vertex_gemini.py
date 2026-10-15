"""Vertex AI Gemini adapter (REST ``generateContent``), behind the same adapter interface
as the vLLM models so ``run_bq.TaskOrchestrator`` drives it unchanged.

What it does (plan: planning/vista_eval_vlm/vlm-tumor-board-questions-v1_6.md, step 5):
  * one ``generateContent`` call per row on the ``global`` endpoint (Gemini 3.1 Pro answers
    only there), user/VM ADC with the ``x-goog-user-project`` header;
  * the answer is forced onto the task's option list (``responseMimeType: text/x.enum``),
    temperature 0; no probability is asked for (Gemini 3.1 Pro refuses token logprobs);
  * CT slices go inline as PNG, before the text, the order the vLLM adapters use;
  * every response is cached on disk under a hash of the exact request, so a rerun only
    calls the API for rows it has not answered (an edited prompt is a new key);
  * 429 / 5xx / connection errors retry with exponential backoff; any other failure comes
    back as ``{"error": ...}`` and run_bq lists the row in the errors file;
  * measured usage (prompt, output, thinking tokens) is returned for every row, cached or not.

No weights: ``load()`` only fetches credentials, so it runs on the CPU box.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import BaseVLMAdapter

DEFAULT_PROJECT = "som-nero-plevriti-deidbdf"
DEFAULT_LOCATION = "global"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# Same keys on every row, so the appended results CSV keeps one column set.
USAGE_KEYS = ("prompt_tokens", "output_tokens", "thinking_tokens", "total_tokens", "image_count")


def _endpoint(project, location, model, method):
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:{method}"


def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_request(question, images, choices=None, max_output_tokens=8192,
                  media_resolution=None, thinking_level=None):
    """The ``generateContent`` body for one row. Pure (no network): unit-tested."""
    parts = [{"inlineData": {"mimeType": "image/png", "data": _png_b64(im)}} for im in (images or [])]
    parts.append({"text": str(question)})
    gen = {"temperature": 0.0, "maxOutputTokens": int(max_output_tokens)}
    if choices:
        gen["responseMimeType"] = "text/x.enum"
        gen["responseSchema"] = {"type": "STRING", "enum": [str(c) for c in choices]}
    if media_resolution:
        gen["mediaResolution"] = media_resolution
    if thinking_level:
        gen["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}


def parse_response(body, choices=None):
    """(answer_text, usage, error) from a ``generateContent`` response body.

    An answer outside ``choices`` (or an empty one, e.g. the output budget ran out while
    thinking) is an error, not a guess.
    """
    um = body.get("usageMetadata", {}) or {}
    usage = {
        "prompt_tokens": um.get("promptTokenCount"),
        "output_tokens": um.get("candidatesTokenCount"),
        "thinking_tokens": um.get("thoughtsTokenCount"),
        "total_tokens": um.get("totalTokenCount"),
    }
    cands = body.get("candidates") or []
    if not cands:
        fb = body.get("promptFeedback", {})
        return None, usage, f"no candidates (promptFeedback={json.dumps(fb)[:300]})"
    cand = cands[0]
    texts = [p.get("text", "") for p in (cand.get("content", {}) or {}).get("parts", []) or []
             if not p.get("thought")]
    text = "".join(texts).strip()
    if not text:
        return None, usage, f"empty answer (finishReason={cand.get('finishReason')})"
    if choices and text not in [str(c) for c in choices]:
        return None, usage, f"answer {text[:100]!r} not in options (finishReason={cand.get('finishReason')})"
    return text, usage, None


class VertexGeminiAdapter(BaseVLMAdapter):
    context_window = 1_048_576
    input_token_limit = 1_048_576
    # Measured 2026-10-14: a 64 px PNG cost 1,089 prompt tokens on 3.1 Pro. Used only for
    # the pre-call overflow check; the bill uses the measured usage.
    image_token_estimate = 1100
    # Below this many characters a prompt cannot reach the limit (>= 1 char per token),
    # so the overflow check skips the countTokens call.
    _COUNT_EXACT_ABOVE_CHARS = 1_000_000

    def __init__(self, model_name, device="auto", cache_dir=None):
        super().__init__(model_name, device, cache_dir)
        self.project = DEFAULT_PROJECT
        self.location = DEFAULT_LOCATION
        self.concurrency = 8
        self.max_retries = 8
        self.media_resolution = None
        self.thinking_level = None
        self.prices = None
        self.max_spend_usd = None
        self.spent_usd = 0.0
        self._inflight = 0
        self._paid_calls = 0
        self.response_cache = Path(cache_dir or ".") / "vertex_gemini_responses"
        self._creds = None
        self._lock = threading.Lock()
        self.api_calls = 0

    def configure(self, opts):
        """Run settings from the config's ``vertex:`` block (run_bq calls this)."""
        opts = opts or {}
        self.project = opts.get("project", self.project)
        self.location = opts.get("location", self.location)
        self.concurrency = int(opts.get("concurrency", self.concurrency))
        self.max_retries = int(opts.get("max_retries", self.max_retries))
        self.media_resolution = opts.get("media_resolution", self.media_resolution)
        self.thinking_level = opts.get("thinking_level", self.thinking_level)
        self.prices = opts.get("price_per_mtok", self.prices)
        self.max_spend_usd = opts.get("max_spend_usd", self.max_spend_usd)

    def call_cost_usd(self, usage):
        """Dollars for one call's measured usage (thinking bills as output)."""
        p = self.prices
        if not p:
            return 0.0
        pt = usage.get("prompt_tokens") or 0
        ot = (usage.get("output_tokens") or 0) + (usage.get("thinking_tokens") or 0)
        if pt > p.get("long_context_over", 200000):
            return (pt * p["input_long"] + ot * p["output_long"]) / 1e6
        return (pt * p["input"] + ot * p["output"]) / 1e6

    # -- auth / transport ---------------------------------------------------------
    def _token(self):
        import google.auth
        import google.auth.transport.requests
        with self._lock:
            if self._creds is None:
                self._creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not self._creds.valid:
                self._creds.refresh(google.auth.transport.requests.Request())
            return self._creds.token

    def _post(self, method, body):
        import requests
        url = _endpoint(self.project, self.location, self.model_name, method)
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                headers = {"Authorization": f"Bearer {self._token()}",
                           "x-goog-user-project": self.project,
                           "Content-Type": "application/json"}
                with self._lock:
                    self.api_calls += 1
                r = requests.post(url, headers=headers, data=json.dumps(body), timeout=600)
                if r.status_code == 200:
                    return r.json(), None
                last = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code not in _RETRY_STATUS:
                    return None, last
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(120, 2 ** attempt) + random.random())
        return None, f"gave up after {self.max_retries + 1} tries: {last}"

    # -- adapter interface --------------------------------------------------------
    def load(self):
        self._token()  # fail now, before any data work, if credentials are missing
        self.response_cache.mkdir(parents=True, exist_ok=True)
        return self, None

    def count_text_tokens(self, text):
        """Prompt-token count for the overflow check: exact (countTokens) only when long."""
        text = str(text)
        if len(text) <= self._COUNT_EXACT_ABOVE_CHARS:
            return len(text) // 4
        body, err = self._post("countTokens", {"contents": [{"role": "user", "parts": [{"text": text}]}]})
        if err:
            raise RuntimeError(f"countTokens failed: {err}")
        return int(body["totalTokens"])

    def create_template(self, item):
        images = item.get("image")
        if images is not None and not isinstance(images, list):
            images = [images]
        return {"question": item.get("question", ""), "images": [im for im in (images or []) if im is not None]}

    def prepare_inputs(self, messages, processor, model):
        return list(messages)

    def stack_inputs(self, input_list, model):
        return input_list

    def _cache_path(self, body):
        key = hashlib.sha256((self.model_name + "\n" + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        return self.response_cache / key[:2] / f"{key}.json"

    def _one(self, msg, max_new_tokens, choices):
        body = build_request(msg["question"], msg["images"], choices, max_new_tokens,
                             self.media_resolution, self.thinking_level)
        path = self._cache_path(body)
        cached = path.exists()
        if cached:
            resp = json.loads(path.read_text())
        else:
            # Hard spend cap (config `vertex.max_spend_usd`): no new call once measured spend
            # this run reaches it; the row is listed in the errors file and a rerun picks it up.
            # Calls already in flight count at the mean measured cost so far (first call: $0.25),
            # so concurrent calls can't carry spend past the cap (stage 1 overshot by $6 without this).
            with self._lock:
                mean = self.spent_usd / self._paid_calls if self._paid_calls else 0.25
                committed = self.spent_usd + (self._inflight + 1) * mean
                if self.max_spend_usd is not None and committed > float(self.max_spend_usd):
                    return {"text": "", "error": f"spend cap ${self.max_spend_usd} reached (spent ${self.spent_usd:.2f}, {self._inflight} calls in flight)"}
                self._inflight += 1
            try:
                resp, err = self._post("generateContent", body)
            finally:
                with self._lock:
                    self._inflight -= 1
            if err:
                return {"text": "", "error": err}
        text, usage, err = parse_response(resp, choices)
        if not cached and err is None:
            # Cache only a readable answer, so an unreadable one is asked again on the next run.
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(resp))
            tmp.replace(path)
        if not cached:
            with self._lock:
                self.spent_usd += self.call_cost_usd(usage)
                self._paid_calls += 1
        usage["image_count"] = len(msg["images"])
        usage = {k: usage.get(k) for k in USAGE_KEYS}
        if err:
            return {"text": "", "error": err, "usage": usage, "cached": cached}
        return {"text": text, "cumulative_logprob": None, "log_probs": None, "usage": usage, "cached": cached}

    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        msgs = inputs if isinstance(inputs, list) else [inputs]
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            return list(ex.map(lambda m: self._one(m, max_new_tokens, constrained_choices), msgs))
