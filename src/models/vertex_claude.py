"""Claude (Opus 5.5) on Vertex AI (``rawPredict``, location global), for the eval set: one plain call per question,
no tools, no agent loop, so it sits next to Argon and Astra as a vision model.

Request shape (Claude API reference, checked 2026-10-31):
  * images first (base64 PNG), each after its "Image n:" label, the question last, all in one user turn;
  * ``cache_control`` on the last prefix part (the last image), so a scan's later questions read the images from
    Anthropic's prompt cache (5 min). The eval-set runner already sends each scan's lead question first and the rest
    after it finishes (``evalset.run.ask_chunk``), which is what this cache needs;
  * the answer is forced with structured output (``output_config.format``: JSON {"answer": "yes"|"no"}). Opus 5.5
    rejects forced ``tool_choice``, ``temperature`` and assistant prefill, and its thinking cannot be switched off, so
    the recipe sets ``effort`` (Argon's ``thinking_level: low`` counterpart). Thinking tokens bill as output;
  * no token log-probabilities (refused by Opus 5.5 on Vertex, measured 10-26).

Raw HTTP via ``requests`` rather than the ``anthropic`` SDK on purpose: it reuses the Gemini adapter's VM login,
retries, request-hash answer cache and hard spend cap unchanged, and the repo's API environment has no SDK.
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .vertex_gemini import PARTS_USAGE_KEYS, VertexGeminiAdapter

ANTHROPIC_VERSION = "vertex-2023-10-16"
#: Usage columns for Claude rows: the Gemini set plus tokens written to the prompt cache (billed at 1.25x input).
CLAUDE_USAGE_KEYS = PARTS_USAGE_KEYS + ("cache_write_tokens",)


def _endpoint(project, location, model):
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict"


def answer_schema(choices, reason_field=False):
    props = {}
    if reason_field:
        props["reasoning"] = {"type": "string", "description": "Brief reasoning from the images, written before the answer."}
    props["answer"] = {"type": "string", "enum": [str(c) for c in choices]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def claude_content(parts, prefix_len=0, cache=True):
    """Eval-set parts -> Messages API content blocks, order kept; ``cache_control`` on the last prefix block."""
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": str(p["text"])})
        else:
            out.append({"type": "image", "source": {"type": "base64", "media_type": p.get("mime", "image/png"),
                                                    "data": base64.b64encode(p["data"]).decode("ascii")}})
    if cache and prefix_len:
        out[prefix_len - 1]["cache_control"] = {"type": "ephemeral"}
    return out


def build_parts_request(parts, choices=None, max_output_tokens=4096, effort=None, prefix_len=0, cache=True,
                        reason_field=False):
    """rawPredict body for one question. Pure (no network): unit-tested."""
    body = {"anthropic_version": ANTHROPIC_VERSION, "max_tokens": int(max_output_tokens),
            "messages": [{"role": "user", "content": claude_content(parts, prefix_len, cache)}]}
    oc = {}
    if effort:
        oc["effort"] = effort
    if choices:
        oc["format"] = {"type": "json_schema", "schema": answer_schema(choices, reason_field)}
    if oc:
        body["output_config"] = oc
    return body


def parse_response(body, choices=None, reason_field=False):
    """(answer_text, usage, error). The text returned is JSON {"answer": ...} (plus "reasoning" with
    ``reason_field``), which ``evalset.score.parse_answer`` reads. An answer outside ``choices`` is an error."""
    u = body.get("usage", {}) or {}
    inp, cw, cr = (u.get("input_tokens") or 0), (u.get("cache_creation_input_tokens") or 0), (u.get("cache_read_input_tokens") or 0)
    out_all = u.get("output_tokens") or 0
    think = (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    # prompt_tokens counts every prompt token (uncached + written + read), as the Gemini rows do.
    usage = {"prompt_tokens": inp + cw + cr, "cached_tokens": cr, "cache_write_tokens": cw,
             "output_tokens": out_all - think, "thinking_tokens": think, "total_tokens": inp + cw + cr + out_all}
    stop = body.get("stop_reason")
    if stop == "refusal":
        sd = body.get("stop_details") or {}
        return None, usage, f"refusal (category={sd.get('category')}): {str(sd.get('explanation'))[:200]}"
    text = "".join(b.get("text", "") for b in body.get("content", []) or [] if b.get("type") == "text").strip()
    if not text:
        return None, usage, f"empty answer (stop_reason={stop})"
    if not choices:
        return text, usage, None
    try:
        v = json.loads(text)
        answer = str(v.get("answer", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return None, usage, f"answer is not the schema's JSON: {text[:100]!r} (stop_reason={stop})"
    if answer not in [str(c) for c in choices]:
        return None, usage, f"answer {answer[:100]!r} not in options (stop_reason={stop})"
    rec = {"reasoning": str(v.get("reasoning", "")), "answer": answer} if reason_field else {"answer": answer}
    return json.dumps(rec), usage, None


class VertexClaudeAdapter(VertexGeminiAdapter):
    """Eval-set calls only (``parts`` messages). Login, retries, answer cache and spend cap come from the Gemini adapter."""

    context_window = 1_000_000
    input_token_limit = 1_000_000

    def __init__(self, model_name, device="auto", cache_dir=None):
        super().__init__(model_name, device, cache_dir)
        self.effort = None
        self.prompt_cache = True
        self.response_cache = self.response_cache.parent / "vertex_claude_responses"
        self._lock = threading.Lock()

    def configure(self, opts):
        super().configure(opts)
        opts = opts or {}
        self.effort = opts.get("effort", self.effort)
        self.prompt_cache = bool(opts.get("prompt_cache", self.prompt_cache))

    def call_cost_usd(self, usage):
        """Dollars for one call: uncached input, cache writes, cache reads and output (thinking included)."""
        p = self.prices
        if not p:
            return 0.0
        cr = usage.get("cached_tokens") or 0
        cw = usage.get("cache_write_tokens") or 0
        inp = (usage.get("prompt_tokens") or 0) - cr - cw
        out = (usage.get("output_tokens") or 0) + (usage.get("thinking_tokens") or 0)
        return (inp * p["input"] + cw * p["cache_write"] + cr * p["cached"] + out * p["output"]) / 1e6

    def _body(self, msg, max_new_tokens, choices, cache=None):
        return build_parts_request(msg["parts"], choices, max_new_tokens, self.effort, msg.get("prefix_len", 0),
                                   self.prompt_cache if cache is None else cache, self.reason_field)

    def is_answered(self, msg, max_new_tokens, choices):
        return self._cache_path(self._body(msg, max_new_tokens, choices)).exists()

    def _one(self, msg, max_new_tokens, choices, cached_content=None):
        if "parts" not in msg:
            raise ValueError("VertexClaudeAdapter takes eval-set messages (parts) only")
        body = self._body(msg, max_new_tokens, choices)
        n_images = sum(1 for p in msg["parts"] if p["type"] == "image")
        path = self._cache_path(body)
        key = path.stem
        cached = path.exists()
        if cached:
            resp = json.loads(path.read_text())
        else:
            # Same hard cap as the Gemini adapter: in-flight calls count at the mean measured cost so far.
            with self._lock:
                mean = self.spent_usd / self._paid_calls if self._paid_calls else 0.25
                committed = self.spent_usd + (self._inflight + 1) * mean
                if self.max_spend_usd is not None and committed > float(self.max_spend_usd):
                    return {"text": "", "error": f"spend cap ${self.max_spend_usd} reached (spent ${self.spent_usd:.2f}, {self._inflight} calls in flight)",
                            "request_key": key}
                self._inflight += 1
            try:
                resp, err = self._post(None, body, url=_endpoint(self.project, self.location, self.model_name))
            finally:
                with self._lock:
                    self._inflight -= 1
            if err:
                return {"text": "", "error": err, "request_key": key}
        text, usage, err = parse_response(resp, choices, self.reason_field)
        if not cached and err is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(resp))
            tmp.replace(path)
        if not cached:
            with self._lock:
                self.spent_usd += self.call_cost_usd(usage)
                self._paid_calls += 1
        usage["image_count"] = n_images
        usage = {k: usage.get(k) for k in CLAUDE_USAGE_KEYS}
        if err:
            return {"text": "", "error": err, "usage": usage, "cached": cached, "request_key": key, "raw": resp}
        return {"text": text, "cumulative_logprob": None, "log_probs": None, "usage": usage, "cached": cached,
                "request_key": key, "raw": resp}

    def ask_group(self, msgs, max_new_tokens, choices):
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            return list(ex.map(lambda m: self._one(m, max_new_tokens, choices), msgs))
