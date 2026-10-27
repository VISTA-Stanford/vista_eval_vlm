"""OpenAI models through the Codex CLI (``codex exec``), behind the adapter interface run_bq drives.

Plan: planning/vista_eval_vlm/vlm-radiology-progression-codex.md (on the mount). GPT-5.6 Sol is
reachable here through Stanford's ChatGPT Edu workspace (Phil 2026-10-15), not through an API
key, so each row is one non-interactive ``codex exec`` call:

  * the CT slices are written as image files to a private temp folder (``/tmp``, mode 700) and
    attached with ``-i``; the folder is deleted after the call;
  * the answer is forced onto the task's options by ``--output-schema`` (a JSON object with one
    ``answer`` enum); an answer off the list or an empty one is an error, not a guess;
  * **sealed**: Codex runs as a coding agent, so its tools are switched off (shell, web search,
    apps, browser, image generation, plugins, memories) and the process runs under bubblewrap with the data
    mount, the code folders, the cloud credentials and this tool's own folders hidden. Any tool
    call in the event stream fails the row;
  * every readable answer is cached under a hash of the exact request (model, prompt, image
    bytes, options, settings), so a rerun only calls Codex for rows it has not answered;
  * a failed call is tried once more; still failing, it comes back as ``{"error": ...}`` and
    run_bq lists the row in the errors file. Hitting the subscription's usage limit stops new
    calls for the rest of the run (rows listed as errors, picked up by the next run);
  * measured usage (input, cached input, output, reasoning tokens) is returned for every row, and
    ``codex.max_total_tokens`` caps new tokens per run (checked before each call, so a run can pass
    it by at most one call).

Subscription-billed: there is no per-call price, so spend is reported in tokens.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .base import BaseVLMAdapter

USAGE_KEYS = ("prompt_tokens", "cached_prompt_tokens", "output_tokens", "thinking_tokens",
              "total_tokens", "image_count", "seconds")
DISABLED_FEATURES = ("shell_tool", "apps", "browser_use", "image_generation", "plugins", "memories")
# Event item types a sealed answer may contain; anything else (a command, a tool call) is a breach.
ALLOWED_ITEM_TYPES = {"agent_message", "reasoning", "error"}
# Either stops every later call this run. "out of credits" is the Edu workspace's pool running dry (10-15 pilot).
USAGE_LIMIT_MARKERS = ("usage limit", "rate limit", "hit your limit", "out of credits")


def hidden_paths(home=None):
    """Folders the sealed process must not see: the data mount, code, credentials, this tool's state."""
    home = Path(home or os.path.expanduser("~"))
    return ["/mnt", str(home / "code"), str(home / ".config" / "gcloud"), str(home / ".claude"),
            str(home / "Documents")]


def build_command(model, image_files, schema_file, last_file, workdir, reasoning_effort=None,
                  codex_bin="codex", seal=True, home=None):
    """argv for one sealed ``codex exec`` call (the prompt goes on stdin). Pure: unit-tested."""
    cmd = [codex_bin, "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--ignore-rules", "--json", "-C", str(workdir), "-o", str(last_file),
           "--output-schema", str(schema_file)]
    for feat in DISABLED_FEATURES:
        cmd += ["--disable", feat]
    # Web search is a config key, not a feature: the 10-15 seal check caught it still on.
    cmd += ["-c", 'web_search="disabled"']
    if reasoning_effort:
        cmd += ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
    if image_files:
        # `-i` takes every following value, so the images go last, comma-joined, then `-` (stdin).
        cmd += ["-i", ",".join(str(f) for f in image_files)]
    cmd += ["-"]
    if seal:
        wrap = ["bwrap", "--dev-bind", "/", "/"]
        for p in hidden_paths(home):
            if Path(p).exists():
                wrap += ["--tmpfs", p]
        cmd = wrap + ["--"] + cmd
    return cmd


def answer_schema(choices):
    if isinstance(choices, dict):
        raise NotImplementedError("panel questions (one field per task) are only wired for the AI Hub adapter")
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": [str(c) for c in choices]}},
            "required": ["answer"], "additionalProperties": False}


def parse_events(events_text, last_message, choices=None):
    """(answer, usage, error, usage_limited) from a call's JSONL events and its last message."""
    usage = {k: None for k in USAGE_KEYS if k != "image_count"}
    tool_items, failed, errors, completed = [], [], [], False
    for line in (events_text or "").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "turn.completed":
            completed = True
            u = ev.get("usage") or {}
            usage = {"prompt_tokens": u.get("input_tokens"), "cached_prompt_tokens": u.get("cached_input_tokens"),
                     "output_tokens": u.get("output_tokens"), "thinking_tokens": u.get("reasoning_output_tokens")}
            usage["total_tokens"] = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        elif kind == "turn.failed":
            failed.append(json.dumps(ev)[:500])
        elif kind == "error":
            # Also sent for a dropped connection Codex then recovers from, so not fatal alone.
            errors.append(json.dumps(ev)[:500])
        elif kind == "item.completed":
            itype = (ev.get("item") or {}).get("type")
            if itype not in ALLOWED_ITEM_TYPES:
                tool_items.append(itype)
    blob = " ".join(failed + errors).lower()
    limited = any(m in blob for m in USAGE_LIMIT_MARKERS)
    if tool_items:
        return None, usage, f"sealed call used a tool ({sorted(set(tool_items))}); answer discarded", limited
    if failed or not completed:
        last = (failed or errors or ["no turn.completed event"])[-1]
        return None, usage, f"codex failed: {last}", limited
    text = (last_message or "").strip()
    if not text:
        return None, usage, "empty answer", limited
    try:
        answer = str(json.loads(text).get("answer", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return None, usage, f"answer is not the schema's JSON: {text[:100]!r}", limited
    if choices and answer not in [str(c) for c in choices]:
        return None, usage, f"answer {answer[:100]!r} not in options", limited
    return answer, usage, None, limited


def encode_image(img, fmt="png", quality=95):
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


class CodexExecAdapter(BaseVLMAdapter):
    config_block = "codex"
    # gpt-5.6-sol's window as Codex reports it (models_cache.json, 10-15).
    context_window = 272_000
    input_token_limit = 272_000
    # Measured 10-15 with synthetic 512px images: about 330 tokens each, ~11.6k fixed per call.
    image_token_estimate = 330

    def __init__(self, model_name, device="auto", cache_dir=None):
        super().__init__(model_name, device, cache_dir)
        self.reasoning_effort = None   # None = the model's default (Phil 10-15)
        self.timeout_s = 1800
        self.retries = 1
        self.image_format = "png"
        self.jpeg_quality = 95
        self.codex_bin = "codex"
        self.seal = True
        self.scratch_root = None
        self.max_total_tokens = None
        self.tokens_used = 0
        self.usage_limited = False
        self.api_calls = 0
        self.response_cache = Path(cache_dir or ".") / "codex_exec_responses"
        self._lock = threading.Lock()

    def configure(self, opts):
        """Run settings from the config's ``codex:`` block (run_bq calls this)."""
        opts = opts or {}
        self.reasoning_effort = opts.get("reasoning_effort", self.reasoning_effort)
        self.timeout_s = int(opts.get("timeout_s", self.timeout_s))
        self.retries = int(opts.get("retries", self.retries))
        self.image_format = opts.get("image_format", self.image_format)
        self.jpeg_quality = int(opts.get("jpeg_quality", self.jpeg_quality))
        self.codex_bin = opts.get("codex_bin", self.codex_bin)
        self.seal = bool(opts.get("seal", self.seal))
        self.scratch_root = opts.get("scratch_root", self.scratch_root)
        self.max_total_tokens = opts.get("max_total_tokens", self.max_total_tokens)

    # -- adapter interface --------------------------------------------------------
    def load(self):
        # Fail now, before any data work, if the CLI or the sandbox is missing.
        if shutil.which(self.codex_bin) is None:
            raise RuntimeError(f"{self.codex_bin!r} not on PATH")
        if self.seal and shutil.which("bwrap") is None:
            raise RuntimeError("bwrap (bubblewrap) is needed to seal the Codex call")
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

    def _cache_key(self, question, encoded, choices):
        h = hashlib.sha256()
        h.update(json.dumps({"model": self.model_name, "question": question, "choices": list(choices or []),
                             "effort": self.reasoning_effort, "format": self.image_format,
                             "quality": self.jpeg_quality if self.image_format == "jpeg" else None},
                            sort_keys=True).encode())
        for b in encoded:
            h.update(hashlib.sha256(b).digest())
        return h.hexdigest()

    def _run_codex(self, question, encoded, choices):
        """One sealed call -> (answer, usage, error, usage_limited)."""
        work = Path(tempfile.mkdtemp(prefix="vlm-codex-", dir=self.scratch_root))
        try:
            os.chmod(work, 0o700)
            ext = "jpg" if self.image_format == "jpeg" else "png"
            files = []
            for i, b in enumerate(encoded):
                f = work / f"img_{i:04d}.{ext}"
                f.write_bytes(b)
                files.append(f)
            schema = work / "schema.json"
            schema.write_text(json.dumps(answer_schema(choices or [])))
            last = work / "last.txt"
            run_dir = work / "cwd"
            run_dir.mkdir()
            cmd = build_command(self.model_name, files, schema, last, run_dir, self.reasoning_effort,
                                self.codex_bin, self.seal)
            with self._lock:
                self.api_calls += 1
            try:
                proc = subprocess.run(cmd, input=question, capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return None, {}, f"codex timed out after {self.timeout_s}s", False
            last_text = last.read_text() if last.exists() else ""
            answer, usage, err, limited = parse_events(proc.stdout, last_text, choices)
            if err is None and proc.returncode != 0:
                err = f"codex exit {proc.returncode}: {proc.stderr[-300:]}"
            if not limited:
                limited = any(m in (proc.stderr or "").lower() for m in USAGE_LIMIT_MARKERS)
            return answer, usage, err, limited
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _one(self, msg, choices):
        encoded = [encode_image(im, self.image_format, self.jpeg_quality) for im in msg["images"]]
        key = self._cache_key(msg["question"], encoded, choices)
        path = self.response_cache / key[:2] / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            usage = {k: rec["usage"].get(k) for k in USAGE_KEYS}
            return {"text": rec["answer"], "cumulative_logprob": None, "log_probs": None, "usage": usage, "cached": True}
        err, usage = None, {}
        t0 = time.monotonic()  # wall time for this scan, retries included
        for attempt in range(self.retries + 1):
            with self._lock:
                if self.usage_limited:
                    return {"text": "", "error": "Codex usage limit reached earlier this run; rerun later"}
                if self.max_total_tokens is not None and self.tokens_used >= int(self.max_total_tokens):
                    return {"text": "", "error": f"token cap {self.max_total_tokens} reached ({self.tokens_used} used)"}
            answer, usage, err, limited = self._run_codex(msg["question"], encoded, choices)
            with self._lock:
                self.tokens_used += int((usage or {}).get("total_tokens") or 0)
                if limited:
                    self.usage_limited = True
            if err is None:
                break
            if limited:
                break
            time.sleep(5)
        usage = dict(usage or {})
        usage["image_count"] = len(encoded)
        usage["seconds"] = round(time.monotonic() - t0, 1)
        usage = {k: usage.get(k) for k in USAGE_KEYS}
        if err:
            return {"text": "", "error": err, "usage": usage, "cached": False}
        # Cache only a readable answer, so an unreadable one is asked again on the next run.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"answer": answer, "usage": usage, "model": self.model_name}))
        tmp.replace(path)
        return {"text": answer, "cumulative_logprob": None, "log_probs": None, "usage": usage, "cached": False}

    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        # One call at a time (plan: pacing), in row order.
        msgs = inputs if isinstance(inputs, list) else [inputs]
        return [self._one(m, constrained_choices) for m in msgs]
