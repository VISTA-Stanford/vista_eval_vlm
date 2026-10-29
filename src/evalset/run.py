"""Ask a step's questions with one recipe and write the answers file.

For each case: load the scan(s) from the recipe's image pack (verified against the manifest), build
one call per question (card text + recipe wrapper, images first), and send a case's calls together
so they share the cached image prefix. Answers go to
``<root>/runs/<run>/<recipe>/step<N>_<arm>.csv``, one row per (case, question): the label, the raw
response, the parsed answer, refusal, usage, and (open models) the yes/no token probabilities.
A rerun asks only rows without a parsed answer; API answers also come from the adapters' caches.

    python -m evalset.run --recipe argon --step 2 --arm ct --run check5 --limit 5 --max-spend 5
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from evalset import packs, paths, samples
from evalset.cards import CARDS
from evalset.recipes import Recipe, get
from evalset.score import is_refusal, parse_answer

CHOICES = ["yes", "no"]
KEY_COLS = ["case_id", "question_id", "arm"]


# -- calls ------------------------------------------------------------------------------------------

def scans_for_case(recipe: Recipe, card, row, root=None):
    """[(scan name, [image bytes])] in image order: prior first, then the current scan."""
    n = recipe.slices_per_scan(len(card.scan_names))
    if len(card.scan_names) == 1:
        return [(None, packs.load(recipe, n, row["study"], row["series"], root))]
    prior_name, current_name = card.scan_names
    return [(prior_name, packs.load(recipe, n, row["prior_study"], row["prior_series"], root)),
            (current_name, packs.load(recipe, n, row["study"], row["series"], root))]


def messages_for_case(recipe: Recipe, card, row, arm: str, root=None, questions=None):
    """[(question, Message)] for every question on the card (or only ids in ``questions``), for one case."""
    scans = scans_for_case(recipe, card, row, root) if arm == "ct" else None
    out = []
    for q in card.questions:
        if questions and q.id not in questions:
            continue
        text = card.text(q, recipe.windows_phrase(), arm, summary=row.get("summary"))
        out.append((q, recipe.build(text, scans)))
    return out


class APIBackend:
    """A models.MODEL_REGISTRY adapter (AI Hub, Vertex Gemini). ``ask_many`` sends calls concurrently;
    ``is_cached`` says whether a call's answer is already in the adapter's disk cache (so it would not warm the
    provider's cache); ``per_case`` is set for an explicit-cache model, which is driven one scan at a time."""

    def __init__(self, recipe: Recipe, cache_dir: Path, max_spend: float | None):
        from models import MODEL_REGISTRY
        self.recipe = recipe
        self.adapter = MODEL_REGISTRY[recipe.model_type](recipe.model_name, "cpu", str(cache_dir))
        opts = dict(recipe.settings)
        if recipe.model_type == "vertex_gemini":
            opts.setdefault("price_per_mtok", ARGON_PRICES)
            if max_spend is not None:
                opts["max_spend_usd"] = max_spend
        elif recipe.model_type == "aihub_openai" and max_spend is not None:
            # The AI Hub adapter caps tokens, not dollars: every token counted at the $10 / M input rate. Output
            # is $50 / M, so the cap runs slightly over when answers are long (here: a one-word schema answer
            # plus low-effort reasoning).
            opts["max_total_tokens"] = int(max_spend / 10.0 * 1e6)
        self.adapter.configure(opts)
        self.adapter.load()
        self.max_new_tokens = int(recipe.setting("max_new_tokens", 4096))

        self.per_case = bool(getattr(self.adapter, "explicit_cache", False))

    @staticmethod
    def _wire(m):
        return {"parts": list(m.parts), "prefix_len": m.prefix_len}

    def is_cached(self, m) -> bool:
        return bool(self.adapter.is_answered(self._wire(m), self.max_new_tokens, CHOICES))

    def ask_many(self, msgs):
        return self.adapter.infer(None, None, [self._wire(m) for m in msgs], self.max_new_tokens, CHOICES)

    def ask_case(self, msgs):
        """One scan's questions through an explicit context cache (``ask_group``)."""
        return self.adapter.ask_group([self._wire(m) for m in msgs], self.max_new_tokens, CHOICES)


#: Gemini 4 Argon, Vertex global, USD per million tokens through 2026-11-30 (doubles from 12-01).
ARGON_PRICES = {"input": 2.0, "cached": 0.10, "output": 10.0, "long_context_over": 10 ** 9,
                "input_long": 2.0, "output_long": 10.0}


# -- rows -------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _image_digest(data: bytes) -> bytes:
    # A scan's slices are the same bytes objects for every question on it, so each is hashed once.
    return hashlib.sha256(data).digest()


def message_key(m) -> str:
    """Fingerprint of exactly what the model was sent (every text part, every image's hash, in order)."""
    h = hashlib.sha256()
    for p in m.parts:
        h.update(b"T" + p["text"].encode() if p["type"] == "text" else b"I" + _image_digest(p["data"]))
    return h.hexdigest()


def p_yes(log_probs):
    """P(yes) from the final answer token's top log-probabilities (open models): the yes-token mass over
    yes + no, each summed over spellings (" yes", "Yes", ...). None when neither appears."""
    if not isinstance(log_probs, str) or not log_probs:
        return None
    top = json.loads(log_probs).get("top", {})
    mass = {"yes": 0.0, "no": 0.0}
    for tok, lp in top.items():
        k = tok.strip().lower().strip("*.'\"")
        if k in mass:
            mass[k] += math.exp(lp)
    tot = mass["yes"] + mass["no"]
    return mass["yes"] / tot if tot > 0 else None


def result_row(recipe, step, arm, row, q, out, m=None):
    text = out.get("text") or ""
    err = out.get("error")
    label = row.get(f"label__{q.task}")
    usage = out.get("usage") or {}
    rec = {"case_id": str(row["case_id"]), "person_id": str(row["person_id"]), "split": row.get("split"),
           "step": step, "question_id": q.id, "task": q.task, "arm": arm, "recipe": recipe.name,
           "model": recipe.model_name, "label": None if label is None or (isinstance(label, float) and math.isnan(label)) else int(label),
           "response_text": text, "answer": parse_answer(text, recipe.answer_form) if not err else None,
           # A refusal is a declined answer: disclaimer wording next to a readable yes/no is not one.
           "refused": False, "error": err, "cached": out.get("cached"),
           "log_probs": out.get("log_probs"), "p_yes": p_yes(out.get("log_probs")),
           "message_key": message_key(m) if m is not None else None, "request_key": out.get("request_key")}
    rec["refused"] = rec["answer"] is None and is_refusal(text, err)
    for k, v in usage.items():
        rec[f"usage_{k}"] = v
    return rec


def out_path(run: str, recipe: Recipe, step: int, arm: str, root=None) -> Path:
    return Path(root or paths.root()) / "runs" / run / recipe.name / f"step{step}_{arm}.csv"


def run(recipe: Recipe, step: int, arm: str, run_name: str, limit: int | None = None,
        max_spend: float | None = None, backend=None, root=None, questions=None, variant=None,
        which: str = "sample") -> pd.DataFrame:
    card = CARDS[step]
    df = samples.load(step, root, which)
    if limit:
        df = packs.check_cases(df, step, limit)
    path = out_path(run_name, recipe, step, arm, root)
    done = pd.DataFrame()
    if path.exists():
        prev = pd.read_csv(path, dtype={"case_id": str, "person_id": str})
        done = prev[prev["answer"].isin(CHOICES)]
    done_keys = set(map(tuple, done[KEY_COLS].astype(str).values)) if len(done) else set()
    if backend is None:
        backend = (VLLMBackend(recipe) if recipe.model_type == "vllm"
                   else APIBackend(recipe, Path(root or paths.root()) / "model_cache", max_spend))
    todo = []
    for _, r in df.iterrows():
        r = r.to_dict()
        qm = [(q, m) for q, m in messages_for_case(recipe, card, r, arm, root, questions)
              if (str(r["case_id"]), q.id, arm) not in done_keys]
        if qm:
            todo.append((r, qm))
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = max(1, int(recipe.setting("cache_chunk", 8)))
    rows, outs_all = [], []
    for c in range(0, len(todo), chunk):
        pairs = ask_chunk(backend, todo[c:c + chunk])
        new_rows = [result_row(recipe, step, arm, r, q, o, m) for r, q, m, o in pairs]
        log_attempts(path, new_rows, [o for _, _, _, o in pairs])
        rows += new_rows
        outs_all += [o for _, _, _, o in pairs]
        # Saved after every group, so an interrupted run keeps what it has and a rerun asks only the rest.
        write_answers(path, done, rows)
    log_run(path, recipe, step, arm, limit, questions, rows, outs_all, variant)
    allrows = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    if allrows.empty:
        raise RuntimeError(f"step {step} {arm}: no cases to ask and none answered before; nothing written")
    write_answers(path, done, rows)
    return allrows


def ask_chunk(backend, todo):
    """[(row, question, message, output)] for a group of cases, in two waves so every model's cache can reuse
    each scan's images: first each case's lead question, then all the rest.

    The lead is the case's first question whose answer is not already in the adapter's disk cache (a cached
    answer comes back without a call, so it would not warm the provider's cache for the others). The rest go
    once the leads have finished, while the images are still cached: Google's implicit cache (Argon: ~96% of the
    prompt, 10-28), the AI Hub's prompt cache and vLLM's prefix cache all need an earlier request with the same
    prefix to have completed. Groups are kept small (recipe setting ``cache_chunk``, default 8) so vLLM's prefix
    blocks are not evicted between the waves, and so API calls in a wave run concurrently.
    An explicit-cache model (``per_case``) is driven one scan at a time through its own cache instead."""
    if getattr(backend, "per_case", False):
        out = []
        for r, qm in todo:
            outs = backend.ask_case([m for _, m in qm])
            out += [(r, q, m, o) for (q, m), o in zip(qm, outs)]
        return out
    cached = getattr(backend, "is_cached", lambda m: False)
    leads = []
    for _, qm in todo:
        k = next((i for i, (_, m) in enumerate(qm) if not cached(m)), 0)
        leads.append(k)
    first = backend.ask_many([qm[k][1] for (_, qm), k in zip(todo, leads)]) if todo else []
    rest_idx = [(ci, qi) for ci, ((_, qm), k) in enumerate(zip(todo, leads)) for qi in range(len(qm)) if qi != k]
    rest = backend.ask_many([todo[ci][1][qi][1] for ci, qi in rest_idx]) if rest_idx else []
    got = {(ci, leads[ci]): o for ci, o in enumerate(first)}
    got.update({key: o for key, o in zip(rest_idx, rest)})
    return [(r, q, m, got[(ci, qi)]) for ci, (r, qm) in enumerate(todo) for qi, (q, m) in enumerate(qm)]


def write_answers(path: Path, done, rows):
    allrows = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    if allrows.empty:
        return
    tmp = path.with_suffix(".tmp")
    allrows.to_csv(tmp, index=False)
    tmp.replace(path)


def log_attempts(path: Path, rows, outs):
    """Append every attempt, failed ones included, with the model's full reply body, to
    ``step<N>_<arm>.attempts.jsonl``. Never rewritten: the answers file keeps the latest readable answer
    per question, this log keeps everything ever asked."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.with_suffix(".attempts.jsonl").open("a", encoding="utf-8") as f:
        for rec, o in zip(rows, outs):
            f.write(json.dumps({**rec, "time": now, "raw": o.get("raw")}, default=str) + "\n")


def environment() -> dict:
    """Python and the versions of the packages that shape inputs or calls (missing ones are skipped)."""
    import platform
    from importlib import metadata
    pkgs = {}
    for name in ("vllm", "transformers", "torch", "numpy", "pillow", "nibabel", "pandas", "requests",
                 "google-auth", "huggingface-hub"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": pkgs}


def model_revision(recipe: Recipe, outs) -> dict:
    """What the provider says it ran: Gemini's ``modelVersion`` / the AI Hub's ``model`` from the saved replies
    (distinct values in this invocation), or, for open models, the weights snapshot folder in the cache."""
    if recipe.model_type == "vllm":
        snaps = Path(model_cache_dir()) / ("models--" + recipe.model_name.replace("/", "--")) / "snapshots"
        return {"weights_snapshots": sorted(p.name for p in snaps.glob("*")) if snaps.is_dir() else [],
                "weights_folder": model_cache_dir()}
    seen = set()
    for o in outs or []:
        raw = o.get("raw")
        if isinstance(raw, dict):
            v = raw.get("modelVersion") or raw.get("model")
            if v:
                seen.add(str(v))
    return {"reported": sorted(seen)}


def log_run(path: Path, recipe: Recipe, step: int, arm: str, limit, questions, rows, outs=None, variant=None):
    """Append one line per invocation to ``step<N>_<arm>.runs.jsonl``: when, where, which code and package
    versions, the full recipe (and arms-file variant, if any), the model version reported, the image pack, what
    was asked and how many answers were read."""
    import dataclasses
    import socket
    import subprocess
    from datetime import datetime, timezone
    commit = paths.code_commit()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip())
    card = CARDS[step]
    rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": socket.gethostname(),
           "commit": commit, "uncommitted_changes": dirty, "step": step, "arm": arm, "limit": limit,
           "questions": questions, "recipe": dataclasses.asdict(recipe),
           "pack": recipe.pack_id(recipe.slices_per_scan(len(card.scan_names))) if arm == "ct" else None,
           "n_asked": len(rows), "n_parsed": sum(r["answer"] in CHOICES for r in rows),
           "variant": variant, "environment": environment(), "model_revision": model_revision(recipe, outs)}
    with path.with_suffix(".runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# -- open models (vLLM, weill) ------------------------------------------------------------------------

class VLLMBackend:
    """MedGemma through vLLM's chat API: images as JPEG data URLs (the notebook's form), greedy, with the
    top token log-probabilities kept at the position of the final yes/no."""

    def __init__(self, recipe: Recipe):
        import os
        cache = model_cache_dir()
        # Set before importing vLLM (Hugging Face reads these at import). Tokenizer, processor and chat
        # template load through Hugging Face, not download_dir, so its cache points at the same folder. A folder
        # that already holds the model loads offline; an empty one is filled once from Hugging Face (gated:
        # needs HF_TOKEN with MedGemma access).
        # Overrides any HF_HUB_CACHE already set: the tokenizer must look where the weights check looked.
        os.environ["HF_HUB_CACHE"] = cache
        if model_in_cache(cache, recipe.model_name):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        else:
            from huggingface_hub import get_token  # env HF_TOKEN or a saved `huggingface-cli login`
            if not get_token():
                raise RuntimeError(f"{recipe.model_name} is not in {cache} and no Hugging Face token is set to download it")
        from vllm import LLM, SamplingParams
        self.recipe = recipe
        n_max = max(recipe.n_slices, 2 * recipe.n_slices_pair)
        self.llm = LLM(model=recipe.model_name, dtype="bfloat16", max_model_len=int(recipe.setting("max_model_len")),
                       limit_mm_per_prompt={"image": n_max},
                       # EVALSET_PREFIX_CACHE=0 turns it off, to check cached and uncached runs give the same answers.
                       enable_prefix_caching=os.environ.get("EVALSET_PREFIX_CACHE", "1") != "0",
                       gpu_memory_utilization=0.85, trust_remote_code=True,
                       download_dir=cache)
        self.params = SamplingParams(temperature=float(recipe.setting("temperature", 0.0)),
                                     max_tokens=int(recipe.setting("max_new_tokens", 2000)),
                                     logprobs=int(recipe.setting("logprobs", 20)))

    def ask_many(self, msgs):
        convs = [[{"role": "user", "content": openai_content(m)}] for m in msgs]
        outs = self.llm.chat(convs, self.params, use_tqdm=True)
        return [{"text": o.outputs[0].text, "log_probs": answer_logprobs(o.outputs[0]), "usage":
                 {"prompt_tokens": len(o.prompt_token_ids or []), "output_tokens": len(o.outputs[0].token_ids)}}
                for o in outs]


def model_cache_dir() -> str:
    """Weights folder: ``EVALSET_MODEL_CACHE`` (a local disk is much faster: the mount read CTs at ~11 MB/s
    on weill and the model is 8.1 GB), else the shared cache on the mount, which holds MedGemma 1.5 4B."""
    import os
    return os.environ.get("EVALSET_MODEL_CACHE") or str(paths.mount_root() / paths.BUCKET / "vistabench/vlm/model_cache")


def model_in_cache(cache: str, model_name: str) -> bool:
    """True when the Hugging Face cache folder holds a complete snapshot of ``model_name``: every shard its
    ``model.safetensors.index.json`` names (a half-finished download is not complete), plus ``config.json``."""
    snaps = Path(cache) / ("models--" + model_name.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return False
    for s in snaps.glob("*"):
        if not (s / "config.json").exists():
            continue
        idx = s / "model.safetensors.index.json"
        if idx.exists():
            shards = set(json.loads(idx.read_text()).get("weight_map", {}).values())
            if shards and all((s / f).exists() for f in shards):
                return True
        elif (s / "model.safetensors").exists():
            return True
    return False


def openai_content(m):
    import base64
    out = []
    for p in m.parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        else:
            out.append({"type": "image_url", "image_url": {
                "url": f"data:{p['mime']};base64," + base64.b64encode(p["data"]).decode("ascii")}})
    return out


def answer_logprobs(completion):
    """JSON {position, top: {token: logprob}} at the last generated token that reads yes/no, or None."""
    lps = getattr(completion, "logprobs", None)
    if not lps:
        return None
    for i in range(len(completion.token_ids) - 1, -1, -1):
        pos = lps[i] or {}
        chosen = pos.get(completion.token_ids[i])
        tok = getattr(chosen, "decoded_token", "") or ""
        if tok.strip().lower().strip("*.'\"") in ("yes", "no"):
            return json.dumps({"position": i, "top": {str(getattr(v, "decoded_token", k)): float(v.logprob)
                                                      for k, v in pos.items()}})
    return None


def recipe_for(name: str, arms_file=None, variant=None):
    """(recipe, provenance) for the CLI: the registry recipe, or one variant of it from an arms file."""
    recipe = get(name)
    if bool(arms_file) != bool(variant):
        raise SystemExit("--arms and --variant go together")
    if not arms_file:
        return recipe, None
    from evalset import arms
    over, prov = arms.load(arms_file, variant)
    return arms.apply(recipe, variant, over), prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--step", type=int, nargs="+", required=True)
    ap.add_argument("--arm", default=None, help="default: every arm the card has")
    ap.add_argument("--run", required=True, help="run name, e.g. check5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None, help="USD cap for this invocation (API models)")
    ap.add_argument("--questions", nargs="+", default=None, help="question ids to ask (default: all on each card)")
    ap.add_argument("--arms", default=None, help="arms YAML (see evalset.arms); with --variant")
    ap.add_argument("--variant", default=None, help="variant name in the arms file")
    ap.add_argument("--which", choices=["sample", "test"], default="sample",
                    help="the step's sample (default) or its whole TEST split")
    args = ap.parse_args()
    recipe, variant = recipe_for(args.recipe, args.arms, args.variant)
    # One backend for every step: weights load once, and the spend cap covers the whole invocation.
    backend = (VLLMBackend(recipe) if recipe.model_type == "vllm"
               else APIBackend(recipe, paths.root() / "model_cache", args.max_spend))
    for step in args.step:
        if args.questions and not any(q.id in args.questions for q in CARDS[step].questions):
            continue
        for arm in ([args.arm] if args.arm else list(CARDS[step].arms)):
            res = run(recipe, step, arm, args.run, args.limit, args.max_spend, backend, questions=args.questions,
                      variant=variant, which=args.which)
            n_ok = int(res["answer"].isin(CHOICES).sum())
            print(f"{recipe.name} step {step} {arm}: {n_ok}/{len(res)} parsed -> {out_path(args.run, recipe, step, arm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
