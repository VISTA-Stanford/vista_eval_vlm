"""Gemini 4 Argon as the CT reader, zero-shot, on the ct-findings-5 VAL-200 scans (spec
planning/vista_eval_vlm/2026-11-14-ct-findings-argon-zeroshot.md; arm 2 of the jury draft).

One Vertex call per scan: the scan's 50-slice pack (the evalset argon recipe: lossless PNG, MedGemma's three windows,
each image labelled, images first, media resolution high, thinking low), then the Claude reader's zero-shot wording
(the seed instructions.md and the per-call wrapper's five questions, minus the viewer and file steps that only a Claude
session has), answered as one JSON object: five whole numbers 0-100 and 3-5 reasons, forced by a response schema. A
reply that breaks the shape is a failed answer (scored 50 by the task scorer), never retried; only network errors are
retried (the evalset client's backoff). VAL pool only: TRAIN and TEST are never asked.

The evalset code lives on vista_eval_vlm main, not on this branch: it is imported from a checkout pinned to a main
commit (EVALSET_SRC, default ~/code/vista_eval_vlm-evalset-main/src), and that commit is written into every run log.

    render   the 50-slice pack of every VAL pool scan, into ct-findings-5-v1_7/argon_packs/ (kept if already there)
    check    $0: prompt wording, the parser on good and bad replies, two packs load hash-checked, the run list is the
             VAL pool only (no TRAIN scan, no TEST-sample patient) -> every line PASS
    run      ask Argon: --limit N (first N VAL scans by id) --max-usd D (hard stop) --name NAME; writes answers.json
             (the task's answer shape), calls.jsonl (tokens, dollars, error per scan) and run.json to
             ct-findings-5-v1_7-runs/<host>/argon-zeroshot/<name>/; a rerun skips scans already answered

    cd optimizer/ct-findings-5
    uv run -q --no-project --with numpy --with nibabel --with pillow --with pandas --with pyarrow --with requests \\
        --with google-auth --with pyyaml python -B scripts/argon_alone.py check
    ... argon_alone.py run --limit 5 --max-usd 2 --name check5

Prints counts and dollars only: never an id, a UID, a date or report text.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(TASK))
EVALSET_SRC = Path(os.environ.get("EVALSET_SRC", Path.home() / "code" / "vista_eval_vlm-evalset-main" / "src"))
sys.path.insert(0, str(EVALSET_SRC))

import config  # noqa: E402  (the task's: pools, questions, data paths)

MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
PACK_ROOT = config.DATA_DIR / "argon_packs"
RUNS = MOUNT / "ct-findings-5-v1_7-runs" / socket.gethostname() / "argon-zeroshot"
TEST_SAMPLE = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/samples/step2.parquet")
KEYS = [k for k, _t, _q in config.QUESTIONS]
IMAGE_LINE = ("These are axial slices of a chest CT, ordered from top (head) to bottom (feet), each shown in {windows}.")


def evalset():
    from evalset import packs
    from evalset.recipes.argon import RECIPE
    return packs, RECIPE


def evalset_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(EVALSET_SRC), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def val_scans() -> list[dict]:
    """[{id, study, series, nifti}] for the VAL pool, sorted by id, from the private id map."""
    pools = json.loads((config.DATA_DIR / "pools.json").read_text())
    val = set(pools["val"])
    out = []
    for line in (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines():
        if not line:
            continue
        m = json.loads(line)
        if m["id"] in val and m["split"] == "val":
            name = Path(m["nifti"]).name.removesuffix(".nii.gz")
            study, series = name.split("__")
            out.append({"id": m["id"], "study": study, "series": series, "nifti": m["nifti"]})
    if len(out) != len(val):
        raise SystemExit(f"id map covers {len(out)} of {len(val)} VAL scans")
    return sorted(out, key=lambda s: s["id"])


# -- the prompt -------------------------------------------------------------------------------------------------

def seed_text() -> str:
    """The Claude reader's zero-shot instructions, without the viewer step (Argon is given the images)."""
    lines = (TASK / "program_agentic" / "instructions.md").read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("To look at the scan")).strip()


def prompt_text(windows_phrase: str) -> str:
    questions = " ".join(f'({n}) "{key}": {text}' for n, (key, _t, text) in enumerate(config.QUESTIONS, 1))
    keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key in KEYS)
    return (
        IMAGE_LINE.format(windows=windows_phrase) + "\n\n" + seed_text() + "\n\n"
        f"Answer five questions about this scan: {questions} "
        f"Give each as the chance, from 0 to 100, that the answer is yes. "
        f'Reply with exactly one JSON object: {{{keys}, "reasons": [{{"finding": "<one thing you saw in the scan>", '
        f'"where": "<where in the scan, e.g. right upper lobe, liver, mediastinal nodes>"}}, ...]}}. '
        f"Each number is your probability in percent, 0 to 100; a number outside that range, or a word, counts as no "
        f'answer. In "reasons" list the 3 to 5 things in the scan that moved your probabilities most, most important '
        f"first."
    )


SCHEMA = {
    "type": "OBJECT",
    "properties": {**{k: {"type": "INTEGER"} for k in KEYS},
                   "reasons": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                       "finding": {"type": "STRING"}, "where": {"type": "STRING"}}, "required": ["finding", "where"]}}},
    "required": KEYS + ["reasons"],
    "propertyOrdering": KEYS + ["reasons"],
}


def parse_answer(text: str | None) -> dict:
    """The task's answer shape from Argon's reply: probabilities 0-1, reasons, or an error (scored 50 on all five).
    The same rule as the Claude reader's output.json: each of the five must be a number 0-100 (not a bool)."""
    out = {k: None for k in KEYS} | {"error": None, "reason": None, "reasons": [], "findings_log": None}
    if not text:
        out["error"] = "no reply"
        return out
    try:
        obj = json.loads(text)
    except ValueError:
        out["error"] = "reply is not JSON"
        return out
    if not isinstance(obj, dict):
        out["error"] = "reply is not one JSON object"
        return out
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
            out["error"] = f"{k}: not a number from 0 to 100"
            return {**{kk: None for kk in KEYS}, **{x: out[x] for x in ("error", "reason", "reasons", "findings_log")}}
        out[k] = round(float(v) / 100.0, 4)
    reasons = obj.get("reasons")
    out["reasons"] = [{"finding": str(r.get("finding", ""))[:300], "where": str(r.get("where", ""))[:120]}
                      for r in reasons[:5] if isinstance(r, dict)] if isinstance(reasons, list) else []
    out["reason"] = "; ".join(r["finding"] for r in out["reasons"])[:600] or None
    return out


# -- packs ------------------------------------------------------------------------------------------------------

def _render(scan: dict) -> str:
    packs, recipe = evalset()
    from context import nifti
    n = recipe.slices_per_scan(1)
    d = packs.pack_dir(recipe, n, PACK_ROOT) / packs.scan_id(scan["study"], scan["series"])
    if (d / "manifest.json").is_file():
        return "kept"
    try:
        packs.render_scan(recipe, n, scan["study"], scan["series"], PACK_ROOT, commit=evalset_commit(),
                          stack=nifti.AxialStack.open(scan["nifti"]))
        return "rendered"
    except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the batch; counted by type
        return f"failed:{type(exc).__name__}"


def cmd_render(args) -> int:
    scans = val_scans()[: args.limit or None]
    status: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, st in enumerate(pool.map(_render, scans), 1):
            status[st] = status.get(st, 0) + 1
            if n % 25 == 0:
                print(f"{n}/{len(scans)} {status}", flush=True)
    print(f"done {len(scans)}: {status}")
    return 1 if any(k.startswith("failed") for k in status) else 0


def load_pack(scan: dict) -> list[bytes]:
    packs, recipe = evalset()
    return packs.load(recipe, recipe.slices_per_scan(1), scan["study"], scan["series"], PACK_ROOT)


# -- Argon ------------------------------------------------------------------------------------------------------

def _vertex_module():
    """models.vertex_gemini without running models/__init__.py, which imports torch for the local models. The module
    itself needs only models/base.py (standard library)."""
    import types
    if "models" not in sys.modules:
        pkg = types.ModuleType("models")
        pkg.__path__ = [str(EVALSET_SRC / "models")]
        sys.modules["models"] = pkg
    import models.vertex_gemini as vg
    return vg


def make_client():
    from evalset.run import ARGON_PRICES
    VertexGeminiAdapter = _vertex_module().VertexGeminiAdapter
    _packs, recipe = evalset()
    c = VertexGeminiAdapter(recipe.model_name, "cpu", None)
    c.configure({**dict(recipe.settings), "price_per_mtok": ARGON_PRICES})
    c._token()  # fail now if the machine's Google login is missing
    return c, recipe


def ask(client, recipe, scan: dict) -> tuple[dict, dict]:
    """(answer, call log) for one scan: one generateContent call."""
    vg = _vertex_module()
    build_parts_request, parse_response = vg.build_parts_request, vg.parse_response
    msg = recipe.build(prompt_text(recipe.windows_phrase()), [(None, load_pack(scan))])
    body = build_parts_request(list(msg.parts), None, int(recipe.setting("max_new_tokens", 8192)),
                               recipe.setting("media_resolution"), recipe.setting("thinking_level"))
    body["generationConfig"]["responseMimeType"] = "application/json"
    body["generationConfig"]["responseSchema"] = SCHEMA
    resp, err = client._post("generateContent", body)
    if err:
        return parse_answer(None) | {"error": f"call failed: {err[:200]}"}, {"usage": None, "usd": 0.0, "error": err[:300]}
    text, usage, perr = parse_response(resp)
    usd = client.call_cost_usd(usage)
    answer = parse_answer(text) if not perr else parse_answer(None) | {"error": perr[:200]}
    return answer, {"usage": usage, "usd": round(usd, 5), "error": answer["error"]}


def cmd_run(args) -> int:
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    apath, cpath = out / "answers.json", out / "calls.jsonl"
    answers = json.loads(apath.read_text())["answers"] if apath.is_file() else {}
    spent = sum(json.loads(l)["usd"] for l in cpath.read_text().splitlines() if l) if cpath.is_file() else 0.0
    scans = [s for s in val_scans()[: args.limit or None] if s["id"] not in answers]
    client, recipe = make_client()
    lock = threading.Lock()
    stop = threading.Event()

    def one(scan):
        nonlocal spent
        if stop.is_set():
            return
        answer, log = ask(client, recipe, scan)
        with lock:
            answers[scan["id"]] = answer
            spent += log["usd"]
            with cpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": scan["id"], **log}) + "\n")
            apath.write_text(json.dumps({"answers": answers}, indent=1), encoding="utf-8")
            if spent >= args.max_usd:
                stop.set()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, scans))
    failed = sum(1 for a in answers.values() if a["error"])
    run = {"spec": "planning/vista_eval_vlm/2026-11-14-ct-findings-argon-zeroshot.md", "host": socket.gethostname(),
           "model": recipe.model_name, "recipe": recipe.name, "slices": recipe.slices_per_scan(1),
           "settings": dict(recipe.settings), "evalset_src": str(EVALSET_SRC), "evalset_commit": evalset_commit(),
           "task_commit": subprocess.run(["git", "-C", str(TASK), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
           "pack_root": str(PACK_ROOT), "answered": len(answers), "failed": failed, "usd": round(spent, 4),
           "stopped_at_max_usd": stop.is_set(), "max_usd": args.max_usd}
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    per = spent / max(len(answers), 1)
    print(f"answered {len(answers)}, failed {failed}, ${spent:.2f} (${per:.3f}/scan)"
          f"{'; STOPPED at --max-usd' if stop.is_set() else ''}; results in {out}")
    return 1 if failed or stop.is_set() else 0


# -- the $0 check -----------------------------------------------------------------------------------------------

def cmd_check(_args) -> int:
    import pandas as pd

    bad = 0

    def report(name, ok, detail=""):
        nonlocal bad
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")

    packs, recipe = evalset()
    text = prompt_text(recipe.windows_phrase())
    seed = seed_text()
    report("prompt = image line + the Claude seed wording + the five questions + 0-100 + JSON reply",
           text.startswith("These are axial slices") and seed in text and "To look at the scan" not in text
           and all(f'"{k}"' in text for k in KEYS) and "from 0 to 100" in text, f"{len(text)} chars")
    good = json.dumps({**{k: v for k, v in zip(KEYS, (5, 95, 40, 0, 100))},
                       "reasons": [{"finding": "f", "where": "liver"}]})
    a = parse_answer(good)
    report("parser: a good reply gives five probabilities 0-1 and its reasons",
           a["error"] is None and [a[k] for k in KEYS] == [0.05, 0.95, 0.4, 0.0, 1.0] and len(a["reasons"]) == 1)
    bads = {"word": {**json.loads(good), KEYS[0]: "high"}, "over 100": {**json.loads(good), KEYS[1]: 101},
            "negative": {**json.loads(good), KEYS[2]: -1}, "bool": {**json.loads(good), KEYS[3]: True},
            "missing key": {k: v for k, v in json.loads(good).items() if k != KEYS[4]}}
    fails = {n: parse_answer(json.dumps(o)) for n, o in bads.items()}
    fails["bad JSON"], fails["empty"] = parse_answer("{not json"), parse_answer("")
    report("parser: a word, >100, <0, a bool, a missing key, bad JSON or no reply is a failed answer with no numbers",
           all(f["error"] and all(f[k] is None for k in KEYS) for f in fails.values()), f"{len(fails)} cases")
    report("response schema requires the five integers and reasons", set(SCHEMA["required"]) == set(KEYS) | {"reasons"}
           and all(SCHEMA["properties"][k]["type"] == "INTEGER" for k in KEYS))
    scans = val_scans()
    pools = json.loads((config.DATA_DIR / "pools.json").read_text())
    ids = {s["id"] for s in scans}
    report("run list is exactly the VAL pool, no TRAIN scan", ids == set(pools["val"]) and not ids & set(pools["train"]),
           f"{len(ids)} scans")
    id_map = [json.loads(l) for l in (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines() if l]
    persons = {int(m["person_id"]) for m in id_map if m["id"] in ids}
    test = set(pd.read_parquet(TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    report("no TEST-sample patient in the run list", len(test) > 0 and not persons & test,
           f"{len(test)} TEST-sample patients, {len(persons & test)} in the list")
    two = scans[:2]
    st = [_render(s) for s in two]
    try:
        n_imgs = [len(load_pack(s)) for s in two]
        ok = n_imgs == [recipe.slices_per_scan(1)] * 2
    except Exception as exc:  # noqa: BLE001
        ok, n_imgs = False, [type(exc).__name__]
    report("two VAL packs render and load with every file matching its manifest hash", ok and not any(
        x.startswith("failed") for x in st), f"{st}, images {n_imgs}")
    msg = recipe.build(text, [(None, load_pack(two[0]))]) if ok else None
    n_img = sum(1 for p in msg.parts if p["type"] == "image") if msg else 0
    report("message: 50 labelled images, then the question", bool(msg) and n_img == 50 and msg.parts[0]["text"] ==
           "Image 1:" and msg.parts[-1]["text"] == text, f"{n_img} images")
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("check")
    q = sub.add_parser("run")
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--max-usd", type=float, required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    return {"render": cmd_render, "check": cmd_check, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
