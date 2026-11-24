"""Single-image calls for the small-finding perception study (spec planning/vista_eval_vlm/2026-11-23-small-finding-perception.md).

    layer 1   each positive and negative, alone, as the arm D grid tile and at full resolution
    layer 2   each positive and negative of a bin that passed the layer 1 gate (score.py gate.json), at full
              resolution, after three reference examples of the same track and bin from other patients (full
              resolution, with their size, description and location from the findings extraction)

Models Sonnet 5.5 and Opus 5.5 on Vertex (som-nero-plevriti-deidbdf, global), this host's application-default login.
Every call asks for JSON: answer present/absent, side, where, confidence. A refusal is recorded as its own outcome:
no fallback model, since a fallback would mix two models' answers into one score.

    cd optimizer/ct-findings-5
    uv run -q --no-project --with "anthropic[vertex]" python -B small_finding/run.py cost-check     # 20 items
    uv run -q --no-project --with "anthropic[vertex]" python -B small_finding/run.py layer1
    uv run -q --no-project --with "anthropic[vertex]" python -B small_finding/run.py layer2
    uv run -q --no-project --with "anthropic[vertex]" python -B small_finding/run.py zoom         # zoomed organ view

Results: PRIVATE/results.jsonl, one line per call, keyed (layer, model, view, item); a rerun skips calls already
answered. Spend is the sum of every line's cost; a run stops when spend plus the worst case of the calls in flight
would pass CAP_USD ($90, spec). Prices are Anthropic list prices per million tokens; Vertex bills the GCP project.
Prints counts and dollars only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import items as it  # noqa: E402
import render as rd  # noqa: E402

PROJECT, REGION = "som-nero-plevriti-deidbdf", "global"
MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}  # $ per M tokens: input, output
CAP_USD = float(os.environ.get("SF_CAP_USD", "90"))
EFFORT = os.environ.get("SF_EFFORT", "medium")  # the same for both models (Opus 5.5 defaults to medium, Sonnet to high)
MAX_TOKENS = 4000
CONCURRENCY = int(os.environ.get("SF_CONCURRENCY", "4"))
RESULTS = it.PRIVATE / "results.jsonl"
GATE = it.OUT / "gate.json"

SYSTEM = ("You are a radiologist reading CT images. Answer only from what you can see in the image you are asked "
          "about. Images are axial slices in radiology convention: the patient's right is on the image's left, "
          "anterior at the top.")
QUESTION = {
    "lung": ("Is there at least one lung nodule (solid, part-solid or ground-glass) visible on this axial slice? "
             "If present, give the side and where in the lung."),
    "liver": ("Is there at least one focal liver lesion (a focal area of different density: cyst, hypodensity, mass "
              "or metastasis) visible on this axial slice? If present, give the side (right or left lobe) and where."),
    "lymph_node": ("Is there at least one lymph node on this axial slice (mediastinum, hila or axillae) that a "
                   "radiologist would mention in a report: enlarged, borderline or otherwise notable? If present, give "
                   "the side and the station or region."),
}
QKEY = {"lung_solid": "lung", "lung_ground_glass": "lung", "liver": "liver", "lymph_node": "lymph_node"}
NOUN = {"lung_solid": "lung nodule", "lung_ground_glass": "ground-glass lung nodule", "liver": "liver lesion",
        "lymph_node": "lymph node"}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["answer", "side", "where", "confidence"],
          "properties": {"answer": {"type": "string", "enum": ["present", "absent"]},
                         "side": {"type": "string", "enum": ["right", "left", "both", "midline", "none", "unsure"]},
                         "where": {"type": "string"},
                         "confidence": {"type": "integer", "description": "0-100: how sure you are of the answer"}}}

_lock = threading.Lock()


def _img(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(path.read_bytes()).decode()}}


def content_for(item: dict, view: str, examples: list[dict] | None) -> list[dict]:
    win = rd.WINDOW_NAME[rd.WINDOW[item["track"]]]
    out = []
    if examples and "kind" in examples[0]:  # layer 3: report-matched (oracle) set, the same three kinds for every case
        out.append({"type": "text", "text": (f"First, {len(examples)} reference images from other patients ({win}). "
                                             f"Each says what the radiologist reported on that slice.")})
        for k, ex in enumerate(examples, 1):
            if ex["kind"] == "no":
                cap = f"no {NOUN[item['track']]} on this slice (the report says this region is clear); location: {ex['site']}"
            else:
                cap = f"a confirmed {NOUN[item['track']]}: {ex['desc']}; location: {ex['location']}"
            out.append({"type": "text", "text": f"Reference {k}: {cap}."})
            out.append(_img(rd.path_of(ex["item"], view)))
        out.append({"type": "text", "text": "Now the image to read, from a different patient:"})
    elif examples:
        out.append({"type": "text", "text": (f"First, {len(examples)} reference images from other patients, each "
                                             f"with a confirmed {NOUN[item['track']]} that the radiologist reported "
                                             f"on that slice ({win}).")})
        for k, ex in enumerate(examples, 1):
            out.append({"type": "text", "text": f"Reference {k}: {ex['desc']}; location: {ex['location']}."})
            out.append(_img(rd.path_of(ex["item"], "zoom" if view == "zoom" else "full")))  # references match the target's view
        out.append({"type": "text", "text": "Now the image to read, from a different patient:"})
    out.append(_img(rd.path_of(item["item"], view)))
    out.append({"type": "text", "text": f"({win}) {QUESTION[QKEY[item['track']]]}"})
    return out


def worst_case(model: str) -> float:
    pin, pout = PRICE[model]
    return (20000 * pin + MAX_TOKENS * pout) / 1e6


def call(client, model: str, content: list[dict]) -> dict:
    import anthropic

    t0 = time.time()
    try:
        r = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
                                   messages=[{"role": "user", "content": content}],
                                   output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
    except (anthropic.BadRequestError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        return {"status": "error", "error": type(e).__name__, "cost_usd": 0.0, "seconds": time.time() - t0}
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        return {"status": "retry", "error": type(e).__name__, "cost_usd": 0.0, "seconds": time.time() - t0}
    pin, pout = PRICE[model]
    u = r.usage
    cost = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
    base = {"cost_usd": cost, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "stop_reason": r.stop_reason, "seconds": round(time.time() - t0, 2)}
    if r.stop_reason == "refusal":
        return {"status": "refusal", **base}
    try:
        ans = json.loads(next(b.text for b in r.content if b.type == "text"))
    except Exception:  # noqa: BLE001 - e.g. max_tokens cut the JSON; counted, retried on a rerun
        return {"status": "unparsed", **base}
    return {"status": "ok", **base, **{k: ans.get(k) for k in ("answer", "side", "where", "confidence")}}


def done_keys() -> tuple[set, float]:
    keys, spent = set(), 0.0
    if RESULTS.is_file():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            spent += r.get("cost_usd") or 0.0
            if r["status"] in ("ok", "refusal", "error"):
                keys.add((r["layer"], r["model"], r["view"], r["item"]))
    return keys, spent


def run_jobs(jobs: list[dict], label: str) -> int:
    import anthropic

    client = anthropic.AnthropicVertex(project_id=PROJECT, region=REGION, max_retries=4)
    keys, spent = done_keys()
    jobs = [j for j in jobs if (j["layer"], j["model"], j["view"], j["item"]["item"]) not in keys]
    print(f"{label}: {len(jobs)} calls to make; spent so far ${spent:.2f} of ${CAP_USD:.0f}", flush=True)
    state = {"spent": spent, "reserved": 0.0, "stopped": False}
    st = Counter()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    def one(j):
        w = worst_case(j["model"])
        with _lock:
            if state["stopped"] or state["spent"] + state["reserved"] + w > CAP_USD:
                state["stopped"] = True
                return "skipped (cap)"
            state["reserved"] += w
        res = {"status": "retry"}
        for attempt in range(3):
            res = call(client, j["model"], content_for(j["item"], j["view"], j.get("examples")))
            if res["status"] != "retry":
                break
            time.sleep(10 * (attempt + 1))
        row = {"layer": j["layer"], "model": j["model"], "view": j["view"], "item": j["item"]["item"],
               "examples": [e["item"] for e in j.get("examples") or []], "effort": EFFORT, **res}
        with _lock:
            state["reserved"] -= w
            state["spent"] += res.get("cost_usd") or 0.0
            with RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        return res["status"]

    with ThreadPoolExecutor(CONCURRENCY) as ex:
        for n, f in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            st[f.result()] += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} {dict(st)} spent ${state['spent']:.2f}", flush=True)
    print(f"{label} done: {dict(st)}; spent ${state['spent']:.2f}", flush=True)
    return 2 if state["stopped"] else 0


def test_items() -> list[dict]:
    return [i for i in it.load_items() if i["role"] in ("positive", "negative")]


def layer1_jobs(items: list[dict]) -> list[dict]:
    jobs = [{"layer": 1, "model": m, "view": v, "item": i} for i in items for m in MODELS for v in ("tile", "full")]
    random.Random(it.SEED).shuffle(jobs)  # a partial run still covers every bin
    return jobs


def cmd_cost_check(_args) -> int:
    """20 items, both models, both views (80 calls): positives and negatives spread over tracks and bins. Prints the
    projected total for the whole study and refuses (exit 3) when it passes the cap."""
    items = test_items()
    rng = random.Random(it.SEED + 1)
    by = defaultdict(list)
    for i in items:
        by[(i["track"], i["bin"], i["role"])].append(i)
    cells = sorted(by)
    rng.shuffle(cells)
    pick = [rng.choice(by[c]) for c in cells[:20]]
    rc = run_jobs(layer1_jobs(pick), "cost check")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l]
    picked = {i["item"] for i in pick}
    rows = [r for r in rows if r["item"] in picked and r["layer"] == 1]
    mean = defaultdict(list)
    for r in rows:
        mean[(r["model"], r["view"])].append(r)
    n_test = len(items)
    proj1 = 0.0
    print("\nper call (cost check):")
    for (m, v), rs in sorted(mean.items()):
        c = sum(r["cost_usd"] for r in rs) / len(rs)
        tin = sum(r.get("input_tokens", 0) for r in rs) / len(rs)
        tout = sum(r.get("output_tokens", 0) for r in rs) / len(rs)
        proj1 += c * n_test
        print(f"  {m:<18} {v:<5} n={len(rs):>3}  ${c:.4f}  input {tin:.0f} tok  output {tout:.0f} tok  "
              f"statuses {dict(Counter(r['status'] for r in rs))}")
    # layer 2: full view plus three more full images, on at most every bin; image input priced from the full view
    proj2 = 0.0
    for m in MODELS:
        rs = mean.get((m, "full"), [])
        if rs:
            pin, _ = PRICE[m]
            c_full = sum(r["cost_usd"] for r in rs) / len(rs)
            img_in = sum(r.get("input_tokens", 0) for r in rs) / len(rs)
            proj2 += (c_full + 3 * img_in * pin / 1e6) * n_test
    _, spent = done_keys()
    total = proj1 + proj2
    print(f"\nprojection: layer 1 ${proj1:.2f} + layer 2 (if every bin passes) ${proj2:.2f} = ${total:.2f}; "
          f"cap ${CAP_USD:.0f}; spent so far ${spent:.2f}")
    if total > CAP_USD:
        print("OVER CAP: stop and come back to Phil (spec)")
        return 3
    print("UNDER CAP")
    return rc


def cmd_layer1(_args) -> int:
    return run_jobs(layer1_jobs(test_items()), "layer 1")


def cmd_zoom(_args) -> int:
    """Layer 1 on the zoomed organ view (render.py zoom; Phil 2026-11-24): same items, both models, no examples.
    Items whose organ is absent from the mask on that slice have no zoom image and are skipped (counted)."""
    items = test_items()
    have = [i for i in items if rd.path_of(i["item"], "zoom").is_file()]
    print(f"zoom: {len(have)} of {len(items)} items have a zoom view")
    jobs = [{"layer": 1, "model": m, "view": "zoom", "item": i} for i in have for m in MODELS]
    random.Random(it.SEED + 3).shuffle(jobs)
    return run_jobs(jobs, "zoom")


def layer2_jobs(all_items: list[dict], gate: dict, view: str = "full", bins: tuple | None = None) -> tuple[list[dict], set, set]:
    """(jobs, bins passing for at least one model, bins skipped for want of 3 examples). A model runs layer 2 only
    on the bins where it passed its own gate; every positive and negative of such a bin gets three references drawn
    (seeded per item) from that bin's example pool, the same three for both models."""
    ok = {(g["model"], g["track"], g["bin"]) for g in gate["bins"] if g["pass"] or bins}  # bins given: run them all
    if bins:
        ok = {x for x in ok if x[2] in bins}
    passed = {(t, b) for _m, t, b in ok}
    pool = defaultdict(list)
    for i in all_items:
        if i["role"] == "example":
            pool[(i["track"], i["bin"])].append(i)
    jobs, skipped = [], set()
    for i in (x for x in all_items if x["role"] in ("positive", "negative")):
        cell = (i["track"], i["bin"])
        if cell not in passed:
            continue
        exs = sorted(pool[cell], key=lambda e: e["item"])
        if len(exs) < 3:  # liver <5 mm has no spare lesions to show: reported as skipped, not run
            skipped.add(cell)
            continue
        k = random.Random(f"{it.SEED}:{i['item']}").sample(exs, 3)
        jobs += [{"layer": 2, "model": m, "view": view, "item": i, "examples": k} for m in MODELS if (m, *cell) in ok]
    random.Random(it.SEED + 2).shuffle(jobs)
    return jobs, passed, skipped


def cmd_layer2(_args) -> int:
    jobs, passed, skipped = layer2_jobs(it.load_items(), json.loads(GATE.read_text()))
    print(f"layer 2: bins passing for at least one model: {len(passed)}; skipped for want of 3 examples: "
          f"{sorted(' '.join(c) for c in skipped)}")
    return run_jobs(jobs, "layer 2")


def cmd_layer2_zoom(_args) -> int:
    """Layer 2 on the zoom view (Phil 2026-11-24): zoomed target plus three zoomed references, both models, on the
    5-9 and 10-19 mm bins of every track (where zoom moved layer 1), no gate. Same seeded references as layer 2."""
    jobs, _passed, skipped = layer2_jobs(it.load_items(), json.loads(GATE.read_text()), view="zoom", bins=("5-9", "10-19"))
    print(f"layer 2 zoom: {len(jobs)} calls; skipped for want of 3 examples: {sorted(' '.join(c) for c in skipped)}")
    return run_jobs(jobs, "layer 2 zoom")


def layer3_jobs(all_items: list[dict], bins=("5-9", "10-19"), view: str = "zoom") -> tuple[list[dict], dict]:
    """Layer 3 (spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md): every positive and negative of the
    bins the zoomed layer 2 ran, with its report-matched set (oracle_match.perception_sets: yes at its site, no at its
    site, decoy elsewhere, shuffled), both models, the zoom view (layer-1 AUROC higher at 5-9 mm, 2026-11-24)."""
    sys.path.insert(0, str(HERE.parent))
    import oracle_match as om

    sets, close = om.perception_sets(all_items, it.SEED, bins=bins)
    by = {i["item"]: i for i in all_items}
    jobs = [{"layer": 3, "model": m, "view": view, "item": by[k], "examples": v} for k, v in sorted(sets.items()) for m in MODELS]
    random.Random(it.SEED + 3).shuffle(jobs)
    return jobs, dict(close)


def cmd_layer3(args) -> int:
    jobs, close = layer3_jobs(it.load_items())
    if args.limit:
        keep = sorted({j["item"]["item"] for j in jobs})[: args.limit]
        jobs = [j for j in jobs if j["item"]["item"] in keep]
    print(f"layer 3: {len(jobs)} calls; matching {close}")
    return run_jobs(jobs, "layer 3" + (f" (first {args.limit} items)" if args.limit else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    l3 = sub.add_parser("layer3")
    l3.add_argument("--limit", type=int, default=0, help="the first N items only (the cost check)")
    for name in ("cost-check", "layer1", "layer2", "zoom", "layer2-zoom"):
        sub.add_parser(name)
    args = ap.parse_args()
    return {"cost-check": cmd_cost_check, "layer1": cmd_layer1, "layer2": cmd_layer2, "zoom": cmd_zoom, "layer2-zoom": cmd_layer2_zoom, "layer3": cmd_layer3}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
