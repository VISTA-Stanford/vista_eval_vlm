"""Score the small-finding perception calls (spec planning/vista_eval_vlm/2026-11-23-small-finding-perception.md).

    layer1   per track x size bin x model x view: sensitivity (present on positives), false-positive rate (present on
             negatives), their difference with a 95% interval, and the secondary where-agreement (the said side vs
             the side in the lesion's location, where the location names one). Writes OUT/layer1.csv and
             OUT/gate.json: the layer 2 gate on the full-resolution view.
    layer2   per bin that ran layer 2: layer 2 minus layer 1 (full view) on the same items, for sensitivity and the
             false-positive rate, with a paired bootstrap 95% interval. Writes OUT/layer2.csv.

Gate (spec): sensitivity - false-positive rate > 0.1 and the interval's low end > 0. The interval is the normal
approximation, sqrt(p1(1-p1)/n1 + p0(1-p0)/n0), so it can be checked by hand. A refused or unparsed call counts as
"absent" (it did not report the finding) and is counted beside the table.

    cd optimizer/ct-findings-5
    python3 -B small_finding/score.py layer1
    python3 -B small_finding/score.py layer2

Counts and rates only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GATE_MIN = 0.1
Z95 = 1.96


def gate(pos: list[int], neg: list[int]) -> dict:
    if not pos or not neg:
        return {"sens": None, "fpr": None, "diff": None, "lo": None, "hi": None, "pass": False}
    p1, p0 = sum(pos) / len(pos), sum(neg) / len(neg)
    se = math.sqrt(p1 * (1 - p1) / len(pos) + p0 * (1 - p0) / len(neg))
    d = p1 - p0
    lo, hi = d - Z95 * se, d + Z95 * se
    return {"sens": p1, "fpr": p0, "diff": d, "lo": lo, "hi": hi, "pass": d > GATE_MIN and lo > 0}


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k of n: well behaved at 0 and n, where the normal approximation collapses."""
    if not n:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return centre - half, centre + half


def paired_boot(a: list[int], b: list[int], n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """mean(b) - mean(a) on paired items, with a bootstrap 95% interval."""
    if not a:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    k = len(a)
    d = sum(b) / k - sum(a) / k
    ds = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        ds.append(sum(b[i] - a[i] for i in idx) / k)
    ds.sort()
    return d, ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def net_boot(pos: tuple[list, list], neg: tuple[list, list], n: int = 2000, seed: int = 9):
    """Change in (sensitivity - false-positive rate) from layer 1 to layer 2: did the examples improve telling
    lesion from no lesion, or only move the threshold? Positives and negatives are resampled separately (different
    items), each keeping its layer 1 / layer 2 pairing."""
    (p1, p2), (q1, q2) = pos, neg
    if not p1 or not q1:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    f = lambda a, b, idx: sum(b[i] - a[i] for i in idx) / len(idx)
    d = f(p1, p2, range(len(p1))) - f(q1, q2, range(len(q1)))
    ds = sorted(f(p1, p2, [rng.randrange(len(p1)) for _ in p1]) - f(q1, q2, [rng.randrange(len(q1)) for _ in q1])
                for _ in range(n))
    return d, ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def _load():
    import items as it

    items = {i["item"]: i for i in it.load_items()}
    rows = [json.loads(l) for l in (it.PRIVATE / "results.jsonl").read_text(encoding="utf-8").splitlines() if l]
    latest = {}
    for r in rows:  # one answer per key: the last ok/refusal/unparsed line
        if r["status"] in ("ok", "refusal", "unparsed", "error"):
            latest[(r["layer"], r["model"], r["view"], r["item"])] = r
    return it, items, latest


def said(r: dict) -> int:
    return int(r["status"] == "ok" and r.get("answer") == "present")


def cmd_layer1(_args) -> int:
    it, items, latest = _load()
    cells = defaultdict(lambda: {"pos": [], "neg": [], "side_ok": 0, "side_n": 0, "not_ok": Counter()})
    for (layer, model, view, iid), r in latest.items():
        if layer != 1:
            continue
        i = items[iid]
        c = cells[(i["track"], i["bin"], model, view)]
        if r["status"] != "ok":
            c["not_ok"][r["status"]] += 1
        (c["pos"] if i["role"] == "positive" else c["neg"]).append(said(r))
        if i["role"] == "positive" and said(r) and i.get("side") and r.get("side") in ("right", "left"):
            c["side_n"] += 1
            c["side_ok"] += int(r["side"] == i["side"])
    out, gate_rows = [], []
    for (t, b, m, v), c in sorted(cells.items(), key=lambda kv: (it.TRACKS.index(kv[0][0]), it.BINS.index(kv[0][1]),
                                                                    kv[0][2], kv[0][3])):
        g = gate(c["pos"], c["neg"])
        s_lo, s_hi = wilson(sum(c["pos"]), len(c["pos"]))
        f_lo, f_hi = wilson(sum(c["neg"]), len(c["neg"]))
        # the gate's normal-approximation interval is too narrow when a rate is 0 or 1, or with few items
        note = "; ".join(x for x in ("few items" if min(len(c["pos"]), len(c["neg"])) < 40 else "",
                                     "a rate at 0 or 1: gate interval too narrow" if {sum(c["pos"]), sum(c["neg"])} & {0}
                                     or sum(c["pos"]) == len(c["pos"]) or sum(c["neg"]) == len(c["neg"]) else "") if x)
        row = {"track": t, "bin": b, "model": m, "view": v, "n_pos": len(c["pos"]), "n_neg": len(c["neg"]),
               **{k: (round(x, 3) if isinstance(x, float) else x) for k, x in g.items()},
               "sens_lo": round(s_lo, 3), "sens_hi": round(s_hi, 3), "fpr_lo": round(f_lo, 3), "fpr_hi": round(f_hi, 3),
               "note": note,
               "side_agree": round(c["side_ok"] / c["side_n"], 3) if c["side_n"] else None, "side_n": c["side_n"],
               "not_ok": dict(c["not_ok"])}
        out.append(row)
        if v == "full":
            gate_rows.append({"track": t, "bin": b, "model": m, "pass": g["pass"], "diff": row["diff"], "lo": row["lo"]})
    it.OUT.mkdir(parents=True, exist_ok=True)
    with (it.OUT / "layer1.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (it.OUT / "gate.json").write_text(json.dumps({"rule": f"sens - fpr > {GATE_MIN} and 95% low end > 0 (full view)",
                                                  "bins": gate_rows}, indent=1))
    print("track              bin    model              view  n+/n-   sens   fpr   diff [95%]          gate  side")
    for r in out:
        print(f"{r['track']:<18} {r['bin']:<6} {r['model']:<18} {r['view']:<5} {r['n_pos']:>2}/{r['n_neg']:<2}  "
              f"{r['sens']:.2f}  {r['fpr']:.2f}  {r['diff']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]  "
              f"{'PASS' if r['pass'] else 'fail':<4}  {r['side_agree'] if r['side_agree'] is not None else '-'}"
              f"{'  not ok ' + str(r['not_ok']) if r['not_ok'] else ''}{'  (' + r['note'] + ')' if r['note'] else ''}")
    return 0


def cmd_layer2(_args) -> int:
    it, items, latest = _load()
    pairs = defaultdict(lambda: {"pos": ([], []), "neg": ([], [])})
    for (layer, model, view, iid), r in latest.items():
        if layer != 2:
            continue
        r1 = latest.get((1, model, "full", iid))
        if not r1:
            continue
        i = items[iid]
        a, b = pairs[(i["track"], i["bin"], model)]["pos" if i["role"] == "positive" else "neg"]
        a.append(said(r1))
        b.append(said(r))
    out = []
    for (t, bn, m), p in sorted(pairs.items()):
        ds = paired_boot(*p["pos"])
        df = paired_boot(*p["neg"], seed=8)
        dn = net_boot(p["pos"], p["neg"])
        out.append({"net_change": round(dn[0], 3), "net_lo": round(dn[1], 3), "net_hi": round(dn[2], 3),"track": t, "bin": bn, "model": m, "n_pos": len(p["pos"][0]), "n_neg": len(p["neg"][0]),
                    "sens_l1": round(sum(p["pos"][0]) / max(len(p["pos"][0]), 1), 3),
                    "sens_l2": round(sum(p["pos"][1]) / max(len(p["pos"][1]), 1), 3),
                    "sens_change": round(ds[0], 3), "sens_lo": round(ds[1], 3), "sens_hi": round(ds[2], 3),
                    "fpr_l1": round(sum(p["neg"][0]) / max(len(p["neg"][0]), 1), 3),
                    "fpr_l2": round(sum(p["neg"][1]) / max(len(p["neg"][1]), 1), 3),
                    "fpr_change": round(df[0], 3), "fpr_lo": round(df[1], 3), "fpr_hi": round(df[2], 3)})
    if not out:
        print("no layer 2 results")
        return 1
    with (it.OUT / "layer2.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['track']:<18} {r['bin']:<6} {r['model']:<18} n {r['n_pos']}/{r['n_neg']}  sens {r['sens_l1']:.2f}->"
              f"{r['sens_l2']:.2f} ({r['sens_change']:+.2f} [{r['sens_lo']:+.2f},{r['sens_hi']:+.2f}])  fpr "
              f"{r['fpr_l1']:.2f}->{r['fpr_l2']:.2f} ({r['fpr_change']:+.2f} [{r['fpr_lo']:+.2f},{r['fpr_hi']:+.2f}])  "
              f"net {r['net_change']:+.2f} [{r['net_lo']:+.2f},{r['net_hi']:+.2f}]")
    return 0


def prob(r: dict) -> float:
    """The model's probability that the finding is there, from its answer and its 0-100 confidence (a refused or
    unparsed call counts as 0: it did not report the finding)."""
    if r["status"] != "ok" or r.get("confidence") is None:
        return 0.0
    c = max(0, min(100, int(r["confidence"]))) / 100
    return c if r.get("answer") == "present" else 1 - c


def auroc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative, ties counted half."""
    if not pos or not neg:
        return float("nan")
    s = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def auroc_boot(pos, neg, n: int = 2000, seed: int = 11) -> tuple[float, float]:
    rng = random.Random(seed)
    v = sorted(auroc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def auroc_change_boot(p1, p2, q1, q2, n: int = 2000, seed: int = 12) -> tuple[float, float]:
    """Paired: the same items scored two ways (e.g. full vs zoom, or layer 1 vs layer 2)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ip = [rng.randrange(len(p1)) for _ in p1]
        iq = [rng.randrange(len(q1)) for _ in q1]
        out.append(auroc([p2[i] for i in ip], [q2[i] for i in iq]) - auroc([p1[i] for i in ip], [q1[i] for i in iq]))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


def cmd_auroc(_args) -> int:
    """AUROC per track x bin x model x view (layer 1: tile, full, zoom) with a bootstrap 95% interval, the paired
    change zoom - full and layer 2 - layer 1 (full) on the same items. Writes OUT/auroc.json."""
    it, items, latest = _load()
    cells = defaultdict(lambda: {"pos": {}, "neg": {}})
    for (layer, m, v, iid), r in latest.items():
        i = items[iid]
        cells[(layer, i["track"], i["bin"], m, v)]["pos" if i["role"] == "positive" else "neg"][iid] = prob(r)
    rows = []
    for (layer, t, b, m, v), c in sorted(cells.items()):
        p, q = list(c["pos"].values()), list(c["neg"].values())
        lo, hi = auroc_boot(p, q)
        row = {"layer": layer, "track": t, "bin": b, "model": m, "view": v, "n_pos": len(p), "n_neg": len(q),
               "auroc": round(auroc(p, q), 3), "lo": round(lo, 3), "hi": round(hi, 3)}
        # paired change on the items both have: layer 1 zoom vs layer 1 full; layer 2 vs layer 1 in the same view
        base = (cells.get((1, t, b, m, v)) if layer == 2 else
                cells.get((1, t, b, m, "full")) if v == "zoom" else None)
        if base:
            ip = [k for k in c["pos"] if k in base["pos"]]
            iq = [k for k in c["neg"] if k in base["neg"]]
            p1, p2 = [base["pos"][k] for k in ip], [c["pos"][k] for k in ip]
            q1, q2 = [base["neg"][k] for k in iq], [c["neg"][k] for k in iq]
            if ip and iq:
                dlo, dhi = auroc_change_boot(p1, p2, q1, q2)
                row.update({"vs_base": f"layer 1 {v}" if layer == 2 else "layer 1 full", "vs_full_l1": round(auroc(p2, q2) - auroc(p1, q1), 3), "vs_lo": round(dlo, 3),
                            "vs_hi": round(dhi, 3), "paired_n": f"{len(ip)}/{len(iq)}"})
        rows.append(row)
    (it.OUT / "auroc.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        extra = (f"  vs {r['vs_base']} {r['vs_full_l1']:+.2f} [{r['vs_lo']:+.2f},{r['vs_hi']:+.2f}] on {r['paired_n']}"
                 if "vs_full_l1" in r else "")
        print(f"L{r['layer']} {r['track']:<18} {r['bin']:<6} {r['model']:<18} {r['view']:<5} n {r['n_pos']}/{r['n_neg']}  "
              f"AUROC {r['auroc']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]{extra}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layer1")
    sub.add_parser("layer2")
    sub.add_parser("auroc")
    args = ap.parse_args()
    return {"layer1": cmd_layer1, "layer2": cmd_layer2, "auroc": cmd_auroc}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
