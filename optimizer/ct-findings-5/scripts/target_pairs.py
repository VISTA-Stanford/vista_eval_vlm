"""Two-target AUROC (mean of lymph nodes and pneumonitis) for VAL passes, and paired bootstrap intervals between them
(spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md: Half A's win is oracle minus forced-similarity on
this mean). Scans resampled together across passes, 2,000 draws; a failed answer counts 0.5 (the task scorer's rule).

    python3 scripts/target_pairs.py --out OUT.json NAME=answers.json [NAME=answers.json ...]

Every pass after the first two names is paired against the first two (the oracle arms), and the first two against
each other. Prints AUROCs and intervals only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
TARGETS = ("lesion_lymph_nodes", "pneumonitis")


def auroc(pairs) -> float | None:
    pos = [p for g, p in pairs if g == 1]
    neg = [p for g, p in pairs if g == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def tmean(ans: dict, gold: dict, ids) -> float | None:
    vals = []
    for q in TARGETS:
        pairs = [(gold[i][q], 0.5 if ans[i].get("error") or ans[i].get(q) is None else ans[i][q])
                 for i in ids if gold[i].get(q) is not None]
        a = auroc(pairs)
        if a is None:
            return None
        vals.append(a)
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("passes", nargs="+")
    args = ap.parse_args()
    import config

    gold = {json.loads(l)["id"]: json.loads(l) for l in (config.DATA_DIR / "splits" / "val.jsonl").read_text().splitlines() if l}
    P = {}
    for spec in args.passes:
        name, path = spec.split("=", 1)
        P[name] = json.loads(Path(path).read_text())["answers"]
    ids = sorted(set.intersection(*[set(a) for a in P.values()]))
    names = list(P)
    pairs = [(names[0], names[1])] + [(o, x) for o in names[:2] for x in names[2:]]
    rng = random.Random(0)
    out = {"scans": len(ids), "targets_mean": {n: round(tmean(P[n], gold, ids), 4) for n in names}, "paired": {}}
    for a, b in pairs:
        d0 = tmean(P[a], gold, ids) - tmean(P[b], gold, ids)
        ds = []
        while len(ds) < 2000:
            s = [rng.choice(ids) for _ in ids]
            x, y = tmean(P[a], gold, s), tmean(P[b], gold, s)
            if x is not None and y is not None:
                ds.append(x - y)
        ds.sort()
        out["paired"][f"{a} - {b}"] = {"diff": round(d0, 4), "ci95": [round(ds[50], 4), round(ds[1949], 4)]}
        print(f"{a} - {b}: {d0:+.3f} [{ds[50]:+.3f}, {ds[1949]:+.3f}]")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["targets_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
