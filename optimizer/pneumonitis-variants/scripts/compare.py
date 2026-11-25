"""The paired comparison (spec, Done when): per-variant one-vs-rest AUROC of the zero-shot and guide arms on the same VAL
scans, the paired difference (guide - zero-shot) on the five-variant mean with a patient bootstrap 95% interval
(2,000 resamples of patients, both arms scored on each resample), and the MerlinOnc probe on the same scans beside
them. Win = that interval above zero. A failed answer counts 0.5 on every variant (adapters.prob), as in scoring.
Counts and AUROC only.

    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/compare.py --zero ZERO_ANSWERS --guide GUIDE_ANSWERS --out DIR
    ... --arm atlas=ATLAS_ANSWERS --pair atlas:zero-shot --pair atlas:guide --pair guide:zero-shot   (the atlas arm)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import adapters as ad  # noqa: E402
import config  # noqa: E402

VARIANTS = tuple(k for k, _t, _q in config.QUESTIONS)
N_BOOT = 2000


def mean_auroc(gold: dict, probs: dict, ids: list[str]) -> tuple[float, dict]:
    per = {v: ad.auroc([(gold[i][v], probs[i][v]) for i in ids]) for v in VARIANTS}
    return sum(per.values()) / len(per), per


MERGED = ("radiation", "drug", "cop", "infection")


def merged_auroc(gold: dict, probs: dict, ids: list[str]) -> tuple[float, dict]:
    """Four classes, other drug and immunotherapy merged into "drug" (gold: either; probability: the larger of the
    two), one-vs-rest AUROC each and their mean (spec 2026-11-25-ct-pneumonitis-library.md)."""
    def g(i, v):
        return int(gold[i]["other_drug"] or gold[i]["immunotherapy"]) if v == "drug" else gold[i][v]

    def p(i, v):
        return max(probs[i]["other_drug"], probs[i]["immunotherapy"]) if v == "drug" else probs[i][v]

    per = {v: ad.auroc([(g(i, v), p(i, v)) for i in ids]) for v in MERGED}
    return sum(per.values()) / len(per), per


def arm_probs(answers: dict) -> dict:
    return {i: {v: ad.prob(a, v) for v in VARIANTS} for i, a in answers.items()}


def paired_boot(gold, a, b, groups, ids, seed=0, n=N_BOOT) -> tuple[float, list[float]]:
    """(observed b - a on the mean, [2.5%, 97.5%]) over patient resamples."""
    by = {}
    for i in ids:
        by.setdefault(groups[i], []).append(i)
    pats = sorted(by)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [i for g in (rng.choice(pats) for _ in pats) for i in by[g]]
        diffs.append(mean_auroc(gold, b, s)[0] - mean_auroc(gold, a, s)[0])
    diffs.sort()
    obs = mean_auroc(gold, b, ids)[0] - mean_auroc(gold, a, ids)[0]
    return obs, [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zero", type=Path, help="the zero-shot arm's answers (same as --arm zero-shot=PATH)")
    ap.add_argument("--guide", type=Path, help="the guide arm's answers (same as --arm guide=PATH)")
    ap.add_argument("--arm", action="append", default=[], help="NAME=ANSWERS_JSON, repeatable")
    ap.add_argument("--pair", action="append", default=[], help="B:A, the paired B - A on the mean; repeatable "
                                                                "(default guide:zero-shot)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms = {}
    if args.zero:
        arms["zero-shot"] = args.zero
    if args.guide:
        arms["guide"] = args.guide
    for a in args.arm:
        name, path = a.split("=", 1)
        arms[name] = Path(path)
    pairs = [tuple(p.split(":", 1)) for p in args.pair] or [("guide", "zero-shot")]
    gold = {r["id"]: r for r in ad.read_split("val")}
    groups = json.loads((config.DATA_DIR / "groups.json").read_text())
    answers = {n: json.loads(p.read_text())["answers"] for n, p in arms.items()}
    ids = sorted(set(gold).intersection(*answers.values()))
    if len(ids) != len(gold):
        print(f"note: {len(gold) - len(ids)} VAL scans are missing from an arm; compared on {len(ids)}")
    probs = {n: arm_probs(a) for n, a in answers.items()}
    pred_path = config.ID_MAP_DIR / "probe_val_predictions.json"
    if pred_path.is_file():
        probs["MerlinOnc probe"] = json.loads(pred_path.read_text())
    out = {"scans": len(ids), "patients": len({groups[i] for i in ids}),
           "positives": {v: sum(gold[i][v] for i in ids) for v in VARIANTS}, "arms": {}, "pairs": {}}
    for n, pr in probs.items():
        m, per = mean_auroc(gold, pr, ids)
        mm, mper = merged_auroc(gold, pr, ids)
        out["arms"][n] = {"mean": round(m, 4), **{v: round(x, 4) for v, x in per.items()},
                          "merged_drug_mean": round(mm, 4), "merged": {v: round(x, 4) for v, x in mper.items()},
                          "failed": sum(bool(answers[n][i].get("error")) for i in ids) if n in answers else 0}
    for seed, (b, a) in enumerate(pairs):
        obs, ci = paired_boot(gold, probs[a], probs[b], groups, ids, seed=seed)
        out["pairs"][f"{b} - {a}"] = {"mean": round(obs, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "win": ci[0] > 0}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(out, indent=1))
    lines = [f"VAL: {out['scans']} scans, {out['patients']} patients; positives "
             + ", ".join(f"{v} {n}" for v, n in out["positives"].items()), "",
             "| arm | " + " | ".join(VARIANTS) + " | mean |", "|" + "---|" * (len(VARIANTS) + 2)]
    for n, r in out["arms"].items():
        lines.append(f"| {n} | " + " | ".join(f"{r[v]:.3f}" for v in VARIANTS) + f" | {r['mean']:.3f} |")
    lines += ["", "Merged 'any drug' (other drug and immunotherapy as one; four classes):", "",
              "| arm | " + " | ".join(MERGED) + " | mean |", "|" + "---|" * (len(MERGED) + 2)]
    for n, r in out["arms"].items():
        lines.append(f"| {n} | " + " | ".join(f"{r['merged'][v]:.3f}" for v in MERGED) + f" | {r['merged_drug_mean']:.3f} |")
    lines.append("")
    for k, d in out["pairs"].items():
        lines.append(f"{k} on the mean: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                     f"(patient bootstrap {N_BOOT}); interval above zero: {'yes' if d['win'] else 'no'}.")
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
