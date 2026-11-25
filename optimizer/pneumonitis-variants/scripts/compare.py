"""The paired comparison (spec, Done when): per-variant one-vs-rest AUROC of the zero-shot and guide arms on the same VAL
scans, the paired difference (guide - zero-shot) on the five-variant mean with a patient bootstrap 95% interval
(2,000 resamples of patients, both arms scored on each resample), and the MerlinOnc probe on the same scans beside
them. Win = that interval above zero. A failed answer counts 0.5 on every variant (adapters.prob), as in scoring.
Counts and AUROC only.

    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/compare.py --zero ZERO_ANSWERS --guide GUIDE_ANSWERS --out DIR
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
    ap.add_argument("--zero", type=Path, required=True)
    ap.add_argument("--guide", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gold = {r["id"]: r for r in ad.read_split("val")}
    groups = json.loads((config.DATA_DIR / "groups.json").read_text())
    za = json.loads(args.zero.read_text())["answers"]
    ga = json.loads(args.guide.read_text())["answers"]
    ids = sorted(set(za) & set(ga) & set(gold))
    if len(ids) != len(gold):
        print(f"note: {len(gold) - len(ids)} VAL scans are missing from one arm; compared on {len(ids)}")
    zero, guide = arm_probs(za), arm_probs(ga)
    pred_path = config.ID_MAP_DIR / "probe_val_predictions.json"
    probe = json.loads(pred_path.read_text()) if pred_path.is_file() else None
    zm, zper = mean_auroc(gold, zero, ids)
    gm, gper = mean_auroc(gold, guide, ids)
    obs, ci = paired_boot(gold, zero, guide, groups, ids)
    out = {"scans": len(ids), "patients": len({groups[i] for i in ids}),
           "positives": {v: sum(gold[i][v] for i in ids) for v in VARIANTS},
           "zero_shot": {"mean": round(zm, 4), **{v: round(x, 4) for v, x in zper.items()},
                         "failed": sum(bool(za[i].get("error")) for i in ids)},
           "guide": {"mean": round(gm, 4), **{v: round(x, 4) for v, x in gper.items()},
                     "failed": sum(bool(ga[i].get("error")) for i in ids)},
           "guide_minus_zero_shot": {"mean": round(obs, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                                     "win": ci[0] > 0}}
    if probe:
        pm, pper = mean_auroc(gold, probe, ids)
        pobs, pci = paired_boot(gold, probe, guide, groups, ids, seed=1)
        out["merlinonc_probe"] = {"mean": round(pm, 4), **{v: round(x, 4) for v, x in pper.items()}}
        out["guide_minus_probe"] = {"mean": round(pobs, 4), "ci95": [round(pci[0], 4), round(pci[1], 4)]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(out, indent=1))
    head = "| arm | " + " | ".join(VARIANTS) + " | mean |"
    lines = [f"VAL: {out['scans']} scans, {out['patients']} patients; positives "
             + ", ".join(f"{v} {n}" for v, n in out["positives"].items()), "", head,
             "|" + "---|" * (len(VARIANTS) + 2)]
    for name, key in (("zero-shot reader", "zero_shot"), ("guide reader (round 5)", "guide"), ("MerlinOnc probe", "merlinonc_probe")):
        if key in out:
            lines.append(f"| {name} | " + " | ".join(f"{out[key][v]:.3f}" for v in VARIANTS) + f" | {out[key]['mean']:.3f} |")
    d = out["guide_minus_zero_shot"]
    lines += ["", f"Guide - zero-shot on the mean: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                  f"(patient bootstrap {N_BOOT}); win: {'yes' if d['win'] else 'no'}."]
    if "guide_minus_probe" in out:
        e = out["guide_minus_probe"]
        lines.append(f"Guide - MerlinOnc probe on the mean: {e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}].")
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
