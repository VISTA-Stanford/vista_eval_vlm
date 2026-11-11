"""Does the agent just follow the probes? (config.PROBES; spec planning/vista_eval_vlm/2026-11-09-died-2yr-ehr-probe-tool.md)

For one scored VAL pass of the probe arm, per horizon:
  1. tool use: the share of sessions whose transcript has a tool call touching probes.json or probe_risk (reading
     the instructions that mention it does not count);
  2. rank correlation (Spearman) of the agent's probability with the six-probe combination and with each probe;
  3. the share of patients where the agent is within 5 points of the combination;
  4. patients where they differ by more than 15 points: how many, and AUROC of the agent vs the combination on them.
Failed sessions (no usable answer) are counted and left out of 2-4. Stop and come back (spec) if under half the
sessions ran the tool, or the rank correlation with the combination is above 0.95.

    ANSWER=probability WITH_SUMMARY=1 PROBES=1 python scripts/probe_following.py \
        --answers <work folder>/program-outputs/archive/<VAL pass>/answers.json [--out <file.json>]

Reads the pass's transcripts beside answers.json, config.PROBES_DIR/<id>.json and the combination written by
scripts/build_probe_tool.py, and the task's VAL labels. Prints aggregates only, never an id or a row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters as ad  # noqa: E402
import config  # noqa: E402
from build_probe_tool import PROBES  # noqa: E402

COMBINATION = config.PRIVATE_DIR / "probes" / "combination_val.parquet"
HORIZONS = (("1 year", "died_within_1_year", "label_1yr"), ("2 years", "died_within_2_years", "label"))
NEAR, FAR = 0.05, 0.15  # 5 and 15 points on the 0-100 scale; answers are stored 0-1
STOP_TOOL_SHARE, STOP_CORRELATION = 0.5, 0.95
MARKERS = ("probes.json", "probe_risk")


def used_tool(pass_dir: Path, call_id: str) -> bool | None:
    """True if the session's latest transcript has a tool call that touches the probe file or tool; None if no
    transcript."""
    root = pass_dir / "transcripts" / call_id
    events = sorted(root.glob("attempt*/*/*/events.jsonl"))
    if not events:
        return None
    for line in events[-1].read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "assistant":
            continue
        for item in (ev.get("message") or {}).get("content") or []:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                if any(m in json.dumps(item.get("input", {})) for m in MARKERS):
                    return True
    return False


def probe_scores(i: str, words: str) -> dict[str, float]:
    data = json.loads((config.PROBES_DIR / f"{i}.json").read_text(encoding="utf-8"))
    return {p["probe"]: p["horizons"][words]["this_patient_score"] for p in data["probes"]
            if p["horizons"][words]["available"]}


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else None


def main() -> int:
    if not config.PROBES:
        raise SystemExit("run with PROBES=1 (and ANSWER=probability WITH_SUMMARY=1)")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    answers = json.loads(a.answers.read_text(encoding="utf-8"))["answers"]
    gold = {r["id"]: r for r in ad.read_split("val")}
    if set(answers) - set(gold):
        raise SystemExit("the pass holds patients that are not VAL patients of this task")
    comb = pd.read_parquet(COMBINATION)
    tool = {i: used_tool(a.answers.parent, i) for i in answers}
    seen = [v for v in tool.values() if v is not None]
    out = {"pass": a.answers.parent.name, "n": len(answers), "with_transcript": len(seen),
           "ran_tool_share": round(float(np.mean(seen)), 3) if seen else None, "horizons": {}}
    stop = []
    if seen and np.mean(seen) < STOP_TOOL_SHARE:
        stop.append(f"only {np.mean(seen):.0%} of sessions ran the tool (rule: at least {STOP_TOOL_SHARE:.0%})")
    print(f"{out['pass']}: {len(answers)} VAL patients; {len(seen)} transcripts; ran the probe tool in "
          f"{out['ran_tool_share']:.0%}" if seen else f"{out['pass']}: no transcripts found")
    for words, key, field in HORIZONS:
        c = dict(zip(*[comb.loc[comb["horizon"] == words, k] for k in ("id", "score")]))
        ok = [i for i in answers if not answers[i].get("error") and i in c]
        failed = sum(1 for i in answers if answers[i].get("error"))
        agent = np.array([ad.prob(answers[i], key) for i in ok])
        combo = np.array([c[i] for i in ok])
        y = np.array([gold[i][field] for i in ok])
        rho = float(spearmanr(agent, combo).statistic)
        per_probe = {}
        for k, *_ in PROBES:
            pairs = [(ad.prob(answers[i], key), probe_scores(i, words).get(k)) for i in ok]
            pairs = [(x, s) for x, s in pairs if s is not None]
            if len(pairs) > 10:
                per_probe[k] = {"n": len(pairs), "spearman": round(float(spearmanr(*zip(*pairs)).statistic), 3)}
        far = np.abs(agent - combo) > FAR
        h = {"n_scored": len(ok), "failed": failed,
             "spearman_with_combination": round(rho, 3),
             "within_5_points_share": round(float(np.mean(np.abs(agent - combo) <= NEAR)), 3),
             "over_15_points": {"n": int(far.sum()), "died": int(y[far].sum()),
                                "agent_auroc": _auc(y[far], agent[far]), "combination_auroc": _auc(y[far], combo[far])},
             "agent_auroc": _auc(y, agent), "combination_auroc": _auc(y, combo), "spearman_with_probe": per_probe}
        out["horizons"][words] = h
        if rho > STOP_CORRELATION:
            stop.append(f"{words}: rank correlation with the combination {rho:.3f} (rule: at most {STOP_CORRELATION})")
        f = h["over_15_points"]
        fa = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  {words}: n={len(ok)} ({failed} failed); AUROC agent {fa(h['agent_auroc'])} vs combination "
              f"{fa(h['combination_auroc'])}; Spearman with combination {rho:.3f}; within 5 points "
              f"{h['within_5_points_share']:.0%}; >15 points apart: {f['n']} patients ({f['died']} died), AUROC agent "
              f"{fa(f['agent_auroc'])} vs combination {fa(f['combination_auroc'])}")
        print("    Spearman with each probe: " + ", ".join(f"{k} {v['spearman']:.2f} (n={v['n']})"
                                                      for k, v in per_probe.items()))
    out["stop_and_ask"] = stop
    print("STOP AND ASK: " + "; ".join(stop) if stop else "stop rules: none tripped")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
