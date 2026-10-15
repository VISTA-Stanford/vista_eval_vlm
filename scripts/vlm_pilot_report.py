"""Gemini pilot report (tumor-board plan V3; any registry task): readable answers, caching, measured tokens and dollars,
and the extrapolated cost of the full run (all tasks x arms in the full config).

Reads the pilot's results / errors / overflow files off the mount and prints aggregates
only. Extrapolation = each arm's mean measured dollars per call x the number of rows that
arm would send in the full run (scored labels, with a CT for CT arms, with a summary for
summary arms), counted from the board table.

    cd src && python ../scripts/vlm_pilot_report.py --pilot ../configs/all_tasks.tb.gemini_pilot.yaml \
        --full ../configs/all_tasks.tb.gemini.yaml
Exit 1 if any pilot answer is unreadable or any row failed.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from results.final_metrics import answer_to_label  # noqa: E402


def _cost_per_row(df, p):
    pt = pd.to_numeric(df["usage_prompt_tokens"], errors="coerce").fillna(0)
    ot = (pd.to_numeric(df["usage_output_tokens"], errors="coerce").fillna(0)
          + pd.to_numeric(df["usage_thinking_tokens"], errors="coerce").fillna(0))
    long = pt > p["long_context_over"]
    return ((pt * p["input"] + ot * p["output"]).where(~long, pt * p["input_long"] + ot * p["output_long"])) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--budget", type=float, default=150.0)
    args = ap.parse_args()
    pcfg = yaml.safe_load(open(args.pilot))
    fcfg = yaml.safe_load(open(args.full))
    prices = pcfg["vertex"]["price_per_mtok"]
    base = Path(pcfg["paths"]["base_dir"])
    registry = {t["task_name"]: t for t in json.load(open(base / pcfg["paths"]["valid_tasks"]))}
    model = pcfg["models"][0]["name"].split("/")[-1]

    bad = []
    per_arm = {}
    print(f"{'arm':16s} {'rows':>4s} {'err':>3s} {'ovf':>3s} {'unread':>6s} {'cached':>6s} "
          f"{'prompt tok med':>14s} {'max':>8s} {'think med':>9s} {'$ total':>8s} {'$/call':>7s}")
    for task in pcfg["tasks"]:
        info = registry[task]
        d = Path(pcfg["paths"]["results_dir"]) / info["task_source_csv"] / task / model
        for arm in pcfg["experiments"]:
            fp = d / f"{task}_results_{arm}.csv"
            df = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
            n_err = len(pd.read_csv(d / f"{task}_errors_{arm}.csv")) if (d / f"{task}_errors_{arm}.csv").exists() else 0
            n_ovf = len(pd.read_csv(d / f"{task}_overflow_{arm}.csv")) if (d / f"{task}_overflow_{arm}.csv").exists() else 0
            if df.empty:
                bad.append(f"{arm}: no results")
                continue
            unread = int(df["model_response"].apply(lambda r: answer_to_label(r, info["mapping"]) is None).sum())
            cost = _cost_per_row(df, prices)
            per_arm[(task, arm)] = float(cost.mean())
            pt = pd.to_numeric(df["usage_prompt_tokens"], errors="coerce")
            th = pd.to_numeric(df["usage_thinking_tokens"], errors="coerce")
            print(f"{arm:16s} {len(df):4d} {n_err:3d} {n_ovf:3d} {unread:6d} {int(df['cached_response'].sum()):6d} "
                  f"{int(pt.median()):14d} {int(pt.max()):8d} {int(th.median()):9d} {cost.sum():8.2f} {cost.mean():7.3f}")
            if unread:
                bad.append(f"{arm}: {unread} unreadable answers")
            if n_err:
                bad.append(f"{arm}: {n_err} failed rows")

    # Full-run row counts per arm, from the board table.
    from vista_run.run_bq import TaskOrchestrator, _CT_EXPERIMENTS, attach_prior_ct
    from context.presets import SUMMARY_EXPERIMENTS
    orch = TaskOrchestrator(args.full, fcfg["models"][0]["type"], fcfg["models"][0]["name"])
    total = 0.0
    not_piloted = set()
    print("\nExtrapolated full run (each task's own mean $/call from the pilot, else the mean of the\n"
          "piloted tasks for that arm, x rows the arm would send):")
    for task in fcfg["tasks"]:
        full, _, _ = orch._load_bq_table_task_data(registry[task])
        has_ct = full["image_series_uid"].notna() & full["image_study_uid"].notna()
        has_sum = full["compact_summary"].notna() & (full["compact_summary"].astype(str).str.strip() != "")
        for arm in fcfg["experiments"]:
            keep = pd.Series(True, index=full.index)
            if arm in _CT_EXPERIMENTS:
                keep &= has_ct
            if arm in SUMMARY_EXPERIMENTS:
                keep &= has_sum
            spec = registry[task].get("prior_ct")
            if spec and arm in spec.get("arms", sorted(_CT_EXPERIMENTS)):
                with_prior, _ = attach_prior_ct(full[keep], orch._prior_link(spec), spec)
                n = len(with_prior)
            else:
                n = int(keep.sum())
            rate = per_arm.get((task, arm))
            note = ""
            if rate is None:
                same_arm = [v for (t, a), v in per_arm.items() if a == arm]
                rate = sum(same_arm) / len(same_arm) if same_arm else None
                note = "  (rate borrowed from the piloted tasks)"
            if rate is None:
                not_piloted.add(arm)
                print(f"  {task:36s} {arm:16s} rows {n:4d}  not piloted")
                continue
            c = n * rate
            total += c
            print(f"  {task:36s} {arm:16s} rows {n:4d}  ${c:8.2f}{note}")
    if not_piloted:
        print(f"\nNot in the total (not piloted yet): {sorted(not_piloted)}")
    print(f"\nEXTRAPOLATED TOTAL, piloted arms ${total:.2f} (stop-and-ask line ${args.budget:.0f}): "
          + ("OVER, ask before the full run" if total > args.budget else "under"))
    print("PILOT " + ("PASS" if not bad else "FAIL"))
    for b in bad:
        print("  FAIL: " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
