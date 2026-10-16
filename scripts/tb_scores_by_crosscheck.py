"""Tumor-board progression scores split by whether the board and the radiology report agreed.

Spec: planning/vista_bench/tb-label-crosscheck-report-and-treatment.md (program
tumor-board-label-crosscheck, stage 1). The cross-check is built in vista_bench
(``python -m vistabench.tb_crosscheck``) and its per-answer parquet sits on the mount; this joins it
to each arm's per-patient answers and scores every stratum with the same forced-answer metrics and
class-stratified bootstrap as ``results.final_metrics --tb``, pooled over splits.

Strata: board vs report agree / disagree / mixed (report said mixed response) / no_call (no CT, no
report, procedure report, or a PET/CT with no usable trajectory), and agree / disagree split again by
report source (CT report vs PET/CT). Each arm's agreement strata add back to its pooled n in
tb_scores.csv; the script refuses to write if they don't.

    cd src && python ../scripts/tb_scores_by_crosscheck.py --config ../configs/all_tasks.tb.gemini.yaml \
        --crosscheck /mnt/su-vista-uscentral1/vistabench/ad_hoc/tb_label_crosscheck/tb_crosscheck_rows.parquet
Writes tb_scores_by_crosscheck.csv beside the parquet. Aggregates only, not suppressed by default: the file
stays on the shared mount, inside the secure environment (Phil, 2026-10-16). --suppress_small_cells hides
groups under 11 for a table that will leave the mount.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from results.final_metrics import (  # noqa: E402
    answer_to_label, bootstrap_ci, extract_experiment_from_filename, forced_answer_metrics)

TASK = "progression_assessment_answer"
SMALL_CELL = 11   # groups under this are suppressed: n shown as "<11", metrics blanked
SOURCE = {"ct_call": "ct_report", "petct_call": "petct_report"}


def strata(xc: pd.DataFrame) -> pd.DataFrame:
    """person_id -> agreement and report source, from the cross-check's progression rows."""
    p = xc[xc["task"] == TASK][["person_id", "board_report_agreement", "report_source"]].copy()
    if p["person_id"].duplicated().any():
        raise ValueError("cross-check has more than one progression row for a patient")
    p["agreement"] = p["board_report_agreement"].fillna("no_call").replace({"board_indeterminate": "no_call"})
    p["source"] = p["report_source"].map(SOURCE).fillna("none")
    return p[["person_id", "agreement", "source"]]


def score_strata(answers: pd.DataFrame, classes, n_boot=1000) -> list[dict]:
    """Rows for each agreement stratum and each agree/disagree x source cell."""
    out = []
    groups = [("agreement", g, sub) for g, sub in answers.groupby("agreement")]
    agreed = answers[answers["agreement"].isin(["agree", "disagree"])]
    groups += [("agreement_x_source", f"{a}|{s}", sub) for (a, s), sub in agreed.groupby(["agreement", "source"])]
    for kind, name, sub in groups:
        y = sub["label"].astype(int).tolist()
        pred = [None if pd.isna(v) else int(v) for v in sub["pred"]]
        row = {"stratum_kind": kind, "stratum": name, **forced_answer_metrics(y, pred, classes)}
        for k, (lo, hi) in bootstrap_ci(y, pred, classes, n_boot=n_boot).items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        out.append(row)
    return out


def suppress(rows: list[dict]) -> list[dict]:
    """Blank every group with n in 1..SMALL_CELL-1, then any group whose n could be worked out by
    subtraction: an arm's agreement groups add up to its pooled n (printed in tb_scores.csv), and
    the agree|source / disagree|source groups add up to the agree / disagree row. A group with
    exactly one hidden member gets its next smallest member hidden too; a hidden total keeps at
    least one member hidden."""
    rows = [dict(r) for r in rows]
    small = {i for i, r in enumerate(rows) if 0 < r["n"] < SMALL_CELL}
    hidden = set(small)

    def families():
        by = {}
        for i, r in enumerate(rows):
            if r["stratum_kind"] == "agreement":
                by.setdefault((r["model"], r["arm"], "agreement"), []).append(i)
            else:
                by.setdefault((r["model"], r["arm"], r["stratum"].split("|")[0]), []).append(i)
        totals = {(r["model"], r["arm"], r["stratum"]): i for i, r in enumerate(rows) if r["stratum_kind"] == "agreement"}
        return by, totals

    by, totals = families()
    while True:
        more = set()
        for (model, arm, fam), members in by.items():
            total_hidden = fam != "agreement" and totals.get((model, arm, fam)) in hidden
            n_hidden = sum(i in hidden for i in members)
            if (n_hidden == 1 and not total_hidden) or (total_hidden and n_hidden == 0):
                vis = [(rows[i]["n"], i) for i in members if i not in hidden]
                if vis and len(members) > 1:
                    more.add(min(vis)[1])
                elif total_hidden and vis:
                    more.add(min(vis)[1])
        if not more - hidden:
            break
        hidden |= more
    for i in hidden:
        r = rows[i]
        for k in list(r):
            if k not in ("task", "model", "arm", "arm_pooled_n", "stratum_kind", "stratum"):
                r[k] = None
        r["n"] = f"<{SMALL_CELL}" if i in small else "hidden"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--crosscheck", required=True, help="tb_crosscheck_rows.parquet from vistabench.tb_crosscheck")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--suppress_small_cells", action="store_true",
                    help="Hide groups under SMALL_CELL and those that would give them back (for tables leaving the mount).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = Path(cfg["paths"]["results_dir"])
    registry = {t["task_name"]: t for t in json.load(open(Path(cfg["paths"]["base_dir"]) / cfg["paths"]["valid_tasks"]))}
    info = registry[TASK]
    mapping = info["mapping"]
    unscored = [str(u) for u in info.get("unscored_labels", [-1])]
    classes = sorted(int(k) for k in mapping if str(k) not in unscored)
    st = strata(pd.read_parquet(args.crosscheck))
    pooled = pd.read_csv(results_dir / "tb_scores.csv")
    pooled = pooled[(pooled["task"] == TASK) & (pooled["split"] == "pooled")].set_index(["model", "arm"])["n"]

    rows = []
    task_dir = results_dir / info["task_source_csv"] / TASK
    for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        for fp in sorted(model_dir.glob(f"{TASK}_results_*.csv")):
            arm = extract_experiment_from_filename(fp.name)
            df = pd.read_csv(fp).drop_duplicates(subset=["index"], keep="last")
            df = df[df["label"].astype(str).isin([str(c) for c in classes])]
            df["pred"] = df["model_response"].apply(lambda r: answer_to_label(r, mapping, unscored))
            df = df.merge(st, on="person_id", how="left")
            if df["agreement"].isna().any():
                raise SystemExit(f"{fp.name}: {int(df['agreement'].isna().sum())} answered patients missing from the cross-check")
            expect = int(pooled.get((model_dir.name, arm), -1))
            if len(df) != expect:
                raise SystemExit(f"{model_dir.name}/{arm}: {len(df)} scored rows here vs pooled n {expect} in tb_scores.csv")
            arm_rows = score_strata(df, classes, args.n_boot)
            if sum(r["n"] for r in arm_rows if r["stratum_kind"] == "agreement") != expect:
                raise SystemExit(f"{model_dir.name}/{arm}: agreement strata do not add up to {expect}")
            rows += [{"task": TASK, "model": model_dir.name, "arm": arm, "arm_pooled_n": expect, **r} for r in arm_rows]

    out = pd.DataFrame(suppress(rows) if args.suppress_small_cells else rows)
    path = Path(args.crosscheck).with_name("tb_scores_by_crosscheck.csv")
    out.to_csv(path, index=False)
    cols = ["arm", "stratum", "n", "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi"]
    print(out[out["stratum_kind"] == "agreement"][cols].to_string(index=False))
    print(f"Wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
