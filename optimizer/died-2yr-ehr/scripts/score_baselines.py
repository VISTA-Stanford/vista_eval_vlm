"""Score the published vista_bench v1.7 baselines for died_any_cause_2_yr (and, with --horizons both,
died_any_cause_1_yr) on the same VAL patients as a run.

The baselines are vista-eval's linear probes plus clinical XGBoost (``clinical_pretx_xgb``, Phil 2026-10-31), already scored per patient in the published v1.7 results
(``<modality>/progression_recurrence_survival/<model>_per_example.parquet`` under RESULTS). This script subsets
those per-patient scores to a set of this task's VAL ids and reports AUROC with a 95% bootstrap interval, so the
agent's number and the baselines' compare on identical patients. Radiology-report text covers only patients with
a report before the cutoff: it is scored on its covered subset, with n reported.

    python scripts/score_baselines.py reproduce          AUROC of clinical_pretx on every published VAL patient,
                                                         checked against the published per-task AUROC to 3 decimals
    python scripts/score_baselines.py score --ids pool   every baseline on the run's VAL pool (pools.json, 200)
    python scripts/score_baselines.py score --ids val    every baseline on all VAL patients of the task (811)
    python scripts/score_baselines.py score --ids <file.json>   any {"ids": [...]} file, e.g. a pass's batch
        --out <file.json>   also write the table as JSON (to the run's results folder on the mount)
        --horizons both     score death within 1 year too (the probability run's table); default 2yr only

    python scripts/score_baselines.py survival --ids pool|val [--out <file.json>]
        survival mode's table (spec planning/vista_eval_vlm/2026-11-01-died-ehr-survival.md): every monthly-grid
        survival row (GRID_RESULTS, written by vista-eval scripts/run_survival_monthly_grid_v1_7.py: clinical Cox and
        XGBoost, clinical LLM Cox, the patient-summary probe, the re-fitted radiology-text and MOTOR probes) and the
        average curve (TRAIN Kaplan-Meier), each scored on the task's patients by the agent's own scorer
        (survival_metrics.py): integrated Brier (lower is better) with a 95% bootstrap interval, C-index, Brier and
        time-dependent AUC at 12/24/36/60 months. Patients with no os_survival time are left out (pool 198, VAL 804).

Needs pandas, numpy and scikit-learn (the vista_eval_vlm radprog environment has them). Reads the CT task's
private id map (id -> person_id) to join; prints aggregates only, never an id or a row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = Path(os.environ.get("VISTA_RESULTS", "/mnt/su-vista-uscentral1/vistabench/results/v1_7"))
DATASET = "progression_recurrence_survival"
TASK = "died_any_cause_2_yr"
#: (words, vista_bench task, gold field in splits/*.jsonl). The yes/no-era table scored 2 years only.
HORIZONS = {"2yr": (("2 years", TASK, "label"),),
            "both": (("1 year", "died_any_cause_1_yr", "label_1yr"), ("2 years", TASK, "label"))}
#: The task's splits and pools are the CT task's (scripts/build_data.py copies them), so either folder serves.
DATA_DIR = Path(os.environ.get("TASK_SPLITS_DIR", "/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7"))
CT_DATA_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7")
ID_MAP = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7-private/id_map.jsonl")
#: (name shown, modality folder, result stem). Linear probe = the stem without _knn / _mlp. Spec, Decided list.
BASELINES = (
    ("clinical_pretx", "clinical", "clinical_pretx"),
    ("clinical_llm_pretx", "clinical", "clinical_llm_pretx"),
    ("clinical_pretx_xgb (XGBoost)", "clinical", "clinical_pretx_xgb"),
    ("patient summary, Gemini", "text_llm_summary", "gemini_emb_001"),
    ("run-vista-02 (MOTOR)", "ehr_code", "run-vista-02_512_medspace_v1_tokfix"),
    ("motor-arm-A-codes-notes", "ehr_code", "motor-arm-A-codes-notes_512_medspace_v1_tokfix"),
    ("radiology text, Gemini", "text_radiology", "gemini_emb_001"),
    ("radiology text, GatorTron", "text_radiology", "gatortron_base"),
    ("radiology text, Qwen3", "text_radiology", "qwen3emb8b"),
    ("radiology findings, Gemini", "text_radiology_findings", "gemini_emb_001"),
    # the probe tool's other two probes (spec 2026-11-09-died-2yr-ehr-probe-tool.md: each probe alone is a bar)
    ("pathology text, Gemini", "text_pathology", "gemini_emb_001"),
    ("CT, MerlinOnc pan-cancer ep99", "ct", "merlinonc_top_centered_pancancer_postdx_ep99"),
)
#: The probe tool's headline bar (spec planning/vista_eval_vlm/2026-11-09-died-2yr-ehr-probe-tool.md): a logistic
#: regression on the six probes' scores, fit on TRAIN out-of-fold scores, written by scripts/build_probe_tool.py keyed
#: by this task's ids. Scored as one more row when the file exists.
COMBINATION = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7-private/probes/combination_val.parquet")
COMBINATION_NAME = "six-probe combination (probe tool bar)"
BOOTSTRAP = 2000
SEED = 0
#: Survival mode: the monthly-grid rows, one folder each (vista-eval configs/runs/v1_7/survival_monthly_grid/).
GRID_RESULTS = Path(os.environ.get(
    "VISTA_SURVIVAL_GRID", "/mnt/su-vista-uscentral1/vistabench/results/v1_7/survival_monthly_grid/progression_recurrence_survival"))
#: (name shown, folder) in table order.
SURVIVAL_ROWS = (
    ("clinical, Cox", "clinical_pretx_cox"),
    ("clinical, XGBoost survival", "clinical_pretx_xgb"),
    ("clinical LLM, Cox", "clinical_llm_pretx_cox"),
    ("patient summary, Gemini, Cox", "summary_gemini_cox"),
    ("run-vista-02 (MOTOR), Cox", "motor_run_vista_02_cox"),
    ("motor-arm-A-codes-notes, Cox", "motor_arm_A_cox"),
    ("radiology text, Gemini, Cox", "radiology_gemini_cox"),
    ("radiology text, GatorTron, Cox", "radiology_gatortron_cox"),
    ("radiology text, Qwen3, Cox", "radiology_qwen3_cox"),
)
SURVIVAL_BOOTSTRAP = 200


def data_dir() -> Path:
    return DATA_DIR if (DATA_DIR / "pools.json").is_file() else CT_DATA_DIR


def per_example(modality: str, stem: str, split: str = "val", task: str = TASK) -> pd.DataFrame:
    path = RESULTS / modality / DATASET / f"{stem}_per_example.parquet"
    df = pd.read_parquet(path, columns=["task", "split", "person_id", "label", "score"])
    df = df[(df["task"] == task) & (df["split"] == split)]
    if df["person_id"].duplicated().any():
        raise SystemExit(f"{stem}: more than one row per patient for {task}/{split}")
    return df.assign(person_id=df["person_id"].astype(str))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of AUROC over patient resamples (resamples missing a class are redrawn)."""
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_reproduce(_args) -> int:
    published = pd.read_parquet(RESULTS / "clinical" / DATASET / "clinical_pretx_per_task.parquet")
    row = published[(published["task"] == TASK) & (published["split"] == "val")]
    if len(row) != 1:
        raise SystemExit("no single published VAL row for clinical_pretx")
    want, want_n = float(row["auroc"].iloc[0]), int(row["n"].iloc[0])
    df = per_example("clinical", "clinical_pretx")
    got = roc_auc_score(df["label"].to_numpy(), df["score"].to_numpy())
    ok = round(got, 3) == round(want, 3) and len(df) == want_n
    print(f"clinical_pretx {TASK} VAL: published AUROC {want:.4f} on n={want_n}; recomputed {got:.4f} on n={len(df)} "
          f"from the per-example file -> {'MATCH' if ok else 'MISMATCH'} to 3 decimals")
    return 0 if ok else 1


def wanted_ids(spec: str) -> tuple[str, list[str]]:
    d = data_dir()
    if spec == "pool":
        return "the run's VAL pool", sorted(json.loads((d / "pools.json").read_text())["val"])
    if spec == "val":
        return "all VAL patients", [json.loads(l)["id"] for l in (d / "splits" / "val.jsonl").read_text().splitlines() if l]
    return f"ids in {Path(spec).name}", sorted(json.loads(Path(spec).read_text())["ids"])


def cmd_score(args) -> int:
    what, ids = wanted_ids(args.ids)
    d = data_dir()
    rows = {json.loads(l)["id"]: json.loads(l) for l in (d / "splits" / "val.jsonl").read_text().splitlines() if l}
    if set(ids) - set(rows):
        raise SystemExit(f"{len(set(ids) - set(rows))} requested ids are not VAL patients of this task")
    to_pid = {}
    with ID_MAP.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            to_pid[m["id"]] = str(m["person_id"])
    pid_ids = {to_pid[i]: i for i in ids}
    table, out = [], {"what": what, "n": len(ids), "horizons": {}}
    for words, task, field in HORIZONS[args.horizons]:
        died = sum(rows[i][field] for i in ids)
        out["horizons"][words] = {"task": task, "died": died}
        print(f"{what}: {len(ids)} patients, {died} died within {words} ({died / len(ids):.1%}). {task}, VAL; "
              f"AUROC with a 95% bootstrap interval ({BOOTSTRAP} resamples)")
        for name, modality, stem in BASELINES:
            df = per_example(modality, stem, task=task)
            df = df[df["person_id"].isin(pid_ids)]
            task_gold = df["person_id"].map(lambda p: rows[pid_ids[p]][field])
            disagree = int((task_gold.to_numpy() != df["label"].to_numpy()).sum())
            y, sc = task_gold.to_numpy(), df["score"].to_numpy()
            row = {"horizon": words, "task": task, "baseline": name, "modality": modality, "stem": stem,
                   "n": int(len(df)), "of": len(ids), "died": int(y.sum()), "label_disagreements": disagree}
            if len(df) and 0 < y.sum() < len(y):
                row["auroc"] = float(roc_auc_score(y, sc))
                row["auroc_lower"], row["auroc_upper"] = bootstrap_ci(y, sc)
            table.append(row)
            auc = f"{row['auroc']:.3f} [{row['auroc_lower']:.3f}, {row['auroc_upper']:.3f}]" if "auroc" in row else "n/a"
            flag = f"; {disagree} labels differ from the task's" if disagree else ""
            print(f"  {name:<30} n={row['n']:>4}/{len(ids)} ({row['died']} died)  AUROC {auc}{flag}")
        if COMBINATION.is_file():
            c = pd.read_parquet(COMBINATION)
            c = c[(c["horizon"] == words) & c["id"].isin(ids)]
            y, sc = np.array([rows[i][field] for i in c["id"]]), c["score"].to_numpy()
            row = {"horizon": words, "task": task, "baseline": COMBINATION_NAME, "modality": "probes",
                   "stem": "combination", "n": int(len(c)), "of": len(ids), "died": int(y.sum()),
                   "label_disagreements": 0}
            if len(c) and 0 < y.sum() < len(y):
                row["auroc"] = float(roc_auc_score(y, sc))
                row["auroc_lower"], row["auroc_upper"] = bootstrap_ci(y, sc)
            table.append(row)
            auc = f"{row['auroc']:.3f} [{row['auroc_lower']:.3f}, {row['auroc_upper']:.3f}]" if "auroc" in row else "n/a"
            print(f"  {COMBINATION_NAME:<30} n={row['n']:>4}/{len(ids)} ({row['died']} died)  AUROC {auc}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if args.horizons == "2yr":  # the yes/no-era file layout, unchanged
            h = out["horizons"]["2 years"]
            body = {"what": what, "n": len(ids), "died": h["died"], "task": TASK, "baselines": table}
        else:
            body = {**out, "baselines": table}
        Path(args.out).write_text(json.dumps(body, indent=1), encoding="utf-8")
        print(f"written {args.out}")
    return 0


def cmd_survival(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import survival_metrics as sm

    what, ids = wanted_ids(args.ids)
    d = data_dir()
    key = {}
    for line in (d / "survival.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            key[r["id"]] = {"id": r["id"], "time_days": float(r["time_days"]), "event": int(r["event"])}
    train_ids = [json.loads(l)["id"] for l in (d / "splits" / "train.jsonl").read_text().splitlines() if l]
    train = [key[i] for i in train_ids if i in key]
    kept = [i for i in ids if i in key]
    to_id = {}
    with ID_MAP.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            to_id[str(m["person_id"])] = m["id"]
    xs, ys = sm.kaplan_meier([r["time_days"] / sm.DAYS_PER_MONTH for r in train], [r["event"] for r in train])
    avg = sm.curve_at(list(zip([0.0, *xs], [1.0, *ys])), sm.GRID_MONTHS)
    rows_all = [key[i] for i in kept]
    print(f"{what}: {len(ids)} patients, {len(kept)} with an os_survival time ({sum(r['event'] for r in rows_all)} "
          f"deaths); scoring months {sm.scoring_months(train, rows_all)[0]}-{sm.scoring_months(train, rows_all)[-1]}; "
          f"integrated Brier (lower is better) with a 95% bootstrap interval ({SURVIVAL_BOOTSTRAP} resamples)")
    rng = np.random.default_rng(SEED)

    def score(name, folder, curves):
        rows = [key[i] for i in kept if i in curves]
        m = sm.survival_metrics(rows, curves, train)
        boots = []
        for _ in range(SURVIVAL_BOOTSTRAP):
            pick = rng.integers(0, len(rows), len(rows))
            rs = [dict(rows[k], id=f"{rows[k]['id']}#{n}") for n, k in enumerate(pick)]
            cs = {r["id"]: curves[r["id"].split("#")[0]] for r in rs}
            if any(r["event"] for r in rs):
                b = sm.survival_metrics(rs, cs, train).get("integrated_brier")
                if b is not None and b == b:
                    boots.append(b)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
        row = {"baseline": name, "folder": folder, "n": len(rows), "of": len(kept), **m,
               "integrated_brier_lower": float(lo), "integrated_brier_upper": float(hi)}
        print(f"  {name:<32} n={len(rows):>4}/{len(kept)}  integrated Brier {m.get('integrated_brier', float('nan')):.4f} "
              f"[{lo:.4f}, {hi:.4f}]  C-index {m.get('c_index', float('nan')):.4f}  "
              f"Brier 12/24/36/60m " + "/".join(f"{m.get(f'brier_{k}m', float('nan')):.3f}" for k in sm.REPORT_MONTHS)
              + (f"  ({row['gold_disagreements']} gold disagreements)" if row.get("gold_disagreements") else ""),
              flush=True)
        return row

    table = [score("average curve (TRAIN Kaplan-Meier)", None, {i: avg for i in kept})]
    for name, folder in SURVIVAL_ROWS:
        files = sorted((GRID_RESULTS / folder).glob("*_survival_curves.parquet"))
        if len(files) != 1:
            print(f"  {name:<32} no curves file in {GRID_RESULTS / folder}")
            table.append({"baseline": name, "folder": folder, "missing": True})
            continue
        c = pd.read_parquet(files[0])
        c = c[(c["task"] == "os_survival") & (c["split"] == "val")]
        if c["person_id"].duplicated().any():
            raise SystemExit(f"{folder}: more than one VAL row for a patient")
        grid = list(c["horizons_months"].iloc[0])
        if grid != list(sm.GRID_MONTHS):
            raise SystemExit(f"{folder}: curves on months {grid[0]}-{grid[-1]}, not the task's grid")
        curves, disagree = {}, 0
        for pid, t, e, s in zip(c["person_id"].astype(str), c["time_to_event_days"], c["event_observed"], c["survival"]):
            i = to_id.get(pid)
            if i in key and i in set(kept):
                curves[i] = [float(x) for x in s]
                disagree += int(abs(float(t) - key[i]["time_days"]) > 1e-6 or int(e) != key[i]["event"])
        row = score(name, folder, curves)
        row["gold_disagreements"] = disagree
        if disagree:
            print(f"    {disagree} patients' time or event differs from the task's key")
        table.append(row)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"what": what, "n": len(kept), "answer": "survival",
                                               "baselines": table}, indent=1, default=float), encoding="utf-8")
        print(f"written {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reproduce")
    s = sub.add_parser("score")
    s.add_argument("--ids", default="pool", help="pool | val | a JSON file with an 'ids' list")
    s.add_argument("--out", default="")
    s.add_argument("--horizons", choices=("2yr", "both"), default="2yr")
    v = sub.add_parser("survival")
    v.add_argument("--ids", default="pool", help="pool | val | a JSON file with an 'ids' list")
    v.add_argument("--out", default="")
    args = ap.parse_args()
    return {"reproduce": cmd_reproduce, "score": cmd_score, "survival": cmd_survival}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
