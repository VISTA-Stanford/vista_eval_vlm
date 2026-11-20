"""Score the published vista_bench v1.7 baselines for died_any_cause_1_yr and died_any_cause_2_yr on the same VAL
patients as a died-2yr-ct run (spec planning/vista_eval_vlm/2026-11-10-ct-agent-probability-climbs.md: the death
winner's VAL-200 AUROC next to MerlinOnc and clinical XGBoost on the same patients). Copied from died-2yr-ehr's
scripts/score_baselines.py (2026-11-10), with the CT probes in place of the EHR ones.

The baselines are vista-eval's CT linear probes and clinical XGBoost, already scored per patient in the published
v1.7 results (``<modality>/progression_recurrence_survival/<stem>_per_example.parquet`` under RESULTS). This script
subsets those per-patient scores to a set of this task's VAL ids and reports AUROC per horizon with a 95% bootstrap
interval, so the agent's number and the baselines' compare on identical patients. A probe that has no score for some
of the patients is scored on the ones it covers, with n reported.

    python scripts/score_baselines.py reproduce          AUROC of each baseline on every published VAL patient,
                                                         checked against its published per-task AUROC to 3 decimals
    python scripts/score_baselines.py score --ids pool   every baseline on the run's VAL pool (pools.json, 200)
    python scripts/score_baselines.py score --ids val    every baseline on all VAL patients of the task (811)
    python scripts/score_baselines.py score --ids <file.json>   any {"ids": [...]} file, e.g. a pass's batch
        --out <file.json>   also write the table as JSON (to the run's results folder on the mount)

Needs pandas, numpy and scikit-learn. Reads the task's private id map (id -> person_id) to join; prints aggregates
only, never an id or a row.
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
#: (words, vista_bench task, gold field in splits/*.jsonl), as config.HORIZONS.
HORIZONS = (("1 year", "died_any_cause_1_yr", "label_1yr"), ("2 years", "died_any_cause_2_yr", "label"))
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7"))
ID_MAP = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7-private/id_map.jsonl")
#: (name shown, modality folder, result stem): linear probes (the stem without _knn / _mlp). Spec, Done when.
BASELINES = (
    ("MerlinOnc 10-crop v2.2", "ct", "merlinonc_10crops_v2.2"),
    ("MerlinOnc top-centered v2.2", "ct", "merlinonc_top_centered_v2.2"),
    ("MerlinOnc pan-cancer ep99", "ct", "merlinonc_top_centered_pancancer_postdx_ep99"),
    ("clinical_pretx_xgb (XGBoost)", "clinical", "clinical_pretx_xgb"),
)
BOOTSTRAP = 2000
SEED = 0


def per_example(modality: str, stem: str, task: str, split: str = "val") -> pd.DataFrame:
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
    """The instrument check: recompute each baseline's published VAL AUROC from its per-example file."""
    bad = 0
    for name, modality, stem in BASELINES:
        published = pd.read_parquet(RESULTS / modality / DATASET / f"{stem}_per_task.parquet")
        for _w, task, _f in HORIZONS:
            row = published[(published["task"] == task) & (published["split"] == "val")]
            if len(row) != 1:
                print(f"{name} {task}: no single published VAL row")
                bad += 1
                continue
            want, want_n = float(row["auroc"].iloc[0]), int(row["n"].iloc[0])
            df = per_example(modality, stem, task)
            got = roc_auc_score(df["label"].to_numpy(), df["score"].to_numpy())
            ok = round(got, 3) == round(want, 3) and len(df) == want_n
            bad += not ok
            print(f"  {'MATCH' if ok else 'MISMATCH'}  {name:<30} {task}: published {want:.4f} n={want_n}; "
                  f"recomputed {got:.4f} n={len(df)}")
    return 1 if bad else 0


def wanted_ids(spec: str) -> tuple[str, list[str]]:
    if spec == "pool":
        return "the run's VAL pool", sorted(json.loads((DATA_DIR / "pools.json").read_text())["val"])
    if spec == "val":
        return "all VAL patients", [json.loads(l)["id"] for l in (DATA_DIR / "splits" / "val.jsonl").read_text().splitlines() if l]
    return f"ids in {Path(spec).name}", sorted(json.loads(Path(spec).read_text())["ids"])


def cmd_score(args) -> int:
    what, ids = wanted_ids(args.ids)
    rows = {json.loads(l)["id"]: json.loads(l) for l in (DATA_DIR / "splits" / "val.jsonl").read_text().splitlines() if l}
    if set(ids) - set(rows):
        raise SystemExit(f"{len(set(ids) - set(rows))} requested ids are not VAL patients of this task")
    to_pid = {}
    with ID_MAP.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            to_pid[m["id"]] = str(m["person_id"])
    pid_ids = {to_pid[i]: i for i in ids}
    if len(pid_ids) != len(ids):
        raise SystemExit("two task ids map to one patient")
    table, out = [], {"what": what, "n": len(ids), "horizons": {}}
    for words, task, field in HORIZONS:
        died = sum(rows[i][field] for i in ids)
        out["horizons"][words] = {"task": task, "died": died}
        print(f"{what}: {len(ids)} patients, {died} died within {words} ({died / len(ids):.1%}). {task}, VAL; "
              f"AUROC with a 95% bootstrap interval ({BOOTSTRAP} resamples)")
        for name, modality, stem in BASELINES:
            df = per_example(modality, stem, task)
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
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({**out, "baselines": table}, indent=1), encoding="utf-8")
        print(f"written {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reproduce")
    s = sub.add_parser("score")
    s.add_argument("--ids", default="pool", help="pool | val | a JSON file with an 'ids' list")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    return {"reproduce": cmd_reproduce, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
