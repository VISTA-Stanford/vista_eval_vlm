"""Score the published vista_bench v1.7 baselines for died_any_cause_2_yr on the same VAL patients as a run.

The baselines are vista-eval's linear probes, already scored per patient in the published v1.7 results
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
#: The task's splits and pools are the CT task's (scripts/build_data.py copies them), so either folder serves.
DATA_DIR = Path(os.environ.get("TASK_SPLITS_DIR", "/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7"))
CT_DATA_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7")
ID_MAP = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7-private/id_map.jsonl")
#: (name shown, modality folder, result stem). Linear probe = the stem without _knn / _mlp. Spec, Decided list.
BASELINES = (
    ("clinical_pretx", "clinical", "clinical_pretx"),
    ("clinical_llm_pretx", "clinical", "clinical_llm_pretx"),
    ("run-vista-02 (MOTOR)", "ehr_code", "run-vista-02_512_medspace_v1_tokfix"),
    ("motor-arm-A-codes-notes", "ehr_code", "motor-arm-A-codes-notes_512_medspace_v1_tokfix"),
    ("radiology text, Gemini", "text_radiology", "gemini_emb_001"),
    ("radiology text, GatorTron", "text_radiology", "gatortron_base"),
    ("radiology text, Qwen3", "text_radiology", "qwen3emb8b"),
    ("radiology findings, Gemini", "text_radiology_findings", "gemini_emb_001"),
)
BOOTSTRAP = 2000
SEED = 0


def data_dir() -> Path:
    return DATA_DIR if (DATA_DIR / "pools.json").is_file() else CT_DATA_DIR


def per_example(modality: str, stem: str, split: str = "val") -> pd.DataFrame:
    path = RESULTS / modality / DATASET / f"{stem}_per_example.parquet"
    df = pd.read_parquet(path, columns=["task", "split", "person_id", "label", "score"])
    df = df[(df["task"] == TASK) & (df["split"] == split)]
    if df["person_id"].duplicated().any():
        raise SystemExit(f"{stem}: more than one row per patient for {TASK}/{split}")
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
    gold = {json.loads(l)["id"]: json.loads(l)["label"] for l in (d / "splits" / "val.jsonl").read_text().splitlines() if l}
    if set(ids) - set(gold):
        raise SystemExit(f"{len(set(ids) - set(gold))} requested ids are not VAL patients of this task")
    to_pid = {}
    with ID_MAP.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            to_pid[m["id"]] = str(m["person_id"])
    pid_ids = {to_pid[i]: i for i in ids}
    died = sum(gold[i] for i in ids)
    print(f"{what}: {len(ids)} patients, {died} died ({died / len(ids):.1%}). {TASK}, VAL, linear probe; "
          f"AUROC with a 95% bootstrap interval ({BOOTSTRAP} resamples)")
    table = []
    for name, modality, stem in BASELINES:
        df = per_example(modality, stem)
        df = df[df["person_id"].isin(pid_ids)]
        task_gold = df["person_id"].map(lambda p: gold[pid_ids[p]])
        disagree = int((task_gold.to_numpy() != df["label"].to_numpy()).sum())
        y, s = task_gold.to_numpy(), df["score"].to_numpy()
        row = {"baseline": name, "modality": modality, "stem": stem, "n": int(len(df)), "of": len(ids),
               "died": int(y.sum()), "label_disagreements": disagree}
        if len(df) and 0 < y.sum() < len(y):
            row["auroc"] = float(roc_auc_score(y, s))
            row["auroc_lower"], row["auroc_upper"] = bootstrap_ci(y, s)
        table.append(row)
        auc = f"{row['auroc']:.3f} [{row['auroc_lower']:.3f}, {row['auroc_upper']:.3f}]" if "auroc" in row else "n/a"
        flag = f"; {disagree} labels differ from the task's" if disagree else ""
        print(f"  {name:<28} n={row['n']:>4}/{len(ids)} ({row['died']} died)  AUROC {auc}{flag}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"what": what, "n": len(ids), "died": died, "task": TASK,
                                              "baselines": table}, indent=1), encoding="utf-8")
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
