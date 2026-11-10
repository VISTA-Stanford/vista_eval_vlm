"""Score the published vista_bench v1.7 CT probes on the five findings questions, on the same VAL scans as a
ct-findings-5 run (spec planning/vista_eval_vlm/2026-11-10-ct-agent-probability-climbs.md, Done when: the findings
winner's per-question and mean AUROC on VAL-200 next to Pillar0, COLIPRI and MerlinOnc ep99 on the same scans).
Built from died-2yr-ct's scripts/score_baselines.py (2026-11-10); scans are joined by accession number, not patient.

The probes are vista-eval's linear probes on frozen CT embeddings, scored per scan in the published results
(``ct/per_ct_diagnostic_tasks/<stem>_per_example.parquet`` under RESULTS: task, split, person_id, accession_number,
label, score). Each AUROC is against this task's labels (splits/val.jsonl), with how many published labels differ
reported; a probe with no score for some scans is scored on the ones it covers, with n reported. Mean AUROC is over
the five questions, as the climb's score.

    python scripts/score_baselines.py reproduce          each probe's VAL AUROC per question from its per-example
                                                         file, checked against its published per-task AUROC (3 dp)
    python scripts/score_baselines.py score --ids pool   every probe on the run's VAL pool (pools.json, 200 scans)
    python scripts/score_baselines.py score --ids <file.json>   any {"ids": [...]} file of VAL ids
        --out <file.json>   also write the table as JSON (to the run's results folder on the mount)

    uv run -q --no-project --with pandas --with pyarrow --with scikit-learn python -B scripts/score_baselines.py score --ids pool

Reads the task's private id map (id -> accession number) to join; prints aggregates only, never an id or a row.
Only VAL rows of the published files are read.
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
DATASET = "per_ct_diagnostic_tasks"
#: (answer key = gold field in splits/val.jsonl, vista_bench task), as config.QUESTIONS.
QUESTIONS = (("lesion_liver", "per_ct_lesion_liver"), ("lesion_lungs", "per_ct_lesion_lungs"),
             ("lesion_lymph_nodes", "per_ct_lesion_lymph_nodes"), ("lung_lobe_rul", "per_ct_lung_lobe_rul"),
             ("pneumonitis", "per_ct_pneumonitis"))
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7"))
ID_MAP = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-private/id_map.jsonl")
#: (name shown, result stem): linear probes (the stem without _mlp). Spec, Done when.
BASELINES = (
    ("Pillar0", "pillar0"),
    ("COLIPRI", "colipri"),
    ("MerlinOnc pan-cancer ep99", "merlinonc_top_centered_pancancer_postdx_ep99"),
)
BOOTSTRAP = 2000
SEED = 0


def per_example(stem: str) -> pd.DataFrame:
    path = RESULTS / "ct" / DATASET / f"{stem}_per_example.parquet"
    df = pd.read_parquet(path, columns=["task", "split", "accession_number", "label", "score"],
                         filters=[("split", "==", "val")])
    df = df[df["task"].isin([t for _k, t in QUESTIONS])]
    if df.duplicated(["task", "accession_number"]).any():
        raise SystemExit(f"{stem}: more than one VAL row for a scan and task")
    return df.assign(accession_number=df["accession_number"].astype(str))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of AUROC over scan resamples (resamples missing a class are redrawn)."""
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_reproduce(_args) -> int:
    """The instrument check: recompute each probe's published VAL AUROC per question from its per-example file."""
    bad = 0
    for name, stem in BASELINES:
        published = pd.read_parquet(RESULTS / "ct" / DATASET / f"{stem}_per_task.parquet")
        df_all = per_example(stem)
        for _k, task in QUESTIONS:
            row = published[(published["task"] == task) & (published["split"] == "val")]
            if len(row) != 1:
                print(f"  MISSING  {name} {task}: no single published VAL row")
                bad += 1
                continue
            want, want_n = float(row["auroc"].iloc[0]), int(row["n"].iloc[0])
            df = df_all[df_all["task"] == task]
            got = roc_auc_score(df["label"].to_numpy(), df["score"].to_numpy())
            ok = round(got, 3) == round(want, 3) and len(df) == want_n
            bad += not ok
            print(f"  {'MATCH' if ok else 'MISMATCH'}  {name:<26} {task:<26} published {want:.4f} n={want_n}; "
                  f"recomputed {got:.4f} n={len(df)}")
    return 1 if bad else 0


def wanted_ids(spec: str) -> tuple[str, list[str]]:
    if spec == "pool":
        return "the run's VAL pool", sorted(json.loads((DATA_DIR / "pools.json").read_text())["val"])
    return f"ids in {Path(spec).name}", sorted(json.loads(Path(spec).read_text())["ids"])


def cmd_score(args) -> int:
    what, ids = wanted_ids(args.ids)
    rows = {r["id"]: r for r in map(json.loads, (DATA_DIR / "splits" / "val.jsonl").read_text().splitlines()) if r}
    if set(ids) - set(rows):
        raise SystemExit(f"{len(set(ids) - set(rows))} requested ids are not VAL scans of this task")
    to_acc = {m["id"]: str(m["accession_number"]) for m in map(json.loads, ID_MAP.read_text().splitlines()) if m}
    acc_ids = {to_acc[i]: i for i in ids}
    if len(acc_ids) != len(ids):
        raise SystemExit("two task ids map to one accession number")
    table, out = [], {"what": what, "n": len(ids), "questions": {}}
    print(f"{what}: {len(ids)} scans. VAL; AUROC against this task's labels, 95% bootstrap interval ({BOOTSTRAP})")
    frames = {stem: per_example(stem) for _n, stem in BASELINES}
    for key, task in QUESTIONS:
        scored = [i for i in ids if rows[i][key] is not None]
        yes = sum(rows[i][key] for i in scored)
        out["questions"][key] = {"task": task, "scored": len(scored), "yes": yes}
        print(f"{key} ({task}): {len(scored)} scored scans, {yes} yes")
        for name, stem in BASELINES:
            df = frames[stem]
            df = df[(df["task"] == task) & df["accession_number"].isin({to_acc[i] for i in scored})]
            gold = df["accession_number"].map(lambda a: rows[acc_ids[a]][key]).to_numpy()
            disagree = int((gold != df["label"].to_numpy()).sum())
            sc = df["score"].to_numpy()
            row = {"question": key, "task": task, "baseline": name, "stem": stem, "n": int(len(df)),
                   "of": len(scored), "yes": int(gold.sum()), "label_disagreements": disagree}
            if len(df) and 0 < gold.sum() < len(gold):
                row["auroc"] = float(roc_auc_score(gold, sc))
                row["auroc_lower"], row["auroc_upper"] = bootstrap_ci(gold, sc)
            table.append(row)
            auc = f"{row['auroc']:.3f} [{row['auroc_lower']:.3f}, {row['auroc_upper']:.3f}]" if "auroc" in row else "n/a"
            flag = f"; {disagree} published labels differ" if disagree else ""
            print(f"  {name:<26} n={row['n']:>3}/{len(scored)} ({row['yes']} yes)  AUROC {auc}{flag}")
    means = {}
    for name, _stem in BASELINES:
        vals = [r["auroc"] for r in table if r["baseline"] == name and "auroc" in r]
        means[name] = float(np.mean(vals)) if len(vals) == len(QUESTIONS) else None
        print(f"mean AUROC over the 5, {name:<26} {means[name]:.3f}" if means[name] is not None else
              f"mean AUROC over the 5, {name}: n/a (a question has no AUROC)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({**out, "baselines": table, "mean_auroc": means}, indent=1),
                                  encoding="utf-8")
        print(f"written {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reproduce")
    s = sub.add_parser("score")
    s.add_argument("--ids", default="pool", help="pool | a JSON file with an 'ids' list")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    return {"reproduce": cmd_reproduce, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
