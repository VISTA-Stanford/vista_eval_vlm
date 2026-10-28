"""The fixed sample per step, drawn once from the vista_bench v1.7 tables, seeded, saved to the mount.

Each step's file (``<root>/samples/step<N>.parquet``, plus ``step<N>.json`` with counts and settings) has
one row per case: ``case_id`` (patient for steps 1 and 4, CT accession for 2 and 3), ``person_id``,
``split``, the CT's ``study`` / ``series`` (and ``prior_study`` / ``prior_series`` for two-scan steps),
and one ``label__<task>`` column per question (NaN where that question has no scored label). Every
model and the Claude Code agent read these files, so they all answer about the same cases.

  step 1  death 1 / 2 yr: 200 TEST patients with a pre-treatment CT and a summary, 100 died within
          2 years and 100 did not (the 1-year answer follows; see ``STEP1_NOTE``)
  step 2  17 lesion items: TEST scans chosen so every answer of every item has at least 100 scans
          (rarest answer first, reusing scans already chosen; the 10-25 sizing rule, 522 scans then)
  step 3  progression vs prior: the 400 TEST pairs GPT-6 Astra answered on 10-19 (100 per four-way
          class, so 200 yes / 200 no two-way)
  step 4  board progression: the 334 board cases the rev-5 board CT run answered (scored answer, CT and
          prior CT; all splits, 45 TEST reported separately). One case has no split in the board table: it
          counts in the all-cases score and in no per-split score

    python -m evalset.samples build --steps 1 2 3 4

Prints counts only, never an identifier.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from context import nifti
from evalset import paths
from evalset.cards import CARDS
from evalset.cards.progression import TWO_WAY

SEED = 20261027
PER_ANSWER = 100
ASTRA_400 = ("vistabench/vlm/results_radprog_v1_7/astra_400/radiology_progression_tasks/radiology_progression/"
             "gpt-6-astra/radiology_progression_results_ct_pair.csv")
BOARD_334 = ("vistabench/vlm/results_tb_v1_7_rev5/tb_classification_tasks/progression_assessment_answer/"
             "gemini-3.1-pro-preview/progression_assessment_answer_results_axial_all_image.csv")
STEP1_NOTE = ("Balanced on the 2-year answer (100 / 100). The 1-year answer is then about 60 yes / 140 no: 200 "
              "patients cannot hold 100 per answer on both horizons without dropping the deaths between 1 and 2 years.")


def _bq():
    from google.cloud import bigquery
    return bigquery.Client(project=paths.PROJECT)


def _query(client, sql, params=None):
    from google.cloud import bigquery
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter(k, "STRING", v)
                                                    for k, v in (params or {}).items()])
    return client.query(sql, job_config=cfg).to_dataframe()


def _on_mount(study, series) -> bool:
    p = nifti.ct_path(study, series, mount_root=paths.mount_root())
    return p is not None and p.is_file()


def _pivot(rows: pd.DataFrame, key: str, tasks: list[str]) -> pd.DataFrame:
    """One row per ``key`` with ``label__<task>`` columns (scored 0/1 only; else NaN)."""
    rows = rows.assign(label=rows["label"].astype(float))
    rows.loc[~rows["label"].isin([0, 1]), "label"] = np.nan
    if rows.duplicated([key, "task"]).any():
        raise ValueError(f"more than one row per {key} and task")
    lab = rows.pivot(index=key, columns="task", values="label")
    lab.columns = [f"label__{t}" for t in lab.columns]
    for t in tasks:
        if f"label__{t}" not in lab.columns:
            lab[f"label__{t}"] = np.nan
    return lab


def greedy_cover(labels: pd.DataFrame, k: int, seed: int) -> list:
    """Cases such that every answer of every column has at least ``k`` cases (or all it has).

    The 10-25 sizing rule: answers in order of how few cases carry them (rarest first); for each,
    count the chosen cases that already carry it and add seeded-random others until it reaches k.
    """
    rng = np.random.default_rng(seed)
    order = []
    for col in sorted(labels.columns):
        s = labels[col].dropna()
        for v, grp in s.groupby(s):
            order.append((len(grp), col, v, sorted(grp.index)))
    order.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen: set = set()
    for _, _, _, idx in order:
        have = sum(1 for c in idx if c in chosen)
        rest = [c for c in idx if c not in chosen]
        rng.shuffle(rest)
        chosen.update(rest[: max(0, k - have)])
    return sorted(chosen)


def step1(client) -> pd.DataFrame:
    tasks = [q.task for q in CARDS[1].questions]
    rows = _query(client, f"""select person_id, split, task, label, image_study_uid, image_series_uid, compact_summary
        from `{paths.table('progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr')}`
        where task in unnest(@tasks) and split = 'test'""", {"tasks": tasks})
    lab = _pivot(rows, "person_id", tasks)
    # The CT and summary of the 2-year row: the scan vista_bench pairs with died_any_cause_2_yr, the same
    # one the CT probes and the died-2yr agent use.
    two = rows[rows["task"] == "died_any_cause_2_yr"].set_index("person_id")
    df = lab.join(two[["split", "image_study_uid", "image_series_uid", "compact_summary"]], how="inner")
    df = df.rename(columns={"image_study_uid": "study", "image_series_uid": "series", "compact_summary": "summary"})
    df = df[df[[f"label__{t}" for t in tasks]].notna().all(axis=1) & df["study"].notna() & df["series"].notna()]
    df = df[df["summary"].notna() & (df["summary"].astype(str).str.strip() != "")]
    df = df[[_on_mount(a, b) for a, b in zip(df["study"], df["series"])]]
    from vista_run.run_bq import per_class_sample
    pool = df.reset_index().assign(label=lambda d: d["label__died_any_cause_2_yr"].astype(int))
    out = per_class_sample(pool, "person_id", PER_ANSWER, SEED).drop(columns="label")
    return out.assign(case_id=out["person_id"].astype(str))


def step2(client) -> pd.DataFrame:
    tasks = [q.task for q in CARDS[2].questions]
    rows = _query(client, f"""select person_id, split, task, label, ct_accession_number, image_study_uid, image_series_uid
        from `{paths.table('per_ct_diagnostic_tasks')}`
        where task in unnest(@tasks) and split = 'test' and image_series_uid is not null""", {"tasks": tasks})
    lab = _pivot(rows, "ct_accession_number", tasks)
    scan = rows.drop_duplicates("ct_accession_number").set_index("ct_accession_number")
    df = lab.join(scan[["person_id", "split", "image_study_uid", "image_series_uid"]], how="inner")
    df = df.rename(columns={"image_study_uid": "study", "image_series_uid": "series"})
    keep = greedy_cover(df[[f"label__{t}" for t in tasks]], PER_ANSWER, SEED)
    df = df.loc[keep]
    missing = [k for k, a, b in zip(df.index, df["study"], df["series"]) if not _on_mount(a, b)]
    if missing:
        raise RuntimeError(f"step 2: {len(missing)} chosen scans have no NIfTI on the mount")
    df = df.reset_index()
    return df.assign(case_id=df["ct_accession_number"].astype(str))


def step3(_client) -> pd.DataFrame:
    src = paths.mount_root() / paths.BUCKET / ASTRA_400
    a = pd.read_csv(src, dtype={"ct_accession_number": str, "image_study_uid": str, "image_series_uid": str,
                                "prior_image_study_uid": str, "prior_image_series_uid": str})
    a = a.drop_duplicates("ct_accession_number")
    if len(a) != 400:
        raise RuntimeError(f"step 3: expected the 400 Astra pairs, found {len(a)}")
    a = a.rename(columns={"image_study_uid": "study", "image_series_uid": "series",
                          "prior_image_study_uid": "prior_study", "prior_image_series_uid": "prior_series"})
    a["label__radiology_progression_4way"] = a["label"].astype(int)
    a["label__radiology_progression"] = a["label"].astype(int).map(TWO_WAY)
    for s, r in (("study", "series"), ("prior_study", "prior_series")):
        if not all(_on_mount(x, y) for x, y in zip(a[s], a[r])):
            raise RuntimeError(f"step 3: some {s} NIfTIs are not on the mount")
    cols = ["ct_accession_number", "person_id", "split", "study", "series", "prior_study", "prior_series",
            "prior_gap_days", "label__radiology_progression", "label__radiology_progression_4way"]
    return a[cols].assign(case_id=a["ct_accession_number"])


def step4(_client) -> pd.DataFrame:
    """The 334 cases the rev-5 board CT run answered, with the CT and prior it used (read from that run's
    results). Rebuilding from today's prior-CT link gives 427: the link has grown since 10-14."""
    src = paths.mount_root() / paths.BUCKET / BOARD_334
    a = pd.read_csv(src, dtype=str).drop_duplicates("person_id")
    if len(a) != 334:
        raise RuntimeError(f"step 4: expected the 334 board cases, found {len(a)}")
    a = a.rename(columns={"image_study_uid": "study", "image_series_uid": "series",
                          "prior_image_study_uid": "prior_study", "prior_image_series_uid": "prior_series"})
    a["label__progression_assessment_answer"] = a["label"].astype(int)
    if not a["label__progression_assessment_answer"].isin([0, 1]).all():
        raise RuntimeError("step 4: a board case without a scored answer")
    a["split"] = a["split"].fillna("")
    a["prior_gap_days"] = pd.to_numeric(a["prior_gap_days"], errors="coerce")
    for s, r in (("study", "series"), ("prior_study", "prior_series")):
        if not all(_on_mount(x, y) for x, y in zip(a[s], a[r])):
            raise RuntimeError(f"step 4: some {s} NIfTIs are not on the mount")
    cols = ["person_id", "split", "ct_accession_number", "study", "series", "prior_study", "prior_series",
            "prior_gap_days", "label__progression_assessment_answer"]
    return a[cols].assign(case_id=a["person_id"]).reset_index(drop=True)


BUILDERS = {1: step1, 2: step2, 3: step3, 4: step4}


def counts(step: int, df: pd.DataFrame) -> dict:
    out = {"n": len(df), "by_split": df["split"].value_counts().sort_index().to_dict(), "answers": {}}
    for q in CARDS[step].questions:
        col = f"label__{q.task}"
        out["answers"][q.task] = {str(int(k)): int(v) for k, v in df[col].dropna().astype(int).value_counts().sort_index().items()}
    return out


def save(step: int, df: pd.DataFrame, root: Path | None = None) -> Path:
    root = Path(root or paths.root()) / "samples"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"step{step}.parquet"
    df.to_parquet(out, index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    meta = {"step": step, "built": date.today().isoformat(), "seed": SEED, "per_answer": PER_ANSWER,
            "code_commit": commit, "counts": counts(step, df)}
    if step == 1:
        meta["note"] = STEP1_NOTE
    (root / f"step{step}.json").write_text(json.dumps(meta, indent=1))
    return out


def load(step: int, root: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(root or paths.root()) / "samples" / f"step{step}.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    client = _bq()
    for s in args.steps:
        df = BUILDERS[s](client)
        out = save(s, df)
        print(f"step {s}: {json.dumps(counts(s, df))} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
