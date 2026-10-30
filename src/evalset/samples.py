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
  step 5  positive control (IV contrast, female sex): scans of the step-2 sample, balanced on contrast, answer
          keys from each series' DICOM header on the mount (no BigQuery)

    python -m evalset.samples build --steps 1 2 3 4

Prints counts only, never an identifier.
"""

from __future__ import annotations

import argparse
import json
import re
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


def _step1_pool(client) -> pd.DataFrame:
    """Every TEST patient eligible for step 1 (both labels scored, CT on the mount, a summary)."""
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
    return df.reset_index()


def step1(client) -> pd.DataFrame:
    from vista_run.run_bq import per_class_sample
    pool = _step1_pool(client).assign(label=lambda d: d["label__died_any_cause_2_yr"].astype(int))
    out = per_class_sample(pool, "person_id", PER_ANSWER, SEED).drop(columns="label")
    return out.assign(case_id=out["person_id"].astype(str))


def _step2_table(client) -> pd.DataFrame:
    """Every TEST scan with at least one of the 17 items labelled, indexed by accession (mount not yet checked)."""
    tasks = [q.task for q in CARDS[2].questions]
    rows = _query(client, f"""select person_id, split, task, label, ct_accession_number, image_study_uid, image_series_uid
        from `{paths.table('per_ct_diagnostic_tasks')}`
        where task in unnest(@tasks) and split = 'test' and image_series_uid is not null""", {"tasks": tasks})
    lab = _pivot(rows, "ct_accession_number", tasks)
    per_acc = rows.groupby("ct_accession_number")[["image_study_uid", "image_series_uid"]].nunique()
    if (per_acc > 1).any(axis=None):
        raise RuntimeError("step 2: an accession maps to more than one CT series; the scan would be ambiguous")
    scan = rows.drop_duplicates("ct_accession_number").set_index("ct_accession_number")
    df = lab.join(scan[["person_id", "split", "image_study_uid", "image_series_uid"]], how="inner")
    df = df.rename(columns={"image_study_uid": "study", "image_series_uid": "series"})
    return df[df[[f"label__{t}" for t in tasks]].notna().any(axis=1)]


#: Step 2 for v1 (Phil 10-29): a simple random 1,000 TEST scans, drawn once and kept fixed for every model.
STEP2_N = 1000
STEP2_SEED = 20261029


def step2(client) -> pd.DataFrame:
    """A simple random STEP2_N of the labelled TEST scans (seed STEP2_SEED), every label column kept."""
    df = _step2_table(client).sort_index()
    df = df.sample(n=STEP2_N, random_state=STEP2_SEED)
    missing = [k for k, a, b in zip(df.index, df["study"], df["series"]) if not _on_mount(a, b)]
    if missing:
        raise RuntimeError(f"step 2: {len(missing)} chosen scans have no NIfTI on the mount")
    df = df.sort_index().reset_index()
    return df.assign(case_id=df["ct_accession_number"].astype(str))


def step2_greedy_526(client) -> pd.DataFrame:
    """The 10-28 step-2 sample (every answer of all 17 items at >=100), superseded 10-29; kept for reference."""
    tasks = [q.task for q in CARDS[2].questions]
    df = _step2_table(client)
    df = df.loc[greedy_cover(df[[f"label__{t}" for t in tasks]], PER_ANSWER, SEED)].reset_index()
    return df.assign(case_id=df["ct_accession_number"].astype(str))


def step3(client) -> pd.DataFrame:
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
    # Priors from the task table, the scan each label was read against (as the whole-TEST list does). The Astra
    # file's priors came from the shared prior link on 10-19; 1 of the 400 differs (10-29 check).
    pri = _query(client, f"""select ct_accession_number, prior_image_study_uid, prior_image_series_uid, prior_gap_days
        from `{paths.table('radiology_progression_tasks')}` where task = 'radiology_progression'""")
    pri = pri.drop_duplicates("ct_accession_number").set_index(pri.drop_duplicates("ct_accession_number")["ct_accession_number"].astype(str))
    k = a["ct_accession_number"].astype(str)
    if not k.isin(pri.index).all():
        raise RuntimeError("step 3: some Astra pairs are no longer in the task table")
    a["prior_study"] = k.map(pri["prior_image_study_uid"]).values
    a["prior_series"] = k.map(pri["prior_image_series_uid"]).values
    a["prior_gap_days"] = k.map(pri["prior_gap_days"]).values
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
    a["label__progression_assessment_answer"] = pd.to_numeric(a["label"]).astype(int)
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


def step1_test(client) -> pd.DataFrame:
    """Whole TEST split for step 1: every eligible patient (860 on 10-28)."""
    df = _step1_pool(client)
    return df.assign(case_id=df["person_id"].astype(str))


def step2_test(client) -> pd.DataFrame:
    """Whole TEST split for step 2: every labelled scan whose NIfTI is on the mount (missing ones counted)."""
    df = _step2_table(client)
    on = [_on_mount(a, b) for a, b in zip(df["study"], df["series"])]
    print(f"    step 2 test: {len(df)} labelled scans, {len(df) - sum(on)} without a NIfTI on the mount (dropped)")
    df = df[on].reset_index()
    return df.assign(case_id=df["ct_accession_number"].astype(str))


def step3_test(client) -> pd.DataFrame:
    """Whole TEST split for step 3: every scored scan with its prior (the task table's own prior columns), both
    NIfTIs on the mount (missing ones counted)."""
    rows = _query(client, f"""select person_id, split, label, ct_accession_number, image_study_uid, image_series_uid,
          prior_image_study_uid, prior_image_series_uid, prior_gap_days
        from `{paths.table('radiology_progression_tasks')}`
        where task = 'radiology_progression' and split = 'test' and label in (0, 1, 2, 3)
          and image_series_uid is not null and prior_image_series_uid is not null""")
    if rows["ct_accession_number"].duplicated().any():
        raise RuntimeError("step 3 test: more than one row per accession")
    a = rows.rename(columns={"image_study_uid": "study", "image_series_uid": "series",
                             "prior_image_study_uid": "prior_study", "prior_image_series_uid": "prior_series"})
    on = [_on_mount(x, y) and _on_mount(p, q) for x, y, p, q in zip(a["study"], a["series"], a["prior_study"], a["prior_series"])]
    print(f"    step 3 test: {len(a)} pairs, {len(a) - sum(on)} with a NIfTI missing on the mount (dropped)")
    a = a[on].copy()
    a["label__radiology_progression_4way"] = a["label"].astype(int)
    a["label__radiology_progression"] = a["label"].astype(int).map(TWO_WAY)
    a["ct_accession_number"] = a["ct_accession_number"].astype(str)
    return a.drop(columns="label").assign(case_id=a["ct_accession_number"]).reset_index(drop=True)


def step4_test(client) -> pd.DataFrame:
    """Whole TEST split for step 4: the TEST cases among the 334 (the sample already holds every board case)."""
    df = step4(client)
    return df[df["split"] == "test"].reset_index(drop=True)


#: Step 5 control (spec 2026-10-30-medgemma-fair-eval): one-slice DICOM header per series, ``{study}__{series}.dcm``.
DICOM_HEADERS = "chaudhari_lab/ct_data/ct_scans/vista/dicom_metadata"
STEP5_SEED = 20261030
STEP5_RULE = ("step-2 sample scans whose study description and contrast-agent field agree; every scan without contrast, "
              "the same number with (seeded); female from PatientSex")


def contrast_from_description(desc) -> float:
    """1 / 0 from a study description that says with / without IV contrast; NaN when it says both or neither."""
    d = re.sub(r"\s+", " ", str(desc or "").upper())
    if re.search(r"\bW(ITH)? AND WO\b|WITH AND WITHOUT", d):
        return np.nan  # two-phase study: the stored series could be either
    if re.search(r"\bWO\b|WITHOUT", d):
        return 0.0
    if re.search(r"\bW\b|\bWITH\b", d):
        return 1.0
    return np.nan


def contrast_from_agent(agent) -> float:
    """1 from a named iodinated agent (Isovue, Omnipaque and their misspellings) or 'APPLIED' / 'CE'; 0 from an
    empty field, 'NONE', 'UNDEFINED' or water alone ('H20', 'H2O'); NaN otherwise."""
    a = str(agent or "").upper().strip()
    if re.search(r"ISO|IOS|OMNI", a) or a in ("APPLIED", "CE"):
        return 1.0
    if a in ("", "NONE", "UNDEFINED", "H20", "H2O"):
        return 0.0
    return np.nan


def _header_fields(study, series) -> dict:
    import pydicom
    p = paths.mount_root() / paths.BUCKET / DICOM_HEADERS / f"{study}__{series}.dcm"
    h = pydicom.dcmread(p, stop_before_pixels=True)
    return {"desc": str(h.get("StudyDescription", "")), "agent": str(h.get("ContrastBolusAgent", "")),
            "sex": str(h.get("PatientSex", ""))}


def step5_labels(base: pd.DataFrame, read=_header_fields, workers: int = 32) -> pd.DataFrame:
    """``base`` with ``label__meta_iv_contrast`` (NaN unless description and agent agree) and ``label__meta_female``."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        h = pd.DataFrame(list(ex.map(read, base["study"], base["series"])), index=base.index)
    d, a = h["desc"].map(contrast_from_description), h["agent"].map(contrast_from_agent)
    sex = h["sex"].str.upper().str.strip().map({"F": 1.0, "M": 0.0})
    return base.assign(label__meta_iv_contrast=d.where(d == a), label__meta_female=sex)


def step5(_client, read=_header_fields) -> pd.DataFrame:
    """Every step-2 sample scan without contrast and as many with (seeded), both labels from the DICOM header."""
    cols = ["case_id", "person_id", "split", "ct_accession_number", "study", "series"]
    df = step5_labels(load(2)[cols], read)
    neg = df[df["label__meta_iv_contrast"] == 0]
    pos = df[df["label__meta_iv_contrast"] == 1].sort_values("case_id")
    if len(pos) < len(neg):
        raise RuntimeError(f"step 5: {len(pos)} scans with contrast, fewer than the {len(neg)} without")
    pos = pos.sample(n=len(neg), random_state=STEP5_SEED)
    return pd.concat([neg, pos]).sort_values("case_id").reset_index(drop=True)


BUILDERS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5}
#: ``which="test"``: the whole TEST split, the second stage after the samples (spec, Scale).
TEST_BUILDERS = {1: step1_test, 2: step2_test, 3: step3_test, 4: step4_test}


def _name(step: int, which: str) -> str:
    if which not in ("sample", "test"):
        raise ValueError(f"which must be 'sample' or 'test', not {which!r}")
    return f"step{step}" if which == "sample" else f"step{step}_test"


def counts(step: int, df: pd.DataFrame) -> dict:
    out = {"n": len(df), "by_split": df["split"].value_counts().sort_index().to_dict(), "answers": {}}
    for q in CARDS[step].questions:
        col = f"label__{q.task}"
        out["answers"][q.task] = {str(int(k)): int(v) for k, v in df[col].dropna().astype(int).value_counts().sort_index().items()}
    return out


def case_hash(df: pd.DataFrame) -> str:
    """sha256 of the sorted case ids: names the case set, independent of file encoding."""
    import hashlib
    return hashlib.sha256("\n".join(sorted(df["case_id"].astype(str))).encode()).hexdigest()


def save(step: int, df: pd.DataFrame, root: Path | None = None, which: str = "sample", force: bool = False) -> Path:
    """Write the step's file. A file that already exists with a different case set is never replaced unless
    ``force``: samples are fixed once drawn (Phil 10-29, "keep those 1k fixed")."""
    root = Path(root or paths.root()) / "samples"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{_name(step, which)}.parquet"
    if out.exists() and not force:
        old = pd.read_parquet(out)
        if case_hash(old) != case_hash(df):
            raise RuntimeError(f"{out} exists with a different case set; samples are fixed once drawn (force to replace)")
    df.to_parquet(out, index=False)
    commit = paths.code_commit()
    import hashlib
    meta = {"step": step, "which": which, "built": date.today().isoformat(), "code_commit": commit,
            "case_set_sha256": case_hash(df), "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "counts": counts(step, df)}
    if step == 2 and which == "sample":
        meta.update({"rule": f"simple random {STEP2_N} of the labelled TEST scans with a NIfTI on the mount",
                     "seed": STEP2_SEED, "pool": "per_ct_diagnostic_tasks TEST, any of the 17 items labelled"})
    elif step == 5:
        meta.update({"rule": STEP5_RULE, "seed": STEP5_SEED, "pool": "samples/step2.parquet", "headers": DICOM_HEADERS})
    elif which == "sample":
        meta.update({"seed": SEED, "per_answer": PER_ANSWER})
    if step == 1 and which == "sample":
        meta["note"] = STEP1_NOTE
    (root / f"{_name(step, which)}.json").write_text(json.dumps(meta, indent=1))
    return out


def load(step: int, root: Path | None = None, which: str = "sample") -> pd.DataFrame:
    return pd.read_parquet(Path(root or paths.root()) / "samples" / f"{_name(step, which)}.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--which", choices=["sample", "test"], default="sample")
    ap.add_argument("--force", action="store_true", help="replace an existing file with a different case set")
    args = ap.parse_args()
    # Step 5 reads DICOM headers only, so it builds without BigQuery.
    client = _bq() if any(s != 5 for s in args.steps) else None
    for s in args.steps:
        df = (BUILDERS if args.which == "sample" else TEST_BUILDERS)[s](client)
        out = save(s, df, which=args.which, force=args.force)
        print(f"step {s}: {json.dumps(counts(s, df))} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
