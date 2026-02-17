"""
Load full test split data from bigquery_data_2_3 for each task in all_tasks.yaml.
For each task: filter task==task_name and split=='test', run remove_imaging_report
logic to add patient_string, and save as parquet with name {task_name}_full.parquet
in the task folder (task_source_csv).
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv
from data_tools.OMOP_meds_query.remove_imaging_report import process_dataframe

import meds_reader
from meds2text.ontology import OntologyDescriptionLookupTable

SPLIT_COL = "split"
SPLIT_TEST = "test"
TASK_COL = "task"
BQ_DATA_DIR = "bigquery_data_2_3"


def _resolve_bq_file(bq_dir: Path, table_name: str) -> Optional[Path]:
    """Resolve BQ file path; try table_name and table_name_v1_1 if not found."""
    candidates = [bq_dir / table_name, bq_dir / f"{table_name}_v1_1"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _process_one_task(
    task_name: str,
    source_csv: str,
    base_path: Path,
    local_bq_data_dir: Path,
    meds_db_path: str,
    ontology_path: str,
    overwrite: bool,
) -> tuple[str, str, int | None]:
    """
    Process a single task (worker function for parallel execution).
    Returns (task_name, status, row_count) where status is 'ok', 'skip', or 'error'.
    """
    bq_file = _resolve_bq_file(local_bq_data_dir, source_csv)
    if bq_file is None:
        return (task_name, "skip", None)

    out_dir = base_path / source_csv
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_full.parquet"

    if out_path.exists() and not overwrite:
        return (task_name, "skip", None)

    try:
        df_all = pd.read_csv(bq_file, sep=None, engine="python", on_bad_lines="warn")
    except Exception as e:
        return (task_name, f"error: {e}", None)

    if SPLIT_COL not in df_all.columns or TASK_COL not in df_all.columns:
        return (task_name, "skip", None)
    if "embed_time" not in df_all.columns or "label" not in df_all.columns:
        return (task_name, "skip", None)

    label_ok = (df_all["label"] != -1) & (df_all["label"].astype(str) != "-1")
    df_test = df_all[
        (df_all[SPLIT_COL].astype(str).str.strip().str.lower() == SPLIT_TEST.lower())
        & (df_all[TASK_COL].astype(str) == task_name)
        & df_all["embed_time"].notna()
        & label_ok
    ].copy()

    if df_test.empty:
        return (task_name, "skip", None)

    lookup = OntologyDescriptionLookupTable()
    lookup.load(ontology_path)
    database = meds_reader.SubjectDatabase(meds_db_path)

    try:
        df_out = process_dataframe(
            df_test,
            database=database,
            lookup_table=lookup,
            embed_time_col="embed_time",
            person_id_col="person_id",
            months_before=6,
        )
        df_out.to_parquet(out_path, index=False)
        return (task_name, "ok", len(df_out))
    except Exception as e:
        return (task_name, f"error: {e}", None)


def run(
    config_path: str,
    valid_tasks_json_path: str,
    meds_db_path: str,
    ontology_path: str,
    local_bq_data_dir: str | Path | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    """
    For each task in config:
    1. Load data from bigquery_data_2_3 where task matches and split=='test'
    2. Run remove_imaging_report logic to add patient_string
    3. Save as {task_name}_full.parquet in the task folder

    When workers > 1, tasks are processed in parallel.
    """
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)

    if local_bq_data_dir is None:
        local_bq_data_dir = base_path / BQ_DATA_DIR
    else:
        local_bq_data_dir = Path(local_bq_data_dir)

    if not local_bq_data_dir.exists():
        print(f"Error: BQ data dir not found: {local_bq_data_dir}")
        return

    # Build list of (task_name, source_csv) to process
    work_items = []
    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue
        work_items.append((task_name, source_csv))

    if not work_items:
        print("No tasks to process.")
        return

    workers = workers if workers > 1 else 1
    print(f"Processing {len(work_items)} tasks with {workers} worker(s)")

    if workers == 1:
        # Sequential: reuse shared lookup/database
        lookup = OntologyDescriptionLookupTable()
        lookup.load(ontology_path)
        database = meds_reader.SubjectDatabase(meds_db_path)

        for task_name, source_csv in work_items:
            bq_file = _resolve_bq_file(local_bq_data_dir, source_csv)
            if bq_file is None:
                print(f"[SKIP] BQ file not found for task '{task_name}' (table: {source_csv})")
                continue

            out_dir = base_path / source_csv
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task_name}_full.parquet"

            if out_path.exists() and not overwrite:
                print(f"[SKIP] Output exists (use overwrite=True): {out_path.name}")
                continue

            print(f"Loading: {bq_file.name} for task '{task_name}'")
            try:
                df_all = pd.read_csv(bq_file, sep=None, engine="python", on_bad_lines="warn")
            except Exception as e:
                print(f"[ERROR] Failed to read {bq_file}: {e}")
                continue

            if SPLIT_COL not in df_all.columns or TASK_COL not in df_all.columns:
                continue
            if "embed_time" not in df_all.columns or "label" not in df_all.columns:
                continue

            label_ok = (df_all["label"] != -1) & (df_all["label"].astype(str) != "-1")
            df_test = df_all[
                (df_all[SPLIT_COL].astype(str).str.strip().str.lower() == SPLIT_TEST.lower())
                & (df_all[TASK_COL].astype(str) == task_name)
                & df_all["embed_time"].notna()
                & label_ok
            ].copy()

            if df_test.empty:
                print(f"[SKIP] No rows with task=='{task_name}' and split=='test'")
                continue

            print(f"  Rows: {len(df_test)}. Running remove_imaging_report logic...")
            try:
                df_out = process_dataframe(
                    df_test,
                    database=database,
                    lookup_table=lookup,
                    embed_time_col="embed_time",
                    person_id_col="person_id",
                    months_before=6,
                )
                df_out.to_parquet(out_path, index=False)
                print(f"  Wrote {len(df_out)} rows to {out_path}")
            except Exception as e:
                print(f"[ERROR] {task_name}: {e}")
                raise
    else:
        # Parallel: each worker initializes its own database/lookup
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one_task,
                    task_name,
                    source_csv,
                    base_path,
                    local_bq_data_dir,
                    meds_db_path,
                    ontology_path,
                    overwrite,
                ): (task_name, source_csv)
                for task_name, source_csv in work_items
            }

            for future in as_completed(futures):
                task_name, source_csv = futures[future]
                try:
                    result_task, status, row_count = future.result()
                    if status == "ok":
                        print(f"[OK]   {task_name}: wrote {row_count} rows")
                    elif status == "skip":
                        print(f"[SKIP] {task_name}: skipped")
                    else:
                        print(f"[ERROR] {task_name}: {status}")
                except Exception as e:
                    print(f"[ERROR] {task_name}: {e}")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build full test parquet files with patient_string from bigquery_data_2_3."
    )
    parser.add_argument(
        "--config",
        default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
        help="Path to all_tasks.yaml",
    )
    parser.add_argument(
        "--valid-tasks",
        default="/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks.json",
        help="Path to valid_tasks JSON",
    )
    parser.add_argument(
        "--meds-db",
        # default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/vista_thoracic_cohort_v0_db",
        default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/thoracic_cohort_meds_femr_db",
        help="Path to meds reader DB",
    )
    parser.add_argument(
        "--ontology",
        default="/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/athena_omop_ontologies",
        help="Path to ontology",
    )
    parser.add_argument(
        "--bq-dir",
        default=None,
        help="Path to bigquery_data_2_3 (default: {base_dir}/bigquery_data_2_3)",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        help="Overwrite existing _full.parquet files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 1, max: 4 recommended)",
    )
    args = parser.parse_args()

    workers = min(max(1, args.workers), 4)

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        meds_db_path=args.meds_db,
        ontology_path=args.ontology,
        local_bq_data_dir=args.bq_dir,
        overwrite=args.overwrite,
        workers=workers,
    )
