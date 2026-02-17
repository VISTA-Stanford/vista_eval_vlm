"""
Overwrite _subsampled.csv files using person_ids from retrieval_subsample_50.csv.

For each task in all_tasks.yaml:
  1. Load person_ids from vista_bench/v1_2/{task}/retrieval_subsample_50.csv (filter by task)
  2. Get column names from vista_bench/bigquery_data_2_3/{task} CSV (or existing _subsampled)
  3. Filter bigquery_data_2_3 rows by person_id and task
  4. Add patient_string (patient timeline from embed_time to 6 months before, via get_described_events_window + get_llm_event_string with exclude_report=False)
  5. Ensure note_text column is present
  6. Overwrite vista_bench/v1_2/{task}/{subtask}_subsampled.csv

Note: patient_string and note_text are not in the bigquery table; patient_string is built from MEDS.
"""

import sys
from pathlib import Path

import pandas as pd

# Support large CSV fields (e.g. note_text)
import csv
csv.field_size_limit(sys.maxsize)

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv
from data_tools.utils.meds_timeline_utils import get_llm_event_string

RETRIEVAL_SUBSAMPLE_FILENAME = "retrieval_subsample_50.csv"
BIGQUERY_DATA_DIR = "bigquery_data_2_3"
DEFAULT_MEDS_DB = "/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/thoracic_cohort_meds_femr_db"
DEFAULT_ONTOLOGY = "/home/rdcunha/vista_project/vista_bench/thoracic_cohort_meds/athena_omop_ontologies"


def run(
    config_path: str,
    valid_tasks_json_path: str,
    base_dir: str | Path | None = None,
    bigquery_dir: str | Path | None = None,
    meds_db_path: str | Path | None = None,
    ontology_path: str | Path | None = None,
    overwrite: bool = True,
) -> None:
    tasks, cfg_base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir or cfg_base_dir)
    v1_2_path = base_path / "v1_2"
    bq_path = Path(bigquery_dir or base_path / BIGQUERY_DATA_DIR)

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    if not bq_path.exists():
        print(f"Error: bigquery_data_2_3 not found: {bq_path}")
        return

    # Init MEDS database and ontology for patient_string (patient timeline)
    database = None
    lookup_table = None
    meds_path = Path(meds_db_path or DEFAULT_MEDS_DB)
    ont_path = Path(ontology_path or DEFAULT_ONTOLOGY)
    if meds_path.exists() and ont_path.exists():
        try:
            import meds_reader
            from meds2text.ontology import OntologyDescriptionLookupTable

            database = meds_reader.SubjectDatabase(str(meds_path))
            lookup_table = OntologyDescriptionLookupTable()
            lookup_table.load(str(ont_path))
            print("Loaded MEDS database and ontology for patient_string.")
        except Exception as e:
            print(f"[WARN] Could not load MEDS/ontology for patient_string: {e}")
    else:
        print(f"[WARN] MEDS db or ontology not found (meds={meds_path}, ontology={ont_path}). Skipping patient_string.")

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        retrieval_path = v1_2_path / source_csv / RETRIEVAL_SUBSAMPLE_FILENAME
        if not retrieval_path.exists():
            print(f"[SKIP] {task_name}: retrieval_subsample_50.csv not found at {retrieval_path}")
            continue

        # BigQuery CSV path (no .csv extension in vista_bench)
        bq_csv_path = bq_path / source_csv
        if not bq_csv_path.exists():
            # Try with .csv extension
            bq_csv_path = bq_path / f"{source_csv}.csv"
        if not bq_csv_path.exists():
            print(f"[SKIP] {task_name}: bigquery CSV not found at {bq_path / source_csv}")
            continue

        subsampled_path = v1_2_path / source_csv / f"{task_name}_subsampled.csv"
        if subsampled_path.exists() and not overwrite:
            print(f"[SKIP] {task_name}: {subsampled_path.name} exists (use --overwrite to replace)")
            continue

        try:
            # 1. Load person_ids from retrieval_subsample_50.csv (filter by task)
            df_ret = pd.read_csv(
                retrieval_path, sep=None, engine="python", on_bad_lines="warn"
            )
            task_col = next((c for c in df_ret.columns if c.lower() == "task"), None)
            person_id_col = next(
                (c for c in df_ret.columns if c.lower() == "person_id"), None
            )
            if not task_col or not person_id_col:
                print(
                    f"[SKIP] {task_name}: retrieval CSV missing 'task' or 'person_id' column"
                )
                continue

            df_task_ret = df_ret[df_ret[task_col].astype(str) == task_name]
            person_ids = set(df_task_ret[person_id_col].dropna().astype(str).unique())
            if len(person_ids) == 0:
                print(f"[SKIP] {task_name}: no person_ids in retrieval CSV for this task")
                continue

            # 2. Get column names: from existing _subsampled if present, else from bigquery
            if subsampled_path.exists():
                df_sub = pd.read_csv(
                    subsampled_path, sep=None, engine="python", on_bad_lines="warn", nrows=0
                )
                output_columns = list(df_sub.columns)
            else:
                output_columns = None  # will use bigquery columns

            # 3. Load bigquery CSV and filter by person_id and task
            df_bq = pd.read_csv(
                bq_csv_path
            )
            bq_task_col = next((c for c in df_bq.columns if c.lower() == "task"), None)
            bq_person_col = next(
                (c for c in df_bq.columns if c.lower() == "person_id"), None
            )
            if not bq_task_col or not bq_person_col:
                print(
                    f"[SKIP] {task_name}: bigquery CSV missing 'task' or 'person_id' column"
                )
                continue

            mask_task = df_bq[bq_task_col].astype(str) == task_name
            mask_person = df_bq[bq_person_col].astype(str).isin(person_ids)
            df_out = df_bq[mask_task & mask_person].copy()

            if len(df_out) == 0:
                print(
                    f"[SKIP] {task_name}: no matching rows in bigquery for person_ids from retrieval"
                )
                continue

            # 4. Select columns (same as _subsampled or bigquery), ensure note_text
            if output_columns is not None:
                # Keep only columns that exist in both
                cols = [c for c in output_columns if c in df_out.columns]
                # missing = [c for c in output_columns if c not in df_out.columns]
                # if missing:
                #     print(
                #         f"[WARN] {task_name}: bigquery missing columns {missing}, output may differ"
                #     )
                df_out = df_out[cols]
            else:
                df_out = df_out

            # Ensure note_text column (bigquery has it; add empty if missing)
            if "note_text" not in df_out.columns:
                df_out["note_text"] = ""

            # 5. Add patient_string (patient timeline: embed_time to 6 months before)
            # Uses get_described_events_window + get_llm_event_string with exclude_report=False
            if database is not None and lookup_table is not None:
                try:
                    from meds_tools import patient_timeline

                    embed_times = pd.to_datetime(
                        df_out["embed_time"], errors="coerce"
                    )
                    start_times = embed_times - pd.DateOffset(months=6)
                    new_strings = []
                    for i, row in df_out.iterrows():
                        subject_id = row["person_id"]
                        end_time = embed_times.loc[i]
                        start_time = start_times.loc[i]
                        if pd.isna(end_time) or pd.isna(subject_id):
                            new_strings.append("")
                            continue
                        try:
                            window_df = patient_timeline.get_described_events_window(
                                database=database,
                                lookup_table=lookup_table,
                                subject_id=subject_id,
                                end_time=end_time,
                                start_time=start_time,
                            )
                            if (
                                not window_df.empty
                                and "time" not in window_df.columns
                                and hasattr(window_df.index, "names")
                                and window_df.index.names
                            ):
                                window_df = window_df.reset_index()
                            patient_string = get_llm_event_string(
                                window_df,
                                include_text=True,
                                exclude_report=False,
                            )
                        except Exception as e:
                            print(
                                f"  [WARN] Row {i} (person_id={subject_id}): {e}"
                            )
                            patient_string = ""
                        new_strings.append(patient_string)
                    df_out["patient_string"] = new_strings
                except Exception as e:
                    print(f"[WARN] {task_name}: patient_string failed: {e}")
                    df_out["patient_string"] = ""
            else:
                df_out["patient_string"] = ""

            subsampled_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(subsampled_path, index=False)
            print(f"[OK]   {task_name}: wrote {subsampled_path.name} ({len(df_out)} rows)")

        except Exception as e:
            print(f"[ERR]  {task_name}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overwrite _subsampled.csv files with person_ids from retrieval_subsample_50.csv.",
    )
    parser.add_argument(
        "--config",
        default="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
        help="Path to all_tasks.yaml",
    )
    parser.add_argument(
        "--valid-tasks",
        default="/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks_v1_2.json",
        help="Path to valid_tasks JSON",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: from config paths.base_dir)",
    )
    parser.add_argument(
        "--bigquery-dir",
        type=str,
        default=None,
        help=f"Path to bigquery data (default: {{base_dir}}/{BIGQUERY_DATA_DIR})",
    )
    parser.add_argument(
        "--meds-db",
        type=str,
        default=DEFAULT_MEDS_DB,
        help="Path to MEDS database for patient_string (patient timeline)",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=DEFAULT_ONTOLOGY,
        help="Path to ontology for patient_string",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing _subsampled files (default: True)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing _subsampled files",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        base_dir=args.base_dir,
        bigquery_dir=args.bigquery_dir,
        meds_db_path=args.meds_db,
        ontology_path=args.ontology,
        overwrite=args.overwrite,
    )
