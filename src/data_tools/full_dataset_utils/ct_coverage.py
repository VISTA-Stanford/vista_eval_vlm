"""
Report CT coverage: for each task in all_tasks.yaml, count unique person_id in
vista_bench/bigquery_data_2_3 where task matches, split=='test', and both
nifti_path and _accession_number are not None.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv

SPLIT_COL = "split"
SPLIT_TEST = "test"
TASK_COL = "task"
BQ_DATA_DIR = "bigquery_data_2_3"
NIFTI_COL = "nifti_path"
ACCESSION_COL = "_accession_number"


def _resolve_bq_file(bq_dir: Path, table_name: str) -> Optional[Path]:
    """Resolve BQ file path; try table_name and table_name_v1_1 if not found."""
    candidates = [bq_dir / table_name, bq_dir / f"{table_name}_v1_1"]
    for p in candidates:
        if p.exists():
            return p
    return None


def run(
    config_path: str,
    valid_tasks_json_path: str,
    local_bq_data_dir: str | Path | None = None,
) -> None:
    """
    For each task in config, count unique person_id with:
    - task == task_name
    - split == 'test'
    - nifti_path not None
    - _accession_number not None
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

    print(f"CT coverage (split=test, nifti_path and _accession_number non-null)")
    print(f"BQ dir: {local_bq_data_dir}")
    print("-" * 60)

    results = []
    all_person_ids = set()
    for task_name in sorted(tasks):
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"{task_name}: [SKIP] No task_source_csv")
            results.append((task_name, None, "no task_source_csv"))
            continue

        bq_file = _resolve_bq_file(local_bq_data_dir, source_csv)
        if bq_file is None:
            print(f"{task_name}: [SKIP] BQ file not found (table: {source_csv})")
            results.append((task_name, None, "bq file not found"))
            continue

        try:
            df_all = pd.read_csv(bq_file, sep=None, engine="python", on_bad_lines="warn")
        except Exception as e:
            print(f"{task_name}: [ERROR] {e}")
            results.append((task_name, None, str(e)))
            continue

        if SPLIT_COL not in df_all.columns or TASK_COL not in df_all.columns:
            print(f"{task_name}: [SKIP] Missing split/task columns")
            results.append((task_name, None, "missing columns"))
            continue

        if NIFTI_COL not in df_all.columns:
            print(f"{task_name}: [SKIP] No '{NIFTI_COL}' column")
            results.append((task_name, None, "no nifti_path column"))
            continue

        if ACCESSION_COL not in df_all.columns:
            print(f"{task_name}: [SKIP] No '{ACCESSION_COL}' column")
            results.append((task_name, None, "no _accession_number column"))
            continue

        df_filtered = df_all[
            (df_all[SPLIT_COL].astype(str).str.strip().str.lower() == SPLIT_TEST.lower())
            & (df_all[TASK_COL].astype(str) == task_name)
            & df_all[NIFTI_COL].notna()
            & (df_all[NIFTI_COL].astype(str).str.strip() != "")
            & df_all[ACCESSION_COL].notna()
            & (df_all[ACCESSION_COL].astype(str).str.strip() != "")
        ]

        n_unique = df_filtered["person_id"].nunique()
        n_rows = len(df_filtered)
        print(f"{task_name}: {n_unique} unique person_id ({n_rows} rows)")
        results.append((task_name, n_unique, None))
        all_person_ids.update(df_filtered["person_id"].dropna().astype(str).unique())

    print("-" * 60)
    print(f"Unique person_id across all tasks: {len(all_person_ids)}")

    out_dir = base_path / "ct_coverage_all_vb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "person_ids.csv"
    pd.DataFrame({"person_id": sorted(all_person_ids)}).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Report CT coverage: unique person_id per task with nifti_path and _accession_number."
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
        "--bq-dir",
        default=None,
        help="Path to bigquery_data_2_3 (default: {base_dir}/bigquery_data_2_3)",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        local_bq_data_dir=args.bq_dir,
    )
