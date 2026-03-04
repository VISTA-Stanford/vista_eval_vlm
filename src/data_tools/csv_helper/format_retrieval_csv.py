"""
Create _subsampled_retrieval CSV files for tasks in all_tasks.yaml.

Uses retrieval_subsample_50.csv when available (multi-task CSV with person_ids
present in all retrieval tasks). Falls back to per-task _subsampled.csv otherwise.

Output CSVs include only columns needed for retrieval-based inference (no note_text).
"""

from pathlib import Path

import pandas as pd

from data_tools.utils.config_utils import load_tasks_and_base_dir, load_task_source_csv

RETRIEVAL_COLUMNS = [
    "person_id",
    "split",
    "embed_time",
    "label",
    "task",
    "task_group",
    "question",
    "label_description",
    "latest_img_date",
    "_accession_number",
    "modality",
    "anatomic_site_source_value",
    "image_study_uid",
    "image_series_uid",
    "local_path",
    "nifti_path",
]

RETRIEVAL_SUBSAMPLE_FILENAME = "retrieval_subsample_50.csv"


def run(
    config_path: str,
    valid_tasks_json_path: str,
    overwrite: bool = True,
    retrieval_csv_path: str | Path | None = None,
) -> None:
    tasks, base_dir = load_tasks_and_base_dir(config_path)
    task_to_source = load_task_source_csv(valid_tasks_json_path)
    base_path = Path(base_dir)
    v1_2_path = base_path / "v1_2"

    if not base_path.exists():
        print(f"Error: base_dir not found: {base_path}")
        return

    # Load retrieval_subsample_50.csv if it exists (per source_csv)
    retrieval_df_by_source: dict[str, pd.DataFrame] = {}
    if retrieval_csv_path:
        path = Path(retrieval_csv_path)
        if path.exists():
            source_csv = path.parent.name
            try:
                df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="warn")
                if "task" in df.columns:
                    retrieval_df_by_source[source_csv] = df
                    print(f"Loaded retrieval CSV: {path} ({len(df)} rows)")
            except Exception as e:
                print(f"[WARN] Could not load {path}: {e}")
    for source_csv in set(task_to_source.values()):
        if source_csv in retrieval_df_by_source:
            continue
        retrieval_path = v1_2_path / source_csv / RETRIEVAL_SUBSAMPLE_FILENAME
        if not retrieval_path.exists():
            continue
        try:
            df = pd.read_csv(retrieval_path, sep=None, engine="python", on_bad_lines="warn")
            if "task" in df.columns:
                retrieval_df_by_source[source_csv] = df
                print(f"Loaded retrieval CSV: {retrieval_path} ({len(df)} rows)")
        except Exception as e:
            print(f"[WARN] Could not load {retrieval_path}: {e}")

    for task_name in tasks:
        source_csv = task_to_source.get(task_name)
        if not source_csv:
            print(f"[SKIP] No task_source_csv for task '{task_name}'")
            continue

        out_dir = v1_2_path / source_csv
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task_name}_subsampled_retrieval.csv"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path.name}: already exists (use --overwrite to replace)")
            continue

        df_out = None

        # 1. Try retrieval_subsample_50.csv (filter by task)
        if source_csv in retrieval_df_by_source:
            df_ret = retrieval_df_by_source[source_csv]
            task_col = next((c for c in df_ret.columns if c.lower() == "task"), None)
            if task_col:
                df_task = df_ret[df_ret[task_col].astype(str) == task_name]
                if len(df_task) > 0:
                    missing = [c for c in RETRIEVAL_COLUMNS if c not in df_task.columns]
                    if not missing:
                        df_out = df_task[[c for c in RETRIEVAL_COLUMNS if c in df_task.columns]].copy()
                        # Ensure column order matches RETRIEVAL_COLUMNS
                        df_out = df_out[[c for c in RETRIEVAL_COLUMNS if c in df_out.columns]]

        # 2. Fallback: per-task _subsampled.csv
        if df_out is None:
            subsampled_path = base_path / source_csv / f"{task_name}_subsampled.csv"
            if not subsampled_path.exists():
                subsampled_path = v1_2_path / source_csv / f"{task_name}_subsampled.csv"
            if not subsampled_path.exists():
                print(f"[SKIP] No source for task '{task_name}' (retrieval CSV has no rows, _subsampled.csv not found)")
                continue

            try:
                df = pd.read_csv(subsampled_path, sep=None, engine="python", on_bad_lines="warn")
            except Exception as e:
                print(f"[ERR]  {subsampled_path.name}: {e}")
                continue

            missing = [c for c in RETRIEVAL_COLUMNS if c not in df.columns]
            if missing:
                print(f"[SKIP] {task_name}: missing columns: {missing}")
                continue

            df_out = df[RETRIEVAL_COLUMNS].copy()

        df_out.to_csv(out_path, index=False)
        print(f"[OK]   {out_path.name}: {len(df_out)} rows")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create _subsampled_retrieval CSVs from _subsampled CSVs for config tasks.",
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
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing _subsampled_retrieval files (default: True)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing _subsampled_retrieval files",
    )
    parser.add_argument(
        "--retrieval-csv",
        type=str,
        default=None,
        help="Path to retrieval_subsample_50.csv (default: v1_2/{source_csv}/retrieval_subsample_50.csv)",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        valid_tasks_json_path=args.valid_tasks,
        overwrite=args.overwrite,
        retrieval_csv_path=args.retrieval_csv,
    )
