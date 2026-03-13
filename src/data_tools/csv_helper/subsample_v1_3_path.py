"""Subsample CSVs from vista_bench/bigquery_v1_3 for v1_3 path-based tasks.

Reads from bigquery CSVs where the file name matches task_source_csv from valid_tasks_v1_3.json.
Filters rows where: split == 'test', label != -1, and path_image_path exists (non-null, non-empty).
Writes {task_name}_path_subsampled.csv to vista_bench/v1_3/{task_source_csv}/.
Copies all columns from the source bigquery CSV. No row limit; keeps all matching rows.
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

# Paths (vista_bench_cached)
BIGQUERY_DIR = Path("/data/fries/users/rdcunha/vista_bench_cached/vista_bench/bigquery_v1_3")
VALID_TASKS_JSON = Path("/data/fries/users/rdcunha/vista_bench_cached/vista_bench/tasks/valid_tasks_v1_3.json")
OUTPUT_BASE = Path("/data/fries/users/rdcunha/vista_bench_cached/vista_bench/v1_3")


def load_valid_tasks(valid_tasks_json_path: str | Path) -> list[dict]:
    """Load valid tasks from JSON. Returns list of task dicts with task_name and task_source_csv."""
    with open(valid_tasks_json_path) as f:
        tasks = json.load(f)
    return tasks


def subsample_one_task_path(
    source_csv_path: Path,
    task_name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[str, str, str]:
    """
    Filter one task from a bigquery CSV: split=='test', label != -1, path_image_path exists.
    Returns (status, path_or_name, message).
    """
    output_path = output_dir / f"{task_name}_path_subsampled.csv"
    if output_path.exists() and not overwrite:
        return (
            "skip",
            str(output_path),
            "output file already exists (use overwrite=True to replace)",
        )

    try:
        df = pd.read_csv(source_csv_path, sep=None, engine="python", on_bad_lines="warn")

        # Filter by task
        task_col = next((c for c in df.columns if c.lower() == "task"), None)
        if task_col is None:
            return ("skip", str(source_csv_path), f"no 'task' column. cols={list(df.columns)}")
        df = df[df[task_col].astype(str) == str(task_name)]

        if len(df) == 0:
            return ("skip", str(output_path), f"no rows for task '{task_name}'")

        # Filter: split == 'test'
        split_col = next((c for c in df.columns if c.lower() == "split"), None)
        if split_col is None:
            return ("skip", str(source_csv_path), f"no 'split' column. cols={list(df.columns)}")
        df = df[df[split_col].astype(str).str.strip().str.lower() == "test"]

        if len(df) == 0:
            return ("skip", str(output_path), f"no test split rows for task '{task_name}'")

        # Filter: label != -1 (exclude only -1; keep all other values including NaN)
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is not None:
            # Exclude rows where label is -1 (int, float, or string "-1"/"-1.0")
            str_vals = df[label_col].astype(str).str.strip()
            exclude_str = str_vals.isin(("-1", "-1.0"))
            try:
                exclude_numeric = (df[label_col] == -1) | (df[label_col] == -1.0)
            except (TypeError, ValueError):
                exclude_numeric = False
            df = df[~(exclude_str | exclude_numeric)]

        if len(df) == 0:
            return ("skip", str(output_path), f"no rows with label != -1 for task '{task_name}'")

        # Filter: path_image_path exists (non-null, non-empty)
        path_col = next((c for c in df.columns if c.replace(" ", "").lower() == "path_image_path"), None)
        if path_col is None:
            path_col = next((c for c in df.columns if "path_image" in c.lower() or "image_path" in c.lower()), None)
        if path_col is None:
            return (
                "skip",
                str(source_csv_path),
                f"no 'path_image_path' (or similar) column. cols={list(df.columns)}",
            )
        mask_path = df[path_col].notna() & (df[path_col].astype(str).str.strip() != "")
        df = df[mask_path]

        if len(df) == 0:
            return (
                "skip",
                str(output_path),
                f"no rows with path_image_path for task '{task_name}'",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return (
            "ok",
            str(output_path),
            f"wrote {output_path.name} ({len(df)} rows)",
        )

    except Exception as e:
        return ("err", str(output_path), repr(e))


def subsample_v1_3_path(
    bigquery_dir: str | Path | None = None,
    valid_tasks_json_path: str | Path | None = None,
    output_base: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """
    Subsample path-based tasks from bigquery_v1_3 into vista_bench/v1_3.

    For each task in valid_tasks_v1_3.json: reads the bigquery CSV for task_source_csv,
    keeps rows with split=='test', label != -1, and path_image_path present; writes
    {task_name}_path_subsampled.csv to output_base / task_source_csv /.

    Args:
        bigquery_dir: Directory with bigquery CSVs (default: BIGQUERY_DIR)
        valid_tasks_json_path: Path to valid_tasks_v1_3.json (default: VALID_TASKS_JSON)
        output_base: Output base directory (default: OUTPUT_BASE)
        overwrite: If True, overwrite existing _path_subsampled.csv files
    """
    bigquery_dir = Path(bigquery_dir) if bigquery_dir is not None else BIGQUERY_DIR
    valid_tasks_json_path = (
        Path(valid_tasks_json_path) if valid_tasks_json_path is not None else VALID_TASKS_JSON
    )
    output_base = Path(output_base) if output_base is not None else OUTPUT_BASE

    tasks = load_valid_tasks(valid_tasks_json_path)
    print(f"Loaded {len(tasks)} tasks from {valid_tasks_json_path}")

    # Group by task_source_csv
    source_to_tasks: dict[str, list[str]] = {}
    for t in tasks:
        src = t.get("task_source_csv")
        name = t.get("task_name")
        if not src or not name:
            continue
        source_to_tasks.setdefault(src, []).append(name)

    ok = skip = err = 0
    for task_source_csv, task_names in source_to_tasks.items():
        source_path = bigquery_dir / task_source_csv
        if not source_path.exists():
            source_path = bigquery_dir / f"{task_source_csv}.csv"
        if not source_path.exists():
            print(f"[SKIP] Source not found: {task_source_csv}")
            skip += len(task_names)
            continue

        out_task_dir = output_base / task_source_csv
        for task_name in task_names:
            status, path_or_name, msg = subsample_one_task_path(
                source_csv_path=source_path,
                task_name=task_name,
                output_dir=out_task_dir,
                overwrite=overwrite,
            )
            if status == "ok":
                ok += 1
                print(f"[OK]   {task_source_csv}/{task_name}: {msg}")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {task_source_csv}/{task_name}: {msg}")
            else:
                err += 1
                print(f"[ERR]  {task_source_csv}/{task_name}: {msg}")

    print(f"\nDone. OK={ok}, SKIP={skip}, ERR={err}")


if __name__ == "__main__":
    subsample_v1_3_path(overwrite=True)
