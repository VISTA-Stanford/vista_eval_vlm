"""
1. Gather person_ids from retrieval_subsample_50.csv
2. Check how many are present in bigquery_data for each task in all_tasks.yaml
3. Add new rows to retrieval_subsample_50.csv: for each (person_id, task) from config tasks
   present in bigquery, add an entry (if not already in CSV) with all columns from bigquery,
   but task and question updated from valid_tasks_v1_2.json. Does not delete any existing rows.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Support large CSV fields (e.g. note_text)
import csv
csv.field_size_limit(sys.maxsize)

RETRIEVAL_SUBSAMPLE_CSV = Path(
    "/home/rdcunha/vista_project/vista_bench/v1_2/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/retrieval_subsample_50.csv"
)
BIGQUERY_DATA_PATH = Path(
    "/home/rdcunha/vista_project/vista_bench/bigquery_data_2_3/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "all_tasks.yaml"
VALID_TASKS_PATH = Path(
    "/home/rdcunha/vista_project/vista_bench/tasks/valid_tasks_v1_2.json"
)


def load_valid_tasks_questions(valid_tasks_path: Path) -> dict:
    """Load task_name -> {question, group_name} from valid_tasks_v1_2.json."""
    with open(valid_tasks_path, "r") as f:
        tasks = json.load(f)
    return {
        t["task_name"]: {
            "question": t.get("question", ""),
            "group_name": t.get("group_name", ""),
        }
        for t in tasks
    }


def main():
    # 1. Load existing retrieval_subsample_50.csv (we will append to it, never delete)
    if not RETRIEVAL_SUBSAMPLE_CSV.exists():
        print(f"Error: retrieval_subsample_50.csv not found at {RETRIEVAL_SUBSAMPLE_CSV}")
        return

    df_existing = pd.read_csv(RETRIEVAL_SUBSAMPLE_CSV, sep=None, engine="python", on_bad_lines="warn")
    person_id_col = next((c for c in df_existing.columns if c.lower() == "person_id"), None)
    task_col_ret = next((c for c in df_existing.columns if c.lower() == "task"), None)
    if not person_id_col:
        print("Error: retrieval CSV missing 'person_id' column")
        return

    person_ids = set(df_existing[person_id_col].dropna().astype(str).unique())
    existing_pairs = set()
    if task_col_ret:
        existing_pairs = set(
            zip(
                df_existing[person_id_col].astype(str),
                df_existing[task_col_ret].astype(str),
            )
        )
    print(f"Total unique person_ids in retrieval_subsample_50.csv: {len(person_ids)}")
    print(f"Existing rows: {len(df_existing)}\n")

    # 2. Load tasks from all_tasks.yaml
    if not CONFIG_PATH.exists():
        print(f"Error: config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    tasks = config.get("tasks", [])
    if not tasks:
        print("Error: no tasks defined in config")
        return

    # 3. Load valid_tasks for question and group_name
    if not VALID_TASKS_PATH.exists():
        print(f"Error: valid_tasks not found at {VALID_TASKS_PATH}")
        return

    task_info = load_valid_tasks_questions(VALID_TASKS_PATH)

    # 4. Load bigquery data
    if not BIGQUERY_DATA_PATH.exists():
        print(f"Error: bigquery data not found at {BIGQUERY_DATA_PATH}")
        return

    df_bq = pd.read_csv(BIGQUERY_DATA_PATH, sep=None, engine="python", on_bad_lines="warn")
    task_col = next((c for c in df_bq.columns if c.lower() == "task"), None)
    bq_person_col = next((c for c in df_bq.columns if c.lower() == "person_id"), None)
    task_group_col = next((c for c in df_bq.columns if c.lower() == "task_group"), None)
    question_col = next((c for c in df_bq.columns if c.lower() == "question"), None)
    if not task_col or not bq_person_col:
        print("Error: bigquery data missing 'task' or 'person_id' column")
        return

    # 5. For each task: filter by person_id and task, print total, build new rows to add
    rows_to_add = []
    for task_name in tasks:
        df_task = df_bq[
            (df_bq[task_col].astype(str) == task_name)
            & (df_bq[bq_person_col].astype(str).isin(person_ids))
        ]
        count = df_task[bq_person_col].nunique()
        print(f"{task_name}: {count} person_ids present")

        # Add entries with task and question updated from valid_tasks (only if not already in CSV)
        info = task_info.get(task_name, {})
        question = info.get("question", "")
        group_name = info.get("group_name", "")

        for _, row in df_task.iterrows():
            pid = str(row[bq_person_col])
            if (pid, task_name) in existing_pairs:
                continue
            existing_pairs.add((pid, task_name))

            row_dict = row.to_dict()
            row_dict[task_col] = task_name
            if question and question_col:
                row_dict[question_col] = question
            if group_name and task_group_col:
                row_dict[task_group_col] = group_name
            rows_to_add.append(row_dict)

    # 6. Append new rows to existing CSV and write back to retrieval_subsample_50.csv
    if rows_to_add:
        df_new = pd.DataFrame(rows_to_add)
        # Align columns: use existing CSV columns, fill missing in df_new with NaN
        for c in df_existing.columns:
            if c not in df_new.columns:
                df_new[c] = pd.NA
        df_new = df_new[df_existing.columns]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(RETRIEVAL_SUBSAMPLE_CSV, index=False)
        print(f"\nAdded {len(rows_to_add)} rows. Total rows now: {len(df_combined)}")
        print(f"Updated {RETRIEVAL_SUBSAMPLE_CSV}")
    else:
        print("\nNo new rows to add.")


if __name__ == "__main__":
    main()
