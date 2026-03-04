"""Subsample retrieval CSV: find person_ids present in all specified tasks (split=test, label in [0,1])."""

from pathlib import Path

import numpy as np
import pandas as pd

# Tasks required for retrieval subsampling (person_id must appear in ALL of these)
RETRIEVAL_TASKS = [
    "has_recurrence_1_yr",
    # "has_recurrence_2_yr",
    "progression_recurrence_free_survival_1_yr",
    # "progression_recurrence_free_survival_2_yr",
    "has_progression_nonrecurrence_1_yr",
    # "has_progression_nonrecurrence_2_yr",
    # "died_of_cancer_1_yr",
    # "died_of_cancer_2_yr",
    # "died_any_cause_1_yr",
    # "died_any_cause_2_yr",
]

# Columns that must be present (non-null, non-empty) for a person_id to be valid
REQUIRED_COLS = ["_accession_number", "note_text", "nifti_path"]


def subsample_retrieval_csv(
    csv_path,
    output_dir,
    target_n=50,
    split="test",
    tasks=None,
    seed=42,
):
    """Create a list of person_ids that appear in ALL specified tasks with split=test and label in [0,1].
    Randomly sample target_n person_ids if more fit the criteria.
    Report prevalence (0 vs 1) for each task.
    Save the filtered CSV to output_dir."""
    if tasks is None:
        tasks = RETRIEVAL_TASKS

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")

    # Normalize column names
    split_col = next((c for c in df.columns if c.lower() == "split"), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    task_col = next((c for c in df.columns if c.lower() == "task"), None)
    person_id_col = next((c for c in df.columns if c.lower() == "person_id"), None)

    if not all([split_col, label_col, task_col, person_id_col]):
        raise ValueError(
            f"Missing required columns. split={split_col}, label={label_col}, "
            f"task={task_col}, person_id={person_id_col}"
        )

    # Resolve required columns (_accession_number, note_text, nifti_path)
    required_col_map = {}
    for col_name in REQUIRED_COLS:
        c = next((c for c in df.columns if c.strip().lower() == col_name.lower()), None)
        if c is None:
            raise ValueError(f"Missing required column '{col_name}' for valid person_id")
        required_col_map[col_name] = c

    # Filter: split=test, label in [0,1]
    mask_split = df[split_col].astype(str).str.strip().str.lower() == split.lower()
    mask_label = df[label_col].apply(
        lambda x: x in (0, 1) or (isinstance(x, (int, float)) and x in (0.0, 1.0))
    )
    df_filtered = df[mask_split & mask_label].copy()

    # Filter: require _accession_number, note_text, nifti_path all present (non-null, non-empty)
    def is_present(val):
        if pd.isna(val):
            return False
        s = str(val).strip()
        return len(s) > 0

    for col_name, col in required_col_map.items():
        mask = df_filtered[col].apply(is_present)
        before = len(df_filtered)
        df_filtered = df_filtered[mask]
        print(f"  After requiring '{col_name}': {len(df_filtered)} rows (dropped {before - len(df_filtered)})")

    print(f"After filter (split={split}, label in [0,1], required cols present): {len(df_filtered)} rows")

    # For each task, get person_ids present
    person_ids_by_task = {}
    for t in tasks:
        task_rows = df_filtered[df_filtered[task_col].astype(str) == t]
        pids = task_rows[person_id_col].dropna().astype(str).unique().tolist()
        person_ids_by_task[t] = set(pids)
        print(f"  {t}: {len(pids)} person_ids")

    # Intersection: person_ids in ALL tasks
    common = person_ids_by_task[tasks[0]].copy()
    for t in tasks[1:]:
        common &= person_ids_by_task[t]

    total_fitting = len(common)
    print(f"\nPerson_ids present in ALL {len(tasks)} tasks: {total_fitting}")

    # For each task, get person_ids with label 1 (within common)
    label1_by_task = {}
    for t in tasks:
        task_rows = df_filtered[
            (df_filtered[task_col].astype(str) == t) & (df_filtered[label_col] == 1)
        ]
        pids = set(task_rows[person_id_col].dropna().astype(str).unique()) & common
        label1_by_task[t] = pids

    common_list = sorted(common)
    rng = np.random.default_rng(seed)
    min_label1_per_task = 5

    if len(common_list) > target_n:
        # Stratified sampling: ensure at least min_label1_per_task with label 1 per task (if possible)
        selected = set()
        for t in tasks:
            current_label1_count = len(selected & label1_by_task[t])
            needed = min_label1_per_task - current_label1_count
            if needed <= 0:
                continue
            available = list(label1_by_task[t] - selected)
            if not available:
                n_avail = len(label1_by_task[t])
                print(
                    f"  Task {t}: fewer than {min_label1_per_task} label-1 in pool "
                    f"(only {n_avail} available)"
                )
                continue
            n_to_add = min(needed, len(available))
            chosen = rng.choice(available, size=n_to_add, replace=False)
            selected.update(chosen.tolist() if hasattr(chosen, "tolist") else [chosen])

        # Fill remaining slots with random sample from common
        remaining_slots = target_n - len(selected)
        if remaining_slots > 0:
            available = list(common - selected)
            n_to_add = min(remaining_slots, len(available))
            chosen = rng.choice(available, size=n_to_add, replace=False)
            selected.update(chosen.tolist() if hasattr(chosen, "tolist") else [chosen])

        n_label1_ensured = sum(
            1 for t in tasks if len(selected & label1_by_task[t]) >= min_label1_per_task
        )
        print(
            f"Stratified sample: {len(selected)} person_ids "
            f"({n_label1_ensured}/{len(tasks)} tasks with >= {min_label1_per_task} label-1, seed={seed})"
        )
    else:
        selected = set(common_list)
        print(f"Using all {len(selected)} person_ids (fewer than target_n={target_n})")

    # Filter original df to selected person_ids and the 6 tasks
    mask_pid = df[person_id_col].astype(str).isin(selected)
    mask_task = df[task_col].astype(str).isin(tasks)
    mask_split_full = df[split_col].astype(str).str.strip().str.lower() == split.lower()
    mask_label_full = df[label_col].apply(
        lambda x: x in (0, 1) or (isinstance(x, (int, float)) and x in (0.0, 1.0))
    )
    out_df = df[mask_pid & mask_task & mask_split_full & mask_label_full].copy()

    # Report prevalence for each task
    print("\nPrevalence (label 0 vs 1) for the selected patients:")
    print("-" * 60)
    for t in tasks:
        task_rows = out_df[out_df[task_col].astype(str) == t]
        n0 = (task_rows[label_col] == 0).sum()
        n1 = (task_rows[label_col] == 1).sum()
        n_total = len(task_rows)
        print(f"  {t}: 0={n0}, 1={n1} (total={n_total})")

    # Save output
    dataset_name = csv_path.stem if csv_path.suffix else csv_path.name
    out_subdir = output_dir / dataset_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / "retrieval_subsample_50.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")

    # Also save a summary with person_id list and prevalence
    summary_path = out_subdir / "retrieval_subsample_50_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total person_ids fitting criteria: {total_fitting}\n")
        f.write(f"Selected: {len(selected)} person_ids\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("Prevalence per task:\n")
        for t in tasks:
            task_rows = out_df[out_df[task_col].astype(str) == t]
            n0 = (task_rows[label_col] == 0).sum()
            n1 = (task_rows[label_col] == 1).sum()
            f.write(f"  {t}: 0={n0}, 1={n1}\n")
        f.write("\nSelected person_ids:\n")
        for pid in sorted(selected):
            f.write(f"  {pid}\n")
    print(f"Saved summary to {summary_path}")

    return list(selected), out_path


if __name__ == "__main__":
    CSV_PATH = "/home/rdcunha/vista_project/vista_bench/bigquery_data_2_3/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
    OUTPUT_DIR = "/home/rdcunha/vista_project/vista_bench/v1_2"
    subsample_retrieval_csv(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        target_n=171,
        split="test",
        tasks=RETRIEVAL_TASKS,
    )
