import os
import pandas as pd
import yaml
import csv
import sys
from pathlib import Path
from google.cloud import storage
from data_tools.utils.config_utils import load_tasks_from_config

# Increase the limit to handle very large clinical text fields
csv.field_size_limit(sys.maxsize)

BQ_DATA_DIR = "bigquery_data_2_3"
CT_COVERAGE_PERSON_IDS = "ct_coverage_all_vb/person_ids.csv"
DEFAULT_DOWNLOAD_DIR_CT_COVERAGE = "/data/fries/datasets/vista_bench_ryan/downloaded_ct_scans"
RETRIEVAL_SUBSAMPLE_50_CSV = "v1_2/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/retrieval_subsample_50.csv"


def _resolve_bq_file(bq_dir: Path, table_name: str) -> Path | None:
    """Resolve BQ file path; try table_name and table_name_v1_1 if not found."""
    candidates = [bq_dir / table_name, bq_dir / f"{table_name}_v1_1"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_nifti_paths_from_retrieval_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load person_id and nifti_path directly from a retrieval-style CSV that already
    contains both columns. Returns DataFrame with columns: path, person_id, task.
    Deduplicates by path (unique person_ids may share the same nifti_path across rows).
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
    person_id_col = next((c for c in df.columns if "person_id" in c.lower()), None)
    nifti_path_col = next((c for c in df.columns if c.lower() == "nifti_path"), None)
    if not nifti_path_col:
        nifti_path_col = next((c for c in df.columns if c.lower() == "local_path"), None)
    if not person_id_col or not nifti_path_col:
        raise ValueError(
            f"CSV must have 'person_id' and 'nifti_path' (or 'local_path') columns: {csv_path}"
        )
    df = df[[person_id_col, nifti_path_col]].rename(
        columns={nifti_path_col: "path", person_id_col: "person_id"}
    )
    df = df.dropna(subset=["path"])
    df["path"] = df["path"].astype(str).str.strip()
    df = df[df["path"] != ""]
    df["task"] = csv_path.parent.name  # e.g. progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr
    return df.drop_duplicates(subset=["path"])


def _load_nifti_paths_from_person_ids(
    base_path: Path,
    person_ids_csv: Path,
) -> pd.DataFrame:
    """
    Load person_ids from CSV, then look up nifti_path from bigquery_data_2_3.
    Returns DataFrame with columns: path, person_id, task (task from first matching row).
    """
    person_ids_df = pd.read_csv(person_ids_csv)
    person_id_col = next((c for c in person_ids_df.columns if "person_id" in c.lower()), None)
    if not person_id_col:
        raise ValueError(f"person_ids.csv must have 'person_id' column: {person_ids_csv}")
    person_ids = set(person_ids_df[person_id_col].dropna().astype(str).unique())

    bq_dir = base_path / BQ_DATA_DIR
    if not bq_dir.exists():
        raise FileNotFoundError(f"BQ data dir not found: {bq_dir}")

    # Discover BQ files (unique table names)
    seen_tables = set()
    bq_files = []
    for p in bq_dir.iterdir():
        if p.is_file() and not p.name.startswith("."):
            # Normalize: progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1 -> progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr
            base_name = p.name.replace("_v1_1", "")
            if base_name not in seen_tables:
                seen_tables.add(base_name)
                bq_files.append(p)

    all_records = []
    for bq_file in bq_files:
        try:
            df = pd.read_csv(bq_file, sep=None, engine="python", on_bad_lines="warn")
        except Exception as e:
            print(f"  [WARN] Failed to read {bq_file.name}: {e}")
            continue

        if "person_id" not in df.columns or "nifti_path" not in df.columns:
            continue
        if "split" not in df.columns:
            continue

        df_filt = df[
            (df["person_id"].astype(str).isin(person_ids))
            & (df["split"].astype(str).str.strip().str.lower() == "test")
            & df["nifti_path"].notna()
            & (df["nifti_path"].astype(str).str.strip() != "")
        ][["person_id", "nifti_path", "task"]].copy()
        df_filt = df_filt.rename(columns={"nifti_path": "path"})

        if not df_filt.empty:
            all_records.append(df_filt)

    if not all_records:
        raise ValueError("No nifti_path found for any person_id in bigquery_data_2_3")

    return pd.concat(all_records, ignore_index=True).drop_duplicates(subset=["path"])


def download_ct_scans(base_path='/home/rdcunha/vista_project/vista_bench',
                      bucket_name='su-vista-uscentral1',
                      prefix='chaudhari_lab/ct_data/ct_scans/vista/nov25',
                      dry_run=True,
                      config_path=None,
                      download_base_dir='/home/rdcunha/vista_project/downloaded_ct_scans',
                      file_suffix='_subsampled',
                      person_ids_csv=None,
                      retrieval_subsample_csv=None):
    """
    Checks or downloads NIfTI files from GCP, reporting Task and Person ID.
    Finds all CSV files with the specified suffix and processes them.
    dry_run=True: Only counts present/missing files without downloading.
    config_path: Optional path to YAML config file. If provided, only processes tasks listed in config.
    download_base_dir: Base directory where files will be downloaded, maintaining bucket structure.
    file_suffix: Suffix to look for in CSV filenames (e.g., '_subsampled' or '_all_ct'). Default: '_subsampled'.
    person_ids_csv: If provided, read person_ids from this file and look up nifti_path from bigquery_data_2_3.
                    Uses download_base_dir=/data/fries/datasets/vista_bench_ryan/downloaded_ct_scans.
    retrieval_subsample_csv: If provided, read person_id and nifti_path directly from this CSV (e.g.
                    retrieval_subsample_50.csv). Skips bigquery lookup. Downloads only missing files.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"Error: Base directory {base_path} not found.")
        return

    # Mode: retrieval_subsample_csv (CSV with person_id + nifti_path, e.g. retrieval_subsample_50.csv)
    if retrieval_subsample_csv is not None:
        retrieval_path = Path(retrieval_subsample_csv)
        if not retrieval_path.is_absolute():
            retrieval_path = base_dir / retrieval_path
        if not retrieval_path.exists():
            print(f"Error: Retrieval subsample CSV not found: {retrieval_path}")
            return
        download_base_dir = DEFAULT_DOWNLOAD_DIR_CT_COVERAGE
        print(f"Mode: Retrieval subsample (person_id + nifti_path from {retrieval_path.name})")
        print(f"Download dir: {download_base_dir}")
        try:
            unique_df = _load_nifti_paths_from_retrieval_csv(retrieval_path)
        except Exception as e:
            print(f"Error loading retrieval CSV: {e}")
            return
        print(f"Loaded {len(unique_df)} unique nifti paths for {unique_df['person_id'].nunique()} person_ids")
    # Mode: person_ids_csv (CT coverage all VB)
    elif person_ids_csv is not None:
        person_ids_path = Path(person_ids_csv)
        if not person_ids_path.is_absolute():
            person_ids_path = base_dir / person_ids_path
        if not person_ids_path.exists():
            print(f"Error: person_ids CSV not found: {person_ids_path}")
            return
        download_base_dir = DEFAULT_DOWNLOAD_DIR_CT_COVERAGE
        print(f"Mode: CT coverage (person_ids from {person_ids_path.name})")
        print(f"Download dir: {download_base_dir}")
        try:
            unique_df = _load_nifti_paths_from_person_ids(base_dir, person_ids_path)
        except Exception as e:
            print(f"Error loading nifti paths: {e}")
            return
        print(f"Loaded {len(unique_df)} unique nifti paths for {unique_df['person_id'].nunique()} person_ids")
        # Fall through to download loop (skip CSV discovery)
    else:
        unique_df = None

    if unique_df is None:
        # Load tasks from config if provided
        valid_tasks = None
        if config_path:
            valid_tasks = load_tasks_from_config(config_path)
            if valid_tasks:
                print(f"Loaded {len(valid_tasks)} tasks from config: {sorted(valid_tasks)}")
            else:
                print(f"Warning: No tasks found in config or error loading config. Processing all tasks.")
        
        # Find all CSV files with the specified suffix
        all_csv_files = [p for p in base_dir.rglob("*.csv") if p.stem.endswith(file_suffix)]
        
        if not all_csv_files:
            print(f"No CSV files with suffix '{file_suffix}' found in {base_path}")
            return
        
        # Filter by task names if config provided
        csv_files = []
        if valid_tasks:
            for csv_path in all_csv_files:
                # Extract task name from filename (remove file_suffix suffix)
                task_name = csv_path.stem.replace(file_suffix, '')
                if task_name in valid_tasks:
                    csv_files.append(csv_path)
            print(f"Filtered to {len(csv_files)} CSVs with suffix '{file_suffix}' matching tasks from config (out of {len(all_csv_files)} total).")
        else:
            csv_files = all_csv_files
            print(f"Found {len(csv_files)} CSV files with suffix '{file_suffix}'.")
        
        if not csv_files:
            print(f"No matching CSV files with suffix '{file_suffix}' found.")
            return
        
        # Read all matching CSV files and combine
        all_records = []
        for csv_file in csv_files:
            try:
                # Extract task name from filename
                task_name = csv_file.stem.replace(file_suffix, '')
                
                # Read CSV file
                df = pd.read_csv(csv_file, sep=None, engine='python', on_bad_lines='warn')
                
                # Check for required columns - prefer nifti_path, fallback to local_path
                nifti_path_col = next((c for c in df.columns if c.lower() == 'nifti_path'), None)
                local_path_col = next((c for c in df.columns if c.lower() == 'local_path'), None)
                person_id_col = next((c for c in df.columns if c.lower() in ['person_id', 'patient_id']), None)
                
                # Use nifti_path if available, otherwise fallback to local_path
                path_col = nifti_path_col if nifti_path_col else local_path_col
                
                if not path_col:
                    print(f"  [SKIP] {csv_file.name}: Missing 'nifti_path' or 'local_path' column")
                    continue
                
                if not person_id_col:
                    print(f"  [SKIP] {csv_file.name}: Missing 'person_id' or 'patient_id' column")
                    continue
                
                # Select and rename columns
                df_selected = df[[path_col, person_id_col]].copy()
                df_selected = df_selected.rename(columns={path_col: 'path', person_id_col: 'person_id'})
                df_selected['task'] = task_name
                
                # Filter valid paths only
                df_selected = df_selected.dropna(subset=['path'])
                
                if not df_selected.empty:
                    all_records.append(df_selected)
                    print(f"  + Loaded {len(df_selected)} rows from {csv_file.name}")
            
            except Exception as e:
                print(f"  [ERROR] Failed to process {csv_file.name}: {e}")
        
        if not all_records:
            print(f"No valid records found in CSV files with suffix '{file_suffix}'.")
            return
        
        # Combine all dataframes
        df = pd.concat(all_records, ignore_index=True)
        
        # Deduplicate based on path to avoid checking the same file twice
        # We keep the first occurrence of metadata for reporting
        unique_df = df.drop_duplicates(subset=['path'])
    
    stats = {"present": 0, "missing": 0, "already_on_vm": 0}
    
    # Setup download base directory
    download_base = Path(download_base_dir)
    
    print(f"{' [DRY RUN MODE] ' if dry_run else ' [LIVE DOWNLOAD MODE] '}")
    print(f"Processing {len(unique_df)} unique paths...")
    if not dry_run:
        print(f"Files will be downloaded to: {download_base_dir}")

    for _, row in unique_df.iterrows():
        path_str = row['path']
        task_name = row['task']
        person_id = row['person_id']
        
        try:
            # 1. Path Normalization - Handle different nifti_path formats
            # Remove /mnt/ prefix if present
            if path_str.startswith('/mnt/'):
                path_str = path_str[5:]
            
            # Remove bucket name prefix if present
            if path_str.startswith(f'{bucket_name}/'):
                path_str = path_str[len(bucket_name) + 1:]
            
            # Check if path_str is already a bucket-relative path (contains prefix)
            if path_str.startswith(prefix):
                # Already a full bucket path, use it directly
                blob_path = path_str
                # Extract filename for local download path
                bucket_filename = path_str.split('/')[-1]
            else:
                # Extract filename from path (handles both local paths and just filenames)
                parts = path_str.split('/')
                filename = parts[-1]
                
                # If filename doesn't have .nii.gz extension, construct it from parts
                if not filename.endswith('.nii.gz'):
                    # Assume format: {study_uid}__{series_uid}.nii.gz or construct from path
                    if len(parts) >= 2:
                        filename_no_ext = parts[-1].replace('.zip', '')
                        bucket_filename = f"{parts[-2]}__{filename_no_ext}.nii.gz"
                    else:
                        # Just filename, assume it's already correct or needs .nii.gz
                        bucket_filename = filename if filename.endswith('.nii.gz') else f"{filename}.nii.gz"
                else:
                    bucket_filename = filename
                
                blob_path = f"{prefix}/{bucket_filename}"
            
            # 2. Local Path Setup - Use bucket structure
            # Construct path as: download_base_dir/prefix/bucket_filename
            local_download_path = download_base / prefix / bucket_filename

            # 3. Check Bucket
            blob = bucket.blob(blob_path)
            exists_in_bucket = blob.exists()

            if exists_in_bucket:
                stats["present"] += 1
                status_msg = f"[PRESENT] (Task: {task_name}, ID: {person_id})"
                
                if local_download_path.exists():
                    stats["already_on_vm"] += 1
                    # Optional: Comment out to reduce noise if you have many files
                    # print(f"  {status_msg} - Already on VM")
                
                if not dry_run and not local_download_path.exists():
                    local_download_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  [DOWNLOADING] {blob_path} -> {local_download_path}")
                    blob.download_to_filename(str(local_download_path))
                elif dry_run:
                     print(f"  {status_msg} - {blob_path}")

            else:
                stats["missing"] += 1
                print(f"  [MISSING] {blob_path} (Task: {task_name}, ID: {person_id})")

        except Exception as e:
            print(f"  [ERROR] Failed to process {path_str}: {e}")

    # 4. Final Summary
    print("\n" + "="*40)
    print(" SCAN SUMMARY ")
    print("="*40)
    print(f"Total Paths in CSV:    {len(unique_df)}")
    print(f"Present in Bucket:     {stats['present']}")
    print(f"Missing from Bucket:   {stats['missing']}")
    print(f"Already on VM:         {stats['already_on_vm']}")
    
    if len(unique_df) > 0:
        availability = (stats['present'] / len(unique_df)) * 100
        print(f"Bucket Availability:   {availability:.2f}%")
    print("="*40)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CT scans from GCP bucket.")
    parser.add_argument(
        "--person-ids-csv",
        default=None,
        help=f"Use person_ids from this file (e.g. {CT_COVERAGE_PERSON_IDS}), look up nifti_path from bigquery_data_2_3, download to {DEFAULT_DOWNLOAD_DIR_CT_COVERAGE}.",
    )
    parser.add_argument(
        "--retrieval-subsample-csv",
        default=RETRIEVAL_SUBSAMPLE_50_CSV,
        help=f"Use person_id and nifti_path from this CSV (default: {RETRIEVAL_SUBSAMPLE_50_CSV}). Downloads CT scans for unique person_ids. Ignored if --legacy or --person-ids-csv is set.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy mode: discover _subsampled CSVs from config tasks instead of person_ids.csv",
    )
    parser.add_argument(
        "--base-path",
        default="/home/rdcunha/vista_project/vista_bench",
        help="Base path for vista_bench",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only report, do not download (default: True)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files (disables dry-run)",
    )
    args = parser.parse_args()

    if args.legacy:
        person_ids_csv = None
        retrieval_subsample_csv = None
    elif args.person_ids_csv:
        person_ids_csv = args.person_ids_csv
        if not Path(person_ids_csv).is_absolute():
            person_ids_csv = str(Path(args.base_path) / person_ids_csv)
        retrieval_subsample_csv = None
    else:
        person_ids_csv = None
        retrieval_subsample_csv = args.retrieval_subsample_csv
        if retrieval_subsample_csv and not Path(retrieval_subsample_csv).is_absolute():
            retrieval_subsample_csv = str(Path(args.base_path) / retrieval_subsample_csv)

    download_ct_scans(
        base_path=args.base_path,
        dry_run=not args.download,
        config_path="/home/rdcunha/vista_project/vista_eval_vlm/configs/all_tasks.yaml",
        file_suffix="_subsampled",
        person_ids_csv=person_ids_csv,
        retrieval_subsample_csv=retrieval_subsample_csv,
    )