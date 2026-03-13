"""
Download path images from GCS for v1_3 path_subsampled CSVs.

Scans /data/fries/users/rdcunha/vista_bench_cached/vista_bench/v1_3 for subfolders,
finds CSVs whose name contains 'path_subsampled', and downloads the 'path_image_path'
column (gs:// URLs) to /data/fries/datasets/vista_bench_ryan/download_path.

- Dry run: report how many paths exist in bucket vs missing, and how many already local.
- Live: download only if not already present locally; skip if not in bucket.
"""

import csv
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage

csv.field_size_limit(sys.maxsize)

V1_3_BASE = Path("/data/fries/users/rdcunha/vista_bench_cached/vista_bench/v1_3")
DOWNLOAD_BASE = Path("/data/fries/datasets/vista_bench_ryan/download_path")
PATH_SUBSAMPLED_SUBSTRING = "path_subsampled"
PATH_IMAGE_PATH_COL = "path_image_path"


def _parse_gs_url(url: str) -> tuple[str, str] | None:
    """Parse gs://bucket/path/to/object into (bucket_name, blob_path). Returns None if invalid."""
    s = (url or "").strip()
    if not s.startswith("gs://"):
        return None
    s = s[5:]
    if "/" not in s:
        return None
    bucket_name, _, blob_path = s.partition("/")
    if not bucket_name or not blob_path:
        return None
    return bucket_name, blob_path


def _find_path_subsampled_csvs(v1_3_base: Path) -> list[Path]:
    """Return list of CSV paths under v1_3_base whose filename contains PATH_SUBSAMPLED_SUBSTRING."""
    if not v1_3_base.is_dir():
        return []
    out = []
    for subfolder in v1_3_base.iterdir():
        if not subfolder.is_dir():
            continue
        for p in subfolder.iterdir():
            if p.is_file() and p.suffix.lower() == ".csv" and PATH_SUBSAMPLED_SUBSTRING in p.name:
                out.append(p)
    return sorted(out)


def _collect_gs_paths(csv_paths: list[Path], path_col_name: str) -> list[tuple[str, str, str]]:
    """
    Read path_image_path from each CSV. Return list of (gs_url, task_folder, source_csv_name).
    Only includes non-empty gs:// URLs.
    """
    rows = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", on_bad_lines="warn")
        except Exception as e:
            print(f"  [WARN] Failed to read {csv_path}: {e}")
            continue

        path_col = next(
            (c for c in df.columns if c.replace(" ", "").lower() == path_col_name.lower()),
            None,
        )
        if path_col is None:
            path_col = next(
                (c for c in df.columns if "path_image" in c.lower() or "image_path" in c.lower()),
                None,
            )
        if path_col is None:
            print(f"  [SKIP] {csv_path.name}: no '{path_col_name}' column. cols={list(df.columns)}")
            continue

        for _, row in df.iterrows():
            val = row.get(path_col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            gs_url = str(val).strip()
            if not gs_url.startswith("gs://"):
                continue
            task_folder = csv_path.parent.name
            rows.append((gs_url, task_folder, csv_path.name))
    return rows


def _gs_to_local_path(gs_url: str, download_base: Path) -> Path | None:
    """
    Map gs://bucket/path/to/object to local path: download_base / path/to/object.
    Returns None if gs_url is not a valid gs:// URL.
    """
    parsed = _parse_gs_url(gs_url)
    if parsed is None:
        return None
    _, blob_path = parsed
    # Use forward slashes for blob_path; Path handles them
    local = download_base / blob_path.replace("\\", "/")
    return local


def run(
    v1_3_base: str | Path | None = None,
    download_base: str | Path | None = None,
    dry_run: bool = True,
) -> None:
    """
    Scan v1_3 subfolders for path_subsampled CSVs, collect path_image_path (gs://),
    then report (dry_run) or download to download_base.

    Args:
        v1_3_base: Base directory containing v1_3 subfolders (default: V1_3_BASE).
        download_base: Local directory for downloads (default: DOWNLOAD_BASE).
        dry_run: If True, only report counts (present in bucket, missing, already local).
                 If False, download files that are in bucket and not yet local.
    """
    v1_3_base = Path(v1_3_base) if v1_3_base is not None else V1_3_BASE
    download_base = Path(download_base) if download_base is not None else DOWNLOAD_BASE

    if not v1_3_base.is_dir():
        print(f"Error: v1_3 base not found: {v1_3_base}")
        return

    csv_paths = _find_path_subsampled_csvs(v1_3_base)
    if not csv_paths:
        print(f"No CSVs containing '{PATH_SUBSAMPLED_SUBSTRING}' found under {v1_3_base}")
        return

    print(f"Found {len(csv_paths)} path_subsampled CSVs under {v1_3_base}")
    all_rows = _collect_gs_paths(csv_paths, PATH_IMAGE_PATH_COL)
    # Deduplicate by gs_url to avoid repeated bucket checks
    seen = set()
    unique_rows = []
    for gs_url, task_folder, csv_name in all_rows:
        if gs_url not in seen:
            seen.add(gs_url)
            unique_rows.append((gs_url, task_folder, csv_name))

    if not unique_rows:
        print("No gs:// path_image_path values found in those CSVs.")
        return

    print(f"Collected {len(unique_rows)} unique gs:// paths from path_image_path")
    client = storage.Client()
    stats = {"present_in_bucket": 0, "missing_in_bucket": 0, "already_local": 0, "downloaded": 0}
    would_download_printed = 0
    max_would_download_print = 30

    mode = " [DRY RUN] " if dry_run else " [LIVE DOWNLOAD] "
    print(f"{mode} Checking bucket and local paths...")
    if not dry_run:
        print(f"Download directory: {download_base}")

    for gs_url, task_folder, csv_name in unique_rows:
        parsed = _parse_gs_url(gs_url)
        if parsed is None:
            print(f"  [SKIP] Invalid gs URL: {gs_url[:80]}...")
            continue
        bucket_name, blob_path = parsed
        local_path = _gs_to_local_path(gs_url, download_base)
        if local_path is None:
            continue

        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            exists_in_bucket = blob.exists()
        except Exception as e:
            print(f"  [ERROR] Bucket check failed for {gs_url[:60]}...: {e}")
            continue

        if not exists_in_bucket:
            stats["missing_in_bucket"] += 1
            if stats["missing_in_bucket"] <= 20:
                print(f"  [MISSING] {blob_path}")
            continue

        stats["present_in_bucket"] += 1
        if local_path.exists():
            stats["already_local"] += 1
            if not dry_run:
                continue
            if stats["already_local"] <= 10:
                print(f"  [ALREADY LOCAL] {local_path.relative_to(download_base)}")
            continue

        if dry_run:
            if would_download_printed < max_would_download_print:
                print(f"  [WOULD DOWNLOAD] {blob_path} -> {local_path.relative_to(download_base)}")
                would_download_printed += 1
            continue

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            stats["downloaded"] += 1
            if stats["downloaded"] <= 50:
                print(f"  [DOWNLOADED] {local_path.relative_to(download_base)}")
        except Exception as e:
            print(f"  [ERROR] Download failed {blob_path}: {e}")

    would_download = stats["present_in_bucket"] - stats["already_local"]
    print("\n" + "=" * 50)
    print(" SUMMARY ")
    print("=" * 50)
    print(f"Unique gs:// paths:        {len(unique_rows)}")
    print(f"Present in bucket:         {stats['present_in_bucket']}")
    print(f"Missing in bucket:         {stats['missing_in_bucket']}")
    print(f"Already on disk:           {stats['already_local']}")
    if dry_run:
        print(f"Would download (not local): {would_download}")
    else:
        print(f"Downloaded this run:       {stats['downloaded']}")
    if len(unique_rows) > 0:
        pct = 100.0 * stats["present_in_bucket"] / len(unique_rows)
        print(f"Bucket availability:       {pct:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download path_image_path (gs://) from v1_3 path_subsampled CSVs."
    )
    parser.add_argument(
        "--v1-3-base",
        default=str(V1_3_BASE),
        help="Base directory for v1_3 (subfolders with path_subsampled CSVs)",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DOWNLOAD_BASE),
        help="Local directory to download files into",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only report counts; do not download (default)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files (disables dry-run)",
    )
    args = parser.parse_args()

    run(
        v1_3_base=args.v1_3_base,
        download_base=args.download_dir,
        dry_run=not args.download,
    )
