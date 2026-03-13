"""
Build a CSV of all .tiff slide paths under download_path for use with tile_wsi.py.

Scans /data/fries/datasets/vista_bench_ryan/download_path subfolders for .tiff/.tif
files and writes a CSV with a 'slide_path' column (the format tile_wsi.py expects).
Output: /data/fries/datasets/vista_bench_ryan/download_path/tile_manifest.csv
"""

from pathlib import Path

import pandas as pd

DOWNLOAD_PATH = Path("/data/fries/datasets/vista_bench_ryan/download_path")
OUTPUT_CSV = DOWNLOAD_PATH / "tile_manifest.csv"
TIFF_SUFFIXES = (".tiff", ".tif")


def collect_tiff_paths(root: Path) -> list[Path]:
    """Return sorted list of .tiff/.tif file paths under root (in subfolders)."""
    paths = []
    if not root.is_dir():
        return paths
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES:
            paths.append(p.resolve())
    return sorted(paths)


def main(
    download_path: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> Path:
    """
    Find all .tiff/.tif under download_path and write a slide_path CSV for tile_wsi.

    Args:
        download_path: Root to scan (default: DOWNLOAD_PATH).
        output_csv: Where to write the CSV (default: download_path/tile_manifest.csv).

    Returns:
        Path to the written CSV.
    """
    root = Path(download_path) if download_path is not None else DOWNLOAD_PATH
    out = Path(output_csv) if output_csv is not None else OUTPUT_CSV

    paths = collect_tiff_paths(root)
    if not paths:
        print(f"No .tiff/.tif files found under {root}")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["slide_path"]).to_csv(out, index=False)
        print(f"Wrote empty CSV: {out}")
        return out

    df = pd.DataFrame({"slide_path": [str(p) for p in paths]})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Found {len(paths)} .tiff/.tif files under {root}")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a slide_path CSV from .tiff files under download_path for tile_wsi.py.",
    )
    parser.add_argument(
        "--download-path",
        default=str(DOWNLOAD_PATH),
        help="Root directory to scan for .tiff/.tif files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output CSV path (default: {{download_path}}/tile_manifest.csv)",
    )
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else Path(args.download_path) / "tile_manifest.csv"
    main(download_path=args.download_path, output_csv=out_path)
