"""
WSI tiling tool for pathology — extracts tiles from whole slide images at a
configurable magnification using Otsu-based tissue detection.

Replicates the tiling approach used by MedGemma for pathology.

Usage:
    uv run python -m src.data_tools.path_tools.tile_wsi /path/to/slide.svs /tmp/tiles/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image

MANIFEST_FILENAME = "tile_manifest.csv"


def _append_to_csv(file_path: Path, data_list: list[dict]) -> None:
    """Append records to CSV, writing header only when the file is new."""
    df = pd.DataFrame(data_list)
    header = not file_path.exists()
    df.to_csv(file_path, mode="a", index=False, header=header)


def tile_wsi(
    slide_input: str | Path,
    output_dir: str | Path,
    *,
    target_mag: float = 10.0,
    native_mag: float | None = None,
    tile_size: int = 256,
    output_format: str = "jpg",
    tissue_threshold: float = 0.50,
    thumbnail_max_dim: int = 2048,
    border_margin_px: int = 5000,
    min_variance: float = 100.0,
) -> Path:
    """Tile one or more WSIs and write a manifest CSV.

    Args:
        slide_input: Single slide path OR path to a CSV with a ``slide_path``
            column.
        output_dir: Root output directory for tiles and manifest.
        target_mag: Desired extraction magnification (default 10x).
        native_mag: Base magnification. Auto-read from slide metadata when
            *None*.
        tile_size: Output tile dimension in pixels.
        output_format: ``"jpg"`` or ``"png"``.
        tissue_threshold: Minimum tissue fraction (0–1) for a tile to be kept.
        thumbnail_max_dim: Max dimension for the thumbnail used in tissue
            detection.
        border_margin_px: Skip tiles whose center falls within this margin (at
            level 0) of the slide edge. Use 0 to disable.
        min_variance: Minimum grayscale variance for kept tiles. Rejects blank
            or uniform regions. Use 0 to disable.

    Returns:
        Path to the manifest CSV.
    """
    slide_input = Path(slide_input)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / MANIFEST_FILENAME

    # --- Resolve slide paths ---
    slide_paths = _resolve_slide_paths(slide_input)

    # --- Load already-processed slides for continuation ---
    processed_slides: set[str] = set()
    if manifest_path.exists():
        try:
            existing = pd.read_csv(manifest_path)
            if "slide_name" in existing.columns:
                processed_slides = set(existing["slide_name"].unique())
        except Exception:
            pass

    for slide_path in slide_paths:
        slide_name = Path(slide_path).stem
        if slide_name in processed_slides:
            print(f"[SKIP] {slide_name} already in manifest — skipping")
            continue

        try:
            _process_slide(
                slide_path=Path(slide_path),
                output_dir=output_dir,
                manifest_path=manifest_path,
                target_mag=target_mag,
                native_mag=native_mag,
                tile_size=tile_size,
                output_format=output_format,
                tissue_threshold=tissue_threshold,
                thumbnail_max_dim=thumbnail_max_dim,
                border_margin_px=border_margin_px,
                min_variance=min_variance,
            )
        except Exception as exc:
            print(f"[WARN] Failed to process {slide_path}: {exc}")
            continue

    return manifest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_slide_paths(slide_input: Path) -> list[str]:
    """Return a list of slide file paths from a single path or CSV."""
    if not slide_input.exists():
        raise FileNotFoundError(f"Input not found: {slide_input}")

    if slide_input.suffix.lower() == ".csv":
        df = pd.read_csv(slide_input)
        if "slide_path" not in df.columns:
            raise ValueError(
                f"CSV {slide_input} must contain a 'slide_path' column. "
                f"Found columns: {list(df.columns)}"
            )
        paths = df["slide_path"].dropna().astype(str).tolist()
        if not paths:
            raise ValueError(f"No slide paths found in {slide_input}")
        return paths

    return [str(slide_input)]


def _get_native_mag(slide: openslide.OpenSlide, user_mag: float | None) -> float:
    """Determine the native objective magnification.

    Tries (in order): user-supplied value, ``openslide.objective-power``
    metadata, then derives magnification from microns-per-pixel (MPP).
    """
    if user_mag is not None:
        return user_mag

    prop = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if prop is not None:
        return float(prop)

    # Fall back to MPP → magnification conversion (10 µm/px ≈ 1x)
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp is not None:
        mag = 10.0 / float(mpp)
        print(f"[INFO] No objective-power metadata; estimated {mag:.1f}x from MPP={mpp}")
        return mag

    raise ValueError(
        "Cannot determine native magnification from slide metadata. "
        "Please provide native_mag explicitly."
    )


def _build_tissue_mask(
    slide: openslide.OpenSlide,
    thumbnail_max_dim: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a binary tissue mask from a slide thumbnail.

    Returns:
        (tissue_mask, thumbnail_size) where tissue_mask is a boolean ndarray
        and thumbnail_size is (width, height) of the thumbnail.
    """
    slide_w, slide_h = slide.dimensions
    scale = thumbnail_max_dim / max(slide_w, slide_h)
    thumb_w = max(1, int(slide_w * scale))
    thumb_h = max(1, int(slide_h * scale))

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    thumbnail_np = np.array(thumbnail.convert("L"))

    otsu_thresh, _ = cv2.threshold(
        thumbnail_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # In H&E, tissue is darker than background
    tissue_mask = thumbnail_np < otsu_thresh

    return tissue_mask, (thumb_w, thumb_h)


def _tissue_fraction_for_tile(
    tissue_mask: np.ndarray,
    thumb_size: tuple[int, int],
    slide_dims: tuple[int, int],
    x0: int,
    y0: int,
    region_size: int,
) -> float:
    """Compute fraction of tissue in a tile's footprint on the thumbnail."""
    slide_w, slide_h = slide_dims
    thumb_w, thumb_h = thumb_size

    sx = thumb_w / slide_w
    sy = thumb_h / slide_h

    tx0 = int(x0 * sx)
    ty0 = int(y0 * sy)
    tx1 = min(int((x0 + region_size) * sx) + 1, thumb_w)
    ty1 = min(int((y0 + region_size) * sy) + 1, thumb_h)

    if tx1 <= tx0 or ty1 <= ty0:
        return 0.0

    patch = tissue_mask[ty0:ty1, tx0:tx1]
    if patch.size == 0:
        return 0.0
    return float(patch.sum()) / patch.size


def _is_within_border(
    x0: int,
    y0: int,
    region_size_l0: int,
    slide_w: int,
    slide_h: int,
    border_margin_px: int,
) -> bool:
    """Return True if tile center is within border margin (should be excluded)."""
    center_x = x0 + region_size_l0 // 2
    center_y = y0 + region_size_l0 // 2
    return (
        center_x < border_margin_px
        or center_x > slide_w - border_margin_px
        or center_y < border_margin_px
        or center_y > slide_h - border_margin_px
    )


def _is_informative_by_variance(
    tile_img: np.ndarray,
    min_variance: float,
) -> bool:
    """Return True if tile has sufficient color variance (reject blank/uniform regions)."""
    gray = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
    return float(gray.var()) >= min_variance


def _process_slide(
    slide_path: Path,
    output_dir: Path,
    manifest_path: Path,
    target_mag: float,
    native_mag: float | None,
    tile_size: int,
    output_format: str,
    tissue_threshold: float,
    thumbnail_max_dim: int,
    border_margin_px: int,
    min_variance: float,
) -> None:
    """Process a single slide: detect tissue, extract tiles, update manifest."""
    slide = openslide.OpenSlide(str(slide_path))
    slide_name = slide_path.stem
    actual_native_mag = _get_native_mag(slide, native_mag)

    downsample = actual_native_mag / target_mag
    best_level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[best_level]

    # Region size at level 0 that corresponds to one tile
    region_size_l0 = int(tile_size * downsample)

    # Build tissue mask
    tissue_mask, thumb_size = _build_tissue_mask(slide, thumbnail_max_dim)

    slide_w, slide_h = slide.dimensions
    slide_dir = output_dir / slide_name
    slide_dir.mkdir(parents=True, exist_ok=True)

    tile_records: list[dict] = []
    tile_index = 0

    print(f"[TILE] {slide_name} | native={actual_native_mag}x target={target_mag}x "
          f"downsample={downsample:.2f} level={best_level} "
          f"region_l0={region_size_l0}px tile={tile_size}px")

    for y0 in range(0, slide_h, region_size_l0):
        for x0 in range(0, slide_w, region_size_l0):
            frac = _tissue_fraction_for_tile(
                tissue_mask, thumb_size, (slide_w, slide_h),
                x0, y0, region_size_l0,
            )
            if frac < tissue_threshold:
                continue

            if border_margin_px > 0 and _is_within_border(
                x0, y0, region_size_l0, slide_w, slide_h, border_margin_px
            ):
                continue

            try:
                region = slide.read_region((x0, y0), best_level, (
                    int(tile_size * downsample / level_downsample),
                    int(tile_size * downsample / level_downsample),
                ))
                tile_img = region.convert("RGB").resize(
                    (tile_size, tile_size), Image.LANCZOS,
                )
                tile_np = np.array(tile_img)

                if min_variance > 0 and not _is_informative_by_variance(
                    tile_np, min_variance
                ):
                    continue

                fname = f"{slide_name}_{x0}_{y0}_{target_mag}x_{tile_index}.{output_format}"
                tile_path = slide_dir / fname
                tile_img.save(str(tile_path))

                tile_records.append({
                    "image_path": str(tile_path.resolve()),
                    "slide_path": str(slide_path),
                    "slide_name": slide_name,
                    "x": x0,
                    "y": y0,
                    "tile_index": tile_index,
                    "target_magnification": target_mag,
                    "native_magnification": actual_native_mag,
                    "tile_size_px": tile_size,
                    "tissue_fraction": round(frac, 4),
                    "output_format": output_format,
                })
                tile_index += 1
            except Exception as exc:
                print(f"[WARN] Tile ({x0},{y0}) failed for {slide_name}: {exc}")
                continue

    if tile_records:
        _append_to_csv(manifest_path, tile_records)
        print(f"[DONE] {slide_name}: {len(tile_records)} tiles saved")
    else:
        print(f"[DONE] {slide_name}: 0 tiles (no tissue above threshold)")

    slide.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Tile whole slide images with Otsu tissue detection.",
    )
    parser.add_argument(
        "slide_input",
        nargs="?",
        default="/data/fries/datasets/vista_bench_ryan/download_path/tile_manifest.csv",
        help="Path to a single WSI or a CSV with a 'slide_path' column.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/data/fries/datasets/vista_bench_ryan/download_path/test_patch",
        help="Root output directory for tiles and manifest CSV.",
    )
    parser.add_argument(
        "--target-mag", type=float, default=10.0,
        help="Target magnification (default: 10.0).",
    )
    parser.add_argument(
        "--native-mag", type=float, default=None,
        help="Native objective magnification (auto-detected if omitted).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Output tile size in pixels (default: 256).",
    )
    parser.add_argument(
        "--output-format", choices=["jpg", "png"], default="jpg",
        help="Tile image format (default: jpg).",
    )
    parser.add_argument(
        "--tissue-threshold", type=float, default=0.50,
        help="Minimum tissue fraction per tile (default: 0.50).",
    )
    parser.add_argument(
        "--thumbnail-max-dim", type=int, default=2048,
        help="Max thumbnail dimension for tissue detection (default: 2048).",
    )
    parser.add_argument(
        "--border-margin-px", type=int, default=10000,
        help="Skip tiles whose center is within this margin (at level 0) of "
             "slide edge (default: 5000). Use 0 to disable.",
    )
    parser.add_argument(
        "--min-variance", type=float, default=150.0,
        help="Minimum grayscale variance per tile (default: 100). Rejects "
             "blank/uniform regions. Use 0 to disable.",
    )
    args = parser.parse_args()

    manifest = tile_wsi(
        slide_input=args.slide_input,
        output_dir=args.output_dir,
        target_mag=args.target_mag,
        native_mag=args.native_mag,
        tile_size=args.tile_size,
        output_format=args.output_format,
        tissue_threshold=args.tissue_threshold,
        thumbnail_max_dim=args.thumbnail_max_dim,
        border_margin_px=args.border_margin_px,
        min_variance=args.min_variance,
    )
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    _cli()
