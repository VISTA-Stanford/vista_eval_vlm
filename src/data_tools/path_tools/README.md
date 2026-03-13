# path_tools — WSI Tiling

Extracts tissue tiles from whole slide images (SVS, TIFF, etc.) using Otsu thresholding for tissue detection. Designed to produce tiles compatible with pathology VLMs like MedGemma.

## Quick Start

All commands run from the repo root (`vista_eval_vlm/`).

```bash
# 1. Install dependencies (one time)
uv sync

# 2. Tile a single slide for MedGemma (10x, 896px patches)
uv run python src/data_tools/path_tools/tile_wsi.py \
    /path/to/slide.svs \
    /path/to/output/ \
    --tile-size 896

# 3. Tile a batch of slides from a CSV
uv run python src/data_tools/path_tools/tile_wsi.py \
    slides.csv \
    /path/to/output/ \
    --tile-size 896
```

## MedGemma Patch Spec

| Parameter | Value |
|---|---|
| Magnification | 10x (default) |
| Patch size | **896 x 896 px** (`--tile-size 896`) |

The default magnification is already 10x. You only need to set `--tile-size 896`.

## Input

The first positional argument is either a **path to a single slide** or a **path to a CSV**.

### Single slide

Any format OpenSlide supports: `.svs`, `.tiff`, `.ndpi`, `.mrxs`, etc.

```bash
uv run python src/data_tools/path_tools/tile_wsi.py /data/slides/case_A.svs ./output/
```

### Batch via CSV

A CSV file with a `slide_path` column. One slide per row:

```csv
slide_path
/data/slides/case_A.svs
/data/slides/case_B.tiff
```

```bash
uv run python src/data_tools/path_tools/tile_wsi.py slides.csv ./output/
```

## Output

The second positional argument is the output directory. The tool creates one subdirectory per slide plus a single `tile_manifest.csv`:

```
output/
    tile_manifest.csv
    case_A/
        case_A_0_0_10.0x_0.jpg
        case_A_1024_0_10.0x_1.jpg
        ...
    case_B/
        case_B_0_0_10.0x_0.jpg
        ...
```

### tile_manifest.csv

One row per tile. This CSV is the input for downstream VQA/inference pipelines.

| Column | Description |
|---|---|
| `image_path` | Absolute path to the tile image |
| `slide_path` | Source WSI path |
| `slide_name` | Slide filename stem |
| `x` | Level-0 X coordinate |
| `y` | Level-0 Y coordinate |
| `tile_index` | Sequential index within slide |
| `target_magnification` | Extraction magnification (e.g. 10.0) |
| `native_magnification` | Slide's native objective power |
| `tile_size_px` | Output tile pixel dimension |
| `tissue_fraction` | Fraction of tile area that is tissue |
| `output_format` | jpg or png |

## How It Works

1. **Tissue detection** — Creates a thumbnail of the slide, converts to grayscale, applies Otsu thresholding. In H&E stains, tissue is darker than background, so pixels below the Otsu threshold are classified as tissue.
2. **Tiling** — Walks a grid over the slide at level-0 coordinates. For each tile position, checks the tissue fraction on the thumbnail mask. Tiles below `--tissue-threshold` (default 50%) are skipped.
3. **Extraction** — Reads the region from the best available OpenSlide level, resizes to the target tile size, and saves.

**Otsu-based tissue detection:** 

1. Specify target magnification, tile size, tissue threshold (part of tile that must be tissue), thumbnail dimension for tissue detection, border margin, min_variance
    - Native magnification (default 10 microns per pixel)
2. Creates tissue mask → downscale slide to thumbnail_max_dim and use cv2 to get automatic threshold
    - H&E (Hematoxylin)**-** tissue is darker than background white
    - Check tissue takes up certain proportion of slide
3. Border check (must be far enough from edge of slide) and check for variance in slide
4. Use OpenSlide to downsample image, pyramid level, region at level 0
    - Level 0 (highest resolution) → convert to this pixel coordinate by scaling magnification

## Native Magnification Detection

The tool resolves native magnification automatically in this order:

1. `--native-mag` flag (if you provide it explicitly)
2. `openslide.objective-power` slide metadata (SVS files typically have this)
3. Derived from `openslide.mpp-x` (microns-per-pixel): `mag = 10 / mpp`

If none are available, the tool errors and asks you to pass `--native-mag`.

## Continuation / Fault Tolerance

Re-running the same command skips slides already in the manifest. This means:
- Safe to resume after interruption
- Safe to add new slides to a CSV and re-run — only new slides get tiled
- If a single slide fails (e.g. corrupt file), it is skipped and the rest continue

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--target-mag` | 10.0 | Target magnification |
| `--native-mag` | auto | Override native objective magnification |
| `--tile-size` | 256 | Output tile size in pixels (**use 896 for MedGemma**) |
| `--output-format` | jpg | `jpg` or `png` |
| `--tissue-threshold` | 0.50 | Min tissue fraction to keep a tile (0.0–1.0) |
| `--thumbnail-max-dim` | 2048 | Thumbnail size for tissue detection |

## Prerequisites

Python packages (managed by `uv`, already in `pyproject.toml`):

- `openslide-python` — Python bindings for OpenSlide
- `openslide-bin` — Bundles the OpenSlide C library (no system install needed)
- `opencv-python-headless` — Otsu thresholding

```bash
uv sync
```
