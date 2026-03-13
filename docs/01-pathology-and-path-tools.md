# Pathology and path tools

This document describes how to obtain and use pathology (whole slide image, WSI) data for Vista Eval VLM experiments: `path`, `path_image_and_report`, and `path_full`.

**NOTE**: Pathology slides are very big (and in a hidden bucket). Be careful when downloading a lot (take up space and high cost from GS)

## Prerequisites

- **GCS access:** Read access to the bucket and paths referenced in `path_image_path` (e.g. `gs://` URLs in the path cohort CSVs).
- **OpenSlide:** For tiling, install OpenSlide (Python bindings and library). The repo uses `openslide-python` and `opencv-python-headless`; see [path_tools/README.md](../src/data_tools/path_tools/README.md).

## Data flow overview

1. **Source:** Task data with pathology comes from BigQuery-backed tables (or local cache) under `bigquery_v1_3`. Task definitions live in `valid_tasks_v1_3.json`. Each row has a `path_image_path` (e.g. `gs://bucket/path/to/slide.tiff`).
2. **Subsample:** Build per-task path CSVs (test split, valid label, non-empty `path_image_path`).
3. **Download:** Download WSI files from GCS to a local base directory (e.g. `path_tile_base`).
4. **Tiling:** Tile each WSI into patches and write them under `path_tile_base/test_patch/{folder_name}/`.
5. **Config:** Set `paths.path_tile_base` in [configs/all_tasks.yaml](../configs/all_tasks.yaml) to the root that contains `test_patch/`.
6. **Inference:** `run_bq.py` loads path task data, attaches up to **100** tile paths per slide, and `vqa_dataset.py` loads those tiles for the model.

## Step-by-step

### 1. Subsample path cohort

Script: [src/data_tools/csv_helper/subsample_v1_3_path.py](../src/data_tools/csv_helper/subsample_v1_3_path.py).

- Reads BigQuery-derived CSVs under `bigquery_v1_3` (file names match `task_source_csv` from `valid_tasks_v1_3.json`).
- Filters: `split == 'test'`, `label != -1`, and `path_image_path` non-null and non-empty.
- Writes `v1_3/{task_source_csv}/{task_name}_path_subsampled.csv` (all columns from source).

Run from repo root after BigQuery data is available (or synced) under `base_dir/bigquery_v1_3/`.

### 2. Download pathology images from GCS

Script: [src/data_tools/path_tools/download_path_subsampled.py](../src/data_tools/path_tools/download_path_subsampled.py).

- Scans `v1_3` for CSVs whose name contains `path_subsampled`.
- Reads the `path_image_path` column (expects `gs://` URLs).
- Downloads each object to the local base (e.g. `path_tile_base`; default in script is `/data/fries/datasets/vista_bench_ryan/download_path`).
- Supports dry-run (report existing vs missing) and skips files already present locally.

### 3. Tiling

1. **Build slide list:** [src/data_tools/path_tools/csv_tile_wsi.py](../src/data_tools/path_tools/csv_tile_wsi.py) scans the download directory for `.tiff`/`.tif` and writes a CSV with a `slide_path` column for [tile_wsi.py](../src/data_tools/path_tools/tile_wsi.py).

2. **Tile WSIs:** [src/data_tools/path_tools/tile_wsi.py](../src/data_tools/path_tools/tile_wsi.py) takes that CSV (or a single slide path) and an output directory. Use `--output-dir` equal to `path_tile_base/test_patch` (or the default in the script). It creates one subdirectory per slide, named by the slide filename stem (e.g. `slide_001` for `slide_001.tiff`), and writes `.jpg` tiles there.

For MedGemma-compatible tiles use `--tile-size 896` and 10x magnification and 125 patches (see [path_tools/README.md](../src/data_tools/path_tools/README.md) for full options and MedGemma patch spec).

### 4. Config

In [configs/all_tasks.yaml](../configs/all_tasks.yaml):

```yaml
paths:
  path_tile_base: "/data/fries/datasets/vista_bench_ryan/download_path"  # must contain test_patch/
```

Tiles must live at `{path_tile_base}/test_patch/{folder_name}/*.jpg`, where `folder_name` is derived from `path_image_path` (see below).

## How run_bq and vqa_dataset use path data

- **run_bq.py** (`_load_path_task_data` and `_load_path_full_task_data` in [src/vista_run/run_bq.py](../src/vista_run/run_bq.py)):
  - Loads `{task_name}_path_subsampled.csv` from `base_dir/v1_3/{source_csv}/`.
  - For each row, derives **folder name** from `path_image_path`: take the last segment after `\` or `/`, then remove `.tiff`/`.tif` (filename stem).
  - Looks for tiles under `path_tile_base/test_patch/{folder_name}/` (`.jpg`/`.jpeg`).
  - Attaches up to **100** tile paths per slide (all if ≤100, otherwise random sample with seed 42) as `path_tile_paths` on the dataframe.
  - Rows with no tiles found are dropped. For `path_full`, the same path data is merged with the patient timeline CSV so each row has tiles + path_note_text + truncated timeline.

- **vqa_dataset.py** ([PromptDataset](../src/vqa_dataset.py)):
  - For experiments `path`, `path_image_and_report`, and `path_full`, each item’s `path_tile_paths` (list) is loaded as PIL images, resized/padded to the model’s target size, and passed as the `image` field (list of images).

## Troubleshooting

- **Missing tiles:** Ensure tiles were written under `path_tile_base/test_patch/` with one folder per slide and that the folder name exactly matches the stem of the file referenced in `path_image_path` (e.g. `path_image_path` ending in `.../slide_001.tiff` → folder `slide_001`).
- **Folder name mismatch:** `run_bq` derives the folder from the last path segment (after `\` or `/`) with `.tiff`/`.tif` stripped. If your GCS paths use a different pattern, the derived name must still match the directory created by `tile_wsi` (which uses the slide filename stem).
- **No rows after load:** If every row gets empty `path_tile_paths`, the orchestrator drops all rows and reports "No rows with tiles under test_patch/...". Re-check `path_tile_base`, `test_patch`, and folder names.
