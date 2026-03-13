# CT scans

This document describes how CT (NIfTI) data is obtained, where it lives, and how it is used in the Vista Eval VLM pipeline.

## Source

NIfTI paths come from BigQuery: the task/cohort tables include a `nifti_path` column. The canonical GCS location uses:

- **Bucket:** `su-vista-uscentral1`
- **Prefix:** `chaudhari_lab/ct_data/ct_scans/vista/nov25`

Defined in [src/vqa_dataset.py](../src/vqa_dataset.py) as `DEFAULT_NIFTI_BUCKET_PREFIX`. Paths may be stored as full bucket paths or relative; the dataset normalizes them to blob path and filename for both GCP download and local lookup.

## Downloading CT data

Script: [src/data_tools/ct_info/download_subsampled_ct.py](../src/data_tools/ct_info/download_subsampled_ct.py).

Two modes:

1. **Person IDs + BQ lookup:** Provide a CSV of `person_id`. The script looks up `nifti_path` from the cached BigQuery data (e.g. under `base_dir/bigquery_data_2_3/`), then downloads only missing NIfTI files to a local directory.
2. **Retrieval-style CSV:** Provide a CSV that already has `person_id` and `nifti_path` (e.g. `retrieval_subsample_50.csv` or similar). The script reads paths from the CSV and downloads only missing files.

Output is written to a local directory (e.g. under `downloaded_ct_scans/...`). Set `paths.ct_dir` in [configs/all_tasks.yaml](../configs/all_tasks.yaml) to this directory so inference can load from disk instead of GCP.

## Config

In [configs/all_tasks.yaml](../configs/all_tasks.yaml):

```yaml
paths:
  ct_dir: "/data/fries/datasets/vista_bench_ryan/downloaded_ct_scans/chaudhari_lab/ct_data/ct_scans/vista/nov25"
```

- If `ct_dir` is set and the NIfTI **filename** (derived from `nifti_path`) exists under `ct_dir`, [vqa_dataset.py](../src/vqa_dataset.py) loads the volume from disk.
- Otherwise it uses the GCP Storage client to download the blob (same bucket/prefix logic) and load from a temporary file.

## Usage in the dataset

For experiments that use CT images, [PromptDataset](../src/vqa_dataset.py) in `vqa_dataset.py`:

1. Reads `nifti_path` from each row.
2. Resolves blob path and filename via `_nifti_path_to_blob_and_filename` (handles `/mnt/` prefixes, bucket prefix, and `.nii.gz` naming).
3. Loads the NIfTI from `ct_dir` if the file exists locally, else from GCP Storage.
4. Samples axial slices from the volume. The number and spacing depend on the experiment:
   - **50 slices** (uniformly spaced): `no_timeline`, `all_vb_image_only`, `axial_all_image`, `retrieved_timeline_with_image`, `retrieved_timeline_per_iteration_summarization_with_image`, `no_report`.
   - **30 slices:** `axial_all_image` and the retrieval+image variants use a 0.1-spacing scheme over depth.
5. Preprocessing:
   - **Gemma-style models:** Multi-window CT (wide, mediastinum, brain) via `window()`, then RGB PIL image, resized/padded to 448.
   - **Other models:** `normalize_slice()` then grayscale PIL, resized/padded to 512.

Experiments that do *not* load CT include: `no_image`, `report`, `timeline_only`, `all_vb_timeline_only`, `retrieved_timeline`, `retrieved_timeline_per_iteration`, `retrieved_timeline_per_iteration_summarization`, and path-only variants when path tiles are present.

## Related scripts

- **download_subsampled_ct.py:** Main script to download NIfTI files (person_ids or retrieval CSV).
- **full_test_ehr_vb_download.py:** Builds the full test split parquet (with `patient_string`, and optionally `nifti_path` / `_accession_number`) used for `all_vb_timeline_only` and `all_vb_image_only`.
- **ct_coverage.py:** Reports CT coverage (e.g. how many person_ids have `nifti_path`) per task using config and BQ/cache data.
