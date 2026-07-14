# CT scans

This document describes how CT (NIfTI) data is obtained, where it lives, and how it is used in the Vista Eval VLM pipeline.

## Source

As of the v1.5 substrate, CTs are linked by **DICOM UID**, not by a path string. The task/cohort
tables carry `image_study_uid` and `image_series_uid`; the string `nifti_path` column is deprecated
(now a placeholder INTEGER) and is **not** read. The canonical GCS location is:

- **Bucket:** `su-vista-uscentral1`
- **Prefix (snapshot):** `chaudhari_lab/ct_data/ct_scans/vista/feb26`
- **Blob:** `{prefix}/{image_study_uid}__{image_series_uid}.nii.gz`

The prefix is a **config value** — `paths.ct_snapshot_prefix` in
[configs/all_tasks.yaml](../configs/all_tasks.yaml), defaulting to
`DEFAULT_CT_SNAPSHOT_PREFIX` in [src/vqa_dataset.py](../src/vqa_dataset.py). Because the snapshot is
config-driven, re-materializing the CT data to a new snapshot (feb26 → …) is a **config/data
change, not a code edit**. The prior `nov25` snapshot has been deleted from the bucket (zero
objects); `feb26` is the only snapshot.

Blob resolution is centralized in `resolve_ct_blob(study_uid, series_uid, prefix)`
([src/vqa_dataset.py](../src/vqa_dataset.py)): it returns `(blob_path, filename)`, or `None` when
either UID is null/empty (the row then fails closed to a no-image result — the same as a missing
CT — and never reaches for `nifti_path`).

## Config

In [configs/all_tasks.yaml](../configs/all_tasks.yaml):

```yaml
paths:
  ct_snapshot_prefix: "chaudhari_lab/ct_data/ct_scans/vista/feb26"
  # ct_dir:  # unset -> force GCS (the old nov25 local dir is deleted; do not re-add nov25)
```

- If `ct_dir` **is** set and the resolved CT **filename** (`{study}__{series}.nii.gz`) exists under
  `ct_dir`, [vqa_dataset.py](../src/vqa_dataset.py) loads the volume from disk.
- Otherwise it uses the GCP Storage client to download the blob (`{ct_snapshot_prefix}/{filename}`)
  and load from a temporary file. Leaving `ct_dir` unset forces this GCS path.

## Usage in the dataset

For experiments that use CT images, [PromptDataset](../src/vqa_dataset.py) in `vqa_dataset.py`:

1. Reads `image_study_uid` and `image_series_uid` from each row.
2. Resolves the blob path and filename via `resolve_ct_blob` (fails closed to no-image on null UIDs).
3. Loads the NIfTI from `ct_dir` if the filename exists locally, else from GCP Storage.
4. Samples axial slices from the volume. The number and spacing depend on the experiment. Every branch spaces slices evenly across `[0, depth)` via `index = int(i/(n-1) * (depth-1))`:
   - **50 slices:** `no_timeline`, `all_vb_image_only`.
   - **30 slices:** `axial_all_image`, `retrieved_timeline_with_image`, `retrieved_timeline_per_iteration_summarization_with_image`.
   - **10 slices:** `no_report`.
5. Preprocessing:
   - **Gemma-style models:** Multi-window CT (wide, mediastinum, brain) via `multi_window_rgb()`, then RGB PIL image, resized/padded to 448.
   - **Other models:** `grayscale()` then grayscale PIL, resized/padded to 512.

Experiments that do *not* load CT include: `no_image`, `report`, `timeline_only`, `all_vb_timeline_only`, `retrieved_timeline`, `retrieved_timeline_per_iteration`, `retrieved_timeline_per_iteration_summarization`, and path-only variants when path tiles are present.

## Downloading CT data (legacy — pending v1.5 migration)

> **Note:** The download / subsample / coverage tooling below is still keyed on the old
> `nifti_path` string + `nov25` prefix and is **off the golden / `run_bq` eval path**. It is
> deferred and will break if run against v1.5 until migrated onto `resolve_ct_blob` + the
> `(study, series)` UID queries. The eval pipeline does not use it.

Script: [src/data_tools/ct_info/download_subsampled_ct.py](../src/data_tools/ct_info/download_subsampled_ct.py).

Two modes:

1. **Person IDs + BQ lookup:** Provide a CSV of `person_id`. The script looks up the CT link from the cached BigQuery data, then downloads only missing NIfTI files to a local directory.
2. **Retrieval-style CSV:** Provide a CSV that already carries the CT link. The script reads it and downloads only missing files.

Output is written to a local directory (e.g. under `downloaded_ct_scans/...`). Set `paths.ct_dir` in [configs/all_tasks.yaml](../configs/all_tasks.yaml) to this directory so inference can load from disk instead of GCP.

## Related scripts

- **download_subsampled_ct.py:** Downloads NIfTI files (legacy nov25 tooling — see note above).
- **full_test_ehr_vb_download.py:** Builds the full test split parquet (with `patient_string`, and image link columns) used for `all_vb_timeline_only` and `all_vb_image_only`.
- **ct_coverage.py:** Reports CT coverage per task (legacy nov25 tooling — see note above).
