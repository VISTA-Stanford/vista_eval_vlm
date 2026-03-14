# Data Setup: Understanding how to get VISTA Bench

This document explains what data Vista Eval VLM expects, how to get it from **BigQuery** or by **downloading from Google Cloud Storage (gs://)**, and the **folder structure** you need under `paths.base_dir` (and related config paths).

## What the data is

Vista Bench is a benchmark built on de-identified oncology data. The pipeline uses:

1. **Task tables** — BigQuery tables (one per cohort/source) with columns such as `person_id`, `task`, `split`, `label`, `embed_time`, `question`, `nifti_path`, etc. Rows are filtered by `task` (e.g. `has_recurrence_1_yr`). Table names are the **task_source_csv** value (e.g. `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`).
2. **Timeline CSVs** — Local CSVs that contain **patient_string** (or **patient_timeline**): one row per patient with clinical-event text. These are merged with BigQuery rows on `person_id`. They may also include `report` (radiology report) and `embed_time`.
3. **Optional:** **Full parquet** files (`{task_name}_full.parquet`) for full-test experiments; **path_subsampled** and **retrieval** CSVs for pathology and retrieval experiments (see [01-pathology-and-path-tools.md](01-pathology-and-path-tools.md) and [05-retrieval.md](05-retrieval.md)).

If you have **direct BigQuery access**, the pipeline can query at runtime and only needs the timeline (and related) CSVs locally **THIS WILL BE EXPENSIVE SO DO NOT JUST DO THIS**. If you **don’t** have BQ access, you need a **local cache** of the BigQuery exports plus the same CSVs. That cache is often obtained by syncing a pre-built **Vista Bench GCS bucket** to your machine.

---

## BigQuery (source of truth)

- **Project:** `som-nero-plevriti-deidbdf`
- **Dataset:** `vista_bench_v1_1` (defined in [query_utils.py](../src/data_tools/utils/query_utils.py) as `VISTA_BENCH_DATASET`). Your org may use a different dataset name (e.g. `vista_bench_v1_2` / `vista_bench_v1_3`) for newer cohorts; the code that fetches task data uses this constant.
- **Tables:** One table per cohort. The table name is the **task_source_csv** from the task registry (e.g. `valid_tasks_v1_3.json`). Example: `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`. Each table has a `task` column; the pipeline queries `WHERE task = @task_name` for each task.

**To export from BigQuery to local CSV (for offline use):**

- Use BigQuery Console or `bq extract` to export the table to GCS (e.g. as CSV), then download or sync to your machine. The pipeline expects **one CSV per source table** in the local cache (see [Folder layout](#folder-layout) below). The CSV should contain all columns and all tasks for that table; the code filters by `task` when loading.
- Alternatively, run the pipeline with BQ credentials; it will query on the fly and can optionally write a local cache (if you add or use a script that saves the fetched DataFrame to `base_dir/bigquery_data_2_3/{source_csv}`).

---

## Downloading from GCS (gs://)

If Vista Bench is distributed as a **GCS bucket** (e.g. `vista_bench`), you can sync it to a local directory and use that as (or under) `paths.base_dir`.

**Sync the whole bucket to a local folder:**

```bash
gsutil -m rsync -r gs://vista_bench/ /path/to/your/vista_bench
```

Then set `paths.base_dir` in [configs/all_tasks.yaml](../configs/all_tasks.yaml) to `/path/to/your/vista_bench` (or to a subfolder of that, if the bucket layout uses a top-level subfolder like `vista_bench/`).

**Sync a specific prefix (e.g. only timeline + BQ cache):**

```bash
# Example: only a cohort subfolder
gsutil -m rsync -r gs://vista_bench/v1_2/ /path/to/your/vista_bench/v1_2
gsutil -m rsync -r gs://vista_bench/bigquery_data_2_3/ /path/to/your/vista_bench/bigquery_data_2_3
```

Use the bucket paths and prefix structure provided by your data owner. Common layouts include:

- `bigquery_data_2_3/<source_csv>` — exported BQ table CSVs (one file per table name).
- `v1_2/<source_csv>/` — retrieval and subsampled timeline CSVs.
- `v1_3/<source_csv>/` — path cohort and path_subsampled CSVs.
- `<source_csv>/` at root — timeline CSVs and optionally `_full.parquet`.
- `tasks/` — `valid_tasks_*.json`, `prompts_by_task.json`, `image_valid_tasks.json`.

**Credentials:** Configure Application Default Credentials (e.g. `gcloud auth application-default login`) or a service account so `gsutil` and the pipeline can access the bucket and BigQuery.

---

## Folder layout

Set **`paths.base_dir`** in `all_tasks.yaml` to the root of this tree (or the path you synced from GCS). All paths below are relative to `base_dir` unless noted.

| Path under base_dir | Purpose |
|---------------------|--------|
| **tasks/** | Task registry and prompts: `valid_tasks_v1_3.json` (or the file named in `paths.valid_tasks`), `prompts_by_task.json` (`paths.prompts`), `image_valid_tasks.json` (`paths.image_prompts`). |
| **bigquery_data_2_3/** | Local BigQuery cache. One **file** per table: the filename is the table (source_csv) name, e.g. `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`. No extension is required; the code uses `resolve_local_bq_cache_path(base_path, source_csv)` which points to a single path per table. If this file exists, the pipeline loads from it instead of querying BigQuery. |
| **bigquery_v1_3/** | Same idea for v1_3 path cohort: one file per table name for path-related BQ exports (used by subsample_v1_3_path and path pipeline). |
| **{source_csv}/** | Per-cohort folder at **root** of base_dir. Contains timeline CSVs: `{task_name}_subsampled.csv`, `{task_name}_subsampled_no_img_report.csv` (when using subsample), and optionally `{task_name}_full.parquet`. Example: `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/has_recurrence_1_yr_subsampled.csv`. |
| **v1_2/{source_csv}/** | v1_2 cohort: retrieval CSVs (`retrieval_subsample_50.csv`, `{task_name}_subsampled_retrieval.csv`) and sometimes timeline CSVs. The pipeline looks here for retrieval data and as a fallback for timeline CSVs if not found under `{source_csv}/`. |
| **v1_3/{source_csv}/** | v1_3 cohort: path CSVs only, e.g. `{task_name}_path_subsampled.csv`. |

**Other config paths (not under base_dir):**

- **paths.results_dir** — Where result CSVs are written; can be anywhere.
- **paths.ct_dir** — Directory for NIfTI files (CT). If set, the pipeline loads CT from disk when the file exists here; otherwise it can use GCP Storage.
- **paths.path_tile_base** — Root that contains **test_patch/** with one folder per pathology slide (see [01-pathology-and-path-tools.md](01-pathology-and-path-tools.md)).

**Example minimal layout (timeline + BQ cache only):**

```
/path/to/vista_bench/
├── tasks/
│   ├── valid_tasks_v1_3.json
│   ├── prompts_by_task.json
│   └── image_valid_tasks.json
├── bigquery_data_2_3/
│   └── progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr   # one file
├── v1_3/
│   └── progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/ per table
        ├── has_recurrence_1_yr_subsampled.csv
        └── has_recurrence_1_yr_subsampled_no_img_report.csv
        └── has_recurrence_1_yr_path_subsampled.csv

```

---

## Checklist

1. **Credentials** — BigQuery and (if needed) GCP Storage access for project `som-nero-plevriti-deidbdf` and any bucket (e.g. `vista_bench`, `su-vista-uscentral1`).
2. **base_dir** — Create or sync a directory that contains at least `tasks/` (valid_tasks, prompts, image_prompts) and either:
   - **bigquery_data_2_3/{source_csv}** for each table you need, or
   - Live BigQuery access so the pipeline can fetch and merge with timeline CSVs.
3. **Timeline CSVs** — Under `{source_csv}/` or `v1_2/{source_csv}/` with the expected names (`_subsampled.csv`, `_subsampled_no_img_report.csv` as per config `subsample`).
4. **config** — Set `paths.base_dir` (and optionally `paths.results_dir`, `paths.ct_dir`, `paths.path_tile_base`) in `configs/all_tasks.yaml` to match your layout.

For pathology, CT, and retrieval data flows, see [01-pathology-and-path-tools.md](01-pathology-and-path-tools.md), [02-ct-scans.md](02-ct-scans.md), and [05-retrieval.md](05-retrieval.md).
