# Running the full pipeline

This document describes how to run the Vista Eval VLM pipeline from config to results: config layout, entry points, flow inside `run_bq.py`, how `vqa_dataset.py` is used, and output layout.

## Config: all_tasks.yaml

Main sections in [configs/all_tasks.yaml](../configs/all_tasks.yaml):

| Section | Purpose |
|--------|---------|
| **paths** | `base_dir` (Vista Bench root), `results_dir`, `ct_dir`, `path_tile_base`, `valid_tasks`, `prompts`, `image_prompts`. |
| **model** | `device` (e.g. `cuda`). |
| **runtime** | `cache_dir` (HF/transformers/vLLM cache), `batch_size`, `max_new_tokens`, `use_constrained_decoding_for_binary`. |
| **models** | List of `{ type, name }` (e.g. `gemma3`, `google/medgemma-4b-it`). |
| **tasks** | List of task names to run (subset of valid_tasks; use config tasks or override via `--tasks`). |
| **experiments** | List of experiment names (e.g. `no_image`, `path_full`, `axial_all_image`, retrieval variants). |
| **retrieval** | Used when any experiment is retrieval-based: `enabled`, `corpus_dir`, `cache_dir`, iterations, batch size, timeline cache, etc. |
| **timeline_truncation** | `mode` (`last_k_events` or `max_chars`), `k` or `max_chars`. |
| **subsample** | If `true`, use `_subsampled*.csv` filenames. |
| **weill** | `gpu_nodes` (list of GPU IDs) for [eval/weill.sh](../eval/weill.sh). |

Edit paths for your environment; set `tasks` and `experiments` to the tasks and modalities you want.

## Entry points

### Weill cluster

Script: [eval/weill.sh](../eval/weill.sh).

- Reads `configs/all_tasks.yaml`, gets `models` and `weill.gpu_nodes` (or `WEILL_GPUS` env / command-line args).
- Runs one `run_bq.py` process per model; GPUs are assigned round-robin. If there are more models than nodes, each node runs its assigned models sequentially.
- Logs go to `logs/<timestamp>/model_<gpu>_<model_name>.log`.
- Set `HF_TOKEN` in .sh file; uses project `.venv` (switch to `llava` if wanting use a llava model).
- Command run: `./eval/weill.sh`

From repo root:

```bash
./eval/weill.sh
# Or specific GPUs: ./eval/weill.sh 0 1 2 3
```

## Flow inside run_bq.py

The following describes how [TaskOrchestrator](../src/vista_run/run_bq.py) in `run_bq.py` works: initialization, data loading per experiment type, prompt building for every experiment, and the inference loop.

### Initialization (`TaskOrchestrator.__init__`)

1. **Load YAML config** from `config_path`; set `base_path` (Vista Bench root) and `results_base` from `paths.base_dir` and `paths.results_dir`.
2. **Set environment variables** via `_set_envs(cache_dir)`: `HF_HOME`, `TRANSFORMERS_CACHE`, `VLLM_CACHE_ROOT`, `TOKENIZERS_PARALLELISM` so all model/tokenizer caches use the same directory.
3. **Load task registries** (paths relative to `base_path`):
   - `valid_tasks` from `paths.valid_tasks` (e.g. `tasks/valid_tasks_v1_3.json`) — list of `{ task_name, task_source_csv, is_binary?, ... }`.
   - `prompts_map` from `paths.prompts` (e.g. `tasks/prompts_by_task.json`) — template per task, usually containing `[PATIENT_TIMELINE]`.
   - `prompts_map_summarized` from `paths.prompts_summarized` if the file exists (used for one retrieval experiment).
   - `image_prompts_map` from `paths.image_prompts` if the file exists (e.g. `tasks/image_valid_tasks.json`) — used for image-only or path experiments.
4. **Initialize BigQuery client** (project `som-nero-plevriti-deidbdf`) and **GCP Storage client** (for NIfTI when not using local `ct_dir`).
5. **Load model:** `load_model_adapter(model_type, model_name, device, cache_dir)` then `adapter.load()` → `self.model`, `self.processor`.
6. **Optional retrieval:** If `retrieval.enabled` is true, import `LocalPatientRetriever` and construct it with `retrieval.corpus_dir` and `retrieval.cache_dir` (requires meds_mcp). Otherwise `self.retriever` is None. `self.retrieval_cfg` always holds the config `retrieval` section.

### Top-level loop: `run_inference(task_names)`

- **Task list:** Use the `valid_tasks` list.
- **Experiment list:** `experiments = cfg.get('experiments', ['no_image'])`.
- **Per task**, the orchestrator decides which **data variants** are needed based on the experiment list:

| Flag | True when any of these experiments are in `experiments` | Data loader used |
|------|--------------------------------------------------------|-------------------|
| `needs_report_timeline` | `no_report`, `timeline_only`, `report` | `_load_task_data(use_no_report_csv=True, require_timeline=True)` |
| `needs_no_timeline` | `no_timeline` | `_load_task_data(use_no_report_csv=True, require_timeline=False)` |
| `needs_retrieved_timeline` | Any in `RETRIEVAL_EXPERIMENTS` | `_load_retrieval_task_data` |
| `needs_all_vb_timeline` | `all_vb_timeline_only` | `_load_all_vb_timeline_task_data` |
| `needs_all_vb_image` | `all_vb_image_only` | `_load_all_vb_image_task_data` |
| `needs_path` | `path`, `path_image_and_report` | `_load_path_task_data` |
| `needs_path_full` | `path_full` | `_load_path_full_task_data` |
| `needs_normal` | Any experiment not in the above (e.g. `no_image`, `axial_all_image`) | `_load_task_data(use_no_report_csv=False, require_timeline=True)` |

Each loader is called at most once per task; the same `(df, timeline_col, source_csv)` is reused for every experiment that uses that variant. Then **for each experiment** the code selects which loaded data to use and calls `_process_single_task_with_data(task_info, experiment, df, timeline_col)`.

**Experiment → data mapping:**

- `no_timeline` → `loaded_no_timeline`
- `all_vb_image_only` → `loaded_all_vb_image`
- `all_vb_timeline_only` → `loaded_all_vb_timeline`
- `path`, `path_image_and_report` → `loaded_path`
- `path_full` → `loaded_path_full`
- `no_report`, `timeline_only`, `report` → `loaded_no_report`
- Any in `RETRIEVAL_EXPERIMENTS` → `loaded_retrieval`
- **All other experiments** (e.g. `no_image`, `axial_all_image`) → `loaded_normal`

If the chosen loader returned None (e.g. missing file), that experiment is skipped for this task.

### Every experiment: data source, prompt, and images

| Experiment | Data source | Timeline column | Prompt construction | Images (in vqa_dataset) |
|------------|-------------|-----------------|---------------------|-------------------------|
| **no_image** | BQ/cache + `_subsampled.csv` | From timeline CSV merge | `prompts_map[task]` with `[PATIENT_TIMELINE]` replaced by truncated timeline | None |
| **axial_all_image** | Same as no_image | Same | Same (timeline in prompt) | 30 axial CT slices from `nifti_path` |
| **no_timeline** | BQ/cache + `_subsampled_no_img_report.csv` (no timeline merge) | None | `image_prompts_map` or `prompts_map` (template only, no timeline) | 50 axial CT slices |
| **no_report** | BQ/cache + `_subsampled_no_img_report.csv` + timeline | From CSV | `prompts_map` with timeline only (no report) | 10 axial CT slices |
| **timeline_only** | Same as no_report | From CSV | `prompts_map` with timeline only | None |
| **report** | Same as no_report | From CSV | `prompts_map` with timeline + `"\nRadiology Report: " + row['report']` | None |
| **all_vb_timeline_only** | `{task}_full.parquet` | `patient_string` | `prompts_map` with truncated `patient_string` | None |
| **all_vb_image_only** | `{task}_full.parquet` (rows with `nifti_path`, `_accession_number`) | None | `image_prompts_map` or `prompts_map` | 50 axial CT slices |
| **path** | `v1_3/.../{task}_path_subsampled.csv` + `path_tile_paths` from test_patch | None | `image_prompts_map` or `prompts_map` (template only) | Pathology tiles from `path_tile_paths` |
| **path_image_and_report** | Same as path | None | Template + `"\n\nPathology note:\n" + path_note_text` | Same (path tiles) |
| **path_full** | Path subsampled + timeline CSV merge | From CSV | Template + pathology note (if present) + `"\n\nPatient timeline:\n" + timeline` | Same (path tiles) |
| **retrieved_timeline** | `_subsampled_retrieval.csv` | N/A (timeline from retrieval) | `build_retrieval_prompts` (single timeline per patient, truncated) | None |
| **retrieved_timeline_per_iteration** | Same | N/A | `build_retrieval_prompts` (per-iteration: one row per patient per iteration) | None |
| **retrieved_timeline_per_iteration_summarization** | Same | N/A | Same, uses `prompts_map_summarized` and optionally summarized timeline | None |
| **retrieved_timeline_with_image** | Same | N/A | Per-iteration timeline + image | 50 axial CT slices |
| **retrieved_timeline_per_iteration_summarization_with_image** | Same | N/A | Summarized per-iteration + image | 50 axial CT slices |

### Prompt building: `_build_prompts_for_experiment(df, task_info, experiment, timeline_col)`

This method returns a DataFrame (possibly with more rows for per-iteration retrieval) with a `dynamic_prompt` column. Branch order:

1. **no_timeline, all_vb_image_only, path** — Use `image_prompts_map.get(task_name)` or `prompts_map.get(task_name)` as the only prompt text; no timeline or report injected.
2. **path_image_and_report** — Same base template; append `"\n\nPathology note:\n" + row[path_note_text]` (or template only if no path_note column).
3. **path_full** — Base template + optional pathology note + `"\n\nPatient timeline:\n" + timeline_col` (truncated timeline).
4. **report** — `prompts_map` with `[PATIENT_TIMELINE]` replaced by `timeline + "\nRadiology Report: " + row['report']`.
5. **Retrieval experiments** (`RETRIEVAL_EXPERIMENTS`) — Delegates to `build_retrieval_prompts(...)` in `prompt_building.py`: runs iterative retrieval (or loads timeline cache), then either one timeline per row (retrieved_timeline) or expands to one row per (patient, iteration) with the corresponding timeline/summary. Uses `prompts_map_summarized` only for `retrieved_timeline_per_iteration_summarization` (and its with_image variant). Truncation config is applied except for per-iteration experiments (truncation applied when building the prompt string).
6. **Default (no_image, axial_all_image, timeline_only, all_vb_timeline_only)** — `prompts_map.get(task_name)` with `[PATIENT_TIMELINE]` replaced by the row’s `timeline_col` (already truncated in the data load step).

### Processing one (task, experiment): `_process_single_task_with_data`

1. **Constrained decoding:** If the task has `is_binary` and config `runtime.use_constrained_decoding_for_binary` is true, set `constrained_choices = ["Yes", "No"]`; otherwise None.
2. **Output and resume:** `_setup_output_and_resume(task_info, experiment)` creates the save dir and `out_file = results_dir/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`. For **retrieval experiments** the file is deleted if it exists (no resume). For others, existing rows are read and their `index` values are collected into `existing_indices` for resume.
3. **Build prompts:** `df_exp = _build_prompts_for_experiment(df, task_info, experiment, timeline_col)` (may expand rows for per-iteration retrieval).
4. **Dataset and DataLoader:** `PromptDataset(df=df_exp, prompt_col='dynamic_prompt', experiment=experiment, storage_client, model_type, ct_dir)` then DataLoader with `batch_size`, `prompt_collate`, `num_workers=4`, `persistent_workers=True`, `pin_memory=True`.
5. **Inference loop:** `_run_inference_loop(loader, existing_indices, constrained_choices, out_file, ...)`.

### Inference loop: `_run_inference_loop`

- **Producer thread:** Iterates the DataLoader; for each batch calls `_prepare_batch_for_inference(batch, existing_indices, constrained_choices)`. Items whose `raw_row['index']` is in `existing_indices` are skipped. Remaining items are turned into model inputs (template + prepare_inputs). For **Gemma**-type models, items are grouped by number of images (0, 1, or N) so that each inference batch has a fixed image count; for other models, the batch is sent as-is. Prepared work is put on a queue (max size 2).
- **Main thread:** Consumes from the queue; for each prepared chunk runs `_run_inference_on_batches` (adapter.infer with `max_new_tokens` and optional `constrained_choices`), then builds one result row per item via `_build_result_row` (drops heavy columns, adds `model_response`, `used_image`, and optionally `input_token_length`, `timeline_token_count`). Appends result rows to a buffer; every 10 batches the buffer is appended to `out_file` via `append_to_csv_util`. Every 20 batches calls `torch.cuda.empty_cache()`.
- **Resume:** Only indices not in `existing_indices` are sent to the model; their results are appended to the CSV. So reruns continue from the last saved indices.

### Result row and output

`_build_result_row` drops from the raw row: the timeline column (if any), `note_text`, `patient_string`, `report`, `path_note_text`. It adds `model_response` (and optionally `cumulative_logprob`, `log_probs`), `used_image` (1 if the item had images), and for some experiments `input_token_length` or `timeline_token_count`. Output path: `{results_dir}/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`.

## vqa_dataset.py: PromptDataset and experiment types

[PromptDataset](../src/vqa_dataset.py) in `vqa_dataset.py`:

- **Input:** DataFrame with `dynamic_prompt` (and optionally `path_tile_paths`, `nifti_path`, `image_path`, etc.), plus `experiment`, `storage_client`, `model_type`, `ct_dir`.
- **Output per item:** Dict with `index`, `question` (prompt text), `image` (None, single image, or list of images), `options`, `raw_row` (for building result rows).

**Experiments that load images:**

- **Path:** `path`, `path_image_and_report`, `path_full` — load from `path_tile_paths` (list of local .jpg paths).
- **CT:** `axial_all_image`, `no_timeline`, `all_vb_image_only`, `no_report`, `retrieved_timeline_with_image`, `retrieved_timeline_per_iteration_summarization_with_image` — load NIfTI from `ct_dir` or GCP and sample axial slices (number and preprocessing depend on experiment and model_type).
- **Single image:** If `image_path` is set and the file exists, it is used when no path/CT image is used.

**Experiments that do not load images (text-only or timeline-only):**

- `no_image`, `report`, `timeline_only`, `all_vb_timeline_only`, `retrieved_timeline`, `retrieved_timeline_per_iteration`, `retrieved_timeline_per_iteration_summarization`.

`prompt_collate` returns the batch as a list of these item dicts (no stacking).

## Output

- **Path:** `{results_dir}/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`
- **Resume:** Rows already present (by `index`) are skipped; new rows are appended every 10 batches and at the end.
- **Columns:** Original row columns minus heavy ones (`note_text`, `patient_string`, `report`, `path_note_text`, and the timeline column used for that experiment), plus `model_response`, `cumulative_logprob`, `log_probs`, `used_image`. Some retrieval experiments add `input_token_length`; `no_image` can add `timeline_token_count`.
- **Retrieval experiments:** Output file is overwritten (no resume) when running retrieval experiments.

## Prerequisites

- **Python environment:** Dependencies per repo (e.g. `uv sync` or requirements); vLLM/transformers for the chosen models.
- **Credentials:** BigQuery and GCP Storage access for project `som-nero-plevriti-deidbdf` (and bucket `su-vista-uscentral1` when loading NIfTI from GCP).
- **Path experiments:** OpenSlide and pathology data pipeline as in [01-pathology-and-path-tools.md](01-pathology-and-path-tools.md) (path_tile_base, test_patch).
- **CT experiments:** Either `paths.ct_dir` with NIfTI files on disk or GCP access so NIfTI can be downloaded on the fly.
- **Optional:** For retrieval experiments, `retrieval.enabled: true` and corpus/cache paths; more info in [05-retrieval.md](05-retrieval.md).
