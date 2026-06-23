Reference: docs/claude_ops.md

# Stand up VISTA multimodal VLM eval on GCP — Bench v1_5 (EHR-timeline + CT + pathology)

**Status: Draft** (2026-06-22) · revised against Codex `/review-plan` (2026-06-22)

## Goal

Get the `vista_eval_vlm` harness running **on a GCP GPU VM in the BAA project** and producing
results for VISTA Bench **v1_5** across three modalities — **EHR patient-timeline text, CT
(NIfTI), and pathology (WSI)** — for the open-weight VLM roster (MedGemma, Qwen3-VL,
InternVL3.5, OctoMed). This is a *stand-up + re-target* effort, not new pipeline code: the
fork (`origin/main`) already implements every experiment type end-to-end; the work is
(a) provisioning GCP, (b) bumping the harness from v1_3 → v1_5 **and the version-hardcoded
path resolvers** (see Versioned data layout contract), and (c) materializing the three
modalities' data for the v1_5 cohort.

**Why GCP, not the Weill B200s:** the whole roster is 4–8B VLMs that fit on a single
A100/H100, so B200s buy throughput, not capability. The bench BQ dataset and both image
buckets already live in the BAA project `som-nero-plevriti-deidbdf` — running in-project
means in-region CT/WSI download, no cross-institution sync of de-id PHI, and simple ADC
auth. The B200 node stays available as a scale-out lever for the full sweep (Ryan's
Weill/B200 login docs to be located if/when needed).

## Context & key references

- Repo: `code/vista_eval_vlm`, branch `origin/main` @ `3794e42` (2026-03-14). **Do NOT merge
  `upstream/main`** — it is Ryan's separate *MMBU* benchmark and deletes the entire VISTA
  BQ + retrieval + CT/pathology integration. The only upstream artifact worth porting later
  is `src/models/api_models.py` (frontier models) — deferred (needs BAA-endpoint rewrite).
- Existing pipeline docs (authoritative, reused — this plan POINTS at them): `docs/00-data-setup.md`,
  `docs/01-pathology-and-path-tools.md`, `docs/02-ct-scans.md`, `docs/03-vista-bench-data-cohort.md`,
  `docs/04-running-the-pipeline.md`.
- Bench task defs: `vista_bench/src/vistabench/task_metadata.py` (`TASK_REGISTRY`); v1_5 dataset
  + 3 tables in `vista_bench/src/vistabench/config.py` (radiation dropped in v1_5).
- **The task-registry JSONs the harness reads are NOT auto-exported by `vista_bench`.** The
  existing `meds_tools/scripts/task_def/02_task_to_prompt.py` only produces `prompts_by_task.json`
  **from an already-existing `valid_tasks.json`** — it does **not** generate `valid_tasks_v1_5.json`
  or `image_valid_tasks.json` from `TASK_REGISTRY`. So the seam is wider than "regen one file"
  (see Phase 1 + Open Q1).
- PHI: medical-data repo (BQ / OMOP / DICOM / WSI). Commits go through `/phi-vet` →
  `/commit-review`. Frontier/public-API models are a BAA violation on this data — out of scope.

### Versioned data layout contract (decide before Phase 2)

Bumping `VISTA_BENCH_DATASET` + `configs/all_tasks.yaml` is **not sufficient** — these
resolvers are version-hardcoded and must each be moved to v1_5 or *intentionally* left shared:

| Artifact | Current resolver (code) | v1_5 decision |
|---|---|---|
| BQ table source | `VISTA_BENCH_DATASET = "vista_bench_v1_1"` (`query_utils.py:236`) | → `vista_bench_v1_5` |
| BQ local cache | `bigquery_data_2_3/{source_csv}` (`run_bq.py:245`/`task_data_utils.py`) | new v1_5 cache **or** delete cache so BQ is queried (avoid stale shadowing) |
| Timeline CSVs | root `{source_csv}/` or `v1_2/` fallback (`task_data_utils.py:14,38,45`) | confirm where v1_5 timeline CSVs live |
| Path subsampled CSVs | `v1_3/{source_csv}/` (`run_bq.py:262`) | v1_5 output base + update `resolve_path_subsampled_csv_path` |
| CT NIfTI prefix | `nov25` (`vqa_dataset.py:12,15`) | vista_bench v1_5 uses `feb26` primary, `nov25` fallback (`vista_bench config.py:53,59`) — verify |
| Task JSONs | `tasks/valid_tasks_v1_3.json` etc. | `tasks/valid_tasks_v1_5.json` + prompts + image_prompts |

Open Q2 (authoritative bucket layout) gates this table.

## Machine posture

Planner = local Mac (this session): authors the plan, no execution. Executor = the **GCP GPU
VM**: runs every step. Verification criteria below are the handoff spec.

## Approach (phased)

### Phase 0 — GCP infra & access
1. GPU VM in / peered to BAA project `som-nero-plevriti-deidbdf`, **region us-central1**
   (matches `gs://su-vista-uscentral1` + BQ; avoids egress). GPU: 1× A100-80GB or H100 (L4 for
   the 4B-only subset). Ample local SSD for NIfTI + WSI tiles.
2. ADC (`gcloud auth application-default login`) with BQ read on `vista_bench_v1_5` + Storage
   read on `gs://vista_bench`, `gs://su-vista-uscentral1`, and the (hidden) WSI bucket.
3. `HF_TOKEN` for gated weights (MedGemma). Export, do not commit.
4. Clone the VISTA fork; `./scripts/setup.sh` (uv `.venv` + `requirements-default.txt`). Skip
   the `llava` env (no LLaVA-Med in v1 roster).

### Phase 1 — Bump to v1_5: dataset + registry + path resolvers
5. `query_utils.py:236` `VISTA_BENCH_DATASET` → `vista_bench_v1_5` (it is currently
   `vista_bench_v1_1`). Apply the rest of the Versioned data layout contract.
6. v1_5 source tables: `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`, `diagnostic_tasks`,
   `tb_classification_tasks`.
7. Produce the three JSONs the harness reads from `base_dir/tasks/`, **as three separate
   outputs** (the meds_tools script covers only the middle one):
   - `valid_tasks_v1_5.json` — `{task_name, task_source_csv, is_binary}` from `TASK_REGISTRY`.
   - `prompts_by_task.json` — text templates w/ `[PATIENT_TIMELINE]` (meds_tools script).
   - `image_valid_tasks.json` — image-only templates (no current generator).
   See Open Q1: reuse meds_tools + hand-build the other two, **or** add a durable
   `vista_bench` exporter that emits all three from `TASK_REGISTRY`.
8. `configs/all_tasks.yaml`: GCP paths (`base_dir`, `results_dir`, `ct_dir`, `path_tile_base`,
   `cache_dir`), `valid_tasks: tasks/valid_tasks_v1_5.json`, and set
   `use_constrained_decoding_for_binary` intentionally (currently `false`, `all_tasks.yaml:17`).

### Phase 2 — EHR patient-timeline (text)  [gated on timeline CSVs]
9. Sync bench data per `docs/00-data-setup.md` (`gsutil -m rsync gs://vista_bench <base_dir>`).
10. **Hard gate (was Open Q3):** `no_image`, `timeline_only`, `report`, `no_report`, and
    `path_full` all return `None` without the timeline CSVs (`run_bq.py:265,274,287,290,558,568`).
    Confirm `{task}_subsampled.csv` **and** `{task}_subsampled_no_img_report.csv` exist per
    `task_source_csv`; if absent, build them via `full_test_ehr_vb_download.py` / the subsample
    timeline producer **before** any timeline/path_full experiment.
11. Smoke then run: `no_image`, `timeline_only`, `report`, `no_report`.

### Phase 3 — CT (NIfTI)
12. `ct_coverage.py` → CT availability per v1_5 task.
13. Resolve the CT prefix first (`feb26` vs `nov25`, Open Q3), then `download_subsampled_ct.py`
    → NIfTIs into `ct_dir` (in-region; mind volume). Per `docs/02-ct-scans.md`.
14. Run `axial_all_image` / `no_timeline` / `all_vb_image_only`.

### Phase 4 — Pathology (WSI) — heaviest lift
15. Path cohort subsample for v1_5. `subsample_v1_3_path.py` defaults hardcode `bigquery_v1_3`
    / `valid_tasks_v1_3.json` / `v1_3` (`:18-21,185`) — running it as-is regenerates v1_3.
    **Parameterize** into `subsample_path.py --version/--bigquery-dir/--valid-tasks/--output-base`
    (Open Q1 modularity; v1_6 is already on the horizon) rather than copy.
16. `download_path_subsampled.py` → WSIs from the hidden bucket (**cost/space cap**, Open Q6).
17. `csv_tile_wsi.py` + `tile_wsi.py` → tile at 896 px / 10× into `path_tile_base/test_patch/`.
    **Note:** runtime samples up to **100** tiles per slide (`run_bq.py:461`), not 125 — pick
    the tile budget deliberately (Open Q5) and assert it downstream.
18. Run `path`, `path_image_and_report`, `path_full`.

### Phase 5 — Models & sweep
19. Roster (vLLM; constrained Yes/No decoding for binary when enabled): `gemma3`
    (`google/medgemma-4b-it`, `google/medgemma-1.5-4b-it`), `qwen3vl` (`Qwen/Qwen3-VL-8B-Instruct`),
    `intern` (`OpenGVLab/InternVL3_5-8B-hf`), `octomed` (`OctoMed/OctoMed-7B`).
20. Run via `eval/bq_gcp.sh` (edit venv path, GPU, `MAX_MODELS`); single-GPU sequential. If a
    multi-GPU VM, adapt the `weill.sh` round-robin.

### Phase 6 — Results
21. `src/results/final_metrics.py` / `results_analyzer.py` → per-task accuracy / weighted-F1.
22. (Optional) feed into `vista-eval` `reports/vlm_report.py` to compare vs embedding baselines.

## Files to modify
- `src/data_tools/utils/query_utils.py` — `VISTA_BENCH_DATASET` → `vista_bench_v1_5`.
- `src/data_tools/utils/task_data_utils.py` — timeline/path/cache resolvers per the layout contract.
- `configs/all_tasks.yaml` — GCP paths, v1_5 registry, model/task/experiment lists, constrained-decoding flag.
- `src/data_tools/csv_helper/subsample_v1_3_path.py` → parameterized `subsample_path.py` (`--version` etc.).
- `eval/bq_gcp.sh` — venv path, GPU, `MAX_MODELS`.
- *(vista_bench, optional, Open Q1)* new `src/vistabench/export_task_registry.py` — emit all three JSONs from `TASK_REGISTRY`.
- Regenerated task JSONs — live in the bench bucket under `tasks/`, not in the repo.

## Open questions
1. **Registry regeneration / seam:** reuse `meds_tools/.../02_task_to_prompt.py` (covers only
   `prompts_by_task.json` from an existing `valid_tasks.json`) + hand-build `valid_tasks_v1_5.json`
   and `image_valid_tasks.json`, **or** add a durable `vista_bench` exporter for all three?
   Lean: exporter — v1_5 adds diagnostic tasks and the sweep will recur (v1_6 pending), so
   realistic reuse exists.
2. **Authoritative v1_5 local-artifact layout:** root `{source_csv}/`, `v1_5/{source_csv}/`, or
   reuse of `bigquery_data_2_3` / `v1_3` names? Gates the Versioned data layout contract; do not
   silently reuse v1_2/v1_3 names for v1_5 artifacts.
3. **CT NIfTI prefix:** vista_bench v1_5 primary is `feb26` (eval repo docs say `nov25`). Use
   `feb26`, `nov25`, or both-with-fallback? Default: `feb26` primary + `nov25` fallback, verified on VM.
4. **Diagnostic scope:** include the v1_5 diagnostic suite (stage/PD-L1/histology/mets)? It pairs
   naturally with pathology but **only if text + image prompt templates are authored/reviewed** —
   gate inclusion on prompt coverage.
5. **Pathology tile budget:** runtime samples 100/slide today; tiling spec mentions 125. Run at
   100, or change the runtime cap to 125? Plus the slide/patient cap for cost+disk.
6. **WSI bucket:** confirm the exact (hidden) bucket name + read grant for the GCP principal.
7. **GPU shape:** single A100/H100 sequential, or multi-GPU VM for parallel model runs?

## Verification & VM handoff
Executed on the GCP VM. Per-phase Expected / Stop (verification must fail loudly on silent
fallbacks — a written CSV with no images / empty responses is a FAILURE, not a pass):

- **Phase 0** — `hostname`; `gcloud config get-value project`;
  `bq ls som-nero-plevriti-deidbdf:vista_bench_v1_5` lists the 3 tables;
  `gsutil ls gs://su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/{feb26,nov25}/ | head`;
  `nvidia-smi`; `uv run python -c "import vllm,torch; print(torch.cuda.get_device_name(0))"`.
  **Stop:** any auth/dataset/GPU failure.
- **Phase 1** — JSON structural validation: every configured task is in `valid_tasks_v1_5.json`;
  every `task_source_csv` ∈ the 3 v1_5 tables; every text prompt exists + contains
  `[PATIENT_TIMELINE]` where the experiment needs it; every image/path task has an image prompt.
  BQ schema+rowcount per table/task (cols `task,split,label,person_id,embed_time` + modality cols
  `nifti_path,_accession_number,path_image_path,path_note_text`). **Print the resolved data SOURCE
  per task and assert any local cache is a v1_5 export or absent** (guards `bigquery_data_2_3`
  shadowing). **Stop:** unresolved table/task, missing prompt, stale cache shadowing v1_5.
- **Phase 2** — input-file inventory (`{task}_subsampled.csv` AND `_subsampled_no_img_report.csv`);
  nonzero inner-join of timeline CSV `person_id` against v1_5 BQ rows; 1-task×1-model smoke writes
  a non-empty results CSV with `model_response` populated. **Stop:** missing CSV / empty CSV / all rows skipped.
- **Phase 3** — sample 20 v1_5 `nifti_path`, normalize via `_nifti_path_to_blob_and_filename`,
  assert local file or GCS blob exists; CT smoke asserts `used_image.sum() > 0` (PromptDataset
  silently catches NIfTI load errors and continues text-only — hard-fail this). **Stop:** 0 blobs resolve / `used_image` all 0.
- **Phase 4** — path subsampled rowcount > 0; local WSI download count; tile-manifest rowcount;
  folder stem matches `path_image_path`; result rows carry the chosen tile count and `used_image==1`.
  **Stop:** all path rows dropped ("No rows with tiles…") / tile-count mismatch.
- **Phase 5** — one results CSV per (task, model, experiment), resume intact; `nvidia-smi` headroom.
  **Stop:** OOM (drop batch_size / model count).
- **Phase 6** — per-task metric sanity: evaluated-row count == expected; no all-empty
  `model_response`; label-parser coverage reported; accuracy + weighted-F1 (incl. multiclass).

`/vm-handoff` renders this section into a runnable `docs/vm-status/<date>-<sha>.md` on the VM.
