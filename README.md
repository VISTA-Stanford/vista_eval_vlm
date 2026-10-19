# Vista Eval VLM

Evaluation pipeline for vision-language models (VLMs) on the Vista Bench benchmark. Supports timeline-only, CT imaging, pathology slides, and retrieval-based experiments.

**Full pipeline documentation** (pathology, CT, Vista Bench cohort, retrieval, running the pipeline): [docs/](docs/README.md).

---

## Setup

- **Python:** 3.11+
- **Environment:** Use the project root `.venv` for all models except llava models require `llava`.

**scripts/setup.sh (recommended)**

```bash
# From repo root
./scripts/setup.sh
# Install any extra deps from external repos
```
- Creates `.venv` with `uv`, installs the package and `requirements-default.txt`.
- Clones LLaVA-Med into `src/` and creates a separate `llava` env for LLaVA-Med models.

Edit `scripts/setup.sh` if you use different paths or env names. For all runs besides with llava models the default `.venv` is sufficient.

**Tests** (synthetic inputs; no network, weights or patient data):

```bash
UV_PYTHON=3.12 uv sync --extra test && uv run pytest
```

Python 3.14 breaks the sync on the VMs; use 3.12.

---

## Configs

All runtime configuration is in **`configs/all_tasks.yaml`**.

| Section | What to set |
|--------|--------------|
| **paths** | `base_dir`, `results_dir`, `ct_dir`, `path_tile_base` to your Vista Bench and data locations; `valid_tasks`, `prompts`, `image_prompts` (paths relative to `base_dir`). |
| **model** | `device` (e.g. `cuda`). |
| **runtime** | `cache_dir` (HF/transformers/vLLM cache), `batch_size`, `max_new_tokens`, `use_constrained_decoding_for_binary`. |
| **models** | List of `{ type, name }` for each VLM to run (see [Models](#models)). |
| **tasks** | Task names to evaluate (must exist in `valid_tasks` JSON). |
| **experiments** | Experiment(s) to run (e.g. `no_image`, `path_full`, `axial_all_image`, retrieval variants). |
| **retrieval** | If any experiment is retrieval-based: `enabled: true`, `corpus_dir`, `cache_dir`, and other options (see [docs/05-retrieval.md](docs/05-retrieval.md)). |
| **timeline_truncation** | `mode` and `k` (or `max_chars`) for truncating patient timelines. |
| **subsample** | `true` to use `_subsampled*.csv` data. |
| **weill** | `gpu_nodes` (list of GPU IDs) when using `eval/weill.sh`. |

---

## Eval scripts

Run from **repo root**. Each script reads `configs/all_tasks.yaml` and runs `src/vista_run/run_bq.py` for every model in the config.

| Script | Use case |
|--------|----------|
| **`eval/weill.sh`** | Weill cluster (or any multi-GPU machine). Uses `weill.gpu_nodes` from config; override with `./eval/weill.sh 0 1 2 3` or `WEILL_GPUS="0 2 4 6" ./eval/weill.sh`. Activates `.venv` from project root (or `VISTA_VENV`). |
| **`eval/bq_gcp.sh`** | GCP VM with BigQuery; runs models sequentially. **Edit the script:** set venv path and any `MAX_MODELS` / paths for your machine. |
| **`eval/gcp.sh`** | GCP VM (non-BQ). **Edit the script:** set venv path and paths for your machine. |

**Before running**

- Set **`HF_TOKEN`** in the script (or export it) if your models need Hugging Face auth. The token in the repo may be expired; use your own.
- For Weill: ensure `weill.gpu_nodes` in `all_tasks.yaml` matches your GPU IDs (e.g. `[0,1,2,3,4,5,6,7]`).

---

## Models

Models are defined in the config as `type` + `name`. The **type** must match a key in the adapter registry in `src/models/__init__.py`. Supported types and example names:

| type | Example name | Notes |
|------|----------------|------|
| **gemma3** | `google/medgemma-1.5-4b-it`, `google/medgemma-4b-it`, `google/gemma-3-4b-it` | vLLM |
| **qwen3vl** | `Qwen/Qwen3-VL-8B-Instruct` | vLLM |
| **intern** | `OpenGVLab/InternVL3_5-8B-hf` | vLLM |
| **octomed** | `OctoMed/OctoMed-7B` | vLLM |
| **qwen2vl** | `Qwen/Qwen2-VL-2B-Instruct` | |
| **qwen2_5vl** | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` | |
| **medvlm** | `JZPeterPan/MedVLM-R1` | |
| **lingshu** | `lingshu-medical-mllm/Lingshu-7B` | |
| **llava** | `llava-hf/llava-1.5-7b-hf` | |
| **llavamed** | `microsoft/llava-med-v1.5-mistral-7b` | LLaVA-Med needs llava env (see setup). |
| **vertex_gemini** | `gemini-3.1-pro-preview` | Vertex AI REST API, no GPU; settings in the config's `vertex:` block. Responses cached under `{cache_dir}/vertex_gemini_responses/`. |

**Note:** Gemma3, Qwen3-VL, InternVL3.5, and OctoMed are wired for vLLM inference and constrained decoding. For other models, check adapter support in `src/models/` before adding to config.

---

## Tumor-board questions (vista_bench_v1_7)

Three no-rubric board questions (`progression_assessment_answer`, `radiation_oligoprogression_answer`,
`pneumonitis_infection_answer`) read straight from `vista_bench_v1_7.tb_classification_tasks` (registry
`"loader": "bq_table"`, per-task `dataset`), with prediction time a week before the board. Arms:
`no_image`, `axial_all_image`, `no_notes` (EHR without clinical notes), `summary` and `summary_image`
(the board-time LLM summary instead of the EHR), plus `no_image_uncut` (no EHR cut; Gemini only, test
split only via config `arm_splits`). Progression's CT arms also get the **prior CT** (30 + 30 slices)
from `vista_bench_v1_7.per_ct_prior_ct_link`, joined by current accession for any registry task with a
`prior_ct` entry. Registry and prompts: `{base_dir}/tasks/valid_tasks.tb_v1_7.json`,
`prompts_by_task.tb_v1_7.json` (the `tb_v1_6` files are what the first pilot ran).

```bash
cd src && python ../scripts/vlm_preflight.py --config ../configs/all_tasks.tb.gemini.yaml   # no model, checks inputs
cd src && python -m vista_run.run_bq --config ../configs/all_tasks.tb.gemini_pilot.yaml --type vertex_gemini --name gemini-3.1-pro-preview
cd src && python -m results.final_metrics --tb --config ../configs/all_tasks.tb.gemini.yaml   # {results_dir}/tb_scores.csv + tb_scores_matched_comparisons.csv
```

Beside each results CSV: `_overflow_` (prompts over the model's token limit, with their count),
`_missing_` (rows the arm could not be built for, with the reason) and `_errors_` (failed calls,
retried on the next run; a CT that can't be read is listed there too, never answered text-only). Gemini
answers are cached by exact request (only readable ones), and `vertex.max_spend_usd` is a hard cap on new
spend per run, counting calls in flight. Scores carry class-stratified 95% bootstrap intervals and an
always-majority baseline row; the matched-comparisons file gives each arm minus the trimmed EHR (and
summary + CT minus summary) on the same rows. Results hold per-patient answers and stay on the mount.

## Radiology progression vs the prior CT (Codex)

The per-CT progression task (`vista_bench_v1_7.radiology_progression_tasks`: stable / progression /
response / mixed against the prior CT) asked of GPT-5.6 Sol through the Codex CLI (`--type codex_exec`).
Image-only arms, never the scan's report or EHR (the label was read from that report): `ct_pair` (prior
then current CT) and `ct_current`. Every slice up to 250 per scan, native 512 px, chest windows as RGB,
sent as JPEG 95. The task's class names come from vista_bench's registry through vista-eval's
`vistabench-eval` package (registry field `shared_task`), and rows key on the scan, not the patient.
Plan: `planning/vista_eval_vlm/vlm-radiology-progression-codex.md` (on the mount).

Each call is sealed: Codex's tools are off (shell, web search, apps, browser, plugins, memories) and it
runs under bubblewrap with the data mount, code folders and cloud credentials hidden. Needs `codex`
(signed in) and `bwrap` on the box, and an env with `vistabench-eval` installed (see the plan).

```bash
.venv/bin/python scripts/codex_seal_check.py                      # proves the call can't see the mount
.venv/bin/python scripts/radprog_preflight.py --config configs/radiology_progression.codex_pilot.yaml   # no model
.venv/bin/python scripts/codex_synthetic_smoke.py --config configs/radiology_progression.codex_pilot.yaml
cd src && ../.venv/bin/python -m vista_run.run_bq --config ../configs/radiology_progression.codex_pilot.yaml --type codex_exec --name gpt-5.6-sol
cd src && ../.venv/bin/python -m results.final_metrics --tb --config ../configs/radiology_progression.codex_pilot.yaml
cd src && ../.venv/bin/python -m results.final_metrics --probe --config ../configs/radiology_progression.codex_pilot.yaml
```

**GPT-6 Astra on the Stanford Health Care AI Hub** (`--type aihub_openai`, key `AI_HUB_API_KEY` in the
environment or `~/.config/vista/aihub.env`): Azure takes at most 50 images per request, so the first look
sends 25 + 25 slices, one scan per class, paired arm only. From phil-sllm-01 the hub's firewall rejects any
request over ~48 KB (about two slices); `scripts/aihub_body_cap_probe.py` measures that limit from the box
you are on, synthetic images only. `bash scripts/setup_api_env.sh` builds the light env (`.venv-api`).

```bash
.venv-api/bin/python scripts/aihub_body_cap_probe.py
cd src && ../.venv-api/bin/python -m vista_run.run_bq --config ../configs/radiology_progression.astra_first_look.yaml --type aihub_openai --name gpt-6-astra
```

Answers are cached by exact request on the mount, so the full run (`radiology_progression.codex.yaml`,
100 per class) reuses the pilot's 10 per class. `codex.max_total_tokens` caps new tokens per run; hitting
the subscription's usage limit stops new calls and a rerun picks up the rest.

## Results

Output CSVs are written under:

`{results_dir}/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`

Resume is by row `index` (except for retrieval experiments, which overwrite). See [docs/04-running-the-pipeline.md](docs/04-running-the-pipeline.md) for details.
