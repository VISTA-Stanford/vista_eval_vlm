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

**Note:** Gemma3, Qwen3-VL, InternVL3.5, and OctoMed are wired for vLLM inference and constrained decoding. For other models, check adapter support in `src/models/` before adding to config.

---

## Results

Output CSVs are written under:

`{results_dir}/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`

Resume is by row `index` (except for retrieval experiments, which overwrite). See [docs/04-running-the-pipeline.md](docs/04-running-the-pipeline.md) for details.
