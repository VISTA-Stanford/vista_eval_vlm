#!/usr/bin/env bash
# Light env for the API-model runs (Codex, AI Hub, Vertex): CPU torch, the data libraries, and
# vista-eval's vistabench-eval (the shared task registry). No vLLM, no GPU. Built in its own folder
# (default .venv-api) so it never replaces a checkout's GPU env.
#   bash scripts/setup_api_env.sh            -> .venv-api
#   VENV=.venv bash scripts/setup_api_env.sh -> another folder
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv-api}"
VISTA_EVAL_REF=79688c2654e75846f3dd66e3aca578af9b7db69d
uv venv -q --clear --python 3.11 "$VENV"
uv pip install -q --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
# pyarrow < 20: newer wheels need numpy 2, and the repo pins numpy 1.26.
uv pip install -q --python "$VENV/bin/python" google-cloud-bigquery google-cloud-storage db-dtypes \
  "indexed_gzip==1.10.3" nibabel "numpy==1.26.4" "pandas==2.3.3" "pillow==12.0.0" "pyyaml==6.0.3" \
  tqdm requests google-auth pytest "pyarrow<20" \
  "vistabench-eval @ git+ssh://git@github.com/VISTA-Stanford/vista-eval.git@${VISTA_EVAL_REF}#subdirectory=packages/vistabench_eval"
"$VENV/bin/python" -c "from vistabench_eval import tasks; assert tasks.inputs_for_task('radiology_progression') == 'current_and_prior'; print('api env ok:', '$VENV')"
