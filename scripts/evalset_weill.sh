#!/usr/bin/env bash
# Eval-set run on weill-p01 for the models that run there: MedGemma 1.5 4B (vLLM, one GPU) and GPT-6 Astra
# (SHC AI Hub; the hub blocks large requests from phil-sllm-01). Reads the samples and image packs that
# phil-sllm-01 rendered to the mount, writes answers to <mount>/vistabench/vlm/evalset_v1/runs/<RUN>/<recipe>/,
# then checks every question was answered and parsed (exit 0 = pass).
#
#   RECIPE=medgemma RUN=check5 LIMIT=5 bash scripts/evalset_weill.sh
#   RECIPE=astra RUN=check5 LIMIT=5 MAX_SPEND=30 bash scripts/evalset_weill.sh
#
# Env: RECIPE (medgemma | astra), RUN (run name), LIMIT (cases per step; unset = the whole sample),
# STEPS (default "1 2 3 4"), QUESTIONS (space-separated question ids; default all), MAX_SPEND (USD cap, Astra), VISTA_VENV (default .venv for medgemma, .venv-api
# for astra), CUDA_VISIBLE_DEVICES (MedGemma's GPU, default 0), EVALSET_MOUNT_ROOT (default weill's).
# Spec: planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md (on the mount).
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE="${RECIPE:?set RECIPE=medgemma or RECIPE=astra}"
RUN="${RUN:?set RUN, e.g. check5}"
STEPS="${STEPS:-1 2 3 4}"
export EVALSET_MOUNT_ROOT="${EVALSET_MOUNT_ROOT:-/data/fries/users/padamson/mnt}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
case "$RECIPE" in
  medgemma) VENV="${VISTA_VENV:-.venv}"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ;;
  astra) VENV="${VISTA_VENV:-.venv-api}" ;;
  *) echo "ERROR: RECIPE must be medgemma or astra (argon runs on phil-sllm-01)"; exit 2 ;;
esac
PY="$PWD/$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no python at $PY (medgemma: bash scripts/setup.sh; astra: bash scripts/setup_api_env.sh)"; exit 2; }
"$PY" -c "import pandas, pyarrow, PIL, requests" || { echo "ERROR: $VENV lacks pandas/pyarrow/pillow/requests"; exit 2; }
[ "$RECIPE" = medgemma ] && { "$PY" -c "import vllm" || { echo "ERROR: $VENV has no vllm"; exit 2; }; }
echo "host $(hostname)  commit $(git log -1 --format=%h)  recipe $RECIPE  run $RUN  steps $STEPS  limit ${LIMIT:-all}"
cd src
LIMIT_ARG=(); [ -n "${LIMIT:-}" ] && LIMIT_ARG=(--limit "$LIMIT")
SPEND_ARG=(); [ -n "${MAX_SPEND:-}" ] && SPEND_ARG=(--max-spend "$MAX_SPEND")
Q_ARG=(); [ -n "${QUESTIONS:-}" ] && read -r -a QS <<< "$QUESTIONS" && Q_ARG=(--questions "${QS[@]}")
# shellcheck disable=SC2086
"$PY" -m evalset.run --recipe "$RECIPE" --step $STEPS --run "$RUN" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} ${SPEND_ARG[@]+"${SPEND_ARG[@]}"} ${Q_ARG[@]+"${Q_ARG[@]}"}
# shellcheck disable=SC2086
"$PY" -m evalset.check --recipe "$RECIPE" --run "$RUN" --steps $STEPS ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} ${Q_ARG[@]+"${Q_ARG[@]}"}
