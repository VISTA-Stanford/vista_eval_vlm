#!/usr/bin/env bash
#
# Rung-0 weighted 0b run — reproduce Ryan's PFS pipeline on the feb26 CT snapshot.
#
# WEIGHTED run (loads medgemma weights, emits result CSVs). GPU box ONLY — phil-sllm-01
# is CPU-only, which is why 0b was backlogged here. See:
#   docs/vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md   (readback + 0b/0c recipe)
#   configs/all_tasks.rung0.yaml                              (the config this drives)
#
# Usage (GPU box, repo on branch worktree-vlm-modular-preprocessing-roadmap):
#   HF_TOKEN=hf_xxx bash eval/run_rung0_gpu.sh
#
# Optional env overrides:
#   VENV    virtualenv to `source $VENV/bin/activate` (default: repo .venv if present)
#   CONFIG  config path (default: configs/all_tasks.rung0.yaml)
#   HF_TOKEN  HuggingFace token for the gated medgemma weights (skip if already `hf auth login`'d)
#
# The preflight halts BEFORE any GPU/weight spend if the box isn't ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/all_tasks.rung0.yaml}"
# Resolve to absolute so it works regardless of cwd (we later cd into src/).
case "$CONFIG" in /*) : ;; *) CONFIG="$REPO_ROOT/$CONFIG" ;; esac

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }

# --- activate env ---
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== rung-0 preflight (halts before GPU spend if the box isn't ready) ==="

# 1. GPU present — the whole reason 0b moved off phil-sllm-01.
command -v nvidia-smi >/dev/null 2>&1 \
    || fail "no nvidia-smi — this box has no GPU. 0b is the WEIGHTED run; it needs one."
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')" || GPU_COUNT=0
[ "${GPU_COUNT:-0}" -ge 1 ] || fail "nvidia-smi found 0 GPUs."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || fail "torch.cuda.is_available()==False (driver/torch mismatch)."
echo "  [ok] GPU(s): $GPU_COUNT; torch.cuda available"

# 2. config present + invariants (force-GCS / feb26 / un-subsampled / constrained).
[ -f "$CONFIG" ] || fail "config not found: $CONFIG"
python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
p = cfg.get("paths", {})
assert not p.get("ct_dir"), "paths.ct_dir is set -> breaks force-GCS (feb26 reroute must go to GCS)."
assert p.get("ct_snapshot_prefix", "").endswith("feb26"), "ct_snapshot_prefix is not feb26."
assert cfg.get("subsample") is False, "subsample must be false (declared delta #2)."
assert cfg.get("runtime", {}).get("use_constrained_decoding_for_binary") is True, \
    "runtime.use_constrained_decoding_for_binary must be true (OQ-R6)."
print("  [ok] config invariants: ct_dir unset, feb26 prefix, subsample=false, constrained=true")
PY

# 3. base_dir mount + task registry readable.
BASE_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['base_dir'])")"
[ -d "$BASE_DIR" ] || fail "base_dir not mounted/readable: $BASE_DIR"
[ -f "$BASE_DIR/tasks/valid_tasks.json" ] || fail "missing $BASE_DIR/tasks/valid_tasks.json"
echo "  [ok] base_dir mounted: $BASE_DIR"

# 4. HF auth — medgemma weights are gated.
python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1 \
    || fail "HuggingFace not authenticated — set HF_TOKEN or run 'hf auth login' (medgemma is gated)."
echo "  [ok] HuggingFace authenticated"

# --- run (single model, pinned to the config's models[0] so there is one source of truth) ---
read -r MODEL_TYPE MODEL_NAME < <(python -c "import yaml; m=yaml.safe_load(open('$CONFIG'))['models'][0]; print(m['type'], m['name'])")

echo "=== preflight PASSED — starting weighted run ==="
LOG="$REPO_ROOT/logs/rung0_$(date +%Y%m%d_%H%M%S)/run_bq.log"
mkdir -p "$(dirname "$LOG")"
echo "Config: $CONFIG"
echo "Model:  $MODEL_TYPE / $MODEL_NAME"
echo "Log:    $LOG"

cd "$REPO_ROOT/src"
python -m vista_run.run_bq --config "$CONFIG" --type "$MODEL_TYPE" --name "$MODEL_NAME" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
cd "$REPO_ROOT"

[ "$STATUS" -eq 0 ] || { echo "run_bq FAILED (exit $STATUS). See $LOG" >&2; exit "$STATUS"; }

echo ""
echo "=== weighted run complete — sanity-check before declaring 0b green ==="
echo "  1. grep -c 'source=local' \"$LOG\"    # MUST be 0 (any local = force-GCS breach)"
echo "  2. grep -c 'source=gcs'   \"$LOG\"    # expect all CT rows"
echo "  3. result CSVs non-empty at {results_dir}/.../{task}_results_{no_image,axial_all_image}.csv"
echo "Then 0c (report, CPU-fine): cd src && python -m results.all_model_response"
