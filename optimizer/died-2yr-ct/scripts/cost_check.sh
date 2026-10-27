#!/usr/bin/env bash
# The 5+5 cost check for died-2yr-ct: one iteration (the seed on 5 TRAIN patients, one optimizer round, the new
# version on 5 VAL patients) in a fresh throwaway work folder, never the real one. Paid: about $5, on the task's route
# (Vertex by default; ALO_ROUTE=anthropic for the seat). Needs Phil's go. Judges itself with scripts/cost_report.py.
#
#   bash optimizer/died-2yr-ct/scripts/cost_check.sh
#
# Pass: it ends "PASS: ..." (fewer than 1 in 20 sessions at the $0.50 cap, no unreadable answer, the run finished),
# and the verdict is on the mount at died-2yr-ct-v1_7-runs/<host>/costcheck-<stamp>/cost_report.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE=${AGENTIC_LABEL_OPT:-$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine}
STAMP=$(date +%m%d-%H%M)
export AGENTIC_LABEL_OPT=$ENGINE
export TASK_WORKDIR=$HOME/.cache/alo-tasks/died-2yr-ct-costcheck-$STAMP
export TASK_SCHEDULE='{"train_size": "constant(5)", "val_size": "constant(5)"}'
export ITERATIONS=1
OUT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7-runs/$(hostname)/costcheck-$STAMP
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$HERE/config.py")
[ "$(git -C "$ENGINE" rev-parse HEAD)" = "$PIN" ] || { echo "FAIL: engine at $ENGINE is not at the pin $PIN"; exit 1; }
cd "$HERE"
alo() { uv run -q --project "$ENGINE" python run.py "$@"; }
alo init
alo pins --write | tail -1
alo run --run-id costcheck --start seed --iterations 1 || true  # judged below, from the receipts and the summary
uv run -q --project "$ENGINE" python scripts/cost_report.py "$TASK_WORKDIR" "$OUT/cost_report.json"
