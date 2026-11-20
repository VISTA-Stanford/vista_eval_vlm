#!/usr/bin/env bash
# The 5+5 cost check for one tb-questions agent: one iteration (the seed on 5 TRAIN patients, one optimizer round, the
# new version on 5 VAL patients) in a fresh throwaway work folder, never the real one. Paid: a few dollars, on Vertex
# (the spec's route). Approved in the spec (hand-off 3: needs its own go only if it extrapolates past the caps).
# Copied from ct-findings-5's (2026-11-20). Judges itself with scripts/cost_report.py, which also extrapolates a
# round and the zero-shot VAL pass from the sessions' costs and fails if either passes the caps.
#
#   TB_AGENT=treatment bash optimizer/tb-questions/scripts/cost_check.sh
#   TB_AGENT=imaging bash optimizer/tb-questions/scripts/cost_check.sh
#
# Then the engine's after-run audit (scripts/audit_vertex_run.py) over the throwaway folder: no session command naming
# the credential, gcloud, bq, BigQuery, storage or a non-model Vertex request, and no refused or unexpected host in the
# saved network logs. Pass: it ends "PASS: ...", and the verdict and audit.txt are on the mount at
# tb-questions-v1_7-runs/<host>/<agent>/costcheck-<stamp>/.
set -euo pipefail
: "${TB_AGENT:?set TB_AGENT=treatment or TB_AGENT=imaging}"
export TB_AGENT
export ALO_ROUTE=${ALO_ROUTE:-vertex}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE=${AGENTIC_LABEL_OPT:-$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine}
STAMP=$(date +%m%d-%H%M)
export AGENTIC_LABEL_OPT=$ENGINE
export TASK_WORKDIR=$HOME/.cache/alo-tasks/tb-questions-$TB_AGENT-costcheck-$STAMP
export TASK_SCHEDULE='{"train_size": "constant(5)", "val_size": "constant(5)"}'
export ITERATIONS=1
OUT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/tb-questions-v1_7-runs/$(hostname)/$TB_AGENT/costcheck-$STAMP
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$HERE/config.py")
[ "$(git -C "$ENGINE" rev-parse HEAD)" = "$PIN" ] || { echo "FAIL: engine at $ENGINE is not at the pin $PIN"; exit 1; }
cd "$HERE"
alo() { uv run -q --project "$ENGINE" python run.py "$@"; }
docker image inspect "alo-tb-questions-$TB_AGENT-optimizer:$(sed -n 's/^CLAUDE_CODE_VERSION = "\([0-9.]*\)".*/\1/p' config.py)" >/dev/null 2>&1 \
    || alo build-images >/dev/null
alo init
alo pins --write | tail -1
alo run --run-id costcheck --start seed --iterations 1 || true  # judged below, from the receipts and the summary
mkdir -p "$OUT"
report=0
uv run -q --project "$ENGINE" python scripts/cost_report.py "$TASK_WORKDIR" "$OUT/cost_report.json" || report=$?
audit=0
python3 -I "$ENGINE/scripts/audit_vertex_run.py" "$TASK_WORKDIR" --route "$ALO_ROUTE" --skip private/archive \
    ${AUDIT_PRINCIPAL:+--principal "$AUDIT_PRINCIPAL"} | tee "$OUT/audit.txt" || audit=$?
if [ "$report" -eq 0 ] && [ "$audit" -eq 0 ]; then
    echo "PASS: the cost check and the after-run audit both passed ($TB_AGENT, route $ALO_ROUTE)"
else
    echo "FAIL: cost check exit $report, after-run audit exit $audit (0 pass, 1 found something, 2 incomplete); see above"
    exit 1
fi
