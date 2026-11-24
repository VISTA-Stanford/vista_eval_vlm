#!/usr/bin/env bash
# Oracle-examples Half A on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md,
# Approved 2026-11-24). The knowledge-base tester (round-4 library, program-v8, the forced-image words) with each scan's
# report-matched sets (kb_oracle.py) in place of the similarity picks, on the 190 VAL scans, in two arms: grid, zoom.
#   1. setup: host, engine pin, Vertex login, disk; the pruner; the sets (kb_oracle.py build) if not built
#   2. per arm: cost check on 10 scans (stops if a scan costs over MAX_PER_SCAN, fewer than half the 10 opened 3+ of
#      their examples, or the projection passes CAP_A_USD), then the 190
#   3. $0: the paired comparison against the forced-similarity, forced-empty and pilot arms, and the two-target interval
# Status line and results in ct-findings-5-v1_7-runs/knowledge-base/oracle/ on the mount. Re-running resumes.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/oracle-examples:optimizer/ct-findings-5/scripts/run_oracle.sh > ~/run_oracle.sh && tmux new -d -s kb-oracle 'bash ~/run_oracle.sh 2>&1 | tee -a ~/kb-oracle.log'
#
# Settings, optional, on the same line: CAP_A_USD (60: Half A's share of the $90; Half B and the report pass take the
# rest), MAX_PER_SCAN (0.40), PROGRAM_MAX_CONCURRENT (3), ARMS ("grid zoom").
set -euo pipefail

BRANCH=feat/oracle-examples
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-oracle-run}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/knowledge-base/oracle
CAP_A_USD=${CAP_A_USD:-60}
MAX_PER_SCAN=${MAX_PER_SCAN:-0.40}
ARMS=${ARMS:-"grid zoom"}
TAG=program-v8  # the round-4 library with the forced-image words (forced re-test, 2026-11-24)
export KB_ARM=1
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
WORK=$HOME/.cache/alo-tasks/ct-findings-5-kb
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
# Helpers run with the prestage switch off (config refuses KB_PRESTAGE=1 without a tag); only val() turns it on.
py() { (cd "$TASK_DIR" && KB_PRESTAGE=0 uv run -q --project "$ENGINE_WT" --with pillow --with numpy --with nibabel --with pandas python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
ledger_answers() {
    python3 -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1])]; e=[x for x in e if x['tag']==sys.argv[2] and x['n']==int(sys.argv[3])]; print(e[-1]['answers'])" \
        "$WORK/private/val_pass_ledger.jsonl" "$TAG" "$1"
}
v8_spent() {  # every program-v8 pass this work folder paid for, from its receipts
    py scripts/cost_report.py "$WORK" "$OUT/cost-work-folder.json" > /dev/null || true
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(sum(p['pass_usd'] for p in d['passes'] if p['pass'].startswith('val-pass-$TAG-')), 4))" "$OUT/cost-work-folder.json"
}

status running "setup on $(hostname)"
git -C "$VLM" fetch -q origin
if [ -d "$TASK_WT" ]; then
    [ -z "$(git -C "$TASK_WT" status --porcelain)" ] || fail "$TASK_WT has uncommitted changes"
    git -C "$TASK_WT" checkout -q --detach "origin/$BRANCH"
else
    git -C "$VLM" worktree add -q --detach "$TASK_WT" "origin/$BRANCH"
fi
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$TASK_DIR/config.py")
[ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || fail "engine at $ENGINE_WT is not at the pin $PIN"
[ "$ALO_ROUTE" != vertex ] || [ -e /home/optimizer/.config/gcloud/application_default_credentials.json ] || fail "no optimizer login for Vertex"
free_gb=$(df -BG --output=avail "$HOME/.cache" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "${MIN_FREE_GB:-30}" ] || fail "only ${free_gb} GB free under ~/.cache"
bash "$TASK_DIR/scripts/prune_kb_copies.sh" > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT
[ -d "$WORK/versions/val-pass-$TAG/library" ] || fail "no written-out $TAG (run the forced re-test first)"
SETS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-private/kb_oracle/sets.json
if [ ! -f "$SETS" ]; then
    status running "building the oracle sets"
    py kb_oracle.py build "$WORK/versions/val-pass-$TAG/library" > "$OUT/sets_summary.json" || fail "set build failed"
fi
[ -f "$OUT/base_spent.txt" ] || v8_spent > "$OUT/base_spent.txt"  # earlier program-v8 passes (the forced arm), not ours
spent_new() { python3 -c "print(round($(v8_spent) - $(cat "$OUT/base_spent.txt"), 2))"; }

val() {  # val ARM [LIMIT]
    local arm=$1 limit=${2:-} ok=0
    for attempt in 1 2 3 4 5; do
        if (cd "$TASK_DIR" && KB_PRESTAGE=1 KB_PRESTAGE_TAG=$TAG KB_ORACLE=$arm uv run -q --project "$ENGINE_WT" --with pillow --with numpy --with nibabel python run.py val-pass --tag "$TAG" ${limit:+--limit "$limit"}); then ok=1; break; fi
        echo "VAL $arm attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    [ "$ok" = 1 ]
}

for arm in $ARMS; do
    if [ -f "$OUT/val-$arm.json" ]; then echo "arm $arm already done"; continue; fi
    if [ ! -f "$OUT/costcheck-$arm.ok" ]; then
        status running "arm $arm: cost check on 10 VAL scans"
        before=$(spent_new)
        val "$arm" 10 > "$OUT/costcheck-$arm.log" 2>&1 || fail "arm $arm cost check failed (costcheck-$arm.log)"
        py scripts/run_kb_loop.py usage "$(ledger_answers 10)" --out "$OUT/usage-costcheck-$arm.json" > /dev/null
        per=$(python3 -c "print(round(($(spent_new) - $before) / 10, 4))")
        read3=$(json_get "$OUT/usage-costcheck-$arm.json" "d['share_read_3_or_more_staged']")
        echo "arm $arm cost check: \$$per a scan; opened 3+ example images: $read3"
        check "$per <= $MAX_PER_SCAN" || fail "arm $arm costs \$$per a scan, above \$$MAX_PER_SCAN"
        check "$read3 >= 0.5" || fail "arm $arm: only $read3 of the cost-check sessions opened 3+ example images"
        n_left=0; for a in $ARMS; do [ -f "$OUT/val-$a.json" ] || n_left=$((n_left + 190)); done  # arms still to run
        check "$(spent_new) + 1.15 * $per * $n_left <= $CAP_A_USD" || fail "arm $arm: projection passes the \$$CAP_A_USD cap"
        echo "per_scan $per read3 $read3" > "$OUT/costcheck-$arm.ok"
    fi
    status running "arm $arm: VAL on the 190"
    val "$arm" > "$OUT/val-$arm.log" 2>&1 || fail "arm $arm VAL did not finish (val-$arm.log)"
    ans=$(ledger_answers 190)
    py scripts/run_kb_loop.py usage "$ans" --out "$OUT/usage-val-$arm.json" > /dev/null
    cp "$ans" "$OUT/val-$arm.json.tmp" && mv "$OUT/val-$arm.json.tmp" "$OUT/val-$arm.json"
    echo "arm $arm done; Half A so far \$$(spent_new)"
done

status running "comparison"
K=$RUNS/knowledge-base
py scripts/retrieval_usage.py arms --pass "oracle-grid=$OUT/val-grid.json" --pass "oracle-zoom=$OUT/val-zoom.json" \
    --pass "forced-library=$K/forced/val-library.json" --pass "forced-empty=$K/forced/val-empty.json" \
    --pass "pilot-empty=$K/val-empty.json" --pair "oracle-grid:forced-library" --pair "oracle-zoom:forced-library" \
    --pair "oracle-zoom:oracle-grid" --pair "oracle-grid:pilot-empty" --pair "oracle-zoom:pilot-empty" --out "$OUT/arms" \
    > "$OUT/arms.log" 2>&1 || fail "paired comparison failed (arms.log)"
py scripts/target_pairs.py --out "$OUT/targets.json" "oracle-grid=$OUT/val-grid.json" "oracle-zoom=$OUT/val-zoom.json" \
    "forced-library=$K/forced/val-library.json" "forced-empty=$K/forced/val-empty.json" "pilot-empty=$K/val-empty.json" \
    > "$OUT/targets.log" 2>&1 || fail "two-target intervals failed (targets.log)"
status done "Half A finished on $(hostname): \$$(spent_new); results in knowledge-base/oracle/ (arms/, targets.json)"
