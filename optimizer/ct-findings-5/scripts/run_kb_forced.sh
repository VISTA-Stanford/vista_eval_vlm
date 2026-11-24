#!/usr/bin/env bash
# The knowledge-base forced-image re-test on phil-sllm-02, Vertex (Phil 2026-11-24, after the pilot's tester opened the
# saved example image in only 61 of 381 opens). Round 4's library (pilot program-v5) and the empty library, both with
# the new tester words (examples pre-drawn into examples/ and every image must be opened), on the same 190 VAL scans.
#   1. setup: host, engine pin, Vertex login, disk; the pruner
#   2. the two versions (scripts/run_kb_loop.py retag program-v0 program-v5)
#   3. cost check: the library arm on 10 VAL scans. Stops if a scan costs more than MAX_PER_SCAN, fewer than half the 10
#      opened 3 or more example images, or the projection (2 x 190 scans) would pass CAP_USD with the pilot's spend
#   4. VAL: empty library, then the library, one pass each
#   5. $0: library use, the paired comparison, the fusion
# Status line and results in ct-findings-5-v1_7-runs/knowledge-base/forced/ on the mount. Re-running resumes.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-knowledge-base:optimizer/ct-findings-5/scripts/run_kb_forced.sh > ~/run_kb_forced.sh && tmux new -d -s kb-forced 'bash ~/run_kb_forced.sh 2>&1 | tee -a ~/kb-forced.log'
#
# Settings, optional, on the same line: CAP_USD (200, pilot included), MAX_PER_SCAN (0.40), PROGRAM_MAX_CONCURRENT (3).
set -euo pipefail

BRANCH=feat/ct-knowledge-base
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-kb-run}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/knowledge-base/forced
CAP_USD=${CAP_USD:-200}
MAX_PER_SCAN=${MAX_PER_SCAN:-0.40}
PILOT_SPENT=${PILOT_SPENT:-107.20}  # the pilot's total (knowledge-base/STATUS.txt, 2026-11-23)
export KB_ARM=1 KB_PRESTAGE=1
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
WORK=$HOME/.cache/alo-tasks/ct-findings-5-kb
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
# Helpers run with the switch off: config refuses KB_PRESTAGE=1 without KB_PRESTAGE_TAG, and only val() sets the tag
# (it killed the first launch on 2026-11-24, after the cost-check pass had finished).
py() { (cd "$TASK_DIR" && KB_PRESTAGE=0 uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
ledger_answers() {  # the latest val-pass answers file for a tag and size
    python3 -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1])]; e=[x for x in e if x['tag']==sys.argv[2] and x['n']==int(sys.argv[3])]; print(e[-1]['answers'])" \
        "$WORK/private/val_pass_ledger.jsonl" "$1" "$2"
}
spent_new() {  # tester sessions of this re-test's passes, from the work folder's receipts (pass names val-pass-<tag>-...)
    py scripts/cost_report.py "$WORK" "$OUT/cost-work-folder.json" > /dev/null || true
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(sum(p['pass_usd'] for p in d['passes'] if any(p['pass'].startswith('val-pass-'+t+'-') for t in sys.argv[2:])), 2))" \
        "$OUT/cost-work-folder.json" "$EMPTY" "$LIB"
}

status running "setup on $(hostname)"
[ "$(hostname)" = phil-sllm-02 ] || echo "note: planned for phil-sllm-02; running on $(hostname)"
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
[ -d "$WORK/repo" ] || fail "no KB work folder at $WORK (the pilot's)"
bash "$TASK_DIR/scripts/prune_kb_copies.sh" > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

status running "the two versions: pilot's empty library and round-4 library with the new tester words"
py scripts/run_kb_loop.py retag program-v0 program-v5 > "$OUT/retag.json" || fail "retag failed"
EMPTY=$(json_get "$OUT/retag.json" "d['program-v0']")
LIB=$(json_get "$OUT/retag.json" "d['program-v5']")
echo "empty library $EMPTY, library $LIB"

val() {  # val TAG [LIMIT]: one VAL pass with the examples drawn from TAG's library; retries a network stop
    local tag=$1 limit=${2:-} ok=0
    for attempt in 1 2 3 4 5; do
        if (cd "$TASK_DIR" && KB_PRESTAGE_TAG=$tag uv run -q --project "$ENGINE_WT" --with pillow --with numpy python run.py val-pass --tag "$tag" ${limit:+--limit "$limit"}); then ok=1; break; fi
        echo "VAL $tag attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    [ "$ok" = 1 ]
}

# 3. cost check
if [ -f "$OUT/costcheck.ok" ]; then
    echo "cost check already passed: $(cat "$OUT/costcheck.ok")"
else
    status running "cost check: library arm on 10 VAL scans"
    val "$LIB" 10 > "$OUT/costcheck.log" 2>&1 || fail "cost check pass failed (costcheck.log)"
    ans=$(ledger_answers "$LIB" 10)
    py scripts/run_kb_loop.py usage "$ans" --out "$OUT/usage-costcheck.json" > /dev/null
    t=$(spent_new)
    per=$(python3 -c "print(round($t / 10, 4))")
    read3=$(json_get "$OUT/usage-costcheck.json" "d['share_read_3_or_more_staged']")
    echo "cost check: \$$per a scan; opened 3+ example images: $read3"
    check "$per <= $MAX_PER_SCAN" || fail "costs \$$per a scan, above the \$$MAX_PER_SCAN stop"
    check "$read3 >= 0.5" || fail "only $read3 of the cost-check sessions opened 3 or more example images (they still skip them)"
    proj=$(python3 -c "print(round($PILOT_SPENT + $t + 1.15 * $per * 380, 2))")
    check "$proj <= $CAP_USD" || fail "projected \$$proj passes the \$$CAP_USD cap"
    echo "per_scan $per read3 $read3 projected $proj" > "$OUT/costcheck.ok"
fi

# 4. VAL, both arms
for arm in empty library; do
    tag=$EMPTY; [ "$arm" = library ] && tag=$LIB
    if [ -f "$OUT/val-$arm.json" ]; then echo "VAL $arm already done"; continue; fi
    status running "VAL $arm library ($tag) on the 190, examples pre-drawn"
    val "$tag" > "$OUT/val-$arm.log" 2>&1 || fail "VAL $arm did not finish after 5 attempts (val-$arm.log)"
    ans=$(ledger_answers "$tag" 190)
    py scripts/run_kb_loop.py usage "$ans" --out "$OUT/usage-val-$arm.json" > /dev/null
    cp "$ans" "$OUT/val-$arm.json.tmp" && mv "$OUT/val-$arm.json.tmp" "$OUT/val-$arm.json"
    echo "VAL $arm done; this re-test so far \$$(spent_new)"
done

# 5. $0: comparison and fusion
status running "comparison and fusion"
py scripts/retrieval_usage.py arms --pass "forced-library=$OUT/val-library.json" --pass "forced-empty=$OUT/val-empty.json" \
    --pass "pilot-library=$RUNS/knowledge-base/val-library.json" --pass "pilot-empty=$RUNS/knowledge-base/val-empty.json" \
    --pass "arm-D=$RUNS/example-ct-arms/arm-d.json" --pair "forced-library:forced-empty" --pair "forced-library:pilot-library" \
    --pair "forced-empty:pilot-empty" --out "$OUT/arms" > "$OUT/arms.log" 2>&1 || fail "paired comparison failed (arms.log)"
(cd "$TASK_DIR" && OMP_NUM_THREADS=4 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with numpy \
    --with scikit-learn python -B scripts/step0_curves.py --kb-forced) > "$OUT/fusion.log" 2>&1 || fail "fusion failed (fusion.log)"
status done "forced-image re-test finished on $(hostname): this re-test \$$(spent_new), pilot \$$PILOT_SPENT; results in knowledge-base/forced/"
