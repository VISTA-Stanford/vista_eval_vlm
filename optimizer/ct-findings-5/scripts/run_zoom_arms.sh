#!/usr/bin/env bash
# The organ-zoom arms on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-24-agent-organ-zoom.md, Approved
# 2026-11-24; these paid steps are its approved Expensive steps). Copied from scripts/run_example_arms.sh.
#   1. setup: host, engine pin, Vertex login, disk; the masks' $0 check (scripts/build_masks.py check); the pruner
#   2. cost check: each arm on the first 10 of the 190 VAL scans (scripts/cost_check.sh). Stops on a failed check, a
#      tool crash, a session error, or a projection (TRAIN check 50 + both VAL arms 190 each) over CAP_USD
#   3. TRAIN check: the tool arm on the 50 find-the-slice scans. Stops only if the tool is broken (a crash, or no
#      session drew a zoomed image); the find rate is reported
#   4. VAL: control, then tool, on the 190. A pass is not started if the spend so far plus 1.15 x its cost-check cost
#      per scan x 190 would cross CAP_USD, and is stopped (Ctrl-C to the pass) if its spend crosses CAP_USD
#   5. $0: usage per arm, the paired comparison (tool - control; three-question mean decides) beside arm D and the
#      retrieval arm
# Status line and every result in ct-findings-5-v1_7-runs/organ-zoom/ on the mount. Re-running resumes.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-organ-zoom:optimizer/ct-findings-5/scripts/run_zoom_arms.sh > ~/run_zoom_arms.sh && tmux new -d -s organ-zoom 'bash ~/run_zoom_arms.sh 2>&1 | tee -a ~/organ-zoom.log'
#
# Settings, optional, on the same line: CAP_USD (150, the spec), PROGRAM_MAX_CONCURRENT (3), MIN_FREE_GB (40).
# Pass: STATUS.txt reads "Status: done"; arms.md holds the table.
set -euo pipefail

BRANCH=feat/ct-organ-zoom
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-organ-zoom-run}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/organ-zoom
CAP_USD=${CAP_USD:-150}
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
export VAL_PASS_STOP_USD=200  # 190 scans x the $1.00 cap; the real stop is CAP_USD, enforced here
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
alo() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" python run.py "$@"); }
py() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with numpy --with nibabel python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
ledger_answers() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['answers'])" "$1"; }
spent_file=$OUT/spent_usd.txt
[ -f "$spent_file" ] || echo 0 > "$spent_file"
add_spent() { python3 -c "import sys; print(round(float(open(sys.argv[1]).read()) + float(sys.argv[2]), 2))" "$spent_file" "$1" > "$spent_file.tmp" && mv "$spent_file.tmp" "$spent_file"; }
spent() { cat "$spent_file"; }
workdir() { echo "$HOME/.cache/alo-tasks/ct-findings-5-zoom-$1"; }  # tool | control | traincheck (config.WORKDIR)

status running "setup on $(hostname)"
[ "$(hostname)" = phil-sllm-02 ] || echo "note: spec names phil-sllm-02; running on $(hostname)"
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
[ "$free_gb" -ge "${MIN_FREE_GB:-40}" ] || fail "only ${free_gb} GB free under ~/.cache"
py scripts/build_masks.py check > "$OUT/mask_check.txt" 2>&1 || fail "mask check failed (mask_check.txt)"

# Each session gets its own CT copy (~40 MB); once it has answered and is 30 minutes old (sessions time out at 30),
# nothing reads it again (a resume restages only unfinished calls). The view log and answers stay.
prune() {
    while :; do
        n=0
        for ct in $(ls -d "$HOME"/.cache/alo-tasks/ct-findings-5-zoom-*/program-outputs/*/*/calls/*/ct 2>/dev/null); do
            [ -f "$(dirname "$ct")/output.json" ] || continue
            [ -n "$(find "$ct" -maxdepth 0 -mmin +35 2>/dev/null)" ] || continue
            chmod -R u+w "$ct" 2>/dev/null; rm -rf "$ct" && n=$((n + 1))
        done
        echo "$(date -u +%FT%TZ) pruned $n CT copies; free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
        sleep 300
    done
}
prune > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

# 2. cost checks
for arm in tool control; do
    if [ -f "$OUT/costcheck-$arm.ok" ]; then echo "cost check $arm already passed: \$$(cat "$OUT/costcheck-$arm.ok") a scan"; continue; fi
    status running "cost check: $arm arm on 10 VAL scans"
    (cd "$TASK_DIR" && ZOOM_ARM=$arm bash scripts/cost_check.sh) > "$OUT/costcheck-$arm.log" 2>&1 || fail "$arm cost check failed (costcheck-$arm.log)"
    ccwd=$(ls -td "$HOME"/.cache/alo-tasks/ct-findings-5-costcheck-* | head -1)
    cc="$RUNS/$(hostname)/$(basename "$ccwd" | sed 's/^ct-findings-5-//')"
    [ -f "$cc/cost_report.json" ] || fail "$arm: no cost report at $cc"
    per=$(json_get "$cc/cost_report.json" "d['passes'][-1]['mean_usd']")
    add_spent "$(json_get "$cc/cost_report.json" "d['patient_sessions_usd']")"
    ans=$(ledger_answers "$ccwd/private/val_pass_ledger.jsonl")
    (cd "$TASK_DIR" && ZOOM_ARM=$arm uv run -q --project "$ENGINE_WT" python -B scripts/zoom_usage.py usage "$ans" --out "$OUT/usage-costcheck-$arm") > /dev/null
    find "$ccwd/program-outputs" -type d -name ct -path '*/calls/*' -exec chmod -R u+w {} + -exec rm -rf {} + 2>/dev/null || true
    crashes=$(json_get "$OUT/usage-costcheck-$arm/usage_summary.json" "d['tool_crashes']")
    errs=$(json_get "$OUT/usage-costcheck-$arm/usage_summary.json" "d['program_errors']")
    echo "cost check $arm: \$$per a scan; tool crashes $crashes; session errors $errs; spent so far \$$(spent)"
    check "$crashes == 0" || fail "$arm cost check: $crashes tool crashes (usage-costcheck-$arm)"
    check "$errs == 0" || fail "$arm cost check: $errs sessions without a readable answer"
    echo "$per" > "$OUT/costcheck-$arm.ok"
done
PER_TOOL=$(cat "$OUT/costcheck-tool.ok"); PER_CTRL=$(cat "$OUT/costcheck-control.ok")
proj=$(python3 -c "print(round($(spent) + 1.15 * ($PER_TOOL * (50 + 190) + $PER_CTRL * 190), 2))")
echo "projection: \$$proj (spent \$$(spent); tool \$$PER_TOOL, control \$$PER_CTRL a scan)" | tee "$OUT/projection.txt"
[ -f "$OUT/val-tool.json" ] || check "$proj <= $CAP_USD" || fail "projected \$$proj passes the \$$CAP_USD cap"

# A pass in the background, watched: Ctrl-C to it if this pass's spend would take the total over CAP_USD.
watched() {  # watched WORKDIR COMMAND...: run, check spend every 2 minutes; returns the command's exit code
    local wd=$1; shift
    local before; before=$(spent)
    "$@" & local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 120
        (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" python -B scripts/cost_report.py "$wd" "$wd/cost-live.json") > /dev/null 2>&1 || true
        local now; now=$(json_get "$wd/cost-live.json" "d['patient_sessions_usd']" 2>/dev/null || echo 0)
        if ! check "$before + $now <= $CAP_USD"; then
            echo "spend \$$before + \$$now crosses the \$$CAP_USD cap: stopping the pass"
            pkill -INT -P "$pid" 2>/dev/null; kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
            fail "stopped at the \$$CAP_USD cap (spent \$$before before this pass, \$$now in it)"
        fi
    done
    wait "$pid"
}
pass_with_retries() {  # pass_with_retries WORKDIR run.py-ARGS...
    local wd=$1; shift
    for attempt in 1 2 3 4 5; do
        if watched "$wd" alo "$@"; then return 0; fi
        echo "pass attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    return 1
}
setup_workdir() { [ -d "$TASK_WORKDIR/repo" ] || alo init; git -C "$TASK_WORKDIR/repo" log --format=%s | grep -q '^setup pins for' || alo pins --write | tail -1; }

# 3. TRAIN check
if [ -f "$OUT/train-check.json" ]; then echo "TRAIN check already done"; else
    status running "TRAIN find-the-slice check: tool arm on 50 scans"
    export ZOOM_ARM=tool ZOOM_TRAIN_CHECK=1 TASK_WORKDIR; TASK_WORKDIR=$(workdir traincheck)
    setup_workdir
    pass_with_retries "$TASK_WORKDIR" train-check --tag program-v0 > "$OUT/train-check.log" 2>&1 || fail "TRAIN check did not finish (train-check.log)"
    py scripts/cost_report.py "$TASK_WORKDIR" "$OUT/cost-traincheck.json" > /dev/null || true
    add_spent "$(json_get "$OUT/cost-traincheck.json" "d['patient_sessions_usd']")"
    ans=$(ledger_answers "$TASK_WORKDIR/private/train_check_ledger.jsonl")
    py scripts/zoom_usage.py usage "$ans" --train-check --out "$OUT/usage-train-check" > /dev/null
    broken=$(json_get "$OUT/usage-train-check/usage_summary.json" "d['tool_broken']")
    found=$(json_get "$OUT/usage-train-check/usage_summary.json" "d['train_check']['found_zoom']")
    echo "TRAIN check: tool broken $broken; zoomed within a slab of the cited lesion on $found of 50; spent so far \$$(spent)"
    [ "$broken" = False ] || fail "TRAIN check: the tool is broken (usage-train-check/usage_summary.json)"
    cp "$ans" "$OUT/train-check.json.tmp" && mv "$OUT/train-check.json.tmp" "$OUT/train-check.json"
    unset ZOOM_TRAIN_CHECK TASK_WORKDIR
fi

# 4. VAL
for arm in control tool; do
    if [ -f "$OUT/val-$arm.json" ]; then echo "VAL $arm already done"; continue; fi
    per=$PER_CTRL; [ "$arm" = tool ] && per=$PER_TOOL
    check "$(spent) + 1.15 * $per * 190 <= $CAP_USD" || fail "VAL $arm would cross the \$$CAP_USD cap (spent \$$(spent), \$$per a scan)"
    status running "VAL $arm arm on the 190"
    export ZOOM_ARM=$arm TASK_WORKDIR; TASK_WORKDIR=$(workdir "$arm")
    setup_workdir
    pass_with_retries "$TASK_WORKDIR" val-pass --tag program-v0 > "$OUT/val-$arm.log" 2>&1 || fail "VAL $arm did not finish (val-$arm.log)"
    py scripts/cost_report.py "$TASK_WORKDIR" "$OUT/cost-$arm.json" > /dev/null || true
    add_spent "$(json_get "$OUT/cost-$arm.json" "d['patient_sessions_usd']")"
    ans=$(ledger_answers "$TASK_WORKDIR/private/val_pass_ledger.jsonl")
    py scripts/zoom_usage.py usage "$ans" --out "$OUT/usage-val-$arm" > /dev/null
    cp "$ans" "$OUT/val-$arm.json.tmp" && mv "$OUT/val-$arm.json.tmp" "$OUT/val-$arm.json"
    unset TASK_WORKDIR
    echo "VAL $arm done; spent so far \$$(spent)"
done
unset ZOOM_ARM

# 5. $0: the paired comparison
status running "paired comparison"
py scripts/retrieval_usage.py arms --pass "zoom-tool=$OUT/val-tool.json" --pass "zoom-control=$OUT/val-control.json" \
    --pass "arm-D=$RUNS/example-ct-arms/arm-d.json" --pass "retrieval=$RUNS/retrieval-arm/retrieval.json" \
    --pair "zoom-tool:zoom-control" --pair "zoom-tool:arm-D" --pair "zoom-control:arm-D" --out "$OUT/arms" \
    > "$OUT/arms.log" 2>&1 || fail "paired comparison failed (arms.log)"
status done "organ-zoom arms finished on $(hostname): spent \$$(spent) of \$$CAP_USD; table in organ-zoom/arms/arms.md"
