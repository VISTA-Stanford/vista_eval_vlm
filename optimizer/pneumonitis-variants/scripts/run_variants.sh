#!/usr/bin/env bash
# The pneumonitis-variants study on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md,
# Approved 2026-11-24; this launcher is its approved Expensive step, <= $250 through VAL). Modelled on ct-findings-5's
# scripts/run_zoom_arms.sh. Re-running resumes: each finished step leaves a marker in OUT.
#   1. setup: host, engine pin, Vertex login, disk, the data check (scripts/build_data.py check), the probe, the pruner
#   2. zero-shot cost check: program-v0 on the first 10 VAL scans, then the engine's after-run audit. Stops on a session
#      at the cap, an unreadable answer, a failed audit, or a projection over CAP_USD
#   3. zero-shot VAL: program-v0 on all VAL scans (first, before any building: Phil 2026-11-24)
#   4. the 5-round builder loop on TRAIN (scripts/run_loop.py loop), with the guide arm's VAL pass kept back under the cap
#   5. guide-arm cost check: round 5's version on the first 10 VAL scans. Stops if fewer than half its sessions opened
#      every drawn example image (spec), or on a projection over CAP_USD
#   6. guide VAL: round 5's version on all VAL scans
#   7. $0: the paired comparison (guide - zero-shot, patient bootstrap) beside the MerlinOnc probe
# Every pass is watched: Ctrl-C to it if the study's total would cross CAP_USD. Status line and results in
# ct-findings-5-v1_7-runs/pneumonitis-variants/ on the mount.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-pneumonitis-variants:optimizer/pneumonitis-variants/scripts/run_variants.sh > ~/run_variants.sh && tmux new -d -s pneumonitis-variants 'bash ~/run_variants.sh 2>&1 | tee -a ~/pneumonitis-variants.log'
#
# Settings, optional, on the same line: CAP_USD (250, the spec), PROGRAM_MAX_CONCURRENT (3), MIN_FREE_GB (40),
# BEFORE_USD (3.74, the infection pass). Pass: STATUS.txt reads "Status: done"; comparison/table.md holds the table.
set -euo pipefail

BRANCH=feat/ct-pneumonitis-variants
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-pneumonitis-run}
TASK_DIR=$TASK_WT/optimizer/pneumonitis-variants
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
OUT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs/pneumonitis-variants
WORK=$HOME/.cache/alo-tasks/pneumonitis-variants  # config.WORKDIR
CAP_USD=${CAP_USD:-250}
BEFORE_USD=${BEFORE_USD:-3.74}
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
export VAL_PASS_STOP_USD=110  # 134 scans x the $0.75 cap; the real stop is CAP_USD, enforced here
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
alo() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python run.py "$@"); }
loop() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B scripts/run_loop.py "$@"); }
py() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
ledger_answers() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['answers'])" "$1"; }
ledger_look() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['look'])" "$1"; }
total() { python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" --before "$BEFORE_USD" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_usd'])"; }
n_val=$(wc -l < /mnt/su-vista-uscentral1/chaudhari_lab/phil/pneumonitis-variants-v1_7/splits/val.jsonl)

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
(cd "$TASK_DIR" && uv run -q --no-project --with numpy python -B scripts/build_data.py check) > "$OUT/data_check.txt" 2>&1 || fail "data check failed (data_check.txt)"
[ -f "$OUT/probe.json" ] || (cd "$TASK_DIR" && OMP_NUM_THREADS=4 uv run -q --no-project --with numpy --with scikit-learn python -B scripts/score_probe.py) > "$OUT/probe.log" 2>&1 || fail "probe failed (probe.log)"

# Each session gets its own CT copy (~40 MB); once its answer (output.json) is 2 minutes old nothing reads it again (a
# resume restages only unfinished calls). The engine leaves the call folder read-only, so it is made writable for the
# delete and read-only again (organ-zoom's first pruner only unlocked ct/ and deleted nothing, 2026-11-24).
prune() {
    while :; do
        n=0
        for ct in $(ls -d "$WORK"/program-outputs/*/*/calls/*/ct 2>/dev/null); do
            d=$(dirname "$ct")
            [ -f "$d/output.json" ] || continue
            [ -n "$(find "$d/output.json" -mmin +2 2>/dev/null)" ] || continue
            chmod u+w "$d" && chmod -R u+w "$ct" && rm -rf "$ct" && n=$((n + 1))
            chmod u-w "$d"
        done
        echo "$(date -u +%FT%TZ) pruned $n CT copies; free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
        sleep 300
    done
}
prune > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

# A command in the background, watched: Ctrl-C to it if the study's total crosses CAP_USD.
watched() {
    "$@" & local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 120
        local t; t=$(total)
        if ! check "$t <= $CAP_USD"; then
            echo "total \$$t crosses the \$$CAP_USD cap: stopping"
            pkill -INT -P "$pid" 2>/dev/null; kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
            fail "stopped at the \$$CAP_USD cap (total \$$t)"
        fi
    done
    wait "$pid"
}
with_retries() {
    for attempt in 1 2 3; do
        if watched "$@"; then return 0; fi
        echo "attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    return 1
}

# The work folder: program-v0, the source scans for the builder, every pin.
[ -d "$WORK/repo" ] || loop init > "$OUT/init.log" 2>&1 || fail "init failed (init.log)"
git -C "$WORK/repo" log --format=%s | grep -q '^setup pins for' || alo pins --write | tail -1
git -C "$WORK/repo" log --format=%s | grep -q '^builder pin for' || loop pins --write | tail -1

cost_check() {  # cost_check NAME TAG: 10 VAL scans; writes $OUT/costcheck-NAME.json; echoes the mean $ a scan
    local name=$1 tag=$2
    if [ -f "$OUT/costcheck-$name.json" ]; then json_get "$OUT/costcheck-$name.json" "d['pass']['mean_usd']"; return; fi
    status running "cost check: $name ($tag) on 10 VAL scans" >&2
    with_retries alo val-pass --tag "$tag" --limit 10 > "$OUT/costcheck-$name.log" 2>&1 || fail "$name cost check did not finish (costcheck-$name.log)"
    local look; look=$(ledger_look "$WORK/private/val_pass_ledger.jsonl")
    python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" --before "$BEFORE_USD" --run-id "val-pass-$tag-look$look" --out "$OUT/costcheck-$name.json.tmp" > /dev/null
    check "$(json_get "$OUT/costcheck-$name.json.tmp" "d['pass']['at_cap']") == 0" || fail "$name cost check: a session reached the \$0.75 cap"
    check "$(json_get "$OUT/costcheck-$name.json.tmp" "d['pass']['unreadable']") == 0" || fail "$name cost check: an answer was unreadable"
    py scripts/usage.py "$(ledger_answers "$WORK/private/val_pass_ledger.jsonl")" --tree "$WORK/versions/val-pass-$tag" --out "$OUT/usage-costcheck-$name.json" > /dev/null
    mv "$OUT/costcheck-$name.json.tmp" "$OUT/costcheck-$name.json"
    json_get "$OUT/costcheck-$name.json" "d['pass']['mean_usd']"
}

val_pass() {  # val_pass NAME TAG: every VAL scan; writes $OUT/val-NAME.json (the answers) and usage-val-NAME.json
    local name=$1 tag=$2
    [ -f "$OUT/val-$name.json" ] && { echo "VAL $name already done"; return; }
    status running "VAL $name ($tag) on $n_val scans"
    with_retries alo val-pass --tag "$tag" > "$OUT/val-$name.log" 2>&1 || fail "VAL $name did not finish (val-$name.log)"
    local ans; ans=$(ledger_answers "$WORK/private/val_pass_ledger.jsonl")
    py scripts/usage.py "$ans" --tree "$WORK/versions/val-pass-$tag" --out "$OUT/usage-val-$name.json" > /dev/null
    cp "$ans" "$OUT/val-$name.json.tmp" && mv "$OUT/val-$name.json.tmp" "$OUT/val-$name.json"
    echo "VAL $name done; total \$$(total)"
}

# 2-3. zero-shot
PER0=$(cost_check zero program-v0)
if [ ! -f "$OUT/audit.ok" ]; then
    python3 -I "$ENGINE_WT/scripts/audit_vertex_run.py" "$WORK" --route "$ALO_ROUTE" --skip private/archive > "$OUT/audit.txt" 2>&1 || fail "after-run audit failed (audit.txt)"
    touch "$OUT/audit.ok"
fi
# Guide-arm sessions open 10 example images and a guide: projected at 1.5x the zero-shot cost; 5 builders at their cap.
proj=$(python3 -c "print(round($(total) + 1.15 * $PER0 * $n_val + 1.5 * 1.15 * $PER0 * (400 + $n_val) + 5 * 15, 2))")
echo "projection: \$$proj (total \$$(total); zero-shot \$$PER0 a scan)" | tee "$OUT/projection.txt"
[ -f "$OUT/loop.json" ] || check "$proj <= $CAP_USD" || fail "projected \$$proj passes the \$$CAP_USD cap"
val_pass zero program-v0

# 4. the loop
if [ ! -f "$OUT/loop.json" ]; then
    status running "builder loop: 5 rounds on TRAIN"
    # The loop counts its own spend from its state; everything else (VAL zero-shot, cost check, infection) goes in.
    already=$(python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" --before "$BEFORE_USD" | python3 -c "import json,sys; d=json.load(sys.stdin); print(round(d['total_usd'] - d['loop_usd'], 2))")
    watched loop loop --run-id loop-r1 --cap "$CAP_USD" --val-scans "$n_val" --already-spent "$already" > "$OUT/loop.log" 2>&1 || fail "loop stopped (loop.log)"
    loop report --run-id loop-r1 --out "$OUT/loop.json.tmp" > /dev/null && mv "$OUT/loop.json.tmp" "$OUT/loop.json"
fi
FINAL=$(json_get "$OUT/loop.json" "d['final']['tag']")
# The builder's local CT copies (~36 MB each, up to ~11 GB) are not needed after the loop: free the shared disk.
find "$WORK/builder-scans" -mindepth 2 -maxdepth 2 -type d -name ct -exec chmod -R u+w {} + -exec rm -rf {} + 2>/dev/null || true
echo "loop done: final version $FINAL; total \$$(total)"

# 5-6. guide arm
PERG=$(cost_check guide "$FINAL")
opened=$(json_get "$OUT/usage-costcheck-guide.json" "d['share_opened_all_drawn']")
echo "guide cost check: \$$PERG a scan; share of sessions that opened every drawn image $opened"
check "$opened is not None and $opened >= 0.5" || fail "guide cost check: only $opened of sessions opened every drawn image (spec: stop under half)"
check "$(total) + 1.15 * $PERG * $n_val <= $CAP_USD" || fail "guide VAL would cross the \$$CAP_USD cap (total \$$(total), \$$PERG a scan)"
val_pass guide "$FINAL"

# 7. comparison
status running "paired comparison"
py scripts/compare.py --zero "$OUT/val-zero.json" --guide "$OUT/val-guide.json" --out "$OUT/comparison" > "$OUT/compare.log" 2>&1 || fail "comparison failed (compare.log)"
python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" --before "$BEFORE_USD" --out "$OUT/spend.json" > /dev/null
status done "pneumonitis variants finished on $(hostname): total \$$(total) of \$$CAP_USD; table in pneumonitis-variants/comparison/table.md"
