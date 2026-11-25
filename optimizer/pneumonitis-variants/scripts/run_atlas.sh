#!/usr/bin/env bash
# The atlas arm on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-atlas-arm.md, Approved
# 2026-11-25; this launcher is its approved Expensive step, <= $300 through VAL). Copied from scripts/run_variants.sh.
# Starts only once the variants study is done: it reuses that study's zero-shot and guide VAL arms (same 134 scans) and
# needs the shared disk. Re-running resumes.
#   1. setup: host, engine pin, Vertex login, disk, data check, the variants study finished, the pruner
#   2. the loop through round 1: round 0 (literature review + first guide; must pass tools/atlas.py check), round 1
#      (the atlas, then the reader on 80 TRAIN scans)
#   3. atlas cost check: round 1's version on the first 10 VAL scans. Stops if a session fails, fewer than half open
#      every atlas image, or the projection passes CAP_USD
#   4. the loop on to round 5
#   5. atlas VAL: round 5's version on all VAL scans
#   6. $0: the comparison: atlas, guide arm, zero-shot and the MerlinOnc probe; atlas - zero-shot, atlas - guide
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-pneumonitis-atlas:optimizer/pneumonitis-variants/scripts/run_atlas.sh > ~/run_atlas.sh && tmux new -d -s pneumonitis-atlas 'bash ~/run_atlas.sh 2>&1 | tee -a ~/pneumonitis-atlas.log'
#
# Settings, optional, on the same line: CAP_USD (300, the spec), PROGRAM_MAX_CONCURRENT (3), MIN_FREE_GB (40).
# Pass: STATUS.txt reads "Status: done"; comparison/table.md holds the table.
set -euo pipefail

BRANCH=feat/ct-pneumonitis-atlas
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-pneumonitis-atlas-run}
TASK_DIR=$TASK_WT/optimizer/pneumonitis-variants
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/pneumonitis-atlas
VARIANTS_OUT=$RUNS/pneumonitis-variants
WORK=$HOME/.cache/alo-tasks/pneumonitis-variants-atlas  # config.WORKDIR with ATLAS=1
CAP_USD=${CAP_USD:-300}
export ATLAS=1
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
export VAL_PASS_STOP_USD=110
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
spend() { python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" "$@"; }
total() { spend | python3 -c "import json,sys; print(json.load(sys.stdin)['total_usd'])"; }
n_val=$(wc -l < /mnt/su-vista-uscentral1/chaudhari_lab/phil/pneumonitis-variants-v1_7/splits/val.jsonl)

status running "setup on $(hostname)"
[ "$(hostname)" = phil-sllm-02 ] || echo "note: spec names phil-sllm-02; running on $(hostname)"
grep -q "^Status: done" "$VARIANTS_OUT/STATUS.txt" 2>/dev/null || fail "the variants study is not done ($VARIANTS_OUT/STATUS.txt): the atlas arm starts after it"
[ -f "$VARIANTS_OUT/val-zero.json" ] && [ -f "$VARIANTS_OUT/val-guide.json" ] || fail "the variants study's VAL arms are missing"
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

prune() {  # as run_variants.sh: a session's CT copy once its answer is 2 minutes old; the call folder is read-only
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
loop_already() { spend | python3 -c "import json,sys; d=json.load(sys.stdin); print(round(d['total_usd'] - d['loop_usd'], 2))"; }

[ -d "$WORK/repo" ] || loop init > "$OUT/init.log" 2>&1 || fail "init failed (init.log)"
git -C "$WORK/repo" log --format=%s | grep -q '^setup pins for' || alo pins --write | tail -1
git -C "$WORK/repo" log --format=%s | grep -q '^builder pin for' || loop pins --write | tail -1

# 2. round 0 (literature) and round 1
if [ ! -f "$OUT/round1.ok" ]; then
    status running "builder loop: literature review (round 0) and atlas round 1"
    watched loop loop --run-id atlas-r1 --cap "$CAP_USD" --val-scans "$n_val" --already-spent "$(loop_already)" --stop-after 1 > "$OUT/loop-r01.log" 2>&1 || fail "loop stopped in rounds 0-1 (loop-r01.log)"
    loop report --run-id atlas-r1 --out "$OUT/loop-r01.json" > /dev/null
    touch "$OUT/round1.ok"
fi
TAG1=$(json_get "$OUT/loop-r01.json" "[r for r in d['rounds'] if r['round'] == 1][0]['tag']")
echo "round 0 literature: $(json_get "$OUT/loop-r01.json" "[r for r in d['rounds'] if r['round'] == 0][0]['library']")"

# 3. atlas cost check on 10 VAL scans with round 1's version
if [ ! -f "$OUT/costcheck-atlas.json" ]; then
    status running "cost check: atlas ($TAG1) on 10 VAL scans"
    with_retries alo val-pass --tag "$TAG1" --limit 10 > "$OUT/costcheck-atlas.log" 2>&1 || fail "atlas cost check did not finish (costcheck-atlas.log)"
    look=$(ledger_look "$WORK/private/val_pass_ledger.jsonl")
    spend --run-id "val-pass-$TAG1-look$look" --out "$OUT/costcheck-atlas.json.tmp" > /dev/null
    check "$(json_get "$OUT/costcheck-atlas.json.tmp" "d['pass']['unreadable']") == 0" || fail "atlas cost check: a session failed or gave no readable answer (spec: image size)"
    check "$(json_get "$OUT/costcheck-atlas.json.tmp" "d['pass']['at_cap']") == 0" || fail "atlas cost check: a session reached the \$0.75 cap"
    py scripts/usage.py "$(ledger_answers "$WORK/private/val_pass_ledger.jsonl")" --tree "$WORK/versions/val-pass-$TAG1" --out "$OUT/usage-costcheck-atlas.json" > /dev/null
    mv "$OUT/costcheck-atlas.json.tmp" "$OUT/costcheck-atlas.json"
fi
PER=$(json_get "$OUT/costcheck-atlas.json" "d['pass']['mean_usd']")
opened=$(json_get "$OUT/usage-costcheck-atlas.json" "d['share_opened_all_drawn']")
check "$opened is not None and $opened >= 0.5" || fail "atlas cost check: only $opened of sessions opened every atlas image (spec: stop under half)"
proj=$(python3 -c "print(round($(total) + 1.15 * $PER * (4 * 80 + $n_val) + 4 * 15, 2))")
echo "atlas cost check: \$$PER a scan; opened every atlas image $opened; projection \$$proj (total \$$(total))" | tee "$OUT/projection.txt"
[ -f "$OUT/loop.json" ] || check "$proj <= $CAP_USD" || fail "projected \$$proj passes the \$$CAP_USD cap"

# 4. rounds 2-5
if [ ! -f "$OUT/loop.json" ]; then
    status running "builder loop: atlas rounds 2-5 on TRAIN"
    watched loop loop --run-id atlas-r1 --cap "$CAP_USD" --val-scans "$n_val" --already-spent "$(loop_already)" > "$OUT/loop.log" 2>&1 || fail "loop stopped (loop.log)"
    loop report --run-id atlas-r1 --out "$OUT/loop.json.tmp" > /dev/null && mv "$OUT/loop.json.tmp" "$OUT/loop.json"
fi
FINAL=$(json_get "$OUT/loop.json" "d['final']['tag']")
find "$WORK/builder-scans" -mindepth 2 -maxdepth 2 -type d -name ct -exec chmod -R u+w {} + -exec rm -rf {} + 2>/dev/null || true
echo "loop done: final version $FINAL; total \$$(total)"

# 5. atlas VAL
if [ ! -f "$OUT/val-atlas.json" ]; then
    check "$(total) + 1.15 * $PER * $n_val <= $CAP_USD" || fail "atlas VAL would cross the \$$CAP_USD cap (total \$$(total))"
    status running "VAL atlas ($FINAL) on $n_val scans"
    with_retries alo val-pass --tag "$FINAL" > "$OUT/val-atlas.log" 2>&1 || fail "VAL atlas did not finish (val-atlas.log)"
    ans=$(ledger_answers "$WORK/private/val_pass_ledger.jsonl")
    py scripts/usage.py "$ans" --tree "$WORK/versions/val-pass-$FINAL" --out "$OUT/usage-val-atlas.json" > /dev/null
    cp "$ans" "$OUT/val-atlas.json.tmp" && mv "$OUT/val-atlas.json.tmp" "$OUT/val-atlas.json"
fi

# 6. comparison
status running "comparison"
py scripts/compare.py --zero "$VARIANTS_OUT/val-zero.json" --guide "$VARIANTS_OUT/val-guide.json" --arm "atlas=$OUT/val-atlas.json" \
    --pair atlas:zero-shot --pair atlas:guide --pair guide:zero-shot --out "$OUT/comparison" > "$OUT/compare.log" 2>&1 || fail "comparison failed (compare.log)"
spend --out "$OUT/spend.json" > /dev/null
status done "pneumonitis atlas finished on $(hostname): total \$$(total) of \$$CAP_USD; table in pneumonitis-atlas/comparison/table.md"
