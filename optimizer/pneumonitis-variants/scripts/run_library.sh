#!/usr/bin/env bash
# The library arm on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-library.md, Approved
# 2026-11-25; this launcher is its approved Expensive step, <= $450 through VAL, discovery <= $175). Built from
# scripts/run_variants.sh. Re-running resumes: each finished step leaves a marker in OUT.
#   1. setup: host, engine pin, Vertex login, disk, the data check (599 TRAIN scans), the pruner
#   2. the work folder: program-v0, the caches (599 TRAIN scans, 252 literature images), every pin
#   3. discovery cost check: the director's plan and the first manager (radiation); stops if 5 more sessions at the
#      dearest so far could pass $175; then the engine's after-run audit
#   4. discovery: the other four managers and the director's synthesis
#   5. climb round 1 (the reader on 80 TRAIN scans with discovery's library); stops if the projection passes CAP_USD
#   6. climb rounds 2-5 (the director revises, then the reader)
#   7. library VAL: the final version on the 134 VAL scans
#   8. $0: the comparison: library, guide arm, zero-shot and the MerlinOnc probe, five causes and merged "any drug"
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-pneumonitis-library:optimizer/pneumonitis-variants/scripts/run_library.sh > ~/run_library.sh && tmux new -d -s pneumonitis-library 'bash ~/run_library.sh 2>&1 | tee -a ~/pneumonitis-library.log'
#
# Settings, optional, on the same line: CAP_USD (450), DISCOVERY_CAP_USD (175), PROGRAM_MAX_CONCURRENT (3),
# MIN_FREE_GB (60: the TRAIN cache is ~22 GB). Pass: STATUS.txt reads "Status: done"; comparison/table.md has the table.
set -euo pipefail

BRANCH=feat/ct-pneumonitis-library
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-pneumonitis-library-run}
TASK_DIR=$TASK_WT/optimizer/pneumonitis-variants
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/pneumonitis-library
VARIANTS_OUT=$RUNS/pneumonitis-variants
WORK=$HOME/.cache/alo-tasks/pneumonitis-variants-library  # config.WORKDIR with LIBRARY=1
RUN_ID=library-r1
CAP_USD=${CAP_USD:-450}
DISCOVERY_CAP_USD=${DISCOVERY_CAP_USD:-175}
export LIBRARY=1
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
export VAL_PASS_STOP_USD=140  # 134 scans x the $1.00 cap; the real stop is CAP_USD, enforced here
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
alo() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python run.py "$@"); }
loop() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B scripts/run_loop.py "$@"); }
py() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
ledger_answers() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['answers'])" "$1"; }
spend() { python3 -I "$TASK_DIR/scripts/spend.py" "$WORK" "$@"; }
total() { spend | python3 -c "import json,sys; print(json.load(sys.stdin)['total_usd'])"; }
loop_already() { spend | python3 -c "import json,sys; d=json.load(sys.stdin); print(round(d['total_usd'] - d['loop_usd'], 2))"; }
n_val=$(wc -l < /mnt/su-vista-uscentral1/chaudhari_lab/phil/pneumonitis-variants-v1_7/splits/val.jsonl)

status running "setup on $(hostname)"
[ "$(hostname)" = phil-sllm-02 ] || echo "note: spec names phil-sllm-02; running on $(hostname)"
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
[ -d "$WORK/train-scans" ] || [ "$free_gb" -ge "${MIN_FREE_GB:-60}" ] || fail "only ${free_gb} GB free under ~/.cache (the TRAIN cache needs ~22 GB)"
(cd "$TASK_DIR" && uv run -q --no-project --with numpy python -B scripts/build_data.py check) > "$OUT/data_check.txt" 2>&1 || fail "data check failed (data_check.txt)"

# Each session's CT copy and its linked library files (hard links: no extra disk, but thousands of names) go once its
# answer is 2 minutes old. The engine leaves the call folder read-only: unlock it for the delete, lock it again.
prune() {
    while :; do
        n=0
        for d in $(ls -d "$WORK"/program-outputs/*/*/calls/* 2>/dev/null); do
            [ -f "$d/output.json" ] || continue
            [ -n "$(find "$d/output.json" -mmin +2 2>/dev/null)" ] || continue
            [ -e "$d/ct" ] || [ -e "$d/library_scans" ] || [ -e "$d/library_assets" ] || continue
            chmod u+w "$d" && for x in ct library_scans library_assets; do [ -e "$d/$x" ] && chmod -R u+w "$d/$x" && rm -rf "$d/$x"; done
            chmod u-w "$d"; n=$((n + 1))
        done
        echo "$(date -u +%FT%TZ) pruned $n call folders; free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
        sleep 300
    done
}
prune > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

watched() {  # watched CAP COMMAND...: Ctrl-C to the command if the study's total crosses CAP
    local cap=$1; shift
    "$@" & local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 120
        local t; t=$(total)
        if ! check "$t <= $cap"; then
            echo "total \$$t crosses the \$$cap cap: stopping"
            pkill -INT -P "$pid" 2>/dev/null; kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
            fail "stopped at the \$$cap cap (total \$$t)"
        fi
    done
    wait "$pid"
}

# 2. the work folder
if [ ! -f "$OUT/init.ok" ]; then
    status running "work folder: caches (599 TRAIN scans, 252 literature images) and pins"
    loop init > "$OUT/init.log" 2>&1 || fail "init failed (init.log)"
    git -C "$WORK/repo" log --format=%s | grep -q '^setup pins for' || alo pins --write | tail -1
    git -C "$WORK/repo" log --format=%s | grep -q '^builder pin for' || loop pins --write | tail -1
    touch "$OUT/init.ok"
fi

# 3. discovery cost check: sessions 1-2
if [ ! -f "$OUT/discovery-check.ok" ]; then
    status running "discovery cost check: the director's plan and the radiation manager"
    watched "$DISCOVERY_CAP_USD" loop discover --run-id "$RUN_ID" --stop-after 2 > "$OUT/discovery-12.log" 2>&1 || fail "discovery sessions 1-2 stopped (discovery-12.log)"
    loop report --run-id "$RUN_ID" --out "$OUT/discovery-12.json" > /dev/null
    proj=$(python3 -c "
import json; d=json.load(open('$OUT/discovery-12.json'))['discovery']; c=[r['builder_usd'] or 0 for r in d if r.get('tag')]
print(round(sum(c) + 5 * max(c), 2))")
    echo "discovery: sessions 1-2 \$$(python3 -c "import json; print(round(sum(r['builder_usd'] or 0 for r in json.load(open('$OUT/discovery-12.json'))['discovery']), 2))"); projection \$$proj" | tee "$OUT/discovery-projection.txt"
    check "$proj <= $DISCOVERY_CAP_USD" || fail "discovery projected \$$proj passes the \$$DISCOVERY_CAP_USD cap"
    python3 -I "$ENGINE_WT/scripts/audit_vertex_run.py" "$WORK" --route "$ALO_ROUTE" --skip private/archive > "$OUT/audit.txt" 2>&1 || fail "after-run audit failed (audit.txt)"
    touch "$OUT/discovery-check.ok"
fi

# 4. discovery: sessions 3-7
if [ ! -f "$OUT/discovery.json" ]; then
    status running "discovery: managers (immunotherapy, other drug, COP, infection) and the director's synthesis"
    watched "$CAP_USD" loop discover --run-id "$RUN_ID" > "$OUT/discovery.log" 2>&1 || fail "discovery stopped (discovery.log)"
    loop report --run-id "$RUN_ID" --out "$OUT/discovery.json.tmp" > /dev/null && mv "$OUT/discovery.json.tmp" "$OUT/discovery.json"
fi
echo "discovery done: $(json_get "$OUT/discovery.json" "d['discovery_final']"); total \$$(total)"

# 5. climb round 1, then its projection
if [ ! -f "$OUT/round1.ok" ]; then
    status running "climb round 1: the reader on 80 TRAIN scans with discovery's library"
    watched "$CAP_USD" loop loop --run-id "$RUN_ID" --cap "$CAP_USD" --val-scans "$n_val" --already-spent "$(loop_already)" --stop-after 1 > "$OUT/climb-r1.log" 2>&1 || fail "climb round 1 stopped (climb-r1.log)"
    touch "$OUT/round1.ok"
fi
loop report --run-id "$RUN_ID" --out "$OUT/climb-r1.json" > /dev/null
PER=$(json_get "$OUT/climb-r1.json" "[r for r in d['rounds'] if r['round'] == 1][0]['reader_usd'] / [r for r in d['rounds'] if r['round'] == 1][0]['reader_sessions']")
proj=$(python3 -c "print(round($(total) + 4 * (1.15 * $PER * 80 + 25) + 1.15 * $PER * $n_val, 2))")
echo "climb round 1: \$$PER a scan; projection \$$proj (total \$$(total))" | tee "$OUT/projection.txt"
[ -f "$OUT/loop.json" ] || check "$proj <= $CAP_USD" || fail "projected \$$proj passes the \$$CAP_USD cap"

# 6. climb rounds 2-5
if [ ! -f "$OUT/loop.json" ]; then
    status running "climb rounds 2-5: the director revises, the reader reads 80 TRAIN scans"
    watched "$CAP_USD" loop loop --run-id "$RUN_ID" --cap "$CAP_USD" --val-scans "$n_val" --already-spent "$(loop_already)" > "$OUT/climb.log" 2>&1 || fail "climb stopped (climb.log)"
    loop report --run-id "$RUN_ID" --out "$OUT/loop.json.tmp" > /dev/null && mv "$OUT/loop.json.tmp" "$OUT/loop.json"
fi
FINAL=$(json_get "$OUT/loop.json" "d['final']['tag']")
echo "climb done: final version $FINAL; total \$$(total)"

# 7. library VAL
if [ ! -f "$OUT/val-library.json" ]; then
    check "$(total) + 1.15 * $PER * $n_val <= $CAP_USD" || fail "library VAL would cross the \$$CAP_USD cap (total \$$(total))"
    status running "VAL library ($FINAL) on $n_val scans"
    watched "$CAP_USD" alo val-pass --tag "$FINAL" > "$OUT/val-library.log" 2>&1 || fail "VAL library did not finish (val-library.log)"
    ans=$(ledger_answers "$WORK/private/val_pass_ledger.jsonl")
    cp "$ans" "$OUT/val-library.json.tmp" && mv "$OUT/val-library.json.tmp" "$OUT/val-library.json"
fi

# 8. comparison
status running "comparison"
py scripts/compare.py --zero "$VARIANTS_OUT/val-zero.json" --guide "$VARIANTS_OUT/val-guide.json" --arm "library=$OUT/val-library.json" \
    --pair library:zero-shot --pair library:guide --pair guide:zero-shot --out "$OUT/comparison" > "$OUT/compare.log" 2>&1 || fail "comparison failed (compare.log)"
spend --out "$OUT/spend.json" > /dev/null
# The TRAIN cache (~22 GB) is not needed once VAL is scored: free the shared disk.
chmod -R u+w "$WORK/train-scans" 2>/dev/null && rm -rf "$WORK/train-scans" || true
status done "pneumonitis library finished on $(hostname): total \$$(total) of \$$CAP_USD; table in pneumonitis-library/comparison/table.md"
