#!/usr/bin/env bash
# Set up and run the death-agent-over-time hill-climb on phil-sllm-01 (Vertex), judging every step; then run it
# detached, resuming after network stops. Needs the VAL and TRAIN records already on this machine's local disk
# (build_points.py export, then export --split train).
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/died-ehr-over-time:optimizer/died-ehr-over-time/scripts/run_climb.sh > ~/run_climb.sh && SETUP_ONLY=1 RUN_ID=climb-r1 bash ~/run_climb.sh
#
# Settings, all optional, on the same line: RUN_ID (climb-r1), BRANCH (feat/died-ehr-over-time), ALO_ROUTE (vertex),
# PROGRAM_MAX_CONCURRENT (2), MAX_RESUMES (20), CLIMB_BUDGET_USD (300, the hard stop), SETUP_ONLY=1.
# Pass: "SETUP PASS"; the log ends "FINISHED 10 iterations"; per-round VAL tables, the run summary and the best
# version's instructions reach /mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-runs/<host>/<run id>/.
set -euo pipefail

BRANCH=${BRANCH:-feat/died-ehr-over-time}
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=$HOME/code/vista_eval_vlm-over-time-run
TASK_DIR=$TASK_WT/optimizer/died-ehr-over-time
D2_DIR=$TASK_WT/optimizer/died-2yr-ehr
ENGINE=$HOME/code/agentic-label-opt
ENGINE_WT=$ENGINE/.claude/worktrees/mortality-engine
PRIVATE=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-private
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export TASK_DATA_DIR=${TASK_DATA_DIR:-$HOME/data/died-ehr-over-time-v1_7}
export CLIMB_WORKDIR=${CLIMB_WORKDIR:-$HOME/.cache/alo-tasks/died-ehr-over-time-climb}
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}
export CLIMB_BUDGET_USD=${CLIMB_BUDGET_USD:-300}
export ITERATIONS=10
export AGENTIC_LABEL_OPT=$ENGINE_WT
RUN_ID=${RUN_ID:-climb-r1}
MAX_RESUMES=${MAX_RESUMES:-20}
RESULTS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-runs/$(hostname)/$RUN_ID
LOG=$CLIMB_WORKDIR/$RUN_ID-run.log

fail() { echo "SETUP FAIL: $*"; exit 1; }
cl() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with pyyaml python -B climb.py "$@"); }
ot() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with pyyaml python -B run_over_time.py "$@"); }

publish() {  # aggregates and the program text to the results folder (no ids, rows or reasons)
    mkdir -p "$RESULTS"
    cl report --run-id "$RUN_ID" > "$RESULTS/report.txt" 2>&1 || true
    cp "$CLIMB_WORKDIR/private/runs/$RUN_ID/run_summary.json" "$RESULTS/" 2>/dev/null || true
    python3 -I -c "import json,sys; [print(json.dumps({k: v for k, v in json.loads(l).items() if k != 'answers'})) for l in open(sys.argv[1]) if l.strip()]" \
        "$CLIMB_WORKDIR/private/climb/val_rounds.jsonl" > "$RESULTS/val_rounds.jsonl" 2>/dev/null || true
    grep -E "^(VAL round|  [a-z0-9]+ +(fresh|memory) )" "$LOG" > "$RESULTS/val_tables.txt" 2>/dev/null || true
    rm -rf "$RESULTS/versions" && mkdir -p "$RESULTS/versions"
    for v in "$CLIMB_WORKDIR"/versions/iter*-program-v*; do  # each scored version's instructions (program text only)
        case "$v" in *-memory) continue;; esac
        [ -f "$v/instructions.md" ] && mkdir -p "$RESULTS/versions/${v##*/}" && cp -r "$v/." "$RESULTS/versions/${v##*/}/"
    done
    cp "$LOG" "$RESULTS/run.log" 2>/dev/null || true
}

if [ "${1:-}" = "--loop" ]; then
    attempt=0; mode=()
    [ -f "$CLIMB_WORKDIR/private/runs/$RUN_ID/run_summary.json" ] && mode=(--resume)
    while :; do
        echo "=== $(date -u +%FT%TZ) climb $RUN_ID ${mode[*]:-fresh} (resume $attempt of $MAX_RESUMES)"
        cl run --run-id "$RUN_ID" "${mode[@]}" 2>&1 | tee "$CLIMB_WORKDIR/last-pass.log" || true
        publish
        if grep -q "^FINISHED" "$CLIMB_WORKDIR/last-pass.log"; then echo "FINISHED"; exit 0; fi
        if grep -qE "STOPPED \((proxy_died|pass_stopped|network)" "$CLIMB_WORKDIR/last-pass.log" && [ "$attempt" -lt "$MAX_RESUMES" ]; then
            attempt=$((attempt + 1)); mode=(--resume); sleep 150; continue  # the NAT holds a closed port for 120 s
        fi
        echo "STOPPED for a reason other than the network (a dollar, cost or leak rule included), or out of resumes: see above; not resumed"
        exit 3
    done
fi

echo "host $(hostname)"
command -v uv >/dev/null || fail "uv is not installed"
docker info >/dev/null 2>&1 || fail "Docker is not running or this user cannot use it"
if [ "$ALO_ROUTE" = vertex ]; then
    LOGIN=/home/optimizer/.config/gcloud/application_default_credentials.json
    [ -f "$LOGIN" ] && [ ! -L "$LOGIN" ] || fail "no optimizer login at $LOGIN"
fi

git -C "$VLM" fetch -q origin "$BRANCH"
if [ -d "$TASK_WT" ]; then
    [ -z "$(git -C "$TASK_WT" status --porcelain)" ] || fail "$TASK_WT has uncommitted changes; not touching it"
    git -C "$TASK_WT" checkout -q --detach "origin/$BRANCH"
else
    git -C "$VLM" worktree add -q --detach "$TASK_WT" "origin/$BRANCH"
fi
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$D2_DIR/config.py")
[ -n "$PIN" ] && [ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || fail "engine at $ENGINE_WT is not at the pin $PIN"
echo "task $(git -C "$TASK_WT" rev-parse --short HEAD), engine $(git -C "$ENGINE_WT" rev-parse --short HEAD)"

(cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow python -B build_points.py local) || fail "build_points.py local"
want=$(( $(wc -l < "$PRIVATE/cutoffs.jsonl") + $(wc -l < "$PRIVATE/train/cutoffs.jsonl") ))
n=$(find "$TASK_DATA_DIR/patients" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" -eq "$want" ] || fail "expected $want VAL+TRAIN record folders in $TASK_DATA_DIR/patients, found $n (TRAIN exported?)"
[ -d "$PRIVATE/runs/over-time-r1/answers" ] || fail "no first-pass answers for round 0 at $PRIVATE/runs/over-time-r1/answers"
echo "records: $n VAL+TRAIN in $TASK_DATA_DIR; round 0 from the first pass"
rc=0; out=$(ot leak-check --split all 2>&1) || rc=$?
echo "$out" | tail -11
[ "$rc" = 0 ] || fail "record leak check"

(cd "$D2_DIR" && docker image inspect "$(uv run -q --project "$ENGINE_WT" python -c 'import config; print(config.OPTIMIZER_IMAGE_TAG)')" >/dev/null 2>&1) \
    || fail "the died-2yr-ehr image is missing: build it with run.py build-images when no died-2yr-ehr run is live"
[ -d "$CLIMB_WORKDIR/repo" ] || cl init || fail "init"
cl pins --write | tail -1 || fail "pins --write"

rm -rf /tmp/over-time-synth
(cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with scikit-learn --with pyyaml python -B scripts/check_task.py synth /tmp/over-time-synth >/dev/null \
    && env -u OVER_TIME_PAST_START_GATE -u CLIMB_WORKDIR -u CLIMB_BUDGET_USD -u ITERATIONS TASK_DATA_DIR=/tmp/over-time-synth TASK_WORKDIR=/tmp/over-time-synth-work uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with scikit-learn --with pyyaml python -B scripts/check_task.py check | tail -1) \
    | tee /dev/stderr | grep -q "checks PASS$" || fail "check_task.py"

if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: climb not started)"; exit 0; fi
mkdir -p "$CLIMB_WORKDIR" "$RESULTS"
cp "$TASK_DIR/scripts/run_climb.sh" "$CLIMB_WORKDIR/run_climb.sh"
setsid nohup env RUN_ID="$RUN_ID" MAX_RESUMES="$MAX_RESUMES" BRANCH="$BRANCH" bash "$CLIMB_WORKDIR/run_climb.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: climb $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time, hard stop \$$CLIMB_BUDGET_USD); log $LOG; results to $RESULTS"
