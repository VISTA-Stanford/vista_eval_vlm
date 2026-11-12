#!/usr/bin/env bash
# Set up and run the death agent over time on this VM (Vertex by default), judging every step; then run it detached,
# resuming after network stops. No BigQuery here: the records are rebuilt on local disk from the mount archive that
# build_points.py export wrote on phil-sllm-01.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/died-ehr-over-time:optimizer/died-ehr-over-time/scripts/run_on_vm.sh > ~/run_over_time.sh && SETUP_ONLY=1 bash ~/run_over_time.sh
#
# Settings, all optional, on the same line: RUN_ID (over-time-r1), BRANCH (feat/died-ehr-over-time), ALO_ROUTE
# (vertex), PROGRAM_MAX_CONCURRENT (2), MAX_RESUMES (20), RUN_BUDGET_USD (250, the hard stop), SETUP_ONLY=1 (stop
# before the paid run), OVER_TIME_PAST_START_GATE=1 (go on past a failed treatment-start check: Phil's go only). Pass: "SETUP PASS"; the log ends "FINISHED"; the score tables and cost reach
# /mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-runs/<host>/<run id>/ after every stop.
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
export TASK_WORKDIR=${TASK_WORKDIR:-$HOME/.cache/alo-tasks/died-ehr-over-time}
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}
export RUN_BUDGET_USD=${RUN_BUDGET_USD:-250}
export AGENTIC_LABEL_OPT=$ENGINE_WT
RUN_ID=${RUN_ID:-over-time-r1}
MAX_RESUMES=${MAX_RESUMES:-20}
RESULTS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-runs/$(hostname)/$RUN_ID
LOG=$TASK_WORKDIR/$RUN_ID-run.log

fail() { echo "SETUP FAIL: $*"; exit 1; }
ot() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with pyyaml python -B run_over_time.py "$@"); }

publish() {  # aggregates to the results folder; per-row answers (reasons text) to the private folder
    mkdir -p "$RESULTS" "$PRIVATE/runs/$RUN_ID"
    ot score --run-id "$RUN_ID" --out "$RESULTS/score.json" > "$RESULTS/score.txt" 2>&1 || true
    cp "$LOG" "$RESULTS/run.log" 2>/dev/null || true
    cp -r "$TASK_WORKDIR/private/runs/$RUN_ID/answers" "$PRIVATE/runs/$RUN_ID/" 2>/dev/null || true
}

if [ "${1:-}" = "--loop" ]; then
    attempt=0
    while :; do
        echo "=== $(date -u +%FT%TZ) run $RUN_ID (resume $attempt of $MAX_RESUMES)"
        rc=0; ot run --run-id "$RUN_ID" 2>&1 | tee "$TASK_WORKDIR/last-pass.log" || rc=$?
        publish
        if grep -q "^FINISHED" "$TASK_WORKDIR/last-pass.log"; then echo "FINISHED"; exit 0; fi
        if grep -q "STOPPED (rule)" "$TASK_WORKDIR/last-pass.log"; then echo "STOPPED by a rule: see above; not resumed (past the treatment-start check only with Phil's go: OVER_TIME_PAST_START_GATE=1 on the launch line)"; exit 3; fi
        if grep -q "STOPPED (pass_stopped)" "$TASK_WORKDIR/last-pass.log" && [ "$attempt" -lt "$MAX_RESUMES" ]; then
            attempt=$((attempt + 1)); sleep 150; continue  # the NAT holds a closed connection's port for 120 s
        fi
        echo "STOPPED for another reason, or out of resumes: see above"; exit 1
    done
fi

echo "host $(hostname)"
command -v uv >/dev/null || fail "uv is not installed"
docker info >/dev/null 2>&1 || fail "Docker is not running or this user cannot use it"
[ -d "$PRIVATE/archive/tables" ] || fail "no record archive at $PRIVATE/archive (run the export on phil-sllm-01 first)"
if [ "$ALO_ROUTE" = vertex ]; then
    LOGIN=/home/optimizer/.config/gcloud/application_default_credentials.json
    [ -f "$LOGIN" ] && [ ! -L "$LOGIN" ] || fail "no optimizer login at $LOGIN"
fi

# The task code at the pushed branch, in its own worktree.
git -C "$VLM" fetch -q origin "$BRANCH"
if [ -d "$TASK_WT" ]; then
    [ -z "$(git -C "$TASK_WT" status --porcelain)" ] || fail "$TASK_WT has uncommitted changes; not touching it"
    git -C "$TASK_WT" checkout -q --detach "origin/$BRANCH"
else
    git -C "$VLM" worktree add -q --detach "$TASK_WT" "origin/$BRANCH"
fi
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$D2_DIR/config.py")
[ -n "$PIN" ] || fail "no ENGINE_PIN in died-2yr-ehr's config.py"
git -C "$ENGINE" fetch -q origin
if [ -d "$ENGINE_WT" ]; then
    [ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || {
        [ -z "$(git -C "$ENGINE_WT" status --porcelain)" ] || fail "$ENGINE_WT has uncommitted changes; not touching it"
        git -C "$ENGINE_WT" checkout -q --detach "$PIN"; }
else
    git -C "$ENGINE" worktree add -q --detach "$ENGINE_WT" "$PIN"
fi
[ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || fail "engine is not at the pin $PIN"
echo "task $(git -C "$TASK_WT" rev-parse --short HEAD), engine $(git -C "$ENGINE_WT" rev-parse --short HEAD)"

# The records on local disk, rebuilt by died-2yr-ehr's rehydrate from this task's archive; then rows, labels, probes.
if [ ! -d "$TASK_DATA_DIR/patients" ]; then
    (cd "$D2_DIR" && TASK_MOUNT_DATA_DIR=$PRIVATE TASK_EXPORT_STAGE=$HOME/.cache/died-ehr-over-time-rehydrate \
        OMP_NUM_THREADS=2 nice -n 19 uv run -q --no-project --with "duckdb==1.5.6" --with pandas --with pyarrow \
        python -B scripts/build_data.py rehydrate) || fail "rehydrate (records from $PRIVATE/archive)"
fi
(cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow python -B build_points.py local) || fail "build_points.py local"
[ ! -e "$TASK_DATA_DIR/registry_dates" ] || fail "$TASK_DATA_DIR/registry_dates exists: raw registry dates must not be in the task copy"
want=$(wc -l < "$PRIVATE/cutoffs.jsonl"); n=$(find "$TASK_DATA_DIR/patients" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" -eq "$want" ] || fail "expected $want record folders in $TASK_DATA_DIR/patients, found $n"
echo "records: $n in $TASK_DATA_DIR"
rc=0; out=$(ot leak-check 2>&1) || rc=$?
echo "$out" | tail -7
[ "$rc" = 0 ] || fail "record leak check"

# The image (died-2yr-ehr's): build only if missing, so a died-2yr-ehr run live here keeps its image.
(cd "$D2_DIR" && docker image inspect "$(uv run -q --project "$ENGINE_WT" python -c 'import config; print(config.OPTIMIZER_IMAGE_TAG)')" >/dev/null 2>&1) \
    || (cd "$D2_DIR" && ANSWER=probability uv run -q --project "$ENGINE_WT" python run.py build-images >/dev/null) || fail "build-images"
ot pins --write | tail -1 || fail "pins --write"

# The $0 checks on synthetic data, with the same code.
rm -rf /tmp/over-time-synth
(cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with scikit-learn --with pyyaml python -B scripts/check_task.py synth /tmp/over-time-synth >/dev/null \
    && env -u OVER_TIME_PAST_START_GATE TASK_DATA_DIR=/tmp/over-time-synth TASK_WORKDIR=/tmp/over-time-synth-work uv run -q --project "$ENGINE_WT" --with pandas --with pyarrow --with "duckdb<2" --with scikit-learn --with pyyaml python -B scripts/check_task.py check | tail -1) \
    | tee /dev/stderr | grep -q "checks PASS$" || fail "check_task.py"

if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: run not started)"; exit 0; fi
mkdir -p "$TASK_WORKDIR" "$RESULTS"
cp "$TASK_DIR/scripts/run_on_vm.sh" "$TASK_WORKDIR/run_on_vm.sh"
setsid nohup env RUN_ID="$RUN_ID" MAX_RESUMES="$MAX_RESUMES" BRANCH="$BRANCH" OVER_TIME_PAST_START_GATE="${OVER_TIME_PAST_START_GATE:-}" bash "$TASK_WORKDIR/run_on_vm.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: run $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time, hard stop \$$RUN_BUDGET_USD); log $LOG; results to $RESULTS"
