#!/usr/bin/env bash
# Set up and run the died-2yr-ehr optimization on this VM, judging every step; then run it detached, resuming
# after network stops. Rebuilds the patients' records on local disk from the mount archive first (no BigQuery). Written for a VM other than phil-sllm-01, whose Cloud NAT runs out of its 64 ports per VM
# when several Claude runs share it (agentic-label-opt stops with proxy_died); another VM has its own 64.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/died-2yr-ehr-agentic:optimizer/died-2yr-ehr/scripts/run_on_vm.sh > ~/run_died2yr_ehr.sh && bash ~/run_died2yr_ehr.sh
#
# Needs: Docker, the su-vista-uscentral1 mount, uv, git access to both repos, and the Claude token at
# ~/.config/alo/oauth_token (mode 600; `claude setup-token` makes one). Settings, all optional, on the same line:
# RUN_ID (first), ITERATIONS (2), PROGRAM_MAX_CONCURRENT (3), MAX_RESUMES (20); SETUP_ONLY=1 stops before the run.
# Pass: it prints "SETUP PASS" and the log path; the log ends in "FINISHED"; results reach
# /mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7-runs/<host>/<run id>/ after every stop, with the
# baselines scored on the same VAL pool (baselines_val_pool.json).
set -euo pipefail

BRANCH=feat/died-2yr-ehr-agentic
ENGINE_BRANCH=fix/vertex-global-session-host
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=$HOME/code/vista_eval_vlm-died2yr-ehr
TASK_DIR=$TASK_WT/optimizer/died-2yr-ehr
ENGINE=$HOME/code/agentic-label-opt
ENGINE_WT=$ENGINE/.claude/worktrees/optimizer-websearch
DATA=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7
# The local working copy the task reads (config.DATA_DIR): rebuilt from $DATA/archive by build_data.py rehydrate.
export TASK_DATA_DIR=${TASK_DATA_DIR:-$HOME/data/died-2yr-ehr-v1_7}
RUN_ID=${RUN_ID:-first}
ITERATIONS=${ITERATIONS:-2}
MAX_RESUMES=${MAX_RESUMES:-20}
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export TASK_WORKDIR=$HOME/.cache/alo-tasks/died-2yr-ehr
RESULTS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7-runs/$(hostname)/$RUN_ID
LOG=$TASK_WORKDIR/$RUN_ID-run.log

fail() { echo "SETUP FAIL: $*"; exit 1; }
alo() { (cd "$TASK_DIR" && uv run --project "$ENGINE_WT" python run.py "$@"); }

publish() {  # the report and run summary, onto the mount (inside the secure environment)
    mkdir -p "$RESULTS"
    alo report --run-id "$RUN_ID" > "$RESULTS/report.txt" 2>&1 || true
    cp "$TASK_WORKDIR/private/runs/$RUN_ID/run_summary.json" "$RESULTS/" 2>/dev/null || true
    cp "$LOG" "$RESULTS/run.log" 2>/dev/null || true
}

if [ "${1:-}" = "--loop" ]; then
    # Detached part: run, and resume after a network stop (a resume reruns only unfinished calls).
    attempt=0
    mode=()
    [ -f "$TASK_WORKDIR/private/runs/$RUN_ID/run_summary.json" ] && mode=(--resume)
    while :; do
        echo "=== $(date -u +%FT%TZ) run $RUN_ID ${mode[*]:-fresh} (resume $attempt of $MAX_RESUMES)"
        alo run --run-id "$RUN_ID" --iterations "$ITERATIONS" "${mode[@]}" 2>&1 | tee "$TASK_WORKDIR/last-pass.log" || true
        publish
        if grep -q "^FINISHED" "$TASK_WORKDIR/last-pass.log"; then echo "FINISHED"; exit 0; fi
        if grep -q "STOPPED (proxy_died)" "$TASK_WORKDIR/last-pass.log" && [ "$attempt" -lt "$MAX_RESUMES" ]; then
            attempt=$((attempt + 1)); mode=(--resume)
            sleep 150  # the NAT holds a closed connection's port for 120 s
            continue
        fi
        echo "STOPPED for a reason other than the network, or out of resumes: see above"; exit 1
    done
fi

echo "host $(hostname)"
command -v uv >/dev/null || fail "uv is not installed"
docker info >/dev/null 2>&1 || fail "Docker is not running or this user cannot use it"
[ -f "$DATA/pools.json" ] || fail "the data is not readable at $DATA (is the su-vista-uscentral1 mount here?)"
[ -d "$DATA/archive/tables" ] || fail "no record archive at $DATA/archive (the export has not been run yet)"
TOKEN=$HOME/.config/alo/oauth_token
[ -f "$TOKEN" ] || fail "no Claude token at $TOKEN: run claude setup-token, save it there with mode 600"
[ "$(stat -c %a "$TOKEN")" = 600 ] || fail "$TOKEN must be mode 600"

# The task code, at the pushed branch.
[ -d "$VLM/.git" ] || git clone -q git@github.com:VISTA-Stanford/vista_eval_vlm.git "$VLM"
git -C "$VLM" fetch -q origin "$BRANCH"
if [ -d "$TASK_WT" ]; then
    [ -z "$(git -C "$TASK_WT" status --porcelain)" ] || fail "$TASK_WT has uncommitted changes; not touching it"
    git -C "$TASK_WT" checkout -q --detach "origin/$BRANCH"
else
    git -C "$VLM" worktree add -q --detach "$TASK_WT" "origin/$BRANCH"
fi
PIN=$(sed -n 's/^ENGINE_PIN = "\([0-9a-f]\{40\}\)".*/\1/p' "$TASK_DIR/config.py")
[ -n "$PIN" ] || fail "no ENGINE_PIN in config.py"

# The engine, at the task's pin.
[ -d "$ENGINE/.git" ] || git clone -q git@github.com:philadamson93/agentic-label-opt.git "$ENGINE"
git -C "$ENGINE" fetch -q origin "$ENGINE_BRANCH"
if [ -d "$ENGINE_WT" ]; then
    [ -z "$(git -C "$ENGINE_WT" status --porcelain)" ] || fail "$ENGINE_WT has uncommitted changes; not touching it"
    git -C "$ENGINE_WT" checkout -q --detach "$PIN"
else
    git -C "$ENGINE" worktree add -q --detach "$ENGINE_WT" "$PIN"
fi
[ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || fail "engine is not at the pin $PIN"
echo "task $(git -C "$TASK_WT" rev-parse --short HEAD), engine $(git -C "$ENGINE_WT" rev-parse --short HEAD)"

# The patients' records, on local disk (tens of thousands of small files are hours on gcsfuse). Only patients/<id>/
# folders are rebuilt; the exporter's registry_dates/ side file is not in the archive, and staging refuses any file
# outside the record whitelist (agentic.record_files).
if [ ! -f "$TASK_DATA_DIR/pools.json" ] || [ ! -d "$TASK_DATA_DIR/patients" ]; then
    rm -rf "$TASK_DATA_DIR/patients"
    (cd "$TASK_DIR" && OMP_NUM_THREADS=2 nice -n 19 uv run -q --no-project --with "duckdb==1.5.6" --with pandas --with pyarrow \
        python -B scripts/build_data.py rehydrate) || fail "rehydrate (records from $DATA/archive into $TASK_DATA_DIR)"
fi
[ ! -e "$TASK_DATA_DIR/registry_dates" ] || fail "$TASK_DATA_DIR/registry_dates exists: raw registry dates must not be in the task copy"
n=$(find "$TASK_DATA_DIR/patients" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" -eq 5549 ] || fail "expected 5549 patient folders in $TASK_DATA_DIR/patients, found $n"
cmp -s "$DATA/pools.json" "$TASK_DATA_DIR/pools.json" || fail "local pools.json differs from the mount's"
echo "records: $n patients in $TASK_DATA_DIR"

# Images, this machine's pins, the token, and the $0 dry run. A rebuild replaces the image under the same tag, so
# a run of this task already going on this machine would lose its pinned image (it did, 2026-10-20): refuse.
for pid in $(pgrep -f "run.py run --run-id" || true); do
    tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -qx run.py || continue  # run.py as its own argument
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in */optimizer/died-2yr-ehr) fail "a died-2yr-ehr run is live here (pid $pid); setup would rebuild its image";; esac
done
alo build-images >/dev/null || fail "build-images"
[ -d "$TASK_WORKDIR/repo" ] || alo init || fail "init"
alo pins --write | tail -1 || fail "pins --write"
alo check-token | grep -q ": PASS" || fail "check-token did not PASS (try: cd $TASK_DIR && uv run --project $ENGINE_WT python run.py check-token)"
out=$(alo dry-run 2>&1) || fail "dry-run exited non-zero"
echo "$out" | grep -q "^FINISHED" && ! echo "$out" | grep -q " FAIL " || { echo "$out" | tail -20; fail "dry-run"; }
echo "$out" | grep "best program-v0"

mkdir -p "$RESULTS"
(cd "$TASK_DIR" && uv run -q --no-project --with pandas --with pyarrow --with scikit-learn python -B scripts/score_baselines.py \
    score --ids pool --out "$RESULTS/baselines_val_pool.json") | tail -9 || fail "score_baselines"

if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: run not started)"; exit 0; fi
mkdir -p "$TASK_WORKDIR"
cp "$TASK_DIR/scripts/run_on_vm.sh" "$TASK_WORKDIR/run_on_vm.sh"
setsid nohup env RUN_ID="$RUN_ID" ITERATIONS="$ITERATIONS" MAX_RESUMES="$MAX_RESUMES" bash "$TASK_WORKDIR/run_on_vm.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: run $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time); log $LOG; results to $RESULTS"
