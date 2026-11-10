#!/usr/bin/env bash
# Set up and run the ct-findings-5 climb on this VM, judging every step; then run it detached, resuming after network
# stops. Copied from died-2yr-ct's launcher (2026-11-10). The spec runs it from phil-sllm-02 on Vertex.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-findings-agentic:optimizer/ct-findings-5/scripts/run_on_vm.sh > ~/run_ct_findings.sh && ALO_ROUTE=vertex bash ~/run_ct_findings.sh
#
# Needs: Docker, the su-vista-uscentral1 mount, uv, git access to both repos, and on the seat route (the default) the
# Claude token at ~/.config/alo/oauth_token (mode 600); with ALO_ROUTE=vertex, the machine's optimizer login
# (/home/optimizer/.config/gcloud/application_default_credentials.json; never read here, only checked to exist).
# Settings, all optional, on the same line: RUN_ID (prob-r1), ITERATIONS (10), PROGRAM_MAX_CONCURRENT (2),
# MAX_RESUMES (20); SETUP_ONLY=1 stops before the run. The run stops after 2 rounds with no new best VAL (which ends it
# as finished) or past $50 a round / $200 a run; setup also scores the CT probes on the same VAL pool into
# baselines_val_pool.json (spec 2026-11-10).
# Pass: it prints "SETUP PASS" and the log path; the log ends in "FINISHED"; results reach
# /mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs/<host>/<run id>/ after every stop.
set -euo pipefail

BRANCH=feat/ct-findings-agentic
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-ct-findings}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE=$HOME/code/agentic-label-opt
ENGINE_WT=$ENGINE/.claude/worktrees/mortality-engine
DATA=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7
RUN_ID=${RUN_ID:-prob-r1}
ITERATIONS=${ITERATIONS:-10}  # spec; the stop rule usually ends it sooner
MAX_RESUMES=${MAX_RESUMES:-20}
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}  # spec: 2 at a time
export AGENTIC_LABEL_OPT=$ENGINE_WT
export TASK_WORKDIR=${TASK_WORKDIR:-$HOME/.cache/alo-tasks/ct-findings-5}  # config.WORKDIR's default
RESULTS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs/$(hostname)/$RUN_ID
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
        if grep -q "STOPPED (no_new_best)" "$TASK_WORKDIR/last-pass.log"; then echo "FINISHED (stop rule: no new best)"; exit 0; fi
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
if [ "${ALO_ROUTE:-anthropic}" = vertex ]; then
    LOGIN=/home/optimizer/.config/gcloud/application_default_credentials.json
    [ -f "$LOGIN" ] && [ ! -L "$LOGIN" ] || fail "no optimizer login at $LOGIN (or run on the seat: unset ALO_ROUTE)"
else
    TOKEN=$HOME/.config/alo/oauth_token
    [ -f "$TOKEN" ] || fail "no Claude token at $TOKEN: run claude setup-token, save it there with mode 600"
    [ "$(stat -c %a "$TOKEN")" = 600 ] || fail "$TOKEN must be mode 600"
fi

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
git -C "$ENGINE" fetch -q origin  # every branch: the pin sits on feat/vertex-login-plus-seat-pause, not main (2026-11-10)
if [ -d "$ENGINE_WT" ]; then
    [ -z "$(git -C "$ENGINE_WT" status --porcelain)" ] || fail "$ENGINE_WT has uncommitted changes; not touching it"
    git -C "$ENGINE_WT" checkout -q --detach "$PIN"
else
    git -C "$ENGINE" worktree add -q --detach "$ENGINE_WT" "$PIN"
fi
[ "$(git -C "$ENGINE_WT" rev-parse HEAD)" = "$PIN" ] || fail "engine is not at the pin $PIN"
echo "task $(git -C "$TASK_WT" rev-parse --short HEAD), engine $(git -C "$ENGINE_WT" rev-parse --short HEAD)"

# Images, this machine's pins, the token, and the $0 dry run. A rebuild replaces the image under the same tag, so
# a run of this task already going on this machine would lose its pinned image (it did, 2026-10-20): refuse.
for pid in $(pgrep -f "run.py run --run-id" || true); do
    tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -qx run.py || continue  # run.py as its own argument
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in */optimizer/ct-findings-5) fail "a ct-findings-5 run is live here (pid $pid); setup would rebuild its image";; esac
done
alo build-images >/dev/null || fail "build-images"
[ -d "$TASK_WORKDIR/repo" ] || alo init || fail "init"
alo pins --write | tail -1 || fail "pins --write"
if [ "${ALO_ROUTE:-anthropic}" != vertex ]; then
    alo check-token | grep -q ": PASS" || fail "check-token did not PASS (try: cd $TASK_DIR && uv run --project $ENGINE_WT python run.py check-token)"
fi
rc=0; out=$(alo dry-run 2>&1) || rc=$?
[ "$rc" = 0 ] || [ "$rc" = 3 ] || fail "dry-run exited $rc"  # 3 = stopped (the probability stop rule can end it)
echo "$out" | grep -qE "^FINISHED|^STOPPED \(no_new_best\)" && ! echo "$out" | grep -q " FAIL " || { echo "$out" | tail -20; fail "dry-run"; }
echo "$out" | grep "best program-v0"
mkdir -p "$RESULTS"  # the comparison the spec reports the winner against, on the same 200 VAL scans
(cd "$TASK_DIR" && uv run -q --no-project --with pandas --with pyarrow --with scikit-learn python -B scripts/score_baselines.py score --ids pool \
    --out "$RESULTS/baselines_val_pool.json") | tail -4 || fail "score_baselines"

if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: run not started)"; exit 0; fi
mkdir -p "$TASK_WORKDIR"
cp "$TASK_DIR/scripts/run_on_vm.sh" "$TASK_WORKDIR/run_on_vm.sh"
setsid nohup env RUN_ID="$RUN_ID" ITERATIONS="$ITERATIONS" MAX_RESUMES="$MAX_RESUMES" bash "$TASK_WORKDIR/run_on_vm.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: run $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time); log $LOG; results to $RESULTS"
