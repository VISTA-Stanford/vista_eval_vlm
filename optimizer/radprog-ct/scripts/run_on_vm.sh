#!/usr/bin/env bash
# Set up and run the radprog-ct climb on this VM, judging every step; then run it detached, resuming after network
# stops. Copied 2026-11-21 from tb-questions' launcher (itself from ct-findings-5's). Spec
# planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md runs the climb on phil-sllm-02, on Vertex.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/radprog-vlm-agent:optimizer/radprog-ct/scripts/run_on_vm.sh > ~/run_radprog_ct.sh && bash ~/run_radprog_ct.sh
#
# Needs: Docker, the su-vista-uscentral1 mount, uv, git access to both repos, and the machine's optimizer login
# (/home/optimizer/.config/gcloud/application_default_credentials.json; never read here, only checked to exist).
# Settings, all optional, on the same line: RUN_ID (r1), ITERATIONS (10), PROGRAM_MAX_CONCURRENT (2), MAX_RESUMES (20),
# START (a version tag to start a fresh run from); SETUP_ONLY=1 stops before the zero-shot pass and the run;
# SKIP_ZERO_SHOT=1 skips the zero-shot pass. The zero-shot VAL pass (the seed on all 200 VAL pairs, paid, ~$20-40) runs
# first unless a zero-shot result is already on the mount; if more than 10% of its sessions fail it stops before the
# climb. The climb stops after 2 rounds with no new best VAL (which ends it as finished), 10 rounds, or past the caps
# ($60 a round / $250 a run).
# Disk flushing (the tumor board imaging climb's fix, sllm-03 2026-11-21): a dry run or a round writes GBs that sit
# unflushed; the next container start then waits on the flush and the engine's 30 s proxy start times out. So setup
# runs `sync` after the dry run, and the detached loop keeps a `sync` every 5 s alive while it runs.
# Pass: it prints "SETUP PASS" and the log path; the log ends in "FINISHED"; results reach
# /mnt/su-vista-uscentral1/chaudhari_lab/phil/radprog-ct-v1_7-runs/<host>/<run id>/ after every stop.
set -euo pipefail

export ALO_ROUTE=${ALO_ROUTE:-vertex}  # the spec's route
BRANCH=feat/radprog-vlm-agent
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-radprog-ct-run}
TASK_DIR=$TASK_WT/optimizer/radprog-ct
ENGINE=$HOME/code/agentic-label-opt
ENGINE_WT=$ENGINE/.claude/worktrees/mortality-engine
DATA=/mnt/su-vista-uscentral1/chaudhari_lab/phil/radprog-ct-v1_7
RESULTS_ROOT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/radprog-ct-v1_7-runs
RUN_ID=${RUN_ID:-r1}
ITERATIONS=${ITERATIONS:-10}  # spec; the stop rule usually ends it sooner
MAX_RESUMES=${MAX_RESUMES:-20}
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export TASK_WORKDIR=${TASK_WORKDIR:-$HOME/.cache/alo-tasks/radprog-ct}  # config.WORKDIR's default
RESULTS=$RESULTS_ROOT/$(hostname)/$RUN_ID
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
    # Detached part: run, and resume after a network stop (a resume reruns only unfinished calls). The flush loop
    # lives as long as this shell (see the header).
    ( while kill -0 $$ 2>/dev/null; do sync; sleep 5; done ) &
    attempt=0
    mode=()
    [ -n "${START:-}" ] && mode=(--start "$START")  # fresh run from a chosen version tag (e.g. program-v1)
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
if [ "$ALO_ROUTE" = vertex ]; then
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
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in */optimizer/radprog-ct) fail "a radprog-ct run is live here (pid $pid); setup would rebuild its image";; esac
done
alo build-images >/dev/null || fail "build-images"
[ -d "$TASK_WORKDIR/repo" ] || alo init || fail "init"
alo pins --write | tail -1 || fail "pins --write"
if [ "$ALO_ROUTE" != vertex ]; then
    alo check-token | grep -q ": PASS" || fail "check-token did not PASS (try: cd $TASK_DIR && uv run --project $ENGINE_WT python run.py check-token)"
fi
rc=0; out=$(alo dry-run 2>&1) || rc=$?
[ "$rc" = 0 ] || [ "$rc" = 3 ] || fail "dry-run exited $rc"  # 3 = stopped (the probability stop rule can end it)
echo "$out" | grep -qE "^FINISHED|^STOPPED \(no_new_best\)" && ! echo "$out" | grep -q " FAIL " || { echo "$out" | tail -20; fail "dry-run"; }
echo "$out" | grep -E "^(FINISHED|STOPPED)" || true
sync  # flush the dry run's GBs now, not during the next container start (see the header)
if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: zero-shot pass and run not started)"; exit 0; fi

# The zero-shot VAL pass (spec, Done when), unless one for this agent is already on the mount (from any host).
done_zs=$(ls "$RESULTS_ROOT"/*/zero-shot/zero_shot_val.json 2>/dev/null | head -1 || true)
if [ -n "${SKIP_ZERO_SHOT:-}" ]; then
    echo "zero-shot pass skipped (SKIP_ZERO_SHOT)"
elif [ -n "$done_zs" ]; then
    # Re-check its stop rule: a stopped zero-shot leaves its file too, and a re-paste must not skip the gate.
    python3 -c 'import json,sys; g=json.load(open(sys.argv[1])).get("gate") or {}; sys.exit(0 if g.get("passed") is True else 1)' "$done_zs" \
        || fail "zero-shot at $done_zs did not pass the spec's stop rule (or has no verdict): stop and come back"
    echo "zero-shot already scored and passed: $done_zs"
else
    zs=0; alo zero-shot > "$TASK_WORKDIR/zero-shot.log" 2>&1 || zs=$?
    tail -3 "$TASK_WORKDIR/zero-shot.log"
    [ "$zs" = 0 ] || fail "zero-shot exited $zs (3 = the spec's stop-and-come-back rule fired); log $TASK_WORKDIR/zero-shot.log"
fi
mkdir -p "$TASK_WORKDIR"
cp "$TASK_DIR/scripts/run_on_vm.sh" "$TASK_WORKDIR/run_on_vm.sh"
setsid nohup env ALO_ROUTE="$ALO_ROUTE" RUN_ID="$RUN_ID" ITERATIONS="$ITERATIONS" MAX_RESUMES="$MAX_RESUMES" START="${START:-}" bash "$TASK_WORKDIR/run_on_vm.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: run $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time); log $LOG; results to $RESULTS"
