#!/usr/bin/env bash
# Set up and run one tb-questions climb on this VM, judging every step; then run it detached, resuming after network
# stops. Copied from ct-findings-5's launcher (2026-11-20). The spec runs the treatment climb on phil-sllm-02 and the
# progression (imaging) climb on phil-sllm-03, on Vertex.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/tb-agent-questions:optimizer/tb-questions/scripts/run_on_vm.sh > ~/run_tb_questions.sh && TB_AGENT=treatment bash ~/run_tb_questions.sh
#
# Needs: Docker, the su-vista-uscentral1 mount, uv, git access to both repos, and the machine's optimizer login
# (/home/optimizer/.config/gcloud/application_default_credentials.json; never read here, only checked to exist).
# Settings, all optional, on the same line: RUN_ID (r1), ITERATIONS (10), PROGRAM_MAX_CONCURRENT (4 treatment, 2
# imaging), MAX_RESUMES (20), START (a version tag to start a fresh run from); SETUP_ONLY=1 stops before the zero-shot
# pass and the run; SKIP_ZERO_SHOT=1 skips the zero-shot pass. The zero-shot VAL pass (the seed on every VAL patient,
# paid, ~$6 treatment / ~$20 imaging) runs first unless a zero-shot result for this agent is already on the mount; if
# the spec's stop rule fires (failures over 10%, or the treatment agent at chance on both questions) it stops before
# the climb. The climb stops after 2 rounds with no new best VAL (which ends it as finished), 10 rounds, or past the
# caps ($40 a round / $150 a run treatment; $60 / $250 imaging).
# Pass: it prints "SETUP PASS" and the log path; the log ends in "FINISHED"; results reach
# /mnt/su-vista-uscentral1/chaudhari_lab/phil/tb-questions-v1_7-runs/<host>/<agent>/<run id>/ after every stop.
set -euo pipefail

: "${TB_AGENT:?set TB_AGENT=treatment, imaging, treatment-record or boards-record}"
export TB_AGENT
export TB_MODEL=${TB_MODEL:-sonnet}  # boards only: sonnet or haiku (config.TB_MODEL)
export TB_QUESTIONS=${TB_QUESTIONS:-board}  # boards only: board or treatment (config.TB_QUESTIONS)
export TB_SEED=${TB_SEED:-v1}  # treatment only: v1 or untuned (config.TB_SEED)
LABEL=$TB_AGENT$([ "$TB_QUESTIONS" = treatment ] && echo -tx || true)$([ "$TB_SEED" = untuned ] && echo -untuned || true)$([ "$TB_MODEL" = haiku ] && echo -haiku || true)  # config.RUN_LABEL: work folder, images, results
export ALO_ROUTE=${ALO_ROUTE:-vertex}  # the spec's route
BRANCH=feat/tb-agent-questions
[ "$TB_AGENT" != treatment-record ] || BRANCH=feat/tb-treatment-record  # the full-record arm's branch (spec 2026-11-21)
[ "$TB_AGENT" != boards-record ] || BRANCH=feat/tb-board-questions  # the board questions agent (spec 2026-11-22)
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-tb-questions-run}
[ "$TB_AGENT" != treatment-record ] || TASK_WT=${TASK_WT_RECORD:-$HOME/code/vista_eval_vlm-tb-record-run}  # its own checkout: the summary runs keep theirs
[ "$TB_AGENT" != boards-record ] || TASK_WT=${TASK_WT_BOARDS:-$HOME/code/vista_eval_vlm-tb-boards-run}
[ "$TB_QUESTIONS" != treatment ] || TASK_WT=${TASK_WT_TX:-$HOME/code/vista_eval_vlm-tb-tx-run}  # its own checkout: never move a live climb's code
TASK_DIR=$TASK_WT/optimizer/tb-questions
ENGINE=$HOME/code/agentic-label-opt
ENGINE_WT=$ENGINE/.claude/worktrees/mortality-engine
DATA=/mnt/su-vista-uscentral1/chaudhari_lab/phil/tb-questions-v1_7/$([ "$TB_QUESTIONS" = treatment ] && echo boards-tx || echo "$TB_AGENT")
RESULTS_ROOT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/tb-questions-v1_7-runs
RUN_ID=${RUN_ID:-r1}
ITERATIONS=${ITERATIONS:-$([ "$TB_AGENT" = boards-record ] && echo 8 || echo 10)}  # specs; the stop rule usually ends it sooner
MAX_RESUMES=${MAX_RESUMES:-20}
if [ "$TB_AGENT" != imaging ]; then export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-4}; else export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}; fi
export AGENTIC_LABEL_OPT=$ENGINE_WT
export TASK_WORKDIR=${TASK_WORKDIR:-$HOME/.cache/alo-tasks/tb-questions-$LABEL}  # config.WORKDIR's default
RESULTS=$RESULTS_ROOT/$(hostname)/$LABEL/$RUN_ID
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
if [ "$TB_AGENT" = treatment-record ] || [ "$TB_AGENT" = boards-record ]; then  # the record folders are local (build_data.py record-export)
    REC=${TASK_RECORD_DIR:-$HOME/data/tb-questions-${TB_AGENT%-record}-record}
    [ -d "$REC/patients" ] || fail "no record folders at $REC/patients: run build_data.py record-export on this machine"
fi
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
    [ "$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TB_AGENT=//p')" = "$TB_AGENT" ] || continue
    pm=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TB_MODEL=//p')
    [ "${pm:-sonnet}" = "$TB_MODEL" ] || continue  # the Sonnet and Haiku arms have their own images
    pq=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TB_QUESTIONS=//p')
    [ "${pq:-board}" = "$TB_QUESTIONS" ] || continue  # the treatment climb has its own images too
    ps_=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^TB_SEED=//p')
    [ "${ps_:-v1}" = "$TB_SEED" ] || continue
    case "$(readlink "/proc/$pid/cwd" 2>/dev/null)" in */optimizer/tb-questions) fail "a tb-questions $TB_AGENT run is live here (pid $pid); setup would rebuild its image";; esac
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
if [ -n "${SETUP_ONLY:-}" ]; then echo "SETUP PASS (SETUP_ONLY: zero-shot pass and run not started)"; exit 0; fi

# The zero-shot VAL pass (spec, Done when), unless one for this agent is already on the mount (from any host).
done_zs=$(ls "$RESULTS_ROOT"/*/"$LABEL"/zero-shot/zero_shot_val.json 2>/dev/null | head -1 || true)
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
setsid nohup env TB_AGENT="$TB_AGENT" TB_MODEL="$TB_MODEL" TB_QUESTIONS="$TB_QUESTIONS" TB_SEED="$TB_SEED" ALO_ROUTE="$ALO_ROUTE" RUN_ID="$RUN_ID" ITERATIONS="$ITERATIONS" MAX_RESUMES="$MAX_RESUMES" START="${START:-}" bash "$TASK_WORKDIR/run_on_vm.sh" --loop > "$LOG" 2>&1 < /dev/null &
echo "SETUP PASS: run $RUN_ID started detached ($PROGRAM_MAX_CONCURRENT at a time); log $LOG; results to $RESULTS"
