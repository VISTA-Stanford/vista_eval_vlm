#!/usr/bin/env bash
# The example-CT arms on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-22-ct-findings-example-ct-arms.md,
# Approved 2026-11-22; these paid passes are its approved Expensive step). For each arm A, B, C in turn: a 10-scan cost
# check, its gates, then the VAL pass (A and B on 200 scans, C on the 190 with organ ranges). Judges every step and
# stops on the first failure; writes a status line and every result to the mount.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-example-arms:optimizer/ct-findings-5/scripts/run_example_arms.sh > ~/run_example_arms.sh && tmux new -d -s example-arms 'bash ~/run_example_arms.sh 2>&1 | tee ~/example-arms.log'
#
# Settings, optional, on the same line: ARMS ("A B C"), CAP_USD (75: total across cost checks and passes; a pass is
# not started if the cost check's cost per scan says it would cross the cap), MAX_PER_SCAN (0.30: the spec's stop),
# PROGRAM_MAX_CONCURRENT (2, as 11-17). Re-running resumes: a finished arm (its answers on the mount) is skipped, and
# an unfinished pass re-runs only its unfinished calls.
# Stops (spec): cost check failed or above $0.30/scan; in A or C fewer than half the cost-check sessions opened a
# neighbour CT; the cap. Pass: STATUS.txt reads "Status: done"; answers and usage in ct-findings-5-v1_7-runs/example-ct-arms/.
set -euo pipefail

BRANCH=feat/ct-example-arms
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-example-arms}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/example-ct-arms
ARMS=${ARMS:-"A B C"}
CAP_USD=${CAP_USD:-75}
MAX_PER_SCAN=${MAX_PER_SCAN:-0.30}
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-2}
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
alo() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" python run.py "$@"); }
py() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" python -B "$@"); }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
ledger_answers() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['answers'])" "$1/private/val_pass_ledger.jsonl"; }
spent_file=$OUT/spent_usd.txt
[ -f "$spent_file" ] || echo 0 > "$spent_file"
add_spent() { python3 -c "import sys; print(round(float(open(sys.argv[1]).read()) + float(sys.argv[2]), 2))" "$spent_file" "$1" > "$spent_file.tmp" && mv "$spent_file.tmp" "$spent_file"; }

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
if [ "$ALO_ROUTE" = vertex ]; then
    [ -e /home/optimizer/.config/gcloud/application_default_credentials.json ] || fail "no optimizer login for Vertex"
fi
free_gb=$(df -BG --output=avail "$HOME/.cache" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "${MIN_FREE_GB:-60}" ] || fail "only ${free_gb} GB free under ~/.cache; need ${MIN_FREE_GB:-60} (cache ~48 GB + call folders)"
python3 -B "$TASK_DIR/scripts/build_organ_ranges.py" check > "$OUT/organ_check.txt" || fail "organ ranges check failed (organ_check.txt)"
status running "copying neighbour CTs into the local cache (one copy each)"
n=$(cd "$TASK_DIR" && EXAMPLE_ARM=A uv run -q --project "$ENGINE_WT" python -B -c "import agentic; print(agentic.warm_cache())")
echo "cached $n neighbours"

for arm in $ARMS; do
    a=$(echo "$arm" | tr 'A-Z' 'a-z')
    if [ -f "$OUT/arm-$a.json" ]; then echo "arm $arm already finished; skipping"; continue; fi
    export EXAMPLE_ARM=$arm
    nscans=200; [ "$arm" = C ] && nscans=190

    if [ -f "$OUT/costcheck-$a.ok" ]; then  # resume: this arm's cost check already passed its gates
    per=$(cat "$OUT/costcheck-$a.ok"); echo "arm $arm cost check already passed: \$$per per scan"
    else
    status running "arm $arm: 10-scan cost check"
    (cd "$TASK_DIR" && bash scripts/cost_check.sh) > "$OUT/costcheck-$a.log" 2>&1 || fail "arm $arm cost check failed (costcheck-$a.log)"
    ccwd=$(ls -td "$HOME"/.cache/alo-tasks/ct-findings-5-costcheck-* | head -1)
    cc="$RUNS/$(hostname)/$(basename "$ccwd" | sed 's/^ct-findings-5-//')"  # by name: mount folder times are unreliable
    [ -f "$cc/cost_report.json" ] || fail "arm $arm: no cost report at $cc"
    per=$(json_get "$cc/cost_report.json" "d['passes'][-1]['mean_usd']")
    add_spent "$(json_get "$cc/cost_report.json" "d['patient_sessions_usd']")"
    echo "arm $arm cost check: \$$per per scan; spent so far \$$(cat "$spent_file")"
    python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$per" "$MAX_PER_SCAN" \
        || fail "arm $arm costs \$$per per scan, above the \$$MAX_PER_SCAN stop"
    py scripts/retrieval_usage.py usage "$(ledger_answers "$ccwd")" --out "$OUT/usage-costcheck-$a" > /dev/null
    if [ "$arm" != B ]; then
        share=$(json_get "$OUT/usage-costcheck-$a/usage_summary.json" "d['share_viewed_a_neighbour_ct']")
        python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.5 else 1)" "$share" \
            || fail "arm $arm: only $share of cost-check sessions opened a neighbour CT (spec stop)"
    fi
    python3 -c "import sys; s,p,n,c=map(float,sys.argv[1:]); sys.exit(0 if s + 1.1*p*n <= c else 1)" \
        "$(cat "$spent_file")" "$per" "$nscans" "$CAP_USD" || fail "arm $arm would cross the \$$CAP_USD cap"
    echo "$per" > "$OUT/costcheck-$a.ok"
    fi

    status running "arm $arm: VAL pass on $nscans scans"
    export TASK_WORKDIR=$HOME/.cache/alo-tasks/ct-findings-5-examples-$a
    [ -d "$TASK_WORKDIR/repo" ] || alo init
    git -C "$TASK_WORKDIR/repo" log --format=%s | grep -q '^setup pins for' || alo pins --write | tail -1  # resume: already committed
    ok=0
    for attempt in 1 2 3 4 5; do  # a network stop kills a pass; the same command re-runs only unfinished calls
        if alo val-pass --tag program-v0; then ok=1; break; fi
        echo "arm $arm pass attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    [ "$ok" = 1 ] || fail "arm $arm VAL pass did not finish after 5 attempts"
    py scripts/cost_report.py "$TASK_WORKDIR" "$OUT/cost-$a.json" > /dev/null || true
    add_spent "$(json_get "$OUT/cost-$a.json" "d['patient_sessions_usd']")"
    py scripts/retrieval_usage.py usage "$(ledger_answers "$TASK_WORKDIR")" --out "$OUT/usage-$a" > /dev/null
    cp "$(ledger_answers "$TASK_WORKDIR")" "$OUT/arm-$a.json.tmp" && mv "$OUT/arm-$a.json.tmp" "$OUT/arm-$a.json"
    unset TASK_WORKDIR
    echo "arm $arm done; spent so far \$$(cat "$spent_file")"
done
status done "arms $ARMS finished on $(hostname); spent \$$(cat "$spent_file"); comparison and readback next (sllm-01)"
