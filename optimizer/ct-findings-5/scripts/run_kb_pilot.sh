#!/usr/bin/env bash
# The CT knowledge-base pilot on phil-sllm-02, Vertex (spec planning/vista_eval_vlm/2026-11-23-ct-knowledge-base-pilot.md,
# Approved 2026-11-23: the cost check and the full pilot within $200 are its approved Expensive steps). In order, judging
# each and stopping on the first failure:
#   1. setup: host, engine pin, Vertex login, disk; the KB data check (scripts/build_kb_data.py check)
#   2. the work folder, the builder's local source copy, pins (scripts/run_kb_loop.py init / pins)
#   3. cost check: one builder session + 10 tester scans (run id kb-costcheck). Stops if a tester scan costs more than
#      MAX_PER_SCAN, fewer than half the 10 opened an entry, the library is almost all text-only entries, or the
#      projected pilot (5 builder sessions + 500 tester scans + 380 VAL scans) passes CAP_USD (spec stops)
#   4. the loop: 5 rounds of builder + 100 TRAIN tester scans (run id kb-r1), best round by the targets' TRAIN AUROC
#   5. VAL, the 190 with organ ranges: the empty library (program-v0) and the best round's library, one pass each
#   6. $0: library use on VAL, the paired comparison and the fusion with the probe and vote
# Status line and every result in ct-findings-5-v1_7-runs/knowledge-base/ on the mount. Re-running resumes.
#
#   git -C ~/code/vista_eval_vlm fetch -q origin && git -C ~/code/vista_eval_vlm show origin/feat/ct-knowledge-base:optimizer/ct-findings-5/scripts/run_kb_pilot.sh > ~/run_kb_pilot.sh && tmux new -d -s kb-pilot 'bash ~/run_kb_pilot.sh 2>&1 | tee ~/kb-pilot.log'
#
# Settings, optional, on the same line: CAP_USD (200, the whole pilot), MAX_PER_SCAN (0.40), PROGRAM_MAX_CONCURRENT (3).
set -euo pipefail

BRANCH=feat/ct-knowledge-base
VLM=${VLM_REPO:-$HOME/code/vista_eval_vlm}
TASK_WT=${TASK_WT:-$HOME/code/vista_eval_vlm-kb-run}
TASK_DIR=$TASK_WT/optimizer/ct-findings-5
ENGINE_WT=$HOME/code/agentic-label-opt/.claude/worktrees/mortality-engine
RUNS=/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs
OUT=$RUNS/knowledge-base
CAP_USD=${CAP_USD:-200}
MAX_PER_SCAN=${MAX_PER_SCAN:-0.40}
export KB_ARM=1
export ALO_ROUTE=${ALO_ROUTE:-vertex}
export AGENTIC_LABEL_OPT=$ENGINE_WT
export PROGRAM_MAX_CONCURRENT=${PROGRAM_MAX_CONCURRENT:-3}
mkdir -p "$OUT"

status() { echo "Status: $1 — $2" > "$OUT/STATUS.txt"; echo "$(date -u +%FT%TZ) $1 — $2" | tee -a "$OUT/progress.log"; }
fail() { status failed "$*"; exit 1; }
py() { (cd "$TASK_DIR" && uv run -q --project "$ENGINE_WT" --with pillow --with numpy python -B "$@"); }
kb() { py scripts/run_kb_loop.py "$@"; }
json_get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {'d': d}))" "$@"; }
check() { python3 -c "import sys; sys.exit(0 if eval(sys.argv[1]) else 1)" "$1"; }
WORK=$HOME/.cache/alo-tasks/ct-findings-5-kb
state() { echo "$WORK/private/kb-loop/$1/state.json"; }

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
[ "$free_gb" -ge "${MIN_FREE_GB:-40}" ] || fail "only ${free_gb} GB free under ~/.cache; need ${MIN_FREE_GB:-40} (source copy 11 GB + pruned call folders)"
(cd "$TASK_DIR" && uv run -q --no-project --with pandas --with pyarrow python -B scripts/build_kb_data.py check) > "$OUT/data_check.txt" 2>&1 \
    || fail "KB data check failed (data_check.txt)"

# Delete each finished session's staged CT copy as the run goes (shared disk; spec 2026-11-23 build note). Peak with it:
# the 11 GB source copy plus the few sessions in flight; without it ~45 GB by the end.
bash "$TASK_DIR/scripts/prune_kb_copies.sh" > "$OUT/prune.log" 2>&1 &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

status running "work folder, source copy and pins"
kb init > "$OUT/init.log" 2>&1 || fail "init failed (init.log)"
kb pins > "$OUT/pins.log" 2>&1 || fail "builder pin failed (pins.log)"
if ! grep -q '"builder"' "$WORK/repo/manifest.json"; then kb pins --write >> "$OUT/pins.log" 2>&1 || fail "builder pin write failed"; fi

# 3. cost check
if [ -f "$OUT/costcheck.ok" ]; then
    echo "cost check already passed: $(cat "$OUT/costcheck.ok")"
else
    status running "cost check: one builder session + 10 tester scans"
    kb loop --run-id kb-costcheck --rounds 1 --tester-scans 10 --cap "$CAP_USD" --already-spent "$(cat "$OUT/prior_spent_usd.txt" 2>/dev/null || echo 0)" > "$OUT/costcheck.log" 2>&1 \
        || fail "cost check stopped (costcheck.log)"
    kb report --run-id kb-costcheck --out "$OUT/costcheck.json" > /dev/null
    S=$(state kb-costcheck)
    b=$(json_get "$S" "d['rounds']['1']['builder_usd']")
    t=$(json_get "$S" "d['rounds']['1']['tester_usd']")
    n=$(json_get "$S" "d['rounds']['1']['tester_sessions']")
    per=$(python3 -c "print(round($t / max($n, 1), 4))")
    share=$(json_get "$S" "d['rounds']['1']['usage']['share_opened_an_entry']")
    entries=$(json_get "$S" "d['rounds']['1']['library']['entries']")
    textonly=$(json_get "$S" "d['rounds']['1']['library']['text_only']")
    echo "cost check: builder \$$b, tester \$$per/scan over $n, opened an entry $share, library $entries entries ($textonly text-only)"
    check "$per <= $MAX_PER_SCAN" || fail "tester costs \$$per per scan, above the \$$MAX_PER_SCAN stop"
    check "$share >= 0.5" || fail "only $share of the cost-check scans opened an entry (spec stop: the tester does not use the library)"
    check "$entries > 0 and $textonly <= 0.8 * $entries" || fail "library of $entries entries, $textonly text-only (spec stop: not an image test)"
    proj=$(python3 -c "print(round($b + $t + 5 * $b * 1.2 + $per * 1.15 * (500 + 380), 2))")
    check "$proj <= $CAP_USD" || fail "projected pilot \$$proj passes the \$$CAP_USD cap"
    echo "builder $b tester_per_scan $per projected $proj" > "$OUT/costcheck.ok"
fi
# Spend before this work folder (an earlier attempt's cost check, prior_spent_usd.txt on the mount) counts toward the cap.
prior=$(cat "$OUT/prior_spent_usd.txt" 2>/dev/null || echo 0)
cc_spent=$(json_get "$(state kb-costcheck)" "$prior + sum(float(r.get('builder_usd') or 0) + float(r.get('tester_usd') or 0) for r in d['rounds'].values())")

# 4. the loop
status running "the loop: 5 rounds (cost check spent \$$cc_spent)"
kb loop --run-id kb-r1 --rounds 5 --tester-scans 100 --cap "$CAP_USD" --already-spent "$cc_spent" > "$OUT/loop.log" 2>&1 \
    || fail "the loop stopped (loop.log)"
kb report --run-id kb-r1 --out "$OUT/loop.json" > /dev/null
best=$(json_get "$(state kb-r1)" "d['best']['tag']")
loop_spent=$(json_get "$(state kb-r1)" "d['spent_usd']")
mkdir -p "$OUT/tester-rounds"
for d in "$WORK"/kb-tester/round-*; do
    mkdir -p "$OUT/tester-rounds/$(basename "$d")"
    cp "$d/summary.json" "$d/entry_use.json" "$OUT/tester-rounds/$(basename "$d")/"
done
# Straight into its own folder: the mount (gcsfuse) cannot rename a folder, which killed the first run here (2026-11-23).
rm -rf "$OUT/library-best" "$OUT/library" && mkdir -p "$OUT/library-best" \
    && git -C "$WORK/repo" archive "$best" library | tar -x -C "$OUT/library-best" --strip-components=1 \
    || fail "copying the best library to the mount failed"
echo "best $best; loop spent \$$loop_spent"

# 5. VAL: empty library and best library, one pass each
builders=$(python3 -c "import json,sys; print(round(float(sys.argv[1]) + sum(float(r.get('builder_usd') or 0) for f in sys.argv[2:] for r in json.load(open(f))['rounds'].values()), 4))" "$(cat "$OUT/prior_spent_usd.txt" 2>/dev/null || echo 0)" "$(state kb-costcheck)" "$(state kb-r1)")
total_spent() {  # builder sessions + every tester session this work folder paid for (its receipts)
    py scripts/cost_report.py "$WORK" "$OUT/cost-work-folder.json" > /dev/null || true
    python3 -c "import json,sys; print(round(float(sys.argv[1]) + json.load(open(sys.argv[2]))['patient_sessions_usd'], 2))" "$builders" "$OUT/cost-work-folder.json"
}
spent=$(total_spent)
echo "spent so far \$$spent"
for arm in empty library; do
    tag=program-v0; [ "$arm" = library ] && tag=$best
    if [ -f "$OUT/val-$arm.json" ]; then echo "VAL $arm already done"; continue; fi
    per=$(json_get "$(state kb-r1)" "sum(r['tester_usd'] for r in d['rounds'].values()) / sum(r['tester_sessions'] for r in d['rounds'].values())")
    check "$spent + 1.15 * $per * 190 <= $CAP_USD" || fail "VAL $arm (~\$$(python3 -c "print(round($per*190,2))")) would pass the cap (spent \$$spent)"
    status running "VAL $arm library ($tag) on the 190"
    ok=0
    for attempt in 1 2 3 4 5; do  # a network stop kills a pass; the same command re-runs only unfinished calls
        if (cd "$TASK_DIR" && TASK_WORKDIR=$WORK uv run -q --project "$ENGINE_WT" python run.py val-pass --tag "$tag") > "$OUT/val-$arm.log" 2>&1; then ok=1; break; fi
        echo "VAL $arm attempt $attempt stopped; retrying in 60 s"; sleep 60
    done
    [ "$ok" = 1 ] || fail "VAL $arm did not finish after 5 attempts (val-$arm.log)"
    ans=$(python3 -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1])]; print([x for x in e if x['tag']==sys.argv[2]][-1]['answers'])" "$WORK/private/val_pass_ledger.jsonl" "$tag")
    py scripts/run_kb_loop.py usage "$ans" --out "$OUT/usage-val-$arm.json" > /dev/null
    cp "$ans" "$OUT/val-$arm.json.tmp" && mv "$OUT/val-$arm.json.tmp" "$OUT/val-$arm.json"
    spent=$(total_spent)
    echo "VAL $arm done; spent so far \$$spent"
done

# 6. $0: comparison and fusion
status running "comparison and fusion"
py scripts/retrieval_usage.py arms --pass "kb-library=$OUT/val-library.json" --pass "kb-empty=$OUT/val-empty.json" \
    --pass "arm-D=$RUNS/example-ct-arms/arm-d.json" --pair "kb-library:kb-empty" --pair "kb-library:arm-D" --out "$OUT/arms" \
    > "$OUT/arms.log" 2>&1 || fail "paired comparison failed (arms.log)"
(cd "$TASK_DIR" && OMP_NUM_THREADS=4 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with numpy \
    --with scikit-learn python -B scripts/step0_curves.py --kb-arms) > "$OUT/fusion.log" 2>&1 || fail "fusion failed (fusion.log)"
status done "pilot finished on $(hostname): best $best; spent \$$(total_spent); results in knowledge-base/ (arms/, fusion.json, loop.json); readback next"
