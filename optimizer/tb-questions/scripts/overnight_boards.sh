#!/usr/bin/env bash
# Overnight run-through for the board questions agent, approved by Phil 2026-11-23 (cost checks, untuned VAL and both
# climbs, within the caps; TEST stays sealed). Each step judges itself; any failure stops here and says why in
# $OUT/progress.log on the mount. Order: record checks -> pruner -> cost checks (Sonnet, Haiku) -> Sonnet untuned VAL
# + stop rule + climb (detached) -> Haiku untuned VAL + climb (detached).
#
#   [FROM=launch] setsid nohup bash optimizer/tb-questions/scripts/overnight_boards.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/mnt/su-vista-uscentral1/chaudhari_lab/phil/tb-questions-v1_7-runs/$(hostname)/boards-record/overnight
CHECK_LOG=${CHECK_LOG:?set CHECK_LOG to the running check_task.py output}
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/progress.log"; }
fail() { say "STOPPED: $*"; exit 1; }
cd "$HERE"

say "start (host $(hostname), free disk $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' '))"
if [ "${FROM:-}" != launch ]; then  # FROM=launch: checks and cost checks already passed, start at the launches
while pgrep -f "[c]heck_task.py check" > /dev/null; do sleep 30; done  # [c]: never match this line's own shell
grep -q "^ALL PASS" "$CHECK_LOG" || { grep -E "FAIL" "$CHECK_LOG" | head -5 | tee -a "$OUT/progress.log"; fail "record checks did not all pass"; }
cp "$CHECK_LOG" "$OUT/check_task.txt"
say "record checks: ALL PASS ($(grep -c '^  PASS' "$CHECK_LOG") checks, staging and leak checks included)"

setsid nohup bash scripts/prune_record_copies.sh >> "$OUT/prune.log" 2>&1 < /dev/null &
say "pruner started (pid $!)"

for m in sonnet haiku; do
    TB_AGENT=boards-record TB_MODEL=$m bash scripts/cost_check.sh > "$OUT/costcheck-$m.log" 2>&1 \
        || { tail -15 "$OUT/costcheck-$m.log" >> "$OUT/progress.log"; fail "cost check $m failed (log $OUT/costcheck-$m.log)"; }
    say "cost check $m: $(grep -E '^extrapolated|^rounds that fit|^PASS' "$OUT/costcheck-$m.log" | tr '\n' ' ')"
done

fi

for m in sonnet haiku; do
    say "launch $m: setup, dry run, untuned VAL pass, then the climb (detached)"
    TB_AGENT=boards-record TB_MODEL=$m PROGRAM_MAX_CONCURRENT=${CONC:-6} bash scripts/run_on_vm.sh > "$OUT/launch-$m.log" 2>&1 \
        || { tail -15 "$OUT/launch-$m.log" >> "$OUT/progress.log"; fail "launch $m failed or the untuned stop rule fired (log $OUT/launch-$m.log)"; }
    say "launch $m: $(grep -E 'ZERO-SHOT|SETUP PASS|zero-shot already' "$OUT/launch-$m.log" | tr '\n' ' ')"
done
say "both climbs running; logs ~/.cache/alo-tasks/tb-questions-boards-record{,-haiku}/r1-run.log, results under tb-questions-v1_7-runs/$(hostname)/boards-record{,-haiku}/"
