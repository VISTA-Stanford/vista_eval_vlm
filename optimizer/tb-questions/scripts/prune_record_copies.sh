#!/usr/bin/env bash
# Keep the boards-record climbs from filling a shared disk (sllm-02 filled twice in November from staged copies).
# Every session gets its own copy of its record (agentic.stage -> calls/<id>/record/, ~2 MB a board, ~1,240 a round).
# Once the session has written its answer (output.json beside it) and the copy is over AGE_MIN minutes old (sessions
# time out at 15), nothing reads the copy again: the optimizer's feedback copies from the master export
# (adapters.patient_files) and a resume restages only unfinished calls. This deletes those copies and logs free disk.
#
#   nohup bash optimizer/tb-questions/scripts/prune_record_copies.sh > ~/.cache/alo-tasks/tb-boards-prune.log 2>&1 &
#
# Settings on the same line: AGE_MIN (30), EVERY_S (300), ROOTS (the boards work folders).
set -uo pipefail
AGE_MIN=${AGE_MIN:-30}
EVERY_S=${EVERY_S:-300}
ROOTS=${ROOTS:-$HOME/.cache/alo-tasks/tb-questions-boards-record*}
while :; do
    n=0
    for rec in $(ls -d $ROOTS/program-outputs/*/*/calls/*/record 2>/dev/null); do
        [ -f "$(dirname "$rec")/output.json" ] || continue
        [ -n "$(find "$rec" -maxdepth 0 -mmin +"$AGE_MIN" 2>/dev/null)" ] || continue
        chmod -R u+w "$rec" 2>/dev/null
        rm -rf "$rec" && n=$((n + 1))
    done
    echo "$(date -u +%FT%TZ) pruned $n record copies; free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
    sleep "$EVERY_S"
done
