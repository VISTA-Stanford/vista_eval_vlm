#!/usr/bin/env bash
# Keep the knowledge-base pilot from filling the shared disk on phil-sllm-02 (it filled 11-17, 11-18 and 11-23 from
# staged CT copies; copied from tb-questions' prune_record_copies.sh). Every tester session gets its own copy of its
# scan (agentic.stage -> calls/<id>/ct/, ~37 MB; 880 sessions in the pilot, ~33 GB unpruned). Once the session has
# written its answer (output.json beside it) and the copy is over AGE_MIN minutes old (sessions time out at 15), nothing
# reads it again: the builder's feedback uses only the findings note and the transcript, and a resume restages only
# unfinished calls. This deletes those copies and logs free disk. scripts/run_kb_pilot.sh starts it and stops it.
#
#   nohup bash optimizer/ct-findings-5/scripts/prune_kb_copies.sh > ~/.cache/alo-tasks/kb-prune.log 2>&1 &
#
# Settings on the same line: AGE_MIN (30), EVERY_S (300), ROOT (the KB work folder).
set -uo pipefail
AGE_MIN=${AGE_MIN:-30}
EVERY_S=${EVERY_S:-300}
ROOT=${ROOT:-$HOME/.cache/alo-tasks/ct-findings-5-kb}
while :; do
    n=0
    for ct in $(ls -d "$ROOT"/program-outputs/*/*/calls/*/ct 2>/dev/null); do
        [ -f "$(dirname "$ct")/output.json" ] || continue
        [ -n "$(find "$ct" -maxdepth 0 -mmin +"$AGE_MIN" 2>/dev/null)" ] || continue
        chmod -R u+w "$ct" 2>/dev/null
        rm -rf "$ct" && n=$((n + 1))
    done
    echo "$(date -u +%FT%TZ) pruned $n CT copies; free $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
    sleep "$EVERY_S"
done
