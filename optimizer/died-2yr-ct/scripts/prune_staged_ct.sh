#!/usr/bin/env bash
# Free disk during a climb: delete the staged CT copies (calls/<id>/ct/ct.npz) inside FINISHED passes of a work
# folder, keeping everything else (answers, findings notes, transcripts, receipts, network logs). Each copy is a
# duplicate of the mount's patients/<id>/ct.npz; the optimizer's mistake files are copied from the mount, not from
# here (adapters.mistake_corpus), so nothing a later round reads is lost. A pass counts as finished only when it is
# under program-outputs/archive/ and its pass.json says "complete"; a copy is deleted only if its size equals the
# mount's. A 1,000-scan pass stages about 40 GB, so without this a climb fills the disk in a few rounds (2026-11-17).
#
#   bash optimizer/died-2yr-ct/scripts/prune_staged_ct.sh ~/.cache/alo-tasks/died-2yr-ct-prob-r2
#   bash optimizer/died-2yr-ct/scripts/prune_staged_ct.sh <work folder> --watch 600 <run log>   repeat every 600 s
#       until the run log ends in FINISHED (or a STOPPED line), then prune once more and exit
#
# Prints counts only. Exit 1 if a copy's size differs from the mount's (that pass is then left alone).
set -uo pipefail
WD=$1
DATA=${TASK_DATA_DIR:-/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7}/patients

prune_once() {
    local rc=0
    for pass in "$WD"/program-outputs/archive/*/; do
        [ -f "$pass/pass.json" ] && grep -q '"state": *"complete"' "$pass/pass.json" || continue
        local n=0 bad=0
        for f in "$pass"calls/*/ct/ct.npz; do
            [ -f "$f" ] || continue
            n=$((n + 1))
            id=$(basename "$(dirname "$(dirname "$f")")")
            [ "$(stat -c %s "$f")" = "$(stat -c %s "$DATA/$id/ct.npz" 2>/dev/null)" ] || bad=$((bad + 1))
        done
        [ "$n" = 0 ] && continue
        if [ "$bad" != 0 ]; then echo "$(date -u +%FT%TZ) $(basename "$pass"): $bad of $n copies differ from the mount; left alone"; rc=1; continue; fi
        find "$pass"calls -mindepth 2 -maxdepth 2 -type d -name ct -exec chmod u+w {} +
        find "$pass"calls -mindepth 3 -maxdepth 3 -path '*/ct/ct.npz' -type f -delete
        echo "$(date -u +%FT%TZ) $(basename "$pass"): deleted $n staged CT copies; disk free $(df -h --output=avail "$WD" | tail -1 | tr -d ' ')"
    done
    return $rc
}

if [ "${2:-}" = "--watch" ]; then
    every=${3:-600}; log=${4:-}
    while :; do
        prune_once
        if [ -n "$log" ] && grep -qE '^FINISHED|^STOPPED' "$log" 2>/dev/null && ! pgrep -f "run.py run --run-id" >/dev/null; then
            prune_once; echo "run finished; pruner exits"; exit 0
        fi
        sleep "$every"
    done
fi
prune_once
