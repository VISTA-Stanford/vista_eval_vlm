#!/usr/bin/env bash
# Copy the hill-climbed agent programs and the CT agents' raw test answers off this box to the shared mount
# (paper-code stage 2, Expensive step: Phil runs it once on phil-sllm-02 and once on phil-sllm-03).
#
# What it copies, from ~/.cache/alo-tasks/<task> work folders of the paper's agent tasks
# (died-2yr-ehr*, died-2yr-ct*, ct-findings-5*, died-ehr-over-time*), whose program history the engine keeps in <task>/repo:
#   - each work folder's whole program history as one git bundle (every version, every tag), named
#     <task>-<head sha 12>.bundle, so any version the paper used can be checked out later by its tag;
#   - for the CT tasks, every TEST pass's answers.json under program-outputs/archive/test-program-*/.
# Into /mnt/su-vista-uscentral1/chaudhari_lab/phil/paper-agent-programs/<host>/ (inside the secure environment).
# The programs carry notes learned from TRAIN patients: they stay on the mount and are NOT committed (PHI look first).
#
# Pass: prints "PASS: <n> bundles, <m> answer files" and exits 0; exits 1 if it found no work folder or a copy is empty.
set -euo pipefail
SRC="${ALO_TASKS:-$HOME/.cache/alo-tasks}"
OUT="${OUT:-/mnt/su-vista-uscentral1/chaudhari_lab/phil/paper-agent-programs}/$(hostname)"
mkdir -p "$OUT/programs" "$OUT/raw-test-answers"
nb=0; na=0
for d in "$SRC"/died-2yr-ehr* "$SRC"/died-2yr-ct* "$SRC"/ct-findings-5* "$SRC"/died-ehr-over-time*; do
  # the engine keeps each task's program history in <work folder>/repo
  repo="$d/repo"
  [ -e "$repo/.git" ] || continue
  name="$(basename "$d")"
  sha="$(git -C "$repo" rev-parse --short=12 HEAD)"
  f="$OUT/programs/${name}-${sha}.bundle"
  git -C "$repo" bundle create "$f.tmp" --all >/dev/null 2>&1
  [ -s "$f.tmp" ] || { echo "ERROR: empty bundle for $name"; exit 1; }
  mv "$f.tmp" "$f"
  git -C "$repo" tag --list 'program-v*' | sort -V > "$OUT/programs/${name}-${sha}.tags.txt"
  nb=$((nb + 1))
  echo "bundle $name ($sha): $(wc -l < "$OUT/programs/${name}-${sha}.tags.txt") program tags"
  case "$name" in
    died-2yr-ct*|ct-findings-5*)
      for t in "$d"/program-outputs/archive/test-program-*/answers.json; do
        [ -f "$t" ] || continue
        dest="$OUT/raw-test-answers/$name/$(basename "$(dirname "$t")")"
        mkdir -p "$dest"; cp "$t" "$dest/answers.json"
        [ -s "$dest/answers.json" ] || { echo "ERROR: empty copy $dest"; exit 1; }
        sha256sum "$dest/answers.json" | cut -c1-64 > "$dest/answers.json.sha256"
        na=$((na + 1))
      done ;;
  esac
done
[ "$nb" -gt 0 ] || { echo "ERROR: no agent work folders under $SRC on $(hostname)"; exit 1; }
echo "PASS: $nb bundles, $na answer files -> $OUT"
