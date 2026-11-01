#!/usr/bin/env bash
# Fair evaluation of an open model (vLLM) on weill-p01, one phase per call, each with its own go
# (spec planning/vista_eval_vlm/2026-11-01-open-vlm-fair-eval.md, on the mount; MedGemma's phases are
# scripts/medgemma_fair_eval.sh, kept as the record of its runs):
#
#   bash scripts/fair_eval.sh <recipe> control <gpu>   # step 5 control (IV contrast, female sex)  -> runs/fair_control/
#   bash scripts/fair_eval.sh <recipe> dry <gpu>       # first 50 cases per step, steps 1-4        -> runs/fair_dry/
#   bash scripts/fair_eval.sh <recipe> ablation <gpu>  # step 2, 50 check cases, slice variants     -> runs/fair_ablation/
#   bash scripts/fair_eval.sh <recipe> full <gpu>      # steps 1 + 3 on the TEST split, 2 + 4 on the sample -> runs/fair_full/
#   bash scripts/fair_eval.sh <recipe> format <run>    # format check only (no GPU)
#
# <recipe>: qwen3vl | octomed. One arm per model (the recipe as is); ablation variants are in the arms file below.
# A step that fails evalset.check (an unparsed answer) does not stop the phase: the format check after it is the
# pass/fail (<= 5% unparsed and <= 5% at the token limit). Each phase first copies its image packs to a local root
# (src/evalset/stage.py; EVALSET_LOCAL_ROOT), as the mount reads slowly one file at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE="${1:?recipe: qwen3vl | octomed}"
PHASE="${2:?phase: control | dry | ablation | full | format}"
MOUNT=/data/fries/users/padamson/mnt
ARMS_FILE="$MOUNT/su-vista-uscentral1/vistabench/vlm/evalset_v1/arms/open_vlm_fair_eval.yaml"
case "$RECIPE" in
  qwen3vl) VARIANTS="qwen3vl_85" ;;
  octomed) VARIANTS="octomed_10 octomed_85" ;;
  *) echo "ERROR: recipe must be qwen3vl or octomed"; exit 2 ;;
esac

format() {  # $1 = run name: unparsed and at-limit share per folder and step
  (cd src && EVALSET_MOUNT_ROOT="$MOUNT" ../.venv/bin/python - "$1" "$RECIPE" "$ARMS_FILE" <<'EOF'
import sys
import pandas as pd
from evalset import arms, paths, recipes
from evalset.score import reparse
run, name, arms_file = sys.argv[1:]
base = recipes.get(name)
for d in sorted((paths.root() / "runs" / run).glob(name + "*")):
    v = d.name.split("__", 1)[1] if "__" in d.name else None
    r = arms.apply(base, v, arms.load(arms_file, v)[0]) if v else base
    cap = int(r.setting("max_new_tokens"))
    for f in sorted(d.glob("step*.csv")):
        df = reparse(pd.read_csv(f), r.answer_form)  # today's parser, not the one stored at run time
        un, lim = df["answer"].isna().mean(), (df["usage_output_tokens"] >= cap).mean()
        flag = "FAIL" if un > 0.05 or lim > 0.05 else "ok"
        print(f"{d.name:22s} {f.stem:14s} n={len(df):5d} unparsed={un:6.1%} at_limit={lim:6.1%} "
              f"median_tokens={df['usage_output_tokens'].median():6.0f} {flag}")
EOF
  )
}
[ "$PHASE" = format ] && { format "${3:?run name}"; exit 0; }

GPU="${3:?gpu index}"
export CUDA_VISIBLE_DEVICES="$GPU" EVALSET_MODEL_CACHE=/data/fries/users/padamson/model_cache
export EVALSET_GPU_MEMORY_UTIL="${EVALSET_GPU_MEMORY_UTIL:-0.85}"
export EVALSET_MOUNT_ROOT="$MOUNT" EVALSET_ROOT="${EVALSET_LOCAL_ROOT:-/data/fries/users/padamson/evalset_local}"

stage() {  # steps which [limit] [variant]
  local v=(); [ -n "${4:-}" ] && v=(--arms "$ARMS_FILE" --variant "$4")
  (cd src && ../.venv/bin/python -m evalset.stage --recipe "$RECIPE" --steps $1 --which "$2" ${3:+--limit "$3"} \
     ${v[@]+"${v[@]}"} --to "$EVALSET_ROOT")
}

one() {  # variant run [VAR=value ...]; plain = the recipe as is
  local variant=$1 run=$2; shift 2
  local v=(); [ "$variant" != plain ] && v=(ARMS="$ARMS_FILE" VARIANT="$variant")
  env ${v[@]+"${v[@]}"} RECIPE="$RECIPE" RUN="$run" "$@" bash scripts/evalset_weill.sh
}

case "$PHASE" in
  control)
    stage 5 sample
    one plain fair_control STEPS=5 || echo "check failed: step 5"
    format fair_control ;;
  dry)
    stage "1 3" test 50; stage "2 4" sample 50
    one plain fair_dry LIMIT=50 STEPS="1 3" WHICH=test || echo "check failed: steps 1 3"
    one plain fair_dry LIMIT=50 STEPS="2 4" WHICH=sample || echo "check failed: steps 2 4"
    format fair_dry ;;
  ablation)
    for v in $VARIANTS; do
      stage 2 sample 50 "$v"
      one "$v" fair_ablation LIMIT=50 STEPS=2 WHICH=sample || echo "check failed: $v step 2"
    done
    format fair_ablation ;;
  full)
    stage "1 3" test; stage "2 4" sample
    one plain fair_full STEPS="1 3" WHICH=test || echo "check failed: steps 1 3"
    one plain fair_full STEPS="2 4" WHICH=sample || echo "check failed: steps 2 4"
    format fair_full ;;
  *) echo "ERROR: phase must be control, dry, ablation, full or format"; exit 2 ;;
esac
