#!/usr/bin/env bash
# MedGemma fair evaluation on weill-p01, one phase per call, each with its own go
# (spec planning/vista_eval_vlm/2026-10-30-medgemma-fair-eval.md, on the mount):
#
#   bash scripts/medgemma_fair_eval.sh control <gpu>          # step 5 control, plain + reason   -> runs/fair_control/
#   bash scripts/medgemma_fair_eval.sh dry <gpu>              # 50 check cases per step, reason + think -> runs/fair_dry/
#   bash scripts/medgemma_fair_eval.sh full <gpu> reason ...  # v1's case sets, the variants that passed -> runs/fair_full/
#   bash scripts/medgemma_fair_eval.sh format fair_dry        # format check only (no GPU)
#
# Plain is v1 itself (runs/v1/medgemma/, same commit and cases), so the full phase never reruns it. Steps 1 and 3 use
# the whole TEST split and steps 2 and 4 the sample, as v1 did. A dry or full step that fails evalset.check (an
# unparsed answer) does not stop the phase: the format check after it is the pass/fail (spec: <= 5% unparsed and
# <= 5% at the token limit, per variant). EVALSET_GPU_MEMORY_UTIL defaults to 0.55 here, to fit beside another job.
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:?phase: control | dry | full | format}"
MOUNT=/data/fries/users/padamson/mnt
ARMS_FILE="$MOUNT/su-vista-uscentral1/vistabench/vlm/evalset_v1/arms/medgemma_fair_eval.yaml"

format() {  # $1 = run name: unparsed and at-limit share per variant and step
  (cd src && EVALSET_MOUNT_ROOT="$MOUNT" ../.venv/bin/python - "$1" "$ARMS_FILE" <<'EOF'
import sys
import pandas as pd
from evalset import arms, paths, recipes
run = sys.argv[1]
for d in sorted((paths.root() / "runs" / run).glob("medgemma*")):
    v = d.name.split("__", 1)[1] if "__" in d.name else None
    r = arms.apply(recipes.get("medgemma"), v, arms.load(sys.argv[2], v)[0]) if v else recipes.get("medgemma")
    cap = int(r.setting("max_new_tokens"))
    for f in sorted(d.glob("step*.csv")):
        df = pd.read_csv(f)
        un, lim = df["answer"].isna().mean(), (df["usage_output_tokens"] >= cap).mean()
        flag = "FAIL" if un > 0.05 or lim > 0.05 else "ok"
        print(f"{d.name:20s} {f.stem:14s} n={len(df):5d} unparsed={un:6.1%} at_limit={lim:6.1%} "
              f"median_tokens={df['usage_output_tokens'].median():6.0f} {flag}")
EOF
  )
}
[ "$PHASE" = format ] && { format "${2:?run name}"; exit 0; }

GPU="${2:?gpu index}"; shift 2
export RECIPE=medgemma CUDA_VISIBLE_DEVICES="$GPU" EVALSET_MODEL_CACHE=/data/fries/users/padamson/model_cache
export EVALSET_GPU_MEMORY_UTIL="${EVALSET_GPU_MEMORY_UTIL:-0.55}"

one() {  # variant run [VAR=value ...]; plain = the recipe as is
  local variant=$1 run=$2; shift 2
  local v=(); [ "$variant" != plain ] && v=(ARMS="$ARMS_FILE" VARIANT="$variant")
  env ${v[@]+"${v[@]}"} RUN="$run" "$@" bash scripts/evalset_weill.sh
}

case "$PHASE" in
  control)
    for v in plain reason; do one "$v" fair_control STEPS=5; done ;;
  dry)
    for v in reason think; do
      one "$v" fair_dry LIMIT=50 STEPS="1 3" WHICH=test || echo "check failed: $v steps 1 3"
      one "$v" fair_dry LIMIT=50 STEPS="2 4" WHICH=sample || echo "check failed: $v steps 2 4"
    done
    format fair_dry ;;
  full)
    [ $# -gt 0 ] || { echo "ERROR: name the variants that passed the dry run, e.g. reason think"; exit 2; }
    for v in "$@"; do
      one "$v" fair_full STEPS="1 3" WHICH=test || echo "check failed: $v steps 1 3"
      one "$v" fair_full STEPS="2 4" WHICH=sample || echo "check failed: $v steps 2 4"
    done
    format fair_full ;;
  *) echo "ERROR: phase must be control, dry, full or format"; exit 2 ;;
esac
