Reference: docs/claude_ops.md

# VM smoke — rung-0 0b decoding fix (force the Yes/No constraint, drop insufficient rows, fail-closed guard)

**Status: Handoff to VM** (2026-07-13)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **uncommitted on the Mac; commit + push first, SHA set at commit time.** Everything below has **never executed.**
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **unmerged; `git fetch` first** · this doc `docs/vm-status/2026-07-13-rung0-0b-decoding-fix.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull` (shared / dirty checkout → `git worktree add ../vista_eval_vlm-rung0fix worktree-vlm-modular-preprocessing-roadmap`).
**Machine posture:** authored on the planner Mac (`DNa825021.SUNet`, no runtime/data/creds). Split run:
- **Step 1** (cheap, weight-free decode-guard proof) → **Claude-Code CPU** box `phil-sllm-01` (shares the `/mnt/su-vista-*` mounts; runs Claude Code → readback here directly).
- **Steps 2–4** (destructive move-aside, weighted 0b, 0c report) → **GPU** box `phil-a100x1-80gb-01` (A100-80GB; no Claude Code → a person runs the script; read results back on the Mac or `phil-sllm-01`).

**Target machine:** Step 1 = Claude-Code CPU (`phil-sllm-01`); Steps 2–4 = GPU (`phil-a100x1-80gb-01`), readback on Mac / `phil-sllm-01`.
**Plans:** [`vlm-rung0-0b-decoding-fix.md`](../plans/vlm-rung0-0b-decoding-fix.md#verification--vm-handoff) ← criteria source of truth (Reviewed: Yes; Codex `/review-implementation` clean 2026-07-13).
**Prior handoffs:** [`2026-07-08-rung0-reproduce-ryan-feb26.md`](./2026-07-08-rung0-reproduce-ryan-feb26.md) — its **0b RAN but INVALID** finding (constrained decoding never engaged) is exactly what this fix corrects.

## Why this doc

The first 0b weighted run (`phil-a100x1-80gb-01`, 2026-07-09) ran **unconstrained, silently**: the Yes/No decode
constraint was AND-gated behind the registry `is_binary` bool at `run_bq.py:910`, and the staged `valid_tasks.json`
marks PFS `is_binary: False` (genuine 3-class), so `constrained_choices` resolved to `None`. medgemma free-generated
~2800-char essays → **0/1238** exactly Yes/No → `predicted_label == -1` ≈ 54%. This branch implements the fix (4 seams,
Codex-clean):

1. **Seam 1** — the decode gate now keys off the task's Yes/No **mapping** (`is_binary_yes_no_task`), not `is_binary`
   (`run_bq.py`). PFS's 3-class mapping passes; medgemma is now constrained to `["Yes","No"]`.
2. **Seam 2** — per-task `[DECODE]` log line + a `[DECODE] summary` count (constraint engagement is now visible in the log).
3. **Seam 3** — a **fail-closed preflight** in `eval/run_rung0_gpu.sh` that goes RED *before weights load* if a selected
   task's mapping isn't Yes/No while constrained decoding is on (closes the silent-no-op gap).
4. **Seam 4** — 0c uses `constrained_all_model_response.py` (drops `ground_truth_label == -1`), registry resolved from
   `paths.valid_tasks` (same key `run_bq` gates on), `--config/--output` CLI, dtype-robust drop mask.

A clean re-run gives a genuinely constrained 0b + a Ryan-adjacent 0c, unblocking rungs 1–2.

**Banked from the prior handoff (not re-run):** the 0a feb26 CT-coverage result (86.9%, 309/2,362 missing — accepted
declared delta; 0c stratifies to `used_image==1`) and the force-GCS proof. This doc re-runs only 0b/0c under the fix.

## Step 0 — get the artifacts onto the VM (both boxes)

```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull
# Confirm the fix seams are present:
grep -n "is_binary_yes_no_task" src/vista_run/run_bq.py src/results/task_mapping.py
grep -n "\[DECODE\]" src/vista_run/run_bq.py
grep -n "decode guard\|paths.valid_tasks\|is_binary_yes_no_task" eval/run_rung0_gpu.sh
grep -n "isin(\[-1\|valid_tasks_rel\|add_argument" src/results/constrained_all_model_response.py
```
**Expected:** clean checkout at the pushed fix SHA; all four greps hit (new `task_mapping.py`, the mapping-based gate,
the `[DECODE]` lines, the `run_rung0_gpu.sh` decode guard, the reducer's `.isin`/registry/CLI).
**STOP:** any grep empty → the Mac hasn't pushed the fix (or the wrong branch is checked out); do not proceed.

## Step 1 — decode-guard proof (Claude-Code CPU `phil-sllm-01`; weight-free, cheap) — the RED-before-spend gate

Prove the guard's decision is correct **off the GPU**, before any weighted run. Runs on `phil-sllm-01` (base_dir mount
present here). This is the plan's Phase-2 "small unit check may stand in" — the predicate + the real registry resolution.

```bash
cd src && python - <<'PY'
import json, os, yaml
from results.task_mapping import is_binary_yes_no_task

# (a) predicate on fixtures: PFS 3-class Yes/No -> True; genuine non-Yes/No -> False
assert is_binary_yes_no_task({"1":"Yes","0":"No","-1":"Insufficient follow-up or missing data"}) is True
assert is_binary_yes_no_task({"1":"Stage I","0":"Stage II","2":"Stage III"}) is False
assert is_binary_yes_no_task({}) is False

# (b) the guard's real decision on the rung-0 config's registry (resolved via paths.valid_tasks)
cfg = yaml.safe_load(open("../configs/all_tasks.rung0.yaml"))
p = cfg["paths"]
reg = os.path.join(p["base_dir"], p.get("valid_tasks", "tasks/valid_tasks.json"))
registry = {t["task_name"]: t for t in json.load(open(reg))}
for t in cfg["tasks"]:
    m = registry.get(t, {}).get("mapping", {})
    ok = is_binary_yes_no_task(m)
    print(f"  {t}: mapping={m} -> is_yes_no={ok}")
    assert ok, f"GUARD would (correctly) go RED for {t} — mapping is not Yes/No"
print("[ok] guard: PFS resolves Yes/No-mappable (GREEN); non-Yes/No fixtures rejected (RED)")
PY
```
**Expected:** all asserts pass; PFS prints `is_yes_no=True`; the `[ok]` line prints. Proves the mapping-based gate
constrains PFS and the fail-closed guard would reject a non-Yes/No task.
**STOP:** PFS prints `is_yes_no=False` → the staged registry's PFS mapping isn't `{"1":"Yes","0":"No",...}` (a data
fact the whole fix assumes) → **class-3 deviation, hand back** (do not run 0b — it would be free-generation again).
Registry not found at the resolved path → fix the mount / `paths.valid_tasks` before proceeding.

*(Optional, on the GPU box in Step 2: a doctored-config dry-run of the full wrapper — `tasks: [__not_a_real_task__]`,
`CONFIG=/tmp/bad.yaml bash eval/run_rung0_gpu.sh` — must exit nonzero at the `decode guard` line **before** the weight
prefetch, proving the in-script fail-close fires end-to-end. Cheap; recommended once.)*

## Full run — operator run-block (GPU `phil-a100x1-80gb-01`, no Claude Code)

On `phil-a100x1-80gb-01` a **person** runs the steps below (no agent). Precondition: Step 1 GREEN on `phil-sllm-01`.

### Step 2 — move the stale free-text CSVs aside (DESTRUCTIVE; not resumable)

`_setup_output_and_resume` resumes by `index`, so a re-run would **append constrained rows to the stale free-text CSVs**.
Archive them first (record the archive path in the readback):

```bash
cd <vista_eval_vlm repo>
RES=/mnt/su-vista-uscentral1/vistabench/vlm/results/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1/progression_recurrence_free_survival_1_yr/medgemma-1.5-4b-it
ARCHIVE="$RES/_stale_freegen_20260709"
mkdir -p "$ARCHIVE"
mv "$RES"/*_results_{no_image,axial_all_image}.csv "$ARCHIVE"/ 2>/dev/null || true
# also move aside any stale reducer output so 0c can't read a leftover:
mv figures/results_stats/*all_model_response.csv figures/results_stats/_stale_20260709/ 2>/dev/null || true
ls "$ARCHIVE"
```
**Expected:** the two `*_results_{no_image,axial_all_image}.csv` from the 2026-07-09 free-gen run are now under
`_stale_freegen_20260709/`; the live results dir has no PFS medgemma result CSVs left.
**STOP:** none — but **record the archive path** so the free-gen run stays recoverable.

### Step 3 — fresh 0b weighted re-run (GPU, expensive)

```bash
# HF_TOKEN if not already `hf auth login`'d; SYNC=1 only if the repo .venv isn't provisioned on this box.
HF_TOKEN=hf_xxx bash eval/run_rung0_gpu.sh
```
The wrapper preflights (GPU + torch.cuda; config force-GCS/feb26/subsample=false/constrained/timeline_truncation
invariants; base_dir mount; **the new fail-closed decode guard**; HF auth; disk; weight prefetch) and halts before GPU
spend if the box isn't ready, then drives `run_bq` for PFS × medgemma × {no_image, axial_all_image}.
**Expected:**
- Preflight prints `[ok] decode guard: … 1/1 selected tasks Yes/No-mappable (use_constrained=True)`.
- Log shows `[DECODE] task=progression_recurrence_free_survival_1_yr  mode=constrained  choices=['Yes','No']` for
  **both** arms, and `[DECODE] summary: 1/1 selected tasks constrained`.
- `[CT] source=gcs` throughout; `source=local` **== 0** (force-GCS).
- `Error in batch` **== 0** AND `Producer error` **== 0** (the two silent-drop paths — re-run before 0c if either > 0).
- Result CSVs non-empty at `$RES/*_results_{no_image,axial_all_image}.csv`; the two arms' row counts ~equal; a spot-check
  shows responses are exact `Yes`/`No` (no essays); `used_image` ≈ 87% for `axial_all_image`.
**STOP:** `mode=free` for PFS (mapping regressed — Step 1 should have caught it); the decode guard exits RED unexpectedly;
weights/HF-auth fail; any `[CT] source=local`; `row_count==0`; batch/producer errors > 0 (re-run after re-archiving).

### Step 4 — 0c report (CPU-fine; constrained reducer, rung-0 `--output`)

```bash
cd src && python -m results.constrained_all_model_response \
    --config ../configs/all_tasks.rung0.yaml \
    --output ../figures/results_stats/rung0_constrained_all_model_response.csv
```
**Expected / QC gates:**
- `[QC] predicted_label == -1 total: 0` — **exactly 0** (proof the constraint engaged). **Any nonzero → 0b still
  invalid, STOP + hand back** (unless a named vLLM structured-output edge is documented).
- `Dropped N rows where ground_truth_label is -1` with **N ≈ 27%** of rows (the insufficient-follow-up class actually
  dropped — verify N is nonzero and ballpark-27%, proving the dtype-robust `.isin([-1,"-1"])` drop works; a printed
  `Dropped 0` means the drop silently no-op'd → STOP).
- The reducer resolved the registry from `paths.valid_tasks` (same file `run_bq` gated on); PFS `mapping` loaded non-empty.
- **Do NOT** write to the default `figures/results_stats/constrained_all_model_response.csv` — that's Ryan's committed
  baseline comparator; the explicit `--output` above keeps it intact.
**Report (no hard gate):** accuracy vs Ryan's committed `figures/results_stats/all_model_response.csv` (constrained
baseline), mapped labels `axial_all_image → image_and_timeline`, `no_image → timeline_only`, image arm stratified to
`used_image==1`. **One weak, confounded signal** within the ~10% informal band (report-presence + 30-vs-10 slices +
un-subsampled n; OQ-R2/R3) — **not** a match claim.
**STOP:** `predicted_label == -1 > 0` → hand back; registry-path mismatch or empty PFS mapping → fix before reporting.

## Report back

On the Claude-Code CPU box (`phil-sllm-01`) or the Mac, append `## VM run results`: Step 1 pass/fail; the archive path
from Step 2; the `[DECODE]` lines + engagement summary; the four Step-3 gate outcomes (with real pandas row counts, not
`wc -l`); the Step-4 QC (`predicted_label == -1 total` = ?, `Dropped N` = ?); and the 0c report read (accuracies +
delta). Read large outputs from the CSVs — **no PHI** (counts / metrics / pass-fail only; never paste rows, timelines,
UIDs, or dates).

## VM run results
_(left empty by the planner; the executor fills this in readback mode)_
