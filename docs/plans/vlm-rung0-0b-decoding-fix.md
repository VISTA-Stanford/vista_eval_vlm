Reference: research-skills/claude_ops.md

# Rung-0 0b decoding fix — force the binary constraint for PFS, drop insufficient-follow-up rows, make the decode mode fail-closed and observable

**Status: Completed — landed on `main` 2026-07-15** (VM-green 2026-07-13; 0c `predicted_label==-1 total: 0`). (2026-07-09; revised after Codex `/review-plan` — `reviews/vlm-rung0-0b-decoding-fix-feedback.md`).
Amends **OQ-R6** of [`vlm-rung0-reproduce-ryan-feb26.md`](vlm-rung0-reproduce-ryan-feb26.md) after a class-3 deviation on the
first 0b run.
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` (continues; uncommitted).
**Machine posture:** authored on the planner Mac (**no code execution here** — no Python/tests). The fresh 0b re-run and
all runtime checks execute on the GPU box `phil-a100x1-80gb-01`; 0c report is CPU-fine (VM-side).

## Context — the mistake this fixes

The first 0b weighted run (GPU box `phil-a100x1-80gb-01`, 2026-07-09, log `logs/rung0_20260709_185445/run_bq.log`) ran
**unconstrained**, silently. OQ-R6 concluded "set `use_constrained_decoding_for_binary: true` to match Ryan" — necessary
but **not sufficient**. The flag is AND-gated behind the registry `is_binary` bool at `run_bq.py:910`:

```python
# run_bq.py:909-910  (_process_single_task_with_data)
use_constrained = self.cfg['runtime'].get('use_constrained_decoding_for_binary', True)
constrained_choices = ["Yes", "No"] if (task_info.get("is_binary", False) and use_constrained) else None
```

The staged `valid_tasks.json` marks PFS **`is_binary: False`** — its label space is genuinely 3-class
(`mapping = {"1":"Yes","0":"No","-1":"Insufficient follow-up or missing data"}`). So `constrained_choices` resolved to
`None` regardless of the flag, medgemma free-generated (~2800-char essays), and the reducer could not reduce them.
Executor's measured evidence (both arms, n=1238, dedup'd by `index`):

| signal | no_image | axial_all_image |
| --- | --- | --- |
| `ground_truth_label == -1` (insufficient follow-up) | 339 = 27.4% | 339 = 27.4% |
| responses exactly `Yes`/`No` | **0 / 1238** | **0 / 1238** |
| `model_response` char length (mean / max) | 2839 / 6393 | 2741 / 6393 |

The "~46% `-1`" first reported was the **clean-extraction rate** (571/1238); the flip side —
`predicted_label == -1` ≈ **54%** (667/1238 no_image) — is *extraction failures*, not the 27% ground-truth insufficient
class. Root cause confirmed end-to-end at `run_bq.py:910`. Two lessons drive this plan:

1. The constraint must key off **what the task actually is** (a Yes/No-scored task = its mapping), not the `is_binary`
   flag, which a 3-class-with-Yes/No registry entry sets to `False`.
2. The no-op was **silent, and only caught after the GPU spend**. The fix must be **fail-closed** — a cheap guard that
   goes RED *before* weights load — not merely logged and QC'd afterward. ("You don't trust; you instrument.")

## Goal

Make 0b genuinely reproduce Ryan, make the reproduction self-evident from the artifacts, and make a silent no-op
**impossible to run to completion**:

1. **Constrain** PFS to `["Yes","No"]` despite `is_binary: False`.
2. **Drop** the `-1` (insufficient-follow-up) rows from the 0c eval entirely — matching Ryan, whose committed baseline is
   all-0/1 precisely because his `constrained_all_model_response.py` drops `ground_truth_label == -1`.
3. **Fail-closed + instrument**: a pre-weights preflight aborts if a selected task's mapping isn't Yes/No while the
   constraint flag is on; the run log records the decode mode per task; 0c hard-STOPs on any residual
   `predicted_label == -1`.

Ryan-parity, not a distortion: forcing Yes/No + dropping `gt == -1` measures exactly the 2-class cohort Ryan scored (the
insufficient rows have no legitimate Yes/No ground truth, so they're excluded either way).

## Approach

Four code seams + a fail-closed preflight + a fresh re-run. Each is small; the value is in getting them jointly correct,
**contract-bound**, and observable.

### 1. Decoding fix — constrain off the mapping, not the `is_binary` flag (`run_bq.py:910`)

Gate the constraint on the task's **mapping being binary Yes/No**, via the canonical predicate
`is_binary_yes_no_task(mapping)` (`final_metrics.py:70`) — True iff `mapping["1"]=="Yes"` and `mapping["0"]=="No"`. PFS's
3-class mapping **passes** (the predicate reads only the `"1"`/`"0"` keys; the `-1` class doesn't disqualify it) — Codex
confirmed this. Faithful to Ryan's baseline commit `5e97abd`, which forced `["Yes","No"]` for every Yes/No-mappable task
*before* the `is_binary` knob existed.

```python
# run_bq.py:909-910  (proposed)
use_constrained = self.cfg['runtime'].get('use_constrained_decoding_for_binary', True)
mapping = task_info.get("mapping", {})
is_yes_no = is_binary_yes_no_task(mapping)          # canonical predicate (see seam 5 for its home)
constrained_choices = ["Yes", "No"] if (is_yes_no and use_constrained) else None
```

`task_info` is a raw entry from the registry `run_bq.py` loads at `:67-71` from `cfg['paths']['valid_tasks']` and iterates
at `:181`, so `task_info.get("mapping")` is available at the gate — **provided the registry entry carries `mapping`**.
That is a VM-side data fact, made fail-closed by seam 3.

### 2. Instrumentation — per-task decode-mode log line (`run_bq.py`, `_process_single_task_with_data`)

Immediately after the gate, emit one line + a per-run summary count of constrained tasks (so constraint *breadth* is
observable, per OQ-F2):

```
[DECODE] task=progression_recurrence_free_survival_1_yr  mode=constrained  choices=['Yes','No']  (mapping is Yes/No; use_constrained=True)
[DECODE] summary: 1/1 selected tasks constrained
```

`mode=free` prints the reason (`mapping not Yes/No` OR `use_constrained=False`). Inference-neutral; costs nothing.

### 3. Fail-closed preflight — assert the mapping BEFORE weights load (`eval/run_rung0_gpu.sh`)

The real fix for the silent no-op: extend the existing preflight (which already asserts
`use_constrained_decoding_for_binary: true` and that `base_dir/…/valid_tasks.json` exists) to **resolve the registry via
the SAME `cfg['paths']['valid_tasks']` `run_bq` uses**, and for each task in the config's `tasks:` list, assert
`is_binary_yes_no_task(mapping)` is True. Any selected task whose mapping isn't Yes/No while the flag is on → **FAIL before
GPU spend** (the "guard that must go RED"). This runs on the GPU box (base_dir is a VM mount), costs no weights, and closes
the exact gap that let the first run complete silently.

### 4. Eval fix — 0c uses the constrained reducer, contract-bound to the same registry, writing a rung-0 output

Use **`constrained_all_model_response.py`**, not `all_model_response.py`. It maps `model_response` directly to Yes/No with
no free-text extraction (`:188`) — correct now that responses *are* Yes/No — and drops `ground_truth_label == -1` rows
(`:259-261`), i.e. the ~27% insufficient class, out of the eval. Three fixes make it safe:

- **Registry contract (Critical):** the reducer hard-codes `base_path/"tasks"/"valid_tasks.json"` (`:127`) while `run_bq`
  reads `cfg['paths']['valid_tasks']` (`:67`). "Same file" holds only for this overlay. **Make the reducer resolve
  `cfg['paths'].get('valid_tasks', 'tasks/valid_tasks.json')`** so 0b gates and 0c scores on the **same** registry by
  contract, not coincidence. (Mirror on `all_model_response.py` for consistency.)
- **`--config`/`--output` CLI:** the reducer's `__main__` calls `main()` with the default `configs/all_tasks.yaml`
  (`:270-271`), so `python -m results.constrained_all_model_response` silently reads the wrong `results_dir`. Add
  `argparse --config/--output` (backward-compatible: no args → current defaults). **0c MUST pass an explicit
  `--output` to a rung-0-specific path** — the default `figures/results_stats/constrained_all_model_response.csv` and its
  `all_model_response.csv` sibling **are Ryan's committed baseline comparators**; writing there overwrites the comparison
  and dirties git.
- **Drop-count print bug:** `:261` prints `len(minus_one_count_gt)` = number of model *groups*, not rows. The plan's 27%
  drop QC would read a garbage number. Fix it to print the actual dropped row count.

### 5. Home for `is_binary_yes_no_task` — extract to a shared dep-free module (OQ-F1 RESOLVED: extract, Phil 2026-07-09)

Move the predicate to `src/results/task_mapping.py` — **imports nothing** (no numpy/pandas/`results_analyzer`/
`context.normalize`), so the `run_bq` inference hot-path and the preflight import it cleanly. `final_metrics.py`
**re-exports** it (imports from `task_mapping`) so existing `from results.final_metrics import is_binary_yes_no_task`
callers keep working. Single source of truth for "is this a Yes/No task" — the semantics whose duplication caused this bug.
*Impl note:* the bash preflight (seam 3) runs from repo-root, so its Python heredoc must `sys.path.insert(0, <repo>/src)`
before importing — otherwise inline the same predicate there is a regression to duplication; keep the import.

### 6. Fresh 0b re-run (not resumable)

`_setup_output_and_resume` (`run_bq.py:702,:713`) resumes by `index`, so it would **append constrained rows to the stale
free-text CSVs**. Before re-running: move aside the existing result CSVs
`{results_dir}/…/medgemma-1.5-4b-it/*_results_{no_image,axial_all_image}.csv` **and** any stale
`figures/results_stats/*all_model_response.csv` (else 0c can read a leftover). Then re-run via `eval/run_rung0_gpu.sh`.

## Files to Modify

- `src/vista_run/run_bq.py` (`_process_single_task_with_data`, ~:905-910) — mapping-based Yes/No gate (seam 1) + `[DECODE]`
  log + summary (seam 2). Import per OQ-F1.
- `eval/run_rung0_gpu.sh` (preflight) — assert each selected task's mapping is Yes/No via `paths.valid_tasks`, fail before
  GPU spend (seam 3).
- `src/results/constrained_all_model_response.py` — resolve registry from `paths.valid_tasks` (:127); add
  `argparse --config/--output` (:270-271); fix the drop-count print (:261) (seam 4).
- `src/results/all_model_response.py` — mirror the `paths.valid_tasks` resolution + `--config/--output` CLI (consistency).
- `src/results/task_mapping.py` (**new**, dep-free) — holds `is_binary_yes_no_task` (OQ-F1: extract).
- `src/results/final_metrics.py` (:70) — import/re-export `is_binary_yes_no_task` from `task_mapping` instead of defining
  it (existing callers keep working).
- `configs/all_tasks.rung0.yaml` (:39) — keep `use_constrained_decoding_for_binary: true`; comment that the gate now keys
  off the task's Yes/No **mapping**, not the registry `is_binary` flag.
- `docs/plans/vlm-rung0-reproduce-ryan-feb26.md` (**OQ-R6**) — record the `is_binary` AND-gate gotcha + this fix; cross-link.
- *(handoff, via `/vm-handoff`)* the re-rendered `docs/vm-status/<date>-<sha>.md` — the complex-tier phasing below.

## Open Questions

- **OQ-F1 — RESOLVED (Phil 2026-07-09): extract.** `is_binary_yes_no_task` → dep-free `src/results/task_mapping.py`,
  imported by the inference gate, the preflight, and `final_metrics` (re-export). Single source of truth for the task
  semantics whose duplication caused this bug. (See seam 5.)
- **OQ-F2 — RESOLVED (Phil 2026-07-09): off-mapping — constrain ANY Yes/No-mappable task**, not a PFS-only allowlist. Seam
  1 as written; their forced guess on `gt==-1` rows is dropped so the metric is unaffected (matches Ryan), and breadth
  stays observable via the `[DECODE] summary` count (seam 2).
- **OQ-F3 — RESOLVED (Phil 2026-07-09): drop `ground_truth_label == -1` only.** Post-fix `predicted_label == -1` must be
  **exactly 0** (proof the constraint engaged) — a **STOP**, never silently dropped (that would hide model failures behind
  inflated accuracy).

## Verification & VM handoff

Complex-tier handoff (multiple phases, a class-2 decode gate, banked prior context, a destructive move-aside). Verification
centers on the guard going RED before spend and the decode mode being observable. Phases render into
`docs/vm-status/<date>-<sha>.md` via `/vm-handoff`.

**Phase 1 — Mac (author, NO code execution).**
- *Purpose:* structural review of the four edits — no Python, no AST-run (planner posture).
- *Expected:* `rg` shows the mapping-based gate + `[DECODE]` lines in `run_bq.py`; the mapping assertion in
  `run_rung0_gpu.sh`; `paths.valid_tasks` resolution + `--config/--output` + fixed drop-print in the reducer(s).
- *Stop:* any edit missing → fix before handoff. *Destructive:* no. *Next:* commit → `/vm-handoff` renders Phases 2–4.

**Phase 2 — GPU box: negative guard (the RED test), cheap.**
- *Purpose:* prove the fail-closed preflight actually aborts. Run the preflight against a deliberately non-Yes/No mapping
  (or flag off) and confirm it exits non-zero *before* weights load; then run it against the real config and confirm it
  passes. *(A small unit check asserting `run_bq`'s gate + the predicate on a `{1:Yes,0:No,-1:…}` fixture may stand in.)*
- *Expected:* guard exits RED on the bad mapping; GREEN on PFS. *Stop:* guard passes a non-Yes/No mapping → the fail-close
  is broken, halt. *Banked:* the 0a feb26 coverage (86.9%) and force-GCS proof from `2026-07-08-…md` — not re-run.

**Phase 3 — GPU box: fresh 0b re-run (expensive).** Move stale CSVs aside first (§6), then `bash eval/run_rung0_gpu.sh`.
- *Purpose:* the constrained weighted run. *Gates:*
  - `[DECODE] task=…progression_recurrence_free_survival_1_yr mode=constrained choices=['Yes','No']` for **both** arms.
  - `[CT] source=gcs` throughout; `source=local == 0` (force-GCS).
  - `Error in batch == 0` **and** `Producer error == 0` (the prior handoff's silent-drop traps); per-arm row counts ~equal;
    empty `model_response` ≈ 0; a spot-check shows exact `Yes`/`No` (no essays).
- *Stop:* `mode=free` for PFS (mapping regressed) — but Phase 2 should have caught this; weights/HF-auth fail; any
  `source=local`; `row_count==0`; batch/producer errors > 0 → re-run after moving CSVs aside. *Destructive:* yes — the
  move-aside; record the archived path.

**Phase 4 — GPU box: 0c report (CPU-fine).**
`cd src && python -m results.constrained_all_model_response --config ../configs/all_tasks.rung0.yaml --output <rung0-output>.csv`
- *Purpose:* the Ryan-adjacent comparison. *QC gates:*
  - printed `predicted_label == -1` count is **exactly 0** (proof the constraint engaged) — **any nonzero is a STOP**
    unless a named vLLM structured-output edge is documented.
  - the **actual** `ground_truth_label == -1` row count ≈ **27%** and those rows dropped (verify via the *fixed* drop
    print, not `len(groupby)`).
  - the resolved registry path 0c used **equals** `paths.valid_tasks`; the PFS `mapping` loaded non-empty.
- *Report (no hard gate):* accuracy vs Ryan's committed `figures/results_stats/all_model_response.csv` (constrained
  baseline), mapped labels `axial_all_image → image_and_timeline`, `no_image → timeline_only`, image arm stratified to
  `used_image==1` — one weak, confounded signal within the ~10% informal band (OQ-R2/R3), **not** a match claim.
- *Stop:* `predicted_label == -1 > 0` → 0b still invalid, hand back; registry-path mismatch or empty mapping → fix before
  reporting. *Next:* readback into the same vm-status doc → `/phi-vet` → `/commit-review`.
