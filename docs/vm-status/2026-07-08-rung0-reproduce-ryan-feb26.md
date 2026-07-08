Reference: docs/claude_ops.md

# VM smoke — rung 0 reproduce Ryan on feb26 (weighted operability), v1_1 substrate

**Status: Handoff to VM** (2026-07-08)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **uncommitted on the Mac; commit + push first, SHA set at commit time.** Everything below has **never executed.**
**Machine posture:** authored on the planner Mac (no runtime/data/creds). Run on the GCP executor VM (`som-nero-plevriti-deidbdf`, holds `gs://vista_bench/` + `su-vista-*` mounts + medgemma weights).
**Plans:** [`vlm-rung0-reproduce-ryan-feb26.md`](../plans/vlm-rung0-reproduce-ryan-feb26.md#verification--vm-handoff) ← criteria source of truth (all OQ-R1–R6 resolved; Reviewed: Yes)
**Prior handoffs:** [`2026-07-06-golden-harness.md`](./2026-07-06-golden-harness.md) — its class-3 DEVIATION (`axial_all_image` on the deleted `nov25` snapshot) is what spawned this feb26 workstream.

## Why this doc

Rung 0 proves **operability**: drive the VM and reproduce Ryan's *weighted* pipeline (loads model weights, emits a
results CSV — **not** the weights-free golden harness) on a **near-untouched v1_1 substrate**, changing only the CT
storage prefix `nov25 → feb26`. It gates the v1.5 cut + 3b refactor (rungs 1–2 in the rebaseline doc). This is a
**report milestone, not a byte/number gate** — 0c is report-only (the comparison is confounded; see the plan).

**Code shipped in this branch (the "seam", authored on the Mac, unrun):** `src/vqa_dataset.py` now reads a
config `ct_snapshot_prefix` (default `…/vista/feb26` via `DEFAULT_CT_SNAPSHOT_PREFIX`) instead of the hard-coded
`nov25` constant, and logs `[CT] source=local|gcs prefix=… blob=… file=…` per CT row for the 0a preflight. The
config read is threaded at the two construction sites `src/vista_run/run_bq.py` (weighted — used here) and
`src/vista_run/golden_harness.py` (weights-free — rungs 1–2). *In-lane note:* the plan's Files-to-modify listed
only `vqa_dataset.py`; the config read necessarily also touches those two caller sites (2 lines each) — mechanical,
design unchanged. Force-GCS needs **no** code: leaving `ct_dir` unset already routes to GCS.

A clean rung-0 result unblocks rungs 1–2.

## Step 0 — get the artifacts onto the VM

```bash
cd <vista_eval_vlm repo on the VM>
git fetch && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull
uv sync --extra dev   # or the repo's usual env sync
# Confirm the seam is present:
grep -n "ct_snapshot_prefix\|DEFAULT_CT_SNAPSHOT_PREFIX" src/vqa_dataset.py src/vista_run/run_bq.py
```
**Expected:** clean checkout at the pushed rung-0 SHA; `grep` shows the seam in all three spots (`DEFAULT_CT_SNAPSHOT_PREFIX` = feb26, the `ct_snapshot_prefix` param, the `run_bq.py` config read).
**STOP:** none — pure setup. (If the seam grep is empty, the Mac hasn't pushed yet — do not proceed.)

## Step 1 — 0a: author the VM-local overlay + preflight (no weights yet; gates GPU spend)

Author a **VM-local, uncommitted** overlay by **copying the committed `configs/all_tasks.yaml`** (already Ryan's
v1_1 layout — `/data/fries/...` `base_dir`, `valid_tasks_v1_3.json`, `subsample: true`) and overriding **only**:

```yaml
# models: exactly one (committed lists 5 > MAX_MODELS=4)
models:
  - gemma3 google/medgemma-1.5-4b-it
tasks:
  - progression_recurrence_free_survival_1_yr    # PFS; uncomment / make it the sole active task
experiments:
  - no_image
  - axial_all_image
paths:
  ct_snapshot_prefix: chaudhari_lab/ct_data/ct_scans/vista/feb26   # feb26 reroute
  # ct_dir: <REMOVE / leave unset>  -> forces GCS (no local nov25 fallback)
runtime:
  use_constrained_decoding_for_binary: true      # match Ryan's constrained baseline (OQ-R6)
  # subsample stays true (inherited)
```
Notes: (a) `ct_snapshot_prefix` goes under `paths:` (next to where `ct_dir` lives — that's where the code reads it);
(b) do **not** start from a `.vm.yaml` (VM-side, points at v1_5 `su-vista-*` mounts — wrong substrate);
(c) do **not** commit the overlay.

**Preflight** (no weights): on a small v1_1 PFS sample, classify stored `nifti_path` shapes, derive `(study, series)`,
and check the corresponding **feb26 blobs exist**; log counts + resolved `source`.

**Expected:** overlay loads (1 model ≤ MAX_MODELS=4); feb26 blob-existence ≈ 100% for the v1_1-selected UIDs; every
`nifti_path` shape maps to a valid feb26 reconstruction; `vista_bench_v1_1` queryable **independent of the local BQ
cache** (`run_bq.py` reads a local cache before BQ — bypass it for a true live-BQ check).
**STOP:**
- feb26 blob-existence **materially < 100%** → superset assumption (feb26 ⊇ nov25 by `(study, series)`) broken →
  report ≤5 offending UIDs-**as-structure** (no PHI), halt **before** GPU spend, hand back (class-3 deviation).
- an unhandled `nifti_path` shape (reconstruction wrong for that form) → halt, report the shape counts.
- `vista_bench_v1_1` not queryable → halt.

## Step 2 — 0b: weighted run

```bash
# with the 0a overlay active:
bash eval/bq_gcp.sh        # (or: python -m vista_run.run_bq ...) — WEIGHTED, loads medgemma weights
```
**Expected:** log shows exactly **one** model, PFS, and **both** `no_image` + `axial_all_image` ran; the `[CT]`
source-log shows `source=gcs` on every CT row (force-GCS holding); `image_count > 0` on CT rows for
`axial_all_image`; results CSVs at `{results_dir}/…/{task}_results_{experiment}.csv`, non-empty, resumable by `index`.
**STOP:**
- weights / HF auth fail → halt.
- **any** CT row logs `source=local` → force-GCS breached (a local nov25 cache survived) → halt; the run is not
  proving the feb26 reroute. (Either the overlay still has `ct_dir` set, or a default cache dir exists.)
- `row_count == 0` for a CT-bearing experiment → link/query broken, halt.

## Step 3 — 0c: Ryan-adjacent comparison (REPORT, not a gate)

**Derivation first — the raw result CSVs lack the accuracy columns.** They carry `model_response` + raw `label`
only; `predicted_label`/`ground_truth_label` are derived downstream by `src/results/all_model_response.py`
(`_extract_answer` + `map_answer_to_label_key(mapping)`), where `mapping` loads from `base_dir/tasks/valid_tasks.json`
— a **different file** than the run registry `valid_tasks_v1_3.json`. So:
```bash
# derive predicted/ground-truth labels from the run's result CSVs:
python -m results.all_model_response   # (against the overlay's results_dir)
```
- **Confirm the PFS `mapping` loaded non-empty** — if `valid_tasks.json` is absent/empty on the v1_1 `base_dir`, the
  mapping is `{}` and every `predicted_label` silently becomes `-1` → garbage accuracy. Point the metrics registry at
  the v1_3 tasks JSON, or verify `valid_tasks.json` exists on `base_dir`, first. (Under constrained decoding the
  response is forced to `Yes`/`No`, so `_extract_answer` is trivial — but the `Yes`/`No`→label map still needs it.)

**Compare** accuracy = `mean(predicted_label == ground_truth_label)` per experiment, vs Ryan's committed
`figures/results_stats/all_model_response.csv` (the **sole** committed baseline; itself **constrained**, so decoding is
now matched — OQ-R6), filtered to the PFS task and the mapped Ryan label (OQ-R2):
`axial_all_image → image_and_timeline`, `no_image → timeline_only`. First confirm the comparator rows exist
(`model_name == medgemma-1.5-4b-it`, PFS task — ~34 rows/cell).

**Expected / Report (no hard gate):** the two mapped pairs side-by-side with their delta, plus the ±image delta —
but treat the delta as **one weak, confounded signal** (report presence + 30-vs-10 slices, OQ-R2; small n ~34/cell,
OQ-R3; ~5% orientation-fixed CTs). Success = a sane, same-ballpark read within the ~10% *informal* band, **not** a
match claim.
**STOP:** none — report-only. If the mapping/derivation/comparator can't be resolved, downgrade to "completed
weighted run, sane label distribution" with no closeness claim (record which).

## Report back

Append under `## VM run results`: per-step pass/fail against each **Expected** block; the 0a blob-existence rate +
`nifti_path` shape counts; confirmation the 0b `[CT]` log is `source=gcs` throughout; the 0c accuracy pairs +
deltas (numbers only — **no** result rows, timelines, UIDs beyond structure, or PHI). Read large outputs from the
result CSVs; do not paste them. If any STOP fired, write a `⚠️ DEVIATION (class 3)` block and flip the `next.md`
pointer to BLOCKED.

## VM run results
_(left empty by the planner; the executor fills this in readback mode)_
