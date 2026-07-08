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

**Status: 0a coverage ACCEPTED as a declared delta (Phil 2026-07-08); feb26 CT-load validated weight-free; 0b (weighted) BACKLOGGED to the GPU machine (this box is CPU-only under Claude Code). 0c pending 0b.** See the **UPDATE (2026-07-08, proceed)** block at the end — it supersedes the initial "class-3 handback" framing below, which is retained for the audit trail (it explains *why* 86.9% is the accepted delta).
Executor `phil-sllm-01`, 2026-07-08, branch `worktree-vlm-modular-preprocessing-roadmap` @ `fa16312` (pulled ff-only). No GPU spent; no weights loaded; no patient rows touched (counts / shape-patterns / UIDs-as-structure only).

### Step 0 — setup + seam — PASS
- Pulled ff-only to `fa16312`; working tree clean. Seam grep shows all three spots: `DEFAULT_CT_SNAPSHOT_PREFIX = chaudhari_lab/ct_data/ct_scans/vista/feb26` + the `ct_snapshot_prefix` param in `vqa_dataset.py` + the config read in `run_bq.py:916,920`.
- gcloud project `som-nero-plevriti-deidbdf`, authed `padamson@stanford.edu`; repo `.venv` (Python 3.11.15) resolves.
- ⚠ **No GPU on this box** (`nvidia-smi` absent — consistent with the prior golden run's `libcuda.so.1: cannot open`). So the *weighted* 0b is not runnable via Claude Code here regardless; it belongs to the separate GPU machine. Only 0a (weight-free, the gate) was in scope here — which is exactly the step that gates GPU spend.

### Step 1 — 0a preflight (weight-free) — **FAILED THE GATE**
Ran the real resolver `vqa_dataset._nifti_path_to_blob_and_filename(prefix=feb26)` over the **live-BQ** v1_1 PFS cohort (bypassing the local BQ cache — queried `vista_bench_v1_1.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1 WHERE task='progression_recurrence_free_survival_1_yr'` directly).

- **Cohort / CT rows:** 9,350 PFS rows; **2,362** carry a non-null `nifti_path`.
- **`nifti_path` shape (OQ-R4) — CLEAN, single shape:** all 2,362 are `(/mnt/, snapshot-token=nov25, .nii.gz, nparts=9)` → i.e. `/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/nov25/<studyUID>__<seriesUID>.nii.gz`. The resolver reconstructs all 2,362 to `feb26/<studyUID>__<seriesUID>.nii.gz` with **0** bad reconstructions. The prefix swap is a real reroute (else-branch), not a no-op. No unhandled shape.
- **feb26 blob-existence (the gate):** full cohort **2,053 / 2,362 = 86.9%** exist; **309 (13.1%) missing** (seeded 400-sample agreed at 87.2%). Threaded existence check against `gs://su-vista-uscentral1/…/feb26/`.
- **Missing not recoverable under sibling prefixes:** the missing `<studyUID>__<seriesUID>.nii.gz` are absent under both `…/feb26/` (77,627 objects) and `…/Feb/` (only **1** object — a stray, not a snapshot). Sample offending blobs available in the VM run log (UIDs redacted here as `feb26/<studyUID>__<seriesUID>.nii.gz` to keep them off the committed tree).

**Gate verdict (per Step 1 STOP):** feb26 blob-existence is **materially < 100%** → **Assumption #1 (feb26 ⊇ nov25 keyed by `(study, series)`) is BROKEN for ~13% of the v1_1 PFS CT set** → **halt before GPU spend, class-3 deviation → Mac planner.**

### Steps 2–3 (0b weighted run, 0c comparison) — NOT RUN
Blocked by the 0a gate (and no GPU here). Running 0b as-is would silently degrade the 309 missing-CT rows to `image_count=0` (the resolver's silent no-image fallback), confounding the `axial_all_image` arm of 0c — precisely what the 0a gate exists to prevent. No overlay config was finalized (its only consumer is the weighted 0b; premature given the STOP + the GPU-machine base_dir/mount unknowns).

### Root cause (for the re-plan) — this is the v1_1 ↔ v1_5 CT-set drift
feb26 was re-exported aligned to the **v1_5** materialized dataset's `image_study_uid` / `image_series_uid` keys (rebaseline doc go-forward; a v1_5 UID pair was already confirmed to resolve to a feb26 blob). Rung-0's premise — *"keep v1_1's `nifti_path` selection, change only the storage prefix"* — assumed feb26 is a superset of nov25 on `(study, series)`. It is not: ~13% of v1_1's nov25-era `(study, series)` keys were dropped or re-keyed in the feb26 reprocess, so a filename-identical lookup 404s for them.

### Mac decisions to make (re-plan)
1. **Rung-0 scope:** accept 86.9% CT coverage as a **declared delta** and run 0b on the resolvable subset (309 rows degrade to no-image — a labeled confound on `axial_all_image`, on top of the OQ-R2 report/slice confounds); **or** treat rung-0-on-v1_1 as unbankable and fold operability into rung-1 (v1_5 substrate) directly.
2. **Are the 309 recoverable?** This is exactly **rung-1's Axis A retarget** — resolve CT via the v1_5 `(image_study_uid, image_series_uid)` link instead of the v1_1 `nifti_path` keys. Whether the 309 v1_1 scans exist in feb26 under *different* v1_5 UIDs (renamed) vs are genuinely dropped (reprocess excluded them) needs the v1_5-table person_id/UID join — planner/rung-1 work, not an in-lane executor correction.
3. Fold this gate result into the rung-1/rebaseline plan; rung-0 as written does not clear its own gate.

**Executor state at handback:** no code changed; no commits; no golden/results artifacts written; working tree clean at `fa16312`. Only this doc + `docs/next.md` edited. PHI: counts / shape-patterns / UIDs-as-structure only — no patient rows, no timelines, no scan contents.

---

## UPDATE (2026-07-08, proceed) — 86.9% accepted as declared delta; feb26 CT-load validated; 0b backlogged to GPU

**Phil's call (2026-07-08):** 86.9% feb26 coverage is acceptable — proceed with rung-0. Rationale: `axial_all_image` records per-row image loading (`used_image`/`image_count`), so at **0c we stratify to `used_image==1` rows** and compare on a like (CT-present) cohort; the 309 missing-CT rows fall back to no-image and drop out of the image arm. So the 0a STOP is downgraded from **class-3 handback → class-2 declared delta** (recorded in the overlay header).

### 0a follow-up — feb26 CT-load validated end-to-end (weight-free), partial axial golden banked
The 0a preflight only proved feb26 blobs *exist*. Ran the weights-free golden harness on `axial_all_image` (`--limit 30`, PFS × medgemma-1.5-4b-it) with the rung-0 overlay to prove they actually **load**:
- **Every CT row logged `source=gcs`** (force-GCS holding; `local_exists=False`) — the feb26 reroute is exercised, no local fallback.
- 30 rows: **14 CT-present → each loaded exactly 30 axial slices**; `len(image_hashes)==image_count` and `len(selected_indices)==image_count` for all 14; `assembly_mode=ordered`. 16 rows are CT-null/missing → `image_count=0` (correct silent fallback). (30-slice count = the Phase-0.5 even-spacing fix.)
- This **banks a partial `axial_all_image` legacy golden** (`…_axial_all_image_rung0_feb26_golden.jsonl`, on the PHI mount, git-clean) — the arm the 2026-07-06 handback declared unbankable because `nov25` was deleted. feb26 unblocks it.

### Overlay authored — `configs/all_tasks.rung0.vm.yaml` (uncommitted; `configs/*` gitignored)
Derived from the proven `all_tasks.vm.yaml` staging (`base_dir=/mnt/su-vista-uscentral1/vistabench/vlm/base`, the substrate that ran `no_image` green). Rung-0 overrides: 1 model (`gemma3 medgemma-1.5-4b-it`), PFS only, `experiments:[no_image, axial_all_image]`, `paths.ct_snapshot_prefix=…/feb26`, `ct_dir` unset (force-GCS), `use_constrained_decoding_for_binary: true` (Ryan-match, OQ-R6). **Declared deltas baked in:** (1) 86.9% CT coverage; (2) `subsample: false` — staged base carries only the full `{task}.csv`, no `_subsampled` variant, so this runs Ryan's cohort **un-subsampled** (larger n; 0c stays Ryan-*adjacent*, not exact); (3) `valid_tasks.json` (not `_v1_3`) — the staged registry (label mapping for 0c loads from it; non-empty, proven by the green `no_image` run).

### 0b (weighted) — BACKLOGGED to the GPU machine (not runnable here)
This box (`phil-sllm-01`) has **no GPU** (`nvidia-smi` absent, `torch.cuda.is_available()==False`) — the weighted medgemma run can't run under Claude Code here; per the machine posture it belongs to the separate user-managed GPU machine. **Turnkey command** (GPU box, repo on this branch, mounts present):
```bash
cd src
python -m vista_run.run_bq --config ../configs/all_tasks.rung0.vm.yaml
# (or: bash eval/bq_gcp.sh after pointing CONFIG at the rung-0 overlay; MAX_MODELS=4 OK — 1 model)
```
**0b expected:** exactly one model, PFS, both experiments run; `[CT]` log `source=gcs` throughout; `image_count>0` on CT-present `axial_all_image` rows; result CSVs at `{results_dir}/…/{task}_results_{experiment}.csv`, non-empty, resumable by `index`. **Stop:** weights/HF-auth fail; any `[CT] source=local` (force-GCS breach); `row_count==0`.

### 0c (report) — recipe for after 0b
```bash
cd src && python -m results.all_model_response   # derive predicted/ground_truth from result CSVs
```
- Confirm the PFS `mapping` loaded non-empty from `base_dir/tasks/valid_tasks.json` (else every `predicted_label→-1`).
- **Stratify** `axial_all_image` to `used_image==1` rows (drops CT-null + the 309 feb26-missing) so the ±image comparison is on a like cohort.
- Compare accuracy vs `figures/results_stats/all_model_response.csv` (constrained, sole committed baseline) for PFS, mapped labels: `axial_all_image→image_and_timeline`, `no_image→timeline_only`. Report the two pairs + delta as **one weak, confounded signal** (report-presence + 30-vs-10 slices + un-subsampled n + ~5% orientation-fixed). Success = sane, same-ballpark read within the ~10% informal band — not a match claim.

**Executor state (this UPDATE):** authored `configs/all_tasks.rung0.vm.yaml` (uncommitted, gitignored); banked a partial `axial_all_image` golden on the PHI mount (git-clean); no code changed; working tree edits = this doc + `docs/next.md` only. PHI: counts / structure only.
