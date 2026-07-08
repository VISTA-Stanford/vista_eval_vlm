Reference: research-skills/claude_ops.md

# Rung 0 — reproduce Ryan's weighted pipeline on feb26 (operability smoke)

**Status: Draft** (2026-07-08). Split out (2026-07-08, Phil) from
[`vlm-ct-feb26-v1_5-golden-rebaseline.md`](vlm-ct-feb26-v1_5-golden-rebaseline.md) so the operability
smoke is a small, self-contained plan that **gates** the larger CT/feb26/v1.5 rebaseline work.
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` (continues; not yet committed).
**Machine posture:** authored on the planner Mac (no runtime/data/creds). VM-executed steps run on the
GCP executor VM (`som-nero-plevriti-deidbdf`) that holds the `gs://vista_bench/` bucket + `su-vista-*` mounts.

## Relationship to the rebaseline plan (this GATES it)

This is **rung 0** of a three-rung ladder. The other two rungs live in the rebaseline doc:

- **Rung 0 (this doc):** reproduce Ryan's *weighted* pipeline on feb26, changing only the CT storage prefix.
  Prove we can drive the VM and get sane, Ryan-adjacent results **before** touching the substrate. Report
  milestone, not a byte/number gate.
- **Rung 1 (rebaseline doc, Axes A + B):** substrate cut v1_1 → v1_5/feb26 — retarget CT resolution off the
  `nifti_path` heuristic onto the `(image_study_uid, image_series_uid)` link; flip the dataset constant.
- **Rung 2 (rebaseline doc, Axis C / 3b):** dissolve the legacy `__getitem__` CT branch into the CT adapter,
  gated byte-identically by the golden harness.

**Rung 0 must be green before rungs 1–2 proceed.** The rebaseline doc carries a prerequisite pointer back here.

## Goal

Prove **operability**: drive the VM end-to-end and reproduce Ryan's *actual weighted pipeline* (loads model
weights, emits a results CSV — **not** the weights-free golden harness) on a **near-untouched v1_1 substrate**,
changing **only** the CT storage prefix (nov25 → feb26). This isolates *"can we run it"* from *"did the v1.5
substrate cut change things"* — the axes in the rebaseline doc assume v1.5 "works" for real evals but never
gate it (every Verification step there is the weights-free **golden harness** — `golden_harness.py` iterates
`PromptDataset` directly and never loads weights), so nothing downstream ever loads weights or emits a results
CSV. Rung 0 fills that missing operability gate.
(Citation precision: the **weighted** path is `run_bq.py`, which loads weights via `_ensure_model_loaded`
(`run_bq.py:171`) — its docstring at `:144` only explains *why* weight-loading is kept out of `__init__`, so it
is not the weights-free walker.)

Secondary outcome: exercise the **feb26 CT reroute** through the real GCS path (force-GCS), so we learn the
feb26 blobs resolve and load before the rebaseline work depends on them.

## Background (self-contained)

Only the facts rung 0 depends on — see the rebaseline doc's *What this supersedes* / *Reprocessing & historical
CT test-set visibility* sections for the full substrate story.

- **`nov25` is fully deleted** from `gs://su-vista-uscentral1/…/vista/`; `feb26` is the *only* CT snapshot.
  So even a "reproduce Ryan" run must resolve CTs from feb26 — there is no nov25 blob left to load from GCS.
- **Rung 0 keeps Ryan's substrate untouched otherwise:** `VISTA_BENCH_DATASET = vista_bench_v1_1`
  (`query_utils.py:236`), the `valid_tasks_v1_3.json` task registry, the `nifti_path` CT selection, and
  `subsample: true` (his subsampled cohort, `figures/data_stats/person_id_subsampled.csv`). **No Axis A
  query-retarget, no Axis B dataset flip.** Keeping v1_1's `nifti_path` selection is what pins Ryan's *exact*
  `(study, series)` set — the only variable is where those keys are stored (nov25 → feb26).
- **Cohort-level reproducibility handle:** the committed `figures/data_stats/person_id_subsampled.csv` +
  `figures/results_stats/all_model_response.csv` pin exactly which `(task, experiment, index, person_id)` ran,
  so Ryan's run is re-derivable on the VM. (File/UID-level visibility is *not* recoverable — no past run banked
  `nifti_path`/UIDs; see rebaseline doc.)
- **Repo vs VM artifacts (don't hunt in the repo):** `person_id_subsampled.csv` / `all_model_response.csv` are
  **repo-committed**, but the task/prompt registries `valid_tasks_v1_3.json` (+ `prompts_by_task.json`,
  `image_valid_tasks.json`) are **VM `base_dir` artifacts** — the repo has **no `tasks/` dir**; they resolve
  against `base_dir` at `run_bq.py:67`. This matters for 0c's label mapping (see Verification).

## Approach

### The only code change: pull the `ct_snapshot_prefix` config seam forward + force-GCS

Rung 1's Axis A already makes the CT storage prefix a **config value** (default
`chaudhari_lab/ct_data/ct_scans/vista/feb26`) instead of the hard-coded
`DEFAULT_NIFTI_BUCKET_PREFIX = ".../vista/nov25"` (`vqa_dataset.py:15`). **Pull that seam forward into rung 0**
(Phil 2026-07-08, chosen over a bare constant swap) — it is *not* throwaway (rung 1 needs it regardless), and
it cleanly resolves the two fragilities Codex flagged (silent local-cache fallback; prefix-swap-as-no-op).

The resolver `_nifti_path_to_blob_and_filename` builds `blob = {prefix}/{study}__{series}.nii.gz`
(`vqa_dataset.py:42`); with `ct_snapshot_prefix = feb26` and the v1_1 `nifti_path` selection untouched, Ryan's
*same* `(study, series)` scans resolve to feb26 blobs (feb26 ⊇ nov25 on those keys — see *Assumptions*).

**Resolver mechanism to verify on the VM.** Once the prefix flips to feb26, a stored `nifti_path` embedding
`.../nov25/...` no longer matches the `startswith(prefix)` early-return (`vqa_dataset.py:28-31`) and falls to
the else-branch (`:32-43`), which **reconstructs** `{feb26}/{study}__{series}.nii.gz` from the filename for
`.nii.gz`-terminated paths — so the repoint is *not* a blind no-op. But non-`.nii.gz` shapes (`.zip`, bare
`study/series`) reconstruct differently, and the stored shape is a VM unknown → the **0a preflight classifies
`nifti_path` shapes** and logs sampled `(input form, blob_path, filename, source=local|gcs)` counts before
trusting the reroute (**OQ-R4**).

**Force GCS for the smoke.** The config `ct_dir` points at a *local* nov25 download
(`/data/fries/.../downloaded_ct_scans/.../nov25`), normally tried before GCS (`vqa_dataset.py:160-173`). If it
survives on the VM the run silently loads **local nov25** bytes by filename and never proves the feb26 reroute.
So the rung-0 gate run **forces GCS by leaving `ct_dir` unset** — already the default GCS route with **no new
code** (`vqa_dataset.py:163-169`: `ct_dir=None → use_local=False → GCS branch`); a `force_gcs` flag is optional.
The run then *must* exercise the feb26 path.

**Local-cache "Ryan-exact" is a separate, labeled outcome (not the rung-0 gate).** If the local nov25 cache
survives on the VM it yields **exact** Ryan bytes — worth banking, but as its own run labeled `ryan_exact_local`,
**not** the feb26 gate (naming the rung "on feb26" while loading local nov25 never proves the reroute — Codex).
The gate run forces GCS; the local-cache exact run is an optional bonus.

### Declared deltas (this branch vs Ryan's original code)

The run is **"Ryan + declared deltas,"** not a digit-match:
- The 30-slice even-spacing fix (`vqa_dataset.py:191-210`, commit `04b97d8` — Ryan's sampler *overshot*) sits on
  the `axial_all_image` CT path and therefore can perturb the image arm.
- The `unit_source_value` EHR remap (already committed on this branch).

Both are deterministic input changes → fine for an operability/sanity smoke. *If a true digit reproduce is
wanted, cherry-pick only the prefix swap onto Ryan's baseline SHA instead of running from this branch.*

## Files to modify

- **`src/vqa_dataset.py`** — make the CT-storage prefix read a config `ct_snapshot_prefix` (default feb26)
  instead of the hard-coded `DEFAULT_NIFTI_BUCKET_PREFIX` (`:15`); add source-logging (`source=local|gcs`) for
  the 0a preflight. **Force-GCS needs no new code:** leaving `ct_dir` unset already routes to GCS
  (`vqa_dataset.py:163-169`), so the overlay just omits it; a `force_gcs` flag is optional. The
  `_nifti_path_to_blob_and_filename` resolver body is otherwise unchanged for rung 0 (rung 2 in the rebaseline
  doc replaces it with the UID resolver). **Net-new code = the config prefix read + source-logging.**
- **VM-local overlay config (uncommitted)** — **start from the committed `configs/all_tasks.yaml`** (which is
  already Ryan's v1_1 layout: `/data/fries/...` `base_dir`, `valid_tasks_v1_3.json`, `subsample: true`) and
  override only: 1 model `gemma3 google/medgemma-1.5-4b-it`, task
  `progression_recurrence_free_survival_1_yr` uncommented, `experiments: [no_image, axial_all_image]`,
  `ct_snapshot_prefix: …/feb26`, `ct_dir` unset, `use_constrained_decoding_for_binary: true` (match Ryan's
  constrained baseline — OQ-R6). Inherit the committed `base_dir`/`results_dir`/`cache_dir`. Do
  **not** commit; do **not** start from a `.vm.yaml` — that file is VM-side and points at the **v1_5**
  `su-vista-*` mounts (the wrong substrate for rung 0's v1_1 reproduce).

No other code changes in rung 0. (Axis A's `nifti_path` → UID retarget, the `VISTA_BENCH_DATASET` flip, and the
3b adapter dissolution all belong to rungs 1–2 in the rebaseline doc.)

## Decisions

- **Force-GCS is the rung-0 gate (OQ-R1, Phil 2026-07-08).** Pull the `ct_snapshot_prefix` seam forward, set
  feb26, unset `ct_dir` so the smoke can't fall back to local nov25 and *must* prove the feb26 reroute. The
  local-cache exact run is a separate optional `ryan_exact_local` bonus, not the gate.

## Assumptions not verifiable in this repo (validate on the VM *before* GPU spend)

Two data facts underpin rung 0 and cannot be checked from this repo's code:

1. **feb26 ⊇ nov25 keyed by `(study, series)`** (the reprocess re-exported the same keys — Phil / vista_bench
   v1.5 changelog). *Validation gate:* the 0a preflight resolves a small v1_1 PFS sample's `(study, series)` →
   confirms the feb26 blobs exist → only then run weighted inference. A missing-blob rate materially above noise
   breaks the superset assumption and **stops the rung before GPU spend**.
2. **~5% of scans orientation-corrected** (upside-down fix), accepted as within VLM stochastic noise — this is
   part of why rung 0 is a *sanity* smoke, not a digit reproduce.

## Open questions

- **OQ-R1 — force-GCS vs local-cache-exact: RESOLVED (Phil 2026-07-08) = force GCS is the gate.** See Decisions.
- **OQ-R2 — experiment-name mapping: RESOLVED (repo + git-history investigation, 2026-07-08).** The committed
  baseline `figures/results_stats/all_model_response.csv` uses four labels; the current code emits `no_image` /
  `axial_all_image`. The apples-to-apples mapping — and the working hypothesis in the earlier draft — was
  checked against the actual code + git history:
  - **Current experiment semantics** (`src/context/presets.py:78-85`, `docs/04-running-the-pipeline.md:96-97`):
    `no_image` = **full timeline, no image, no separate report section**; `axial_all_image` = the **same full
    timeline + 30 axial CT slices**. They are a clean **±image pair** (identical text, differ only by the image).
  - **Ryan's baseline labels are display-renames** of raw experiments via `EXPERIMENT_DISPLAY_NAMES` (introduced
    in commit `96fd42f`, which also regenerated the CSV): `image_only`←`no_timeline` (image, **zero timeline
    text**), `image_and_timeline`←`no_report` (image + report-**stripped** timeline), `report_and_timeline`←
    `report` (report-stripped timeline + re-appended report, no image), `timeline_only` (pass-through,
    report-stripped timeline, no image).
  - **Corrected mapping:**
    - `axial_all_image ↔ image_and_timeline` — **the earlier `axial_all_image ↔ image_only` hypothesis is
      REFUTED.** `image_only` (raw `no_timeline`) feeds the model an image with **zero timeline text**, whereas
      `axial_all_image` feeds the **full timeline + image**. The only "image + timeline" Ryan label is
      `image_and_timeline`.
    - `no_image ↔ timeline_only` — **CONFIRMED** as the ±image-consistent counterpart (Ryan's text-only
      ablation of `image_and_timeline`). Mapping `no_image → report_and_timeline` instead is rejected: Ryan
      never ran image+report+timeline, so `axial_all_image` would then have no counterpart.
  - **⚠ The comparison is confounded — on BOTH absolute accuracy AND the ±image delta.** The current arms and
    Ryan's arms differ in *two* ways beyond the image:
    1. **Report presence.** Current `no_image`/`axial_all_image` use the **full, report-inclusive** timeline
       (`use_no_report_csv=False` — `golden_harness.py:113-116`, `run_bq.py:201-202`); Ryan's
       `timeline_only`/`image_and_timeline` use the **report-stripped** timeline (`_subsampled_no_img_report.csv`,
       `remove_imaging_report.py`). So current timelines embed the radiology report describing that very CT;
       Ryan's do not.
    2. **Slice count.** `axial_all_image` loads **30** CT slices (`presets.py:84` `_ct_block(30)`,
       `vqa_dataset.py:191-210`); Ryan's `image_and_timeline` (= raw `no_report`) loads **10** (`presets.py:92`
       `_ct_block(10)`).
    Absolute accuracies are therefore **not byte-comparable**. And the **±image delta is confounded too**, not
    "confound-robust": the delta measures the image's *marginal* value under different text+dose conditions
    (adding a 30-slice image to text that already narrates the scan ≠ adding a 10-slice image to text that
    doesn't). Treat the delta as **one weak directional signal**, not a clean apples-to-apples number. This is
    why 0c stays **report-only** (OQ-R3).
  - *If a tighter numeric reproduce is wanted later:* run Ryan's actual `no_report` (→`image_and_timeline`, 10
    slices) and raw `timeline_only` experiments, which use the same report-stripped timelines — out of scope for
    this smoke (different slice count + a separate report-stripped CSV dependency).
- **OQ-R3 — "Ryan-adjacent" tolerance: RESOLVED (soft) — 0c is report-only (Phil 2026-07-08).** A ~10%
  absolute-accuracy band is the *informal* "adjacent" heuristic, **not** a hard pass/fail gate ("idk 10%ish?
  we'll see" — Phil). Two reasons the band is soft: (1) the OQ-R2 confounds (report presence + 30-vs-10 slices)
  make both the absolute numbers and the ±image delta non-apples-to-apples; (2) **small n** — the PFS_1yr
  baseline has ≈34 rows per (experiment, model) cell (image arms fewer, being CT-restricted), so the SE on
  accuracy is ~8–9% and a ~10% band sits inside one SE for both the absolute and the delta. So a "pass" is a
  sanity read, not evidence of a match. Revisit the number after the first real 0c results.
- **OQ-R4 — prefixed-`nifti_path` handling: RESOLVED via the config seam + 0a preflight.** No one-off substring
  rewrite: the resolver reads `ct_snapshot_prefix`, and the 0a preflight classifies stored `nifti_path` shapes
  so the executor confirms the feb26 reconstruction *per shape* rather than assuming a single form.
- **OQ-R5 — task count: RESOLVED (Phil 2026-07-08) = PFS-only for the first pass.** The run is `subsample: true`
  either way, so runtime is driven by the number of task × experiment arms, not cohort size. Keep rung 0 to the
  single canonical PFS task × `{no_image, axial_all_image}` (2 arms — fastest, a minimal "does it run" test). The
  tiny-n caveat stands but is acceptable for a report-only 0c: the PFS baseline has ≈**34** rows per
  (experiment, model) cell vs ≈**57** for `has_recurrence_1_yr` — **both from Ryan's *subsampled* baseline** (the
  earlier "912" was has_recurrence's per-*task* total = 57 × 16 cells, not a per-cell or subsample-vs-full
  figure). **Cheap follow-up (not now):** if the PFS 0c read is too noisy to see anything, add
  `has_recurrence_1_yr` as a second task in the overlay — one line, no code change, no new gate.
- **OQ-R6 — decoding-mode parity: RESOLVED from the repo (git-history + CSV contents, 2026-07-08).** Ryan's
  committed baseline was generated under **constrained** decoding, and there is **no** committed
  `constrained_all_model_response.csv` — the single committed baseline `figures/results_stats/all_model_response.csv`
  *is* the constrained artifact (despite the un-prefixed name; that name belongs to a *builder script* that was
  never run to a committed output). Four converging signals: its last-write commit is `5e97abd`
  *"add in constrained decoding…"* (2026-02-11); `run_bq.py` at that commit (`@5e97abd:380`) forced `["Yes","No"]`
  for **every** binary task with no opt-out (the `use_constrained_decoding_for_binary` knob did not exist yet —
  added ~a month later: code `c3f985a`, config `584609d`); the config then used
  `results_dir: results_constrained_decoding_v2`; and the CSV's `model_response_cleaned` column is 100%
  `Yes`/`No` (11,048 / 2,408) with zero free text — the forced-decode signature.
  - **Decision → match Ryan: set `use_constrained_decoding_for_binary: true` in the rung-0 overlay.** The committed
    `all_tasks.yaml:17` `false` is precisely what *diverges* from Ryan; matching **removes the decoding confound
    from 0c** (leaving only OQ-R2's report/slice confounds + small n). No VM check needed — the baseline's decoding
    is pinned by the code/config at the CSV's own last-write commit plus the CSV's contents.
  - **⚠ Knob trap:** `run_bq.py:909` defaults the *absent* key to `True`, so a stripped-out line yields
    constrained (baseline-matching) — set it **explicitly** to avoid ambiguity.
  - *If instead you want rung 0 to smoke-test the go-forward **unconstrained** default,* that's a deliberate scope
    change (say so); rung 0 as written reproduces Ryan → constrained.

## PHI-in-history (workstream-wide — pointer)

Real DICOM Study/Series UIDs sit in git *history* (not the current tree) — see the rebaseline doc's
*Reprocessing & historical CT test-set visibility* section for the enumerated exposures. **No history rewrite
without Phil's sign-off.** Rung-0 reports use counts / field-names / UIDs-as-structure only.

## Verification & VM handoff

Executed on the GCP VM (Mac is planner-only). Canonical smoke = `progression_recurrence_free_survival_1_yr`
(PFS) × `gemma3 google/medgemma-1.5-4b-it`. **Weighted** (`eval/bq_gcp.sh` → `run_bq.py`), *not* the golden
harness. `/vm-handoff` renders this into a runnable `docs/vm-status/<date>-<sha>.md`. Three sub-steps; the
preflight (0a) gates GPU spend.

- **0a — config overlay + preflight (no weights yet).** The committed `configs/all_tasks.yaml` will **not** run
  the smoke as-is: it lists **5 models** (`eval/bq_gcp.sh:8` `MAX_MODELS=4` rejects >4), its only active task is
  `has_recurrence_1_yr` (PFS commented, `:40`), and its **sole active experiment is `path_full`** (`:83`) — not
  the smoke's `no_image`/`axial_all_image` (all other experiments are commented; `run_bq`'s `['no_image']`
  default at `run_bq.py:178` only fires when `experiments:` is *empty*, which it isn't). Author a **VM-local
  uncommitted overlay** by **starting from the committed `all_tasks.yaml`** (already Ryan's v1_1 layout —
  `/data/fries/...` `base_dir`, `valid_tasks_v1_3.json`, `subsample: true`) and overriding only: single model
  `gemma3 google/medgemma-1.5-4b-it`, task `progression_recurrence_free_survival_1_yr` uncommented,
  `experiments: [no_image, axial_all_image]`, `ct_snapshot_prefix: …/feb26`, `ct_dir` unset (the default GCS
  route), `use_constrained_decoding_for_binary: true` (match Ryan's constrained baseline — OQ-R6). Inherit the
  committed `base_dir`/`results_dir`/`cache_dir`. Do **not** commit; do **not** start from a
  `.vm.yaml` (VM-side, v1_5 `su-vista-*` mounts — wrong substrate).
  **Preflight:** on a small v1_1 PFS sample, classify `nifti_path` shapes, derive `(study, series)`, and check
  the corresponding **feb26 blobs exist**; log counts + resolved `source`.
  **Expected:** overlay loads (1 model ≤ MAX_MODELS); feb26 blob-existence ≈ 100% for v1_1-selected UIDs; every
  `nifti_path` shape maps to a valid feb26 reconstruction; `vista_bench_v1_1` queryable **independent of the
  local BQ cache** (`run_bq.py` reads a local cache before querying BQ, so a live-BQ check must bypass it).
  **Stop:** feb26 blob-existence materially < 100% (superset assumption broken → report ≤5 offending
  UIDs-as-structure, no PHI); an unhandled `nifti_path` shape; v1_1 not queryable.
- **0b — weighted run.** `eval/bq_gcp.sh` (or `run_bq.py`) with the 0a overlay, **weighted**.
  **Expected:** log shows exactly one model, PFS, and **both** `no_image` + `axial_all_image` ran;
  `image_count > 0` on CT rows for `axial_all_image`; results CSVs at
  `{results_dir}/…/{task}_results_{experiment}.csv`, non-empty, resumable by `index`. **Stop:** weights / HF
  auth fail; any CT row loads from `source=local` (force-GCS breached); `row_count == 0` for a CT-bearing
  experiment.
- **0c — Ryan-adjacent comparison (REPORT, not a gate).**
  **Derivation (do not skip — the raw CSVs lack the accuracy columns).** The weighted run's result CSVs contain
  `model_response` + the raw `label` only (`run_bq.py:_build_result_row :806-834`) — **not**
  `predicted_label`/`ground_truth_label`. Those are derived *downstream* by `src/results/all_model_response.py`
  via `_extract_answer` + `map_answer_to_label_key(mapping)`, where `mapping` is loaded from
  `base_dir/tasks/valid_tasks.json` (`all_model_response.py:69`, `final_metrics.py:233`) — a **different file**
  than the run registry `valid_tasks_v1_3.json`. So 0c must: (a) run the derivation (regenerate via
  `python -m results.all_model_response` against the overlay, or apply `_extract_answer` + the PFS `mapping` by
  hand); and (b) **confirm the PFS `mapping` loaded non-empty** — if `valid_tasks.json` is absent/empty on
  Ryan's v1_1 `base_dir`, the mapping is `{}` and every `predicted_label` silently becomes `-1`
  (`all_model_response.py:50,57`) → garbage accuracy. Point the metrics registry at the v1_3 tasks JSON or
  verify `valid_tasks.json` exists on `base_dir` first. (Under constrained decoding (OQ-R6) the response is forced
  to `Yes`/`No`, so `_extract_answer` is trivial — but the `Yes`/`No`→label mapping still needs a non-empty
  `valid_tasks.json`.)
  **Compare.** Accuracy = `mean(predicted_label == ground_truth_label)` per experiment, vs Ryan's committed
  `figures/results_stats/all_model_response.csv` filtered to the PFS task and the **mapped** Ryan label (OQ-R2):
  `axial_all_image` → `image_and_timeline`; `no_image` → `timeline_only`. **First confirm the comparator
  exists** — verify the exact `model_name` string (Ryan may have run a different medgemma version) and the exact
  `task` value present in the CSV before filtering; if no matching rows exist there is **no direct comparator**.
  **Report** the two mapped pairs side-by-side with their delta, plus the current-vs-baseline **±image delta**
  — but treat the delta as **one weak directional signal, itself confounded** (report presence + 30-vs-10
  slices, OQ-R2); do **not** over-read it. Remaining confounds to note in the report: **small n** (~34/cell,
  OQ-R3) and **~5% orientation-fixed CTs**. (Decoding mode is now **matched**, not a confound — the overlay sets
  `use_constrained_decoding_for_binary: true` to Ryan's constrained baseline, OQ-R6; and `all_model_response.csv`
  is the *sole* committed baseline, itself constrained.) **Success = a sane, same-ballpark read within the
  OQ-R3 ~10% *informal* band**, not a match claim. *If the mapping/derivation/comparator are unresolved,
  downgrade 0c to "completed weighted run, sane label distribution" with no closeness claim.*

**Rung 0 green → rungs 1–2 (the rebaseline doc) proceed.**

**PHI:** counts / field-names / UIDs-as-structure only — never paste result rows, timelines, or scan contents.
`/phi-vet` gates every commit.
