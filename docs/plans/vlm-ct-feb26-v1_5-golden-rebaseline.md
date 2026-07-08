Reference: research-skills/claude_ops.md

# VLM CT resolution → v1_5/feb26 dataset link + golden rebaseline on feb26

**Status: Draft** (2026-07-07) — supersedes the CT/substrate slice of
[`vlm-modular-preprocessing-and-context-viewer-roadmap.md`](vlm-modular-preprocessing-and-context-viewer-roadmap.md).
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` (continues; not yet committed).
**Machine posture:** authored on the planner Mac (no runtime/data/creds). VM-executed steps below run on the
GCP executor VM (`som-nero-plevriti-deidbdf`) that holds the `gs://vista_bench/` bucket + `su-vista-*` mounts.
**Handback source:** [`docs/vm-status/2026-07-06-golden-harness.md`](../vm-status/2026-07-06-golden-harness.md)
— the *DEVIATION → Mac planner (2026-07-07)* block. The `no_image`/EHR golden baseline is banked + green;
`axial_all_image` bounced back as a class-3 deviation (the `nov25` CT snapshot is deleted).

**Phasing (2026-07-08, Phil):** the ladder is **rung 0** reproduce Ryan (feb26, via pulling the
`ct_snapshot_prefix` config seam forward + force-GCS) → **rung 1** substrate cut to v1.5 (Axes A+B) → **rung 2**
3b modular refactor byte-identity net. **Rung 0 was split into its own gating plan
([`vlm-rung0-reproduce-ryan-feb26.md`](vlm-rung0-reproduce-ryan-feb26.md), 2026-07-08) to keep things simple** —
prove we can drive the VM and reproduce Ryan's *actual weighted pipeline* (results CSV, not the weights-free
golden harness) on a near-untouched v1_1 substrate *before* the v1.5 cut. **This doc now covers rungs 1–2 only**
(the v1.5 substrate cut + the 3b refactor); rung 0 is a **prerequisite** — see the gate below.

> **⚠ Prerequisite gate:** the rung-0 operability smoke
> ([`vlm-rung0-reproduce-ryan-feb26.md`](vlm-rung0-reproduce-ryan-feb26.md)) must be **green** before the axes
> and Steps below run. Rung 0 introduces the `ct_snapshot_prefix` config seam that Axis A builds on, and proves
> the feb26 CT reroute works end-to-end.

## What this supersedes

The roadmap's **Phase 0 v1_5 substrate contract** and **Back-compat golden** sections assumed:
- CT NIfTI resolves via a string `nifti_path` column, with `feb26` as *primary* and `nov25` as *fallback*.
- The 3b CT-dissolution byte-identity gate (Gate 1 / Gate 2) is banked against a **legacy `nov25` baseline**.

The VM disproved both:
1. **`nov25` is fully deleted** from `gs://su-vista-uscentral1/…/vista/` (zero objects). `feb26` is the *only*
   CT snapshot. The "nov25 fallback" line is dead.
2. **v1_5 drops the string `nifti_path`** (now a deprecated INTEGER) and links CTs **only** via
   `image_study_uid` + `image_series_uid` → `…/vista/feb26/{study}__{series}.nii.gz` (the feb26 blob was
   verified to exist for a real (study, series) pair). The `nifti_path`-heuristic loader
   (`_nifti_path_to_blob_and_filename` + `DEFAULT_NIFTI_BUCKET_PREFIX`) cannot resolve v1_5 CTs at all.
3. Because the feb26 link keys live **only in v1_5**, the old nov25 baseline is unusable as a byte-identity
   "before" for the 3b CT refactor. The reason is **not** a proven pixel diff (an earlier draft asserted
   "nov25 vs feb26 load different bytes" as fact — it isn't): `nov25` is deleted, so whether the reprocess
   re-rendered voxels (new spacing/orientation) or losslessly re-exported them is **unverifiable** — no nov25
   bytes and no banked nov25 pixel-hash survive to compare. Independently, the CT **series-selection** logic
   (which series to include per study) has **changed since v1.1** (Phil, 2026-07-07), so feb26 for a given
   person may legitimately load a *different series* than v1.1 did. Either way D1 sidesteps it: both golden
   sides run on feb26, isolating the 3b refactor from the substrate/selection move. See *Reprocessing &
   historical CT test-set visibility* below.

## Goal

Restore the 3b CT-dissolution byte-identity net **on the go-forward substrate**, and repoint CT resolution
off the dead `nov25` heuristic onto the v1_5/feb26 materialized dataset link — so a future
re-materialization (feb26 → …) is a *data/config* change, not a code edit. Fold in the substrate cut
(v1_1 → v1_5) that the feb26 link forces, and the LUMIA `numeric_value` gate-3 declared delta.

## Decisions (resolved 2026-07-07 via AskUserQuestion)

- **D1 — CT golden net: rebuild on feb26.** Make the *legacy* `__getitem__` CT path resolve feb26 via the
  v1_5 study/series link **first** (a small pre-refactor edit), bank a "before" golden on feb26, run 3b,
  diff byte-identically. Full Gate 1 (imaging/selection/assembly) + Gate 2 (per-model windowing hash) net,
  on the substrate the eval will actually use. (Rejected: dropping the CT diff, and hunting a nov25 archive.)
- **D2 — cut to v1_5 now.** Flip BQ dataset + cohort table + task JSONs to v1_5, re-bank the EHR `no_image`
  baseline on the v1_5 cohort, and treat **v1_1 → v1_5 as an expected results change, NOT byte-gated**.
  The feb26 link keys only exist in v1_5, so this is a prerequisite of D1, not a parallel choice.
- **D3 — `numeric_value`: declared delta now, decide after the first gate-3 diff.** Keep `numeric_value`
  as the OQ-K declared delta with the minimal allowlist normalizer; let the first real 3b gate-3 golden diff
  surface the actual legacy-`VALUE:` vs LUMIA-`NOTE:` divergence, **then** decide accept-vs-parse.
  - **Grounding (repo search, 2026-07-07):** the repo has **no** text→number parser. Every VALUE line is
    built from a *discrete* field — `get_llm_event_string` (`meds_timeline_utils.py:225`),
    `format_events.py:103`, `test_meds_tools.py:43` — sourced (legacy path) from the DB/MEDS query, never
    parsed from text. The LUMIA `.xml` is generated **upstream** (no generator in this repo), so the
    number's element-text format is defined outside this codebase. "Parse numeric from element text" =
    a brand-new speculative parser against an upstream format → correctly deferred.
  - **Fallback path (if the diff shows a material, unrecoverable VALUE-line delta):** prefer fixing it
    **upstream** (have the LUMIA generator emit a discrete `numeric_value` attr — cross-repo, out of scope
    here → OQ-N) over an in-repo text parser. In-repo parsing is the last resort.

## The two axes (keep them separate — this is the crux)

- **Refactor axis (3b):** legacy `__getitem__` CT branch → CT adapter. **Byte-identical**, both sides on
  **feb26**. This is what the golden gates.
- **Substrate axis (v1_1 → v1_5/feb26):** expected to change cohort + image bytes. **NOT byte-gated** —
  validated by row-count sanity + the re-banked EHR baseline, not a before/after diff.

Holding the substrate *fixed on feb26* across the 3b refactor is what makes the byte-identity diff
meaningful again.

> **Rung 0 (reproduce Ryan on feb26) lives in its own plan** —
> [`vlm-rung0-reproduce-ryan-feb26.md`](vlm-rung0-reproduce-ryan-feb26.md). It introduces the
> `ct_snapshot_prefix` config seam (default feb26) + a force-GCS toggle, keeps Ryan's v1_1 substrate otherwise
> untouched, and runs his **weighted** pipeline as an operability smoke. **It must be green before the axes
> below run.** Axes A/B/C here describe **rungs 1–2** (the v1.5 substrate cut + the 3b refactor).

## Approach

### Axis A — CT resolution: `nov25` heuristic → v1_5/feb26 dataset link

- **`src/vqa_dataset.py`** — drop `DEFAULT_NIFTI_BUCKET_PREFIX` (`:15`) and the
  `_nifti_path_to_blob_and_filename` heuristic (`:18-43`). Replace with a deterministic resolver keyed on
  `(image_study_uid, image_series_uid)`: blob = `{ct_snapshot_prefix}/{study}__{series}.nii.gz` in bucket
  `su-vista-uscentral1`, filename `{study}__{series}.nii.gz`. **`ct_snapshot_prefix` is a config value**
  (default `chaudhari_lab/ct_data/ct_scans/vista/feb26`), not a hard-coded constant — this is the
  modular-preprocessing thesis (re-materialization = config/data change). Keep the local-`ct_dir`-first →
  GCS-fallback order unchanged.
- **`__getitem__`** — read `row.get('image_study_uid')` / `row.get('image_series_uid')` instead of
  `row.get('nifti_path')`. `nifti_path` is a deprecated INTEGER in v1_5 → do not use it.
- **`src/data_tools/utils/query_utils.py`** — the four CT queries currently key on `nifti_path`
  (`get_person_id_nifti_paths_query :156`, `get_ct_available_person_ids_query :167`,
  `..._fallback :178`, `fetch_person_id_nifti_paths :189`). Retarget to select
  `image_study_uid, image_series_uid` and guard on `image_series_uid IS NOT NULL` (the "CT available"
  predicate). Return `(person_id, study_uid, series_uid)` tuples.

### Axis B — substrate cut v1_1 → v1_5

- **Dataset constant** — `VISTA_BENCH_DATASET` `vista_bench_v1_1` → `vista_bench_v1_5` at
  `src/data_tools/utils/query_utils.py:236` (**confirmed — the roadmap citation was correct, not stale**).
  The two runtime consumers (`task_data_utils.py:129`, `run_bq.py:261`) reference the constant, so flipping it
  flips both. **Also collapse the 3 stray hard-coded `"vista_bench_v1_1"` default-arg literals** that bypass
  the constant: `query_utils.py:269` (`check_ct_available_batch`, on the CT path), `subsample_csv.py:194`,
  `subsample_csv_from_bq.py:256` (CSV-materialization helpers) — refactor to `= VISTA_BENCH_DATASET` rather
  than flip-in-place; + the `ct_test.py:94` test default. Keep it a **local** constant (do not import from
  vista_bench — see OQ-M).
- **Cohort table** — the v1_5 materialized
  `vista_bench_v1_5.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr` (VM: 374k rows, 249.6k with CT).
- **Task JSONs** — config `valid_tasks: tasks/valid_tasks_v1_3.json` → `tasks/valid_tasks.json` (the v1_5
  bucket has no `_v1_5`/`_v1_3` variant). `image_valid_tasks.json` is **absent** in the bucket but is
  optional/guarded in `TaskOrchestrator.__init__` (VM-confirmed) — leave the guard, do not require the file.
- **Config** — the executor keeps a localized `configs/all_tasks.vm.yaml` pointing at the `su-vista-*`
  mounts (already staged by the VM). The committed `configs/all_tasks.yaml` carries the **GCS `gs://vista_bench/`
  layout** for v1_5 — do **not** commit the `.vm` copy as default.
- **Re-bank the EHR `no_image` baseline on the v1_5 cohort** (the current green baseline is v1_1 — a
  different cohort). This is the new gate-3 "before".

### Axis C — golden rebuild on feb26 (sequence)

1. **C0 (pre-refactor)** — apply Axis A to *today's* legacy `__getitem__` so the **legacy** CT path resolves
   feb26. (Phase 0.5's 30-slice even-spacing fix is already in this branch — `vqa_dataset.py:191-210` — so
   the "before" is banked on the fixed sampler, as the roadmap requires.)
2. **C1 — bank the "before"** — `golden_harness … --experiments axial_all_image --tag legacy_feb26` on the
   v1_5 cohort (+ the optional non-gemma InternVL grayscale run for Gate 2). **Pin the before-cohort
   deterministically** to the committed person set (`figures/data_stats/person_id_subsampled.csv`, the
   canonical PFS smoke) so the "before" is reproducible and its feb26 `(study, series)` UIDs are recoverable
   via the v1_5 BQ join — the harness captures `person_id`/`index`/pixel-hash, **not** file identity, so the
   cohort is the only reproducibility handle. Captures `selected_indices`, `image_hashes`, `image_count` on
   **feb26** images.
3. **C2 — implement 3b** — dissolve the `__getitem__` CT branch into the `src/context/adapters/ct.py`
   adapter; it resolves feb26 identically and exposes the same `selected_indices`.
4. **C3 — bank the "after"** — `--tag adapter_feb26`.
5. **C4 — diff** — `diff_golden.py`: Gate 1 (`selected_indices`/`image_count`/`assembly_mode` byte-identical)
   + Gate 2 (`image_hashes` identical per model). Both sides feb26 → the diff isolates the refactor.

Gate 3 (EHR string) rides the already-defined `no_image` path, re-banked on v1_5 (Axis B).

## Files to modify

Rungs 1–2 (the rung-0 `ct_snapshot_prefix` config seam + force-GCS toggle are introduced by the
[rung-0 plan](vlm-rung0-reproduce-ryan-feb26.md); the changes below build on that seam):
- `src/vqa_dataset.py` — drop `DEFAULT_NIFTI_BUCKET_PREFIX` + `_nifti_path_to_blob_and_filename`; new
  `(study_uid, series_uid) → feb26 blob` resolver (config-driven prefix, from the rung-0 seam); `__getitem__`
  reads the UID cols.
- `src/data_tools/utils/query_utils.py` — retarget the 4 CT queries off `nifti_path` onto
  `image_study_uid`/`image_series_uid`.
- `src/data_tools/utils/query_utils.py:236` — flip `VISTA_BENCH_DATASET` `vista_bench_v1_1` →
  `vista_bench_v1_5`, and collapse the 3 stray `"vista_bench_v1_1"` default-arg literals onto the constant
  (`query_utils.py:269`, `subsample_csv.py:194`, `subsample_csv_from_bq.py:256`; + `ct_test.py:94` test).
- `configs/all_tasks.yaml` — v1_5 GCS layout: dataset, cohort table, `valid_tasks: tasks/valid_tasks.json`,
  `ct_snapshot_prefix: …/vista/feb26`, `subsample: false`.
- `docs/02-ct-scans.md` — replace the `nifti_path`/`nov25` resolution description with the feb26 dataset-link
  model.
- **Roadmap doc** — mark the Phase 0 substrate contract + Back-compat golden sections superseded by this doc
  (nov25-fallback / nifti_path-string / legacy-nov25-baseline are retired).

**No change** to `src/context/adapters/ehr.py` beyond what's banked (the `unit_source_value` remap is already
committed on this branch). `numeric_value` stays the declared delta (D3).

## Open questions — resolved 2026-07-07 (Phil feedback + repo/vista_bench exploration)

- **OQ-M — RESOLVED. Definition site confirmed; keep a *local* constant, do not import from vista_bench.**
  The roadmap citation was **correct, not stale**: `VISTA_BENCH_DATASET = "vista_bench_v1_1"` lives at
  `src/data_tools/utils/query_utils.py:236`, and the two runtime consumers (`task_data_utils.py:129`,
  `run_bq.py:261`) reference it. **But 3 stray `"vista_bench_v1_1"` default-arg literals bypass the constant**
  and must be collapsed onto it: `query_utils.py:269` (`check_ct_available_batch`, on the CT path),
  `subsample_csv.py:194`, `subsample_csv_from_bq.py:256` (+ `ct_test.py:94` test). *Import from vista_bench
  with an optional override?* — **No.** vista_bench installs as `vistabench` but exports nothing usable:
  `src/vistabench/__init__.py` is empty and there is no `v1_5`/version constant. The closest importable symbol
  is `vistabench.config.VISTABENCH_DATASET_ROOT = "som-nero-plevriti-deidbdf.vista_bench_v1_5"` — a
  fully-qualified `project.dataset` path with the version buried mid-string and **no override hook**. Taking a
  new `vistabench` dependency just to parse a version out of that path is more coupling for less clarity, and
  the sibling consumer (vista-eval) already keeps its own local `DEFAULT_VERSION` rather than importing from
  vista_bench. vista_bench's stated convention is placeholder-in-*docs* + real-value-in-*code*
  ([[docs-point-to-canonical-vistabench-ref]]). **Decision:** keep `VISTA_BENCH_DATASET` as this repo's single
  source of truth, flip it to `vista_bench_v1_5`, collapse the 3 stray literals onto it.
- **OQ-N — RESOLVED: not in scope.** The upstream LUMIA-generator `numeric_value` fix is *not* part of this
  plan. If the first gate-3 diff surfaces a material, unrecoverable VALUE-line delta, it becomes a
  **separate** cross-repo ask — the last-resort in-repo parser stays deferred (D3).
- **OQ-P (was "Precondition risk") — RESOLVED via vista_bench release notes: not identical to v1.1, but
  queryable.** `embed_time` is **unchanged** v1_1→v1_5 (same column name/semantics; CHANGELOG: "0 real
  `embed_time` changes" v1.4→v1.5 — anchored to the diagnosis landmark, not the CT), and the cohort table
  `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr` is same-named with **stable membership** (identical
  9,350-person set). So the timeline/`embed_time` surface **is** materialized + queryable in v1_5, and this
  repo reads `embed_time` per-row from `ctx` (`ehr.py:195`, `ehr_filters.py:73`) — safe across the cut.
  **What differs vs v1.1:** the CT-adjacent columns were renamed at v1.3 (carried into v1.5) —
  `_accession_number` → `ct_accession_number`, `latest_img_date` → `ct_image_date`. Axis A already retargets
  CT resolution onto `image_study_uid`/`image_series_uid`, so the golden path is clear — **but the legacy
  names survive in off-golden-path tooling** (`data_tools/csv_helper/format_retrieval_csv.py:25-26`,
  `subsample_csv_from_bq.py`, `data_tools/OMOP_meds_query/get_report_note.py`) and would break if that tooling
  is run against v1_5. Flag for a follow-up; not a blocker here. **Also (Phil, 2026-07-07):** the CT
  **series-selection** (which series to include per study) has changed since v1.1, so the go-forward feb26 CT
  set for a person is *not* assumed identical to v1.1's — characterized by the **soft Step 1b coverage report**
  (report + Phil's input, never a hard stop), not gated. **Residual VM check (keep):** Step 2 still
  confirms the actual cohort *shape* (counts) on v1_5 — release notes retire the "is it queryable" worry, but
  the re-bank is exercised for shape, not assumed.

### Rung-0 open questions — moved

The rung-0 open questions (OQ-R1..R4) now live in the
[rung-0 plan](vlm-rung0-reproduce-ryan-feb26.md#open-questions). All four are resolved there (R1 force-GCS
gate; R2 experiment-name mapping — `axial_all_image ↔ image_and_timeline`, `no_image ↔ timeline_only`, with a
report-inclusion confound; R3 tolerance = report-only ~10% soft; R4 prefixed-`nifti_path` via the config seam +
0a preflight).

## Reprocessing & historical CT test-set visibility (2026-07-07)

Prompted by Phil: `nov25` was deleted *because the CT data was reprocessed into `feb26`*, keyed by the same
`(study_uid, series_uid)`, so the same scans likely still exist at the feb26 path — **do we have visibility
into which scans earlier CT runs tested?** Repo sweep (planner Mac; VM-only items flagged):

- **File/UID-level visibility: mostly NO.** `ct_test.py` records only integer counts (no scan ids). The golden
  harness captures `person_id`/`index`/`image_hashes` — a `sha256` of the *loaded pixels* (`golden_harness.py:149-155`),
  **never** `nifti_path`/UIDs/accession — so no past golden JSONL (even one surviving on the PHI mount) names a
  file. The old `nifti_path` *is* UID-decomposable (`{study}__{series}.nii.gz`, `vqa_dataset.py:37`), but no
  concrete nov25 path values live in the current tree.
- **Cohort-level visibility: YES, recoverable.** Committed `figures/data_stats/person_id_subsampled.csv` +
  `figures/results_stats/all_model_response.csv` pin exactly which `(task, experiment, index, person_id)` ran
  in the image arms. `person_id → (study, series)` is a stable v1_5 BQ join (OQ-P: 9,350-person cohort stable),
  so the historical test set is re-derivable on the VM — this is the reproducibility handle C1 pins to.
- **Reprocessing byte-change is UNVERIFIABLE.** nov25 is deleted and no nov25 pixel-hash was ever banked, so we
  cannot prove whether feb26 voxels differ from nov25 (see supersedes §3). Series-selection has also changed
  since v1.1, so per-person CT continuity is not assumed — the soft Step 1b report characterizes it.
- **⚠ PHI-in-history (separate from the golden work; remediation TBD — Phil to decide).** Real DICOM Study/Series
  UIDs sit in git *history*: `notebooks/inspect_early_stage_management_outputs.ipynb` @ `a638a0a` (Ryan Nayebi,
  Feb 13 2026) ties 3 concrete nov25 paths to `person_id`+`accession`; `src/data_tools/ct_error/ct_filenames.txt`
  @ `089b87a`/`62cb81a` (Ryan DCunha, Feb 5–6 2026) holds 27 CT-*error* UID filenames (rendered today as the
  binary `figures/ct_info/ct_error_scans.pdf`). None are in the current tree. Flagged as a known exposure; **no
  history rewrite without Phil's sign-off.**

## Verification & VM handoff

Executed on the GCP VM (Mac is planner-only). Canonical smoke = `progression_recurrence_free_survival_1_yr`
(PFS) × `gemma3 medgemma-1.5-4b-it`; add `intern OpenGVLab/InternVL3_5-8B-hf` for the Gate 2 grayscale path.
`/vm-handoff` renders this into a runnable `docs/vm-status/<date>-<sha>.md`.

> **Prerequisite — rung 0 first.** The weighted operability smoke (the former "Step 0") now lives in the
> [rung-0 plan](vlm-rung0-reproduce-ryan-feb26.md) and **must be green before Steps 1–5 below run**. The Steps
> below are all the **weights-free golden harness** (rungs 1–2); rung 0 is the only weighted / results-CSV gate.

- **Step 1 — v1_5 CT resolution smoke.** With Axis A + B applied, resolve one v1_5 `(study_uid, series_uid)`
  → feb26 blob and load it.
  **Expected:** the feb26 blob downloads; `image_count > 0`; `selected_indices` = 30 evenly-spaced ints; a
  logged resolver trace shows the blob came from the `(study_uid, series_uid)` path and `nifti_path` (deprecated
  INTEGER) was never read (behavioral, not a code-property assertion). **Stop:** any feb26 404 (prefix/UID
  wrong); a fallback to the dropped `nov25` heuristic; `image_count == 0` across all rows (link/query broken).
- **Step 1b — reprocessing-coverage report (REPORT, do NOT stop — decision gate to Phil).** For the pinned
  C1 person-cohort, resolve each person's v1_5 `(study_uid, series_uid)` and check how many map to an existing
  feb26 blob. **Report (counts only):** cohort size; count resolving to a feb26 blob; count with null/missing
  UIDs; and — if derivable — count whose resolved series differs from what v1.1 would have selected. **Do NOT
  fail or halt on divergence** — CT series-selection has changed since v1.1 (Phil), so a shifted/reduced set is
  *expected*, not a bug. Surface the numbers + ≤5 examples (UIDs-as-structure only, no PHI) and **get Phil's input**
  on whether the go-forward coverage is acceptable before banking C1 (Step 3). This is the "are these the same
  scans we tested?" check, deliberately soft.
- **Step 2 — re-bank the EHR `no_image` baseline on v1_5.** Re-run the banked `no_image` harness on the v1_5
  cohort.
  **Expected:** non-null `adapter_prompt_string`/`dynamic_prompt`; `image_count == 0`;
  `assembly_mode == "ordered"`; `row_count == jsonl lines`, sorted by `(person_id, index)`; golden stays on
  the PHI mount (`git status` clean). **Stop:** the v1_5 cohort timeline/`embed_time` surface is missing or
  differs in shape (OQ-P — queryability confirmed via release notes; *shape* still VM-checked here) → report
  the cohort shape (counts only) and hold.
- **Step 3 — bank the axial "before" on feb26 (C1).** `--experiments axial_all_image --tag legacy_feb26`,
  gemma + intern.
  **Expected:** per (experiment, model) `_legacy_feb26_golden.jsonl` + `.meta.json`, non-empty, sorted;
  `image_count > 0` for CT-bearing rows; matching-length `image_hashes`; weight-free (no GPU/weights in
  logs). **Stop:** any traceback; `row_count == 0` for a CT-bearing cohort; weights loading.
- **Step 4 — 3b refactor diff (C3 + C4).** After 3b, bank `--tag adapter_feb26` and
  `diff_golden.py legacy_feb26 adapter_feb26`.
  **Expected:** **Gate 1** (`selected_indices`/`image_count`/`assembly_mode`) byte-identical; **Gate 2**
  (`image_hashes`) identical per model. **Stop:** any Gate-1/Gate-2 drift — the CT dissolution is not a no-op.
- **Step 5 — gate-3 EHR diff (D3).** Diff the LUMIA-live `no_image` render vs the v1_5-rebanked baseline
  under `--mode allowlist`.
  **Expected:** event/line order identical; deltas confined to the declared `numeric_value` VALUE-line
  allowlist. **Stop:** gate-3 outside allowlist *beyond* the `numeric_value` delta (an undeclared render
  divergence) → record the concrete diff for the accept-vs-parse (D3) decision.

**PHI:** counts / field-names / UIDs-as-structure only — never paste golden rows, timelines, or `.xml`
contents. Golden output is written only to the `su-vista-*` PHI mount (git-ignored); `/phi-vet` gates every
commit.
