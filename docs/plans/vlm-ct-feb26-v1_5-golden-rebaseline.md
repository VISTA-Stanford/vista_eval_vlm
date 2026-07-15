Reference: research-skills/claude_ops.md

# VLM CT resolution → v1_5/feb26 dataset link + golden rebaseline on feb26

**Status: Completed — landed on `main` 2026-07-15** (3b CT-dissolution proven a byte-identity no-op, VM-green; Step-5 gate-3 EHR golden deferred to the LUMIA-live follow-up). — supersedes the CT/substrate slice of
[`vlm-modular-preprocessing-and-context-viewer-roadmap.md`](vlm-modular-preprocessing-and-context-viewer-roadmap.md).
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` (continues; not yet committed).
**Machine posture:** authored on the planner Mac (no runtime/data/creds). VM-executed steps below run on the
Claude-Code CPU executor `phil-sllm-01` (holds the `su-vista-*` mounts + BQ creds for project
`som-nero-plevriti-deidbdf`; runs Claude Code → readback here directly). *(`som-nero-plevriti-deidbdf` is the BQ
project, not a VM — the earlier draft used it as a hostname.)*
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

> **✅ Prerequisite gate — MET.** The rung-0 operability smoke
> ([`vlm-rung0-reproduce-ryan-feb26.md`](vlm-rung0-reproduce-ryan-feb26.md), + its decoding-fix amendment
> [`vlm-rung0-0b-decoding-fix.md`](vlm-rung0-0b-decoding-fix.md)) **landed green on 2026-07-13**
> (`docs/vm-status/2026-07-13-rung0-0b-decoding-fix.md`: constrained decoding both arms, `predicted_label==-1
> total: 0`; Ryan accuracy crosscheck in-band). Rung 0 introduced the `ct_snapshot_prefix` config seam that
> Axis A builds on and proved the feb26 CT reroute works end-to-end. Rung-0 implementation/review is **out of
> scope here** (see *Out of scope* below) — this doc consumes its seam, not its code.

## Deviation & re-plan (2026-07-14) — v1_5 timeline base staging

**Step-0 STOP fired** on the first VM run of Doc 1
([`docs/vm-status/2026-07-13-rungs1-2-v1_5-feb26-prebank.md`](../vm-status/2026-07-13-rungs1-2-v1_5-feb26-prebank.md),
readback `5b63eeb` on `phil-sllm-01`): the Axis A/B **code** verified clean (all four seam greps hit; the
`resolve_ct_blob` fail-closed contract passes weight-free), but every data-touching step (1b/2/3) blocked because
**no v1_5 local EHR-timeline base exists on the mount** — only the v1_1 base. `run_bq._load_task_data` reads
labels/UIDs from BQ `vista_bench_v1_5.<source_csv>` (present) but then **inner-joins on `person_id` a local timeline
CSV `base_dir/<source_csv>/<task>.csv` carrying `patient_string`** (`run_bq.py:288-317`); flipping the substrate to
v1_5 repointed that lookup at a v1_5-named dir that isn't staged → returns `None`.

**Finding (Mac investigation):** the `<task>.csv` `patient_string` is a **MEDS-rendered** timeline over
`[embed_time − 6mo, embed_time]` (`remove_imaging_report.py:process_dataframe`; described at
`subsampled_retrieval_csv.py:8,12`) — **not** in BQ and **not** the LUMIA `.xml`. No committed script produces the bare
`<task>.csv`: the real producers emit *suffixed* `_subsampled.csv` / `_full.parquet` and need a `meds_reader` DB +
ontology we can't confirm on the VM. The v1_1 base was `gsutil rsync`'d from `gs://vista_bench/` and is exactly what
the rung-0 `timeline_only` arm read (0.515). The LUMIA `.xml` corpus
(`gs://vista_bench/thoracic_cohort_lumia/{person_id}.xml`, static / version-invariant) is a **second render of the same
MEDS timeline** (`ehr.py:12,141` — "LUMIA is generated from the MEDS timeline… reproduces today's `patient_string` by
construction"), consumed only by the still-deferred modular EHR adapter.

**Decision (Phil, 2026-07-14):** the `patient_string` CSVs are a **MEDS artifact, stable v1_1→v1_5**, so **copy the
v1_1 timeline CSV into the unsuffixed v1_5 cohort dir and continue** (Phase 0 / new Step 0 in *Verification & VM
handoff*). **Do not** produce a fresh v1_5 base or dig for Ryan's source CSVs — we own this now. The **LUMIA-direct
loader** (read the static `.xml` via `ehr.py` + a per-`(person, look-back window, snapshot)` render cache, retiring the
pre-baked-CSV dependency) is the better endgame but is **deferred to a separate follow-up plan** — see *Follow-ups
spun out*.

## Deviation & re-plan (2026-07-15) — C3 intern leg limited to a spot-check (I/O routing resolved)

**Partial VM run of Doc 2**
([`docs/vm-status/2026-07-15-3b-ct-adapter-golden-diff.md`](../vm-status/2026-07-15-3b-ct-adapter-golden-diff.md),
readback `f1fd8a8` on `phil-sllm-01`): Step 0 green, and **gemma C3 (`adapter_feb26`) banked full and GREEN** —
1,238 rows, CT-bearing split 943/295 == C1, and **byte-size identical to C1's gemma `legacy_feb26` golden**
(289,997,040 B — a strong preliminary no-op signal). The VM then **handed back on a routing finding, not a
defect**: the weight-free bank is pure mount-latency (~85 min at 0 % CPU, ~13 CTs/min over 943 CT-bearing rows),
so the *full* intern leg would either tie up the interactive Claude-Code box for another ~85 min or force a
relocation to a high-throughput CPU box — which would break this plan's "Claude-Code CPU throughout, no
standalone runner script, no run/readback split" posture (see *Handoff phasing*).

**Decision (Phil, 2026-07-15): limit the intern leg to a spot-check.** The intern C3 leg re-reads *the same 943
feb26 CT volumes gemma already read*; the only thing that differs from gemma is the **windowing branch applied
after load** (`grayscale` vs gemma's `multi_window_rgb`). That branch is (a) Codex-verified as a **verbatim
relocation** into `CTAdapter` (selector `evenly_spaced_k` verbatim, float64 cast preserved — implementation
review), and (b) already validated on the full 943-volume cohort *for gemma*. A full second 85-min cohort read to
prove the grayscale half is disproportionate. Instead:
- **gemma stays FULL** (already banked) — the full-cohort byte-identity proof on the primary model.
- **intern becomes a LIMITED spot-check** — `golden_harness … --limit N` on the deterministic
  `(person_id, index)` head, diffed strict against a matching `head -n N` slice of the full intern `legacy_feb26`
  before (required because `diff_golden` **fails closed on any index-set inequality** —
  `diff_golden.py:125-126,156-158`). A CT-bearing-row floor keeps the grayscale branch genuinely exercised.

**Net effect:** the intern leg drops from ~85 min to a few minutes, so **all remaining work stays on
`phil-sllm-01`** (Claude-Code native readback) — the hcpu relocation / runner-script / run-readback split is **not
needed** and the plan's single-box posture holds. This is a deliberate **rigor trade** (full intern parity →
gemma-full + Codex-static verbatim proof + an intern spot-check), owned by Phil, not a silent relaxation. Step 4
is revised accordingly; its original full-parity criteria are superseded **for intern only** (gemma unchanged).

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
  - **Extract it as one shared helper — do not inline the string assembly at each call site.**
    `resolve_ct_blob(study_uid, series_uid, prefix) -> (blob_path, filename)` (module-level in
    `vqa_dataset.py`, or a small `data_tools/utils/ct_utils.py` fn if that reads cleaner). **Contract:**
    inputs = the two UID strings + the prefix (defaulted from `ct_snapshot_prefix`); output =
    `(f"{prefix}/{study}__{series}.nii.gz", f"{study}__{series}.nii.gz")`; **null behavior** = if either UID is
    null/empty, return `None` (→ the caller treats the row as no-image, same as a missing CT today — see the
    fail-closed schema guard in Step 1); **logging** = the per-row `[CT] source=local|gcs prefix= blob= file=`
    trace (from the rung-0 seam) stays. Realistic reuse (not a speculative seam): `__getitem__` (load path), the
    **Step 1b coverage report** (resolve → bucket-existence check), and any future snapshot bump all call the
    *same* helper — no duplicated `{study}__{series}` assembly to drift.
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
  mounts. The committed `configs/all_tasks.yaml` carries the **GCS `gs://vista_bench/`
  layout** for v1_5 — do **not** commit the `.vm` copy as default. ⚠ **The v1_5 local timeline base is *not*
  auto-staged** (2026-07-14 deviation — the roadmap's "already staged by the VM" assumption was false): the loader's
  `base_dir/<source_csv>/<task>.csv` must be staged first — **Step 0 (Phase 0)** copies the v1_1 CSV into the
  **unsuffixed** v1_5 cohort dir (MEDS-stable) and flips the staged `valid_tasks.json` `task_source_csv` to the
  unsuffixed v1_5 table.
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
   adapter. **Respect the existing adapter boundary — do not turn `CTAdapter` into a storage resolver.**
   `CTAdapter`'s contract (its own docstring, `ct.py:9-11`) is: "NIfTI *loading* (GCP/local resolution) stays
   in the caller (`vqa_dataset`); this adapter operates on a **loaded volume**." So 3b moves only the
   **slice-selection + windowing** branch into `CTAdapter.contextualize` (which already does exactly that via
   `resolve_ct_selector` + `multi_window_rgb`/`grayscale`); the `(study,series)→blob` resolution (the Axis A
   `resolve_ct_blob` helper) and the NIfTI load stay **outside** the adapter, called identically on both golden
   sides. Byte-identity is therefore gated on selection/windowing, with the resolver/loader held constant — the
   adapter exposes the same `selected_indices`/`image_hashes` because it receives the same loaded volume.
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
  `vista_bench_v1_5`. The two runtime consumers ride the constant: `src/data_tools/utils/task_data_utils.py:129`
  and `src/vista_run/run_bq.py:271` (**not `:261`** — line drift; verify at edit time). Then collapse the 3 stray
  `"vista_bench_v1_1"` default-arg literals onto the constant, **full paths:** `src/data_tools/utils/query_utils.py:269`
  (`check_ct_available_batch`), `src/data_tools/csv_helper/subsample_csv.py:194`,
  `src/data_tools/csv_helper/subsample_csv_from_bq.py:256`; + the `src/tests/ct_test.py:94` test default.
- `configs/all_tasks.yaml` — v1_5 GCS layout: dataset, cohort table, `valid_tasks: tasks/valid_tasks.json`,
  `ct_snapshot_prefix: …/vista/feb26`, `subsample: false`.
- `docs/02-ct-scans.md` — replace the `nifti_path`/`nov25` resolution description with the feb26 dataset-link
  model.
- **Roadmap doc** — mark the Phase 0 substrate contract + Back-compat golden sections superseded by this doc
  (nov25-fallback / nifti_path-string / legacy-nov25-baseline are retired).

**No change** to `src/context/adapters/ehr.py` beyond what's banked (the `unit_source_value` remap is already
committed on this branch). `numeric_value` stays the declared delta (D3).

**Not touched here:** `src/vista_run/run.py:168` (an unthreaded `PromptDataset` caller) and ~10 other
off-golden-path `nifti_path`/`nov25` tooling files are deferred — enumerated + STOP-gated in *Out of scope*.

## Open questions — resolved 2026-07-07 (Phil feedback + repo/vista_bench exploration)

- **OQ-M — RESOLVED. Definition site confirmed; keep a *local* constant, do not import from vista_bench.**
  The roadmap citation was **correct, not stale**: `VISTA_BENCH_DATASET = "vista_bench_v1_1"` lives at
  `src/data_tools/utils/query_utils.py:236`, and the two runtime consumers
  (`src/data_tools/utils/task_data_utils.py:129`, `src/vista_run/run_bq.py:271`) reference it. **But 3 stray `"vista_bench_v1_1"` default-arg literals bypass the constant**
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

### Resolved 2026-07-13 (Codex-review grounding + repo sweep; Phil delegated the calls)

The 2026-07-13 Codex review surfaced these five; a repo sweep (all 13 off-golden-path files characterized) +
the rung-0 vm-status facts + the `/vm-handoff` skill made them decidable. Phil delegated the calls.

- **OQ-Q1 — legacy off-golden-path CT/CSV tooling → DEFER (the whole batch).** Sweep verdict: **0 of the 13
  files are on the golden path** (none imported by `run_bq.py`/`golden_harness.py`/`vqa_dataset.py`/`query_utils.py`,
  none import-reachable); all are standalone `__main__` scripts (CT-download / cohort-subsample / retrieval-prep /
  post-hoc PDF) + the shared `ct_utils.py` helper. Break modes on v1_5 are almost all **silent-empty / silent-wrong**
  (empty cohort, 0-coverage), not crashes. The subsample/download/coverage trio share `ct_utils`'s `nov25`-blob
  helper *and* the in-scope `query_utils` CT queries, so they must migrate **together, reusing the new
  `resolve_ct_blob`**, in a follow-up once Axis A lands — migrating now would duplicate the retarget. Keep the STOP
  guard. Follow-up should add a **loud "nov25 gone" guard** so a manual v1_5 run errors instead of silently emptying.
- **OQ-Q2 — `src/vista_run/run.py` → `run_bq.py`-only; `run.py` legacy/out-of-scope.** `run.py` is the CSV-driven
  **predecessor** to `run_bq.py` (defines its *own* duplicate `TaskOrchestrator` nothing imports; golden/BQ/rung-0
  all use `run_bq.py`). It has no `nifti_path`/`nov25` of its own — un-threaded `PromptDataset` (`:168`) silently
  drops CT on v1_5. **Do NOT thread it in lockstep** (effort on a dead path). It *is* still wired to
  `eval/gcp.sh:97,99`, so it's not fully orphaned → **follow-up: retire `run.py` + `eval/gcp.sh` together** (out of
  scope here; don't expand this plan to a deletion).
- **OQ-Q3 — Step 1b threshold → executor REPORTS the coverage and runs it by Phil, who decides in flight on the VM
  (Phil, 2026-07-13); ~87% is his reference prior, not an auto-gate.** rung-0 measured 86.9% (2,053/2,362) via the
  *crude v1_1 nov25-key reconstruction*; Axis A queries the **native v1_5 UID link**, which should recover **≥** that.
  The golden gate only compares CT rows present on *both* feb26 sides, so coverage <100% doesn't break Gate 1/2 — the
  smoke needs ≥1 CT-bearing row, not statistical power. **Rule:** the executor surfaces the number and **Phil makes the
  accept/STOP call in flight** (he's present on the VM). Reference prior: **at/above ~87% = healthy** (native link at
  least as good as the reconstruction); **materially below (~<80%) = suspect the v1_5 link/query**, not series-selection
  drift. Link-health check, not an eval bar.
- **OQ-Q4 — Step 1b readback → aggregate metrics only in the committed doc; detailed results to the GCP-mounted
  bucket (Phil, 2026-07-13).** The committed handoff carries **aggregate/counts metrics only** — no raw Study/Series
  UIDs. The **detailed per-row coverage output** (including any UID-level breakdown) is **written to the GCP-mounted
  `su-vista-*` bucket** (the PHI mount, git-ignored), where results belong — not the git tree. Grounding: raw UIDs add
  PHI risk with **~zero decision value** — the historical CT test-set UIDs were never banked (golden captures
  pixel-hash, not UIDs), so there's nothing to compare a raw UID against; the cohort-level `person_id` join is the only
  handle. Matches the rung-0 precedent + the golden-output-on-PHI-mount design.
- **OQ-Q5 — handoff rendering → TWO docs (the "superseding chain" framing was the wrong mechanism).** `/vm-handoff`
  renders **one doc per handoff *session*** and treats **every machine switch as a phase boundary**; the Mac 3b
  interlude *is* a session boundary. "Supersede" = *replace a stale/blocked* doc, not continue after planned work —
  continuations use the **`Prior handoffs:`** field. So: **Doc 1** = Phases 1+2 (Steps 1/1b/2/3), pre-3b; **Doc 2** =
  Phase 3 (Steps 4/5), authored *after* the 3b commit (post-3b SHA), `Prior handoffs: [Doc 1]`, C1 banked-by-SHA.

**Follow-ups spun out (out of scope — track separately):** (1) migrate the 13-file off-golden-path tooling batch onto
`resolve_ct_blob` + the v1_5 UID queries, with a loud `nov25`-gone guard; (2) retire `run.py` + `eval/gcp.sh` together;
(3) **LUMIA-direct loader (2026-07-14 deviation endgame)** — point the timeline source at the static LUMIA `.xml` via
the `ehr.py` adapter, wire the deferred window/code `select` chain (`presets.py:31-44`), and add a per-`(person,
look-back window, snapshot)` render cache — retiring the pre-baked-`patient_string`-CSV dependency (and dissolving the
gate-3 legacy-vs-LUMIA diff: LUMIA *becomes* the baseline). Validate against the banked legacy reference (`timeline_only`
≈ 0.515; `ehr.py:12` claims byte-repro of `patient_string`, now checkable against the copied v1_1 CSV).

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

- **Step 0 — stage the v1_5 timeline base (the 2026-07-14 deviation fix; run before Step 1).** The v1_5 golden steps
  need a local `base_dir/<v1_5 source_csv>/<task>.csv` with `person_id` + `patient_string`; only the v1_1 base is on the
  mount. Because `patient_string` is a MEDS render and MEDS is stable v1_1→v1_5 (Phil), **copy, don't reproduce**:
  1. **Copy the timeline CSV** from the v1_1 dir to the **unsuffixed** v1_5 cohort dir (create it first):
     `mkdir -p <base_dir>/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr && cp -n
     <base_dir>/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1/progression_recurrence_free_survival_1_yr.csv
     <base_dir>/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/` (copy the whole `…_v1_1/` dir if staging all
     tasks, not just PFS). **Copy the timeline dir only — NOT any `bigquery_data_2_3/<source_csv>` cache** — so the
     v1_5 labels/UIDs are queried **live** from the v1_5 BQ table (`phil-sllm-01` has creds), never served stale from a
     v1_1 cache.
  2. **Flip the staged `valid_tasks.json`** so the PFS entry's `task_source_csv` is the **unsuffixed**
     `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr` — it drives *both* the BQ table (`vista_bench_v1_5.<that>`,
     which exists) *and* the local dir the copy just created. This clears the second latent mismatch the readback
     flagged (`…_v1_1` source × `vista_bench_v1_5` dataset → a table that does not exist).

  **Expected:** `<base_dir>/<unsuffixed>/progression_recurrence_free_survival_1_yr.csv` exists, non-empty, has a
  `patient_string` column; `valid_tasks.json` PFS `task_source_csv` = the unsuffixed name.
  **Timeline-coverage guard (instrument, don't trust — distinct from Step 1b's CT-blob coverage):** on the Step-2
  (and Step-1b) load, the loader prints `Matched N out of M rows with patient timelines` (`run_bq.py:318`). **Record N
  and M.** N≈M (~100%) confirms Phil's MEDS-stability assumption — the v1_1 person set covers the v1_5 PFS slice.
  **Materially N<M** ⇒ the v1_5 PFS cohort grew a person set the copied v1_1 CSV doesn't cover, and those persons
  **silently inner-join-drop** → **report the gap** (class-2 decision gate, same as Step 1b; *not* an auto-stop — the
  golden smoke only needs ≥1 CT-bearing row — but Phil records accept/revise before Step 3 banks C1).
  **Stop:** the unsuffixed BQ table doesn't resolve (creds / wrong name); the copied CSV lacks a `patient_string`
  column; `Matched 0 out of M` (person-id keyspace mismatch — the v1_1 CSV and v1_5 BQ don't share `person_id` space).
- **Step 1 — v1_5 CT resolution smoke.** With Axis A + B applied, resolve one v1_5 `(study_uid, series_uid)`
  → feb26 blob and load it.
  **Expected:** the feb26 blob downloads; `image_count > 0`; `selected_indices` = 30 evenly-spaced ints; the
  per-row `[CT] source=… prefix=… blob=… file=…` resolver trace (grep this exact tag — it is the rung-0 seam's
  log) shows `source=gcs`, the blob built from the `(study_uid, series_uid)` path, and that `nifti_path`
  (deprecated INTEGER) was never read (behavioral, not a code-property assertion). **Stop:** any feb26 404
  (prefix/UID wrong); any `[CT]` line showing a fall-through to the dropped `nov25` heuristic; `image_count == 0`
  across all rows (link/query broken).
  - **Accept/reject schema guard (add to the resolver — verify here).** A row **with** `image_series_uid` must
    resolve to a feb26 blob; a row **missing** `image_study_uid`/`image_series_uid` must **fail closed to a
    known no-image result** (`resolve_ct_blob → None`, `image_count == 0`), **not** silently reach for
    `nifti_path`. Exercise one of each and confirm the reject case is the deliberate no-image path.
- **Step 1b — reprocessing-coverage report (soft — never fails on divergence; a human decision gate before
  Step 3).** For the pinned C1 person-cohort, resolve each person's v1_5 `(study_uid, series_uid)` via
  `resolve_ct_blob` and check how many map to an existing feb26 blob. **Report — aggregate metrics only in the
  committed doc** (OQ-Q4; detailed per-row output → the GCP-mounted `su-vista-*` bucket, git-ignored): cohort size;
  count resolving to a feb26 blob; count with null/missing UIDs; and — if derivable — count whose resolved series
  differs from what v1.1 would have selected. **Control flow:** Steps 1, 1b, 2 complete in the
  same VM doc; **coverage divergence is never a failure** (CT series-selection has changed since v1.1 — Phil —
  so a shifted/reduced set is *expected*); but **Step 3 must not bank C1 until Phil records an accept/revise
  decision.** Pre-encoded routing: **accept → proceed to Step 3** (bank on the go-forward set); **not acceptable
  → STOP + re-plan** (the smoke cohort isn't representative); **null/missing UIDs beyond the shape Phil agrees
  to → class-2 deviation** (record, route back). *Floor (OQ-Q3, resolved): the executor **reports the coverage and
  Phil decides in flight on the VM**; ~87% (≈ rung-0's 86.9%) = healthy reference prior, materially below (~<80%) =
  suspect the v1_5 link/query (not series drift). Link-health check, not an eval bar (the golden gate only compares
  CT rows present on both feb26 sides).*
  This is the deliberately-soft "are these the same scans we tested?" gate.
- **Step 2 — re-bank the EHR `no_image` baseline on v1_5.** Re-run the banked `no_image` harness on the v1_5
  cohort.
  **Expected:** non-null `adapter_prompt_string`/`dynamic_prompt`; `image_count == 0`;
  `assembly_mode == "ordered"`; `.meta.json` `row_count == jsonl line count`, sorted by `(person_id, index)`;
  golden stays on the PHI mount (`git status` clean). **Report the cohort shape as explicit units:** total rows,
  unique `person_id`, CT-bearing UID rows, timeline-non-null rows, task-filtered PFS count. **Stop:**
  `row_count == 0`; `.meta.json` `row_count` ≠ jsonl lines; a missing / zero-byte output file; a dirty repo
  containing golden output (leaked off the PHI mount); or the v1_5 timeline/`embed_time` surface missing /
  shape-shifted (OQ-P — queryability confirmed via release notes; *shape* still VM-checked) → report shape
  (counts only) and hold.
- **Step 3 — bank the axial "before" on feb26 (C1).** Two explicit harness invocations, both
  `--experiments axial_all_image --tag legacy_feb26`: `--model-type gemma3 --model google/medgemma-1.5-4b-it`
  and `--model-type intern --model OpenGVLab/InternVL3_5-8B-hf` (the Gate-2 grayscale path).
  **Expected:** per (experiment, model) `_legacy_feb26_golden.jsonl` + `.meta.json`, non-empty, sorted;
  `image_count > 0` for CT-bearing rows; matching-length `image_hashes`; weight-free (no GPU/weights in logs).
  **Stop:** any traceback; `row_count == 0` for a CT-bearing cohort; a missing / zero-byte `.jsonl`/`.meta.json`;
  index set not unique-and-sorted; an unexpected retrieval-skip / "no data" message (the harness skips retrieval
  experiments *by design* — for `axial_all_image` a skip or zero-row output is a **STOP**, not a benign note);
  weights loading.
- **Step 4 — 3b refactor diff (C3 + C4).** After 3b, bank `--tag adapter_feb26` and diff strict per model.
  **Asymmetric by design (2026-07-15 deviation): gemma full, intern a limited spot-check** — see the deviation
  note above for the rationale. `strict` is the `diff_golden.py` default — state it anyway so the executor
  doesn't rely on the default. Both models weight-free on `phil-sllm-01`; no hcpu / runner-script.
  - **gemma (full — already banked at `f1fd8a8`).** Bank done: `golden_harness … --type gemma3 --name
    google/medgemma-1.5-4b-it --experiments axial_all_image --tag adapter_feb26` → 1,238 rows, split 943/295 ==
    C1. Diff `python -m vista_run.diff_golden <gemma legacy_feb26>.jsonl <gemma adapter_feb26>.jsonl --mode
    strict`.
    **Expected:** shared-index set identical (all 1,238); **Gate 1** structure byte-identical over the tool's full
    hard set — `image_hashes`, `selected_indices`, `image_count`, `assembly_mode`, `path_tile_count` (`None` for
    CT rows) — and **Gate 2** `image_hashes` identical; `RESULT: ALL GATES PASS`.
  - **intern (limited spot-check).** Bank `golden_harness … --type intern --name OpenGVLab/InternVL3_5-8B-hf
    --experiments axial_all_image --tag adapter_feb26 --limit 100`. `--limit` takes a deterministic
    `(person_id, index)` head, so the after's N rows match the head of the full before. Build the paired
    before-slice (needed because `diff_golden` fails closed on any index-set inequality) and diff:
    ```bash
    cd src
    BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
    AFTER=$(find "$BASE" -path '*InternVL*' -name '*_axial_all_image_adapter_feb26_golden.jsonl')
    BEFORE_FULL="${AFTER/_adapter_feb26_/_legacy_feb26_}"              # full 1,238-row C1 intern before
    N=$(wc -l < "$AFTER")                                             # actual limited-after row count
    BEFORE_LTD="${BEFORE_FULL/_golden.jsonl/_limit${N}_golden.jsonl}"  # keeps _golden.jsonl → gitignore backstop
    head -n "$N" "$BEFORE_FULL" > "$BEFORE_LTD"                       # first N by (person_id,index) == after's head
    python -m vista_run.diff_golden "$BEFORE_LTD" "$AFTER" --mode strict
    ```
    **Expected:** the limited intern golden contains **≥ 30 CT-bearing rows** (`image_count > 0`) so the
    `grayscale` windowing branch is genuinely exercised on real volumes — **report the sample's CT-bearing /
    no-image split**; shared-index set identical over the N rows (the `head -n N` slice guarantees it); Gate 1 +
    Gate 2 byte-identical; `RESULT: ALL GATES PASS`.
    **If the sample has < 30 CT-bearing rows** (the head landed CT-thin), re-bank with a larger `--limit`
    (e.g. 200) and re-slice — a few extra minutes, still single-box.
  - **Stop (both models):** any Gate-1/Gate-2 drift (the CT dissolution is not a no-op → hard class-3 halt, back
    to the Mac); an **index-set mismatch** (for intern this means the `head -n N` slice didn't pair — re-derive N
    from the after's `wc -l` and re-slice) or a **`[WARN] no shared indices`** (an empty intersection cannot
    *prove* a no-op — treat as failure, not pass). **Two `ALL GATES PASS` (gemma full + intern spot-check) → 3b is
    a proven no-op; the CT dissolution is retired → `/land`.**
- **Step 5 — gate-3 EHR diff (D3).** Diff the LUMIA-live `no_image` render vs the v1_5-rebanked baseline
  under `diff_golden.py … --mode allowlist`.
  **Expected:** event/line order identical; deltas confined to the declared `numeric_value` VALUE-line
  allowlist. **Stop:** gate-3 outside allowlist *beyond* the `numeric_value` delta (an undeclared render
  divergence) → hand back for the accept-vs-parse (D3) decision. **PHI-clean readback contract:** record
  **counts by field, affected indices, and ≤5 field-name/index examples + a scrubbed characterization only** —
  **do NOT paste `diff_golden.py`'s BEFORE/AFTER output** into the handoff: its `_preview` prints raw
  `dynamic_prompt`/`adapter_prompt_string` values (`src/vista_run/diff_golden.py:169-170,185-186`), which carry
  timeline PHI.

**PHI:** counts / field-names / affected-indices only — never paste golden rows, timelines, `.xml` contents, or
`diff_golden` BEFORE/AFTER previews. **Raw DICOM Study/Series UIDs are identifiers → they stay on the
`su-vista-*` PHI mount and do not enter the handoff doc** (reconciles with the canonical PHI-clean rule,
`verification-and-handoff-design.md:85-89`); Step 1b's examples are **hashed/ordinal structural** stand-ins, not
raw UIDs, unless `/phi-vet` explicitly clears specific UID values for Phil's adjudication. Golden output is
written only to the `su-vista-*` PHI mount (git-ignored); `/phi-vet` gates every commit.

### Handoff phasing

Complex handoff (class-2 human decision gate at 1b + bank-before/diff-after goldens + a **Mac implementation
interlude** between banking the "before" and the "after"). Per claude_ops.md, decompose into phases.
**Executor class throughout = Claude-Code CPU** on the GCP VM (`som-nero-plevriti-deidbdf`): every step is the
**weights-free** golden harness, so there is **no GPU / high-throughput runner, no standalone runner script, and
no run-vs-readback split** (rung 0 was the only weighted/GPU gate). `/vm-handoff` renders each phase.

- **Phase 1 — v1_5/feb26 pre-bank characterization.** Steps 1, 1b, 2. Purpose: prove Axis A/B read v1_5/feb26
  and re-bank the EHR baseline. Banked-from-prior: rung-0 green (prerequisite only; not re-run). Gate: Step 1b
  reports coverage → **Phil accepts/revises before C1**. Destructive: no (golden on PHI mount only). Stop:
  Steps 1/2 hard-stop; Step 1b divergence *pauses* before Step 3, not a fail. Next-doc trigger: Phil accepts.
- **Phase 2 — C1 before-golden bank.** Step 3. Banked-from-prior: Phase-1 Step-2 `no_image` baseline *iff*
  SHA/config unchanged (else un-bank + rerun). Gate: Step 3 Expected/Stop. Destructive: no. Stop: no C1 on
  zero rows / weights load / missing output+meta / dirty repo / unexpected skip. Next-doc trigger: green →
  hand back to **Mac** for C2 (3b) implementation.
- **Phase 3 — C3/C4 after the Mac implements 3b.** Step 4 (Step 5 gate-3 deferred to the LUMIA-live follow-up —
  the runtime still renders the passthrough `patient_string`, so `--mode strict` is the applicable gate here).
  **Split (2026-07-15 deviation): gemma C3 full — banked `f1fd8a8`, GREEN; intern C3 a limited spot-check** (the
  full-cohort read proved I/O-bound at ~85 min → limited rather than relocated to hcpu, preserving the
  Claude-Code-CPU-throughout / no-runner-script posture above — see the 2026-07-15 deviation note). Executor
  class unchanged = Claude-Code CPU (`phil-sllm-01`), no run/readback split. Banked-from-prior: Phase-2 C1
  before-golden + gemma C3 (both frozen; the intern before is `head`-sliced from C1, not re-banked). Gate: Step 4
  Expected/Stop. Destructive: no. Stop: any Gate-1/2 strict drift = hard class-3 halt; **two `ALL GATES PASS`
  (gemma full + intern spot-check) → retire the CT dissolution → `/land`.**

**Mac-interlude / banked-by-SHA rule (the reason to phase):** Step 3 banks "before", then 3b is implemented
**on the Mac** (the SHA moves), then Step 4 banks "after" — so C1 and C3/C4 are at *different SHAs by
construction*. Render as **two vm-status docs** (Phases 1+2 in one; Phase 3 in a superseding doc authored after
the 3b commit), or one superseding chain that records Steps 1/1b/2/3 as **banked-by-SHA**. Either way: if any
code/config/data input moves after a banked phase, **un-bank and rerun** the affected golden — a banked "before"
is only valid while its inputs are frozen. (Phase-2-own-doc vs superseding-chain is a `/vm-handoff` rendering
choice — see Open questions.)

## Out of scope

A fresh implementer should treat everything below as explicitly excluded from rungs 1–2:
- **Rung-0 implementation & review** — landed green 2026-07-13; this doc consumes its `ct_snapshot_prefix` seam,
  not its code.
- **Weighted / results-CSV evals** — rung 0 owned that gate; Steps 1–5 here are weights-free golden harness.
- **Upstream LUMIA-generator `numeric_value` fix** (OQ-N) — a separate cross-repo ask, only if a material
  gate-3 VALUE-line delta surfaces (D3).
- **PHI-in-history remediation** — real DICOM UIDs sit in git *history* (see *Reprocessing & historical CT
  test-set visibility*). **No history rewrite without Phil's sign-off.**
- **Off-golden-path legacy CT/CSV/report tooling still keyed on `nifti_path`/`nov25`** — **deferred, not
  silently broken.** Axis A retargets the golden + `run_bq` path; these files are *not* on it and would break
  only if run against v1_5: `src/vista_run/run.py:168` (unthreaded `PromptDataset`),
  `src/data_tools/csv_helper/{format_retrieval_csv.py, subsample_v1_2.py, all_ct_csv.py}`,
  `src/data_tools/utils/ct_utils.py`, `src/data_tools/full_dataset_utils/ct_coverage.py`,
  `src/data_tools/ct_info/{ct_pdf_ex.py, download_subsampled_ct.py}`,
  `src/results/plot_code/{check_image_usage.py, generate_pdf.py}`, `src/retrieval/subsample_retrieval_csv.py`.
  **STOP guard:** the rungs-1–2 golden/`run_bq` path must not call any of these — if 3b or the config cut pulls
  one in, that's a scope breach → re-plan. Migrate-vs-defer is a Phil scope call (Open questions); **default =
  defer** + a follow-up doc.

## Landing & cleanup

- **Branch:** `worktree-vlm-modular-preprocessing-roadmap` (continues — rung 0 already landed on it; rungs 1–2
  build on the same branch).
- **Landing gate:** rung-0 green ✅ (2026-07-13); rungs 1–2 VM handoff green (Steps 1–5 Expected met, Gate-1/2
  strict `ALL GATES PASS`, gate-3 within the declared allowlist or D3 resolved); `/review-implementation` clean;
  `/phi-vet` sign-off (medical repo — golden touches CT + EHR); `/review-plan` sign-off (this pass).
- **Merge sequence:** single branch → `main` via `/land` (ff-only or PR) after the gate. No sibling branch must
  land first (rung 0 is already in this branch's history). Lands independently of the parent roadmap branch — it
  only marks roadmap sections *superseded* (a doc edit), no code overlap.
- **Cleanup on land:** `/land` Phase 4 — prune the branch (local + remote) + the
  `.claude/worktrees/vlm-modular-preprocessing-roadmap` worktree; set this plan `Status: Completed`; update the
  `docs/plans/README.md` row; prune the `docs/next.md` rungs-1–2 pointer. Retire the rung-0-specific
  `configs/all_tasks.rung0*.yaml` overlays if nothing else references them.
