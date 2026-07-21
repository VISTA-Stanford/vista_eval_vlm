Reference: docs/claude_ops.md

# vista_eval_vlm — Step 6: modernize pathology to live `vista_bench_v1_6` substrate

## Context

Pathology inference (`path` / `path_image_and_report` / `path_full` presets) currently reads from
a frozen, hand-generated CSV — `{task}_path_subsampled.csv` under `<base_path>/v1_3/...`
(`resolve_path_subsampled_csv_path`, `src/data_tools/utils/task_data_utils.py:41-45`), produced by
a **one-shot, non-parameterized script** (`subsample_v1_3_path.py`) hardcoded to one engineer's
(`rdcunha`) local VM cache — never re-run since. Meanwhile the rest of the repo has moved on:
`VISTA_BENCH_DATASET = "vista_bench_v1_5"` (`query_utils.py:257`) drives live CT/EHR/text cohort
queries. Pathology is now **two dataset versions behind** (v1_3 vs v1_5), and no doc in this repo
flags that gap — `docs/01-pathology-and-path-tools.md` / `00-data-setup.md` /
`03-vista-bench-data-cohort.md` all present the v1_3 CSV as normal, current behavior.

Meanwhile `vista_bench`'s actual current pathology substrate has moved substantially past even
v1_5: the real slide-linking pipeline (`(note_id, specimen, block)` → `dx_blocks` → diagnostic-slide
pick, stain classification, LLM-extracted histologic labels) only landed feeding
`vista_bench_v1_6.diagnostic_tasks`. Per Phil's explicit call, this plan bumps the whole repo's
`VISTA_BENCH_DATASET` to v1_6 (not a pathology-only decoupled knob) — accepted trade-off: broader
blast radius (CT/EHR/text cohort queries move too), which this plan re-verifies as **Phase 0**,
gated before any pathology work begins.

This mirrors Step 5's shape (frozen CSV → live-sourced adapter) but the failure mode is different:
Step 5 was a byte-diff-alignment problem against an existing legacy baseline. Pathology has **no
legacy baseline to diff against** — it's genuinely new-live-substrate work, so the landing gate is
Phase 2 human visual-QA (reusing the existing config-context viewer), same as Step 5 ultimately
converged on.

## Goal

Replace the frozen v1_3 pathology CSV with pathology cohort rows pulled the **same way CT/EHR/text
already are** — live BQ, task-scoped, via the existing `_load_task_data`/`fetch_task_data_from_bq`
machinery — sourced from the current `vista_bench_v1_6.diagnostic_tasks` table, so pathology
inference reflects real, current slide-linking + labels instead of a 2-versions-stale, one-off hand
export. Re-tile whatever slide delta that surfaces.

## Approach — 4 phases, each gating the next

**Phase 0 — Dataset bump + re-verify existing modalities (prerequisite, not pathology-specific)**
Bump `VISTA_BENCH_DATASET` v1_5→v1_6 (`query_utils.py:257`). Before touching pathology, confirm
this doesn't silently break CT/EHR/text, which are pinned to v1_5-derived golden fixtures
(including Step 5's just-landed live-EHR-adapter work). Cheap-before-expensive: a schema-diff
first, a scoped byte-diff re-bank second.

**Status (2026-07-21):** dataset bump committed (`ac1561a`). Step 0a (schema-diff) ran on the VM
and passed. Step 0b's original byte-diff vehicle turned out to be unrunnable (a pre-existing
`vista_bench` registry issue, not a bump problem) and was retargeted — see the Verification & VM
handoff section below for the corrected vehicle and full detail
(`docs/vm-status/2026-07-21-ac1561a.md`).

**Phase 1 — Reuse the existing task-scoped BQ loader for pathology + coverage/tiling-gap
characterization**
**Revised per Phil's feedback:** no new BQ query function needed. Every other modality resolves
its BQ table purely from `task_source_csv` in `valid_tasks.json` (`WHERE task = @task_name`,
`docs/00-data-setup.md:21`) — pathology should mirror this instead of introducing a separate
deduped manifest table. Since `vista_bench`'s pathology-tagged `diagnostic`-group tasks (`stage`,
`histologic_type` + subtypes, `pdl1_expression`, metastases, the `_llm` suite) are already
materialized into `vista_bench_v1_6.diagnostic_tasks`, pathology tasks' `task_source_csv` almost
certainly already resolves to `diagnostic_tasks` — **VM Step 0 of this phase confirms this against
the live `valid_tasks.json`** (not assumed; that file lives on the VM, not this repo). If
confirmed, `_load_path_task_data` (`run_bq.py:477`) can call `_load_task_data`
(`run_bq.py:250-343`) directly — the exact function CT/EHR/text already use, local-cache-first,
else live BQ — with **no bespoke pathology query code at all**. The task-filtered rows are already
one-per-person-per-task (same as every other modality), so the dedup concern that originally
motivated a separate manifest table doesn't apply once rows are pulled this way. After the live
rows come back, apply the existing `PathologyAdapter.materialize(...)` post-processing (tile-folder
resolution + seeded sampling) unchanged. Characterize the gap this surfaces: how the new
`diagnostic_tasks`-sourced person/slide set compares to the old v1_3 CSV, and — critically — how
many resolved `path_image_path` values have **no existing tile folder** under `test_patch/` (the
old tiles were only ever generated for v1_3-selected slides).

**Phase 2 — VM-side re-tiling for the delta (gated on Phase 1's coverage numbers)**
Reuse existing tooling unchanged: `download_path_subsampled.py` (already a repeatable CLI, just
point it at an exported manifest CSV for the delta slides) → `tile_wsi.py` (already resumable,
MedGemma-spec correct — the roadmap plan calls this "canonical and good", no changes needed).
Machine class depends on delta size, decided from Phase 1's count (see Verification section).

**Phase 3 — Wire one preset, QA all three + docs**
**Per Phil's feedback:** start with `path_full` as the one actively wired/enabled preset in
`configs/all_tasks.yaml` (matching today's production config), but QA-render **all three** —
`path`, `path_image_and_report`, `path_full` — via the existing config-context viewer (same
weight-free QA mechanism Step 5 used), since the loader fix is shared code and all three should be
confirmed working even if only one ships enabled. This is the landing gate, since there's no legacy
pathology baseline to byte-diff against. Update the three stale docs to describe the live-pull flow
instead of the v1_3 CSV flow.

## Files to Modify

- `src/data_tools/utils/query_utils.py` — bump `VISTA_BENCH_DATASET` (line 257) v1_5→v1_6. **No new
  query function** (revised — see Phase 1): pathology reuses `fetch_task_data_from_bq` as-is.
- `src/data_tools/utils/task_data_utils.py` — **no change expected** (revised — see Phase 1):
  pathology reuses the existing `resolve_local_bq_cache_path` (`:12-14`), same cache path CT/EHR/
  text already use, once `_load_path_task_data` routes through `_load_task_data`. Only touch this
  file if VM Step 0 finds pathology's `task_source_csv` does NOT already resolve to
  `diagnostic_tasks` and a fallback path is needed.
- `src/vista_run/run_bq.py` — rewrite `_load_path_task_data` (~477-548) to call `_load_task_data`
  directly (mirroring CT/EHR/text), then apply the existing `PathologyAdapter.materialize(...)` on
  the returned rows; `_load_path_full_task_data` (~550) needs no change, it wraps the above.
- `src/data_tools/path_tools/download_path_subsampled.py` — minor: point its manifest-CSV input at
  the new BQ-exported manifest (already has `--v1-3-base`/`--download-dir`/`--dry-run` args,
  `:226-258` — likely just a default/doc change, not new code).
- `src/data_tools/path_tools/tile_wsi.py` — unchanged, reused as-is.
- `configs/all_tasks.yaml` — confirm/update `paths.path_tile_base` (line 10) if tile output
  location changes for the delta slides.
- `docs/01-pathology-and-path-tools.md`, `docs/00-data-setup.md`, `docs/03-vista-bench-data-cohort.md`
  — update to describe the live-pull flow + v1_6 (the mermaid diagram in the third doc currently
  shows pathology as the only modality *not* fed by the live `BQ` node — that changes).
- New plan doc: `docs/plans/vlm-step6-pathology-live-substrate.md` (this plan).

## Open Questions

- **Per-slide table choice — RESOLVED (Phil, 2026-07-21): use `diagnostic_tasks`.** "Already linked
  to tasks; should mirror what the radiology is doing... shared logic here almost certainly." This
  is the bigger decision it looks like — see the rewritten Phase 1 / Files to Modify above:
  pathology reuses the existing `_load_task_data`/`fetch_task_data_from_bq` machinery with **zero
  new BQ query code**, sourced from `diagnostic_tasks` (task-filtered rows are already
  one-per-person-per-task, so the dedup concern that motivated the original manifest-table default
  doesn't apply). VM Step 0 of Phase 1 confirms pathology's `task_source_csv` registry entry
  actually resolves to `diagnostic_tasks` before relying on it.
- **Preset scope — RESOLVED (Phil): "let's start with one but test all."** `path_full` stays the
  one actively wired/enabled preset in `configs/all_tasks.yaml` (matching today's production
  config); Phase 3's QA render explicitly covers all three presets (`path`,
  `path_image_and_report`, `path_full`) even though only one ships enabled.
- **Known vista_bench-side caveats this plan inherits, doesn't fix — ACKNOWLEDGED (Phil:
  "understood").** Naming in the updated docs so this isn't rediscovered later: WSI coverage only
  from ~2022+ (~770/~9,350 persons); stain classifier only ~2/3 coverage (unknown-stain fallback,
  graceful); the leak-safe removal-gated window is dormant by default — production pathology is a
  leakage-tolerant `[dx−180, dx+180]` window by design, not something to second-guess here;
  readmission task has zero pathology linkage; radiation task's pathology columns are all-NULL
  (predates 2022 WSI); `is_metastatic_llm` is registered but **not materialized** — won't appear in
  `diagnostic_tasks`, don't expect it in the pathology-fed presets; only ~2,748 of ~747,603
  pathology-titled notes are WSI-linked, so `path_note_text` coverage is a small fraction of total
  pathology report volume.
- **PHI project boundary — CORRECTED (Phil, 2026-07-21): "approved. note this table IS deid
  though."** The original framing overstated the PHI exposure here. `vista_bench_v1_6.diagnostic_tasks`
  (like `vista_bench_v1_5`/`v1_6` generally) lives in the **already-de-identified**
  `som-nero-plevriti-deidbdf` project — the same project + credential scope CT/EHR/text already
  query today. The raw-PHI sourcing (`OMOP_CDM_ROOT_PHI`/`som-rit-phi-oncology-prod`,
  `_whole_slide_imaging_alpha`) happens **upstream, inside `vista_bench`'s own producer pipeline**,
  materializing into the de-identified table — that's out of scope for this plan entirely. This
  plan's live query touches no new PHI boundary and needs no new credential access beyond what
  already works for CT/EHR/text. (Downstream WSI-tile *pixel* storage/access via `path_image_path`
  is unchanged from the existing pipeline either way — not a new consideration this plan
  introduces.)

## Verification & VM handoff

**Target machine:** Claude-Code CPU (`phil-sllm-01`) for Phases 0, 1, 3 and Phase 2's readback.
Phase 2's bulk download+tile step may route to high-throughput CPU (`phil-hcpu`) if the delta is
large — sized from Phase 1's count (decision gate below); if small, stays on Claude-Code CPU.

**Phase 0 — dataset bump + re-verify (not destructive; ordered cheap → expensive)**
- Step 0a (cheap): BQ schema-diff query comparing `vista_bench_v1_5` vs `vista_bench_v1_6` column
  lists for the tables CT/EHR/text loaders actually read.
  **Expected:** no non-additive changes (nothing renamed/removed) among CT/EHR/text-relevant
  columns — pathology columns are additive/new, fine to differ.
  **Stop:** a breaking schema change on a column CT/EHR/text actually reads — report back, do not
  proceed to 0b.
  **Status (2026-07-21):** already run + confirmed clean on the VM
  (`docs/vm-status/2026-07-21-ac1561a.md`) — of the 4 distinct `task_source_csv` tables the 49
  registry tasks reference, only 1 exists in BQ at all, and it changed purely additively (32→38
  cols). Not re-run; this verdict stands. Caveat for any future re-run: a table **absent from
  both** dataset versions also reports `removed/renamed: none` (two empty column sets look
  "clean") — treat an absent-in-both table as `ABSENT (not compared)`, not a pass.
- Step 0b (moderate, only if 0a clean): re-run `diff_golden.py --mode strict` against a small
  representative re-bank sourced from v1_6 (not the full cohort — scope per `vm-smoke-scope-limit`
  convention), banked to a **new** path (not overwriting the existing v1_5 golden bank in place).
  **Verification vehicle: `progression_recurrence_free_survival_1_yr`, `no_image` +
  `axial_all_image`** — already the exact task `configs/all_tasks.rung0.yaml` declares, with both
  experiments enabled, so it needs no new config. **(Re-plan 2026-07-21):** the original vehicle,
  `has_recurrence_1_yr`, was found on the VM to have a `task_source_csv` registry entry
  (`..._5yr_v1_1`) absent from **both** `vista_bench_v1_5` and `vista_bench_v1_6` — a pre-existing
  `vista_bench` task-registry↔BQ naming mismatch (48 of the 49 registry tasks reference nonexistent
  `_v1_*`-suffixed table names in both dataset versions; tracked as a backlog item in
  `docs/next.md`, out of scope to fix here). `progression_recurrence_free_survival_1_yr` is the one
  task whose non-suffixed source table (`progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`)
  actually resolves in both dataset versions — see `docs/vm-status/2026-07-21-ac1561a.md` for the
  full finding.
  **Expected:** byte-identity holds for CT/EHR/text content.
  **Stop:** unexpected divergence in CT/EHR/text content — report back, do not proceed to Phase 1.
- **Decision gate:** Phase 0 clean → proceed to Phase 1. Phase 0 red → STOP, re-plan; pathology
  work does not proceed on top of an unverified dataset bump.

**Phase 1 — reuse existing loader + coverage characterization**
- Step 1a (cheap, structural): confirm a pathology task's `task_source_csv` entry in the live
  `valid_tasks.json` resolves to `diagnostic_tasks`.
  **Expected:** confirmed as assumed.
  **Stop:** it resolves to something else — do not silently adapt; report back with the actual
  value so the Files-to-Modify list (a fallback cache-path resolver) can be revisited.
- Step 1b: exercise `_load_task_data` for one pathology task through the now-rewritten
  `_load_path_task_data` (structural: non-empty df, `path_image_path` populated) + a coverage
  report: old-CSV persons vs new `diagnostic_tasks`-sourced persons (added/dropped), and a count of
  resolved `path_image_path` values with no existing tile folder under `test_patch/`.
  **Expected:** loader succeeds (same de-identified project/credentials CT/EHR/text already use —
  see the corrected PHI Open Question); report produced with concrete numbers.
  **Stop:** BQ access failure — unexpected, since this reuses an already-working credential path;
  report back if it happens.
- **Decision gate (pre-encoded, resolves inline — no round-trip needed):** if the no-local-tile
  count is small (<~10% of resolved slides) → Phase 2 is a quick top-up, stays on Claude-Code CPU.
  If large (most slides are new) → Phase 2 is a real bulk job, route to `phil-hcpu`, readback on
  `phil-sllm-01`.

**Phase 2 — re-tile the delta (additive only, not destructive)**
- `download_path_subsampled.py` against the exported delta manifest → `tile_wsi.py`.
  **Expected:** new tile folders appear under `test_patch/<folder>/` for delta slides; no errors.
  **Stop:** systemic (not isolated-row) GCS/OpenSlide failures.

**Phase 3 — wire one, QA all three (landing gate)**
- Config-context viewer render for **all three** presets (`path`, `path_image_and_report`,
  `path_full`) on a small N — same mechanism and PHI-crossing discipline as Step 5's Phase 2:
  report file paths / existence / card counts / exit codes only, never rendered content.
  `path`/`path_image_and_report` may need a temporary config declaration to render (the viewer
  validates `--experiment` against the config's declared list) even though only `path_full` ships
  enabled in the landed config. **Phil reads all three HTML files himself** — that's the landing
  gate, same as Step 5.
  **Expected:** all three HTMLs render clean, self-contained, non-empty tiles/prompt per card, no
  STOP/traceback.
  **Stop:** same criteria pattern as Step 5's Phase 2, for any of the three.

## Landing & cleanup

- **Branch:** `feat/vlm-step6-pathology-live-substrate` (feature branch — touches shared cohort
  infra, not a doc-only/minor fix).
- **Landing gate:** Phases 0–3 all VM-green + Phil's manual QA read of the pathology viewer HTML.
- **Merge sequence:** single-branch plan → `/land` at the end (no sibling branches currently touch
  these files, per the earlier Step-5 `/land` sibling scan).
- **Cleanup on land:** prune branch + worktree; mark this plan `Completed`; update
  `docs/plans/README.md` + `docs/next.md`; the three doc updates already listed above land in the
  same branch, not as a follow-up.
