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

Replace the frozen v1_3 pathology CSV with a live BigQuery-sourced pathology slide manifest
(mirroring the CT adapter's live-BQ-with-local-cache pattern), on the current `vista_bench_v1_6`
substrate, so pathology inference reflects real, current slide-linking + labels instead of a
2-versions-stale, one-off hand export — and re-tile whatever slide delta that surfaces.

## Approach — 4 phases, each gating the next

**Phase 0 — Dataset bump + re-verify existing modalities (prerequisite, not pathology-specific)**
Bump `VISTA_BENCH_DATASET` v1_5→v1_6 (`query_utils.py:257`). Before touching pathology, confirm
this doesn't silently break CT/EHR/text, which are pinned to v1_5-derived golden fixtures
(including Step 5's just-landed live-EHR-adapter work). Cheap-before-expensive: a schema-diff
first, a scoped byte-diff re-bank second.

**Phase 1 — Live pathology manifest query + coverage/tiling-gap characterization**
Add a new BQ query path for pathology (there isn't one today — `subsample_v1_3_path.py` never
queries BQ at all) targeting `vista_bench_v1_6.diagnostic_wsi_slides_for_embedding` — the deduped,
one-row-per-slide manifest (`path_image_path` + `(accession, specimen, block, slide)` identity),
not the long-format `diagnostic_tasks` (would need in-repo dedup). Rewrite
`_load_path_task_data` (`run_bq.py:477`) to mirror `_load_task_data`'s pattern: local-cache-first,
else live BQ (`run_bq.py:250-343`, `fetch_task_data_from_bq`). Characterize the gap this surfaces:
how the new manifest's person/slide set compares to the old v1_3 CSV, and — critically — how many
resolved `path_image_path` values have **no existing tile folder** under `test_patch/` (the old
tiles were only ever generated for v1_3-selected slides).

**Phase 2 — VM-side re-tiling for the delta (gated on Phase 1's coverage numbers)**
Reuse existing tooling unchanged: `download_path_subsampled.py` (already a repeatable CLI, just
point it at an exported manifest CSV for the delta slides) → `tile_wsi.py` (already resumable,
MedGemma-spec correct — the roadmap plan calls this "canonical and good", no changes needed).
Machine class depends on delta size, decided from Phase 1's count (see Verification section).

**Phase 3 — Wire + human-QA + docs**
Confirm the presets render correctly via the existing config-context viewer (same weight-free QA
mechanism Step 5 used) — this is the landing gate, since there's no legacy pathology baseline to
byte-diff against. Update the three stale docs to describe the live-pull flow instead of the v1_3
CSV flow.

## Files to Modify

- `src/data_tools/utils/query_utils.py` — bump `VISTA_BENCH_DATASET` (line 257) v1_5→v1_6; add a
  new `fetch_pathology_manifest_from_bq(...)` mirroring `fetch_task_data_from_bq`, querying
  `{VISTA_BENCH_DATASET}.diagnostic_wsi_slides_for_embedding`.
- `src/data_tools/utils/task_data_utils.py` — add a cache-path resolver for the pathology manifest
  (mirrors `resolve_local_bq_cache_path`, `:12-14`), replacing `resolve_path_subsampled_csv_path`'s
  hardcoded `v1_3` path as the sole source.
- `src/vista_run/run_bq.py` — rewrite `_load_path_task_data` (~477-548) to the cache-then-live
  pattern; `_load_path_full_task_data` (~550) needs no change, it wraps the above.
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

- **Per-slide table choice** — defaulting to `diagnostic_wsi_slides_for_embedding` (clean,
  deduped) over `diagnostic_tasks` (long-format, would need dedup logic here). Flag, not fully
  locked — cheap to revisit at VM time if the manifest table is missing a needed column.
- **Preset scope** — only `path_full` is actually enabled in `configs/all_tasks.yaml` today (`path`
  / `path_image_and_report` are commented out, `:86-87`). The loader fix is automatic for all
  three (shared code path); Phase 3's QA render covers `path_full` at minimum. Re-enabling the
  other two in config is optional, decide after Phase 3's QA looks clean.
- **Known vista_bench-side caveats this plan inherits, doesn't fix** (worth naming in the updated
  docs so this isn't rediscovered later): WSI coverage only from ~2022+ (~770/~9,350 persons);
  stain classifier only ~2/3 coverage (unknown-stain fallback, graceful); the leak-safe
  removal-gated window is dormant by default — production pathology is a leakage-tolerant
  `[dx−180, dx+180]` window by design, not something to second-guess here; readmission task has
  zero pathology linkage; radiation task's pathology columns are all-NULL (predates 2022 WSI);
  `is_metastatic_llm` is registered but **not materialized** — won't appear in `diagnostic_tasks`,
  don't expect it in the pathology-fed presets; only ~2,748 of ~747,603 pathology-titled notes are
  WSI-linked, so `path_note_text` coverage is a small fraction of total pathology report volume.
- **PHI project boundary** — all pathology sourcing reads `OMOP_CDM_ROOT_PHI`
  (`som-rit-phi-oncology-prod`); no de-identified WSI source exists yet (a separate, unmerged
  `vista_bench` probe confirmed the de-id release's WSI table is a non-bridging 50-row stub). The
  old frozen CSV's data originally came from this same PHI-scoped source via a one-off hand export,
  so credential access should already exist — confirmed as a VM Step 0 precondition, not assumed.

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
- Step 0b (moderate, only if 0a clean): re-run `diff_golden.py --mode strict` against a small
  representative re-bank sourced from v1_6 (not the full cohort — scope per `vm-smoke-scope-limit`
  convention), banked to a **new** path (not overwriting the existing v1_5 golden bank in place).
  **Expected:** byte-identity holds for CT/EHR/text content.
  **Stop:** unexpected divergence in CT/EHR/text content — report back, do not proceed to Phase 1.
- **Decision gate:** Phase 0 clean → proceed to Phase 1. Phase 0 red → STOP, re-plan; pathology
  work does not proceed on top of an unverified dataset bump.

**Phase 1 — live manifest + coverage characterization**
- Exercise the new loader for one pathology task (structural: non-empty df, `path_image_path`
  populated) + a coverage report: old-CSV persons vs new-manifest persons (added/dropped), and a
  count of resolved `path_image_path` values with no existing tile folder under `test_patch/`.
  **Expected:** loader succeeds; report produced with concrete numbers.
  **Stop:** BQ access/credential failure on the PHI-scoped project — surfaces the Open Questions
  precondition; report back.
- **Decision gate (pre-encoded, resolves inline — no round-trip needed):** if the no-local-tile
  count is small (<~10% of resolved slides) → Phase 2 is a quick top-up, stays on Claude-Code CPU.
  If large (most slides are new) → Phase 2 is a real bulk job, route to `phil-hcpu`, readback on
  `phil-sllm-01`.

**Phase 2 — re-tile the delta (additive only, not destructive)**
- `download_path_subsampled.py` against the exported delta manifest → `tile_wsi.py`.
  **Expected:** new tile folders appear under `test_patch/<folder>/` for delta slides; no errors.
  **Stop:** systemic (not isolated-row) GCS/OpenSlide failures.

**Phase 3 — wire + human-QA (landing gate)**
- Config-context viewer render for `path_full` (+ `path` if re-enabled) on a small N — same
  mechanism and PHI-crossing discipline as Step 5's Phase 2: report file paths / existence / card
  counts / exit codes only, never rendered content. **Phil reads the HTML himself** — that's the
  landing gate, same as Step 5.
  **Expected:** HTML renders clean, self-contained, non-empty tiles/prompt per card, no
  STOP/traceback.
  **Stop:** same criteria pattern as Step 5's Phase 2.

## Landing & cleanup

- **Branch:** `feat/vlm-step6-pathology-live-substrate` (feature branch — touches shared cohort
  infra + PHI-scoped query code, not a doc-only/minor fix).
- **Landing gate:** Phases 0–3 all VM-green + Phil's manual QA read of the pathology viewer HTML.
- **Merge sequence:** single-branch plan → `/land` at the end (no sibling branches currently touch
  these files, per the earlier Step-5 `/land` sibling scan).
- **Cleanup on land:** prune branch + worktree; mark this plan `Completed`; update
  `docs/plans/README.md` + `docs/next.md`; the three doc updates already listed above land in the
  same branch, not as a follow-up.
