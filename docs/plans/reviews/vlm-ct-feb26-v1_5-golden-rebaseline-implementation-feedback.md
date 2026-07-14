Reference: research-skills/claude_ops.md

# Implementation Feedback: VLM rungs 1-2 (Axis A + Axis B) feb26/v1_5 rebaseline

## Verdict

Ready to commit. The implementation matches the requested first chunk: Axis A/B are wired, the CT slice-selection branch remains untouched for the later 3b Mac interlude, and no in-scope critical drift was found.

Caveat: `.claude/references/implementation-review-checklist.md` is absent in this repo, so this review could not apply a repo-local implementation checklist; a repo-grounded checklist would sharpen the review.

## Plan Coverage

| Slice / section | Status | Evidence: path:line | Notes |
|---|---|---|---|
| Axis A: remove `DEFAULT_NIFTI_BUCKET_PREFIX` / `_nifti_path_to_blob_and_filename` | Done | src/vqa_dataset.py:14, src/vqa_dataset.py:21 | The nov25 constant and nifti-path heuristic are gone; the only CT snapshot default is feb26. Grep found no dangling references to either removed symbol in the reviewed scope. |
| Axis A: shared `resolve_ct_blob(study_uid, series_uid, prefix)` contract | Done | src/vqa_dataset.py:21, src/vqa_dataset.py:43, src/vqa_dataset.py:51 | Single helper builds `f"{prefix}/{study}__{series}.nii.gz"` and filename once; no duplicated UID string assembly at call sites in the reviewed diff. |
| Axis A: null/NaN fail-closed resolver behavior | Done | src/vqa_dataset.py:43, src/vqa_dataset.py:45, src/vqa_dataset.py:49 | `None`, pandas/NumPy NaN, and empty strings return `None`, preventing `nan__nan.nii.gz` construction. |
| Axis A: `__getitem__` reads UID columns, not `nifti_path` | Done | src/vqa_dataset.py:170, src/vqa_dataset.py:172 | CT load path uses `image_study_uid` / `image_series_uid` and never reads `nifti_path`; missing columns resolve to no-image through `row.get(..., None)` plus `resolve_ct_blob -> None`. |
| Axis A: preserve local-`ct_dir` first, GCS fallback | Done | src/vqa_dataset.py:177, src/vqa_dataset.py:185, src/vqa_dataset.py:188 | Existing source precedence remains: local file when configured and present, otherwise GCS when a storage client exists. |
| Axis A: keep CT source trace | Done | src/vqa_dataset.py:182 | `[CT] source=... prefix=... blob=... file=...` remains present. |
| Axis A: preserve pre-3b slice-selection branch | Done | src/vqa_dataset.py:207, src/vqa_dataset.py:232, src/vqa_dataset.py:254 | `git diff -U0 -- src/vqa_dataset.py` shows no changes inside the 30/50/10 slice loops. Absence of Axis C / 3b is correct for this chunk. |
| Axis A: retarget CT BQ queries to UID link | Done | src/data_tools/utils/query_utils.py:156, src/data_tools/utils/query_utils.py:169, src/data_tools/utils/query_utils.py:181, src/data_tools/utils/query_utils.py:193 | The CT query helpers select/use `image_study_uid` / `image_series_uid`; `fetch_person_id_nifti_paths` intentionally keeps its old name but now returns `(person_id, study_uid, series_uid)`. |
| Axis A: CT availability predicate moved off `nifti_path` | Done | src/data_tools/utils/query_utils.py:164, src/data_tools/utils/query_utils.py:176, src/data_tools/utils/query_utils.py:188 | Queries guard on `image_series_uid IS NOT NULL` plus non-empty string, matching the plan's stated CT-available predicate. |
| Axis B: `VISTA_BENCH_DATASET` flipped to v1_5 | Done | src/data_tools/utils/query_utils.py:257 | Runtime consumers import this constant. |
| Axis B: hard-coded v1_1 defaults collapsed | Done | src/data_tools/utils/query_utils.py:290, src/data_tools/csv_helper/subsample_csv.py:195, src/data_tools/csv_helper/subsample_csv_from_bq.py:257, src/tests/ct_test.py:94 | The reviewed default args now point at `VISTA_BENCH_DATASET`; no in-scope hard-coded `vista_bench_v1_1` literals remain except quoted historical roadmap text. |
| Axis B: `configs/all_tasks.yaml` cut | Done | configs/all_tasks.yaml:8, configs/all_tasks.yaml:9, configs/all_tasks.yaml:11, configs/all_tasks.yaml:120 | `ct_dir` is unset/commented, `ct_snapshot_prefix` is feb26, `valid_tasks` is `tasks/valid_tasks.json`, and `subsample` is false. |
| Axis B: CT docs rewritten | Done | docs/02-ct-scans.md:7, docs/02-ct-scans.md:22, docs/02-ct-scans.md:59 | Docs describe UID-based feb26 resolution and explicitly mark legacy download/subsample/coverage tooling as deferred. |
| Axis B: roadmap sections superseded | Done | docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:231, docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:377 | Phase 0 CT/substrate and Back-compat CT baseline assumptions are marked superseded by this plan. |
| Golden-path UID flow through `run_bq.py` | Done | src/vista_run/run_bq.py:271, src/vista_run/run_bq.py:943, src/vista_run/run_bq.py:945 | Normal BQ task load selects all columns from v1_5, and `ct_snapshot_prefix` is threaded into `PromptDataset`. |
| Golden harness threads `ct_snapshot_prefix` | Done | src/vista_run/golden_harness.py:204, src/vista_run/golden_harness.py:206 | The harness passes the same config prefix into `PromptDataset`. |

## Critical Drift

- None found in the requested Axis A/B chunk.

## Missing Pieces

- None for the requested chunk.

## Contract Violations

- None found. The schema guard is fail-closed: rows missing `image_study_uid` or `image_series_uid` resolve to `None` and skip CT loading rather than falling back to `nifti_path` (src/vqa_dataset.py:170, src/vqa_dataset.py:172, src/vqa_dataset.py:173).

## Test Gaps

- Low severity | Resolver behavior is not directly unit-tested | Evidence: src/vqa_dataset.py:21, src/vqa_dataset.py:43 | A focused test for valid UID output plus `None`/NaN/empty UID no-image behavior would lock the fail-closed contract. This is a confidence gap, not a commit blocker for this scoped implementation.
- Low severity | Golden-path wiring was inspected statically only | Evidence: src/vista_run/run_bq.py:271, src/vista_run/golden_harness.py:206 | The actual VM smoke still needs to prove v1_5 rows carry UID columns and feb26 blobs load. That is already the plan's Phase 1 execution gate, not a Mac-side implementation miss.

## Defensible Deviations

- Deferred off-golden subsample tools still consume the changed 3-tuple return shape through legacy bucket-check code | Evidence: src/data_tools/csv_helper/subsample_csv.py:151, src/data_tools/csv_helper/subsample_csv.py:155, src/data_tools/csv_helper/subsample_csv_from_bq.py:217, src/data_tools/csv_helper/subsample_csv_from_bq.py:221 | This is explicitly intended by OQ-Q1 for this chunk: only the dataset literal was collapsed onto the constant; CT-consumption migration is deferred.
- `run_bq.py` still has an `all_vb_image_only` full-parquet loader keyed on `nifti_path` | Evidence: src/vista_run/run_bq.py:418, src/vista_run/run_bq.py:445, src/vista_run/run_bq.py:453 | This is outside the requested axial golden path inspected here. The normal task load used by `axial_all_image` goes through the v1_5 `SELECT *` path and into `PromptDataset` with UID columns. Track `all_vb_image_only` with the broader deferred legacy CT tooling if it needs v1_5 support.
- Historical roadmap text still contains `vista_bench_v1_1` / `nifti_path` examples under a superseded block | Evidence: docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:244, docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:249 | The new superseded notice immediately above states the go-forward contract and retirement of these assumptions.

## Suggested Code Edits

- None required before commit for this chunk.

## Questions For The Author

- Should the follow-up off-golden CT tooling migration explicitly include `run_bq.py`'s `all_vb_image_only` full-parquet path, or is that experiment also being retired with the legacy `_full.parquet` generation flow? Evidence: src/vista_run/run_bq.py:418, src/vista_run/run_bq.py:445.

## Audit Trail (files inspected, paths only)

- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- configs/all_tasks.yaml
- docs/02-ct-scans.md
- docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md
- src/data_tools/csv_helper/subsample_csv.py
- src/data_tools/csv_helper/subsample_csv_from_bq.py
- src/data_tools/utils/query_utils.py
- src/tests/ct_test.py
- src/vqa_dataset.py
- src/vista_run/run_bq.py
- src/vista_run/golden_harness.py
