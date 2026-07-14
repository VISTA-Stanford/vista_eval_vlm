Reference: docs/claude_ops.md

# Feedback: VLM CT feb26/v1_5 golden rebaseline (rungs 1-2) (Codex review)

## Verdict
Revise. The core rung-1/2 direction is sound and the `ct_snapshot_prefix` seam already exists, but the plan is not yet handoff-ready: it lacks required out-of-scope and landing sections, misses or underspecifies several legacy CT/tooling surfaces, and needs explicit VM handoff phasing around the Step 1b human decision gate.

## Critical Gaps
- Severity: Critical | Gap | Missing `## Landing & cleanup` section | Why it matters | `claude_ops.md` requires the branch, landing gate, merge sequence, and cleanup so `/land` is following a reviewed plan rather than inventing it at the end. The plan names the branch but never states how `worktree-vlm-modular-preprocessing-roadmap` reaches `main`, how rung-0 is ordered relative to this branch, or how plan/README/next/worktree cleanup happens. | Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:7 and docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:248 | Required fix | Add `## Landing & cleanup` with branch, landing gate (`rung 0 green`, rungs 1-2 VM handoff green, PHI-vet, review sign-off), merge sequence, and cleanup of branch/worktree/tracker/plan status.
- Severity: High | Gap | Missing explicit `## Out of scope` section | Why it matters | The plan contains scattered exclusions, but a fresh agent needs one authoritative scope guard for non-rung-1/2 work, off-golden legacy tooling, PHI history remediation, upstream LUMIA `numeric_value`, weighted/results-CSV gates, and rung 0. Without it, implementation can drift into tooling migrations or cross-repo fixes. | Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:14, docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:194, docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:241 | Required fix | Add `## Out of scope` that explicitly excludes rung 0 implementation/review, weighted evals, upstream LUMIA changes, PHI history rewrite, and legacy CSV/download/report tooling unless converted to follow-up docs.
- Severity: High | Gap | Verification has a human decision gate but no explicit handoff phasing | Why it matters | The canonical spec says complex handoffs with decision gates need phases and bank/un-bank rules. Step 1b requires Phil's input before Step 3, so a single unphased Steps 1-5 sequence leaves `/vm-handoff` unclear whether to stop and append, continue, or create a follow-up handoff. | Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:264 | Required fix | Add `Handoff phasing`: Phase 1 implementation + Steps 1/1b/2; gate Phil acceptance of coverage before banking CT; Phase 2 bank `legacy_feb26`; Phase 3 3b + `adapter_feb26` + diffs. State banked artifacts and SHA invalidation rules.

## Failure Modes
- Scenario | v1_5 CT subsampling/materialization helpers still filter on `nifti_path`/nov25 | Why the plan misses it | Axis A lists the four query helpers, but `subsample_csv.py`, `subsample_csv_from_bq.py`, `ct_utils.py`, and `ct_test.py` still combine `nifti_path` queries with a nov25 prefix. The plan mentions two default-arg literals but not the coupled GCS-prefix and path-pair logic. | What to add | Either include these files in scope with a UID-pair bucket-existence contract, or put them in `Out of scope` plus a follow-up and a STOP that the golden path does not call them.
- Scenario | `src/vista_run/run.py` remains an unthreaded `PromptDataset` caller | Why the plan misses it | The seam is verified in `run_bq.py` and `golden_harness.py`, but `run.py` instantiates `PromptDataset` without config, storage client, or `ct_snapshot_prefix`. If this path is still runnable for CT, it will rely on defaults and cannot honor a future snapshot config. | What to add | State whether `run.py` is obsolete/out-of-scope or update it in lockstep.
- Scenario | The CT adapter refactor forks loading/resolution logic | Why the plan misses it | Axis C says the adapter “resolves feb26 identically,” but the current adapter docstring says NIfTI loading stays in the caller and it operates on a loaded volume. | What to add | Specify the exact reuse boundary: one shared resolver/loader helper used by both legacy pre-bank and post-3b path, with `CTAdapter` only selecting/windowing loaded volumes, or explicitly change the adapter contract.
- Scenario | Step 1b reports PHI-like UID examples | Why the plan misses it | The plan says “UIDs-as-structure only” and “≤5 examples,” but DICOM Study/Series UIDs are treated as PHI-sensitive elsewhere in the same plan history discussion. | What to add | Report counts and structural categories only, or explicitly require `/phi-vet` approval before any UID examples leave the VM.

## Contract Checks
- In-repo CT loader contract: plan correctly identifies `DEFAULT_NIFTI_BUCKET_PREFIX` and `_nifti_path_to_blob_and_filename`; current code still resolves `nifti_path` through local `ct_dir` first, then GCS. Evidence: src/vqa_dataset.py:15, src/vqa_dataset.py:24, src/vqa_dataset.py:164, src/vqa_dataset.py:171.
- In-repo `ct_snapshot_prefix` seam: present and threaded through `PromptDataset`, `run_bq.py`, and `golden_harness.py`; Axis A can build on it. Evidence: src/vqa_dataset.py:21, src/vqa_dataset.py:52, src/vista_run/run_bq.py:943, src/vista_run/golden_harness.py:205.
- In-repo CT query contract: the four named helpers currently key on `nifti_path`, matching the plan. Evidence: src/data_tools/utils/query_utils.py:156, src/data_tools/utils/query_utils.py:167, src/data_tools/utils/query_utils.py:178, src/data_tools/utils/query_utils.py:189.
- In-repo dataset-version contract: `VISTA_BENCH_DATASET = "vista_bench_v1_1"` and runtime consumers import the constant. The plan's consumer paths are slightly imprecise; use `src/data_tools/utils/task_data_utils.py`, not `src/data_tools/task_data_utils.py`. Evidence: src/data_tools/utils/query_utils.py:236, src/data_tools/utils/task_data_utils.py:129, src/vista_run/run_bq.py:271.
- In-repo hard-coded literal contract: the plan correctly names `query_utils.py:269`, `subsample_csv.py:194`, `subsample_csv_from_bq.py:256`, and `ct_test.py:94`, but it should use full paths under `src/data_tools/csv_helper/` and `src/tests/`. Evidence: src/data_tools/csv_helper/subsample_csv.py:194, src/data_tools/csv_helper/subsample_csv_from_bq.py:256, src/tests/ct_test.py:94.
- In-repo golden reproducibility contract: the harness captures `person_id`, `index`, `selected_indices`, and SHA256 image hashes, not `nifti_path` or UIDs, so C1's cohort-pinning argument is valid. Evidence: src/vista_run/golden_harness.py:149, src/vista_run/golden_harness.py:217.
- Sibling repo `vista_bench` contract: version pin is `vista_bench_v1_5`; CT table is `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`; CT link columns are `image_study_uid` and `image_series_uid`; task registry is `tasks/valid_tasks.json`; renamed columns are `_accession_number` to `ct_accession_number` and `latest_img_date` to `ct_image_date`. The plan names these, but the facts are VM/changelog assumptions, not code-verifiable here. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:124, docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:126, docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:203.
- Sibling repo `vista-ct` contract: feb26 storage layout is `chaudhari_lab/ct_data/ct_scans/vista/feb26/{study}__{series}.nii.gz`; nov25 deletion, v1_5 `nifti_path` integer deprecation, 9,350-person stable cohort, and unchanged `embed_time` are external assumptions requiring VM/changelog validation. The plan labels most of these, but Step 1/2 should restate which are code-verifiable vs VM-validated.

## Modularity vs. YAGNI
- Decision point | `ct_snapshot_prefix` as config value | Plan's current choice | Keep config seam from rung 0, default feb26. | Modular alternative + the realistic use case | This is already the modular alternative; realistic use case is a future CT re-materialization after feb26 where the same `(study, series)` link convention points at a new snapshot prefix. | Recommendation, OR "raise to user" | Keep it. This is justified, not speculative, because rung 0 already needed a prefix seam and the plan's stated substrate design depends on re-materialization being config/data, not code.
- Decision point | Resolver keyed on `(study_uid, series_uid)` | Plan's current choice | New deterministic resolver in `vqa_dataset.py`. | Modular alternative + the realistic use case | Extract a shared `resolve_ct_blob(study_uid, series_uid, prefix)` helper reused by `PromptDataset`, VM preflight, and any bucket-existence tooling. Future snapshots and Step 1b coverage checks need the same behavior. | Recommendation, OR "raise to user" | Add the shared helper contract to the plan; do not duplicate string assembly in several scripts.
- Decision point | Axis C adapter reuse | Plan's current choice | “Dissolve” legacy branch into `src/context/adapters/ct.py`, but says the adapter resolves feb26. | Modular alternative + the realistic use case | Reuse `CTAdapter` for selection/windowing only and share loader/resolver outside it, matching the current adapter contract. | Recommendation, OR "raise to user" | Revise the wording so implementation does not fork loader logic or mutate the adapter into a storage resolver by accident.

## Verification Gaps
- Step 1 should include a concrete command or script name for the “nifti_path was never read” behavioral check. A log assertion alone is fragile unless the resolver trace is named and grep-able. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:258.
- Step 1b is a well-motivated soft report, but it must state exact decision outcomes: if Phil accepts coverage, proceed to Step 3; if not, stop and re-plan; if UID nulls exceed an agreed shape, classify as deviation. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:264.
- Step 2 should include the exact cohort-shape units to report: rows, unique `person_id`, CT-bearing UID rows, timeline non-null, and task-filtered PFS count. The current “differs in shape” STOP is too broad to execute consistently. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:276.
- Step 3 says “gemma + intern” but should spell out the two harness invocations or exact model type/name pairs so the executor does not infer from prose. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:279.
- No negative/schema guard verifies v1_5 rejects legacy assumptions: e.g. `nifti_path` absent/non-string is tolerated only because UID columns are present; missing `image_series_uid` should fail closed or produce a known no-image result. Add accept and reject cases per the canonical schema/contract archetype.
- PHI readback contract should forbid raw Study/Series UID examples unless explicitly vetted; counts and category labels are sufficient for Step 1b.

## Handoff Readiness
- File paths and line ranges are mostly useful, but fix drift: `task_data_utils.py`, `subsample_csv.py`, `subsample_csv_from_bq.py`, and `ct_test.py` need full repo paths. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:118 and docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:120.
- Schema-of-record is mostly stated for v1_5 CT links, but the plan should directly state required columns and dtypes/nullability assumptions for `person_id`, `index`, `task`, `patient_string`/timeline col, `image_study_uid`, `image_series_uid`, `ct_accession_number`, and `ct_image_date`.
- Cross-stage surface list is incomplete unless legacy CT tooling is explicitly out-of-scope. Grep found additional `nifti_path`/nov25 surfaces in `src/data_tools/utils/ct_utils.py`, `src/data_tools/ct_info/download_subsampled_ct.py`, `src/data_tools/full_dataset_utils/ct_coverage.py`, `src/data_tools/csv_helper/subsample_v1_2.py`, and `src/vista_run/run.py`.
- Concrete success criterion exists for golden gates, but the Step 1b decision gate lacks pass/fail/continue routing and banked-SHA semantics.
- Coordination with in-flight work is partially stated through rung 0, but the plan should update its header: the user says rung 0 landed green on 2026-07-13, while the plan still says “must be green” and `Status: Draft` from 2026-07-07. Evidence: docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:5 and docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:22.
- Add `## Landing & cleanup`; currently absent. Include branch/worktree/tracker cleanup and docs/plans README status update.
- Add `## Out of scope`; currently absent. The scattered “not in scope” notes are not enough for fresh-agent handoff.
- No unresolved ambiguity appears buried in “Open questions — resolved”; OQ-M/N/P are resolved records, but OQ-P contains follow-up tooling breakage that should move to explicit out-of-scope/follow-up.

## Verification & Handoff Design (VM-handoff-bound)
- Archetypes selected correctly: schema/contract smoke for v1_5 CT resolution, data-sanity coverage report, migration/re-anchoring baseline rebuild, before/after refactor parity, declared allowlist diff, PHI readback. These fit the canonical menu.
- Missing/weak archetype: add accept/reject schema cases for UID resolution and a silent-fallback STOP that proves `nifti_path` cannot be used after Axis A. This is important because `nifti_path` exists as a deprecated integer in v1_5.
- Expected-vs-unexpected envelope is partly good: Step 1b is deliberately soft and recognizes series-selection drift. It is not yet a well-formed decision gate because it does not predeclare what Phil can accept inline versus what forces re-plan, nor where the executor records the decision.
- Phasing is insufficient for a complex handoff. The plan has a mid-sequence human decision gate plus bank-before/diff-after golden sequence; the canonical spec requires a `Handoff phasing` decomposition. A single-session framing is not adequate unless the plan explains that Phil will be available synchronously and how banked artifacts survive the gate.
- Machine selection is acceptable as Claude-Code CPU/GCP VM for weights-free golden checks, but the plan should explicitly say no GPU/high-throughput runner script is needed because all Steps 1-5 are weights-free golden harness/readback on the Claude-Code CPU executor.

## Suggested Revisions
- Add `## Out of scope` and `## Landing & cleanup` sections per `claude_ops.md`.
- Add a `Handoff phasing` block with phases, banked artifacts, gates, destructive status, and next-doc trigger.
- Replace shorthand file paths with full repo-relative paths for all planned edits and follow-up surfaces.
- Add a shared UID resolver contract: inputs, null behavior, output blob/filename, prefix source, logging fields, and failure mode.
- Clarify whether legacy CT tooling is migrated now or explicitly deferred; if deferred, add a follow-up item and verification that the golden path does not depend on it.
- Rewrite Axis C to match the existing CT adapter boundary: loader/resolver outside the adapter, CT adapter reused for selected indices and windowing, unless the author intentionally wants to change that boundary.
- Update the prerequisite/header to reflect rung 0 green on 2026-07-13, while keeping rung 0 out of this plan.
- Tighten Step 1b PHI readback to counts/categories only unless `/phi-vet` allows UID examples.

## Questions For The Author
- Should the legacy CT subsampling/download/coverage utilities be migrated in this plan, or explicitly deferred as follow-up because rungs 1-2 only need the golden/run path?
- Should `src/vista_run/run.py` remain supported for CT after the v1_5 cut, or is `run_bq.py` the only runtime path for this plan?
- For Step 1b, what coverage outcome should Phil be allowed to accept inline, and what outcome must force a re-plan?

## Audit Trail
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- src/vqa_dataset.py
- src/data_tools/utils/query_utils.py
- src/data_tools/utils/task_data_utils.py
- src/data_tools/utils/ct_utils.py
- src/data_tools/csv_helper/subsample_csv.py
- src/data_tools/csv_helper/subsample_csv_from_bq.py
- src/tests/ct_test.py
- src/vista_run/golden_harness.py
- src/vista_run/run_bq.py
- src/vista_run/run.py
- src/context/adapters/ct.py
- configs/all_tasks.yaml
- configs/all_tasks.rung0.yaml
- docs/02-ct-scans.md
- docs/03-vista-bench-data-cohort.md
- docs/05-retrieval.md
- docs/next.md
- docs/vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md
- figures/data_stats/person_id_subsampled.csv
