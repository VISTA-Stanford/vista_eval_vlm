Reference: docs/claude_ops.md

# Feedback: vista_eval_vlm Step 5 — LUMIA-live EHR adapter (Codex review)

## Verdict
Revise. The core implementation shape is viable and mostly reuses the existing EHR adapter, but the plan has verification-contract gaps around `diff_golden`, truncation idempotence, cohort pinning, and VM handoff specificity that would leave the executor making judgment calls.

## Critical Gaps
- Severity: Critical | Gap | Phase 1 is not actually self-resolving with the current golden-diff contract | Why it matters | `diff_golden.py` refuses metadata mismatches and fails on index-set mismatches, but the plan says to compare a legacy `no_image` bank against live `no_image` variants after fail-closed row dropping and then infer the `timeline` `code_filter` choice from strict PASS/FAIL. If missing LUMIA files or different `--limit` heads change the index set, a strict failure does not distinguish code-filter scope from cohort mismatch. | Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:217; src/vista_run/diff_golden.py:102; src/vista_run/diff_golden.py:156; src/vista_run/diff_golden.py:190 | Required fix | Specify exactly how both before/after banks are pinned to the same Phase-0 covered indices, or add an explicit prefilter/`--out`/task subset mechanism. State that Phase 1 PASS/FAIL is meaningful only after `task`, `experiment`, `model_type`, row_count, and index set match.
- Severity: Critical | Gap | The truncation safety argument is false | Why it matters | The plan relies on reapplying `truncate_timeline` being harmless, but the actual function prepends a newline for several modes and can re-truncate already-truncated text. That undermines the "safe fallback" claim and can create text drift unrelated to LUMIA rendering. | Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:109; src/data_tools/utils/meds_timeline_utils.py:83; src/data_tools/utils/meds_timeline_utils.py:86; src/data_tools/utils/meds_timeline_utils.py:104; src/data_tools/utils/meds_timeline_utils.py:150 | Required fix | Remove the idempotence claim. Make live rendering replace the timeline column with exactly one truncation pass, and explicitly state that passthrough rows are not re-contextualized through `EHRAdapter` with truncation.
- Severity: High | Gap | Phase 2 asks the VM executor to invent the allowlist implementation from PHI-bearing diffs | Why it matters | `diff_golden.py` currently prints BEFORE/AFTER text previews on failures, and `normalize_text` intentionally only strips trailing whitespace. The plan says to inspect residual `TEXT_FIELDS` and codify regexes, but does not provide a PHI-clean readback contract or require small local implementation review before accepting broad normalization. | Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:231; src/vista_run/diff_golden.py:57; src/vista_run/diff_golden.py:183 | Required fix | Add a Phase-2 sub-step that reports only counts, affected fields, indices, and scrubbed delta classes. Require any allowlist rule to be minimal, event-order-preserving, and re-run in non-lenient `--mode allowlist`; forbid pasting prompt/timeline previews into vm-status.
- Severity: High | Gap | The plan does not update all affected docs/contracts | Why it matters | This changes the inference hot path, row counts, config schema, and prompt provenance, but Files to Modify omits existing pipeline docs that still describe `patient_string` CSV substitution as the EHR source. | Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:142; docs/04-running-the-pipeline.md:102; docs/04-running-the-pipeline.md:113; docs/00-data-setup.md:10 | Required fix | Add doc updates for at least `docs/04-running-the-pipeline.md` and `docs/00-data-setup.md`, including fail-closed LUMIA row exclusion and `paths.lumia_corpus_dir`.

## Failure Modes
- Scenario | Live adapter drops rows and resume skips stale prior outputs | Why the plan misses it | `_setup_output_and_resume` keys resume by `index`; after fail-closed filtering, an existing result CSV for the old full cohort can make the run silently skip rows or compare against mixed-N outputs unless rerun output is isolated. | What to add | Require fresh output tags/results dirs or deletion for wired experiments when LUMIA-live is enabled; report before/after row counts and dropped-N in result metadata.
- Scenario | `person_id` is missing in a future normal experiment routed through `_build_prompts_for_experiment` | Why the plan misses it | `_load_task_data` fail-closes if BQ/CSV lacks `person_id`, and `_load_path_full_task_data` does too, but `_build_prompts_for_experiment` is also callable from `context_capture.py` and custom experiments can reach the default branch. | What to add | `_apply_ehr_adapter` should explicitly check `person_id` and raise a clear error for whitelisted experiments. Evidence: src/vista_run/run_bq.py:307; src/vista_run/run_bq.py:555; src/vista_run/context_capture.py:190.
- Scenario | `axial_all_image` Phase 3 is "optional" and misses the CT+EHR composition risk | Why the plan misses it | The plan wires `axial_all_image` in `_EHR_ADAPTER_EXPERIMENTS`, but visual QA makes it optional even though this is the only wired multimodal hot path. | What to add | Make `axial_all_image` mandatory in Phase 3 unless Phase 0/1 stops before CT-bearing verification. Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:65; docs/plans/vlm-step5-lumia-live-ehr-adapter.md:252.
- Scenario | The LUMIA corpus schema differs from `parse_lumia`'s assumptions | Why the plan misses it | The plan cites `meds2text`/LUMIA behavior but does not name the upstream schema source or require a schema sample check before relying on `.xml` shape. | What to add | Add a Phase 0 schema check using field names/counts only: root/encounter/entry/event shape, timestamp location, event attrs, and `unit_source_value`/text-value coverage. Evidence: src/context/adapters/ehr.py:127; src/context/adapters/ehr.py:134; docs/vm-status/2026-07-06-golden-harness.md:219.

## Contract Checks
- In-repo hot path contract: The plan's claim is correct that `_build_prompts_for_experiment` currently substitutes raw dataframe timeline text and never calls `EHRAdapter`. Evidence: src/vista_run/run_bq.py:737, src/vista_run/run_bq.py:789, src/vista_run/run_bq.py:811.
- Preset contract: The plan's claim is correct that `_ehr_block` currently only stamps `variant` plus serialization style, with no `select` chain. `get_preset(name: str) -> dict` exists and returns a deep copy. Evidence: src/context/presets.py:26, src/context/presets.py:46, src/context/presets.py:145.
- Loader provenance: For `_load_task_data` experiments, `person_id` is required and `embed_time` is carried from CSV into the merged dataframe when present. Evidence: src/vista_run/run_bq.py:307, src/data_tools/utils/task_data_utils.py:91, src/data_tools/utils/task_data_utils.py:103. `axial_all_image` uses the same normal `_load_task_data` route through `needs_normal`. Evidence: src/vista_run/run_bq.py:200, src/vista_run/run_bq.py:206.
- Pathology precedent: The plan correctly points to inline adapter construction in `_load_path_task_data`, but `PathologyAdapter` uses a cohort-level `materialize(...)` method, while the EHR sketch uses per-row `ingest(...).contextualize(...)`; this is appropriate for EHR but should be described as analogous construction, not identical materialization. Evidence: src/vista_run/run_bq.py:516, src/context/adapters/pathology.py:74, src/context/adapters/ehr.py:169.
- Config/path precedent: `configs/all_tasks.yaml:93` is the contributor-specific `/data/fries/.../thoracic_cohort_lumia` path and prior VM status documents it as absent on `phil-sllm-01`; VM-local overlay is the established fix. Evidence: configs/all_tasks.yaml:93; docs/vm-status/2026-07-06-golden-harness.md:167; docs/vm-status/2026-07-06-golden-harness.md:267.
- Cross-repo/external contract: The LUMIA `.xml`/`meds2text` markup is effectively an upstream producer contract. The plan should cite the concrete schema source or prior VM field-coverage result more explicitly, because `parse_lumia` depends on `<encounter>/<entry timestamp>/<event>` shape. Evidence: src/context/adapters/ehr.py:21; src/context/adapters/ehr.py:127; docs/vm-status/2026-07-06-golden-harness.md:220.
- Diff contract: `diff_golden.py --mode allowlist` is designed for real allowlist fill-in after VM diffs, but it still hard-fails structure, index-set, no-shared-index, and incompatible metadata. Phase 2 should treat those as preconditions, not residual classes for human interpretation. Evidence: src/vista_run/diff_golden.py:29; src/vista_run/diff_golden.py:121; src/vista_run/diff_golden.py:156.

## Modularity vs. YAGNI
- Decision point | `timeline` variant `code_filter` | Plan's current choice | One hardcoded `select = []` line, flipped after Phase 1, no runtime toggle. | Modular alternative + the realistic use case it would serve | A config flag could support different cohorts that intentionally include or exclude STANFORD-coded imaging reports in full timelines. | Recommendation, OR "raise to user" when realistic-use is unclear. | Keep the YAGNI choice for now. The code already supports dict experiment overrides via `normalize_experiments`, so future cohort-specific behavior can be expressed as a custom block config without adding a global toggle. Evidence: src/context/normalize.py:82.
- Decision point | `paths.lumia_corpus_dir` default | Plan's current choice | Add unset/placeholder config key and require VM-local overlay. | Modular alternative + the realistic use case it would serve | A committed bucket-relative value plus resolver could mirror CT and simplify executor setup. | Recommendation, OR "raise to user" when realistic-use is unclear. | Keep VM-local-only, but add a commented example and actionable error naming `configs/all_tasks.vm.yaml` and `gs://vista_bench/thoracic_cohort_lumia/`. LUMIA `LocalPatientRetriever` expects a resolved local directory, not a bucket-relative prefix. Evidence: src/retrieval/local_retriever.py:49; src/retrieval/local_retriever.py:60.
- Decision point | Reuse-vs-rewrite helper | Plan's current choice | `_apply_ehr_adapter` reuses `EHRAdapter.ingest()` and `.contextualize()`. | Modular alternative + the realistic use case it would serve | Inline parsing/filtering would expose row-level logs or special-case missing files, but duplicates adapter logic. | Recommendation, OR "raise to user" when realistic-use is unclear. | Current reuse is right. Add only validation/logging around it, not parser/filter reimplementation.

## Verification Gaps
- Add a static/unit-level check for `truncate_timeline` double-application or remove the claim entirely; current function behavior contradicts idempotence.
- Add a preflight command that verifies `paths.lumia_corpus_dir` exists locally and contains `.xml` files before any golden/viewer run, not only GCS listing.
- Add row-count/index-set checks to every golden phase: legacy row count, live row count after drops, shared indices, missing/extra counts, and dropped-N.
- Make `axial_all_image` mandatory in Phase 3 because it is in the wired experiment set and exercises CT+EHR composition.
- Add PHI-clean diff readback rules for Phase 2; `diff_golden.py` prints text previews that must not be pasted into repo docs.
- Add docs verification: pipeline docs updated, `.gitignore` still covers golden/viewer outputs, and VM-local config remains untracked.

## Handoff Readiness
- The plan needs exact commands for Phase 0's "restrict to the PFS-1yr subsampled task cohort"; the SQL block currently contains a comment where the actual join/filter should be.
- The schema of the `select` chain is present in code sketches but should be restated as a config contract: list of dicts with `fn`, `window.before/after`, `code_filter.exclude_stanford`.
- The out-of-scope boundary mostly covers `all_vb_timeline_only` and `path_full`, but should also state retrieval experiments remain untouched because they route through `build_retrieval_prompts`. Evidence: src/vista_run/run_bq.py:798.
- The Open Questions section contains a resolved strikethrough item. Move the fail-closed decision into Approach/Goal and remove it from Open Questions; leave only true unresolved items.
- Landing plan references branch off `885dc6a` and "no worktree needed" from plan time. Given shared-checkout standards, revise to say re-check `git status`, branch, and worktrees at implementation start rather than relying on stale plan-time state. Evidence: docs/plans/vlm-step5-lumia-live-ehr-adapter.md:274.

## Verification & Handoff Design
- Archetype selection is broadly right: Phase 0 precondition, Phase 1 decision gate, Phase 2 declared-delta allowlist, Phase 3 human visual QA.
- Expected-vs-unexpected envelope is incomplete. Phase 1 treats strict FAIL as only the code-filter fork, but current tool failures also include metadata mismatch, index-set mismatch, structure drift, no shared rows, or non-LUMIA text deltas.
- Phasing is mostly cheap-to-expensive: coverage/schema before golden diffs before viewer. However, Phase 2 says "extend N there, don't rebank from scratch" without specifying how the existing Phase-1 bank is extended reproducibly; golden files are append/write outputs, so the executor needs concrete commands and file naming.
- Bank-forward is under-specified. Phase 0's covered set is supposed to feed Phase 1, but neither `golden_harness.py` nor the plan exposes a way to pass that exact set. Add a concrete subset artifact path on the VM, with PHI-safe readback limited to counts.
- Phase 3 reuses the viewer precedent correctly, but should not skip self-containment/PHI checks just because prior Phase 2 validated them; this plan emits new PHI HTML and should rerun the cheap grep/card-count checks. Evidence: docs/vm-status/2026-07-15-phase2-config-context-viewer.md:111.

## Suggested Revisions
- Replace the truncation paragraph with: legacy CSV timelines are already truncated in loaders; live LUMIA timelines are rendered and truncated once by `EHRAdapter`; passthrough experiments remain outside `_EHR_ADAPTER_EXPERIMENTS`.
- Add an `_apply_ehr_adapter` contract: requires whitelisted experiment, non-null `timeline_col`, `person_id`, configured existing `paths.lumia_corpus_dir`, and `embed_time` for windowed variants; log input N, dropped N, output N.
- Rewrite Phase 1 to pin identical covered indices before banking. Treat non-matching index set as STOP distinct from code-filter FAIL.
- Rewrite Phase 2 readback to avoid prompt/timeline excerpts and to require a reviewed minimal normalizer before "green".
- Add `docs/04-running-the-pipeline.md` and `docs/00-data-setup.md` to Files to Modify.
- Move resolved fail-closed policy out of Open Questions and fold it into Approach plus Verification expected output.
- Add retrieval experiments to the explicit untouched boundary.

## Questions For The Author
- Should Phase 1 compare only patients with exact legacy/live index-set parity, or should the implementation add a small VM-only subset artifact to force both banks onto the same covered cohort?
- Is `paths.lumia_corpus_dir` intentionally separate from `retrieval.corpus_dir` long term, or should one fall back to the other when only one is configured?
- What minimum Phase-0 coverage threshold should block landing for the full default run, not just PFS-1yr smoke?

## Audit Trail
- ../research-skills/claude_ops.md
- docs/plans/vlm-step5-lumia-live-ehr-adapter.md
- src/context/adapters/ehr.py
- src/context/selectors/ehr_filters.py
- src/context/presets.py
- src/vista_run/run_bq.py
- src/vista_run/context_capture.py
- src/vista_run/diff_golden.py
- src/vista_run/golden_harness.py
- src/results/context_viewer.py
- src/vqa_dataset.py
- src/data_tools/utils/task_data_utils.py
- src/data_tools/utils/meds_timeline_utils.py
- configs/all_tasks.yaml
- src/retrieval/local_retriever.py
- src/context/normalize.py
- src/context/adapters/pathology.py
- docs/vm-status/2026-07-06-golden-harness.md
- docs/vm-status/2026-07-15-phase2-config-context-viewer.md
- docs/00-data-setup.md
- docs/04-running-the-pipeline.md
