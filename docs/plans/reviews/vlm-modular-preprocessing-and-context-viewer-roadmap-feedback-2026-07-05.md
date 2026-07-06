Reference: docs/claude_ops.md

# Feedback: VLM eval roadmap — modality-adapter ContextBlocks + hierarchical selectors + config-context viewer (Codex review)

## Verdict
Revise. The plan is directionally sound and grounded in the current code, but it still leaves three load-bearing contracts under-specified: true inline image placement for target VLM adapters, the claimed pure no-op golden baseline, and the missing/punted v1_5 VM substrate.

## Critical Gaps
- High | `inline_by_timestamp` is not yet backed by the current model input contract | Current inference items are still `{question, image}` and Gemma/Qwen adapters prepend every image before one trailing text item, so an assembler that "injects" blocks into timeline text will not necessarily create image tokens at those timeline positions | Evidence: src/vqa_dataset.py:250, src/vqa_dataset.py:254, src/models/gemma3.py:49, src/models/gemma3.py:58, src/models/gemma3.py:66, src/models/qwen3.py:46, src/models/qwen3.py:55, src/models/qwen3.py:62 | Required fix: make the assembler output a typed multimodal content sequence, require each model adapter to either preserve that sequence or declare `inline_by_timestamp: unsupported`, and add per-model fallback/error semantics before Phase 1 verification.
- High | The "pure no-op golden diff" is not credible as written | Phase 1 changes more than slice selection plumbing: it replaces `is_gemma` dispatch with `by_model`, moves CT windowing, and routes EHR serialization toward `meds_tools`; any one can alter full-string prompts or rendered image bytes even if intended behavior is equivalent | Evidence: src/vqa_dataset.py:78, src/vqa_dataset.py:94, src/vista_run/run_bq.py:810, src/data_tools/utils/meds_timeline_utils.py:185, docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:352 | Required fix: split the golden into explicit legacy-serializer/legacy-preprocess equivalence tests first, then separately test meds_tools convergence and by-model preprocessing deltas; do not call the full refactor a pure no-op unless those compatibility shims are part of Phase 1.
- High | EHR convergence risks changing serialized strings without a pinned cross-repo contract | The local formatter skips any code containing `STANFORD`, rounds numeric values to one decimal, preserves current row order, and returns a specific empty message; the plan says delegate to `meds_tools` but only gates import/signature, not byte-equivalent serialization or accepted deltas | Evidence: src/data_tools/utils/meds_timeline_utils.py:199, src/data_tools/utils/meds_timeline_utils.py:215, src/data_tools/utils/meds_timeline_utils.py:225, src/data_tools/utils/meds_timeline_utils.py:240, src/data_tools/csv_helper/subsampled_retrieval_csv.py:190 | Required fix: pin the exact `meds_tools.patient_timeline` version/commit and define a fixture comparing local-vs-delegated output for ordering, spacing, rounding, STANFORD filtering, `exclude_report`, and empty output.
- High | Phase 0/v1_5 contract is dangling in this worktree | The roadmap delegates the runnable substrate to `docs/plans/gcp-vlm-eval-v1_5-multimodal-standup.md`, but that file is absent here; current checked-in config/code still point at v1_3/v1_1-era paths, so VM verification lacks a repo-grounded dataset and input contract | Evidence: docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:197, configs/all_tasks.yaml:6, src/data_tools/utils/query_utils.py:236, docs/03-vista-bench-data-cohort.md:8 | Required fix: either restore/include the Phase 0 doc in this branch or inline the needed v1_5 dataset/version/path resolver requirements, including expected v1_5 columns and cardinality checks.
- Medium | Downstream config normalization is broader than the plan's examples | The plan correctly names `final_metrics` and `all_model_response`, but `ct_experiment_plot.py` also parses YAML experiment comments as string list items and builds `set(experiments_list)`, so dict-style experiments will still break or lose display names unless the normalized-name API covers plotting display metadata | Evidence: src/results/final_metrics.py:47, src/results/all_model_response.py:65, src/results/plot_code/ct_experiment_plot.py:53, src/results/plot_code/ct_experiment_plot.py:184, src/results/plot_code/ct_experiment_plot.py:186 | Required fix: make `normalize_experiments` return names plus display labels, and update all result readers and plot display-name parsing in one change.
- Medium | Current line citations in the plan are partly stale | The underlying `.gitignore` claim is correct, but cited lines do not match this checkout; stale evidence weakens future reviews and implementation handoff | Evidence: .gitignore:46, .gitignore:60, .gitignore:63 | Required fix: refresh cited line numbers before implementation review.

## Failure Modes
- Scenario | Inline assembler creates a correct-looking viewer but inference sends all image tokens before text | Why the plan misses it: the viewer spec focuses on rendered order, while current adapters own chat-template serialization and currently append image parts before text | What to add: a test that inspects the actual adapter-produced prompt string/placeholders and `multi_modal_data` order per target model.
- Scenario | New dict-style experiment runs inference but metrics silently omit it | Why the plan misses it: it names the `set(...)` TypeError, but not comment/display parsing and filename filtering everywhere | What to add: end-to-end result discovery fixtures with one legacy string and one block-list dict through `final_metrics`, `all_model_response`, `check_image_usage`, and `ct_experiment_plot`.
- Scenario | meds_tools delegation changes prompt text enough to invalidate historical comparisons | Why the plan misses it: OQ-G asks about availability, not semantic/byte output equivalence | What to add: canonical MEDS window fixtures and a declared allowlist of acceptable serialization diffs, if any.
- Scenario | Hierarchical selectors pass through levels that v1_5 cannot populate, hiding bad metadata | Why the plan misses it: OQ-A asks the right question but Phase 1 still scaffolds all levels before cardinality is known | What to add: VM cardinality query results for CT study/series and pathology specimen/block/slide, plus validation that required metadata exists or the level is explicitly marked synthetic/pass-through.
- Scenario | Viewer cannot remain weight-free once model-backed selectors or summarization appear | Why the plan misses it: the caveat is present but not converted into config validation | What to add: viewer preflight that rejects or marks unevaluated model-backed selector steps unless a selector runtime is supplied.

## Contract Checks
- In-repo: `run_bq.py` currently has six `_load_*` methods plus `run_inference` dispatch by raw experiment strings; output path, resume-by-index, and retrieval overwrite are real contracts at src/vista_run/run_bq.py:155, src/vista_run/run_bq.py:216, src/vista_run/run_bq.py:310, src/vista_run/run_bq.py:350, src/vista_run/run_bq.py:392, src/vista_run/run_bq.py:443, src/vista_run/run_bq.py:541, src/vista_run/run_bq.py:718, src/vista_run/run_bq.py:719, src/vista_run/run_bq.py:722.
- In-repo: CT selection and preprocessing claims are accurate: `axial_all_image` branch samples 30 with the overshoot bug, image-only branches sample 50, `no_report` samples 10, and `is_gemma` controls windowing/size at src/vqa_dataset.py:78, src/vqa_dataset.py:82, src/vqa_dataset.py:94, src/vqa_dataset.py:198, src/vqa_dataset.py:208, src/vqa_dataset.py:227.
- In-repo: pathology sampling is inline and must be lifted without duplicating extraction: folder resolution and seed/count live at src/vista_run/run_bq.py:461, src/vista_run/run_bq.py:486, src/vista_run/run_bq.py:507, while `tile_wsi.py` is the extraction utility at src/data_tools/path_tools/tile_wsi.py:32.
- Cross-repo: `meds_tools.patient_timeline.get_described_events_window` is used locally, but `get_llm_event_string` is still the repo fork; pin function name, kwargs, return formatting, and dependency version before delegation.
- Cross-repo: vista_bench v1_5 needs explicit columns/cardinality: `person_id`, `index`, `patient_string`/timeline col, `nifti_path`, `_accession_number` or study/series identifiers, pathology `path_image_path`, `path_note_text`, and whether CT/path hierarchy levels are one-to-one or one-to-many.
- Downstream: result readers assume filename token `{task}_results_{experiment}.csv` and string experiment filters; normalized `name` must be the only output/resume/metrics token.

## Modularity vs. YAGNI
- Decision point | Plan's current choice | Realistic near-term use (named) or speculative | Recommendation or "raise to user".
- CT study/series levels | Build full contract, pass-through `all` where single-valued | Realistic if v1_5 exposes multiple `image_study_uid`/`image_series_uid`; current repo only hints at these fields in OMOP utilities | Raise to user after VM cardinality query, not as a cut.
- Pathology specimen/block/slide levels | Scaffold full hierarchy, active patch selector now | Speculative until path-team confirms specimen/block/slide metadata and v1_5 cardinality | Raise to user; require metadata source before implementing non-leaf selectors.
- Model-backed selector interface | Permit learned slice selector and EHR summarization, deterministic selectors now | Realistic named uses exist: learned CT slice selector and timeline summarizer | Keep interface, but require viewer/runtime validation for non-weight-free selectors.
- `inline_by_timestamp` assembler | Build now as config-selected strategy | Realistic clinical context use, but model support is unresolved | Keep only behind explicit per-model capability gates until adapter prompt-order tests pass.
- Summarize seam | Include in EHR filter chain | Realistic for long timelines and retrieval summarization | Keep as interface-only now; mark unavailable in weight-free viewer unless configured with a selector model.

## Verification Gaps
- The plan needs adapter-level behavioral tests for actual serialized multimodal prompts, not only viewer order or selected indices.
- The golden test should compare full `dynamic_prompt`, selected indices, rendered/preprocessed image bytes or hashes, and final adapter prompt/placeholders for each legacy preset.
- VM recipes should name the v1_5 config file, dataset constant value, expected local directories for CT/WSI/timeline CSVs, and expected row-count comparisons per loader; the missing Phase 0 doc currently prevents that.
- Add VM cardinality queries for v1_5 CT study/series and pathology specimen/block/slide before implementing pass-through hierarchy defaults.
- Add downstream result-reader tests for both legacy string and dict experiments across all four named readers.
- Add repo-local plan-review checklist. None exists in this checkout; a repo-grounded checklist would sharpen future reviews and reduce repeated framing drift.

## Suggested Revisions
- Add a "Model Adapter Inline Contract" section defining assembler output as ordered content parts and requiring each adapter to preserve, reject, or downgrade inline placement explicitly.
- Rewrite the Phase 1 golden-output claim from "pure no-op" to staged gates: legacy equivalence, then `by_model` equivalence, then meds_tools convergence with declared deltas.
- Restore or inline the missing v1_5 stand-up substrate details: dataset version, path resolver changes, required columns, materialized input locations, and row-count smoke expectations.
- Extend `normalize_experiments` spec to return display labels and update `ct_experiment_plot.py` comment parsing/filtering.
- Make OQ-A and OQ-G blocking preconditions for Phase 1 implementation, not just open questions.
- Refresh stale line citations, especially `.gitignore`.

## Questions For The Author
- For models that cannot honor mid-prompt image tokens, should `inline_by_timestamp` fail closed or downgrade to `ordered` with a warning in results metadata?
- Is byte-identical prompt text a hard requirement for Phase 1, or is a declared EHR serialization delta acceptable once meds_tools is pinned?
- Where is the authoritative v1_5 Phase 0 plan in this branch, and should this roadmap be self-contained if that plan lives elsewhere?
- What exact v1_5 fields should populate CT `study/series` and pathology `specimen/block/slide`, if any?

## Audit Trail
- Files inspected (paths only).
- docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md
- docs/claude_ops.md (missing)
- src/vista_run/run_bq.py
- src/vqa_dataset.py
- src/results/final_metrics.py
- src/results/all_model_response.py
- src/results/plot_code/check_image_usage.py
- src/results/plot_code/ct_experiment_plot.py
- src/data_tools/utils/meds_timeline_utils.py
- src/data_tools/csv_helper/subsampled_retrieval_csv.py
- src/data_tools/OMOP_meds_query/test_meds_tools.py
- src/models/__init__.py
- src/models/base.py
- src/models/gemma3.py
- src/models/qwen3.py
- configs/all_tasks.yaml
- docs/02-ct-scans.md
- docs/01-pathology-and-path-tools.md
- .gitignore
- src/data_tools/path_tools/tile_wsi.py
- src/data_tools/utils/query_utils.py
- docs/03-vista-bench-data-cohort.md
- src/tests/ct_test.py
