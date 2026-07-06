Reference: docs/claude_ops.md

# Implementation Feedback: VLM modular preprocessing framework (foundation slice)

## Verdict
Revise before commit. The Phase 0.5 CT fix and most foundation mechanics match the scoped plan, but the downstream-reader normalization currently imports the full adapter stack and fails in this local reader environment before normalization can run.

Repo-local checklist: none provided. A repo-grounded checklist would sharpen the review by separating deliberate deferrals from exact foundation-slice acceptance criteria.

## Plan Coverage

| Section | Status | Evidence: path:line | Notes |
|---|---|---|---|
| Phase 0.5 axial overshoot fix | Done | `src/vqa_dataset.py:193` | `axial_all_image` and retrieval+image now use 30 slices with `i/(num_slices - 1) * (depth - 1)`, retaining the never-firing clamp at `src/vqa_dataset.py:208`. |
| Phase 0.5 docs | Done | `docs/02-ct-scans.md:44` | Docs now say 50 for `no_timeline`/`all_vb_image_only`, 30 for `axial_all_image` and retrieval+image, 10 for `no_report`. |
| ContextBlock | Done | `src/context/block.py:24` | Uniform `{id, modality, representation, payload, metadata}` exists; representation guard is present. |
| Adapter ABC and registry | Done | `src/context/adapters/base.py:48`, `src/context/adapters/__init__.py:8` | `ModalityAdapter`, `ModelCaps`, `resolve_model_caps`, and registry for `ehr`/`ct`/`pathology` are present. |
| CT selector + windowing | Done | `src/context/selectors/ct_selectors.py:16`, `src/context/windowing.py:20` | Selector reproduces legacy formula and clamp; window/norm/grayscale are lifted from current sources. |
| CT adapter | Done | `src/context/adapters/ct.py:46` | Uses selector indices, legacy Gemma RGB path, legacy grayscale path, and `pad_to_size`. |
| Pathology selector + adapter | Partial | `src/context/selectors/path_selectors.py:26`, `src/context/adapters/pathology.py:74` | Seed-once, per-row sampler is structurally correct; missing-row dropping remains caller-dependent rather than implemented in `materialize`. |
| EHR adapter | Done | `src/context/adapters/ehr.py:33`, `src/context/adapters/ehr.py:53` | Imports canonical `get_llm_event_string`/`truncate_timeline`; `_lumia_event_to_row` isolates extra LUMIA fields and defaults them to `None`. |
| EHR STANFORD skip seam | Done | `src/context/selectors/ehr_filters.py:102`, `src/context/adapters/ehr.py:190` | `code_filter(exclude_stanford=True)` drops by the same substring predicate and render passes `exclude_report=False`. |
| Assembler inline seam | Done | `src/context/assembler.py:38`, `src/models/base.py:33` | `inline_by_timestamp` fails closed through `supports_inline`; `BaseVLMAdapter.supports_inline = False` exists. |
| Presets | Done | `src/context/presets.py:61` | CT counts, path `n=100/seed=42`, cohort, and prompt_style are encoded for legacy names. |
| normalize_experiments | Done | `src/context/normalize.py:54` | Resolves strings and dicts to normalized objects; unknown bare strings pass through with `unknown=True`. |
| Downstream reader normalization | Drifted | `src/results/final_metrics.py:19`, `src/results/plot_code/ct_experiment_plot.py:9`, `src/results/plot_code/generate_pdf.py:16` | Reader call sites use `context.normalize`, but that import triggers full `context/__init__.py` and adapter imports. Confirmed local failure: `ModuleNotFoundError: No module named 'numpy'`. |
| Hot-path dissolution wiring | Missing-but-deferred | `src/vqa_dataset.py:125`, `src/vista_run/run_bq.py:738` | Still branches on experiment as requested; not flagged. |
| Phase 2 context viewer | Missing-but-deferred | n/a | Not in this foundation slice. |
| Phase 1.5 inline interleaving | Missing-but-deferred | `src/context/assembler.py:86` | Real interleaving intentionally raises until later. |

## Critical Drift

- High | Plan says result readers should consume normalized names without expanding the preprocessing stack; code imports `context.normalize`, but Python loads `src/context/__init__.py` first, which imports `.adapters`, which imports CT/pathology/EHR dependencies. In this environment, `PYTHONPATH=src python -c "from context.normalize import experiment_names"` fails at `src/context/adapters/ct.py:16` with `ModuleNotFoundError: No module named 'numpy'`. | Evidence: `src/context/__init__.py:17`, `src/context/adapters/__init__.py:4`, `src/context/adapters/ct.py:16`, `src/results/final_metrics.py:19` | Required fix: make `context.normalize` importable without importing adapters/PIL/numpy/pandas/top-level project modules, e.g. slim `context/__init__.py`, lazy adapter exports, or move normalization/presets into a dependency-light package path used by readers.

## Missing Pieces

- Medium | The pathology adapter says rows with no tiles are dropped “as legacy does,” but `materialize` only appends `[]` and returns the full DataFrame. If future wiring trusts `materialize` as the lifted cohort behavior, it will retain rows legacy `_load_path_task_data` drops. | Evidence: `src/context/adapters/pathology.py:77`, `src/context/adapters/pathology.py:85`, legacy drop at `src/vista_run/run_bq.py:525` | Required fix: either move the drop into `materialize` for byte-identical cohort behavior or document/require the caller to apply the same row filter immediately after materialization.

## Contract Violations

- High | Downstream normalization contract is not usable in a lightweight reader environment because `context.normalize` is coupled to adapter imports. | Evidence: `src/context/__init__.py:17`, `src/context/adapters/__init__.py:4`, `src/context/adapters/ct.py:16`, `src/context/adapters/ehr.py:34` | The plan explicitly called out reader import footprint risk; this is now a confirmed blocker for those readers outside the full inference dependency environment.

## Byte-identity Risks

- Medium | Pathology byte identity depends on preserving the legacy row filter after sampling. The sampler itself matches legacy `random.seed(42)` + per-row `random.sample`, but retaining empty-tile rows would change row counts and downstream output alignment. | Evidence: `src/context/selectors/path_selectors.py:26`, `src/context/adapters/pathology.py:81`, `src/vista_run/run_bq.py:507`, `src/vista_run/run_bq.py:525` |
- Low | `parse_lumia` sorts all rows chronologically after parsing. That is defensible for the plan’s “event order must be identical” target only if LUMIA document order is chronological or legacy renderer input was chronological. | Evidence: `src/context/adapters/ehr.py:128`, `src/data_tools/utils/meds_timeline_utils.py:205` |
- Low | LUMIA `description` falls back to `name` and `text_value` may come from element text. This is intentionally isolated, but VM field coverage still determines whether gate 3 is byte-identical or declared-delta. | Evidence: `src/context/adapters/ehr.py:62`, `src/context/adapters/ehr.py:67`, `src/context/adapters/ehr.py:135` |

## Test Gaps

- Add a reader-only smoke test: `from context.normalize import experiment_names, display_names` in an environment without imaging deps. This would have caught the current import failure. Evidence: reader imports at `src/results/final_metrics.py:19`, `src/results/plot_code/ct_experiment_plot.py:9`, `src/results/plot_code/generate_pdf.py:16`.
- Add a mixed experiments config test containing a legacy string, a dict entry, and an unknown bare string; assert `experiment_names` and `display_names` do not touch adapters. Evidence: `src/context/normalize.py:56`, `src/context/normalize.py:76`, `src/context/normalize.py:114`.
- Add a pathology materialization parity fixture with two rows, one missing tile folder, and one oversize tile list; assert row filtering, sample order, and `path_tile_paths` match legacy. Evidence: `src/context/adapters/pathology.py:74`, `src/vista_run/run_bq.py:507`.
- Add an assembler preflight test for `inline_by_timestamp` against default `ModelCaps` and default `BaseVLMAdapter`; assert `AssemblyError` before content generation. Evidence: `src/context/assembler.py:49`, `src/models/base.py:33`.

## Defensible Deviations

- EHR renderer and truncator are imported canonically rather than copied, which is stricter than the plan text’s “vendor/lift” language and avoids a third formatter copy. Evidence: `src/context/adapters/ehr.py:33`.
- `summarize` is a fail-loud seam, not a silent raw-timeline fallback. That matches the plan’s requirement to decide the failure mode. Evidence: `src/context/selectors/ehr_filters.py:126`.
- Unknown bare experiment names pass through for result discovery instead of failing. This matches the requested tolerant passthrough. Evidence: `src/context/normalize.py:68`.

## Suggested Code Edits

- Decouple `context.normalize` from `context.__init__` adapter imports so metrics/plot/PDF readers can run without imaging/EHR runtime dependencies.
- Make `PathologyAdapter.materialize` either perform the legacy empty-tile row drop or return both the materialized DataFrame and an explicit mask/contract that callers must apply.
- Keep `parse_lumia` chronological sorting only if a VM sample confirms it matches legacy DataFrame ordering; otherwise preserve source order and sort only when explicitly requested by config.

## Questions For The Author

- Should `normalize_experiments(...).spec` include `prompt_style` and `legacy_retrieval`, or is that intentionally kept outside the plan’s `{name, display, cohort, assembly, blocks}` contract until hot-path wiring?
- For pathology materialization, do you want the adapter to own the legacy row dropping now, or should the future loader wiring own it?
- Is chronological sorting of parsed LUMIA events guaranteed by upstream LUMIA generation, or should the adapter preserve XML order until the VM equivalence check?

## Audit Trail

- Read plan: `docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md`.
- Read unstaged diff for scoped tracked files and confirmed staged diff empty.
- Read all untracked `src/context/**` files in full.
- Read legacy sources: `src/data_tools/utils/meds_timeline_utils.py`, `src/vista_run/utils/utils_inference.py`, `src/vqa_dataset.py`, `src/vista_run/run_bq.py`.
- Checked `meds2text/docs/markup.md`; file is absent in this worktree.
- Ran `git status --short`; scoped tracked files are modified and `src/context/` is untracked.
- Ran normalization import smoke with `PYTHONPATH=src`; it failed before use because `context.normalize` imports the full adapter stack through package initialization.
