Reference: docs/claude_ops.md

# Implementation Feedback: VLM Phase 2 — config-context viewer

## Verdict
Revise before commit. The golden capture extraction appears byte-preserving by static field comparison, and the viewer covers the core CLI/rendering/tokenizer requirements, but the model-backed `summarize` preflight is incomplete for the repo's selector-chain representation.

## Plan Coverage
| Slice / section | Status | Evidence: path:line | Notes |
|---|---|---|---|
| Shared capture core extracted from golden harness | Done | src/vista_run/context_capture.py:175; src/vista_run/golden_harness.py:117 | Loop shape matches the pre-refactor harness: load data, build prompts, stable sort by `(person_id, index)`, apply `limit`, build `PromptDataset`, iterate `dataset[i]`. |
| Golden byte-identity record reconstruction | Done | src/vista_run/golden_harness.py:117; src/vista_run/golden_harness.py:200 | `image_count` / `image_hashes` still come from `_image_summary(ex.images)`, and `ex.images` is assigned from `item.get("image")` in the capture core. `json.dumps(record, sort_keys=True, ensure_ascii=False)` is unchanged. |
| Captured scalar field provenance | Done | src/vista_run/context_capture.py:218 | `index`, `person_id`, `dynamic_prompt`, `adapter_prompt_string`, `selected_indices`, `path_tile_count`, and `windowing` map to the same source expressions/coercions as the pre-refactor `HEAD:src/vista_run/golden_harness.py`. |
| Viewer CLI `--config --type --name --task --experiment [--limit] [--batches] [--out]` | Done | src/results/context_viewer.py:493 | Matches Q2 and keeps `--experiment` singular. |
| Invalid `--type` before BQ/GCS client build | Done | src/results/context_viewer.py:506 | `MODEL_REGISTRY` is checked before `TaskOrchestrator(...)`. |
| Experiment resolution through `normalize_experiments(orch.cfg)` | Done | src/results/context_viewer.py:167; src/results/context_viewer.py:531 | Unknown and duplicate normalized names fail closed. |
| Retrieval reject | Done | src/results/context_viewer.py:534 | Checks both `RETRIEVAL_EXPERIMENTS` and `norm.legacy_retrieval` before tokenizer/capture. |
| Model-backed `summarize` reject | Partial | src/results/context_viewer.py:192; src/context/selectors/ehr_filters.py:149 | See Critical Drift: direct flags/name checks miss `select: [{fn: summarize}]`, the selector-chain form documented by the code. |
| Tokenizer-only load hard STOP, no whitespace fallback | Done | src/results/context_viewer.py:117; src/results/context_viewer.py:149; src/results/context_viewer.py:549 | `AutoProcessor`/`AutoTokenizer` only; raises `TokenizerLoadError` on failure; no `split()` fallback. |
| `context_window` weight-free seam | Done | src/models/base.py:40; src/models/gemma3.py:9; src/models/qwen3.py:12; src/models/internvl3_5.py:13; src/models/octomed.py:12; src/models/llava.py:6 | Class attributes are set on the five specified adapters; table-`None` adapters inherit `BaseVLMAdapter.context_window = None`. Viewer reads `orch.adapter` without `load()`. |
| Self-contained HTML assets | Done | src/results/context_viewer.py:238; src/results/context_viewer.py:458 | Images are emitted as `data:image/png;base64,...`; CSS is inline; no external CSS/JS/font refs in the template. |
| No correctness coloring / no results CSV dependency | Done | src/results/context_viewer.py:13; src/results/context_viewer.py:493 | No `--results-csv`; grep found no `results_analyzer` import in `context_viewer.py`. Input-sanity warnings are limited to 0-image/empty-prompt. |
| `.gitignore` viewer PHI globs | Done | .gitignore:74 | Adds `*_context_view.html` and `context_view/` next to golden output globs. |
| Docs usage section | Done | docs/04-running-the-pipeline.md:168 | Documents VM-only, PHI output, CLI, tokenizer hard stop, and fail-closed preflight. |

## Critical Drift
- Severity: Critical | Plan says "`summarize` = fail-closed at preflight" because silently skipping it would misrepresent context (docs/plans/vlm-phase2-config-context-viewer.md:59), and the VM edge cases require a "`summarize`-configured experiment -> rejected" (docs/plans/vlm-phase2-config-context-viewer.md:84). Code checks only `summariz` in the experiment name or direct `block.config.summarize` / `block.config.summarization` flags (src/results/context_viewer.py:192), but the repo's selector seam represents this as a filter function: `EHR_FILTERS["summarize"]` and `MODEL_BACKED_FILTERS = {"summarize"}` (src/context/selectors/ehr_filters.py:149). A block-list dict with `config: {select: [{fn: summarize, ...}]}` can pass preflight and render as though it were weight-free. | Required fix: update `_uses_model_backed_summarize()` to inspect EHR block selector chains, including dict or list forms under `config["select"]`, and reject any selector whose `fn` is in `MODEL_BACKED_FILTERS` or equals `"summarize"`.

## Missing Pieces
- Plan item: fail-closed model-backed summarize detection | Where it should land: `src/results/context_viewer.py:_uses_model_backed_summarize` | Why it matters: the viewer's core contract is not to render a context that would require weights or skip a model-backed preprocessing step | Suggested code change: import or mirror `MODEL_BACKED_FILTERS` from `context.selectors.ehr_filters`, walk `norm.blocks[*]["config"]["select"]`, normalize a single dict to a list, and reject when any selector dict has `fn` in that set.

## Contract Violations
- No manifest / SQL / output-path / cross-repo / labels contract drift found in static inspection.

## Test Gaps
- The plan's landing gate requires VM-side golden byte-identity regression and viewer/preflight smoke tests (docs/plans/vlm-phase2-config-context-viewer.md:81). I did not run them here because this machine is planner-only per the operating standards. The author still needs VM evidence for `diff_golden --mode strict`, CT/no-image HTML generation, retrieval reject, summarize reject, unknown/ambiguous experiment reject, invalid type reject, and tokenizer-miss STOP.
- Add a focused test or smoke fixture for the missed selector-chain case: a normalized experiment block with `config.select: [{"fn": "summarize"}]` must exit non-zero before capture/render.

## Defensible Deviations
- `--out` is constrained to `_context_view.html` (src/results/context_viewer.py:513). The plan required default output under `results_dir`; this extra suffix guard is a defensible PHI backstop.
- The viewer page notes that display pagination is not exact Gemma runtime batching (src/results/context_viewer.py:412). The plan asked to paginate by `runtime.batch_size`; this caveat is a reasonable clarification because the inference path may regroup Gemma batches by image count.

## Suggested Code Edits
- `src/results/context_viewer.py:192`: replace `_uses_model_backed_summarize()` with a selector-aware walker. Concretely, check `block_cfg.get("select")`; if it is a dict, wrap it in a list; if it is a list, scan each dict for `fn == "summarize"` or `fn in MODEL_BACKED_FILTERS`. Keep the existing name/direct-flag checks as conservative extras.
- `src/results/context_viewer.py:192`: consider using `context.selectors.ehr_filters.MODEL_BACKED_FILTERS` instead of hard-coding only summarize, so future model-backed filters fail closed without another viewer change.

## Questions For The Author
- Should custom block-list experiments whose `name` is not a legacy preset be considered runnable by this viewer today, or should the viewer reject `norm.unknown` / unsupported names until `run_bq` consumes block specs directly? Static inspection shows the current capture core still passes only `norm.name` into the legacy prompt/dataset path.

## Resolution (2026-07-15, applied)
- **Critical (model-backed summarize selector-chain) — APPLIED.** `_uses_model_backed_summarize` (src/results/context_viewer.py) now walks each block's `config['select']` (single dict OR list) and rejects any selector whose `fn` is in `MODEL_BACKED_FILTERS`, imported from `context.selectors.ehr_filters` (so future model-backed filters fail-closed with no viewer edit); the name/direct-flag checks are kept as conservative extras.
- **Defensible Deviations — CONFIRMED KEEP** (both): the `--out` `_context_view.html` suffix guard (PHI backstop, mirrors golden's `--out` guard) and the display-pagination-vs-Gemma-batching caveat.
- **Question For The Author — RESOLVED (Phil): keep current behavior.** A config experiment whose `name` is not a legacy preset (`norm.unknown`) is accepted and passed to the default loader — the same tolerance `run_bq` has; the viewer is deliberately not stricter than the runner (you could otherwise run an experiment you can't view). No code change.
- **Test Gaps** — VM-side smoke is the plan's gate (`/vm-handoff` next). The selector-chain summarize unit assertion was offered via `/review-tests`; Phil chose to proceed to the VM handoff.

## Audit Trail
- docs/plans/vlm-phase2-config-context-viewer.md
- .gitignore
- docs/04-running-the-pipeline.md
- src/models/base.py
- src/models/gemma3.py
- src/models/internvl3_5.py
- src/models/llava.py
- src/models/octomed.py
- src/models/qwen3.py
- src/models/__init__.py
- src/vista_run/golden_harness.py
- src/vista_run/context_capture.py
- src/results/context_viewer.py
- src/vqa_dataset.py
- src/context/normalize.py
- src/context/presets.py
- src/context/selectors/ehr_filters.py
- src/vista_run/run_bq.py
- src/vista_run/diff_golden.py
