Reference: research-skills/claude_ops.md

# Feedback: VLM Phase 2 — config-context viewer (Codex review)

## Verdict
Revise. The core architecture is sound: current code confirms the viewer should reuse the golden weight-free `PromptDataset` capture path, not render from `Assembler.to_flat_item`. The plan needs tighter CLI/config resolution and a corrected Verification & VM handoff classification/envelope before handoff.

## Critical Gaps
- High | Tokenizer fallback is both a STOP and an inline decision gate | The handoff says an estimated token count is a Step-2 STOP, then says the same condition should not fail and should be noted for Phil; executor behavior is ambiguous and the canonical spec treats any class-2 decision gate as complex, not simple | Evidence: docs/plans/vlm-phase2-config-context-viewer.md:69; docs/plans/vlm-phase2-config-context-viewer.md:73 | Required fix: choose one: either make tokenizer load required and STOP on fallback, or pre-declare estimate as accepted with a capped envelope; if kept as a gate, classify the handoff as complex and add the phasing schema.
- High | Singular `--experiment` resolution is not specified for dict experiments | The config schema supports bare strings and block-list dicts via `normalize_experiments`, but current `run_bq` and `golden_harness` still iterate raw config entries; a viewer taking a singular name needs an explicit name-to-normalized-entry lookup or a documented legacy-only restriction | Evidence: src/context/normalize.py:110; src/vista_run/run_bq.py:183; src/vista_run/golden_harness.py:266; docs/plans/vlm-phase2-config-context-viewer.md:30 | Required fix: state that `context_viewer.py` resolves `--experiment <name>` through `normalize_experiments(orch.cfg)` and passes the normalized name/current legacy token into the shared capture core, or explicitly reject dict entries with a clear preflight error.
- Medium | `context_window` coverage is under-specified for the full registry | The plan names four adapters, but `MODEL_REGISTRY` has ten registered keys; leaving the rest to implementation risks false denominators or inconsistent "unknown" handling | Evidence: src/models/__init__.py:6; src/models/__init__.py:64; docs/plans/vlm-phase2-config-context-viewer.md:36 | Required fix: add a per-registry checklist: gemma3=120000, qwen3vl=120000, intern=120000, llava=4096, octomed=128000, and explicit `None`/source-needed decisions for qwen2vl, qwen2_5vl, medvlm, lingshu, llavamed.

## Failure Modes
- Scenario | Viewer shows a config dict experiment incorrectly or cannot find it | Why the plan misses it: it notes CLI reconciliation but does not bind `--experiment` to the normalized experiment contract | What to add: preflight accept/reject cases for legacy string, dict `{name: ...}`, unknown name, and multiple matches.
- Scenario | Golden extraction changes record bytes even though semantic fields match | Why the plan misses it: it states byte-identity but not exact JSON serialization constraints | What to add: preserve `json.dumps(..., sort_keys=True, ensure_ascii=False)` and golden record field names/order source; verify against a pre-refactor bank before viewer checks.
- Scenario | Tokenizer-only load pulls more than tokenizer artifacts or fails due cache/network | Why the plan misses it: the only anticipated fork is contradictory | What to add: exact tokenizer load order (`AutoProcessor(...).tokenizer`, then `AutoTokenizer`) with `local_files_only` policy if intended, and Expected/STOP behavior for cache miss.
- Scenario | Correctness coloring is wired without model responses | Why the plan misses it: viewer examples come from raw prompt capture, while `is_answer_correct` requires a model response and mapped label | What to add: make correctness coloring optional only when a matching result CSV is present; otherwise omit coloring rather than deriving from labels alone.

## Contract Checks
- In-repo: golden capture contract is verified. `capture_experiment` loads data, calls `_build_prompts_for_experiment`, sorts, constructs `PromptDataset`, and hashes `item["image"]` after `__getitem__`; fields captured include `index`, `person_id`, `task`, `experiment`, `model_type`, `windowing`, `dynamic_prompt`, `adapter_prompt_string`, `assembly_mode`, `image_count`, `path_tile_count`, `selected_indices`, and `image_hashes` | Evidence: src/vista_run/golden_harness.py:186; src/vista_run/golden_harness.py:206; src/vista_run/golden_harness.py:214; src/vista_run/golden_harness.py:227.
- In-repo: model-load split claim is correct. `TaskOrchestrator.__init__` creates BQ/GCS clients, constructs an adapter, and leaves `self.model`/`self.processor` unset until `_ensure_model_loaded()` | Evidence: src/vista_run/run_bq.py:99; src/vista_run/run_bq.py:114; src/vista_run/run_bq.py:143.
- In-repo: token counter claim needs naming correction. Current helper is `_count_prompt_tokens`, not `_count_tokens`, and it uses already-loaded `self.processor`/`self.model`; the viewer still needs the planned tokenizer-only helper | Evidence: src/vista_run/run_bq.py:154.
- In-repo: retrieval reject is correct. Retrieval prompt building passes `self.model`/`self.processor`, so weight-free capture must reject it | Evidence: src/vista_run/run_bq.py:804.
- In-repo: hot path claim is correct. `_build_prompts_for_experiment` creates `dynamic_prompt`, `PromptDataset.__getitem__` builds the flat `question`/`image` item and calls `CTAdapter.contextualize`; `Assembler.to_flat_item` is not the observed inference path | Evidence: src/vista_run/run_bq.py:733; src/vqa_dataset.py:108; src/vqa_dataset.py:226; src/context/assembler.py:91.
- In-repo: PHI ignore block exists and is the right place for viewer globs | Evidence: .gitignore:67; .gitignore:70.
- Cross-repo: no undocumented sibling-repo contract is introduced by the viewer. The roadmap’s study/series and specimen/block/slide hierarchy remains upstream in VISTABench; this plan consumes the given local data and leaf selections only.

## Modularity vs. YAGNI
- Shared capture core | Plan's choice: extract `src/vista_run/context_capture.py` and have both golden harness and viewer call it | Modular alternative + realistic use case: re-implement viewer capture in `src/results/context_viewer.py`, but that would duplicate loader dispatch, sorting, prompt building, image handling, and retrieval rejection | Recommendation: keep the extraction; require the viewer to call the shared generator for all image handling and prompt-building.
- Side-by-side layout | Plan's choice: defer | Modular alternative + realistic use case: build a comparison viewer now for two experiments/models | Recommendation: defer; current single-context viewer already satisfies Phase 2 instrumentation.
- Summarize support | Plan's choice: fail closed | Modular alternative + realistic use case: load model-backed summarization so summarized configs render exactly | Recommendation: keep fail-closed until summarize is actually in the inference path; add an accept/reject preflight test.

## Verification Gaps
- Add a structural HTML contract check: self-contained means no `http://`, `https://`, remote CSS/JS/image refs, and no local absolute image paths in `src`; counts-only readback.
- Add CLI/preflight checks for dict experiment resolution, retrieval rejection, summarize rejection, invalid model type, and output suffix/location under `results_dir`.
- Add token-counter verification with both branches: real tokenizer count for the canonical model and a deliberately simulated tokenizer miss if fallback remains accepted.
- Add optional correctness-coloring verification only if a matching results CSV is supplied; otherwise verify no correctness class is rendered.
- Add a `.gitignore` check for the exact new viewer glob, parallel to `*_golden.jsonl` and `golden/`.

## Handoff Readiness
- Gap: CLI contract remains partially unresolved | Fix: replace the residual note with the final flags, e.g. `--config --type --name --task --experiment [--limit] [--batches] [--out]`, or intentionally use `--model-type/--model-name`; define whether `--model` is a registry key or HF name.
- Gap: output path/suffix is not as strict as golden | Fix: state default path and require an ignored suffix such as `*_context_view.html`; if `--out` is allowed, require the suffix and reject repo-local paths unless they are ignored.
- Gap: fresh agent does not get a complete adapter context-window table | Fix: list every `MODEL_REGISTRY` key and value/unknown decision in the plan.
- Gap: the plan points at `generate_pdf.py` but not its stale CT helper caveat | Fix: say to reuse pagination/correctness-coloring patterns only; do not copy its CT slice extraction/count constants.
- Gap: in-flight coordination is mostly present but should mention raw experiment normalization | Fix: add that Phase 2 must not broaden cohort-source/config-schema behavior beyond the landed hot path unless `normalize_experiments` resolution is explicitly used.

## Verification & Handoff Design
- Archetype selection: correct core archetypes are before/after refactor parity, schema/output contract checks, edge/reject checks, silent-fallback STOPs, and PHI-clean readback. Add explicit accept/reject preflight cases and HTML contract checks.
- Expected-vs-unexpected envelope: weak. Step 1 has a clear zero-diff expectation. Step 2 needs a clearer envelope for tokenizer fallback, optional correctness coloring, and no-image/CT-bearing counts. The current tokenizer fork conflicts with Step-2 STOP language.
- Simple-vs-complex classification: as written, not simple if the tokenizer fallback remains a decision gate; the canonical classifier says any class-2 decision gate makes it complex. If the plan removes the gate by making fallback either accepted or a STOP, the single-repo Claude-Code CPU handoff can be simple.
- Phasing: one Claude-Code CPU phase is appropriate only after the tokenizer fork is resolved. No cross-repo SHA ripple, no bank/un-bank beyond the pre-refactor golden baseline, no destructive writes, and no non-Claude runner script are needed.

## Suggested Revisions
- Replace the tokenizer fork with unambiguous behavior: "tokenizer miss is STOP" or "estimated token count is accepted for this run and read back as estimated"; update Expected/STOP accordingly.
- Define final CLI flags and normalize `--experiment` through `context.normalize.normalize_experiments`.
- Add the full `MODEL_REGISTRY` `context_window` table, including `octomed=128000` and explicit `None` decisions where no local max is evident.
- State that `generate_pdf.py` is a layout/coloring reference only; do not reuse its stale CT extraction constants.
- Add preflight tests for retrieval, summarize, unknown experiment, dict experiment name, and output suffix/location.
- Add an HTML contract line: one self-contained file, no external URLs, no local image-path leakage, base64 thumbnails only.

## Questions For The Author
- Should tokenizer cache miss be a hard STOP, or is an estimated token bar acceptable for the first Phase 2 landing?
- Should `--model` mean registry key, HF model name, or should the viewer mirror golden exactly with `--type` and `--name`?
- Is correctness coloring in scope only when a matching result CSV exists, or should the first viewer omit it entirely?

## Audit Trail
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md
- docs/plans/vlm-phase2-config-context-viewer.md
- src/vista_run/golden_harness.py
- src/vista_run/run_bq.py
- src/vqa_dataset.py
- src/context/assembler.py
- src/context/adapters/ct.py
- src/context/adapters/pathology.py
- src/context/normalize.py
- src/models/base.py
- src/models/__init__.py
- src/models/gemma3.py
- src/models/qwen3.py
- src/models/llava.py
- src/models/internvl3_5.py
- src/models/qwen2.py
- src/models/qwen2_5.py
- src/models/medvlm.py
- src/models/lingshu.py
- src/models/octomed.py
- src/models/llava_med.py
- src/results/plot_code/generate_pdf.py
- src/results/results_analyzer.py
- .gitignore
