Reference: research-skills/claude_ops.md

# Implementation Feedback: Golden Harness (Task 3b gate)

## Verdict
Revise before commit. The harness mostly calls the real weight-free data/prompt/selection surfaces and the `selected_indices` instrumentation is inference-neutral, but the diff gate schema is internally inconsistent for `dynamic_prompt`, and the PHI output override can bypass the promised results-dir/ignore protection.

## Plan Coverage
| Item | Status | Evidence: path:line | Notes |
|---|---|---|---|
| Add golden harness script | Done | src/vista_run/golden_harness.py:178 | Captures rows through loaders, `_build_prompts_for_experiment`, and direct `PromptDataset` iteration. |
| Weight-free: no `run_inference` / `_ensure_model_loaded` | Done | src/vista_run/golden_harness.py:180, src/vista_run/golden_harness.py:185, src/vista_run/golden_harness.py:197 | The harness does not call `run_inference` or `_ensure_model_loaded`; the methods it calls are data loaders / prompt builders / dataset item loading. Retrieval is skipped before prompt building. |
| Reject retrieval experiments | Done | src/vista_run/golden_harness.py:95, src/vista_run/golden_harness.py:262 | Matches plan because `_build_prompts_for_experiment` passes `self.model` / `self.processor` for retrieval at src/vista_run/run_bq.py:794. |
| Loader dispatch mirrors `run_inference` | Done | src/vista_run/golden_harness.py:100, src/vista_run/run_bq.py:211 | The one-experiment dispatch matches the orchestrator branches for `no_timeline`, `all_vb_*`, path, `path_full`, no-report/report/timeline, retrieval rejection, and normal. |
| Capture `adapter_prompt_string` from post-truncate timeline column | Done | src/vista_run/golden_harness.py:218, src/vista_run/run_bq.py:318 | Loaders truncate `timeline_col` before prompt building; harness snapshots `raw.get(timeline_col)` before `_build_result_row` would drop it at src/vista_run/run_bq.py:809. |
| Capture raw `dynamic_prompt`, not `question` | Done | src/vista_run/golden_harness.py:217, src/vqa_dataset.py:102 | Harness reads the raw dataframe column, so `add_options` cannot append `" Options:"` to the golden prompt. |
| True `image_count` and deterministic `image_hashes` | Done | src/vista_run/golden_harness.py:155, src/vista_run/golden_harness.py:146 | Counts list length rather than `used_image` 0/1; hash includes mode, size, dtype, shape, and bytes. |
| `selected_indices` item instrumentation | Done | src/vqa_dataset.py:109, src/vqa_dataset.py:206, src/vqa_dataset.py:228, src/vqa_dataset.py:249 | CT 30/50/10 branches and pathology branch populate the field; inference consumers only read `raw_row`, `image`, and `question` in src/vista_run/run_bq.py:594, src/vista_run/run_bq.py:640, src/vista_run/run_bq.py:808. |
| Sorted deterministic output and limit after sort | Done | src/vista_run/golden_harness.py:190, src/vista_run/golden_harness.py:193 | Sort is stable and `--limit` is applied after sorting. Direct dataset iteration is single-process at src/vista_run/golden_harness.py:203. |
| Diff script structure-vs-text gates | Drifted | src/vista_run/diff_golden.py:38, src/vista_run/golden_harness.py:76 | `golden_harness.GATE1_FIELDS` includes `dynamic_prompt`, but `diff_golden.STRUCTURE_FIELDS` does not. See Critical Drift. |
| Default PHI output under results dir and gitignore backstop | Partial | src/vista_run/golden_harness.py:227, .gitignore:65 | Default path is under `results_base/golden`, and globs exist; `--out` is unconstrained and can choose a non-ignored in-repo filename. |

## Critical Drift
- High | Plan says Gate 1 includes full-string `dynamic_prompt` as part of legacy-equivalence imaging/structure, while code removes it from the hard structure gate and only compares it under text mode. | Plan: docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:379; harness schema: src/vista_run/golden_harness.py:76; diff schema: src/vista_run/diff_golden.py:38 | Required fix: make the gate schema explicit and consistent. Either put `dynamic_prompt` in `STRUCTURE_FIELDS` for hard equality, or split it into strict non-EHR segments plus an allowlisted EHR segment. Do not leave metadata saying Gate 1 includes `dynamic_prompt` while the actual gate does not.
- High | Plan says golden output is written only under config `results_dir`; code allows arbitrary `--out`, including an in-repo path that may not match `*_golden.jsonl`. | Plan: docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:429; code: src/vista_run/golden_harness.py:252, src/vista_run/golden_harness.py:267; ignore backstop: .gitignore:68 | Required fix: either reject `--out` outside `orch.results_base` or force/validate a `_golden.jsonl` basename and document that `--out` is an escape hatch. The current default is fine; the override violates the PHI contract.

## Missing Pieces
- `diff_golden.py` has no experiment/task/model compatibility preflight; it joins only on `index`. Evidence: src/vista_run/diff_golden.py:66, src/vista_run/diff_golden.py:87. Required before VM use: fail if row metadata disagree on `task`, `experiment`, or `model_type`, so an accidental cross-experiment pair cannot produce a misleading gate report.
- `--out` with multiple `--experiments` overwrites the same file for each experiment. Evidence: src/vista_run/golden_harness.py:247, src/vista_run/golden_harness.py:267. Required fix: reject multi-experiment runs with explicit `--out`, or treat `--out` as a directory/prefix.

## Contract Violations
- `path_tile_count` only counts in-memory list/tuple/ndarray values, not stringified lists. Evidence: src/vista_run/golden_harness.py:164. This is OK for the current `_load_path_task_data` path, which materializes a list at src/context/adapters/pathology.py:95, but it does not satisfy the requested edge-case robustness for `path_tile_paths` as a stringified value.

## Test Gaps
- No structural smoke exists for `diff_golden.py` itself: duplicate index failure, missing/extra index failure, structure drift nonzero exit, strict text drift nonzero exit, allowlist trailing-whitespace pass, and lenient residual text warning behavior. Evidence: src/vista_run/diff_golden.py:66, src/vista_run/diff_golden.py:124, src/vista_run/diff_golden.py:145.
- No local static smoke asserts that `golden_harness.py` rejects retrieval before calling `_build_prompts_for_experiment`. Evidence: src/vista_run/golden_harness.py:95, src/vista_run/run_bq.py:788.

## Defensible Deviations
- The harness instantiates `TaskOrchestrator`, which creates BQ/GCS clients and a model adapter, but it does not load weights. Evidence: src/vista_run/run_bq.py:96, src/vista_run/run_bq.py:105, src/vista_run/run_bq.py:111, src/vista_run/run_bq.py:140. This is consistent with the planner-Mac / executor-VM split and the plan's VM-runnable golden harness.
- Direct `PromptDataset` iteration instead of a `DataLoader` is correct for determinism and still exercises the real `__getitem__` selection path. Evidence: docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md:405, src/vista_run/golden_harness.py:203.

## Suggested Code Edits
- src/vista_run/diff_golden.py:38: align `STRUCTURE_FIELDS` with `golden_harness.GATE1_FIELDS`, or rename/split the fields so `dynamic_prompt` classification is deliberate and documented in both files.
- src/vista_run/golden_harness.py:267: when `args.out` is set, validate it is under `orch.results_base`, or require a `_golden.jsonl` suffix and reject in-repo paths outside `results_base`.
- src/vista_run/golden_harness.py:267: reject `len(experiments) > 1 and args.out` unless `--out` is changed to mean output directory.
- src/vista_run/diff_golden.py:87: after loading rows, compare constant metadata fields (`task`, `experiment`, `model_type`) across files before joining on `index`.
- src/vista_run/golden_harness.py:164: parse stringified `path_tile_paths` only if this harness is expected to tolerate CSV-reloaded path rows; otherwise explicitly assert/list-only and document why.

## Questions For The Author
- Should `dynamic_prompt` be hard byte-identical in Gate 1 for all experiments, or allowlisted only when it embeds a LUMIA-rendered timeline (`path_full`, `report`, normal timeline presets)? The plan currently says both in different places, and the code/meta disagree.
- Is `--lenient` intended to be available in the committed gate runner, or only as a local adjudication aid? Evidence: src/vista_run/diff_golden.py:169. It cannot mask structure drift, but it can return zero despite residual text drift.

## Audit Trail
- Files inspected:
- docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md
- .gitignore
- src/vista_run/golden_harness.py
- src/vista_run/diff_golden.py
- src/vista_run/run_bq.py
- src/vqa_dataset.py
- src/context/adapters/pathology.py

## Resolution (applied 2026-07-06, Claude)
All findings agreed (no disagreements/uncertain). Applied:
- **Critical #1 (gate-schema drift + Q1):** renamed `golden_harness` constants `GATE1_FIELDS/GATE3_FIELDS`
  → `STRUCTURE_FIELDS/TEXT_FIELDS`, now byte-for-byte in lockstep with `diff_golden`. `dynamic_prompt`
  is a TEXT field (mode-dependent): byte-identical under `--mode strict` (Gate 1, passthrough run),
  allowlist-normalised under `--mode allowlist` (Gate 3, LUMIA-live run). Meta now writes
  `structure_fields`/`text_fields`. Comments in both files state the reconciliation.
- **Critical #2 (`--out` PHI bypass):** `--out` must end `_golden.jsonl` (so the `.gitignore` backstop
  always applies) and is rejected when combined with multiple `--experiments` (clobber).
- **Missing (diff preflight):** `diff_golden._check_compatible` aborts unless both dumps agree on
  `task`/`experiment`/`model_type` (constant within each file, equal across the pair).
- **Contract (`path_tile_count`):** documented list-only-by-design (no speculative stringified parse;
  the whole path pipeline assumes a real list).
- **Q2 (`--lenient`):** help text clarified — local Gate-3 adjudication aid, not the committed/VM gate.
- Added a `[WARN]` + non-zero result when the BEFORE/AFTER index intersection is empty (no vacuous pass).

Defensible deviations confirmed (kept): TaskOrchestrator constructs BQ/GCS clients + adapter but loads
NO weights (planner-Mac/executor-VM split); direct `PromptDataset` iteration for determinism.

Not applied here (routed to `/review-tests`): the two Test Gaps (diff_golden gate-logic smoke;
retrieval-reject smoke).
