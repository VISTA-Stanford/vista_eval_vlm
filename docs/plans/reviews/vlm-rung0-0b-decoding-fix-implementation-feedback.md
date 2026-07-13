Reference: research-skills/claude_ops.md

# Implementation Feedback: Rung-0 0b decoding fix

## Verdict
Ready to commit. The uncommitted implementation matches the plan's four seams, fail-closed preflight, and predicate extraction; the only unrun items are the planned GPU/0c runtime checks, which are outside this read-only planner audit.

## Plan Coverage
Seam / section | Status (Done / Partial / Missing / Drifted) | Evidence: path:line | Notes
--- | --- | --- | ---
Seam 1: mapping-based Yes/No decode gate | Done | src/vista_run/run_bq.py:922 | `constrained_choices` is now gated by `is_binary_yes_no_task(mapping)` at src/vista_run/run_bq.py:923-925, not `task_info["is_binary"]`. A 3-class `{"1":"Yes","0":"No","-1":"..."}` mapping passes because the predicate only checks `"1"`/`"0"` at src/results/task_mapping.py:26-30.
Seam 2: `[DECODE]` per-task logging and run summary | Done | src/vista_run/run_bq.py:927 | The gate logs `mode=constrained`/`mode=free` at src/vista_run/run_bq.py:928-934. The ledger dedups by task name at src/vista_run/run_bq.py:935-938, and the summary is emitted once at `run_inference` level after the task/experiment loops at src/vista_run/run_bq.py:237-240.
Seam 3: fail-closed preflight | Done | eval/run_rung0_gpu.sh:121 | The guard resolves `paths.valid_tasks` at eval/run_rung0_gpu.sh:129-130, imports `results.task_mapping.is_binary_yes_no_task` after inserting repo `src` on `sys.path` at eval/run_rung0_gpu.sh:121-124, and exits nonzero for non-Yes/No selected tasks when constrained mode is true at eval/run_rung0_gpu.sh:136-143. It runs before HF auth and weight prefetch at eval/run_rung0_gpu.sh:149-178.
Seam 4a: constrained reducer registry contract | Done | src/results/constrained_all_model_response.py:132 | The reducer reads `cfg["paths"].get("valid_tasks", "tasks/valid_tasks.json")` at src/results/constrained_all_model_response.py:132-135 instead of hardcoding `base_path/tasks/valid_tasks.json`.
Seam 4b: constrained reducer CLI | Done | src/results/constrained_all_model_response.py:292 | `--config` and `--output` are added with defaults that preserve existing behavior at src/results/constrained_all_model_response.py:292-307.
Seam 4c: constrained reducer drop-count print | Done | src/results/constrained_all_model_response.py:266 | The row mask is computed once at src/results/constrained_all_model_response.py:266-267, the actual row count is `int(gt_minus1.sum())` at src/results/constrained_all_model_response.py:281, and the drop uses the same mask at src/results/constrained_all_model_response.py:282-283.
Mirror: `all_model_response.py` registry + CLI | Done | src/results/all_model_response.py:73 | The free-text reducer mirrors registry resolution via `paths.valid_tasks` at src/results/all_model_response.py:73-76 and adds backward-compatible `--config`/`--output` at src/results/all_model_response.py:179-193.
Seam 5: dep-free predicate extraction | Done | src/results/task_mapping.py:17 | New module imports nothing heavy and defines the single predicate at src/results/task_mapping.py:17-30. `final_metrics.py` re-exports it from that module at src/results/final_metrics.py:20-22, and its internal use still resolves at src/results/final_metrics.py:285-288. No duplicate `def is_binary_yes_no_task` remains under `src/` except the new source of truth.
Config/doc updates | Done | configs/all_tasks.rung0.yaml:39 | Rung-0 config documents the mapping gate at configs/all_tasks.rung0.yaml:39-43. OQ-R6 documents the old `is_binary` AND-gate gotcha and cross-links the fix at docs/plans/vlm-rung0-reproduce-ryan-feb26.md:221-228.
QC contract: `predicted_label == -1` checkable | Done | src/results/constrained_all_model_response.py:271 | The constrained reducer prints `[QC] predicted_label == -1 total` at src/results/constrained_all_model_response.py:271-273 before dropping insufficient ground-truth rows.

## Critical Drift
- None.

## Missing Pieces
- None in the implementation scope. Runtime proof remains pending by design: this planner audit did not run the GPU script, reducers, Python checks, or the eval pipeline.

## Contract Violations
- None found.

## Test Gaps
- The negative preflight guard was not executed, so the “guard goes RED before spend” behavior is structurally present but not runtime-proven in this audit.
- The 0c reducer was not run, so the actual `[QC] predicted_label == -1 total: 0` and `Dropped N rows` values remain to be verified on the generated CSVs.

## Defensible Deviations
- The `.isin([-1, "-1"])` change in `constrained_all_model_response.py` is correct and in-lane. `map_label_to_answer()` normalizes labels to registry keys and returns `mapping.get(key, label)` at src/results/results_analyzer.py:279-289; with JSON mappings, `label == -1` maps to the string answer for key `"-1"`. The reducer then maps that answer back through `map_yes_no_to_label()`, which returns the mapping key itself at src/results/constrained_all_model_response.py:114-116, so `ground_truth_label` can be the string `"-1"`. Matching both `-1` and `"-1"` at src/results/constrained_all_model_response.py:266-267 prevents the planned insufficient-follow-up drop from silently dropping zero rows. It does not over-match: it only matches the integer fallback sentinel and the exact JSON string key.

## Suggested Code Edits
- None required before commit.

## Questions For The Author
- None.

## Audit Trail
- Files inspected (paths only)
- docs/plans/vlm-rung0-0b-decoding-fix.md
- docs/plans/vlm-rung0-reproduce-ryan-feb26.md
- configs/all_tasks.rung0.yaml
- eval/run_rung0_gpu.sh
- src/vista_run/run_bq.py
- src/results/task_mapping.py
- src/results/final_metrics.py
- src/results/constrained_all_model_response.py
- src/results/all_model_response.py
- src/results/results_analyzer.py
