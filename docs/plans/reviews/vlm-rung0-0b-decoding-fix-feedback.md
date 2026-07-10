Reference: research-skills/claude_ops.md

# Feedback: Rung-0 0b decoding fix (Codex review)

## Verdict
Revise. The plan correctly identifies the `is_binary` AND-gate bug and chooses the right observable (`[DECODE]` plus residual `predicted_label == -1`), but its "fail-closed" story is still too weak: absent/malformed mapping falls back to free decoding unless the run is actively stopped before inference.

## Critical Gaps
- High | Missing mapping is not actually fail-closed | The proposed code path sets `constrained_choices=None` when mapping is absent/malformed, which is the exact silent no-op class being fixed. A log line plus 0c QC catches it after a bad run; it does not prevent wasting 0b unless the executor tails logs or the code aborts. | Evidence: docs/plans/vlm-rung0-0b-decoding-fix.md:79, docs/plans/vlm-rung0-0b-decoding-fix.md:158, src/vista_run/run_bq.py:905 | Required fix: add an explicit runtime guard/preflight: when `use_constrained_decoding_for_binary` is true for a selected Yes/No-expected task, missing/non-YesNo mapping must raise before inference, or the GPU handoff must run a cheap mapping preflight before `eval/run_rung0_gpu.sh`.
- High | Run registry and reducer registry are only "same file" for the current overlay, not by code contract | `run_bq.py` loads `self.cfg['paths']['valid_tasks']`, while both reducers hard-code `base_path / "tasks" / "valid_tasks.json"`. The plan says they are the same file, but that is contingent on `configs/all_tasks.rung0.yaml`; a newer registry cut or config change can make 0b gate on one mapping and 0c score with another. | Evidence: src/vista_run/run_bq.py:67, src/results/constrained_all_model_response.py:127, src/results/all_model_response.py:69, configs/all_tasks.rung0.yaml:29 | Required fix: state this contract explicitly and either update reducers to honor `paths.valid_tasks` from config or add a Phase-2/Phase-3 check that prints and compares both resolved registry paths plus the PFS mapping shape.
- Medium | Phase-3 relies on a misleading drop-count print | `constrained_all_model_response.py` computes grouped counts, then prints `len(minus_one_count_gt)`, which is number of model groups, not number of rows dropped. The plan expects "`Dropped ... rows`" to support the 27% drop QC. | Evidence: src/results/constrained_all_model_response.py:257, src/results/constrained_all_model_response.py:260, src/results/constrained_all_model_response.py:261, docs/plans/vlm-rung0-0b-decoding-fix.md:169 | Required fix: either fix the reducer to print the actual row count before/after filtering, or make the handoff compute `ground_truth_label == -1` count from the output/intermediate dataframe rather than trusting that line.
- Medium | Fresh rerun cleanup only names result CSVs, not derived report artifacts | Moving result CSVs aside is necessary because resume is by `index`, but stale `figures/results_stats/constrained_all_model_response.csv` or `all_model_response.csv` can survive if 0c finds no files or is accidentally pointed at the wrong config/output. | Evidence: src/vista_run/run_bq.py:713, src/results/constrained_all_model_response.py:134, src/results/constrained_all_model_response.py:263 | Required fix: require Phase 3 to write to a rung-0-specific `--output` path or move/mtime-check existing `figures/results_stats/*all_model_response.csv` before 0c.

## Failure Modes
- Scenario | Why the plan misses it | What to add
- `task_info["mapping"]` absent on the VM registry | The proposed predicate returns false and the run proceeds unconstrained; `[DECODE] mode=free` is only useful if checked before inference completes. | Add a pre-GPU mapping assertion or raise in `run_bq` for configured constrained rung-0 tasks.
- 0b uses `paths.valid_tasks`, 0c reads hard-coded `tasks/valid_tasks.json` | The current rung-0 overlay happens to set that path, but source code does not guarantee it. | Make reducers use `paths.valid_tasks` or log/compare registry path and mapping in both 0b and 0c.
- Reducer command runs with wrong `--config` or no matching files | `constrained_all_model_response.py` returns after "No result files found" without deleting old output. | Gate on result-file count, output mtime after command start, and expected task/model/experiment rows.
- Structured decoding yields `" Yes"`/`"No\n"` | This is handled; Gemma strips text before storing, and reducer strips/case-folds. | Keep the spot-check, but count exact allowed stripped values rather than only eyeballing samples.
- Structured decoding yields empty/non-YesNo despite `[DECODE] mode=constrained` | Reducer maps to `-1`, and Phase-3 `predicted_label == -1` catches it. | Make "materially > 0" concrete: for constrained PFS, any residual `predicted_label == -1` should STOP unless a named known vLLM edge is documented.

## Contract Checks
- In-repo: `run_bq.py` currently loads raw task entries from `paths.valid_tasks`; `task_info` can carry `mapping` if the VM JSON carries it, but the repo does not contain that JSON, so this is a VM-side contract needing an explicit preflight.
- In-repo: `is_binary_yes_no_task(mapping)` returns True for `{"1":"Yes","0":"No","-1":"Insufficient..."}` because it only checks keys `"1"` and `"0"`. That part of the plan is correct.
- In-repo: importing `is_binary_yes_no_task` from `final_metrics.py` would pull `numpy`, `pandas`, `yaml`, `results.results_analyzer`, and `context.normalize` into the inference path. The plan is right to treat direct import as risky.
- Cross-repo/VM: staged `valid_tasks.json` from sibling `vista_bench` must preserve PFS mapping keys `"1"`, `"0"`, `"-1"` with Yes/No/Insufficient semantics. The plan should specify what to do if a newer registry changes `is_binary`, key types, or mapping shape.
- Repo-local checklist: none was provided; note that a repo-grounded checklist would sharpen exact VM recipes, expected row counts, and registry-path contracts.

## Modularity vs. YAGNI
- Decision point | Plan's choice | Modular alternative + realistic use case | Recommendation, or "raise to user".
- Yes/No predicate placement | Lean inline in `run_bq`, referencing `final_metrics.py` | Extract to a light module such as `results/task_mapping.py` or `data_tools/utils/task_mapping.py`; useful immediately because inference and metrics both need the same contract without importing reporting dependencies. | Raise to user only if they want minimal file churn; my recommendation is extract now because this bug came from duplicated/implicit task semantics.
- Constraint breadth | Off-mapping general constraint | Per-task allowlist avoids changing future Yes/No-mappable 3-class tasks unintentionally. | Off-mapping is reasonable for Ryan parity, but require a log/count of all constrained tasks in the run so breadth is observable.
- Reducer CLI | Duplicate `argparse --config/--output` in two modules | Shared tiny CLI helper for result reducers; useful if more reducers already exist (`final_metrics.py` has same default-only pattern). | For this fix, duplicated boilerplate is acceptable if only two modules are touched; if `final_metrics.py` also gets CLI cleanup, extract a helper.

## Verification Gaps
- Add a cheap pre-0b mapping gate that reads the exact VM `base_dir` and config, prints the resolved valid-tasks path, and asserts PFS mapping has `"1": "Yes"` and `"0": "No"` before model load.
- Make `[DECODE] mode=free` a code-level STOP for this rung-0 run, not merely a readback finding after inference.
- Gate on exact result row counts per arm after moving CSVs aside; prior VM handoff already warned batch errors can be swallowed, so non-empty CSVs are insufficient.
- Require zero `Error in batch` and zero `Producer error` in the log; the target plan says "inherit prior Stops" but omits these prior silent-drop traps from the latest Phase 2.
- Require output freshness for 0c: output file path, mtime after reducer start, expected task/model/experiment labels, and no stale default-config artifact.

## Verification & Handoff Design
- Archetypes: the plan selects the right silent-fallback trap (`[DECODE]`) and metric-sanity trap (`predicted_label == -1`), but it misses the canonical "guard that must go RED" archetype. Add a cheap negative check with a malformed mapping or disabled flag proving the guard/log/abort path fires.
- Expected-vs-unexpected envelope: "≈0" and "materially > 0" are too loose for a constrained Yes/No direct map. Expected should be exactly zero residual `predicted_label == -1` after stripping/case-folding, with any nonzero count a STOP unless pre-declared.
- Phasing: this is complex-tier by the canonical spec because it has multiple phases, class-2 decisions, banked prior context, and destructive/stale-artifact moves. The plan should use the full phase schema fields (`purpose`, `banked-from-prior`, `gates`, `destructive?`, `stop/deviation`, `next-doc trigger`) rather than prose-only phasing.
- Cheap-to-expensive ordering: Phase 1's local AST check is fine, but it must not run Python on the planner Mac if repo standards say no code execution on this machine. Use structural read-only review locally; run AST/import checks on the VM or make them optional executor checks.

## Suggested Revisions
- Replace "if mapping absent/malformed, constrained_choices=None" with a real fail-closed behavior for rung-0: raise before inference or add a preflight that must pass before `eval/run_rung0_gpu.sh`.
- Add a registry-path contract section: 0b and 0c must resolve the same task registry path, or reducers must honor `paths.valid_tasks`.
- Update Phase 2 to include prior silent-drop log gates: `Error in batch == 0`, `Producer error == 0`, expected row counts per arm, and no empty `model_response`.
- Update Phase 3 to use `--output` with a rung-0-specific artifact and require output freshness plus task/model/experiment row counts.
- Fix or avoid the misleading `Dropped {len(minus_one_count_gt)}` reducer print.
- Make `predicted_label == -1` STOP threshold exact unless the author documents a known structured-output edge.
- Convert handoff phasing to the canonical complex-phase schema and explicitly mark the CSV move-aside as destructive/stale-artifact handling.

## Questions For The Author
- Should a selected rung-0 Yes/No task with malformed/missing mapping abort in `run_bq`, or should this remain a VM preflight-only guard?
- Do you want a lightweight shared mapping helper now, or is inline duplication acceptable for this one hot-path fix?
- Should reducer outputs for this rung stay in `figures/results_stats/`, or should 0c write a separate rung-0 output path to avoid clobbering/staleness around committed baseline files?

## Audit Trail
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-rung0-reproduce-ryan-feb26.md
- docs/plans/vlm-rung0-0b-decoding-fix.md
- src/vista_run/run_bq.py
- src/results/final_metrics.py
- src/results/constrained_all_model_response.py
- src/results/all_model_response.py
- src/models/gemma3.py
- configs/all_tasks.rung0.yaml
- docs/vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md
