Reference: docs/claude_ops.md

# Feedback: VLM CT feb26/v1_5 golden rebaseline + rung-0 reproduce-Ryan (Codex review)

## Verdict
Revise. The rung order is basically sound and the weighted-vs-golden split is correctly identified, but rung 0 is not yet handoff-ready: the CT prefix change can be masked or no-op, the executor recipe does not actually select the requested model/task/experiments from the committed config, and "Ryan-adjacent" is not operational enough for VM pass/fail.

## Critical Gaps
- High | Rung 0 can silently fail to exercise the feb26 GCS reroute | `PromptDataset` checks local `ct_dir` before GCS; if the nov25 local cache exists, the run uses local nov25 bytes based only on filename and never proves the feb26 prefix/blob path works. This contradicts "reproduce Ryan on feb26" unless the intent is "exact Ryan if cache exists, feb26 only if cache missing." | Evidence: `src/vqa_dataset.py:160-173`, `configs/all_tasks.yaml:4`, plan `docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:105-108` | Required fix: make rung 0 choose and document one mode: either force feb26 GCS by disabling/renaming `ct_dir` for the smoke, or explicitly call the local-cache path "Ryan exact, not feb26" and add a separate VM blob-existence/load check for the v1_1-selected UIDs.
- High | Prefix swap can be a literal no-op for prefixed `nifti_path` values | The helper strips `/mnt/` and bucket name, then returns immediately when `path_str.startswith(prefix)`. If the stored value already starts with the current nov25 prefix, changing the default prefix to feb26 means the stored nov25 path no longer starts with the new prefix; the later fallback may build a malformed `{prefix}/{.../nov25/...}.nii.gz` depending on path shape, not simply rewrite it. If the code instead leaves the prefix as nov25 and only edits constants elsewhere, it uses nov25 as-is. The plan's "also rewrite embedded nov25->feb26" caveat is necessary but too underspecified for a VM executor. | Evidence: `src/vqa_dataset.py:18-43`, plan `docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:95-103` | Required fix: specify the exact row-shape cases and the exact code behavior for each: bare filename, `{study}/{series}`, bucket-relative prefixed nov25, bucket-qualified prefixed nov25, `/mnt/...` prefixed nov25. Require logging sampled `(input nifti_path form, blob_path, filename, source=local|gcs)` counts.
- High | Step 0's run recipe does not select the stated model/task/experiments | `eval/bq_gcp.sh` reads `configs/all_tasks.yaml`; the committed config currently has five models, `has_recurrence_1_yr`, and `path_full`, while PFS and `no_image`/`axial_all_image` are commented out. `MAX_MODELS=4` would also reject the current five-model config. | Evidence: `eval/bq_gcp.sh:7-8`, `eval/bq_gcp.sh:21-35`, `eval/bq_gcp.sh:110-113`, `configs/all_tasks.yaml:19-40`, `configs/all_tasks.yaml:67-83`, README `README.md:47-55` | Required fix: add a concrete VM-only config overlay or temporary edit recipe for rung 0: one model `gemma3/google/medgemma-1.5-4b-it`, task `progression_recurrence_free_survival_1_yr`, experiments `no_image` and `axial_all_image`, `subsample: true`, v1_1 dataset/task registry. State whether this overlay is committed, uncommitted VM-local, or supplied via a copied config path.
- High | "Ryan-adjacent band" is not a checkable Expected/Stop criterion | `claude_ops.md` requires metric ranges or concrete stop criteria. The baseline file is row-level predictions, not an accuracy summary, and uses experiment labels like `image_only`, not `axial_all_image`; quick inspection did not find `no_image` labels. A VM executor cannot know what "close" means or which rows to compare. | Evidence: `research-skills/claude_ops.md:116-124`, plan `docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:116-121` and `:286-289`, `figures/results_stats/all_model_response.csv` header/labels | Required fix: define the comparison procedure: exact filter keys, experiment-name mapping (`axial_all_image` vs `image_only`, `no_image` vs timeline-only/report-only if applicable), accuracy formula, expected baseline values, acceptable tolerance, and how to handle stochastic decode/resume rows.
- Medium | Data facts are presented as accepted assumptions but need stronger assumption labeling in Step 0 | `feb26 ⊇ nov25 keyed by (study, series)` and "~5% orientation fixed" are cross-repo/data claims not verifiable from this repo. The plan says Phil/changelog, and it is honest that the reproduce is close not exact, but Step 0's Expected still relies on the superset as if established by code. | Evidence: plan `docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:97-101`, `:286-290`; no in-repo code validates this | Required fix: promote these to named assumptions in Step 0 and make the first VM UID/blob existence sweep the validation gate before weighted inference.

## Failure Modes
- Scenario | The local nov25 cache exists and all CT rows load locally | Why the plan misses it: it treats cache survival as "exact" but still names the rung "on feb26"; the prefix/GCS code may never be exercised. | What to add: source-of-image logging and either force-GCS mode or a separate feb26 blob load smoke.
- Scenario | v1_1 `nifti_path` values are fully prefixed nov25 paths | Why the plan misses it: it says "small substring rewrite" but does not define whether to rewrite the input string before helper parsing, alter the helper, or normalize via structured UID extraction. | What to add: a VM preflight query that prints counts by path form and a deterministic normalization table for each form.
- Scenario | `eval/bq_gcp.sh` runs the wrong config surface | Why the plan misses it: the plan names the desired run but not how to make the script run it. | What to add: a runnable command/config snippet and a check that the log lists exactly one model, PFS, `no_image`, and `axial_all_image`.
- Scenario | The Ryan baseline comparison filters the wrong rows | Why the plan misses it: committed Ryan output uses legacy experiment names and row-level predictions. | What to add: a baseline extraction recipe and expected two baseline accuracies, or explicitly downgrade Step 0 to "completed run only" with no closeness criterion.
- Scenario | BigQuery queryability is not tested because local BQ cache is present | Why the plan misses it: `run_bq.py` reads local cache before querying BQ. | What to add: an explicit VM queryability check for `vista_bench_v1_1` independent of the weighted run, or require cache bypass for the smoke.

## Contract Checks
- In-repo `PromptDataset` CT contract: `nifti_path` string → `(blob_path, filename)`; local `ct_dir` wins before GCS; `axial_all_image` samples 30 CT slices. Covered, but rung 0 needs source logging because local-first changes what is verified. Evidence: `src/vqa_dataset.py:154-210`.
- In-repo query contract: four CT helpers currently select/filter `nifti_path`; Axis A correctly names these as v1.5 retarget points. Evidence: `src/data_tools/utils/query_utils.py:156-236`.
- In-repo weighted eval contract: `run_bq.py` loads weights at `run_inference()` and writes results through the normal pipeline; README documents `eval/bq_gcp.sh -> run_bq.py`. Covered conceptually. Evidence: `src/vista_run/run_bq.py:141-171`, README `README.md:47-55`, `README.md:85-90`.
- In-repo golden contract: `golden_harness.py` is weights-free and captures inputs/images only. Covered correctly. Evidence: `src/vista_run/golden_harness.py:28-33`, `src/vista_run/golden_harness.py:186-209`.
- Cross-repo `vista_bench` contract: v1_1 has usable `nifti_path`; v1_5 has `image_study_uid`/`image_series_uid`, stable cohort table name, valid task registry, and column names. The plan addresses the major contract, but Step 0's v1_1 queryability and rung 1's v1_5 column/version pins need VM checks stated as commands/queries.
- Cross-repo `vista-ct` contract: feb26 storage key is `chaudhari_lab/ct_data/ct_scans/vista/feb26/{study}__{series}.nii.gz`; feb26 is a superset of nov25 by `(study, series)`; orientation fix affects ~5%. The plan names these, but they remain assumptions until VM blob checks validate them.
- Output contract: weighted results path is `{results_dir}/{source_csv}/{task_name}/{model_name}/{task_name}_results_{experiment}.csv`; plan names this but should include exact expected files for the two rung-0 experiments and expected non-empty row counts.

## Modularity vs. YAGNI
- Decision point | Rung 0 bare constant/substr rewrite vs rung 1 config-driven `ct_snapshot_prefix`.
- Plan's choice | Keep rung 0 as a minimal Ryan-substrate perturbation, then introduce config in Axis A.
- Modular alternative + realistic use case | Add `ct_snapshot_prefix` before rung 0 and set it to feb26 for both the Ryan reproduce and v1.5 cut, avoiding a one-off constant edit and making future re-materializations uniform.
- Recommendation | Raise to user. The split is coherent if rung 0 is intentionally a disposable, few-line smoke on legacy code. It is less coherent if the executor will iterate on path normalization and source logging anyway; at that point using the config seam earlier may reduce throwaway logic.

## Verification Gaps
- Step 0 needs a concrete preflight recipe: query a small v1_1 PFS sample, classify `nifti_path` shapes, derive `(study, series)`, check corresponding feb26 blobs, and report counts before weighted inference.
- Step 0 needs an image-source check: for `axial_all_image`, report CT rows loaded from local `ct_dir` vs GCS and fail/branch according to the chosen mode.
- Step 0 needs exact run selection: either a VM-local config path or explicit temporary config changes, plus log checks proving one model, PFS, and the two requested experiments ran.
- Step 0 needs an operational baseline comparison: map legacy experiment names in `all_model_response.csv`, compute accuracy from `predicted_label == ground_truth_label`, state expected values and tolerance, and include row-count expectations.
- Step 1 should state the VM command or harness invocation that proves `nifti_path` is never read after Axis A/B; "never read" is otherwise a code-property claim, not an output-vs-input behavioral check.
- Step 1b's "few examples (UIDs-as-structure)" should specify PHI-safe formatting and max count.
- Steps 3-5 are stronger: golden byte-identity on `selected_indices`, `image_count`, `assembly_mode`, and `image_hashes` is the right behavioral gate for the 3b CT dissolution. Avoid adding substring assertions on generated source.

## Suggested Revisions
- Add a "Rung 0 VM recipe" block with exact config overlay, command, expected output files, log assertions, and baseline calculation.
- Replace "few-line CT prefix repoint" with "minimal legacy resolver patch" and enumerate path-shape handling; keep the simple constant swap only for the bare filename/non-prefixed case.
- Decide whether rung 0 is "feb26 GCS" or "Ryan exact local cache if present"; if both are allowed, make them explicit branches with different success labels.
- Move the superset/orientation facts into a named "Assumptions not verifiable in this repo" subsection and require VM blob-count validation before running weights.
- Add a local-cache/BQ-cache precondition: either bypass caches when the goal is to test live services, or separately test live BQ/GCS before using caches for inference.
- Add exact Ryan baseline rows/labels to compare, including the legacy experiment-name mapping.
- Clarify that the from-branch run is "Ryan + declared deltas": the 30-slice even-spacing fix is on the `axial_all_image` path (`src/vqa_dataset.py:188-210`) and therefore can perturb the CT reproduce.

## Questions For The Author
- Should rung 0 force feb26 GCS even if the local nov25 cache exists, or is an exact local-cache Ryan reproduce acceptable as a separate outcome?
- What is the intended mapping between current experiments (`no_image`, `axial_all_image`) and Ryan's committed labels (`timeline_only`, `image_only`, `image_and_timeline`, `report_and_timeline`)?
- What numeric tolerance defines "Ryan-adjacent" for the two accuracies?
- If the v1_1 `nifti_path` values are prefixed paths, should the executor implement a one-off substring rewrite for rung 0 or skip directly to the structured UID resolver?

## Audit Trail
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- src/vqa_dataset.py
- src/data_tools/utils/query_utils.py
- src/vista_run/run_bq.py
- src/vista_run/golden_harness.py
- configs/all_tasks.yaml
- README.md
- eval/bq_gcp.sh
- figures/results_stats/all_model_response.csv
