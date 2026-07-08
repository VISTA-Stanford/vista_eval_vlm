Reference: research-skills/claude_ops.md

# Feedback: Rung 0 — reproduce Ryan on feb26 (Fresh Claude Code review)

## Verdict
**Revise.** The design is substantively sound — the crux experiment-name mapping (OQ-R2) is
**correct and well-evidenced against current code**, the resolver / force-GCS mechanism is
understood correctly, the weighted-vs-weights-free split holds, and the substrate pins verify.
But four concrete issues would trip a fresh VM executor: (1) the ±image delta is **not**
"confound-robust" as claimed; (2) 0c's accuracy derivation is under-specified and depends on a
different task registry than the run; (3) the overlay recipe points at a config file that does not
exist in the repo/history and under-specifies paths; (4) two factual errors in the config-recipe
reasoning. All are plan-edit fixes; none invalidate the approach.

## Critical Gaps

- **Severity: Gap | The ±image delta is not "confound-robust" — the delta itself is confounded, not
  just the absolute accuracy.** The plan (OQ-R2 ⚠ block; 0c) argues the report-inclusion offset is
  "symmetric across both arms" so the ±image delta stays valid and is "the confound-robust signal."
  It isn't. The two compared deltas differ in **two** ways beyond report text:
  (a) **slice count 30 vs 10** — `axial_all_image` loads 30 CT slices
  (`src/context/presets.py:82-85` `_ct_block(30)`; `src/vqa_dataset.py:197` `num_slices = 30`;
  `docs/04-running-the-pipeline.md`), while Ryan's `image_and_timeline` (= raw `no_report`) loads
  **10** (`presets.py:90-93` `_ct_block(10)`; `vqa_dataset.py:240` `num_slices = 10`);
  (b) **report presence** — the current image arm sits on a report-*inclusive* timeline that already
  embeds the radiology report describing that very CT, so the image's *marginal* contribution is
  smaller than when the report is stripped (Ryan's arm). Adding an image to text that already
  narrates the scan ≠ adding it to text that doesn't. The delta measures the image's marginal value
  under different text+dose conditions, so it is **not** apples-to-apples. Evidence:
  `src/vista_run/golden_harness.py:113-116` (`no_report`/`timeline_only` → `use_no_report_csv=True`;
  `no_image`/`axial_all_image` → default `use_no_report_csv=False`); `run_bq.py:201-202`.
  **Required fix:** demote the delta from "confound-robust / more meaningful" to "one weak
  directional signal," and surface the 30-vs-10 slice gap as a *delta* confound in 0c/OQ-R2 (today it
  is buried only in the "tighter numeric reproduce" aside, framed as an absolute-accuracy issue).

- **Severity: Gap | 0c accuracy derivation is under-specified and hits a task-registry mismatch.**
  0c says "Compute accuracy = `mean(predicted_label == ground_truth_label)` … from the new results
  CSVs," but the run's result CSVs contain only `model_response` + the raw `label`
  (`run_bq.py:_build_result_row :806-834`) — **not** `predicted_label`/`ground_truth_label`. Those
  are derived downstream by `src/results/all_model_response.py` via `_extract_answer` +
  `map_answer_to_label_key(mapping)`, where `mapping` comes from
  `base_path/tasks/valid_tasks.json` (`all_model_response.py:69`; `final_metrics.py:233`) — a
  **different file** than the run's registry `valid_tasks_v1_3.json`. On Ryan's v1_1 `base_dir`, if
  `valid_tasks.json` is absent/empty the mapping is `{}` → every `predicted_label` = -1
  (`all_model_response.py:50,57`) → accuracy is garbage, silently. **Required fix:** 0c must (a) name
  the derivation step (regenerate via `python -m results.all_model_response` with the overlay, or
  apply `_extract_answer` + the PFS `mapping` by hand), and (b) require the executor to confirm the
  PFS `mapping` is actually loaded (non-empty), pointing the metrics registry at the v1_3 tasks JSON
  or verifying `valid_tasks.json` exists on `base_dir`.

## Failure Modes

- **Overlay built from the wrong base config → wrong substrate.** 0a and Files-to-modify say "mirror
  the existing `configs/all_tasks.vm.yaml` pattern," but that file **exists neither in the repo nor
  anywhere in git history** (verified). The sibling rebaseline doc describes `.vm.yaml` as pointing
  at `su-vista-*` mounts for **v1_5** — the opposite substrate from rung 0's v1_1 reproduce. An
  executor who "mirrors" it could run the v1.5 mounts. The committed `configs/all_tasks.yaml` is
  precisely Ryan's v1_1 layout (`/data/fries/users/rdcunha/...`, `valid_tasks_v1_3.json`,
  `subsample: true`). Say "start from the committed `all_tasks.yaml`, override
  experiments/task/models/`ct_snapshot_prefix`, unset `ct_dir`" instead.
- **Small-n makes the "~10% band" meaningless.** The baseline has only **544 PFS_1yr rows total**
  across 4 experiments × 4 models (≈34 per cell; image arms likely fewer, since they restrict to
  CT-bearing rows). SE on accuracy at n≈34 is ~8-9% — the whole OQ-R3 ~10% band is within one
  standard error of noise, for both the absolute and the delta. Add an explicit n-caveat so a
  "pass" isn't over-read (and consider whether a higher-n task should back the sanity read —
  `has_recurrence_1_yr` has 912 baseline rows).
- **Decoding-mode mismatch (absolute-accuracy confound).** The committed config sets
  `use_constrained_decoding_for_binary: false` (`all_tasks.yaml:17`). If Ryan's committed baseline
  was generated under a different decoding setting, the decode distribution differs — another reason
  the absolute numbers aren't comparable. Note it (or confirm Ryan's baseline decoding).

## Contract Checks

- **vista_bench (owner) — `vista_bench_v1_1` BQ table + `nifti_path` shape.** In-repo pin verified:
  `VISTA_BENCH_DATASET = "vista_bench_v1_1"` at `src/data_tools/utils/query_utils.py:236`. The stored
  `nifti_path` *shape* (`.nii.gz` vs `.zip` vs bare `study/series`) is the real unknown; the 0a
  shape-classification gate is the right instrument for it. Adequate.
- **vista-ct (owner) — feb26 blob key + feb26 ⊇ nov25 + ~5% orientation-fix.** Correctly labeled
  "not verifiable in this repo"; the 0a blob-existence sweep (stop if < ~100%) is an adequate gate
  for the superset assumption before GPU spend. Adequate as written.
- **In-repo — local-BQ-cache-before-live-BQ.** The plan's 0a note ("`run_bq` reads a local cache
  before querying BQ, so a live-BQ check must bypass it") is **correct and well-caught** —
  `run_bq.py:246-260` reads `resolve_local_bq_cache_path(...)` first, only querying BQ if absent.
- **In-repo — task/prompt registries are VM `base_dir` artifacts, not repo files.** The repo has **no
  `tasks/` directory**; `valid_tasks_v1_3.json`, `prompts_by_task.json`, `image_valid_tasks.json`
  all resolve against `base_dir` on the VM (`run_bq.py:67`). The plan's Background lists
  `valid_tasks_v1_3.json` as a substrate "handle" next to the repo-committed
  `person_id_subsampled.csv` / `all_model_response.csv` — clarify it is a VM artifact so the
  executor doesn't hunt for it in the repo.

## Modularity vs. YAGNI

- **Decision point:** pull the `ct_snapshot_prefix` config seam forward into rung 0 vs a bare
  constant swap. **Plan's choice:** pull it forward. **Assessment: endorse — do not raise to user.**
  Rung 1 / Axis A in the sibling doc explicitly builds on this exact seam
  (`vlm-ct-feb26-v1_5-golden-rebaseline.md` Axis A: "`ct_snapshot_prefix` is a config value … from
  the rung-0 seam"), so it is not throwaway, and it kills the two real fragilities better than a
  constant swap. Note also that the **force-GCS half needs zero code**: unsetting `ct_dir` already
  routes to GCS (`vqa_dataset.py:163-169` — `ct_dir=None → local_path=None → use_local=False →`
  `storage_client` branch). So the actual net-new code is just the prefix read. The plan slightly
  over-describes this as "add a `force_gcs` path" (Files-to-modify) — recommend "honor an unset
  `ct_dir` (already the default GCS route); a `force_gcs` flag is optional."

## Verification Gaps

- **0c derivation recipe (see Critical Gap 2):** add the concrete step to turn raw result CSVs into
  `predicted_label`/`ground_truth_label` and the `valid_tasks.json`-vs-`valid_tasks_v1_3.json`
  mapping check.
- **Overlay path block unspecified:** the plan enumerates the semantic overrides (1 model, PFS,
  `experiments`, `subsample`, dataset, prefix, force-GCS) but not the concrete
  `base_dir`/`results_dir`/`cache_dir` values. State "inherit the committed `all_tasks.yaml` paths,
  unset `ct_dir`" so a fresh executor has a runnable file, not a pattern to reverse-engineer.
- **0a "feb26 blob-existence ≈ 100%" and 0b "`image_count > 0` / `source=gcs` / non-empty resumable
  CSV" are concrete and checkable — good.** No gap there.

## Suggested Revisions

- **Fix factual error (config recipe):** experiments are **not** "all commented out." `- path_full`
  is active at `configs/all_tasks.yaml:83`, so the committed config would run `path_full`, not
  "default to `no_image`." The correct statement (which still supports "won't run the smoke as-is"):
  "the sole active experiment is `path_full` — not the smoke's `no_image`/`axial_all_image` — and the
  5 listed models exceed `MAX_MODELS=4`." (`run_bq.py:178`'s `['no_image']` default only fires when
  `experiments:` is absent/empty, which it isn't.)
- **Fix citation:** `run_bq.py:144` is the docstring line of `_ensure_model_loaded` — the
  weight-*loading* method (called at `run_bq.py:171`) — not the weight-free walker. The weights-free
  path is `golden_harness.py` (iterates `PromptDataset` directly, never calls `_ensure_model_loaded`).
  Re-attribute the "without a GPU or weights" phrase accordingly; the substantive weighted-vs-
  weights-free split is correct.
- **Replace** "mirror the existing `configs/all_tasks.vm.yaml` pattern" with an explicit "start from
  committed `all_tasks.yaml`, override …, unset `ct_dir`."
- **Clarify** `valid_tasks_v1_3.json` is a VM `base_dir` artifact (repo has no `tasks/`).
- **Add** small-n and decoding-mode caveats to 0c.
- **Sharpen** the ±image-delta confound (30-vs-10 slices + report presence) in OQ-R2/0c per Critical
  Gap 1.
- **Housekeeping (non-blocking):** on completion, add a row to `docs/plans/README.md` (no rung-0
  entry today) per claude_ops "After Completing a Plan."

## Questions For The Author

- **Which base config does the rung-0 overlay start from** — Ryan's committed `/data/fries` v1_1
  layout, or the VM `su-vista` `.vm.yaml` (v1_5 mounts)? These are different substrates; rung 0 needs
  v1_1, and the referenced `.vm.yaml` isn't in the repo.
- **Is the PFS comparator's tiny n (~34 per cell) acceptable** for even a "report-only" adjacency
  read, given the ~10% band is within one SE of noise? If not, should a higher-n task
  (`has_recurrence_1_yr`, 912 baseline rows) back the sanity check alongside the canonical PFS smoke?
- **Should 0c compare against `constrained_all_model_response.csv`** (or confirm decoding parity)
  if Ryan's baseline was generated under a different `use_constrained_decoding_for_binary` setting?

## Audit Trail

- docs/plans/vlm-rung0-reproduce-ryan-feb26.md
- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- src/vqa_dataset.py
- src/context/presets.py
- src/vista_run/golden_harness.py
- src/vista_run/run_bq.py
- src/results/all_model_response.py
- src/results/constrained_all_model_response.py
- src/data_tools/OMOP_meds_query/remove_imaging_report.py
- src/data_tools/utils/query_utils.py
- src/data_tools/utils/task_data_utils.py (resolve_local_bq_cache_path)
- src/results/final_metrics.py (mapping registry path)
- configs/all_tasks.yaml
- eval/bq_gcp.sh
- docs/04-running-the-pipeline.md
- figures/results_stats/all_model_response.csv (header + distinct experiment/model/task values)
- figures/data_stats/person_id_subsampled.csv (existence)
- git history: commits 96fd42f (CSV regen), 04b97d8 (30-slice fix); branch worktree-vlm-modular-preprocessing-roadmap @ 5904489
