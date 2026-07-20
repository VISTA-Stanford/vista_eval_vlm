Reference: docs/claude_ops.md

# Feedback: vista_eval_vlm — Step 5 re-plan: window-scope crop for live LUMIA `timeline` rendering (Codex review)

## Verdict

Revise. The core in-repo implementation idea is well bounded and consistent with current code, but the VM handoff is not executable as written: it depends on an unnamed restricted input/config and an ephemeral prior `$CLAUDE_JOB_DIR/tmp/` script, uses a `golden_harness` CLI shape the current code does not support, and has a too-vague residual envelope for the decisive Phase 1 gate.

## Critical Gaps

- Severity: Critical | Gap | Phase 1 command is not runnable against current `golden_harness.py` | Why it matters | A fresh VM executor cannot execute the handoff. The plan uses singular `--experiment` and `--input`, but the harness requires `--config` and exposes plural `--experiments`; there is no `--input` argument. | Evidence: docs/plans/vlm-step5-lumia-window-scope-replan.md:121; docs/plans/vlm-step5-lumia-window-scope-replan.md:124; src/vista_run/golden_harness.py:155; src/vista_run/golden_harness.py:159 | Required fix | Replace the command with the actual CLI, name the exact restricted scratch config or require creating one as a deliverable/VM step.
- Severity: Critical | Gap | Restricted `person_id` artifact is not locatable by a fresh executor | Why it matters | The plan says "reuse, don't regenerate" but gives no path, filename, config, or reconstruction recipe. The 2026-07-18 readback says the run used plain `all_tasks.viewer.vm.yaml`, not a separate restricted CSV, because coverage was 100%. | Evidence: docs/plans/vlm-step5-lumia-window-scope-replan.md:112; docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md:141; docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md:143 | Required fix | STATE-DIRECTLY either "use `configs/all_tasks.viewer.vm.yaml`; no restricted artifact exists/needed because prior banks matched 1238/1238" or provide the exact VM path plus `.meta.json` checks. If the artifact may be absent, provide a self-provisioning recipe.
- Severity: Critical | Gap | Characterization script is ephemeral and not specified in the plan | Why it matters | The canonical spec explicitly warns not to inherit prior `/tmp` scratch. `$CLAUDE_JOB_DIR/tmp/phase1_render_alignment_characterize.py` may not exist for the next executor/session. | Evidence: docs/plans/vlm-step5-lumia-window-scope-replan.md:131; docs/vm-status/2026-07-20-7ed0248.md:307; /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md:217 | Required fix | Paste the fixed script into the handoff section, commit it as a small utility, or state a self-provisioning copy-from-2026-07-20-doc recipe. Do not rely on `$CLAUDE_JOB_DIR/tmp/`.

## Failure Modes

- Scenario | Override config silently never applies or is applied after adapter construction | Why the plan misses it | The intended insertion point is correct only if it lands before `EHRAdapter(config=...)`; the plan should make that a hard target, because `EHRAdapter.__init__` resolves filters immediately. | What to add | STATE-DIRECTLY "mutate `ehr_block['config']['select']` before line `adapter = EHRAdapter(...)`; verify the adapter's filter list includes `window(before='24mo')` or the override."
- Scenario | Phase 1 "residual small" becomes subjective and creates another round trip | Why the plan misses it | The decision gate says "collapses sharply", "small", and "meaningful" without a numeric or shape envelope. | What to add | Pre-declare thresholds in units already produced by the script, e.g. total excess, live/legacy ratio, known-cause counts, and domain buckets that are benign vs STOP.
- Scenario | A missing `embed_time` makes `window()` return full history, preserving the bug | Why the plan misses it | `window()` intentionally returns the input unchanged when `embed_time` is missing, but Phase 1 does not check anchor coverage. | What to add | Add a data-sanity STOP: every Phase 1 row must have non-null parseable `embed_time`; report count only.
- Scenario | Documentation remains inconsistent after changing the live path | Why the plan misses it | The plan updates two docs but not `ehr_filters.py`, whose docstring still says `presets.py` does not carry resolved select chains and describes `timeline` as pending. | What to add | Add `src/context/selectors/ehr_filters.py` docstring cleanup or explicitly explain why it is out of scope.

## Contract Checks

- In-repo preset contract: `no_image` and `axial_all_image` are the only `timeline`-variant presets and currently map through `_ehr_block("timeline")`; `no_report`, `timeline_only`, and `report` use `_ehr_block("no_img_report")`. Evidence: src/context/presets.py:84; src/context/presets.py:88; src/context/presets.py:96.
- In-repo filter contract: `window(before="24mo", after="0d")` is supported by the existing selector; `mo` parses to `pd.DateOffset(months=int(qty))`. Evidence: src/context/selectors/ehr_filters.py:33; src/context/selectors/ehr_filters.py:58; src/context/selectors/ehr_filters.py:65.
- In-repo adapter contract: `_apply_ehr_adapter` must mutate preset config before constructing `EHRAdapter`, because filters are resolved in `EHRAdapter.__init__`. Evidence: src/vista_run/run_bq.py:752; src/vista_run/run_bq.py:753; src/context/adapters/ehr.py:264.
- Legacy-window factual claim is correct: `subsampled_retrieval_csv.py` uses `pd.DateOffset(months=24)` while the adjacent comment still says six months; blame confirms `6a4c2ead` on line 180. Evidence: src/data_tools/csv_helper/subsampled_retrieval_csv.py:171; src/data_tools/csv_helper/subsampled_retrieval_csv.py:180.
- Cross-repo meds2text scope is correctly bounded for this plan. The meds2text transform-stack contract mattered to the prior `VALUE:`/`NOTE:` and `start|end` fixes, but this plan changes only this repo's `select` window configuration. It does not need to restate or re-verify meds2text behavior unless Phase 1 residual analysis reopens render semantics.

## Modularity vs. YAGNI

- Decision point | Config surface only for `timeline` | Plan's current choice | New flat top-level `ehr_timeline_window_before`, leaving `no_img_report`'s `"6mo"` hardcoded. | Modular alternative + realistic use case | A nested per-variant config, e.g. `ehr_windows.timeline.before` and `ehr_windows.no_img_report.before`, would support a near-term production-window-tuning request for `no_report`/`timeline_only` without another plumbing change. | Recommendation | Accept the scoped `timeline` knob for now because Phil's production issue is specifically `no_image`/`axial_all_image`, but add an explicit rationale that `no_img_report` is reproducing a separate legacy cohort contract and should not be tuned until requested.
- Decision point | Where the override lives | Plan's current choice | Read `ehr_timeline_window_before` directly in `_apply_ehr_adapter`. | Modular alternative + realistic use case | Passing full config into `presets.py` would make presets less pure and broader than needed. | Recommendation | Keep the current design. `_ehr_block` is pure and has no config access; `_apply_ehr_adapter` is the right runtime seam.

## Verification Gaps

- Phase 1 needs exact artifact discovery: locate `legacy_small_v2` and `lumia_live_windowed` under `results_dir/golden`, verify `.meta.json` task/experiment/model/tag/limit/config, and STOP on mismatch. The 2026-07-20 doc already proved broad `/mnt` find can time out and used the config-scoped results path instead. Evidence: docs/vm-status/2026-07-20-7ed0248.md:286.
- Add a structural smoke before the expensive bank: construct or inspect one tiny adapter run and assert the resolved filter list contains `window(before='24mo', after='0d')`, and with override set contains that override. This catches config-plumbing mistakes without waiting for a full golden run.
- Add anchor coverage: non-null `embed_time` for all 20 rows, because null anchors bypass windowing. Evidence: src/context/selectors/ehr_filters.py:73.
- Add explicit expected-vs-unexpected envelope: e.g. expected live line count should drop toward the prior `total_legacy_lines=6987`, expected excess should be mostly in known residual buckets, and new domain-wide excess outside known buckets is STOP. Current "collapses sharply" is not enough.
- Add `diff_golden` invocation as `python -m vista_run.diff_golden` from `src` or an equivalent path-known command; bare `diff_golden` is not established in the repo docs.
- Phase 2 should copy the original Step 5 Phase 3 structural checks instead of saying "unchanged" only. This plan is intended to be a fresh handoff source; `/vm-handoff` should not have to chase another plan for mandatory expected/stop lines.

## Handoff Readiness

The implementation section points at the right files and the sister precedent in `no_img_report`, but the handoff is not ready for a fresh executor.

- POINT-AT fix: replace "same restricted `person_id` input" with an exact config/file path or a self-provisioning recipe. If the real prior artifact is just `configs/all_tasks.viewer.vm.yaml`, say that directly and explain why it preserves the index set.
- POINT-AT fix: replace `$CLAUDE_JOB_DIR/tmp/` with a durable source for the characterization script.
- STATE-DIRECTLY fix: spell out actual commands using `--config` and `--experiments`, and include `cd src` where module execution requires it.
- STATE-DIRECTLY fix: name the branch state gate as ancestry-based, not exact short SHA if the plan/doc commit may move HEAD, matching the 2026-07-18 in-lane correction pattern.
- Landing plan is mostly complete, but it should mention updating the superseded render-alignment plan header and `docs/plans/README.md` as part of landing, since those are listed under Files to Modify.

## Verification & Handoff Design

- Archetype selection: correct broad selection is focused VM smoke, schema/contract check for the resolved adapter config, data-sanity checks for line counts/index sets/embed anchors, PHI-clean readback, and human visual QA. The plan has the focused smoke and PHI readback, but misses explicit config-contract and embed-anchor checks.
- Expected-vs-unexpected envelope: incomplete. The plan has a decision gate, but it does not pre-declare numeric bands or domain-shape expectations for "small" vs "meaningful" residual. This is exactly the class-2 envelope the canonical spec expects.
- Handoff phasing: two phases are directionally sound for a complex handoff: cheap byte-diff/characterization first, then human HTML QA. But Phase 1 violates the spec's self-provisioning input rule by depending on prior scratch state, and Phase 2 is too referential to be rendered cleanly without reopening the original plan.
- Machine class: Claude-Code CPU is appropriate. No GPU/high-throughput runner script is needed because this is weight-free rendering plus BQ/GCS/result reads.

## Suggested Revisions

- Rewrite Phase 1 commands with actual `golden_harness.py` CLI: `--config <...> --experiments no_image`; remove `--input` unless the implementation first adds that CLI.
- Add a "Preconditions and artifact discovery" block: results root, exact legacy/live golden glob pattern, required `.meta.json` fields, and STOPs.
- Either paste the fixed characterization script into the plan/handoff section or add it as a planned committed helper. Do not require a prior `$CLAUDE_JOB_DIR/tmp/`.
- Replace "collapses sharply", "small", and "meaningful" with quantitative gates in script output units.
- Add a tiny config-resolution verification for default `24mo` and an override value.
- Add `embed_time` non-null coverage as a data-sanity STOP.
- Include `src/context/selectors/ehr_filters.py` docstring in Files to Modify or state why its stale preset-wiring prose is intentionally left alone.
- Expand Phase 2 with the original Step 5 Phase 3 exact command and Expected/Stop checks so this plan is self-contained.

## Questions For The Author

- Does the prior "restricted input" actually exist as a durable VM artifact, or should this plan explicitly use `configs/all_tasks.viewer.vm.yaml` because prior coverage was 1238/1238 and index sets matched?
- What numeric threshold should resolve Phase 1 inline: maximum residual excess lines, maximum live/legacy ratio, or known-cause-only bucket criteria?

## Audit Trail

- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-step5-lumia-window-scope-replan.md
- docs/vm-status/2026-07-20-7ed0248.md
- docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md
- docs/plans/vlm-step5-lumia-render-alignment-replan.md
- docs/plans/vlm-step5-lumia-live-ehr-adapter.md
- src/context/presets.py
- src/vista_run/run_bq.py
- src/context/selectors/ehr_filters.py
- src/context/adapters/ehr.py
- src/vista_run/golden_harness.py
- src/data_tools/csv_helper/subsampled_retrieval_csv.py
- docs/04-running-the-pipeline.md
- docs/00-data-setup.md
