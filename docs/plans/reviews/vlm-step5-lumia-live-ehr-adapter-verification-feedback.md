Reference: docs/claude_ops.md

# Feedback: vista_eval_vlm Step 5 — LUMIA-live EHR adapter (Codex verification & handoff-design review)

## Verdict
Revise. The four-phase shape is directionally right, but the VM handoff is not yet self-executing: Phase 0's cohort source/threshold envelope, Phase 1's "covered sample" mechanics, Phase 2's allowlist guardrails, and Phase 3's structural QA all leave executor judgment that the canonical spec says should be pre-declared.

## Archetype Coverage
- Selected correctly: data-sanity / coverage precondition (Phase 0), before/after parity via two-checkout golden diff (Phase 1), migration/re-anchoring diff with declared text deltas (Phase 2), PHI-clean human visual QA readback (Phase 3). These match the spec's data-sanity, before/after refactor parity, pre-declared diff-envelope, and PHI-clean readback archetypes (`../research-skills/references/verification-and-handoff-design.md:49`, `../research-skills/references/verification-and-handoff-design.md:53`, `../research-skills/references/verification-and-handoff-design.md:85`, `../research-skills/references/verification-and-handoff-design.md:106`).
- Missing: a performance/runtime sanity check for replacing a CSV timeline read with per-row XML parse/render. The plan explicitly changes the hot path to parse each patient's LUMIA XML live (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:27`, `docs/plans/vlm-step5-lumia-live-ehr-adapter.md:88`), but the verification only smokes N~20-50 plus N~5 renders. Claude-Code CPU is fine for the smoke, but the plan should require wall-clock/per-row timing on Phase 2 and a scale caveat before full eval.
- Missing: a downstream reporting contract for dropped rows. The plan says 80-95% coverage may proceed if dropped-N is called out wherever results are reported (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:203`), but it does not name the downstream result/report/leaderboard surface that must display this. `run_bq._build_result_row` only emits per-example result rows and token counts, not cohort/drop metadata (`src/vista_run/run_bq.py:816`), so this can still become invisible after the handoff.
- Missing: a guard that the new fail-closed path actually goes red. The spec calls out proving fail-closed guards fire (`../research-skills/references/verification-and-handoff-design.md:75`). The plan asserts missing `paths.lumia_corpus_dir` should raise (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:123`), but no phase runs a tiny negative check for unset/bad corpus dir.

## Expected/Unexpected Envelope Gaps
- Phase 0: the thresholds need exact boundary language and exact units. As written, ">=95", "80-95", and "<80" makes exactly 95% overlap both buckets and leaves exactly 80% only implicitly accepted (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:202`). Revise to `coverage_pct >= 95.0 proceed`, `80.0 <= coverage_pct < 95.0 proceed-with-drop-reporting`, `<80.0 STOP`.
- Phase 0: the cohort recipe is not executable enough. It says use the exact task CSV set, but the SQL shown is the BQ table plus a comment to restrict to the subsampled cohort (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:185`, `docs/plans/vlm-step5-lumia-live-ehr-adapter.md:190`). Existing loading actually merges BQ rows with a local task CSV (`src/vista_run/run_bq.py:287`, `src/vista_run/run_bq.py:313`). Add STOP if the BQ-derived row count, local CSV row count, and golden-harness loaded row count disagree, and name which one is authoritative for coverage.
- Phase 1: the "sample N~20 from Phase 0's covered set" is not currently guaranteed by `golden_harness --limit 20`. The harness applies `--limit` only after `_build_prompts_for_experiment`, stable sort, and `head(limit)` (`src/vista_run/context_capture.py:190`, `src/vista_run/context_capture.py:195`, `src/vista_run/context_capture.py:198`). The plan needs either a concrete VM-local filtered config/CSV for covered IDs or an explicit check that the deterministic first 20 after fail-closed filtering are all in the Phase 0 covered set.
- Phase 1: the self-resolving gate hides setup preconditions. The precedent handoff spells out throwaway worktree creation, absolute config path use, expected row_count/index-set match, and cleanup (`docs/vm-status/2026-07-15-phase2-config-context-viewer.md:66`, `docs/vm-status/2026-07-15-phase2-config-context-viewer.md:84`, `docs/vm-status/2026-07-15-phase2-config-context-viewer.md:93`). Add those here, plus STOP if the parent SHA cannot construct with the same VM config or if before/after row counts/index sets differ.
- Phase 1: "whichever configuration passes byte-identical is the answer" is objective only if exactly one passes. Add the tie case: if both `select=[]` and `code_filter` pass strict, choose the narrower/no-filter preset only if the rendered timelines are byte-identical between the two after-banks; otherwise STOP because the decision gate is not discriminating.
- Phase 2: the allowlist is too open-ended. `diff_golden --mode allowlist` only passes text deltas after `normalize_text`, which currently strips trailing whitespace only (`src/vista_run/diff_golden.py:57`, `src/vista_run/diff_golden.py:143`). "Codify whichever regex/normalization actually appears" (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:239`) lets the executor invent a loose regex at VM time. Require regexes to be field-local and event-line-local, preserve line count/order, preserve code/description/note text byte-for-byte, parse numeric VALUE equivalence with a bounded tolerance, and explicitly forbid catch-all substitutions.
- Phase 2: final gate should prohibit `--lenient`. The tool can exit 0 despite residual text drift under `--lenient` (`src/vista_run/diff_golden.py:206`), so the handoff should state `--lenient` is allowed only for local inspection while editing the normalizer; the committed final VM gate must run without it and print `RESULT: ALL GATES PASS`.
- Phase 3: "HTML exists, non-empty, self-contained, 5 cards" is too thin for the human-QA step (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:265`). The prior viewer handoff checked no external refs, exact card count, image count expectations, token bars, and no weights/GPU (`docs/vm-status/2026-07-15-phase2-config-context-viewer.md:118`). Add executor-side structural checks that every card has non-empty prompt/timeline text, non-empty token count/bar, expected no-image vs CT image counts where applicable, and no `STOP:`/traceback text in the HTML.

## Handoff Phasing
- Ordering is mostly sound: cheap coverage gate first, then small strict diff, then allowlist fill-in, then human render. It follows cheap-to-expensive sequencing (`../research-skills/references/verification-and-handoff-design.md:136`).
- Bank-forward correctness needs tightening. Phase 2 says extend the Phase 1 bank "don't rebank from scratch" (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:231`), but `golden_harness` writes one output per tag and limit (`src/vista_run/golden_harness.py:139`, `src/vista_run/golden_harness.py:161`); changing N from 20 to 50 is a new input. The plan should say Phase 1's parent worktree/config decision is banked, not the 20-row file itself, unless the exact same N is reused.
- Stop-vs-decision-gate clarity is mixed. Phase 1's neither-passes path is a clear class-3 STOP (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:226`), but Phase 2's "declared-delta fraction gets so pervasive" is intentionally unthresholded (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:243`). Either give a concrete report-only rule or make it an explicit STOP requiring planner/Phil review.
- Machine-class claim is acceptable for the planned smoke. The harness and viewer are documented weight-free and avoid model weights (`src/vista_run/golden_harness.py:26`, `src/results/context_viewer.py:535`), and the Claude-Code CPU class is intended for BigQuery/GCS queries and moderate eval runs (`../research-skills/references/verification-and-handoff-design.md:180`). Add the runtime caveat above for scaling beyond smoke.

## Destructive-Write / Rollback
- No rollback plan is needed for the verification phases as written. The plan reads BigQuery/GCS and writes local golden/context-view artifacts under `results_dir`; the harness and viewer both document PHI outputs outside the repo / git-ignored (`src/vista_run/golden_harness.py:39`, `src/results/context_viewer.py:488`), and `.gitignore` has the defensive globs (`.gitignore:67`, `.gitignore:74`).
- Still state `destructive? no` per phase in the required phasing schema. The canonical schema asks for this field explicitly (`../research-skills/references/verification-and-handoff-design.md:262`), and the current plan uses prose phases rather than the full schema.

## Multi-Target / Cross-Repo Ripple
- No cross-repo dependency is apparent from the plan. The behavioral contract change is inside `vista_eval_vlm`: five presets shrink their eval cohorts when LUMIA XML is missing (`docs/plans/vlm-step5-lumia-live-ehr-adapter.md:106`, `docs/plans/vlm-step5-lumia-live-ehr-adapter.md:203`).
- The in-repo ripple is real: downstream result aggregators consume result CSVs under `results_dir`, and `_build_result_row` drops heavy timeline columns without adding a cohort-denominator/drop-count field (`src/vista_run/run_bq.py:816`, `src/vista_run/run_bq.py:839`). The plan should flag that any report comparing these presets to historical full-cohort results must carry the post-drop denominator.

## Suggested Revisions
- Add a compact phase schema for each phase: purpose, machine, banked-from-prior, gates, destructive?, stop/deviation, next-doc trigger, matching the canonical spec (`../research-skills/references/verification-and-handoff-design.md:262`).
- Phase 0: replace the SQL/comment sketch with an exact VM recipe for deriving the golden-iterated cohort; define boundary thresholds exactly; STOP on source row-count disagreement; report matched/missing/coverage only.
- Phase 1: spell out parent SHA/worktree/config setup; require same task/experiment/model/N and same index set; define covered-sample mechanics despite `--limit`; add both-pass and parent-bank-failure outcomes.
- Phase 2: predeclare normalizer guardrails and final no-`--lenient` gate; require at least one reject case showing a forbidden event-order/content drift still fails after the normalizer.
- Phase 3: copy the stronger structural checks from the 2026-07-15 viewer precedent: no external refs/local paths, exact card count, expected image/no-image counts, non-empty rendered text regions, token bars/counts, no weights/GPU, no traceback/STOP text.
- Add one small negative check for unset/bad `paths.lumia_corpus_dir` to prove the fail-closed guard fires.
- Add per-row/wall-clock timing readback for Phase 2/3 and require the downstream result/report path to surface dropped-N and post-drop denominator for wired presets.

## Questions For The Author
- Which artifact is authoritative for Phase 0 coverage: the local task CSV after `_load_task_data` merge, the BQ table, or the golden-harness produced row_count?
- Where exactly should dropped-N/post-drop denominator be surfaced for later comparisons: console logs only, result CSV metadata, generated HTML, or a separate run manifest?

## Audit Trail
- ../research-skills/claude_ops.md
- ../research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-step5-lumia-live-ehr-adapter.md
- src/vista_run/diff_golden.py
- src/vista_run/golden_harness.py
- src/vista_run/context_capture.py
- src/vista_run/run_bq.py
- src/results/context_viewer.py
- docs/vm-status/2026-07-06-golden-harness.md
- docs/vm-status/2026-07-15-phase2-config-context-viewer.md
- .gitignore
