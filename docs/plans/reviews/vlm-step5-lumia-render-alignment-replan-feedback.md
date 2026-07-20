Reference: docs/claude_ops.md

# Feedback: vista_eval_vlm Step 5 re-plan #2 — render alignment (Codex review)

## Verdict
Revise. The core diagnosis is well grounded in current `meds2text` and `vista_eval_vlm` code, but the plan is not yet handoff-ready: the complex VM handoff is under-phased, and the numeric reconstruction fix overstates byte-identity guarantees for zero/rounded/ontology-overridden values.

## Critical Gaps
- Severity: Critical | Gap: The Verification section spans a pre-implementation decision and post-decision implementation/re-run in one A→B sequence without the required complex-tier Handoff phasing schema | Why it matters: Step A can change the code shape for `diff_golden.py`, so Step B cannot be rendered as the same runnable handoff at one known SHA; the canonical spec says split or gate checks whose implementation depends on unresolved evidence, and complex handoffs must state phase purpose, machine, banked inputs, gates, stop routing, and next-doc trigger | Evidence: docs/plans/vlm-step5-lumia-render-alignment-replan.md:153; docs/plans/vlm-step5-lumia-render-alignment-replan.md:170; ../research-skills/references/verification-and-handoff-design.md:132; ../research-skills/references/verification-and-handoff-design.md:262 | Required fix: Add an explicit `Handoff phasing` block. Phase 1 should be Step A only, read-only, banked from the 2026-07-18 goldens, with a next-doc trigger. Phase 2 should run after the Mac lands the `ehr.py` fixes and any Step-A-resolved `diff_golden.py` mechanism, at a named new SHA.
- Severity: Critical | Gap: The numeric parse assumes every genuine numeric MEDS value survives into XML element text, but `event_to_xml` drops falsy numeric zero before formatting | Why it matters: legacy renders `numeric_value == 0` as `VALUE: 0.0`; live XML may have no element text for numeric zero, so the proposed parser cannot reconstruct it and byte-identity can still fail on zero-valued labs/measurements | Evidence: ../meds2text/src/meds2text/textify.py:1283; ../meds2text/src/meds2text/textify.py:1319; src/data_tools/utils/meds_timeline_utils.py:222 | Required fix: State zero-valued numeric events as an explicit unrecoverable-from-XML residual unless VM data proves none occur in the compared sample; add a masked count check for blank-text measurement/lab rows whose legacy line has `VALUE: 0.0`.
- Severity: Gap | Gap: Step A's decision gate has a ≥80% accept branch and a <50% STOP branch, but the 50-80% band is not executable | Why it matters: the executor will hit a known middle case and have no class-2 routing, causing the exact round-trip the spec is meant to avoid | Evidence: docs/plans/vlm-step5-lumia-render-alignment-replan.md:114; docs/plans/vlm-step5-lumia-render-alignment-replan.md:145; ../research-skills/references/verification-and-handoff-design.md:101 | Required fix: Pre-declare the 50-80% outcome: either STOP as ambiguous/class-3, proceed with Phil approval only, or run a named secondary characterization that resolves it inline.

## Failure Modes
- Scenario: Numeric XML text is already rounded to two decimals before this repo reparses and the legacy renderer rounds the original value to one decimal | Why the plan misses it: it treats `float(element_text)` as equivalent to the original MEDS `numeric_value`, but `f"{value:.2f}"` is a lossy intermediate | What to add: A VM masked comparison for measurement rows near one-decimal rounding boundaries, plus an Expected envelope for `VALUE:` lines that still differ only by rounding.
- Scenario: `event_to_xml` overwrites numeric-looking element text with an ontology description for `str(value)` | Why the plan misses it: it cites the numeric/text collapse but does not account for the later `ontology.get_description(str(value))` override | What to add: A contract note and VM count of non-STANFORD events where element text is nonnumeric after the numeric/text winner path; classify those as unrecoverable or prove count zero for the sample.
- Scenario: The implementing agent inserts the plan's numeric block without removing the current earlier element-text-to-`text_value` branch | Why the plan misses it: the current code assigns `text_value` before `_num`, so the new branch must replace that block, not supplement it | What to add: State the exact edit: move `_num` above element-text handling and make element text populate exactly one of `numeric_value` or `text_value`.
- Scenario: The split-interval excess is partly, but not dominantly, responsible for volume drift | Why the plan misses it: 50-80% is acknowledged as ambiguous in Open Questions but not operationalized in Verification | What to add: A defined middle-band handback or secondary metric, such as table/domain breakdown plus residual top-template histogram threshold.

## Contract Checks
- Owner repo: `meds2text`; surface: `src/meds2text/textify.py::event_to_xml`. Verified: `numeric_value`/`text_value` are removed as attributes and one winner is written as element text, with numeric formatting only for Python `float`/`int` values; the ontology override can replace that text afterward. The plan should make this a cross-repo contract note because a future generator change could silently invalidate the parser heuristic.
- Owner repo: `meds2text`; surface: `src/meds2text/textify.py::omop_split_interval_events`. Verified: the transform emits only `"start"`, `"end"`, or `"start|end"` as interval state text, so `_STATE_TOKENS = {"start", "end", "start|end", "", None}` covers the current generator.
- Owner repo: `vista_eval_vlm`; surface: `src/data_tools/utils/meds_timeline_utils.py::get_llm_event_string`. Verified: `numeric_value` renders as `VALUE: {round(float(...), 1)}` and `text_value` renders as `NOTE: ...`; no renderer change is needed if the adapter populates the right column.
- Owner repo: `vista_eval_vlm` plus sibling `meds_tools`; surface: legacy `patient_string` construction. Verified in `subsampled_retrieval_csv.py` and `../meds_tools/src/meds_tools/patient_timeline.py` that the path uses `meds_reader.SubjectDatabase` → `get_described_events_window` → this repo's `get_llm_event_string`, not `meds2text.apply_transforms`.

## Modularity vs. YAGNI
- Decision point: Shape of the new interval-divergence exclusion in `diff_golden.py` | Plan's current choice: defer until Step A real data | Modular alternative + realistic use case: Commit now to a typed, line-level exclusion/counting interface similar to `--exclude-line-patterns` and `--exclude-if-legacy-missing`, but scoped by declared domain/table/code prefixes plus state-less duplicate-pair signatures; this would support future accepted export-vs-legacy divergences without bespoke code each time | Recommendation, or "raise to user" if unclear: Keep the exact pair-collapse/count-tolerance mechanics deferred, but require the plan to name the likely CLI contract and reject broad substring-only stripping for interval domains unless Step A proves it cannot hide legitimate same-domain lines.

## Verification Gaps
- Add a cheap structural/unit-style parser check to the plan even if local execution happens on the VM: XML snippets for numeric text, nonnumeric text, empty text, zero/blank behavior, `"start"`, `"end"`, and `"start|end"` should produce typed rows before the real golden re-run.
- Add a masked VM report for numeric parse coverage: count of live XML event-text values parsed to `numeric_value`, count left as `text_value`, count blank/nonparseable in measurement/lab domains, and count of residual legacy `VALUE:` lines without a live `VALUE:` counterpart.
- Add an explicit rounding-envelope check for `VALUE:` lines: after the fix, report whether remaining `VALUE:` mismatches are count-zero, rounding-only, ontology-overridden, or missing-source.
- Add a PHI-clean readback instruction for `diff_golden` failure previews. The prior handoff warned that `diff_golden.py` prints BEFORE/AFTER previews on failure; this plan should repeat that no raw previews, raw timeline text, person_ids, or dates are pasted back.
- Add carried-forward bank metadata: Step A reuses `legacy_small_v2`/`lumia_live_fixed`, but the plan should name their producing readback/SHA and say they must be un-banked if the result paths, config, or source branch moved.

## Handoff Readiness
- The plan points at the right files and most relevant functions, but several line references are stale or approximate. Current `_lumia_event_to_row` has `_STATE_TOKENS` at `ehr.py:60`, element-text-to-`text_value` at `ehr.py:95-99`, `_num` at `ehr.py:101-105`, and the return's `numeric_value` at `ehr.py:116`; the plan should cite these exact current locations.
- Approach #1's code block is directionally correct but should state "replace the existing note-body block" explicitly. Otherwise a fresh implementer could leave `text_value` populated before the numeric parse and never emit `VALUE:`.
- The success criterion should be more concrete than "residual fully explained": specify expected counts or bounded classes for post-fix `VALUE:`/`NOTE:` drift, dual-value rows, zero/rounding residuals, and interval excess.
- The landing plan is clear enough for branch ownership, but the Files to Modify list should include any VM-side characterization helper/script if Step A needs nontrivial analysis beyond ad hoc notebook commands.

## Verification & Handoff Design
- Archetype selection: Good selections include before/after parity diff, data-sanity characterization, schema/cross-repo contract checks, and PHI-clean readback. Missing selections are explicit edge/boundary parser checks for numeric/zero/state-token cases and named silent-corruption STOPs for broad interval exclusions.
- Expected-vs-unexpected envelope: Incomplete. The plan pre-declares ≥80% and <50% for split intervals, but not 50-80%; it accepts an approximate dual-value residual but does not state how much drift is acceptable after numeric parsing; it omits expected handling for zero, rounding, and ontology-overridden element text.
- Handoff phasing soundness: Not sound for complex-tier. The plan has a class-2 gate and banked prior goldens, so the canonical schema is required. Step A should be a first handoff/phase that resolves the interval signature and mechanism choice; Step B should be a later handoff after implementation at a new SHA, not bundled as if `/vm-handoff` can run both without a Mac-side code interlude.

## Suggested Revisions
- Add a `Handoff phasing` section using the canonical schema: Phase 1 read-only characterization; Phase 2 implementation verification after code is landed; name banked inputs, gates, destructive status, stop/deviation routing, and next-doc trigger for each.
- Rewrite Step A's gate as: `>=80%` accept interval divergence and implement the named mechanism; `50-80%` explicit ambiguous STOP or secondary inline check; `<50%` class-3 STOP.
- Amend Approach #1 to document zero-valued numeric loss, two-decimal pre-rounding, and ontology-override risks from `event_to_xml`; decide which are expected residuals vs. STOPs.
- Add a VM verification recipe that reports masked counts for parsed numeric element text, residual `NOTE:` numeric-looking lines, and remaining legacy `VALUE:` lines after the fix.
- Add a cross-repo contract note: this adapter relies on current `meds2text.event_to_xml` behavior; if `meds2text` starts emitting `numeric_value`/`text_value` attributes or changes formatting/override behavior, this parser should prefer the explicit attributes and re-run the golden gate.
- Make the `diff_golden.py` extension contract more concrete: likely CLI name, accepted inputs, and guardrails against broad substring exclusions that could hide legitimate interval-table content.

## Questions For The Author
- Should the 50-80% split-interval band be a hard class-3 handback, or should the VM run a second pre-declared characterization to decide inline?
- Are zero-valued and ontology-overridden numeric events acceptable declared residuals if found, or should their presence block the live-render byte-identity gate?
- Should `diff_golden.py` grow a general pair-collapse/count-tolerance mechanism now, or should this remain a one-off interval-divergence mechanism scoped only to the Step A evidence?

## Audit Trail
- ../research-skills/claude_ops.md
- ../research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-step5-lumia-render-alignment-replan.md
- docs/plans/vlm-step5-lumia-live-ehr-adapter.md
- docs/plans/vlm-step5-lumia-demographics-flowsheet-replan.md
- docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md
- docs/vm-status/2026-07-06-golden-harness.md
- src/context/adapters/ehr.py
- src/data_tools/utils/meds_timeline_utils.py
- src/vista_run/diff_golden.py
- src/data_tools/csv_helper/subsampled_retrieval_csv.py
- ../meds2text/src/meds2text/textify.py
- ../meds_tools/src/meds_tools/patient_timeline.py
