Reference: docs/claude_ops.md

# Feedback: vista_eval_vlm Step 5 re-plan #2 — Verification & Handoff Design (Codex review)

## Verdict
Revise. Step A and Step B choose broadly appropriate VM verification shapes, but the handoff is not yet inline-resolvable: the ≥80% accept threshold and <50% STOP threshold leave a 50-80% gap, and the section lacks the complex-handoff phase schema required by the canonical spec.

## Archetype Selection
- Step A is directionally the right archetype: a read-only data-sanity / migration-characterization check over already-banked before/after goldens, with PHI-clean masked structural counts. That matches the plan's unresolved claim: whether live's excess event volume is mostly explained by split interval start/end pairs rather than by another root cause.
- Step A should explicitly name the archetype and unit it verifies: "masked structural count-divergence characterization over live-only / count-mismatched line templates, measured as percent of excess live event lines or event instances." Without the denominator, the ≥80% / <50% thresholds are not attached to the exact quantity the spec requires.
- Step B is also directionally right: a before/after two-checkout golden diff at `6ded1e6` versus branch HEAD after code fixes, with accumulated exclusion mechanisms. This is a before/after parity plus declared-reanchoring diff, not just a smoke test.
- Step B's verification archetype should be stated as a rendering/text-diff correctness claim plus an attributed residual-envelope check. The current text says a clean PASS is not required, which is acceptable only if the allowed residual classes and predicted counts are pinned tightly enough for the executor to classify every residual without inventing a new rule.

## Expected/Unexpected Envelope
- The envelope has a silent middle band. Approach #3 says accept if the split-interval signature explains ≥80% of excess, while Step A STOPs only at <50%. The 50-80% range is neither accepted nor stopped, and the Open Question says to "Flag for Phil" there. That is a planned round-trip, not an inline class-2 decision gate.
- Step A's Expected is too thin for a consequential gate. "a %, plus the domain/table breakdown" should become exact reported units: total legacy non-excluded lines, total live non-excluded lines, excess live lines, count of excess lines / pairs attributed to split-interval signature, percent of excess, top domain/table buckets, and count/percent unattributed.
- Step A's STOP should include precondition failures: banked `legacy_small_v2` or `lumia_live_fixed` file missing, wrong task/experiment/tag/limit, index-set mismatch between the two banks, unreadable / absent diff histogram inputs if the intended method depends on them, or inability to keep readback PHI-clean.
- The raw XML fallback is under-specified. "plus 1-2 raw `.xml` files if needed" does not say what question XML is allowed to answer, how to choose files, or how to avoid identifiers/timeline text in readback. If retained, it should be limited to confirming the structural signature and event attributes for already-selected masked categories, with counts only.
- Step B's Expected/STOP also depends on Step A's predicted scale but does not define tolerance. "materially larger than Step A's characterization predicted" should be converted to a concrete band, for example "unattributed residual = 0; interval residual within Step A predicted count ±X% or ±N lines; any new domain/table bucket outside Step A's breakdown → STOP."
- The accepted dual-value gap is described as "~8% of non-STANFORD lines" in Approach #1, but Step B does not pin the denominator or tolerance. If it is an accepted residual class, Step B needs the same count/percent envelope as the interval divergence.

## Handoff Phasing
- The two-step split is defensible only if Step A is truly a decision-first pre-implementation gate. The prior demographics replan's Phase 0.5 justified a separate round-trip because the Mac lacked real XML conventions needed to write the fix. Here, fixes #1 and #2 are code-provable and already intended to land without VM confirmation; Step A only decides whether to add an interval-divergence exclusion in Step B.
- If the plan keeps Step A separate, the handoff section should use the canonical complex schema per phase: purpose, machine, banked-from-prior, gates, destructive?, stop/deviation, next-doc trigger. The current Step A/Step B prose has Expected/Stop/Destructive, but omits banked-from-prior, gates, and next-doc trigger.
- Batching discipline argues for either one self-resolving handoff or a tighter two-doc gate. Since Step A reads already-banked files and Step B requires Mac-side implementation after Step A, a separate Step A handoff may be justified only if the 50-80% / ≥80% / <50% outcomes are fully pre-encoded. Otherwise it knowingly creates the round-trip the spec is trying to avoid.
- "No new banking run" for Step A is structurally supported by the 2026-07-18 readback: `legacy_small_v2` and `lumia_live_fixed` were banked successfully and persisted on the results mount. However, the plan should say Step A must first verify those exact bank files and their `.meta.json` provenance before using them.
- Step B's "local golden-bank writes only" destructive characterization is accurate for the described mechanism: it uses a throwaway worktree at `6ded1e6`, branch HEAD, `golden_harness` outputs under the ignored results tree, `diff_golden`, and optional worktree removal. It should still mark git worktree add/remove and local result writes explicitly as non-destructive / local-only.

## Round-Trip Necessity
- A VM round-trip for Step A is not clearly proven necessary. The latest readback already contains the key masked histogram shape: live has ~13,774 non-STANFORD lines vs legacy 5,841, line overlap ~8%, live-only `NOTE: start|end` era suffixes, and shared-template count divergence examples. If those persisted histogram outputs are granular enough on the VM results mount, the Mac may only need the readback plus existing banked artifacts' summaries, not a fresh handoff.
- The plan should first ask whether the 2026-07-18 readback's histogram artifact contains per-template counts with domain/table and enough masked timestamp multiplicity to compute the split-interval fraction. If yes, Step A can be pulled into Mac-side analysis or into the implementation handoff as a preflight read of already-generated outputs.
- If answering Step A requires re-parsing raw XML to detect "same `(code, description)` twice per patient at distinct timestamps," then a VM step is more justified, because the XML and PHI-bearing timelines live on `phil-sllm-01`. But that fallback must be specified as a structural parser/report, not an open-ended manual inspection of 1-2 files.

## Suggested Revisions
- Add a "Handoff phasing" block with two phases using the canonical schema. For Step A, include `banked-from-prior: legacy_small_v2 + lumia_live_fixed from docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md @ afeed41; verify meta before use`.
- Replace the threshold language with an exhaustive decision gate, for example: `>=80% of excess explained by split-interval signature → accept declared divergence and proceed to Step B with interval exclusion`; `50% <= x < 80% → proceed only if remaining unattributed buckets are each below a named small-count floor, otherwise STOP`; `<50% → STOP`. If the author wants Phil review for 50-80%, state that Step A intentionally triggers a round-trip.
- Define the denominator for "percent explained": live excess event lines after applying existing STANFORD/demographic exclusions and after masking numeric values, not raw line-template count unless that is the intended unit.
- Add Step A precondition STOPs for missing bank files, mismatched selected indices, missing provenance metadata, missing histogram artifact if required, or any need to report PHI-bearing examples.
- Specify the Step A method: consume banked JSONLs and the digit-masked diff histogram output first; use raw XML only to confirm the split-interval signature for counted categories, reporting domain/table/counts only.
- Tighten Step B Expected with count tolerances for every accepted residual class: STANFORD_OBS, demographic-vintage gap, unrecoverable dual-value gap, and interval split divergence if accepted. "Fully explained" should mean zero unattributed residual, not executor judgment.
- Add Step B STOPs for wrong checkout ancestry, inability to create/remove the `6ded1e6` worktree, before/after index-set mismatch, empty or zero-byte golden files, and any `diff_golden` output that would require raw timeline text in readback.
- If Step A remains a separate handoff, add `next-doc trigger`: ≥80% accepted outcome opens the Mac implementation + Step B handoff; STOP outcomes append readback and return to plan mode; middle-band behavior must be one of those explicitly.

## Questions For The Author
- What exact denominator should the ≥80% / <50% thresholds use: excess live lines, excess live line templates, mismatched shared-template counts, or patient-level duplicate signatures?
- Does the 2026-07-18 masked histogram output already persist with enough granularity to compute Step A, or does Step A require a new VM-side script over JSONL/XML?
- In the 50-80% band, should the executor STOP for Phil review, accept with a stricter residual cap, or proceed with a named partial-attribution rule?

## Audit Trail
- ../research-skills/references/verification-and-handoff-design.md
- docs/plans/vlm-step5-lumia-render-alignment-replan.md
- docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md
- docs/plans/vlm-step5-lumia-demographics-flowsheet-replan.md
- docs/plans/vlm-step5-lumia-live-ehr-adapter.md
- docs/plans/reviews/vlm-step5-lumia-live-ehr-adapter-verification-feedback.md
