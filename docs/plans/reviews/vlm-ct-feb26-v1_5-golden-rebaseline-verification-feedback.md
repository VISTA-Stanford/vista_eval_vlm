Reference: docs/claude_ops.md

# Feedback: VLM CT feb26/v1_5 rungs 1-2 — Verification & Handoff Design pass (Codex)

## Verdict

Revise. The selected verification archetypes are mostly right for the two-axis design, and the executor class is correctly Claude-Code CPU / weights-free. The handoff is not yet handoff-ready because it has class-2/human gates, banked goldens, and a Mac implementation interlude but no explicit `## Handoff phasing` decomposition as required for complex handoffs.

## Archetype Selection

- **Step 1 — v1_5 CT resolution smoke:** Sound archetype: a data-sanity / contract smoke for the new `(study_uid, series_uid) -> feb26 blob` resolver, with silent-fallback traps for feb26 404, nov25 fallback, and zero CT images. This matches the spec's data-sanity and silent-corruption STOP guidance (`verification-and-handoff-design.md:53-55`, `:79-84`) and the plan's separation of the substrate axis from byte identity (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:79-84`, `:258-263`).
- **Step 1b — coverage report:** Right archetype as a soft data-sanity / coverage characterization, not a byte gate. That is correct because the plan explicitly says v1_1 -> v1_5/feb26 is not byte-gated (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:63-65`, `:81-84`). The coverage check should stay report-only.
- **Step 2 — EHR `no_image` re-bank:** Right archetype: re-anchoring / re-banking consistency plus schema/shape sanity, not byte parity against v1_1. The Expected line is checkable against harness output: `golden_harness.py` emits `adapter_prompt_string`, `dynamic_prompt`, `image_count`, `assembly_mode`, `person_id`, and `index` (`src/vista_run/golden_harness.py:217-234`) and writes `.meta.json` with `row_count` (`src/vista_run/golden_harness.py:301-316`).
- **Step 3 — axial before golden:** Right archetype: banked before baseline for later before/after refactor parity. The selected fields are checkable: `image_count`, `image_hashes`, `selected_indices`, and model/windowing are emitted by the harness (`src/vista_run/golden_harness.py:217-234`).
- **Step 4 — 3b refactor diff:** Right archetype: before/after refactor parity with exact equality. `diff_golden.py` enforces index-set parity, Gate 1/2 hard structure fields, and per-model file compatibility (`src/vista_run/diff_golden.py:102-146`, `:156-172`). Minor revision: Step 4 should explicitly say `--mode strict` and include the full hard structure set that the tool enforces, including `path_tile_count` even if CT rows should carry `None` (`src/vista_run/diff_golden.py:38-54`, `:204-205`).
- **Step 5 — gate-3 EHR allowlist diff:** Right archetype: declared-delta allowlist diff, correctly separated from the CT byte gate. The D3 decision is correctly deferred until after the first real gate-3 diff because the current normalizer is intentionally minimal and the actual LUMIA delta is not known yet (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:66-78`; `src/vista_run/diff_golden.py:57-67`).
- **Soft coverage report:** Present and correctly non-blocking. The plan should tighten the readback evidence so it reports counts plus at most structural identifiers needed for Phil's decision, without pasting rows, timelines, XML, or diff value previews.

## Expected/Stop Envelope

- **Step 1:** Expected and Stop are concrete and checkable. The resolver-trace criterion is behavioral rather than a substring/code-property assertion, which is the right shape (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:258-263`; spec warns against substring-only proof at `verification-and-handoff-design.md:90-95`).
- **Step 1b:** The soft gate is well motivated, but the control flow should be clearer. It currently says "REPORT, do NOT stop" and also "get Phil's input before banking C1" (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:264-271`). That is a real halt-before-Step-3 human gate, even if divergence is not a failure. Revise it to: Step 1b never fails on coverage divergence; executor records the counts and <=5 structural examples; Steps 1, 1b, and 2 may complete in the same VM doc; Step 3 must not run until Phil records accept/revise. This aligns with decision-gate-over-round-trip discipline (`verification-and-handoff-design.md:142-144`) while acknowledging that this gate cannot be fully pre-encoded because "acceptable go-forward coverage" is a human product decision.
- **Step 2:** Expected is strong. Stop should add explicit failure cases for `row_count == 0`, mismatched `.meta.json` `row_count` vs JSONL lines, missing/non-empty output files, and a dirty repo containing golden output. The current Stop only covers missing/different timeline shape (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:272-278`), while the spec requires every consequential step to name substantive Expected and halt conditions (`verification-and-handoff-design.md:124-128`).
- **Step 3:** Expected is checkable. Stop is mostly sound, but add missing/non-empty output and `.meta.json` checks, index uniqueness/sort failure, and unexpected retrieval-skip/no-data messages. The harness can skip retrieval experiments by design (`src/vista_run/golden_harness.py:98-102`, `:287-290`); for `axial_all_image`, a skip or zero-row output should be a STOP, not a benign note.
- **Step 4:** Expected/Stop are correct at the high level but should pin the command mode and evidence contract. `diff_golden.py` default mode is strict (`src/vista_run/diff_golden.py:204-205`), but the handoff should say it explicitly. Stop should include index-set mismatch and no shared indices, both of which the tool treats as gate failures (`src/vista_run/diff_golden.py:156-163`, `:190-193`).
- **Step 5:** D3 is a valid class-2/human decision gate after the first real allowlist diff, but the current Stop text says to "record the concrete diff" (`docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md:288-292`). Because `diff_golden.py` prints prompt/timeline value previews on failures (`src/vista_run/diff_golden.py:183-186`), revise this to record counts, field names, affected indices, and a PHI-scrubbed characterization only; do not paste BEFORE/AFTER prompt or timeline values into the handoff.

## Handoff Phasing

Yes, this needs an explicit `## Handoff phasing` decomposition. The canonical classifier makes a handoff complex if it has any class-2 decision gates, bank/un-bank logic, or more than one handoff phase (`verification-and-handoff-design.md:278-285`), and `claude_ops.md` requires per-phase purpose, machine, banked-from-prior, gates, destructive?, stop routing, and next-doc trigger for complex handoffs (`/Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md:145-165`).

Concrete proposed phasing:

- **Phase 1 — v1_5/feb26 pre-bank characterization.** Purpose: prove Axis A/B can read v1_5/feb26 and re-bank the EHR baseline. Machine: Claude-Code CPU on `som-nero-plevriti-deidbdf`. Steps: 1, 1b, 2. Banked-from-prior: rung 0 green prerequisite only; no re-run of rung 0. Gates: Step 1b records coverage counts and <=5 structural examples, then Phil accepts/revises before C1. Destructive?: no published overwrite; golden output only on PHI mount. Stop/deviation: Step 1/2 hard stops; Step 1b divergence is not a failure but pauses before Step 3. Next-doc trigger: Phil accepts coverage and C1 is authorized.
- **Phase 2 — C1 before-golden bank.** Purpose: bank the legacy feb26 axial baseline at the accepted SHA/config. Machine: Claude-Code CPU. Steps: 3 only. Banked-from-prior: Step 2 v1_5 `no_image` baseline from Phase 1 if the SHA/config inputs have not moved; otherwise un-bank and rerun. Gates: none beyond Step 3 Expected/Stop. Destructive?: no published overwrite; writes PHI-mount golden. Stop/deviation: no C1 if zero rows, weights load, missing output/meta, dirty repo, or unexpected skip. Next-doc trigger: Phase 2 green hands back to Mac for C2 implementation.
- **Phase 3 — C3/C4/C5 after implementation.** Purpose: after Mac implements 3b, bank adapter feb26 output and run strict CT parity plus allowlist EHR diff. Machine: Claude-Code CPU. Steps: 4 and 5. Banked-from-prior: C1 before golden from Phase 2, and Step 2 EHR baseline if unchanged by SHA/config. Gates: D3 accept-vs-parse decision after the first real Gate-3 diff. Destructive?: no published overwrite. Stop/deviation: any Gate 1/2 strict drift is a hard class-3 halt; Gate 3 residual outside the declared envelope is a D3 decision with PHI-clean evidence.

The Mac implementation interlude is the main reason to split this. A single vm-status doc spanning "VM bank before -> Mac edits code -> VM bank after" is ambiguous because the SHA necessarily changes between C1 and C3/C4. The plan can either use two vm-status docs after Phase 1/2, or one superseding chain with documented banked steps by SHA; the current single straight-line Steps 1-5 presentation does not record that bank/un-bank rule.

## Suggested Revisions

- Add a `### Handoff phasing` block under `## Verification & VM handoff` using the three phases above. Include the banked-by-SHA rule: if code/config/data inputs move after a banked phase, un-bank and rerun the affected golden.
- In Step 1b, replace "do NOT stop" with "do not fail on divergence; pause before Step 3 for Phil's coverage accept/revise decision." Keep Steps 1, 1b, and 2 batched before that pause.
- In Step 2 and Step 3, add Stop conditions for missing/non-empty output files, `.meta.json` missing or row-count mismatch, dirty repo containing golden output, zero rows, and unexpected skip/no-data messages.
- In Step 4, name `diff_golden.py ... --mode strict`, include index-set parity/no shared indices in Expected/Stop, and mention the full hard structure field set enforced by the tool.
- In Step 5, change "record the concrete diff" to a PHI-clean readback contract: counts by field, affected indices, first <=5 field-name/index examples, and scrubbed characterization only. Do not paste `diff_golden.py` BEFORE/AFTER previews because those can contain prompt/timeline PHI (`src/vista_run/diff_golden.py:183-186`).
- In the PHI paragraph, reconcile "UIDs-as-structure only" with the canonical PHI-clean rule that readback should not include identifiers/DICOM UIDs (`verification-and-handoff-design.md:85-89`). If UIDs are necessary for Step 1b adjudication, say they remain on the PHI mount or are hashed/redacted in the handoff.

## Questions For The Author

- For Step 1b, does Phil need actual `(study_uid, series_uid)` values in the handoff to decide coverage, or are counts plus hashed/ordinal structural examples enough?
- Should Phase 2 get its own vm-status doc after Phil accepts Step 1b, or should `/vm-handoff` render a superseding doc that carries Steps 1/1b/2 as banked by SHA?

## Audit Trail

- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/references/verification-and-handoff-design.md
- /Users/philadamson/Documents/Stanford/VISTA/code/research-skills/claude_ops.md
- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- src/vista_run/golden_harness.py
- src/vista_run/diff_golden.py
