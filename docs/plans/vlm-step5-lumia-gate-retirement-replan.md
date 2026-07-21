Reference: docs/claude_ops.md

# vista_eval_vlm — Step 5 re-plan #5: retire byte-diff-to-legacy as the landing gate

**Status: Completed** (2026-07-21) — landed to `main` `df0723d`. Resolved by Phil's decision
(2026-07-20) — see **## Open Questions** below; Phase 2 human visual-QA render ran clean on
`phil-sllm-01` (`docs/vm-status/2026-07-20-828e570.md`) and Phil read all three HTML files
himself, satisfying the landing gate this plan defines.

## Context

Round 4 ([`vlm-step5-lumia-loinc-provenance-replan.md`](./vlm-step5-lumia-loinc-provenance-replan.md))
declared the LOINC residual (2724/2779 excess lines, 98%) a permanent data-source/vintage
divergence and resolved (Phil's decision, `## Resolution`) to pivot Step 5's landing gate toward
Phase 2's human visual-QA read, with one closing action left: re-run the strict byte-diff gate
(`diff_golden.py --mode strict`) with `LOINC/` added to the already-existing
`--exclude-line-patterns` mechanism, so it would read as fully attributable rather than failing.

That closing verification ran on the VM
([`docs/vm-status/2026-07-20-loinc-closing-verification.md`](../vm-status/2026-07-20-loinc-closing-verification.md))
and **did not pass**. Even with `STANFORD_OBS/Flowsheet` (round 2) and `LOINC/` (round 4) both
declared-excluded, the text gate still failed: **40 field mismatches** across the 20 shared rows
(`dynamic_prompt` + `adapter_prompt_string`). Structural gates (`image_hashes`/`selected_indices`/
counts/`assembly_mode`) still PASS — this is a text-content-only finding.

Per that doc's own Stop (b) instruction, the VM did not declare a new exclusion class just to
force the gate green. Instead it characterized the residual (PHI-safe: masked counts/shapes only)
and found it is **not confined to any enumerable domain**:

- 0 residual lines trace to `LOINC/` or `STANFORD_OBS/Flowsheet` — those two exclusions are
  working correctly; the residual is genuinely *other* content.
- Not a formatting difference — whitespace/case normalization collapses the diff by 0 lines on
  either side; line-skeletons are shape-identical on both arms.
- Not a timestamp/value-only drift — stripping the leading timestamp and the trailing value still
  leaves 4168 (legacy) / 3718 (live) event-identity-level mismatches.
- At bare event-identity (vocab/code/description only, no timestamp/value), **~55% of legacy's
  non-excluded events and ~52% of live's have no counterpart in the other arm.**
- Diagnosis: the same data-source/vintage divergence round 4 confirmed for the one on-VM MEDS
  extraction it could inspect (`vista_aug2025_meds`, 2025-08-18, vs. legacy's described
  2026-02-16 frozen snapshot) **generalizes across the entire remaining timeline, across every
  vocabulary** — LOINC was simply the largest single class of a much broader divergence, not a
  special case unique to labs.

## Why this settles round 4's Open Question 3

Round 4 left OQ3 open: *"should the byte-diff-gate methodology itself be reconsidered as a
landing gate for EHR content... not resolved here."* Step 3's result resolves it, **for this
branch**: the divergence is not enumerable-excludable. Round 2 excluded `STANFORD_OBS/Flowsheet`;
round 4 excluded `LOINC/`; roughly half of every remaining event still has no counterpart in the
other arm. Declaring a third, fourth, fifth exclusion class would mean chasing an unbounded,
ever-growing exclusion list against a residual that is — by the VM's own masked characterization —
spread across "the entire remaining timeline," not a nameable set of domains. At that point the
gate stops measuring anything about the *rendering* code (which four rounds of fixes have already
brought to a byte-identical final render function, confirmed in round 4's `## Context`) and starts
measuring an artifact of legacy's baseline being a different data pull than live's corpus. A gate
that structurally cannot pass, for reasons orthogonal to the code under test, is not a landing
gate — it's noise dressed as a Stop.

## Decision

**Retire the strict byte-diff-to-legacy comparison as a required, blocking landing gate for
Step 5 / the LUMIA-live arm.** It is demoted to informational/diagnostic: it already did real
work across rounds 1-4 (caught a demographics parser bug, a `VALUE:`/`NOTE:` field-label bug, and
correctly triaged two genuine data-provenance divergences) — that value doesn't evaporate, but a
"pass" against *this* legacy baseline is not an achievable or meaningful target going forward.

**Step 5's landing gate becomes: Phase 2's human visual-QA render is the primary and sufficient
correctness check for EHR content.** This is not a new mechanism — it is the original Step 5
plan's Phase 3 (`vlm-step5-lumia-live-ehr-adapter.md`), reprised verbatim in round 3's plan
([`vlm-step5-lumia-window-scope-replan.md`](./vlm-step5-lumia-window-scope-replan.md#phase-2--human-visual-qa-render--landing-gate))
but never yet run — Phase 1's byte-diff kept re-opening across rounds 2, 3, and 4, so Phase 2's
own gate ("once Phase 1 above is clean") never fired. This plan removes that precondition: Phase 2
no longer waits on a clean byte-diff.

Structural gates (`image_hashes`/`selected_indices`/`image_count`/`path_tile_count`/
`assembly_mode`) are unaffected by this decision — they remain byte-identical and are orthogonal
to the EHR-content-text divergence; nothing here retires them. Only the *text* gate
(`dynamic_prompt`/`adapter_prompt_string` byte-parity to the legacy baseline) is being retired as
blocking.

## Scope of this decision

Scoped to **this branch, against this specific legacy baseline** (the frozen 2026-02-16
`patient_string` CSVs vs. the live LUMIA corpus). It does not resolve the broader question of
whether *future* EHR-content work should lean on human-QA by design rather than byte-parity by
default — that stays the explicit, already-filed backlog item in `docs/next.md` ("Byte-diff-gate
methodology for EHR content"), unchanged by this doc.

## Files to Modify

- `docs/plans/vlm-step5-lumia-gate-retirement-replan.md` (this plan, new).
- `docs/plans/vlm-step5-lumia-loinc-provenance-replan.md` — status-line update recording Step 3's
  actual (failing) outcome and pointing forward to this plan.
- `docs/plans/vlm-step5-lumia-window-scope-replan.md` — status-line update noting its own Phase 2
  section is un-gated by this plan (no code changes to that plan's content).
- `docs/plans/README.md` — new row.
- `docs/next.md` — update the Step 5 entry: record Step 3's actual result, this plan's decision,
  and the concrete next action (Phil runs Phase 2 on the VM, reads the HTML himself, then `/land`).

## Open Questions

1. **Resolved (2026-07-20) — Phil: drop the legacy byte-diff comparison entirely, move on.** Not
   kept as an informational re-run either — Step 3's already-collected readback
   (`docs/vm-status/2026-07-20-loinc-closing-verification.md`) stands as the final byte-diff report
   for this branch; no further `diff_golden.py` re-runs against the legacy baseline are needed
   before or at landing. Phase 2 human-QA is Step 5's whole landing gate, full stop.

## Verification & VM handoff

**What runs on the VM** — Claude-Code CPU `phil-sllm-01`, same posture as every step on this
branch. Single phase, read-only render, no destructive writes, no code changes to land first.
**Target machine:** Claude-Code CPU (`phil-sllm-01`).

### Step 1 — Phase 2 human visual-QA render (un-gated)

Reprise, verbatim, of round 3's Phase 2 section — no changes to the command or its criteria, only
to when it's allowed to run (no longer waiting on a clean Phase 1 byte-diff).

```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only

cd src
python -m results.context_viewer --config ../configs/all_tasks.viewer.vm.yaml --type gemma3 \
  --name google/medgemma-1.5-4b-it --task progression_recurrence_free_survival_1_yr \
  --experiment no_image --limit 5          # repeat for no_report/timeline_only, axial_all_image
```

**Expected:** each HTML exists, non-empty, self-contained (no external URLs, no local filesystem
paths in `<img src>`), exactly 5 cards; every card has non-empty rendered prompt/timeline text and
a non-empty token-count/bar; `axial_all_image` cards show the expected slice/thumbnail count,
`no_image`/`no_report` cards show 0 images where expected; no `STOP:` or traceback text anywhere.
**Report back:** file paths, existence, card counts, exit codes only — **never paste rendered
content** (PHI). Phil opens the files himself on `phil-sllm-01` to read the rendered text — the
manual QA step no agent can do for him.
**Stop:** missing/empty HTML, non-zero exit, the self-containment grep fails, a card missing
text/token content, or a `STOP:`/traceback string appears where it shouldn't.
**Destructive:** no.

## Landing & cleanup

- **Branch:** `feat/lumia-live-ehr-adapter` (continuing, no new branch — this plan makes no code
  changes).
- **Landing gate:** Phase 2 rendered clean (Step 1 above) **and** Phil has actually opened and
  read the HTML himself. Byte-diff-to-legacy is no longer part of the landing gate (see Decision
  above) — Step 3's already-collected readback
  (`docs/vm-status/2026-07-20-loinc-closing-verification.md`) stands as the final byte-diff report
  for this branch; no further re-runs, informational or otherwise (Phil, per OQ1).
- **Merge sequence:** single branch, `/land` at the end -> `main`, prune branch.
- **Cleanup on land:** mark this plan, the loinc-provenance-replan, and the window-scope-replan
  `Status: Completed`; update `docs/next.md` and `docs/plans/README.md`; carry forward the two
  already-backlogged items (Ryan D'Cunha MEDS-extraction-reconciliation escalation, byte-diff-gate
  methodology for *future* EHR-content branches) unchanged — this plan resolves the
  this-branch instance of round 4's OQ3, not the general policy question.
