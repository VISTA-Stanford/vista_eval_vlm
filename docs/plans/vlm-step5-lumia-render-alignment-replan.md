Reference: docs/claude_ops.md

# vista_eval_vlm — Step 5 re-plan #2: close the live-adapter render-alignment gap

**Status: Draft** (2026-07-19) — supersedes the "Verification & VM handoff" section of
`docs/plans/vlm-step5-lumia-demographics-flowsheet-replan.md` for Phase 1 onward (again). Builds on,
does not replace, that plan (already landed: demographics synthesis + `type`→`table` fix) and the base
plan `docs/plans/vlm-step5-lumia-live-ehr-adapter.md`.

**2026-07-20 update — Approach #3 (interval-split) refuted, superseded for the volume-gap residual.**
The VM Phase 1 re-run (`docs/vm-status/2026-07-20-7ed0248.md`) found the `omop_split_interval_events`
start/end-pair signature explains only **4.0%** of the excess (321/7,990 lines) — a `<50%` STOP,
refuting it as the primary cause of live's ~2× event-volume gap. The two `ehr.py` render fixes below
(`VALUE:`/`NOTE:` field-label mismatch, `start|end` token leak) are unaffected and remain landed/valid
— only the volume-gap hypothesis is refuted. The real cause (a window-scope mismatch: legacy's frozen
`patient_string` was generated with a 24-month lookback, live's `timeline` variant rendered full
unrestricted history) is root-caused and fixed in the next re-plan,
`docs/plans/vlm-step5-lumia-window-scope-replan.md`.

## Context

Phase 1's decision gate was re-run with the demographics fix + two exclusion mechanisms in place
(`docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md`) and hit a **second class-3 deviation**:
strict gate FAILED on all 20/20 rows, both text fields — not the STANFORD_OBS/demographic-vintage
residual the prior replan expected. The actual residual: line-level overlap is only ~8%, live emits
~2.36× legacy's event volume, and the dominant cause is a field-label mismatch — legacy renders every
lab/measurement as `VALUE: D.D`, live renders the *same* lines as `NOTE: D.D` (live never emits
`VALUE:` at all). Resume block routed this back to the Mac to re-plan the render alignment.

Investigating from code (not VM speculation) found the root causes are largely already provable:

1. **The `VALUE:`/`NOTE:` mismatch is a resurfacing of an already-known, never-resolved fork.** The
   2026-07-06 field-coverage VM check (`docs/vm-status/2026-07-06-golden-harness.md:219-249`) already
   found `numeric_value` at **0% coverage** in real LUMIA XML — "genuinely not a discrete field... the
   number is embedded in `<event>` element text" — and explicitly logged an open design fork: *"accept
   the declared delta vs. parse numeric from text"*. That fork was never actually decided before Step 5
   got wired; it was carried forward as a footnote assumed to be minor. It isn't — labs/measurements are
   the bulk of any timeline, so leaving it unresolved is why it's now the dominant blocker.
2. **Confirmed against meds2text's real generator** (`meds2text/src/meds2text/textify.py::event_to_xml`,
   ~L1248-1340, sibling repo `VISTA/code/meds2text`): it reads MEDS `numeric_value`/`text_value`
   properties, picks exactly one (`value = attributes.get("numeric_value") or attributes.get("text_value")
   or None`), then unconditionally `attributes.pop("numeric_value", None); attributes.pop("text_value",
   None)` — **neither ever survives as an XML attribute**. The winning `value` is written only into the
   `<event>` element **text**, formatted as `f"{float(value):.2f}"` (or bare int) for numerics, or the raw
   string otherwise. So real LUMIA XML has no attribute-level signal distinguishing "originally numeric"
   from "originally text" — `_lumia_event_to_row` (`src/context/adapters/ehr.py:79-127`) reads
   `attrib.get("numeric_value")`, which is dead code (that attribute never exists), and only ever
   populates `text_value` from element text. Every lab/measurement value renders under `NOTE:` instead of
   `VALUE:` as a structural consequence.
3. **The secondary "`NOTE: start|end`" leak traces to `omop_split_interval_events`**
   (`textify.py:458-493`): for `visit`/`visit_detail`/`drug_exposure` interval events, this transform sets
   a MEDS `text_value` property to `"start"`, `"end"`, or — when start≈end within 1 minute — the
   **combined** `"start|end"` token, which flows into element text unchanged. `_lumia_event_to_row`'s
   `_STATE_TOKENS` (`ehr.py:60`) already excludes bare `"start"`/`"end"` but not the combined form — that
   one case leaks through as a stray `NOTE: start|end` line, exactly matching the readback's
   `RxNorm/…`/`NUCC/…` finding.
4. **The remaining ~2.36× event-volume / count-mismatch-on-shared-templates residual is only partly
   explained by the above.** `omop_split_interval_events` structurally emits **two** XML events (at
   `time` and at `end`) per non-instantaneous interval, where legacy's `patient_string` pipeline
   (`get_described_events_window` + `get_llm_event_string`, built via a direct
   `meds_reader.SubjectDatabase` query — **confirmed via `/review-plan`'s Codex audit**, citing
   `src/data_tools/csv_helper/subsampled_retrieval_csv.py` and the sibling repo
   `../meds_tools/src/meds_tools/patient_timeline.py`, that this path does **not** go through meds2text's
   `textify.py`/`apply_transforms` stack at all) most plausibly never applies this split. Strong,
   code-grounded, and now independently confirmed — but quantifying *how much* of the volume it explains
   still needs real data.

Findings (1)-(3) are code-provable — this plan lands those two fixes directly rather than gating them
behind another VM round-trip (the "VM confirms conventions first" discipline from the prior replan
applies when the convention is genuinely unknown; here meds2text's own source *is* the convention).
Finding (4) still needs one VM-side, PHI-safe characterization step before a fix-vs-accept call — this
means a small analysis **script over already-banked golden files**, not manual inspection: the
2026-07-18 readback's digit-masked histogram is illustrative (a handful of example line-template counts),
not an aggregate breakdown, so it cannot itself answer "what fraction of the excess is split-interval
duplication" — that requires actually running the count. (`/review-plan`'s verification-design pass asked
whether this round-trip is avoidable; re-checking the 2026-07-18 doc's actual content confirms it isn't —
the needed aggregate doesn't exist yet in any artifact on the Mac.)

**Independent review (2026-07-19):** audited by Codex via `/review-plan` (design pass + a dedicated
verification-and-handoff-design pass — `docs/plans/reviews/vlm-step5-lumia-render-alignment-replan-
*feedback.md`). Findings applied throughout: two new declared-residual classes (Approach #1), a resolved
50-80% decision-gate band (Approach #3), and the two-phase Verification & VM handoff structure below.

## Goal

Fix the two confirmed adapter bugs (`VALUE:`/`NOTE:` mismatch, `start|end` token leak), characterize and
resolve the event-count/scope divergence (fix or accept-as-declared-divergence, decided once quantified),
and re-run Phase 1's decision gate to get an interpretable residual.

## Approach

**1. `_lumia_event_to_row` (`ehr.py:79-127`) — reconstruct `numeric_value` from element text.** Real
LUMIA element text for a genuine numeric MEDS value is always `float()`-parseable (meds2text formats it
as `f"{value:.2f}"` or a bare int string). **Replace** the current element-text-to-`text_value` branch
(`ehr.py:95-99`) and the return dict's dead `numeric_value` read (`ehr.py:116`,
`_num(attrib.get("numeric_value"))`) — not add alongside them — with:
```python
def _num(v):   # already exists at ehr.py:101-105, unchanged — hoist above its first use
    try:
        return float(v) if v is not None and str(v) != "" else None
    except (TypeError, ValueError):
        return None

text_value = attrib.get("text_value")        # dead in practice — meds2text never emits this attribute
numeric_value = _num(attrib.get("numeric_value"))  # also dead in practice, same reason
if text_value is None and numeric_value is None:
    el_text = (event_el.text or "").strip()
    if el_text and el_text.lower() not in _STATE_TOKENS:
        parsed = _num(el_text)
        if parsed is not None:
            numeric_value = parsed
        else:
            text_value = el_text
```
No renderer change needed: `get_llm_event_string` (`meds_timeline_utils.py:222-236`) already branches on
whichever of `numeric_value`/`text_value` is populated.
- **Cross-repo contract note (Codex review):** this fix depends on meds2text's *current* `event_to_xml`
  collapse behavior (`textify.py:1273-1340`). Note in `ehr.py`'s docstring: if meds2text ever starts
  emitting explicit `numeric_value`/`text_value` attributes, or changes its formatting, prefer those
  attributes over this text-parse heuristic and re-run the golden gate.
- **Accepted, not chased further — three declared, byte-level residual classes, all structurally forced
  by meds2text's export design (not adapter bugs):**
  1. ~460/5,841 legacy lines shaped `VALUE: D.D | NOTE: D.D` (both fields on one row) — `event_to_xml`
     keeps only ONE winning value per event; the loser never reaches the `.xml`.
  2. **Zero-valued numeric measurements (Codex review).** `event_to_xml`'s winner selection is
     `attributes.get("numeric_value") or attributes.get("text_value") or None` — Python's `or` treats
     `0.0` as falsy, so a genuine zero-valued lab silently loses to `text_value`/`None`: live's element
     text can be **empty** for a row legacy renders as `VALUE: 0.0`.
  3. **Ontology-override collapse (Codex review).** After picking a winner, `event_to_xml` overwrites the
     element text with an OMOP concept description if `ontology.get_description(str(value))` resolves
     (`textify.py:1335-1338`) — a numeric value whose string form happens to match a concept code loses
     its number entirely; this repo's `float()` parse correctly falls through to `text_value` for that
     row, but the number itself is unrecoverable.
  Phase 1's VM characterization (below) reports masked counts for classes 2-3 so their real prevalence is
  known, not assumed negligible; Phase 2 expects all three as declared residual, not a failure.

**2. `_STATE_TOKENS` (`ehr.py:60`) — also exclude the combined `"start|end"` token:**
```python
_STATE_TOKENS = {"start", "end", "start|end", "", None}
```
One-line fix, directly grounded in `omop_split_interval_events`'s combined-token branch. **Confirmed
sufficient** (Codex review, read `omop_split_interval_events` fully): the transform emits only `"start"`,
`"end"`, or `"start|end"` as interval state text — no other combined shape exists.

**3. Event-count/scope divergence — VM characterization first, then fix-or-accept.** This is now
"Phase 1" of Verification & VM handoff below (renamed from "Step A" per the canonical complex-handoff
schema). Reuse the already-banked `legacy_small_v2`/`lumia_live_fixed` golden files (no new banking run)
for a masked, PHI-safe structural check: of the **excess live lines** (live non-excluded line count minus
legacy non-excluded line count, both after existing STANFORD_OBS/demographic-vintage exclusions — 13,774
vs. 5,841 per the 2026-07-18 readback), what fraction correspond to `(code, description)` pairs appearing
twice per patient at two distinct timestamps — the `omop_split_interval_events` start/end-pair signature —
vs. any other pattern. Needs a small analysis script over the already-banked JSONLs (the existing
digit-masked histogram is illustrative only, not an aggregate — see Context above).
- **Decision gate, fully resolved (no dangling band — Codex review flagged the original ≥80%/<50% split
  left 50-80% unexecutable):**
  - **≥80%** of excess explained → accept as a permanent, declared divergence (same category as the
    already-accepted STANFORD_OBS/demographics gaps) → add a matching `diff_golden.py` exclusion
    mechanism scoped to interval-table codes (illustrative shape, finalized once Phase 1's real counts are
    in: a `--collapse-duplicate-pairs <code-prefix>[,...]` flag that collapses same-`(code, description)`
    pairs at distinct timestamps into one counted occurrence before comparison, scoped to declared
    interval-table prefixes like `RxNorm/`/`NUCC/` only — never a broad substring exclusion). Prefer this
    over making `parse_lumia` re-collapse split intervals itself — that would mean guessing at legacy's
    own interval convention with no way to verify it.
  - **<50%** → STOP, class-3 — a second, unidentified cause is present; hand back.
  - **50-80%** → STOP and hand back to Phil explicitly, same as <50% (not an auto-resolve — this
    workstream has twice found the residual bigger than expected; not worth a third guessed threshold).
    See Open Questions for Phil's override options.

**4. Phase 2 of Verification & VM handoff** — re-run the base plan's Phase 1 decision gate with fixes
(1)+(2) landed and, if Phase 1 above resolved to "accept," its new exclusion mechanism added. Same
mechanism as before: worktree at `6ded1e6` for "before", this branch for "after", `diff_golden --mode
strict` with the existing `--exclude-line-patterns STANFORD_OBS/Flowsheet --exclude-if-legacy-missing
MEDS_BIRTH Ethnicity/ Race/ Gender/` flags plus whatever Phase 1 adds.

## Files to Modify

- `src/context/adapters/ehr.py` — `_lumia_event_to_row`'s numeric-vs-text branch, replacing `ehr.py:95-99`
  and `ehr.py:116` (Approach #1); `_STATE_TOKENS` at `ehr.py:60` (Approach #2); module/function docstrings
  updated to record the resolved 2026-07-06 OQ-K fork, the meds2text cross-repo contract note, and the
  three accepted residual classes (so none of this reads as an open question for the next reader).
- `src/vista_run/diff_golden.py` — new exclusion mechanism for the interval-table start/end-pair
  divergence, **only if** Phase 1's VM check confirms accept-as-declared (illustrative shape in Approach
  #3; exact mechanics finalized against Phase 1's real counts — module docstring gets the rationale, same
  pattern as the existing `--exclude-line-patterns`/`--exclude-if-legacy-missing` flags).
- A small **VM-local analysis script** for Phase 1's aggregate characterization (reads the already-banked
  `legacy_small_v2`/`lumia_live_fixed` JSONLs, computes the excess-line split-interval-signature fraction)
  — one-off, not committed to the repo unless it proves reusable; name it in the handoff doc.
- `docs/plans/vlm-step5-lumia-live-ehr-adapter.md` — record this second deviation + resolution once
  Phase 2 re-resolves (same pattern the demographics replan used in its header note).
- `docs/vm-status/<date>-<sha>.md` (new) — VM handoff doc rendered via `/vm-handoff`, covering both
  phases below.

## Open Questions

- **The 50-80% decision-gate band (Approach #3).** **Deferred (Phil, 2026-07-20):** keep the default —
  STOP, hand back to Phil — rather than pre-authorizing an override now; revisit once the real number is
  in.
- ~~Do the zero-valued-numeric / ontology-override residuals (Approach #1) block Phase 2's gate?~~
  **Resolved (Phil, 2026-07-20):** agreed — accepted, same treatment as the dual-value gap, provided
  Phase 1/2's masked counts show them as a small tail; unexpectedly large is still a Phase 2 STOP.
- ~~Should the new `diff_golden.py` mechanism (if triggered) generalize beyond this one case?~~
  **Resolved (Phil, 2026-07-20):** agreed — YAGNI, stay scoped to Phase 1's specific evidence.

## Verification & VM handoff

**What runs on the VM** — Claude-Code CPU `phil-sllm-01`, same posture as every step on this branch.
**Complex-tier handoff** (class-2 decision gate + more than one phase + depends on prior-banked artifacts)
— phased below per the canonical schema, per `/review-plan`'s verification-and-handoff-design audit.
**PHI discipline throughout** (unchanged from the base plan and both prior handoffs on this branch):
report counts, field names, exit codes, and decision-gate outcomes only — never raw timeline text,
person_ids, or `diff_golden`'s printed BEFORE/AFTER previews (it prints those on failure — summarize
instead).

### Phase 1 — masked structural characterization of the event-count divergence

- **Purpose:** resolve Approach #3's decision gate — does the split-interval signature explain the
  live/legacy volume gap, and how much?
- **Machine:** Claude-Code CPU `phil-sllm-01`. Read-only; no code dependency, so this phase can run
  independently of (before, during, or after) the Mac landing fixes #1-#2 below.
- **Banked-from-prior:** `legacy_small_v2`/`lumia_live_fixed` golden JSONLs from
  `docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md` @ `afeed41` (persisted on the results
  mount). **Precondition:** verify both files exist and their `.meta.json` provenance (config/tag/limit)
  matches this workstream before use; if either is missing or provenance mismatches, that's a STOP
  (re-bank isn't in scope for this phase — hand back).
- **What runs:** a small analysis script over the already-banked JSONLs (no new `golden_harness` run) —
  compute, per patient: legacy non-excluded line count, live non-excluded line count (both after existing
  STANFORD_OBS/demographic-vintage exclusions), excess live lines, and what fraction of the excess is
  `(code, description)` pairs recurring at two distinct timestamps (the split-interval signature) grouped
  by domain/table. Raw `.xml` re-parsing is a fallback only if the banked JSONLs can't answer this
  (e.g., need to confirm a signature's underlying event attributes) — if used, scope strictly to
  confirming the structural signature for already-selected masked categories, counts only, no more than
  1-2 files.
- **Expected:** exact reported units — total legacy/live non-excluded lines, excess line count, count and
  % of excess attributed to the split-interval signature, domain/table breakdown of both attributed and
  unattributed excess, and the same masked counts for the two new Approach-#1 residual classes
  (zero-valued numerics, ontology-override collapse).
- **Gates:** Approach #3's decision gate, resolved inline here — no round-trip for the threshold itself.
- **Destructive:** no — read-only script over already-banked artifacts (+ optional 1-2 raw `.xml` reads).
- **Stop/deviation routing:** precondition failure (missing/mismatched bank files) or gate STOP → append
  readback to the handoff doc, return to Mac plan mode — do not proceed to Phase 2.
- **Next-doc trigger:** ≥80% outcome → Mac lands fixes #1-#2 (always) + the `diff_golden` mechanism
  (Approach #3's illustrative shape, finalized against this phase's real counts) → same handoff doc's
  Phase 2 runs at the new SHA. STOP outcomes end this handoff; no Phase 2.

### Phase 2 — Mac lands code, then re-run the base plan's Phase 1 decision gate

- **Purpose:** confirm the code fixes (+ exclusion mechanism, if Phase 1 triggered one) produce a fully
  attributable residual.
- **Machine:** Claude-Code CPU `phil-sllm-01`.
- **Banked-from-prior:** none reused — this is a fresh bank (same mechanism as the base plan's Phase 1:
  worktree at `6ded1e6` for "before", branch HEAD for "after").
- **What runs:** first, a cheap sanity check before the real re-run — feed a handful of hand-built XML
  `<event>` snippets (numeric text, non-numeric text, empty text, zero-value text, `"start"`/`"end"`/
  `"start|end"`) through `_lumia_event_to_row` and confirm each routes to the expected
  `numeric_value`/`text_value`/neither outcome. Then the real re-run: bank both sides, `diff_golden
  --mode strict --exclude-line-patterns STANFORD_OBS/Flowsheet --exclude-if-legacy-missing MEDS_BIRTH
  Ethnicity/ Race/ Gender/` plus whatever Phase 1 added.
- **Expected:** zero *unattributed* residual. Every remaining mismatch must fall into one of: STANFORD_OBS
  (always), demographic-vintage-gap, the dual-value ~8%-of-lines gap, the zero-valued-numeric residual,
  the ontology-override residual, or (if Phase 1 triggered it) the interval-divergence exclusion — each
  reported as a count/%, with the interval-divergence count checked against Phase 1's predicted scale
  (within a reasonable tolerance; a materially different count, or any new domain/table bucket outside
  Phase 1's breakdown, is a STOP, not a pass).
- **Destructive:** no — local golden-bank writes under the git-ignored results tree, plus the throwaway
  `6ded1e6` worktree (added/removed same as the base plan's Phase 1).
- **Stop/deviation routing:** any unattributed residual class, an index-set mismatch, empty/zero-byte
  golden output, or an interval-divergence count materially off Phase 1's prediction → STOP, append
  readback, return to Mac plan mode (this workstream's own history — two prior rounds where the residual
  was bigger than expected — is exactly why this phase doesn't accept "close enough" silently).
- **Next-doc trigger:** clean attributable result → this plan's work is done; hand back to the base plan's
  Phase 2 (allowlist gate) and Phase 3 (human visual-QA), unchanged from `vlm-step5-lumia-live-ehr-adapter.md`.

## Landing & cleanup

Unchanged from the base plan: single branch `feat/lumia-live-ehr-adapter` off `main`, `/land` at the end
once Phases 1-3 are all green *including* Phil's Phase-3 human read of the rendered HTML (not yet
reached — this plan only unblocks Phase 1). No new branch, no new worktree beyond the existing throwaway
`../vista_eval_vlm-prewiring` pattern already used for Phase 1 banking.
