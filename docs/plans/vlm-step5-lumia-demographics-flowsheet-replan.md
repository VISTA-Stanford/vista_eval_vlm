Reference: docs/claude_ops.md

# vista_eval_vlm — Step 5 re-plan: close the parse_lumia demographics gap, accept the flowsheet divergence, re-verify

**Status: Draft** (2026-07-17) — supersedes the "Verification & VM handoff" section of
`docs/plans/vlm-step5-lumia-live-ehr-adapter.md` for Phase 1 onward. Approach/Goal/Files-to-Modify
sections below are additive to that plan, not a replacement of it.

## Context

Step 5 (wire the LUMIA-live EHR adapter into the real hot path) was implemented on
`feat/lumia-live-ehr-adapter` and handed to the VM for verification. The VM hit a **class-3
deviation** at Step 2 (Phase 1's decision gate): the live LUMIA render and the legacy
`patient_string` are near-disjoint (102/~24,000 shared lines) — no `code_filter` config can make
them byte-match, so the byte-diff verification premise the plan rested on doesn't hold
(`docs/vm-status/2026-07-17-4989a20.md`, `⚠️ DEVIATION` block).

Planner-side investigation this session (reading `meds2text/docs/markup.md`, `meds2text`'s
`textify.py`/`transforms/core.py`, `vista_eval_vlm`'s `ehr.py`/`ehr_filters.py`/`diff_golden.py`,
and git blame/history across both repos) resolved *why*, and it splits into two very different
kinds of gap:

1. **Demographics (birthdate/ethnicity/gender → rendered as `MEDS_BIRTH`/`Ethnicity`/`Gender`
   classes) — a confirmed, plain parser bug.** The real LUMIA XML schema documents `<person>`
   (with `<birthdate>`, `<demographics><ethnicity/gender>`) as a child of `<encounter>`, a *sibling*
   of `<events>` (`meds2text/docs/markup.md:15-36`). `parse_lumia`
   (`src/context/adapters/ehr.py:134-143`) only ever walks into `<events>/<entry>/<event>` — it
   never reads `<person>` at all. Git blame shows `parse_lumia` was authored whole-cloth in one
   commit (`6f5eb9d2`, 2026-07-06) and has never been touched since; every plan/review doc in this
   repo was grepped for "demographic"/`<person>`/"birthdate" and none mention it — this was never a
   design decision, just an oversight (the implementation was scoped around replicating
   `get_llm_event_string`'s per-`<event>` fields, and `<person>` isn't an `<event>`).
2. **Separately found: `parse_lumia` reads the wrong attribute name.** meds2text's actual generator
   (`meds2text/src/meds2text/textify.py`, `event_to_xml`) emits the event-class attribute as
   `table=` (`attribute_order` defaults to `["table", "code", "name"]`), not `type=` as both
   `markup.md` and `_lumia_event_to_row` (`ehr.py:106`) assume. This doesn't explain the
   STANFORD_OBS gap (parse_lumia doesn't filter by type/table at ingestion — it captures every
   `<event>` child unconditionally), but it silently breaks any future filter keyed on `type`
   (`note_type_filter`, unused by this plan's wired presets today but present in `ehr_filters.py`).
3. **STANFORD_OBS (flowsheet observations, 3497→0) — very likely an intentional, upstream
   data-quality exclusion, not a bug.** meds2text's `observation` domain is exported by default and
   has a dedicated, actively-maintained flowsheet parser wired into its default transform pipeline
   — so the LUMIA *generator* doesn't exclude flowsheets by default. But a separate, *earlier*
   pipeline stage (STARR-OMOP → MEDS extraction, `_get_stanford_transformations()` in
   `meds2text/src/meds2text/transforms/core.py:368-386`) unconditionally strips
   `STANFORD_OBS/Flowsheet` events via `_remove_flowsheets` (`core.py:352-363`), with a documented
   reason: *"Flowsheets in STARR-OMOP have known timing bugs, making them unsuitable for use as
   either features or labels."* This is the **default** transform set for any STARR-OMOP extraction,
   introduced by a year-old, settled commit (`d624f3c`, 2025-07-02) — not something invented for
   this corpus. The legacy `patient_string` path is a structurally *separate* live pipeline
   (`src/data_tools/OMOP_meds_query/remove_imaging_report.py` → a direct `meds_reader.SubjectDatabase`
   query — already flagged in the original 2026-07-06 roadmap review as a "divergent DB-refetch
   fork" being retired, not re-homed) that most plausibly draws from a MEDS build that didn't go
   through this cleanup. Confirmed structurally too: for the specific run that showed the gap
   (`no_image`/`timeline` variant, `select=[]` — **zero filtering applied**), `parse_lumia`'s event
   walk doesn't discriminate by code/table at all, so if flowsheet-coded events existed in the
   parsed `.xml`, they'd have flowed straight through with no filter to catch them. Their total
   absence despite zero filtering means they're not in the file to begin with.
4. **Race** doesn't appear in `markup.md`'s documented `<demographics>` fields at all (only
   `ethnicity`/`gender` are listed) — but `markup.md` is already known to be an incomplete
   description of the real schema (it also documented `type` instead of the real `table` attribute,
   and originally only listed `type/code/name/note_id/provider_id/care_site_id` before Phase 0's
   schema recheck found more). Whether real `.xml` files carry a race field beyond what's documented
   needs an empirical VM check, not an assumption either way.

**Phil's decision (2026-07-17):** fix (1) and (2) now as confirmed bugs. Treat (3) — and (4) if the
VM confirms Race is genuinely absent from source — as accepted, expected, permanent divergences: no
further corpus-provenance chase, no `code_filter` tuning to try to fix them. Re-measure Step 2's
byte-diff after the fix to see the actual residual gap before deciding anything further.

## Goal

Close the demographics gap in `parse_lumia`, fix the `type`/`table` attribute bug, add a mechanism
so the existing byte-diff gate can pass despite the now-accepted flowsheet (and possibly Race)
divergence, and re-run Phase 1's decision gate so verification can proceed to Phases 2–3 of the
original plan.

## Approach

**1. VM-side inspection first (Phase 0.5 below) — don't speculate the demographic code/description
conventions on the Mac.** This mirrors the existing plan's own discipline
(`diff_golden.py`'s docstring: *"The real within-line field-order allowlist gets filled in once the
VM shows the actual LUMIA-render vs `patient_string` diffs — do not speculate it now"*). Two things
need VM confirmation before the `parse_lumia` fix can be written correctly:
   - A real `.xml` file's `<person>` block: does it repeat identically across every `<encounter>`
     for a patient (per the schema's nesting, it's declared per-encounter) or does it need deduping
     before synthesizing one `MEDS_BIRTH`/`Ethnicity`/`Gender` row per patient? Does it carry a race
     field beyond markup.md's documented `ethnicity`/`gender`?
   - The **already-banked** legacy "before" golden file's rendered `MEDS_BIRTH`/`Race`/`Ethnicity`/
     `Gender` lines (from Step 2's existing `*_no_image_legacy_small_golden.jsonl`, already on the
     results mount — no new query needed) — report back the exact `code`/`description` string
     *conventions* (format only, e.g. `"MEDS_BIRTH"` as a bare code with no description vs.
     `"Gender/FEMALE"`-style value-in-code) so the synthesized rows match legacy's rendering
     byte-for-byte. **PHI discipline: report the structural pattern only, never a real patient's
     actual line content** (same discipline as the existing plan's Phase-2/Step-3 PHI notes).

**2. Extend `parse_lumia` (`ehr.py`) to synthesize demographic pseudo-event rows from `<person>`.**
Once Phase 0.5 confirms the conventions: for each patient, read `<person>`'s `<birthdate>` and
`<demographics><ethnicity/gender>` (deduped across encounters if they repeat) and append rows to the
same event-row shape `_lumia_event_to_row` already produces (`time`/`code`/`description`/
`numeric_value`/`unit`/`text_value`), using `entry_ts` = the birthdate for `MEDS_BIRTH` and
whichever anchor legacy uses for `Ethnicity`/`Gender`/`Race` (Phase 0.5 confirms). These flow through
the *same* `get_llm_event_string` renderer unmodified — no special-casing needed there, since the
renderer treats any row generically by its `code`/`description`. If Phase 0.5 shows no race field in
real XML, Race joins the accepted-divergence list alongside STANFORD_OBS (step 4 below) rather than
being force-synthesized from nothing.

**3. Fix `_lumia_event_to_row`'s attribute read.** `ehr.py:106` currently reads
`attrib.get("type")`; change to `attrib.get("table")` to match meds2text's real emitted attribute
name (confirmed via `textify.py`'s `event_to_xml`/`attribute_order`). This doesn't affect the
demographics or STANFORD_OBS gap directly but fixes a silently-broken field any future
`type`-keyed filter (`note_type_filter`) would need.

**4. Add a "declared-excluded-class" comparison step to `diff_golden.py` — a new mechanism,
distinct from the existing `normalize_text` allowlist.** `normalize_text`'s own docstring/guardrails
are deliberately line-count- and event-order-preserving (formatting-only normalization); the
STANFORD_OBS (and possibly Race) divergence is a *permanent, event-count-changing* difference
(legacy has ~3497 flowsheet lines every time; live has zero every time), which the existing
mechanism was never designed to absorb and structurally cannot resolve via `code_filter` config
choice alone (confirmed: the "before" bank is generated by the *old*, pre-wiring code path via a
worktree at the parent SHA, which has no concept of `select`/`code_filter` at all — changing the
*after* side's config can only ever reduce *after*'s content, never make legacy's baked-in
`patient_string` exclude flowsheets it already contains). Add a small
`strip_excluded_lines(text, patterns)` helper: split a `dynamic_prompt`/`adapter_prompt_string`
value into lines, drop any line containing a declared substring (e.g. `"STANFORD_OBS/Flowsheet"`),
rejoin. Apply it to **both** `bv`/`av` before the existing `normalize_text` comparison in `diff()`
(safe either way — live already has zero matching lines, so stripping there is a no-op; legacy loses
its known, permanent extra lines). Confirmed sound: `get_llm_event_string`
(`src/data_tools/utils/meds_timeline_utils.py:220`) embeds the literal `code` string in every
rendered line (`f"{row['code']}{desc}"`), the same convention its own `exclude_report` flag already
substring-matches on (`:217`, `if 'STANFORD' in code_val: continue`) — so this is consistent with
how the codebase already reasons about these lines, not a new convention. Wire as a new
`--exclude-line-patterns` CLI flag (default empty, so existing non-EHR callers of `diff_golden.py`
are unaffected) documented clearly as *"declared-excluded class, not a formatting delta — a
permanent, accepted divergence"* so it can't be silently confused with `normalize_text`'s allowlist.

**5. Re-run Phase 1's decision gate** (does the `timeline` variant need `code_filter`?) with the fix
+ exclusion mechanism in place. This re-uses the *existing* plan's Phase 1 mechanism unchanged
(restricted-input worktree banking, `diff_golden --mode strict`) — only the new
`--exclude-line-patterns STANFORD_OBS/Flowsheet` (and `Race` if step 2 concluded it's a genuine
source gap) flag is added to the invocation.

## Files to Modify

- `src/context/adapters/ehr.py` — `parse_lumia` synthesizes demographic rows from `<person>`
  (Approach #2); `_lumia_event_to_row`'s `type` field reads `table` instead of `type` (Approach #3).
- `src/vista_run/diff_golden.py` — new `strip_excluded_lines()` + `--exclude-line-patterns` CLI flag,
  applied before the existing strict/allowlist text comparison (Approach #4). Update the module
  docstring to describe this as a distinct, declared-permanent-divergence mechanism.
- `docs/plans/vlm-step5-lumia-live-ehr-adapter.md` — record the resolved deviation; update Phase 1's
  Expected/Stop criteria to reference the new exclusion flag; note the accepted STANFORD_OBS
  (and possibly Race) divergence explicitly so it stops reading as an open mystery for the next
  reader.
- `docs/04-running-the-pipeline.md` / `docs/00-data-setup.md` — note the accepted STANFORD_OBS
  (+ Race, if confirmed absent from source) exclusion as expected behavior of the live LUMIA path,
  not a defect — alongside the existing "Fail-closed" note already added for Step 5.

## Open Questions

- Exact demographic `code`/`description` string conventions (VM confirms against the already-banked
  legacy golden file's format in Phase 0.5 — do not speculate on the Mac).
- Whether `<person>` repeats identically per-encounter or needs dedup logic (VM confirms against a
  real `.xml`).
- Whether Race is present anywhere in real LUMIA XML beyond `markup.md`'s documented
  `ethnicity`/`gender` (VM confirms) — if absent, it joins the accepted-divergence list; if present,
  it gets synthesized alongside the other demographic fields in Approach #2.

## Verification & VM handoff

**What runs on the VM** — all steps on **Claude-Code CPU `phil-sllm-01`**, same posture as the
original plan (weight-free rendering + read-only inspection; no GPU/hcpu split).

### Phase 0.5 — demographic-convention + schema confirmation (new, read-only, precedes the code fix)

- Inspect 1–2 real `.xml` files' `<person>` block structure (repetition-across-encounters, race
  field presence) — extend the existing Phase 0 schema-recheck script pattern
  (`docs/vm-status/2026-07-17-4989a20.md` Step 1's schema-reconfirmation snippet is the template).
- Inspect the **already-banked** `*_no_image_legacy_small_golden.jsonl` (from Step 2 of the prior
  handoff, already on the results mount) for its rendered `MEDS_BIRTH`/`Race`/`Ethnicity`/`Gender`
  lines — report the `code`/`description` *format* only (never a real patient's actual line).
**Expected:** exact code/description conventions for each demographic class; confirmation of
`<person>`'s repetition behavior; Race field presence/absence in real XML — all reported
structurally (PHI-clean, per the existing plan's PHI discipline).
**Stop:** `<person>` is missing entirely from real files (would contradict `markup.md` more deeply
than expected — hand back, don't guess a fix).
**Destructive:** no — read-only inspection of already-collected artifacts + existing `.xml` files.

*(Mac lands the `ehr.py` fix + `diff_golden.py` exclusion mechanism after Phase 0.5 reports back —
not before, per Approach #1's "don't speculate" discipline.)*

### Phase 1 — re-run the decision gate with the fix + exclusion mechanism

Re-run the existing plan's Phase 1 mechanism unchanged (restricted-input worktree banking,
`diff_golden --mode strict`), adding `--exclude-line-patterns STANFORD_OBS/Flowsheet` (+ `Race` if
Phase 0.5 confirmed it's genuinely absent from source).
**Expected:** demographic classes (`MEDS_BIRTH`/`Ethnicity`/`Gender`) now match between legacy and
live; the only residual difference is the declared-excluded class(es); Phase 1's original decision
gate (does `timeline` need `code_filter`?) resolves per the original plan's logic against this
(now much closer) residual.
**Stop:** any *other*, previously-unseen divergence appears — that would mean something beyond
demographics/flowsheets is at play (a genuine new class-3 deviation) — hand back rather than
patching further.

Then continue to the existing plan's **Phase 2** (allowlist gate) and **Phase 3** (human
visual-QA render) unchanged — both still apply, now against a much smaller expected residual.

## Landing & cleanup

Unchanged from the existing plan's Landing & cleanup section: single branch
(`feat/lumia-live-ehr-adapter`) off `main`, `/land` at the end once all phases are green *including*
Phil's Phase-3 human read of the rendered HTML. This re-plan doesn't change the landing gate, only
what Phases 1–2 verify against.
