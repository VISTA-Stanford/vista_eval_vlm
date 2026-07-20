Reference: docs/claude_ops.md

# VM verify — Step 5 closing verification (LOINC divergence declared, byte-diff gate re-run)

**Status: Handoff to VM** (2026-07-20)
**Branch:** `feat/lumia-live-ehr-adapter` (authored this session, uncommitted at render time —
commit + push before pulling on the VM; SHA set at commit time).
**Locator:** REPO `vista_eval_vlm` · BRANCH `feat/lumia-live-ehr-adapter` — **`git fetch` first**
(local branch is stale) · this doc `docs/vm-status/2026-07-20-loinc-closing-verification.md`.
Reach it: `git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only`
(shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-loinc-closing feat/lumia-live-ehr-adapter`).
**Machine posture:** authored on the planner Mac (no runtime). Run it on the **Claude-Code CPU**
box (`phil-sllm-01`), same posture as every prior handoff on this branch.
**Target machine:** Claude-Code CPU (`phil-sllm-01`). No hcpu/GPU leg — read-only diff over
existing golden JSONLs, no new `golden_harness` run, no writes.
**Plans:** [`vlm-step5-lumia-loinc-provenance-replan.md`](../plans/vlm-step5-lumia-loinc-provenance-replan.md#verification--vm-handoff)
— criteria source of truth (see its `## Resolution` section and Step 3).
**Prior handoffs:** [`2026-07-20-3206e84.md`](./2026-07-20-3206e84.md) — the blocked Step 2 run
this doc closes out (class-3 deviation: legacy MEDS db + `meds_reader` unavailable on this VM;
resolved by Phil on the Mac, see that doc's superseded banner).

## Why this doc

Round 4's Step 2 (raw LOINC event-count check) was blocked on `phil-sllm-01` — no legacy MEDS db
staged, no working `meds_reader` — and could not be resolved on the VM (`docs/vm-status/2026-07-20-3206e84.md`).
Phil decided on the Mac (2026-07-20, plan's `## Resolution` section) to accept Step 1's suggestive
provenance evidence (the only stamped MEDS extraction on this VM is aug-2025 vintage, vs. legacy's
described 2026-02-16 frozen snapshot) given Step 2 is a dead end without material infra investment.
The LOINC residual (2724/2779 excess lines, 98%) is now a **declared, permanent data-provenance
divergence** — same treatment as `STANFORD_OBS/Flowsheet` — and Step 5's landing gate pivots to
Phase 2's human visual-QA read as the primary correctness check. This doc renders the one concrete
verification action left: re-run the byte-diff gate with `LOINC/` added to the **already-existing**
`--exclude-line-patterns` mechanism (`diff_golden.py`, built for `STANFORD_OBS/Flowsheet` in round
2) — no new code — so the gate reads as intentional rather than failing.

**PHI discipline throughout** (unchanged from every prior handoff on this branch): report counts,
ratios, file paths, and exit codes only — **never** person_ids, raw timeline text, or dates.

## Step 0 — get the doc's dependencies onto the VM

```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only   # fetch FIRST — local branch is stale
ls docs/vm-status/2026-07-20-loinc-closing-verification.md   # confirms this doc landed
```

**Expected:** clean checkout at the branch tip; the `ls` succeeds (this doc is present).
**Stop:** the file is missing after `fetch`/`pull --ff-only` — the commit hasn't landed or hasn't
been pushed yet; do not proceed on a stale tree, report back.

## Step 1 — re-run the byte-diff gate with LOINC declared-excluded

Reuses the same golden JSONLs from round 3's run (`docs/vm-status/2026-07-20-a58f5f9.md`, Step
1d) — no new `golden_harness` bank, no new person selection. Identical invocation to round 3's,
with `LOINC/` added to `--exclude-line-patterns`.

```bash
cd <repo-root>/vista_eval_vlm

LEGACY=<results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_legacy_small_v2_golden.jsonl
LIVE=<results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_lumia_live_windowed_golden.jsonl
# verify both .meta.json: task/experiment/model_type/model_name/tag/limit match; STOP on mismatch

cd src
python -m vista_run.diff_golden "$LEGACY" "$LIVE" --mode strict \
  --exclude-line-patterns STANFORD_OBS/Flowsheet LOINC/ \
  --exclude-if-legacy-missing MEDS_BIRTH Ethnicity/ Race/ Gender/
cd ..
```

**Expected:** structural gates (`image_hashes`/`selected_indices`/`image_count`/
`path_tile_count`/`assembly_mode`) PASS as always. With `LOINC/` now declared-excluded alongside
`STANFORD_OBS/Flowsheet`, the text gate should read as **fully attributable** — either clean, or
only the already-declared residual classes remain (no unattributed residual lines).
**Stop:** (a) `.meta.json` provenance mismatch (task/experiment/model_type/model_name/tag/limit) —
precondition failure, report and hand back, don't reinterpret as a verdict. (b) an unattributed
residual remains even with all declared classes excluded — a real, still-undiscovered
render/adapter bug would remain; hand back to the Mac, don't declare a new class just to make the
gate pass.
**Destructive:** no — read-only diff over existing golden JSONLs, no writes.

## Report back

Step 1's PASS/FAIL per gate (structural + text), the exact `diff_golden` exit code, and — if any
residual remains — which lines/domains it falls under and whether they match an already-declared
class or are new. **Not a VM step, but note it in the readback for the record:** Step 5's landing
gate also requires Phil to have opened and read Phase 2's human-QA HTML render
(`context_viewer.py`) — that is a Mac/Phil action, not something to run here. **Never** person_ids,
raw timeline text, or dates from individual records.

## VM run results — readback on `phil-sllm-01`, 2026-07-20 · REPO `vista_eval_vlm` · BRANCH `feat/lumia-live-ehr-adapter` @ `9814abd` (pushed to `origin/feat/lumia-live-ehr-adapter`)

Read-only diff over the round-3 golden JSONLs (`legacy_small_v2` × `lumia_live_windowed`).
No new `golden_harness` bank, no writes. All characterization done with **masked scripts
(counts/format-skeletons only — no person_ids, timestamps, values, or line text)**.

- **Step 0:** ✅ checked out `9814abd`; this doc present at branch tip.
- **Step 1 — byte-diff gate (`diff_golden --mode strict`, `--exclude-line-patterns STANFORD_OBS/Flowsheet LOINC/`):** ❌ **GATE FAILURE (exit 1).**
  - **Precondition (`.meta.json`):** ✅ matched on `task`/`experiment`/`model_type`/`model_name`/`limit` (both `limit=20`); `tag` differs by design (`legacy_small_v2` vs `lumia_live_windowed`). Not a Stop (a).
  - **Structure gate:** ✅ PASS — `image_hashes` / `selected_indices` / counts / `assembly_mode` byte-identical.
  - **Text gate:** ❌ FAIL — **40 field mismatches** (20 shared rows × 2 fields: `dynamic_prompt`, `adapter_prompt_string`). This is **Stop (b)**: an unattributed residual remains even with `LOINC/` + `STANFORD_OBS/Flowsheet` declared-excluded.

- **⚠️ DEVIATION (class 3) — the residual is NOT a render/adapter bug, and NOT enumerable-excludable; it is systemic data-vintage divergence.**
  PHI-safe characterization of the residual (masked, counts only):
  - **Confirmed fully stripped:** 0 residual lines contain `LOINC/` or `STANFORD_OBS/Flowsheet` — the declared exclusions work; the residual is genuinely *other* content.
  - **Not formatting:** whitespace/case-normalization collapses the diff by **0** (6638→6638 legacy-only, 6188→6188 live-only). Character-class line-skeletons are **identical in shape on both sides** (`[<ts>] | <vocab/code (desc)> | <unit: value>`) — both arms render the same way.
  - **Not timestamp/value drift:** stripping the leading `[timestamp]` drops the per-side diff only ~30% (6638→4622 / 6188→4172); stripping timestamp **and** trailing value (→ bare event identity: vocab/code/desc) still leaves **4168 / 3718** unmatched. Of 7596 legacy / 7146 live non-excluded lines, **~55% of legacy events and ~52% of live events have no counterpart in the other arm at the event-identity level.**
  - **Diagnosis:** the same data-provenance vintage divergence already *declared* for LOINC (live = aug-2025 MEDS extraction; legacy = 2026-02-16 frozen snapshot — see `## Resolution` and prior handoff `2026-07-20-3206e84.md`) **generalizes to the entire remaining timeline across all vocabularies**. LOINC was merely the single largest class. A strict per-line byte-diff therefore **cannot be salvaged by declaring more excluded classes** — the divergence spans ~half of every arm's events, not an enumerable set of domains.
  - **Per Stop (b): did NOT declare a new exclusion class to force the gate to pass.** Handing back to the Mac.

- **Net:** ❌ **BLOCKED — back to planner (Mac).** Step 1's byte-diff gate cannot pass and cannot be made to pass by exclusion; this is the class-3 finding that confirms the plan's own pivot — **retire the strict byte-diff as Step 5's landing gate for the LUMIA-live arm; rely on Phase 2's human visual-QA read (`context_viewer.py`) as the primary correctness check** (a Mac/Phil action, not run here). The Mac re-enters plan mode to formalize retiring/downgrading the gate for this arm.

