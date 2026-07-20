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

## VM run results

_(left empty by the planner; the executor fills this in readback mode)_
