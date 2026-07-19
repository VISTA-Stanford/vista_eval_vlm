Reference: docs/claude_ops.md

# VM Phase 1 re-run — demographics fix + flowsheet/demographic-gap exclusion mechanisms

**Status: Handoff to VM** (2026-07-18)
**Branch:** `feat/lumia-live-ehr-adapter`, fix landed at `fd4e831` (pushed to `origin`; HEAD may be a
commit or two ahead from doc touch-ups — verify ancestry, not exact match, see Step 0)
**Locator:** REPO `vista_eval_vlm` · BRANCH `feat/lumia-live-ehr-adapter` — **`git fetch` first** (local branch is stale) · this doc `docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md`. Reach it: `git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only` (shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-lumia-live1 fd4e831`).
**Machine posture:** authored on the planner Mac (no runtime). Everything below has **never executed** — run it on the **Claude-Code CPU** box (`phil-sllm-01`, holds the `/mnt/su-vista-*` PHI mounts). Run + readback co-located there, same posture as the prior two handoffs on this branch.
**Target machine:** Claude-Code CPU (`phil-sllm-01`). No hcpu/GPU leg.
**Plans:** [`vlm-step5-lumia-demographics-flowsheet-replan.md`](../plans/vlm-step5-lumia-demographics-flowsheet-replan.md#verification--vm-handoff) — criteria source of truth (Phase 1, updated 2026-07-18). Base plan: [`vlm-step5-lumia-live-ehr-adapter.md`](../plans/vlm-step5-lumia-live-ehr-adapter.md#phase-1--decision-gate-does-the-timeline-variant-need-code_filter) (Phase 1's Expected/Stop, updated in lockstep).
**Prior handoffs:** [`2026-07-17-4b5239e.md`](./2026-07-17-4b5239e.md) — Phase 0.5 readback (all 3 Open Questions resolved + the 7/20 demographic-omission asymmetry flagged); this doc is the Mac's response to that readback's "Report back / next" section.

## Why this doc

Phase 0.5 confirmed the exact demographic-render conventions and flagged one new asymmetry (7/20
sampled legacy golden rows carry no demographic lines despite full `<person>` source data — a
legacy-vintage omission, not a bug). The Mac has now:

1. Fixed `parse_lumia` (`src/context/adapters/ehr.py`) to synthesize `MEDS_BIRTH`(×2)/`Ethnicity`/
   `Race`/`Gender` pseudo-event rows from each patient's `<person>` block, anchored at `<birthdate>`,
   in the exact order Phase 0.5 confirmed matches legacy (`_demographic_rows`).
2. Fixed `_lumia_event_to_row`'s attribute read (`type=` → `table=`, matching meds2text's real
   emitted attribute name).
3. Added two new `diff_golden.py` mechanisms, both applied to **both** BEFORE/AFTER text before
   comparison, distinct from the existing `normalize_text` formatting allowlist:
   - `--exclude-line-patterns` — unconditional, class-level (for `STANFORD_OBS/Flowsheet`, which
     legacy always has and live never will — a permanent upstream-pipeline exclusion).
   - `--exclude-if-legacy-missing` — **new**, per-row conditional (for the demographic classes):
     strips a pattern from both sides only for rows where BEFORE has none of it at all; rows where
     BEFORE has the class are still verified byte-for-byte. This is Phil's resolution to the 7/20
     asymmetry — verify demographics wherever legacy has a baseline, don't penalize live for
     rendering demographics on rows where legacy's own vintage never captured them, and don't
     blanket-suppress the class everywhere (which would mean Phase 1 never actually proves the fix
     is correct for the 13/20 that do have a baseline).

This doc re-runs Phase 1's decision gate (does the `timeline` preset variant need `code_filter`?)
with the fix + both exclusion mechanisms in place, to see the real residual gap. Phases 2–3 are
**out of scope for this doc** — they depend on Phase 1's outcome (which variant `code_filter`
config to bank against) and get their own handoff once Phase 1 resolves.

**PHI discipline throughout:** report counts, field names, exit codes, and the decision-gate
outcome back into this doc's readback section — **never** raw timeline text, person_ids, or
`diff_golden.py`'s printed BEFORE/AFTER previews (it prints those on failure — summarize instead).

## Step 0 — get the fix onto the VM

The fix is committed + pushed as of this handoff (three commits: `ehr.py`, `diff_golden.py`, docs
+ this doc). On the VM:

```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only   # fetch FIRST — local branch is stale
git merge-base --is-ancestor fd4e831 HEAD && echo "OK: fix landed"   # ancestry, not exact SHA — later doc touch-ups move HEAD past fd4e831
```
**Env:** reuse the existing hand-augmented golden-harness venv (`~/code/vista_eval_vlm/.venv`,
Python 3.11, yaml/torch/transformers already present) per the Phase-0.5 readback's carry-note —
no bare `uv sync` needed. The LUMIA mirror (9350 `.xml`) and `configs/all_tasks.viewer.vm.yaml`
from the prior handoff are still on `phil-sllm-01` and don't need re-fetching.
**Expected:** clean checkout at the committed SHA.
**Stop:** `git rev-parse --short HEAD` doesn't match this doc's header SHA — the commit/push never
landed or wasn't fetched; do not proceed on a stale tree.

## Step 1 — Phase 1: re-run the decision gate with the fix + exclusion mechanisms

Reuse Phase 1's already-validated restricted-input mechanism unchanged (the pre-wiring worktree at
the parent SHA, the same restricted-to-covered-`person_id` input CSV from the prior handoff — no
need to rebuild it if `../vista_eval_vlm-prewiring` or the restricted scratch config are still on
disk; recreate per the base plan's Phase 1 section if not).

```bash
# --- "before" bank (pre-wiring parent 6ded1e6, in a throwaway worktree — recreate if removed) ---
git worktree add ../vista_eval_vlm-prewiring 6ded1e6   # skip if it already exists
cd ../vista_eval_vlm-prewiring/src
python -m vista_run.golden_harness \
  --config <ABS_PATH_TO_RESTRICTED_SCRATCH_CONFIG> \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image \
  --tag legacy_small_v2 --limit 20
cd ../../vista_eval_vlm

# --- "after" bank (this branch, with the fix, select=[] as currently committed) ---
cd src
python -m vista_run.golden_harness \
  --config <RESTRICTED_SCRATCH_CONFIG> \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image \
  --tag lumia_live_fixed --limit 20

# --- diff, with both new exclusion mechanisms ---
python -m vista_run.diff_golden \
  <results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_legacy_small_v2_golden.jsonl \
  <results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_lumia_live_fixed_golden.jsonl \
  --mode strict \
  --exclude-line-patterns STANFORD_OBS/Flowsheet \
  --exclude-if-legacy-missing MEDS_BIRTH Ethnicity/ Race/ Gender/
cd ..
```

**Expected/Stop (resolve inline where the plan pre-encodes it as a decision gate; hand back on
anything else):**
- Index-set mismatch even against the restricted input → **STOP** (a distinct, unexpected
  precondition failure — hand back, don't interpret as a `code_filter` verdict).
- **Strict PASS** → `timeline` needs **no** `code_filter` — `presets.py`'s `_ehr_block` already has
  `select = []` for the `timeline` variant — done, no code change needed. A PASS here means the only
  differences the two exclusion mechanisms absorbed were `STANFORD_OBS` (always) and demographics on
  rows where legacy had no baseline (per-row) — report which of the 20 rows triggered
  `--exclude-if-legacy-missing` (should be close to the ~35% Phase 0.5 sampled) and confirm no *other*
  residual fired.
- **Strict FAIL** (index sets matching, residual NOT fully explained by the two exclusion mechanisms)
  → re-bank the **after** side with `code_filter(exclude_stanford=True)` added
  (`--tag lumia_live_filtered`), re-diff strict with the same exclusion flags. Whichever configuration
  passes byte-identical is the answer — edit `src/context/presets.py`'s `_ehr_block` (`timeline`
  branch's `select = []` → `select = [{"fn": "code_filter", "exclude_stanford": True}]`), update its
  docstring to say the gate resolved this way, commit on this branch, re-run.
- **Both** pass strict → byte-identical on this sample either way (`code_filter` is a no-op here) —
  keep `select = []` (simpler, maximally permissive) and note this outcome explicitly.
- **Neither** passes strict, and the residual is NOT fully explained by the two exclusion mechanisms
  → **STOP** (class-3 — a third variable is at play; hand back, don't guess).
**Clean up the worktree after:** `git worktree remove ../vista_eval_vlm-prewiring` once both banks
are done (or leave it if a Phase 2 re-run at larger N will reuse it shortly — note which you did).
**Destructive:** no — local golden-bank writes under the git-ignored results tree only.

## Report back

Pass/fail vs Step 1's Expected block; the decision-gate outcome (`select=[]` kept, or
`code_filter` added — with the reasoning branch that resolved it); the count of rows where
`--exclude-if-legacy-missing` actually stripped anything (sanity-check against Phase 0.5's ~35%);
any `presets.py`/`diff_golden.py` edits made + committed on this branch; any class-1 in-lane
corrections. A class-3 deviation gets its own `⚠️ DEVIATION` block per `docs/claude_ops.md` / this
skill's Deviation workflow — STOP, don't improvise past it.

## VM run results — readback on `phil-sllm-01`, 2026-07-18 · REPO `vista_eval_vlm` · BRANCH `feat/lumia-live-ehr-adapter` @ `afeed41` (run + readback co-located on the Claude-Code CPU box)

**Net: ❌ BLOCKED — class-3 deviation. Phase 1's strict gate FAILED on all 20 rows / both text fields even with the demographics fix + both exclusion mechanisms active. The residual is NOT the STANFORD_OBS + demographic-vintage-gap the plan's mechanisms were designed to absorb — it is a pervasive, whole-timeline render divergence (a `VALUE:` vs `NOTE:` field-label mismatch on every lab/measurement line, plus live emitting ~2.36× the events). This supersedes the Phase-0.5 demographics root-cause as the primary blocker. Back to the Mac to re-plan the live-adapter render alignment. `code_filter` was analytically ruled out (not run — see below).**

- **Step 0:** ✅ synced `feat/lumia-live-ehr-adapter` → `afeed41`; `git merge-base --is-ancestor fd4e831 HEAD` OK ("fix landed"). Reused the existing hand-augmented golden venv (`~/code/vista_eval_vlm/.venv`, py3.11). LUMIA mirror (9350 `.xml`) + `configs/all_tasks.viewer.vm.yaml` present, not re-fetched.
- **Step 1 — banking:**
  - **BEFORE** (`legacy_small_v2`): ✅ banked from a fresh prewiring worktree at `6ded1e6` (verified ancestor of HEAD; legacy render path `meds_timeline_utils` is unchanged 6ded1e6→HEAD). 20 rows, config `all_tasks.viewer.vm.yaml` + `--limit 20`. Matched 1238/1238 timeline rows.
  - **AFTER** (`lumia_live_fixed`): ✅ banked on `afeed41` (with the fix, `select=[]` as committed). 20 rows, same config + `--limit 20`. Matched 1238/1238. Ran detached (~10-min slow-`/mnt` XML parse — a foreground 10-min timeout killed the first attempt with a 0-byte file; re-ran unbuffered in background, exit 0).
  - *In-lane note:* used the plain `all_tasks.viewer.vm.yaml` (the config both prior banks in this workstream used, per their `.meta.json`) as the "restricted scratch config" rather than building a new covered-only CSV. Coverage is 100% (1238/1238 matched both sides), so `--limit 20` draws the identical index set for both banks — the diff confirmed this (see below). No separate restriction plumbing was needed.
- **Step 1 — diff** (`diff_golden --mode strict --exclude-line-patterns STANFORD_OBS/Flowsheet --exclude-if-legacy-missing MEDS_BIRTH Ethnicity/ Race/ Gender/`): **❌ GATE FAILURE (exit 1)**.
  - `[PASS] structure` — image_hashes / selected_indices / counts / assembly_mode byte-identical (Gates 1/2 pass).
  - **Index sets MATCHED** — `shared indices: 20`, 0 only-in-BEFORE / 0 only-in-AFTER. The precondition-failure STOP did **not** fire.
  - `[FAIL] text drift (strict): 40 field mismatches` — i.e. **all 20 rows × both text fields** (`dynamic_prompt`, `adapter_prompt_string`) diverge. (`--max-report 20` capped the printed rows; total is 40.)
- **Decision gates (class 2):** Phase 1's index-set precondition gate → **did not fire** (sets matched, coverage 100%). The `code_filter` fork was evaluated and **resolved to "not applicable" analytically** — see the deviation below; I did **not** re-bank with `code_filter(exclude_stanford=True)` because the dominant residual is non-STANFORD and even excluding all STANFORD_* classes leaves thousands of divergent LOINC/SNOMED/RxNorm lines. Running it would have cost another ~10-min slow-mount bank for a foregone FAIL.
- **In-lane corrections (class 1):** none — no code/config changed on this leg (banking + diff only). Prewiring worktree removed after banking (`git worktree remove ../vista_eval_vlm-prewiring`); `legacy_small_v2` + `lumia_live_fixed` banks persist on the results mount.

### ⚠️ DEVIATION (class 3) — the residual is a whole-timeline render mismatch, not the two accepted-divergence classes

**Expected (per Step 1's gate):** after the demographics fix + both exclusion mechanisms, a strict PASS (or a residual attributable only to STANFORD_OBS + the ~35% legacy-vintage demographic gap, i.e. `code_filter`-resolvable).

**Found:** near-total divergence. PHI-safe structural characterization of the two banks (all values digit-masked, descriptions collapsed — no timeline text, dates, values, or person_ids surfaced):

1. **Line-level overlap is ~8%.** Of 5,841 legacy timeline lines (STANFORD_OBS excluded), only 454 (7.8%) appear byte-identically in the live render; live has 13,774 lines (≈2.36× legacy). Not a shared-core-plus-extra and not a strict superset — genuinely divergent in both directions.
2. **Dominant cause — `VALUE:` vs `NOTE:` field-label mismatch on measurement lines.** Digit-masked line-template histogram:
   - legacy top templates: `[…] | LOINC/… (…) | VALUE: D.D` (×2071), `… | VALUE: D.D | NOTE: D.D` (×460), `SNOMED/… | VALUE: D.D` (×222) — legacy renders numeric lab/measurement results under a **`VALUE:`** field.
   - live top templates: `[…] | LOINC/… (…) | NOTE: D` (×4486), `… | NOTE: D.D` (×2561), `SNOMED/… | NOTE: D` (×215) — live renders the **same** lab lines under a **`NOTE:`** field, and never emits `VALUE:` at all.
   - Because labs/measurements are the bulk of the timeline, this single field-label convention difference makes essentially every measurement line diverge byte-for-byte on its own.
3. **Secondary — extra era suffixes + event-scope.** Live emits `RxNorm/… | NOTE: start|end` and `NUCC/… | NOTE: start|end` (drug/provider era bounds) that legacy lacks; and even *shared* line-templates differ in count (e.g. `SNOMED/… | NOTE: N` B180 / A352), i.e. a scope/truncation/dedup difference on top of the formatting one.

**Why it blocks:** the plan assumed the only Phase-1 residuals would be STANFORD_OBS (always-excluded) + demographics on the legacy-vintage-gap rows (per-row excluded). Neither mechanism — nor `code_filter(exclude_stanford)` — touches the `VALUE:`/`NOTE:` label convention or the ~2× event scope, which are the actual blockers. The demographics fix landed correctly but addressed only ~5 lines/patient; it was never the primary divergence. Resolving this needs a code change to the **live adapter's render** (align the measurement value field to legacy's `VALUE:` convention, and reconcile event scope/era suffixes) and/or a decision about which render is canonical — a plan-level call, not an executor one.

**Escalating to planner.** Recommended Mac next step: re-enter plan mode; a masked-line inspection (Phase-0.5 style) of `_lumia_event_to_row` / the live render path in `src/context/adapters/ehr.py` to confirm the `VALUE:`-vs-`NOTE:` mapping and the event-scope delta, then supersede this doc with a render-alignment plan. `legacy_small_v2` + `lumia_live_fixed` banks remain on the results mount for that inspection.

## Resume block

```
Resume ▸ vista_eval_vlm   → re-plan on the Mac (class-3 deviation)
  REPO   vista_eval_vlm
  BRANCH feat/lumia-live-ehr-adapter
  DOC    docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md
  SHA    afeed41 (readback ⚠ UNPUSHED until /commit-review)
  SYNC   git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only   # run FIRST; verify: git rev-parse --short HEAD → (readback sha)
```
