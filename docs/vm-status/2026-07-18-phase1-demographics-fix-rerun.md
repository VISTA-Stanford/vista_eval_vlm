Reference: docs/claude_ops.md

# VM Phase 1 re-run — demographics fix + flowsheet/demographic-gap exclusion mechanisms

**Status: Handoff to VM** (2026-07-18)
**Branch:** `feat/lumia-live-ehr-adapter` (commit + push first, SHA set at commit time — see Step 0)
**Locator:** REPO `vista_eval_vlm` · BRANCH `feat/lumia-live-ehr-adapter` — **`git fetch` first** (local branch is stale) · this doc `docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md`. Reach it: `git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only` (shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-lumia-live1 <sha>`).
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

## Step 0 — commit the fix, get it onto the VM

The fix is uncommitted on the Mac as of this handoff. Commit it there first (`/commit-review`,
which this doc's author offers after landing the handoff doc + pointer), then on the VM:

```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only   # fetch FIRST — local branch is stale
git rev-parse --short HEAD   # must show the SHA this doc's header names once committed — if not, the fetch didn't land the handoff
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

## VM run results
_(left empty by the planner; the executor fills this in on readback)_
