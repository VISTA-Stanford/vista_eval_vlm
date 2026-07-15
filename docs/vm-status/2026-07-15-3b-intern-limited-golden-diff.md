Reference: research-skills/claude_ops.md

# VM smoke — 3b CT-adapter dissolution: gemma C4 (full) + intern C3/C4 (limited spot-check), feb26 / v1_5

**Status: Handoff to VM** (2026-07-15)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **this doc + the plan revision are uncommitted on the Mac; commit + push first, SHA set at commit time.** The 3b code (`e0f2b8d`) and the banked gemma C3 golden (`f1fd8a8`) are already pushed; this leg only adds the intern spot-check + both diffs.
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **`git fetch` first** (even on `main`) · this doc `docs/vm-status/2026-07-15-3b-intern-limited-golden-diff.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull --ff-only` (shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-3b-intern <post-revision-sha>`).
**Machine posture:** authored on the planner Mac (`DNa825009.SUNet`, no runtime/data/creds). Every step is the **weights-free golden harness** (`golden_harness.py`, never loads weights) + `diff_golden.py`. The intern leg is now a **limited spot-check (`--limit 100`, ~minutes)**, so everything runs on the **Claude-Code CPU** box `phil-sllm-01` (holds the `/mnt/su-vista-*` mounts + BQ creds; runs Claude Code → readback here directly). **No hcpu / GPU leg, no standalone runner script, no run-vs-readback split** — the 2026-07-15 routing deviation was resolved by *limiting* the intern read, not relocating it.
**Target machine:** Claude-Code CPU (`phil-sllm-01`) for all steps.
**Plans:** [`vlm-ct-feb26-v1_5-golden-rebaseline.md`](../plans/vlm-ct-feb26-v1_5-golden-rebaseline.md#verification--vm-handoff) ← criteria source of truth (Phase 3, Step 4, revised 2026-07-15). See its **"Deviation & re-plan (2026-07-15)"** for why intern is limited.
**Prior handoffs:** [`2026-07-15-3b-ct-adapter-golden-diff.md`](./2026-07-15-3b-ct-adapter-golden-diff.md) — Step 0 green + **gemma C3 `adapter_feb26` banked GREEN** (1,238 rows, split 943/295 == C1, byte-size identical to C1); intern C3 + diffs handed back to the Mac on a routing finding. **Banked-from-prior (do NOT re-run):** gemma C3 golden, both models' C1 `legacy_feb26` before-goldens, the Phase-1 EHR `no_image` baseline — all frozen on the PHI mount.

## Why this doc

The 3b CT dissolution (legacy per-experiment CT slice-select + windowing → `CTAdapter`, driven by the lazy `ct_img.dataobj` proxy) is supposed to be a behavior-preserving no-op on the imaging surface. The prior handoff banked the gemma "after" golden (C3) full and found it **byte-size identical** to the gemma C1 "before" — a strong preliminary no-op signal — but the actual byte-identity **diff** (C4) had not yet run, and the intern leg was deferred because the full-cohort read is I/O-bound at ~85 min.

Per Phil's 2026-07-15 decision (plan deviation note), the intern leg is now a **limited spot-check** rather than a second full 85-min read: the intern run re-reads *the same 943 feb26 CT volumes gemma already read*, differing only in the `grayscale` windowing branch (Codex-verified verbatim relocation). So this doc:
1. runs the **gemma C4 diff** (full-vs-full — the primary-model full-cohort proof), and
2. banks a **limited intern C3** (`--limit 100`) and runs the **intern C4 diff** against a matching `head -n N` slice of the full intern C1.

Two `RESULT: ALL GATES PASS` (gemma full + intern spot-check) → 3b is a proven no-op → the CT dissolution is retired → back to the Mac to `/land`.

## Step 0 — sync + confirm the banked-from-prior artifacts
```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull --ff-only  # fetch FIRST — local branch is stale, even on main
git rev-parse --short HEAD    # must be the post-revision SHA (set at commit time); it carries the revised plan Step 4 + this doc
# Provision (this repo's real path — plain `uv sync` prunes to base deps; see Doc 1's in-lane correction):
uv pip install -e . && uv pip install -r requirements-default.txt

# Confirm 3b is present (behavioral greps, unchanged since the prior handoff):
grep -q "from context.adapters.ct import CTAdapter" src/vqa_dataset.py && echo "[ok] CTAdapter wired into vqa_dataset"
grep -q "np.asarray(volume\[:, :, idx\], dtype=np.float64)" src/context/adapters/ct.py && echo "[ok] adapter float64 slice cast present"

# Confirm banked-from-prior goldens on the PHI mount (do NOT re-run their banks):
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
echo "--- C1 before-goldens (both models, legacy_feb26, expect 2, ~290 MB each) ---"
find "$BASE" -name '*_axial_all_image_legacy_feb26_golden.jsonl' -printf '%s\t%p\n'
echo "--- gemma C3 after-golden (adapter_feb26, banked in the prior handoff, expect 1) ---"
find "$BASE" -path '*medgemma*' -name '*_axial_all_image_adapter_feb26_golden.jsonl' -printf '%s\t%p\n'
```
**Expected:** `git rev-parse --short HEAD` == the pushed post-revision SHA; `uv` provisioning resolves (`indexed_gzip==1.10.3`, torch/transformers present); both `[ok]` greps print; **exactly two** non-empty `*_legacy_feb26_golden.jsonl` (gemma + intern C1) found; **exactly one** gemma `*_adapter_feb26_golden.jsonl` (the banked gemma C3, ~289,997,040 B) found. **STOP:** a `[ok]` grep fails (3b not in the checkout — wrong SHA/branch); fewer than two `legacy_feb26` goldens, or the gemma `adapter_feb26` golden absent (banked-from-prior missing → re-check the prior handoff / mount before proceeding).

## Step 1 — gemma C4: full byte-identity diff (`legacy_feb26` vs `adapter_feb26`, `--mode strict`)
gemma C3 is already banked (prior handoff); this only diffs it against gemma C1. `strict` is the `diff_golden.py` default (stated explicitly so the run doesn't rely on it).
```bash
cd src
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
AFTER=$(find "$BASE" -path '*medgemma*' -name '*_axial_all_image_adapter_feb26_golden.jsonl')
BEFORE="${AFTER/_adapter_feb26_/_legacy_feb26_}"    # paired gemma C1, same model dir
echo "before: $BEFORE"; echo "after : $AFTER"
test -f "$BEFORE" && test -f "$AFTER" || { echo "MISSING gemma golden pair"; }
python -m vista_run.diff_golden "$BEFORE" "$AFTER" --mode strict
echo "(diff exit: $?)"
```
**Expected:** shared-index set identical (all **1,238**); **Gate 1** structure byte-identical over the tool's full hard set — `image_hashes`, `selected_indices`, `image_count`, `assembly_mode`, `path_tile_count` (`None` for CT rows) — and **Gate 2** `image_hashes` identical; prints `RESULT: ALL GATES PASS`, exits 0. **STOP:** any `RESULT: GATE FAILURE` / non-zero exit (Gate-1/Gate-2 drift — the CT dissolution is not a no-op → **hard class-3 halt**, hand back to the Mac); an **index-set mismatch**; a **`[WARN] no shared indices`** (empty intersection cannot prove a no-op — failure, not pass); a `MISSING gemma golden pair` line (Step-0 pairing wrong). **Do NOT `--lenient`.**

## Step 2 — intern C3: bank the limited "after" spot-check (`adapter_feb26`, `--limit 100`)
Same invocation as the gemma bank, `--type intern` + `--limit 100`. `--limit` takes a deterministic `(person_id, index)` head, so these N rows equal the head of the full intern C1 (Step 3 pairs them).
```bash
cd src
python -m vista_run.golden_harness --config ../configs/all_tasks.rungs12.vm.yaml \
  --type intern --name OpenGVLab/InternVL3_5-8B-hf \
  --task progression_recurrence_free_survival_1_yr --experiments axial_all_image \
  --tag adapter_feb26 --limit 100
git -C .. status --short   # golden must NOT appear (lives on the PHI mount, git-ignored)

# Report the CT-bearing / no-image split of the sample (counts only — PHI-clean):
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
python - "$BASE" <<'PY'
import json, glob, sys
base = sys.argv[1]
f = [p for p in glob.glob(base + "/**/*_axial_all_image_adapter_feb26_golden.jsonl", recursive=True) if "InternVL" in p][0]
ct = noimg = 0
for line in open(f):
    r = json.loads(line)
    if r.get("image_count", 0) > 0:
        ct += 1
    else:
        noimg += 1
print(f"[SAMPLE] intern limited golden: {ct} CT-bearing / {noimg} no-image (total {ct + noimg}) -> {f}")
PY
```
**Expected:** writes `…InternVL…_axial_all_image_adapter_feb26_golden.jsonl` + `.meta.json`, non-empty; **row_count == 100** (`meta.json` agrees with the jsonl); rows unique on `index` and **sorted by `(person_id, index)`**; weight-free (no GPU/weights in logs); `git status` clean. **The `[SAMPLE]` line reports ≥ 30 CT-bearing rows** (`image_count > 0`) so the `grayscale` windowing branch is genuinely exercised on real volumes. *(Graceful substrate defects — the same 404 / degenerate-volume rows the prior gemma run hit — are fine; they land as `image_count == 0` on **both** golden sides and don't count toward the 30.)* **STOP:** any traceback; a missing / zero-byte `.jsonl`/`.meta.json`; index set not unique-and-sorted; weights loading; a dirty repo containing golden output; **< 30 CT-bearing rows** in the sample (the head landed CT-thin) → re-run Step 2 with `--limit 200` and re-check (a few extra minutes, still single-box).

## Step 3 — intern C4: limited byte-identity diff (limited `adapter_feb26` vs `head -n N` of full `legacy_feb26`, `--mode strict`)
Slice the full intern C1 before to the same N head rows (required — `diff_golden` fails closed on any index-set inequality), then diff.
```bash
cd src
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
AFTER=$(find "$BASE" -path '*InternVL*' -name '*_axial_all_image_adapter_feb26_golden.jsonl')
BEFORE_FULL="${AFTER/_adapter_feb26_/_legacy_feb26_}"               # full 1,238-row intern C1
N=$(wc -l < "$AFTER")                                              # actual limited-after row count (== 100 unless re-limited)
BEFORE_LTD="${BEFORE_FULL/_golden.jsonl/_limit${N}_golden.jsonl}"   # keeps _golden.jsonl suffix -> gitignore backstop
echo "N=$N"; echo "before_full: $BEFORE_FULL"; echo "before_ltd : $BEFORE_LTD"; echo "after      : $AFTER"
head -n "$N" "$BEFORE_FULL" > "$BEFORE_LTD"                        # first N by (person_id,index) == after's head
python -m vista_run.diff_golden "$BEFORE_LTD" "$AFTER" --mode strict
echo "(diff exit: $?)"
```
**Expected:** shared-index set identical over the **N** rows (the `head -n N` slice guarantees it — the diff's `BEFORE`/`AFTER` row counts both == N); **Gate 1** structure byte-identical + **Gate 2** `image_hashes` identical; prints `RESULT: ALL GATES PASS`, exits 0. **STOP:** any `RESULT: GATE FAILURE` / non-zero exit (Gate drift — 3b not a no-op → **hard class-3 halt**, back to the Mac); an **index-set mismatch** (the `head -n N` slice didn't pair — re-derive `N` from `wc -l "$AFTER"` and re-slice, do not proceed on a mismatched pair); a **`[WARN] no shared indices`**. **Do NOT `--lenient`.**

## Report back
Under `## VM run results`, append per-step pass/fail against each **Expected** block:
- **Step 0:** post-revision HEAD SHA confirmed; both 3b greps `[ok]`; count + sizes of the two `legacy_feb26` C1 goldens + the one gemma `adapter_feb26` C3 golden (banked-from-prior).
- **Step 1 (gemma C4):** `RESULT:` line + exit code; shared-index count (== 1,238?). On failure: counts of mismatched fields by gate + affected indices only (NOT the values).
- **Step 2 (intern C3):** row_count (== 100? or the re-limited N), the `[SAMPLE]` CT-bearing / no-image split (≥ 30 CT-bearing?), meta==jsonl, sorted, `git status` clean.
- **Step 3 (intern C4):** N; `RESULT:` line + exit code; shared-index count (== N?). On failure: mismatched-field counts by gate + affected indices only.
- **Net:** **both** `ALL GATES PASS` (gemma full + intern spot-check) → 3b is a proven no-op; the CT dissolution is retired → hand back to the **Mac** to `/land` (per the plan's *Landing & cleanup*; gate: `/review-implementation` clean + `/phi-vet` + `/review-plan` sign-off). Any gate drift → **class-3 halt**, back to the Mac to re-plan the refactor.

**PHI (per the plan's Verification PHI rule):** counts / field-names / affected-indices / byte-sizes only — **never** paste golden rows, timelines, or `diff_golden.py`'s BEFORE/AFTER `_preview` output (`diff_golden.py:169-170,185-186` prints raw `dynamic_prompt`/`adapter_prompt_string`, which carry timeline PHI). Raw DICOM Study/Series UIDs stay on the `su-vista-*` PHI mount. Golden + diff output are git-ignored on the mount; `/phi-vet` gates the readback commit.

## VM run results
_(left empty by the planner; the executor fills this in readback mode on `phil-sllm-01`)_
