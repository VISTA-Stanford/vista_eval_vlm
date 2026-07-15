Reference: research-skills/claude_ops.md

# VM smoke — 3b CT-adapter dissolution: C3 bank + C4 byte-identity diff (feb26 / v1_5)

**Status: Handoff to VM** (2026-07-15)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **3b is uncommitted on the Mac; commit + push first, SHA set at commit time.** The banked "before" (C1) is frozen at SHA `1b24507`; this doc's "after" (C3) runs on the post-3b SHA.
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **`git fetch` first (even on `main`)** · this doc `docs/vm-status/2026-07-15-3b-ct-adapter-golden-diff.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull --ff-only` (shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-3b <post-3b-sha>`).
**Machine posture:** authored on the planner Mac (`DNa825009.SUNet`, no runtime/data/creds). Every step is the **weights-free golden harness** (`golden_harness.py`, never loads weights) + `diff_golden.py` → runs entirely on the **Claude-Code CPU** box `phil-sllm-01` (holds the `/mnt/su-vista-*` mounts + BQ creds; runs Claude Code → readback here directly). No GPU / high-throughput leg, no run-vs-readback split.
**Target machine:** Claude-Code CPU (`phil-sllm-01`) for all steps.
**Plan:** [`vlm-ct-feb26-v1_5-golden-rebaseline.md`](../plans/vlm-ct-feb26-v1_5-golden-rebaseline.md#verification--vm-handoff) ← criteria source of truth (Phase 3, Step 4). See its **"The two axes"** + **"Axis C"** for why the diff isolates the refactor.
**Prior handoffs:** [`2026-07-14-rungs1-2-v1_5-stage-and-prebank.md`](./2026-07-14-rungs1-2-v1_5-stage-and-prebank.md) — Phases 1+2 (Steps 0–3) GREEN; **C1 `legacy_feb26` banked for both models at SHA `1b24507`**, 1,238 rows each (943 CT-bearing gemma / 944 intern).
**Implementation review:** [`docs/plans/reviews/vlm-ct-feb26-v1_5-golden-rebaseline-oom-loadpath-implementation-feedback.md`](../plans/reviews/vlm-ct-feb26-v1_5-golden-rebaseline-oom-loadpath-implementation-feedback.md) — the load-path fix (`1b24507`) that C1 rides; behavior-preservation CLEAN.

## Why this doc — Phase 3, the Mac-interlude "after"

Doc 1 banked the CT axial **"before"** golden (C1, tag `legacy_feb26`) on the legacy `__getitem__` CT branch at SHA `1b24507`. On the Mac, **3b (C2) is now implemented**: the legacy per-experiment CT branch (30/50/10 evenly-spaced axial slices + windowing) is **dissolved into the `CTAdapter`** (`src/context/adapters/ct.py`), which `__getitem__` now drives by handing it the lazy `ct_img.dataobj` proxy. The blob resolution + NIfTI load stay in the caller, unchanged (the OOM fix from `1b24507` is untouched).

This handoff banks the **"after"** golden (C3, tag `adapter_feb26`) on the same v1_5/feb26 substrate and **diffs it byte-for-byte against C1** (C4). Both sides run on **feb26** with the **same git-ignored config + copied v1_5 CSV + cohort** — only the *code* moved — so the diff isolates the 3b refactor. A clean `RESULT: ALL GATES PASS` per model retires the CT dissolution; the CTAdapter becomes the load path.

**Banked-from-prior (do NOT re-run):** C1 `legacy_feb26` goldens (both models) + the Phase-1 EHR `no_image` baseline are frozen on the PHI mount at `1b24507`. 3b changed only CT slice/windowing code — it does **not** touch the EHR/`no_image` path or the config/CSV/cohort — so the banked "before" stays valid. Step 0 only *confirms they exist*; it never regenerates them.

**Scope note — Step 5 (gate-3 EHR / D3) is deferred, NOT in this doc.** The plan's Phase 3 pairs Step 4 (this) with Step 5 (LUMIA-live `no_image` gate-3, `--mode allowlist`). But the runtime still renders the timeline from the passthrough `patient_string` CSV — the **LUMIA-live render path is the deferred LUMIA-direct-loader follow-up** (`diff_golden.py:31` "the actual LUMIA-render vs `patient_string` diffs — do not speculate it now"). `diff_golden.py` itself documents `--mode strict` as "the passthrough **post-3b** run" and `--mode allowlist` as "the **LUMIA-live** run." 3b is exactly that passthrough post-3b run, so **`--mode strict` (Step 4) is the applicable gate**; gate-3 (Step 5) rides the future EHR-adapter/LUMIA work. *(Flagged at the Mac sign-off gate.)*

## Step 0 — get the artifacts onto the VM + confirm the banked "before"
```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull --ff-only  # fetch FIRST — local branch is stale, even on main
git rev-parse --short HEAD    # must be the post-3b SHA (set at commit time); if it's still 1b24507, the 3b commit didn't land — STOP
# Provision (this repo's real path — plain `uv sync` prunes to base deps; see Doc 1's in-lane correction):
uv pip install -e . && uv pip install -r requirements-default.txt

# Confirm 3b is present in the checkout (behavioral, not a property assert):
grep -q "from context.adapters.ct import CTAdapter" src/vqa_dataset.py && echo "[ok] CTAdapter wired into vqa_dataset"
grep -q "np.asarray(volume\[:, :, idx\], dtype=np.float64)" src/context/adapters/ct.py && echo "[ok] adapter float64 slice cast present"

# Confirm the banked C1 "before" goldens exist on the PHI mount (banked-from-prior; do NOT re-run):
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
find "$BASE" -name '*_axial_all_image_legacy_feb26_golden.jsonl' -printf '%s\t%p\n'   # expect 2 files (gemma + intern), each non-zero
```
**Expected:** `git rev-parse --short HEAD` == the pushed post-3b SHA (NOT `1b24507`); `uv` provisioning resolves (torch/transformers present); both `[ok]` greps print; **exactly two** non-empty `*_legacy_feb26_golden.jsonl` files (one per model) are found on the mount. `indexed_gzip==1.10.3` installed (from `requirements-default.txt`).
**STOP:** HEAD still at `1b24507` (3b never committed → nothing to diff against — commit + push 3b on the Mac first); a `[ok]` grep fails (3b not in the checkout — wrong SHA/branch); fewer than two `legacy_feb26` goldens present (C1 not banked / wrong mount → cannot diff; re-check Doc 1's Phase 2 or the config's `results_dir`).

## Step 1 — C3: bank the axial "after" golden on feb26 (`adapter_feb26`)
Same two harness invocations as C1's Step 3, **only the tag changes** (`legacy_feb26` → `adapter_feb26`); **same git-ignored config, same copied v1_5 CSV, same cohort** — only the 3b code differs.
```bash
cd src
python -m vista_run.golden_harness --config ../configs/all_tasks.rungs12.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr --experiments axial_all_image --tag adapter_feb26
python -m vista_run.golden_harness --config ../configs/all_tasks.rungs12.vm.yaml \
  --type intern  --name OpenGVLab/InternVL3_5-8B-hf \
  --task progression_recurrence_free_survival_1_yr --experiments axial_all_image --tag adapter_feb26
git -C .. status --short   # golden must NOT appear (lives on the PHI mount, git-ignored)
```
**Expected:** per model a `…_axial_all_image_adapter_feb26_golden.jsonl` + `.meta.json`, non-empty, sorted by `(person_id, index)`; **1,238 rows each** (matching C1); `image_count > 0` for CT-bearing rows with `image_hashes` length == `image_count`; weight-free (no GPU/weights in logs); `git status` clean. **Report the per-model CT-bearing / no-image split** — it should match C1 (gemma 943/295, intern 944/294); a shifted split is a signal the CT path diverged and Step 2 will catch it.
**STOP:** any traceback; `row_count == 0` for a CT-bearing cohort; row count ≠ C1's 1,238 (cohort moved — the diff would be meaningless); a missing / zero-byte `.jsonl`/`.meta.json`; index set not unique-and-sorted; an unexpected retrieval-skip / "no data" for `axial_all_image`; weights loading; a dirty repo containing golden output.

## Step 2 — C4: byte-identity diff, C1 vs C3, per model (`--mode strict`)
For each model, diff its frozen `legacy_feb26` "before" against the fresh `adapter_feb26` "after". `strict` is the `diff_golden.py` default (stated explicitly so the run doesn't rely on it); it is the correct mode for this passthrough post-3b run.
```bash
cd src
BASE=/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden
find "$BASE" -name '*_axial_all_image_adapter_feb26_golden.jsonl' | while read -r after; do
  before="${after/_adapter_feb26_/_legacy_feb26_}"        # the paired C1 file, same model dir
  echo "=== model dir: $(basename "$(dirname "$after")") ==="
  test -f "$before" || { echo "MISSING before: $before"; continue; }
  python -m vista_run.diff_golden "$before" "$after" --mode strict
  echo "(diff exit: $?)"
done
```
**Expected:** for **each** model, the shared-index set is identical (same indices both sides); **Gate 1** structure byte-identical over the tool's full hard set — `image_hashes`, `selected_indices`, `image_count`, `assembly_mode`, `path_tile_count` (`None` for CT rows) — and **Gate 2** `image_hashes` identical per model; each run prints `RESULT: ALL GATES PASS` and exits 0. Two models diffed → **two** `ALL GATES PASS`.
**STOP:** any `RESULT: GATE FAILURE` / non-zero exit (Gate-1/Gate-2 drift — the CT dissolution is not a no-op → **hard class-3 halt**, hand back to the Mac); an **index-set mismatch**; a **`[WARN] no shared indices`** (empty intersection cannot *prove* a no-op — treat as failure, not pass); a `MISSING before` line (Step-0 pairing wrong — re-check the banked C1 paths). **Do NOT `--lenient`** on the gate run (it can exit 0 despite text drift).

## Report back
Under `## VM run results`, append per-step pass/fail against each **Expected** block:
- **Step 0:** post-3b HEAD SHA confirmed (≠ `1b24507`); both 3b greps `[ok]`; count + sizes of the two banked `legacy_feb26` goldens.
- **Step 1 (C3):** per (model) row_count (== 1,238?), CT-bearing / no-image split (vs C1's 943/295 · 944/294), `image_hashes`-length match, meta==jsonl, sorted, `git status` clean.
- **Step 2 (C4):** per model, `RESULT:` line + exit code; on failure, the **counts of mismatched fields by gate** and **affected indices only** (NOT the values). If ALL GATES PASS both models → 3b is a proven no-op; the CT dissolution is retired.
- **Net:** both models `ALL GATES PASS` → hand back to the **Mac** to `/land` (per the plan's *Landing & cleanup*; gate: `/review-implementation` clean + `/phi-vet` + `/review-plan` sign-off). Any gate drift → **class-3 halt**, back to the Mac to re-plan the refactor.

**PHI (per the plan's Verification PHI rule):** counts / field-names / affected-indices only — **never** paste golden rows, timelines, or `diff_golden.py`'s BEFORE/AFTER `_preview` output (`diff_golden.py:169-170,185-186` prints raw `dynamic_prompt`/`adapter_prompt_string`, which carry timeline PHI). Raw DICOM Study/Series UIDs stay on the `su-vista-*` PHI mount. Golden + diff output are git-ignored on the mount; `/phi-vet` gates the readback commit.

## VM run results — readback on `phil-sllm-01`, 2026-07-15 · REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` @ `f1fd8a8` (pushed to `origin/<branch>`)

**PARTIAL run — gemma C3 banked + verified; intern C3 (Step 1) and byte-diff (Step 2) DEFERRED, handed back to the Mac for a routing decision (see handback below).**

- **Step 0:** ✅ post-3b HEAD `f1fd8a8` confirmed (≠ `1b24507`); both 3b greps `[ok]` (`CTAdapter` wired into `vqa_dataset`; adapter float64 slice cast present). Provisioning resolved via `uv pip install -e . && uv pip install -r requirements-default.txt` — `indexed_gzip==1.10.3`, `torch==2.8.0`, `transformers==4.57.1` present. Both banked C1 `legacy_feb26` goldens found on the PHI mount and non-empty: intern **289,990,550 B**, gemma **289,997,040 B** (exactly two, one per model).
- **Step 1 (C3) — gemma (`google/medgemma-1.5-4b-it`):** ✅ wrote `…_axial_all_image_adapter_feb26_golden.jsonl` + `.meta.json` (non-empty). **row_count 1,238** (== C1). **CT-bearing 943 / no-image 295** (== C1's 943/295 — split unchanged). `image_hashes` length == `image_count` for **all** rows. Rows unique on `index` and **sorted by `(person_id, index)`**. `meta.json` `row_count`/`tag` (`adapter_feb26`) agree with the jsonl. `git status` clean (golden git-ignored, lives on PHI mount). **Preliminary no-op signal:** the C3 gemma jsonl is **289,997,040 B — byte-for-byte the same size as C1's gemma legacy golden** (not proof; Step 2's diff is the gate).
  - **3 substrate load errors during the CT read** (all handled gracefully; run continued to 1,238 rows): 2× GCS `404 No such object` (missing NIfTI blobs on the `feb26` prefix) + 1× degenerate volume (`height and width must be > 0`). These are **substrate defects on the shared feb26/v1_5 mount, unchanged since C1's `1b24507` bank** — so they hit the C1 "before" golden identically and do not, by themselves, invalidate the diff. Step 2 confirms.
- **Step 1 (C3) — intern (`OpenGVLab/InternVL3_5-8B-hf`):** ⏸️ **DEFERRED** — not started (see handback).
- **Step 2 (C4) — byte-identity diff:** ⏸️ **DEFERRED** — requires both models' C3 goldens.
- **⚠️ Handback (routing / class-3):** the gemma leg took **~85 min wall-clock at 0% CPU** — the weight-free golden bank is entirely **I/O-bound on the slow `/mnt` small-file mount** (~13 CTs/min over 943 CT-bearing rows), not compute-bound. Per Phil, this bulk read belongs on a **high-throughput CPU box**, not the interactive Claude-Code CPU box that also does the diff readback. **Handing back to the Mac planner** to decide routing (re-scope the intern leg + diff to an hcpu run-then-readback split, or parallelize the CT reads) **before** the remaining intern C3 leg and the Step-2 diff run. The gemma C3 golden is banked on the PHI mount and stays valid regardless of where the intern leg runs.
- **Net:** gemma C3 GREEN and banked; **BLOCKED on the routing decision** for intern C3 + diff → back to the Mac. No design deviation in the refactor was observed (the diff that would prove/disprove the no-op has not yet run).

**PHI:** counts / field-names / byte-sizes only — no golden rows, timelines, prompts, or `diff` previews pasted; raw Study/Series UIDs stay on the `su-vista-*` PHI mount.

**Continued → [`2026-07-15-3b-intern-limited-golden-diff.md`](./2026-07-15-3b-intern-limited-golden-diff.md)** (2026-07-15) — routing handback RESOLVED (Phil): the intern leg is limited to a spot-check (`--limit 100`) rather than relocated to hcpu; this doc's gemma C3 golden is **banked-from-prior** (its C4 diff runs there). Not superseded — this doc's gemma result stands.
