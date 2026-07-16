Reference: docs/claude_ops.md

# VM smoke — Phase 2 config-context viewer + golden byte-identity regression

**Status: Handoff to VM** (2026-07-15)
**Branch:** `feat/phase2-config-context-viewer` (commit + push first — the implementation is authored this session but UNCOMMITTED on the Mac; **SHA set at commit time**)
**Locator:** REPO `vista_eval_vlm` · BRANCH `feat/phase2-config-context-viewer` — **`git fetch` first** (local branch is stale) · this doc `docs/vm-status/2026-07-15-phase2-config-context-viewer.md`. Reach it: `git fetch origin && git checkout feat/phase2-config-context-viewer && git pull --ff-only` (shared / dirty checkout, or non-ff → `git worktree add ../vista_eval_vlm-phase2 <sha>`).
**Machine posture:** authored on the planner Mac (no runtime). Everything below has **never executed** — run it on the **Claude-Code CPU** box (`phil-sllm-01`, holds the `/mnt/su-vista-*` PHI mounts + BQ creds + the model cache the golden harness already uses). Run + readback co-located there.
**Target machine:** all steps — Claude-Code CPU (`phil-sllm-01`). No hcpu/GPU leg, no runner-script split.
**Plans:** [`vlm-phase2-config-context-viewer.md`](../plans/vlm-phase2-config-context-viewer.md#verification--vm-handoff) — criteria source of truth. Impl review: [`reviews/vlm-phase2-config-context-viewer-implementation-feedback.md`](../plans/reviews/vlm-phase2-config-context-viewer-implementation-feedback.md) (Codex; Critical summarize-selector-chain fixed).
**Prior handoffs:** [`2026-07-15-3b-intern-limited-golden-diff.md`](./2026-07-15-3b-intern-limited-golden-diff.md) (the 3b CT-dissolution byte-identity diff this builds on; 3b landed `main` `c62edf6`).

## Why this doc

Phase 2 adds a weight-free config-context viewer for pre-run input QA. It **extracts** the golden
harness's capture loop into a shared core (`src/vista_run/context_capture.py`:
`iter_captured_examples` + `CapturedExample`) that both `golden_harness` (hashes images) and the
new viewer (`src/results/context_viewer.py`, base64-thumbnails) consume. The extraction is
supposed to be a **byte-identity no-op** for the golden `.jsonl`; "trust me" is not credible, so we
instrument. A clean result proves (1) the extraction changed no golden bytes and (2) the viewer
renders exactly what inference feeds, self-contained and weight-free — which unblocks `/land`.

Everything runs on a **small subset** (`--limit ≈5`): this is pre-run input QA, not a full pass.

## Step 0 — get the artifacts onto the VM
```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/phase2-config-context-viewer && git pull --ff-only   # fetch FIRST — local branch is stale
git rev-parse --short HEAD   # must show the committed Phase-2 SHA; if it shows 904896a the impl commit didn't land — STOP
uv sync
```
**Expected:** clean checkout at the committed Phase-2 SHA (i.e. **not** `904896a`, which is the pre-refactor parent); `uv sync` resolves.
**STOP:** `git rev-parse --short HEAD` still shows `904896a` (the impl commit never landed — don't smoke the parent). Otherwise none — pure setup.

**Precondition (config):** these steps read a config with the VM's real mounts (`base_dir`,
`results_dir`, `runtime.cache_dir`, `paths.ct_snapshot_prefix: feb26`). Use the **same localized,
git-ignored config the golden harness already runs with on `phil-sllm-01`** (referred to below as
`<VM_CONFIG>`; e.g. `configs/all_tasks.vm.yaml`). The viewer additionally binds `--experiment` to
that config's `experiments:` list (via `normalize_experiments`), so for the viewer smoke the config
must **list the experiments being viewed** and PFS in `tasks:`. Make a viewer copy:
```bash
cp <VM_CONFIG> configs/all_tasks.viewer.vm.yaml   # git-ignored; edit tasks:/experiments: below
```
so its `tasks:` includes `progression_recurrence_free_survival_1_yr` and its `experiments:`
includes at least `no_image` and `axial_all_image` (add `retrieved_timeline` + the edge dict-entry
for Step 3). `<VM_CONFIG_VIEW>` = this copy.
**STOP (precondition):** if no localized golden config exists on the box, or its mounts/cache_dir
are wrong, fix that first (this is the same config posture the 3b golden diffs used).

## Step 1 — golden byte-identity regression (proves the extraction is a no-op) — THE critical gate
Bank a small golden **before** (pre-refactor code `904896a`) and **after** (this SHA) for the same
`(task, experiment, model, N)`, then `diff_golden --mode strict`. The refactor is on the SAME
substrate (v1_5/feb26), so the ONLY difference is the loop extraction → bytes must be identical.

```bash
# --- "after" bank (this branch, post-extraction) ---
cd src
python -m vista_run.golden_harness \
  --config ../<VM_CONFIG> \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments axial_all_image \
  --tag post_ctxcap --limit 5
cd ..

# --- "before" bank (pre-refactor 904896a, in a throwaway worktree so the shared checkout is untouched) ---
git worktree add ../vista_eval_vlm-prerefactor 904896a
cd ../vista_eval_vlm-prerefactor && uv sync && cd src
python -m vista_run.golden_harness \
  --config <ABS_PATH_TO_VM_CONFIG> \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments axial_all_image \
  --tag pre_ctxcap --limit 5
cd ../../vista_eval_vlm

# --- diff (both jsonl land under results_dir/golden/<src>/<task>/<model>/ with the two tags) ---
cd src
python -m vista_run.diff_golden \
  <results_dir>/golden/.../progression_recurrence_free_survival_1_yr_axial_all_image_pre_ctxcap_golden.jsonl \
  <results_dir>/golden/.../progression_recurrence_free_survival_1_yr_axial_all_image_post_ctxcap_golden.jsonl \
  --mode strict
```
**Expected:** `RESULT: ALL GATES PASS`, exit 0. Both banks report the **same row_count** (== N shared
index set); `STRUCTURE_FIELDS` (selected_indices / image_hashes / image_count / path_tile_count /
assembly_mode) **byte-identical**, and under `--mode strict` `dynamic_prompt` + `adapter_prompt_string`
too. The extraction changed **no** golden bytes.
**STOP (hard halt):** any `GATE FAILURE` / non-zero exit / index-set mismatch ⇒ the extraction was
**not** byte-preserving → do not proceed; report the mismatching fields and hand back. This is the
whole safety of "extract shared core." (Reuse-alternative: if a banked C1/C3 axial golden with a
matching `(task, exp, model, N)` still exists on the mount, diff against it instead of re-banking the
"before" — but N must match, so a fresh `--limit 5` before-bank is the reliable path.)
**Clean up the worktree after:** `git worktree remove ../vista_eval_vlm-prerefactor`.

## Step 2 — viewer smoke on a small subset (CT-bearing + no-image)
```bash
cd src
# CT-bearing experiment (expect thumbnails + a real token bar)
python -m results.context_viewer \
  --config ../configs/all_tasks.viewer.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiment axial_all_image --limit 5
# no-image experiment (expect 0 images flagged on every card)
python -m results.context_viewer \
  --config ../configs/all_tasks.viewer.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiment no_image --limit 5
```
Then verify each HTML (written under `<results_dir>/context_view/.../..._context_view.html`) is
**structurally self-contained** — no external refs, no local image paths in `src`:
```bash
# should print NOTHING (no external URLs, no filesystem paths in <img src>):
grep -nE 'https?://|<img src="(?!data:)' <that>.html || echo "SELF-CONTAINED OK"
grep -c 'data:image/png;base64,' <that>.html   # base64 thumbnails present (CT run only)
```
**Expected:**
- both HTML files exist, non-empty, and **self-contained** (grep finds **no** `http(s)://`, no remote
  CSS/JS/font, and **no** local filesystem path in any `<img src>` — base64 data-URIs only);
- exactly **5** example cards each;
- the `axial_all_image` cards show the assembled prompt + the right **slice count** as thumbnails
  (axial_all_image = 30 slices) + a token-budget bar with a **real** token count over denominator
  **120000** (gemma3's `context_window`);
- the `no_image` cards show **0 images** on every card and flag it; no correctness coloring anywhere.
**STOP:** an HTML references any external URL or leaks a local image path in `src` (not
self-contained); a CT-bearing card renders 0 images or a **wrong** slice count (drift — the viewer
isn't reading the inference path); any GPU/weights load occurs (preflight failed to fail-closed).
**Optional (context_window=None):** run one None-table `--type` (e.g. `qwen2vl` / `medvlm`) if its
tokenizer is cached → the bar should read **"context window unknown"** (accepted, not a failure).

## Step 3 — fail-closed preflight edge cases (each rejects cleanly; no weights touched)
Add to `configs/all_tasks.viewer.vm.yaml` an edge dict-entry so the selector-chain summarize case is
reachable (the name-based `*_summarization*` presets get retrieval-rejected first, so they don't
exercise the new selector-aware guard):
```yaml
experiments:
  - no_image
  - axial_all_image
  - retrieved_timeline            # retrieval reject
  - name: axial_all_image_summ    # block-list dict — model-backed summarize in a select chain
    blocks:
      - {id: ehr, modality: text, adapter: ehr, config: {select: [{fn: summarize}]}}
      - {id: ct,  modality: volume3d, adapter: ct, config: {select: {fn: evenly_spaced_k, k: 30}}}
  - name: axial_all_image_ok      # block-list dict, NO model-backed step — must resolve/render
    blocks:
      - {id: ct, modality: volume3d, adapter: ct, config: {select: {fn: evenly_spaced_k, k: 30}}}
```
```bash
cd src
V="--config ../configs/all_tasks.viewer.vm.yaml --type gemma3 --name google/medgemma-1.5-4b-it --task progression_recurrence_free_survival_1_yr --limit 5"
python -m results.context_viewer $V --experiment retrieved_timeline        ; echo "exit=$?"  # (a) retrieval reject
python -m results.context_viewer $V --experiment axial_all_image_summ      ; echo "exit=$?"  # (b) summarize reject (selector-chain)
python -m results.context_viewer $V --experiment does_not_exist            ; echo "exit=$?"  # (c) unknown --experiment reject
python -m results.context_viewer $V --experiment axial_all_image_ok        ; echo "exit=$?"  # (d) dict entry resolves/renders
python -m results.context_viewer --config ../configs/all_tasks.viewer.vm.yaml --type bogus --name x \
  --task progression_recurrence_free_survival_1_yr --experiment no_image   ; echo "exit=$?"  # (e) invalid --type reject
```
**Expected:**
- (a) retrieval reject, (b) summarize reject, (c) unknown-experiment reject, (e) invalid-type reject
  each **exit non-zero** with a clear `STOP:` message and touch **no** weights/GPU. (e) rejects
  **before** any BQ/GCS client is built.
- (d) the non-model-backed dict entry **renders** (exit 0, HTML written) or rejects cleanly — never
  crashes mid-render.
- **Tokenizer-miss STOP (Q1):** if a chosen `--type/--name` tokenizer won't load (wrong `cache_dir` /
  missing `HF_TOKEN` for a gated repo / uncached / offline), the run **STOPs** at preflight with a
  diagnostic naming the model + likely cause — it never reaches rendering with a fake bar. This is a
  **halt-and-report** (Phil picks the remedy), not a code bug: report it if it fires.
**STOP:** any reject case does **not** fire (renders instead of rejecting, or loads weights), or the
dict entry (d) crashes instead of rendering/rejecting cleanly ⇒ preflight is broken → halt.

## Also verify
```bash
grep -nE '\*_context_view\.html|context_view/' ../.gitignore   # viewer PHI globs present
```
**Expected:** `.gitignore` carries `*_context_view.html` and `context_view/` (parallel to
`*_golden.jsonl` / `golden/`) so the HTML can never be committed.

## Report back
Append to `## VM run results`: per-step ✅/❌ vs the Expected block; for Step 1 the `RESULT:` line +
row_count; for Step 2 the self-containment grep result, card counts, slice counts, and the token
numbers; for Step 3 each case's exit code. **PHI:** the HTML embeds real timelines + imagery — report
**counts / file existence / slice counts / token numbers only**; never paste prompt text, timelines,
or image data. Read large content from the file, don't inline it.

## VM run results
_(left empty by the planner; the executor fills this in readback mode)_
