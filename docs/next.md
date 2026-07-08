# Next steps

_Last updated: 2026-07-08_

- **VM smoke pending → [docs/vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md](vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md)** — rung-0 reproduce Ryan's **weighted** pipeline on feb26 (v1_1 substrate, only the CT prefix rerouted nov25→feb26 via the new `ct_snapshot_prefix` seam + force-GCS). Gates the rebaseline (rungs 1–2). Plan [plans/vlm-rung0-reproduce-ryan-feb26.md](plans/vlm-rung0-reproduce-ryan-feb26.md) Reviewed:Yes, all OQ-R1–R6 resolved; seam authored on the Mac (UNCOMMITTED — `vqa_dataset.py`/`run_bq.py`/`golden_harness.py`), commit+push before pulling on the VM. next: `/commit-review` (Mac) → VM runs Step 0 (0a preflight gate → 0b weighted → 0c report).
- **SUPERSEDING PLAN drafted 2026-07-07 → [plans/vlm-ct-feb26-v1_5-golden-rebaseline.md](plans/vlm-ct-feb26-v1_5-golden-rebaseline.md)** (+ explain-plan HTML `5ce6c2ce738a`, UNCOMMITTED, Reviewed:No). Resolves the 3 handed-back Mac decisions: **D1** rebuild the CT golden on feb26 (make the legacy loader feb26-capable first → bank → 3b → diff), **D2** cut substrate v1_1→v1_5 (re-bank the EHR baseline; v1_1→v1_5 is an expected results change, NOT byte-gated), **D3** keep `numeric_value` a declared delta, decide accept-vs-parse after the first gate-3 diff (**repo has NO text→number parser**). Crux: refactor axis (byte-gated, both sides feb26) kept separate from substrate axis. Residual OQ-M (dataset-constant site) / OQ-N (upstream numeric fix scope) / OQ-P (v1_5 cohort timeline materialization — VM-verify). next: Phil reviews the HTML (**feedback pending**) → `/review-plan` (Codex) → `/commit-review` → VM runs the 5-step Verification section.
- **VM smoke UNBLOCKED 2026-07-07 — Step 1 + no_image baseline GREEN:** [docs/vm-status/2026-07-06-golden-harness.md](vm-status/2026-07-06-golden-harness.md)
  — the substrate the prior handoff marked BLOCKED lives in the GCS bucket `gs://vista_bench/` (not the
  `/data/fries` layout). **base_dir = bucket root** (staged to `/mnt/su-vista-uscentral1/vistabench/vlm/`;
  cohort via BQ `vista_bench_v1_1`, timelines from `<source_csv>/<task>.csv` — `subsample:false`, no
  `_subsampled` variants). VM config = `configs/all_tasks.vm.yaml` (do NOT commit as default).
  **Step 1 (LUMIA field-coverage) DONE:** `time/code/description/text_value` present; `numeric_value`
  **not a discrete field** (baked into element text → declared delta OQ-K); `unit` present as
  `unit_source_value` → **one-line adapter remap applied** (`ehr.py`, 0%→42.3%). LUMIA-as-input premise
  holds (class-2, not a re-plan). **Steps 2–3 (no_image) PASS:** weight-free, 1,238/1,238 timelines
  matched, full legacy baseline banked (row_count/sort/non-null all verified), golden stays on the PHI
  mount (repo git clean). **`axial_all_image` = class-3 DEVIATION → HANDBACK to Mac planner:** the legacy
  CT loader is pinned to a hard-coded `…/vista/nov25` prefix (`DEFAULT_NIFTI_BUCKET_PREFIX` in
  `vqa_dataset.py`), but **`nov25` is deleted from the bucket — feb26 is the current snapshot.** So the
  axial byte-identity baseline is unbankable on this VM (all rows would be `image_count=0`). Go-forward
  (validated): `vista_bench_v1_5` links CTs via `image_study_uid`+`image_series_uid` →
  `…/vista/feb26/{study}__{series}.nii.gz` (feb26 blob confirmed to exist) — the CT adapter should resolve
  from the materialized dataset link, drop the hard-coded prefix, and move the substrate v1_1→v1_5. **Mac
  re-plan owns:** (1) anchor compat on the no_image/EHR arm vs hunt a nov25 archive, (2) v1_5/feb26
  dataset-linked CT resolution, (3) numeric_value declared-delta OQ-K — **now folded into the superseding plan (top bullet).**
  See the *DEVIATION → Mac planner* block in the vm-status doc. UNCOMMITTED on VM: `ehr.py` unit fix +
  `all_tasks.vm.yaml` + this doc + vm-status doc.
- **Modular VLM preprocessing + context-viewer roadmap** — plan + **6 review passes** + explain-plan HTML
  (`8567ac8decbc`). **Foundation + Task 3a COMMITTED+pushed** (`cba3de6`): Phase 0.5 overshoot fix, additive
  `src/context/` framework, reader normalization, `supports_inline` seam, 3a wiring (lazy model-load split,
  pathology `materialize`, CT windowing shim). **Golden harness authored + Codex-reviewed 2026-07-06**
  (`golden_harness.py` + `diff_golden.py` + `selected_indices` instrumentation) — UNCOMMITTED, VM baseline
  pending (see top pointer). Selection is leaf-only; **EHR consumes full LUMIA timelines as input — filter +
  flat-render + truncate LIVE at inference** (no offline prep, `meds_tools`/`meds_reader`/ontology dropped);
  inline → Phase 1.5.
  next: land golden harness (`/commit-review`) → VM banks baseline + LUMIA check → **implement Task 3b**
  (`__getitem__`/`_build_prompts` dissolution + LUMIA-live config-gate) developed against the golden, then
  Phase 2 viewer. Still-open small decisions: `summarize` on/off + failure mode · viewer side-by-side.
  → [plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md](plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md)
- **VLM eval GCP / v1_5 stand-up (Phase 0, subsumed)** — the roadmap subsumes this as Phase 0; its
  v1_5 substrate is inlined. Standalone plan on branch `docs/vlm-eval-gcp-v1_5-standup-plan`, Reviewed: No.
  next: fold into the roadmap's Phase 0 execution; resolve the standup's product OQs there.
- **Frontier models (deferred)** — port `upstream/main` `src/models/api_models.py` (GPT-5 / Gemini)
  onto BAA endpoints (Vertex / Azure-OpenAI) before any VISTA-data run; public APIs violate the BAA.
  Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
