# Next steps

_Last updated: 2026-07-08_

- **VM smoke pending → [docs/vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md](vm-status/2026-07-08-rung0-reproduce-ryan-feb26.md)** — rung-0 reproduce Ryan's **weighted** pipeline on feb26 (v1_1 substrate, only the CT prefix rerouted nov25→feb26 via the new `ct_snapshot_prefix` seam + force-GCS). Gates the rebaseline (rungs 1–2). Plan [plans/vlm-rung0-reproduce-ryan-feb26.md](plans/vlm-rung0-reproduce-ryan-feb26.md) Reviewed:Yes, all OQ-R1–R6 resolved; seam + docs **COMMITTED+PUSHED** (`c379822..6680406`). next: **VM** pulls + runs Step 0 (0a preflight gate → 0b weighted → 0c report), readback into the vm-status doc.
- **Rebaseline rungs 1–2 (gated by rung-0 above) → [plans/vlm-ct-feb26-v1_5-golden-rebaseline.md](plans/vlm-ct-feb26-v1_5-golden-rebaseline.md)** — committed this session (Reviewed:No; HTML stale after the rung-0 trim). v1_5/feb26 dataset-link CT resolution (drop `DEFAULT_NIFTI_BUCKET_PREFIX`) + substrate cut v1_1→v1_5 + 3b CT-dissolution byte-identity golden on feb26 (refactor axis byte-gated, substrate axis not). D3 `numeric_value` a declared delta (repo has NO text→number parser). Residual OQ-M/N/P. next: rung-0 green → `/review-plan` → implement rungs 1–2.
- **Prior handoff (context) → [docs/vm-status/2026-07-06-golden-harness.md](vm-status/2026-07-06-golden-harness.md)** — 2026-07-07 VM smoke: `no_image`/EHR legacy baseline banked GREEN (1,238/1,238; `unit_source_value` remap + `numeric_value` declared-delta OQ-K); `axial_all_image` = class-3 DEVIATION (hard-coded `nov25` prefix, but nov25 deleted → feb26). That deviation spawned the rung-0 + rebaseline plans above (both now committed).
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
