# Next steps

_Last updated: 2026-07-06_

- **Modular VLM preprocessing + context-viewer roadmap** — plan drafted + **6 review passes applied**
  (Codex ×2, Phil pivot + 12-OQ pass, LUMIA/meds_tools research, fresh-Claude, **serialization
  interrogation 2026-07-06**) + explain-plan HTML (in-sync `8567ac8decbc`). **UNCOMMITTED**, Reviewed: No.
  Selection is leaf-only (study/series & specimen/block/slide upstream in VISTABench); **EHR consumes
  full LUMIA timelines as input — filter + flat-render + truncate LIVE at inference** (no offline prep,
  no CSV re-materialization, `meds_tools`/`meds_reader`/ontology dropped); inline → Phase 1.5.
  next: `/read-plan` → resolve 3 small non-blocking decisions (PFS 1yr-vs-2yr smoke · `summarize`
  on/off + failure mode · viewer side-by-side) + the **VM LUMIA field-coverage check** (corpus carries
  `numeric_value`/`unit`/`text_value`/`description` + covers eval cohorts) → implement Phase 0.5
  (overshoot fix) then Phase 1.
  → [plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md](plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md)
- **VLM eval GCP / v1_5 stand-up (Phase 0, subsumed)** — the roadmap subsumes this as Phase 0; its
  v1_5 substrate is inlined. Standalone plan on branch `docs/vlm-eval-gcp-v1_5-standup-plan`, Reviewed: No.
  next: fold into the roadmap's Phase 0 execution; resolve the standup's product OQs there.
- **Frontier models (deferred)** — port `upstream/main` `src/models/api_models.py` (GPT-5 / Gemini)
  onto BAA endpoints (Vertex / Azure-OpenAI) before any VISTA-data run; public APIs violate the BAA.
  Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
