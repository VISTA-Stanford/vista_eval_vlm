# Next steps

_Last updated: 2026-07-06_

- **Modular VLM preprocessing + context-viewer roadmap** — plan drafted + **5 review passes applied**
  (Codex ×2, Phil pivot + 12-OQ pass, LUMIA/meds_tools research, fresh-Claude — each Revise → applied)
  + explain-plan HTML (in-sync `5ec87cdfecdd`). **UNCOMMITTED**, Reviewed: No. Selection is leaf-only
  (study/series & specimen/block/slide are upstream in VISTABench); EHR serialize = **offline re-home**
  (writes `patient_string`); meds_tools **depend / pin-by-SHA `e2a2a59`** (prep env only, no `v0.1.0`
  tag, relax the spurious `py>=3.14` floor); inline → Phase 1.5.
  next: `/read-plan` → resolve 3 small non-blocking decisions (PFS 1yr-vs-2yr smoke · `summarize`
  on/off + failure mode · viewer side-by-side) + the VM meds_tools install check → implement Phase 0.5
  (overshoot fix) then Phase 1.
  → [plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md](plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md)
- **VLM eval GCP / v1_5 stand-up (Phase 0, subsumed)** — the roadmap subsumes this as Phase 0; its
  v1_5 substrate is inlined. Standalone plan on branch `docs/vlm-eval-gcp-v1_5-standup-plan`, Reviewed: No.
  next: fold into the roadmap's Phase 0 execution; resolve the standup's product OQs there.
- **Frontier models (deferred)** — port `upstream/main` `src/models/api_models.py` (GPT-5 / Gemini)
  onto BAA endpoints (Vertex / Azure-OpenAI) before any VISTA-data run; public APIs violate the BAA.
  Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
