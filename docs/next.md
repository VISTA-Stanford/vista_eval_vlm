# Next steps

_Last updated: 2026-07-06_

- **VM smoke pending:** [docs/vm-status/2026-07-06-golden-harness.md](vm-status/2026-07-06-golden-harness.md)
  — bank the Task-3b legacy golden baseline + run the LUMIA field-coverage check (likeliest breaker) on
  the GCP VM. Commit+push first (SHA set at commit time); before/after diff is a later handoff.
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
