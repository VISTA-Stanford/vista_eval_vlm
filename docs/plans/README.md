# Plans

Index of design / implementation plans for `vista_eval_vlm`. Substance lives in each plan doc; this
table is the pointer. **Reviewed** = Phil has read the *current* content via `/read-plan` (or an
approved, SHA-in-sync `/explain-plan` HTML). `Stale` = was `Yes`, substantively edited since.

| Plan | Status | Reviewed | Description |
|---|---|---|---|
| [vlm-modular-preprocessing-and-context-viewer-roadmap.md](vlm-modular-preprocessing-and-context-viewer-roadmap.md) | Draft | No | Dissolve the `experiment` god-enum into **ContextBlock modality-adapters** (leaf selectors: CT slice / pathology patch / EHR filter chain; `ordered` assembler + `supports_inline` seam, inline → Phase 1.5) + a weight-free **config-context viewer**. 5 review passes applied. Companion `.html` + 3 review files under `reviews/`. |

**Subsumes:** the *GCP v1_5 multimodal VLM eval stand-up* plan (this roadmap's **Phase 0**) lives on
branch `docs/vlm-eval-gcp-v1_5-standup-plan` (`docs/plans/gcp-vlm-eval-v1_5-multimodal-standup.md`),
Reviewed: No — its v1_5 substrate is **inlined** into the roadmap's Phase 0.
