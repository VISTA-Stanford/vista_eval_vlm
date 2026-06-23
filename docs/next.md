# Next steps

_Last updated: 2026-06-22_

- **VLM eval GCP / v1_5 stand-up** — plan drafted + Codex-reviewed (Revise → applied) + explain-plan HTML; **UNCOMMITTED** on `main`, Reviewed: No.
  next: resolve the 4 product OQs (bucket layout · CT `feb26`-vs-`nov25` · tile 100-vs-125 · diagnostic-suite scope) → `/read-plan` → `/phi-vet` + `/commit-review`.
  → [plans/gcp-vlm-eval-v1_5-multimodal-standup.md](plans/gcp-vlm-eval-v1_5-multimodal-standup.md)
- **Frontier models (deferred)** — port `upstream/main` `src/models/api_models.py` (GPT-5 / Gemini) onto BAA endpoints (Vertex / Azure-OpenAI) before any VISTA-data run; public APIs violate the BAA. Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
