# Next steps

_Last updated: 2026-07-15_

## ✅ Landed to `main` — modular VLM preprocessing roadmap (2026-07-15)

The full modular-preprocessing roadmap landed on `main` this date, all VM gates green:
Phase 0 (v1_5/feb26 dataset-link substrate) → rung-0 (reproduce-Ryan + the 0b constrained-decoding
fix) → rungs 1–2 (v1_5/feb26 golden rebaseline) → **3b (CT slice-select + windowing dissolved into
`CTAdapter`)**. 3b was proven a byte-identity **no-op** on the imaging surface — gemma C4 full-cohort
(1,238 shared) + intern C3/C4 spot-check (100 rows / 77 CT-bearing), both `ALL GATES PASS`
(→ [docs/vm-status/2026-07-15-3b-intern-limited-golden-diff.md](vm-status/2026-07-15-3b-intern-limited-golden-diff.md)).

Provenance (all on `main`): [roadmap plan](plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md) ·
[rungs 1–2 rebaseline plan](plans/vlm-ct-feb26-v1_5-golden-rebaseline.md) ·
[rung-0 0b decoding-fix plan](plans/vlm-rung0-0b-decoding-fix.md) ·
[rung-0 reproduce-Ryan plan](plans/vlm-rung0-reproduce-ryan-feb26.md) · the per-step `docs/vm-status/` handoff docs.

## Live follow-ups (roadmap not-yet-built)

- **Phase 2 — config-context viewer** — the side-by-side context inspector over the ContextBlock
  assembly is not yet built. Small open decision: viewer side-by-side layout.
- **Phase 1.5 — inline image assembly (deferred)** — the `supports_inline` seam is wired; the inline
  assembly path is deferred behind Phase 2.
- **Step 5 — gate-3 EHR golden (deferred, LUMIA-live)** — strict passthrough is the current gate; the
  full gate-3 EHR golden lands with the LUMIA-live filter/render chain. Still-open small decision:
  `summarize` on/off + its failure mode.
- **Subsumed standup branch** — `docs/vlm-eval-gcp-v1_5-standup-plan` became the roadmap's Phase 0 and
  is inlined; retire the branch (doc-only, superseded).

## Backlog

- **Model-roster refresh (SOTA survey 2026-07)** — VLM roster frozen since ~mid-Feb 2026; we're already current-gen but running the *small* variants. Several config-only upgrades on existing adapters; all open-weight/local (no BAA exposure). Priority order:
  1. **MedGemma 1.5 27B** (`google/medgemma-1.5-27b-it`, confirm exact HF id) — same `gemma3` adapter, config-only; materially stronger than the enabled 4B on 3D CT + WSI pathology (paper: +47% macro-F1 pathology, +11%/+3% 3D MRI/CT vs MedGemma 1).
  2. **Lingshu** — enable the already-registered `lingshu-medical-mllm/Lingshu-7B`; add **Lingshu-32B** (reportedly beats GPT-4.1 / Claude Sonnet 4 on medical multimodal QA + report-gen; 12+ modalities incl. CT / histopath / PET). ⚠ `src/models/lingshu.py` is HF-transformers, not vLLM → **no constrained-decoding / logprob path** until ported to the vLLM template (copy `octomed.py` / `gemma3.py`).
  3. **Larger general variants (GPU-gated, config-only)** — bump `Qwen3-VL-8B` → `Qwen3-VL-30B-A3B` (MoE) or `-32B`, and `InternVL3_5-8B` → `-38B` or `-30B-A3B`, for a stronger general ceiling to compare medical models against. (We already run Qwen3 / InternVL3.5, just the 8B entry size.)
  4. **Retire / demote to frozen baselines** — `llava-1.5-7b` + `llava-med-v1.5` (2023-era; LLaVA-Med needs a separate env) and `Qwen2-VL` / `Qwen2.5-VL` / `MedVLM-R1` (~2B) — superseded; keep only as historical baselines.
  NOT for this repo: Merlin / TITAN+CONCH / Prov-GigaPath / CT-CLIP are CLIP/encoder foundation models → the **`vista-eval` embedding / probe-KNN track**, not this generative-VQA (`infer → text` + constrained decoding) repo.
- **Frontier models (deferred, BAA-gated)** — running flagship Gemini / GPT / Claude on VISTA data needs a **BAA-covered endpoint (Vertex AI with GCP BAA, or Azure-OpenAI)**. `upstream/main` `src/models/api_models.py` is NOT a drop-in port — its `GeminiVisionAdapter` hits the **consumer `generativelanguage` (AI-Studio) endpoint**, which is *not* BAA-covered, so this is effectively a **new Vertex adapter** against `BaseVLMAdapter`, not a port (Claude-on-Vertex is available there too). Public/consumer APIs violate the BAA. Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
