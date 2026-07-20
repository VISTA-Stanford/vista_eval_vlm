# Next steps

_Last updated: 2026-07-20_

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

## ✅ Landed to `main` — Phase 2 config-context viewer (2026-07-16)

Weight-free HTML input-QA viewer (assembled prompt + CT-slice / pathology-tile thumbnails +
token-budget bar) over the ContextBlock assembly, for manual pre-run QA on a small subset.
Codex-reviewed (Critical summarize-selector-chain fixed) + VM smoke on `phil-sllm-01` (golden
byte-identity re-bank `diff_golden --mode strict` → ALL GATES PASS, viewer N=5 self-contained
30-slice / 120000-token bar, all 5 fail-closed preflight edges GREEN) →
[docs/vm-status/2026-07-15-phase2-config-context-viewer.md](vm-status/2026-07-15-phase2-config-context-viewer.md).
Landed `main` `166e7f9` (ff-only, branch `feat/phase2-config-context-viewer` pruned) →
[plans/vlm-phase2-config-context-viewer.md](plans/vlm-phase2-config-context-viewer.md).
Side-by-side layout deferred (single-column first cut).

## Live follow-ups (roadmap not-yet-built)

- **Phase 1.5 — inline image assembly (deferred)** — the `supports_inline` seam is wired; the inline
  assembly path is deferred behind Phase 2.
- **Step 5 — LUMIA-live EHR adapter** — on `feat/lumia-live-ehr-adapter`; three rounds of class-3
  render-alignment deviations, each root-caused + re-planned (see [plans/README.md](plans/README.md)
  for the full history). The render-alignment plan's interval-split hypothesis was VM-refuted
  (4.0% attribution, `<50%` STOP — [docs/vm-status/2026-07-20-7ed0248.md](vm-status/2026-07-20-7ed0248.md)).
  Real cause + fix (config-driven 24mo window crop, matching legacy's actual scope) in
  [plans/vlm-step5-lumia-window-scope-replan.md](plans/vlm-step5-lumia-window-scope-replan.md),
  Codex-reviewed + Phil-approved via `/explain-plan` (2026-07-20). Approach #1-4 implemented +
  landed Mac-side (`a58f5f9`). VM Phase-1 re-verify **DONE, band-3 STOP** (`b88947b`,
  [docs/vm-status/2026-07-20-a58f5f9.md](vm-status/2026-07-20-a58f5f9.md)): 24mo window crop is a
  large partial fix (live/legacy ratio 1.93→1.13, excess 7990→2779) but `total_excess_lines=2779
  > ~2500` → a second cause remains; Phase 2 visual-QA **not run** (gated off). Residual is
  overwhelmingly **LOINC lab lines** (2724 of the excess). Planner-side git archaeology exhausted
  the rendering-code explanations (no LOINC-specific code anywhere in the transform stack; live's
  only relevant transforms are dedup-only) and found a new hypothesis: legacy and live may read
  from **two different underlying data extractions** (different source paths, no shared ancestry
  visible from either repo). Round 4 plan:
  [plans/vlm-step5-lumia-loinc-provenance-replan.md](plans/vlm-step5-lumia-loinc-provenance-replan.md)
  — a code-bypassing raw LOINC event-count check across the 20 already-banked persons.
  **VM BLOCKED (class-3) — see doc:** [docs/vm-status/2026-07-20-3206e84.md](vm-status/2026-07-20-3206e84.md).
  Step 2's raw-legacy leg is unrunnable on the VM: `thoracic_cohort_meds_femr_db` is not staged on
  any mount and `meds_reader` isn't a resolvable dep (fails to build under `uv`). Step 1 provenance
  is *suggestive* (only on-VM stamped extraction is aug-2025; legacy is a Feb-2026 frozen snapshot,
  absent) but not the LOINC-domain confirmation. **Re-planned + resolved on the Mac 2026-07-20**
  (plan's new `## Resolution` section): Phil accepted Step 1's suggestive evidence given Step 2 is
  a dead end without material infra investment — declares the LOINC residual (2724/2779 excess
  lines) a **permanent data-provenance divergence** (same treatment as `STANFORD_OBS`) and **pivots
  Step 5's landing gate to Phase 2 human visual-QA as the primary check**. Ryan-D'Cunha escalation
  (OQ1b) explicitly NOT pursued now. **VM smoke BLOCKED (class-3 deviation) @ `9814abd`:**
  [docs/vm-status/2026-07-20-loinc-closing-verification.md](vm-status/2026-07-20-loinc-closing-verification.md)
  — byte-diff gate FAILS even with `LOINC/` excluded (40 field mismatches / 20 rows). Masked
  readback shows the residual is **systemic cross-vocabulary event-set vintage divergence** (~52–55%
  of each arm's events unmatched at event-identity level; 0 formatting/timestamp explanation), **not**
  a render bug and **not** enumerable-excludable. Confirms the pivot: **retire the strict byte-diff as
  Step 5's landing gate for the LUMIA-live arm**; rely on Phase 2 human visual-QA (`context_viewer.py`).
  **NEXT (Mac):** re-enter plan mode to formalize retiring/downgrading the gate, then Phil runs the
  Phase 2 human-QA render → `/land`.
- **Subsumed standup branch** — `docs/vlm-eval-gcp-v1_5-standup-plan` became the roadmap's Phase 0 and
  is inlined; retire the branch (doc-only, superseded).

## Backlog

- **LUMIA/legacy MEDS extraction reconciliation (Step 5 OQ1b, not blocking)** — escalate to Ryan
  D'Cunha whether the live `thoracic_cohort_lumia` LUMIA corpus can be regenerated from / reconciled
  with the same extraction as legacy's frozen `thoracic_cohort_meds_femr_db` snapshot. Only worth
  pursuing if the accepted LOINC divergence (see Step 5 above) turns out to matter beyond what
  Phase 2 human-QA already catches. Not attempted from the Mac (no data access, can't regenerate
  anything); would also need the db itself located/recovered first (not found on any `phil-sllm-01`
  mount — may only exist on Ryan's original machine).
- **Byte-diff-gate methodology for EHR content (Step 5 OQ3, not blocking)** — the round-4 LOINC
  investigation confirmed a real vintage mismatch on the one MEDS extraction inspectable on the VM
  (`vista_aug2025_meds`, 2025-08-18) vs. legacy's described 2026-02-16 frozen snapshot. Worth
  reconsidering whether full byte parity against a single frozen legacy baseline is the right
  landing gate for EHR content going forward, vs. leaning more on Phase 2 human-QA by design rather
  than as a one-off exception. Not resolved; surfaced for a future planning pass, not this branch's
  landing.
- **Model-roster refresh (SOTA survey 2026-07)** — VLM roster frozen since ~mid-Feb 2026; we're already current-gen but running the *small* variants. Several config-only upgrades on existing adapters; all open-weight/local (no BAA exposure). Priority order:
  1. **MedGemma 1.5 27B** (`google/medgemma-1.5-27b-it`, confirm exact HF id) — same `gemma3` adapter, config-only; materially stronger than the enabled 4B on 3D CT + WSI pathology (paper: +47% macro-F1 pathology, +11%/+3% 3D MRI/CT vs MedGemma 1).
  2. **Lingshu** — enable the already-registered `lingshu-medical-mllm/Lingshu-7B`; add **Lingshu-32B** (reportedly beats GPT-4.1 / Claude Sonnet 4 on medical multimodal QA + report-gen; 12+ modalities incl. CT / histopath / PET). ⚠ `src/models/lingshu.py` is HF-transformers, not vLLM → **no constrained-decoding / logprob path** until ported to the vLLM template (copy `octomed.py` / `gemma3.py`).
  3. **Larger general variants (GPU-gated, config-only)** — bump `Qwen3-VL-8B` → `Qwen3-VL-30B-A3B` (MoE) or `-32B`, and `InternVL3_5-8B` → `-38B` or `-30B-A3B`, for a stronger general ceiling to compare medical models against. (We already run Qwen3 / InternVL3.5, just the 8B entry size.)
  4. **Retire / demote to frozen baselines** — `llava-1.5-7b` + `llava-med-v1.5` (2023-era; LLaVA-Med needs a separate env) and `Qwen2-VL` / `Qwen2.5-VL` / `MedVLM-R1` (~2B) — superseded; keep only as historical baselines.
  NOT for this repo: Merlin / TITAN+CONCH / Prov-GigaPath / CT-CLIP are CLIP/encoder foundation models → the **`vista-eval` embedding / probe-KNN track**, not this generative-VQA (`infer → text` + constrained decoding) repo.
- **Frontier models (deferred, BAA-gated)** — running flagship Gemini / GPT / Claude on VISTA data needs a **BAA-covered endpoint (Vertex AI with GCP BAA, or Azure-OpenAI)**. `upstream/main` `src/models/api_models.py` is NOT a drop-in port — its `GeminiVisionAdapter` hits the **consumer `generativelanguage` (AI-Studio) endpoint**, which is *not* BAA-covered, so this is effectively a **new Vertex adapter** against `BaseVLMAdapter`, not a port (Claude-on-Vertex is available there too). Public/consumer APIs violate the BAA. Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
