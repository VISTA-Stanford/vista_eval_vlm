# Next steps

_Last updated: 2026-07-21_

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

## ✅ Landed to `main` — Step 5 LUMIA-live EHR adapter (2026-07-21)

Live LUMIA `.xml` render replaces frozen `patient_string` CSV substitution, wired into the
`no_image`/`no_report`/`axial_all_image` presets. Five rounds of class-3 deviations on the
byte-diff-to-legacy gate (demographics/flowsheet parser bugs, render-alignment field-label +
interval-split issues, a 24mo window-scope mismatch, a LOINC/cross-vocabulary provenance
divergence) — see [plans/README.md](plans/README.md) for the full round-by-round history. Round 5
retired the byte-diff-to-legacy comparison entirely as structurally unpassable (systemic
data-provenance divergence between legacy's frozen snapshot and the live LUMIA corpus, not a
render bug) and made **Phase 2's human visual-QA render the whole landing gate**
([plans/vlm-step5-lumia-gate-retirement-replan.md](plans/vlm-step5-lumia-gate-retirement-replan.md)).
VM smoke PASS on `phil-sllm-01` (all 3 mandatory experiments clean, correct card/image counts, no
STOP/traceback — [docs/vm-status/2026-07-20-828e570.md](vm-status/2026-07-20-828e570.md)); Phil
read all three HTML files himself. Landed `main` `df0723d`, branch `feat/lumia-live-ehr-adapter`
pruned (local + remote).

## Live follow-ups (roadmap not-yet-built)

- **Step 6 — pathology substrate modernization (Phil-approved, in progress)** — pathology still
  reads a frozen, hand-generated `v1_3` CSV (2 dataset versions behind); this bumps the whole repo
  to `vista_bench_v1_6` (re-verifying CT/EHR/text first), then routes pathology through the same
  task-scoped BQ loader CT/EHR/text already use (`diagnostic_tasks`, zero new query code, per
  Phil's explain-plan feedback) → [plans/vlm-step6-pathology-live-substrate.md](plans/vlm-step6-pathology-live-substrate.md) ·
  companion [`.html`](plans/vlm-step6-pathology-live-substrate.html), Reviewed: Yes (in-sync
  `905d034c0a58`). Phase 0 (dataset bump `query_utils.py:257` v1_5→v1_6) committed on
  `feat/vlm-step6-pathology-live-substrate` @ `ac1561a`. **VM smoke pending:**
  [docs/vm-status/2026-07-21-ac1561a.md](vm-status/2026-07-21-ac1561a.md) — BQ schema-diff +
  golden re-bank byte-diff for CT/EHR/text, gated before Phase 1 (pathology loader) begins.
- **Phase 1.5 — inline image assembly (deferred)** — the `supports_inline` seam is wired; the inline
  assembly path is deferred behind Phase 2. Phil flagged wanting this "soon," once Step 6 lands.
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
- **Byte-diff-gate methodology for EHR content (Step 5 OQ3)** — **this-branch instance resolved by
  round 5** ([plans/vlm-step5-lumia-gate-retirement-replan.md](plans/vlm-step5-lumia-gate-retirement-replan.md)):
  the closing-verification readback found the divergence spans ~half of every arm's events across
  all vocabularies (not enumerable-excludable), confirming full byte parity against this legacy
  baseline is unreachable for data-provenance reasons, not a code defect — the byte-diff gate is
  retired as blocking for this branch, Phase 2 human-QA is primary. **Still open as a general
  policy question (not blocking):** should *future* EHR-content branches lean on human-QA by
  design from the start, rather than defaulting to byte-parity-against-a-frozen-baseline and only
  demoting it after it fails? Surfaced for a future planning pass.
- **Model-roster refresh (SOTA survey 2026-07)** — VLM roster frozen since ~mid-Feb 2026; we're already current-gen but running the *small* variants. Several config-only upgrades on existing adapters; all open-weight/local (no BAA exposure). Priority order:
  1. **MedGemma 1.5 27B** (`google/medgemma-1.5-27b-it`, confirm exact HF id) — same `gemma3` adapter, config-only; materially stronger than the enabled 4B on 3D CT + WSI pathology (paper: +47% macro-F1 pathology, +11%/+3% 3D MRI/CT vs MedGemma 1).
  2. **Lingshu** — enable the already-registered `lingshu-medical-mllm/Lingshu-7B`; add **Lingshu-32B** (reportedly beats GPT-4.1 / Claude Sonnet 4 on medical multimodal QA + report-gen; 12+ modalities incl. CT / histopath / PET). ⚠ `src/models/lingshu.py` is HF-transformers, not vLLM → **no constrained-decoding / logprob path** until ported to the vLLM template (copy `octomed.py` / `gemma3.py`).
  3. **Larger general variants (GPU-gated, config-only)** — bump `Qwen3-VL-8B` → `Qwen3-VL-30B-A3B` (MoE) or `-32B`, and `InternVL3_5-8B` → `-38B` or `-30B-A3B`, for a stronger general ceiling to compare medical models against. (We already run Qwen3 / InternVL3.5, just the 8B entry size.)
  4. **Retire / demote to frozen baselines** — `llava-1.5-7b` + `llava-med-v1.5` (2023-era; LLaVA-Med needs a separate env) and `Qwen2-VL` / `Qwen2.5-VL` / `MedVLM-R1` (~2B) — superseded; keep only as historical baselines.
  NOT for this repo: Merlin / TITAN+CONCH / Prov-GigaPath / CT-CLIP are CLIP/encoder foundation models → the **`vista-eval` embedding / probe-KNN track**, not this generative-VQA (`infer → text` + constrained decoding) repo.
- **Frontier models (deferred, BAA-gated)** — running flagship Gemini / GPT / Claude on VISTA data needs a **BAA-covered endpoint (Vertex AI with GCP BAA, or Azure-OpenAI)**. `upstream/main` `src/models/api_models.py` is NOT a drop-in port — its `GeminiVisionAdapter` hits the **consumer `generativelanguage` (AI-Studio) endpoint**, which is *not* BAA-covered, so this is effectively a **new Vertex adapter** against `BaseVLMAdapter`, not a port (Claude-on-Vertex is available there too). Public/consumer APIs violate the BAA. Do NOT merge `upstream/main` (diverged MMBU repo — deletes the VISTA integration).
