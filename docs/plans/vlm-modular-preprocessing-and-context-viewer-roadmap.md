Reference: docs/claude_ops.md

# VLM eval roadmap: modality-adapter ContextBlocks + leaf selectors + config-context viewer

**Status: Draft** (2026-07-06) · *re-baselined around the ContextBlock modality-adapter architecture (Phil's feedback + sketch). Two Codex passes + a fresh-Claude pass applied; **Phil's OQ pass + LUMIA/meds_tools research + fresh-Claude review applied 2026-07-05**; **serialization interrogation applied 2026-07-06** — selection is leaf-only (hierarchy upstream in VISTABench); **EHR consumes full LUMIA timelines as input (given upstream) — the EHR filter chain runs LIVE at inference over the structured LUMIA, rendered to today's flat format; no offline prep, no CSV re-materialization, no `meds_reader`/ontology/`meds_tools` in-repo**; inline **deferred to Phase 1.5**. Phase 1 unblocked.*

> This roadmap **subsumes** the in-flight *GCP v1_5 multimodal VLM eval stand-up* plan
> (branch `docs/vlm-eval-gcp-v1_5-standup-plan`). That plan becomes **Phase 0** here — the runnable
> substrate — and its v1_5 substrate is now **inlined below** (OQ-L); the standup doc is a build
> reference, then retired. Nothing is discarded; it is referenced, not duplicated.

---

## Resolved direction (2026-07-05)

Four scope forks + a 12-item OQ pass + a LUMIA/meds_tools reuse survey + a fresh-Claude design audit
shaped this plan. Anchors:

1. **Core abstraction → ContextBlock modality-adapters.** Each modality runs an *adapter*
   (ingest → contextualize) that emits a uniform `ContextBlock {id, modality, representation,
   payload, metadata}`. Replaces the flat `ImageSpec`/`TextSpec`-into-`PromptDataset` seam as the
   *central* abstraction; typed specs still exist, but *inside* each adapter.
2. **Assembly → ordered now; inline deferred to Phase 1.5.** The assembler emits a **typed ordered
   content sequence**. `ordered` ships in Phase 1; **`inline_by_timestamp` (image injected into the
   EHR text at `metadata.timestamp`) is deferred to Phase 1.5** — no legacy preset or named experiment
   uses it yet, and it forces a `create_template` rewrite across the model roster + capability wiring
   (fresh-Claude review). Phase 1 ships the **`supports_inline` seam** (default off, fail-closed) so
   Phase 1.5 is additive.
3. **Selectors → leaf-level now; hierarchy is upstream.** Study/series (CT) and specimen/block/slide
   (pathology) selection is done **upstream by VISTABench** — verified: vista_bench dedups to one
   imaging row per person (`vista_bench queries.py:531`, `progression_recurrence_survival.py:124`
   `drop_duplicates(subset=["person_id"])`; `queries.py:589` QUALIFY prefers imaging). Selection here is
   per modality at the **leaf**: CT **slice**, pathology **patch**, EHR a filter/transform **chain**.
   ⚠ Re-adding real study/series selection later is a **cross-repo change to vista_bench** (stop
   deduping, hand multiple candidates through), not just a local selector — the seam is upstream, not here.
4. **Sequencing → skip the interim knob phase; framework-first.** Land **only** the `axial_all_image`
   overshoot bug-fix as a standalone patch, then build the framework once. The first v1_5 sweep runs on
   Phase-0 legacy behavior if timing demands — **not** blocked on the framework.

**Fresh-Claude review (2026-07-05) resolved three items the plan had wrongly called "no blocking
preconditions":** (a) the **EHR serialization layer**; (b) **meds_tools install reality**; (c) **inline
scope** → **defer to Phase 1.5** (above).

**Serialization interrogation (2026-07-06 — Phil) supersedes (a) and (b).** The 2026-07-05 answer to (a)
was "re-home the offline prep"; interrogating what *serialization* actually denotes here dissolved that.
"Serialize" conflated two transforms: **`OMOP → LUMIA`** (the DB + ontology-heavy one) and **`LUMIA →
flat patient_string`** (pure `xml.etree` + string). **Decision: assume the full `OMOP → LUMIA` step is done
upstream and the per-patient LUMIA timeline (`gs://vista_bench/thoracic_cohort_lumia/{person_id}.xml`) is
given as input.** The only serialization left in-repo is `LUMIA → flat`, which needs no `meds_reader`, no
ontology, no `meds_tools`. Consequences: **(a′)** the EHR adapter **ingests LUMIA and runs the filter chain
+ flat-render + truncate LIVE at inference** — full parity with the CT/pathology/assembly axes; **no offline
prep, no `patient_string` CSV re-materialization**. **(b′)** the `meds_tools` **SHA-pin / `py>=3.14` floor /
`uv.sources`** apparatus is **moot** — it was all about the `get_described_events_window` *fetch*, which is
no longer called. With these, Phase 1 is unblocked. ⚠ **One VM-verifiable dependency:** byte-identical
reproduction of today's flat string requires the LUMIA corpus to carry the flat renderer's fields
(`numeric_value`/`unit`/note `text_value`, a distinct `description`) and to cover the eval cohorts — a
Phase-1 precondition below; if it doesn't, gate 1 becomes a *declared-delta* (OQ-K), not a blocker.

---

## Goal

Two coupled aims, sequenced as one roadmap:

1. **Dissolve the `experiment` god-enum into composable, config-selected modality adapters.** Today a
   single `experiment` string simultaneously chooses the data loader, the text composition, *and* the
   image selection+preprocessing — branched across three files. The aim: a new
   `(CT-slice-selector × EHR-filter-chain × pathology-patch-selector × assembly)` combination becomes a
   **config edit**, not a three-file code change. **Every axis is a live inference-time edit** — including
   the EHR filter chain, which runs over the **given LUMIA timeline** (parse → filter → flat-render →
   truncate) with **no re-materialization** (serialization interrogation 2026-07-06 — supersedes the earlier
   "EHR is offline/prep" caveat, which was an artifact of re-fetching from `meds_reader` rather than
   consuming LUMIA).

2. **Add a config-context VIEWER (HTML).** Render exactly what a given config feeds each model — the
   **assembled ContextBlocks** (composed prompt text plus the selected CT slices / pathology tiles as
   actually preprocessed, in assembly order) — in a single scrollable HTML you can **page through
   batch-by-batch**. Both a *design tool* and the *loud verification* the stand-up plan demands.

## Why

The `experiment` string is a **god-enum doing three jobs**, branched in three places:

| Job | Where it branches today | Symptom |
|---|---|---|
| Which **data loader** | `run_bq.py:run_inference` + 6 `_load_*` methods | every new combo needs a new `_load_*` |
| How **text/EHR** is composed | `run_bq.py:_build_prompts_for_experiment` (`:732`) + DB-refetch fork `remove_imaging_report.py` | timeline / +report / +path_note / retrieval hardcoded per-name; the filtered variant re-fetches from `meds_reader` + re-serializes offline instead of filtering the LUMIA it already has |
| How **images** are selected + preprocessed | `vqa_dataset.py:__getitem__` (`:112`) | CT slice logic **duplicated across 3 branches (covering 6 experiments)** |

Concrete coupling debt this roadmap pays down:

- **CT slice selection is inlined and inconsistent.** `axial_all_image` samples **30** slices via a
  **buggy** `i*0.1` scheme (`vqa_dataset.py:198-205` — overshoots `1.0`, silently clamps to the last
  slice); `no_timeline`/`all_vb_image_only` sample **50** (`:208-224`); `no_report` samples **10**
  (`:227-242`). Re-implemented across **three** branches. **No `every_n` selector.** Docs
  (`02-ct-scans.md:45`) already contradict the code.
- **CT preprocessing is keyed off `is_gemma`** (`vqa_dataset.py:78,94`), not config.
- **Pathology tile sampling is hardcoded** in `run_bq.py:461` (`num_tiles_per_slide = 100`, `seed = 42`).
- **The EHR string builder is a divergent DB-refetch fork** (`meds_timeline_utils.py:185`, used at
  `remove_imaging_report.py:69`) with **no composable selection** — you cannot say "only radiology
  notes" or "codes within 1yr" without editing code. Worse, its "no imaging report" variant **re-fetches
  the whole timeline from `meds_reader` and re-serializes** rather than filtering the LUMIA it already has.
- **Assembly is implicit and flat** — no way to place a scan inline at its acquisition point (Phase 1.5).
- **The one good seam already exists:** `timeline_truncation: {mode, k}` is config-driven via
  `truncate_timeline`. The leaf selectors + EHR filter chain generalize it.

Principle anchor (`claude_ops.md`): *"You don't trust; you instrument."* The viewer is the instrument
that was missing.

## Reuse findings (verified against sibling repos; EHR stack surveyed by agent + fresh-Claude 2026-07-05)

- **CT slice selector / windowing / resize — NOTHING CANONICAL to import.**
  `vista_bench/src/vistabench/` is SQL/cohort/survival only. The siblings that touch CT pixels
  (`contrastive-3d-onc`, `MerlinOnc`, `vista-ct`) are 3D MONAI/torch encoder pipelines — none expose a
  2D-axial-slice-for-VLM selector. ⇒ **The CT slice selectors are built locally** (pure-numpy index
  math). The `(inputs, ctx) -> selection` signature could later host a learned selector (OQ-C, deferred).
- **EHR timeline — REUSABLE, and now fully at inference (serialization interrogation 2026-07-06 —
  load-bearing).** **"LUMIA" is a meds2text XML *serialization format*** (`OMOP → BQ → MEDS → meds_reader →
  LUMIA`), not a module; the full per-patient timeline is **given as input**: corpus
  `gs://vista_bench/thoracic_cohort_lumia/{person_id}.xml` (schema `meds2text/docs/markup.md`). LUMIA is
  **structured** (`<eventstream>/<encounter>/<events>/<entry timestamp>/<event type code name>`), so the
  whole filter chain maps directly onto it — `window`→`<entry timestamp>`, `note_type_filter`→`<event type>`
  (+ provider `speciality`), `code_filter`/STANFORD-skip→`<event code>`. **Decision: the EHR adapter ingests
  LUMIA and runs filter → flat-render → truncate LIVE at inference** — `meds_reader` DB + ontology never
  enter the repo (they produced LUMIA upstream). Today's DB-refetch fork (`remove_imaging_report.py`,
  `find_bq_timeline_column` at `run_bq.py:257`) is **retired**, not re-homed. Reusable / net-new pieces:
  - **Ingest (net-new, cheap):** read + parse the per-patient LUMIA `.xml` (stdlib `xml.etree`) → an event
    DataFrame. Replaces `get_described_events_window` — the *only* DB+ontology consumer — entirely. No
    `meds_tools`, no `meds_reader`, no ontology.
  - **Filter (net-new seam):** `window` / `note_type_filter` / `code_filter(exclude_stanford)` as DataFrame
    ops over the parsed events. The STANFORD/report skip is just a `code_filter` row-drop (the old
    `exclude_report=True` kwarg disappears).
  - **Flat-render:** reproduce today's line format `[time] | code (description) | VALUE numeric+unit | text`
    **field-for-field** — the local `meds_timeline_utils.get_llm_event_string` (`:185`, imports only
    `re`+`pandas`) is the **spec**; vendor it into the EHR adapter (same discipline as lifting
    `truncate_timeline` verbatim). Chosen over LUMIA-native output to keep golden gate 1 byte-identical.
  - **Truncate:** lift `truncate_timeline` (`meds_timeline_utils.py:62`, `max_chars`/`last_k_events`) —
    at **inference**. ⚠ Latent quirk: `first_4_rows` is init `''` and never populated, so it prepends
    `'' + '\n'`. **Lift verbatim (byte-preserving for golden gate 1) — do NOT "fix" it during the lift.**
  - **Summarize (optional):** lifting `_summarize_timeline_for_context[_batch]`
    (`src/retrieval/iterative_retrieval.py:62-138`, VLM-wired) is **a small cluster, not a one-liner** —
    it drags `_run_vlm_text_only`/`_batch`, `_extract_answer`, `TIMELINE_SUMMARY_TEMPLATE`, and a
    **raw-timeline-on-exception fallback** (`:92,137`). **Decide the fallback (fail loud vs swallow)** —
    per `claude_ops.md`, don't default silently. (Model-backed → not weight-free; the viewer rejects/marks it.)
  - ⚠ **"No third formatter copy" — a third copy ALREADY exists** in
    `src/data_tools/OMOP_meds_query/test_meds_tools.py:6`; consolidate it too.
  - **`meds_tools` (OQ-G) — NO LONGER A DEPENDENCY.** Consuming LUMIA deletes the
    `get_described_events_window` fetch, so the SHA-pin @ `e2a2a59`, the spurious `requires-python >=3.14`
    floor, and the hardcoded local-path `[tool.uv.sources]` (`/home/minwoos/repos/…`) are all **moot** —
    nothing to pin or install.
  - ⚠ **VM precondition (byte-identical hinges on this):** confirm the LUMIA corpus actually carries the
    flat renderer's fields — lab `numeric_value`/`unit`, note-body `text_value`, a `description` distinct
    from `name` — and covers the eval cohorts. `markup.md` documents `<event>` with only
    `type/code/name/note_id/provider_id` (element text = a state token like `start`) and is abbreviated;
    diff one real `.xml` against `get_llm_event_string`'s inputs. If a field is absent, gate 1 drops to a
    **declared-delta** (OQ-K); if coverage is partial, retain a DB-fetch fallback only for the gap.
- **Pathology tiles — vista_eval_vlm owns it.** `tile_wsi.py` (OpenSlide + Otsu extraction) is canonical
  and good. The *sampling* ("100 random, seed 42", `run_bq.py:461`) lifts into the pathology adapter's
  **patch** selector; leave room for `tissue_top_n`. **Specimen/block/slide is upstream** (OQ-A).

## From god-enum to modality adapters

| Old job (god-enum) | New home |
|---|---|
| **Which data loader** (cohort) | `cohort.source` — *separate concern, Phase 3* |
| **Text/EHR composition** | **EHR adapter** — ingest given LUMIA → filter chain → flat-render → `truncate`, **all live at inference** (no prep, no materialized CSV) |
| **Image modality present** | which **adapters** appear in the block list (`ct`, `pathology`) |
| **Image selection** | the adapter's **leaf selector** (CT slice; pathology patch) |
| **Image preprocess** | the adapter's **preprocess** step, `by_model`-resolved off `MODEL_REGISTRY` (OQ-B) |
| *(new)* **Assembly / placement** | the **assembler** — `ordered` now; `inline_by_timestamp` in Phase 1.5 |

### Adjacent workstream: VLM *model* modularity (audit 2026-07-03, still current)

The model adapters are cleanly decoupled from preprocessing (they receive a ready `{image, question}`
item). The coupling runs the *other* way: `PromptDataset` reaches back to model identity via `is_gemma`,
and the orchestrator batches on `'gemma' in model_type`. **Dissolving the `is_gemma` sniff is shared
groundwork** for a future model-registry refactor (abstract `BaseVLMAdapter`, a shared `VLLMChatAdapter`,
unified `infer` return type — **a separate NOT-yet-written plan**). The model-load split out of
`TaskOrchestrator.__init__` (Phase 1) is groundwork both plans want; the Gemma image-count batching leak
is deliberately *preserved* (the viewer relies on it). **`by_model` resolves off `MODEL_REGISTRY`, override
allowed (OQ-B).** The **`supports_inline` capability** (Phase 1 seam) also lands on `BaseVLMAdapter` — see
the inline contract.

## Architecture: adapters, ContextBlocks, leaf selectors, assembler

```
Raw input ──(ingest)──> normalized ──(contextualize: select + preprocess/serialize)──> ContextBlock
                                                                                             │
                     many blocks ──────────────(assemble: ordered  [inline in P1.5])────────>│──> VLM context ──> VLM
```

- **`ContextBlock`** — `{id, modality, representation, payload, metadata}`. `metadata` carries the
  acquisition **timestamp** (for Phase-1.5 inline), the selector params applied, and provenance.
- **`ModalityAdapter`** (ABC) — `ingest(raw) -> normalized`, `contextualize(normalized, cfg) -> ContextBlock`.
  Adapters: `ehr`, `ct`, `pathology`. In `ADAPTER_REGISTRY`. **The EHR adapter is fully inference-time,
  symmetric with CT/pathology: `ingest` = read+parse the given LUMIA `.xml`; `contextualize` = filter chain
  → flat-render → truncate. No prep step, no DB/ontology.**
- **Leaf selector per modality** — `(candidates, ctx) -> selected`, deterministic today.
  - **CT (slice):** `evenly_spaced_k(k)`, `every_n(n)`, `center`, `center_k(k)`, `slice_range(lo,hi)`, `all`.
  - **Pathology (patch):** `random_n(n, seed)`, `all`; future `tissue_top_n`.
  - **EHR (filter chain, applied live over LUMIA):** `window`, `note_type_filter`, `code_filter(exclude_stanford)`,
    optional `summarize`, then flat-render (vendored `get_llm_event_string` spec — not `meds_tools`) → `patient_string`.
  - **Not built here:** study/series (CT), specimen/block/slide (pathology) — upstream in VISTABench (OQ-A).
- **`Assembler`** — `assemble(blocks, strategy) -> typed content sequence`. `ordered` ships in Phase 1;
  `inline_by_timestamp` in Phase 1.5.

### Model-adapter inline contract (Codex + fresh-Claude — load-bearing; seam in Phase 1, wired in Phase 1.5)

Verified: the inference item is a **flat `{question, image}`** (`vqa_dataset.py:250`) and **every** model
adapter's `create_template` appends **all images first, then text** (`gemma3.py:43-77`, `qwen3.py:46-73`).
There are **8+ adapters** in `MODEL_REGISTRY` (`models/__init__.py:6-64`) but only gemma3/qwen3 would get a
rewrite — so mid-prompt interleaving is the *exception*, and **fail-closed is the common case.** The
contract Phase 1 ships (seam) and Phase 1.5 wires:

- The assembler emits a **typed ordered content sequence** (`[{type: text|image, ...}]`) — the ContextBlock
  list *is* the content sequence. (Phase 1.)
- **Capability lives on the model layer, read at preflight (fresh-Claude — spec'd concretely):** add
  `supports_inline: bool = False` as a class attr on `BaseVLMAdapter` (`base.py:33-70`). The assembler
  receives the **resolved adapter (or a capability dict)**, *not* a bare `model_type` — avoiding a
  `src/context` → `src/models` import cycle. When a config requests `inline_by_timestamp` and the resolved
  adapter's `supports_inline` is `False`, **raise a clear, documented error at preflight — before any
  weights load** (so the viewer benefits too). OQ-D resolved: **fail closed, not silent downgrade.**
- Phase-1.5 verification asserts the **adapter-produced** prompt string / placeholder order per
  inline-capable model — not just the viewer's rendered order.

## Phased roadmap

### Phase 0 — v1_5 GCP stand-up *(subsumed prerequisite)*
**First v1_5 sweep runs here on legacy behavior** — not blocked on the framework.

> **v1_5 substrate contract (inlined per OQ-L — standup doc on branch `docs/vlm-eval-gcp-v1_5-standup-plan`
> is a build reference, then retired):**
> - **Infra:** A100-80GB/H100 in BAA project `som-nero-plevriti-deidbdf`, region us-central1 (matches
>   `gs://su-vista-uscentral1` + BQ); ADC BQ + Storage read.
> - **Dataset constant:** `VISTA_BENCH_DATASET = "vista_bench_v1_1"` (`query_utils.py:236`) → `vista_bench_v1_5`.
> - **Path resolvers:** CT NIfTI prefix `nov25` (`vqa_dataset.py:12,15`) → v1_5 primary `feb26`, `nov25`
>   fallback. Task JSONs `tasks/valid_tasks_v1_3.json` → `valid_tasks_v1_5.json` + `prompts_by_task.json` +
>   `image_valid_tasks.json` (3 files; no single generator).
> - **BQ contract:** 3 tables in `vista_bench_v1_5`, cols `task, split, label, person_id, embed_time` +
>   `nifti_path, _accession_number, path_image_path, path_note_text`.
> - **Config/WSI:** `all_tasks.yaml` GCP paths; `valid_tasks: tasks/valid_tasks_v1_5.json`; WSI tiled 896 px
>   / 10× into `path_tile_base/test_patch/`. **Files:** `query_utils.py`, `task_data_utils.py`.
> ⚠ **This branch is a MIXED baseline (fresh-Claude), not uniform v1_3:** BQ dataset = **`vista_bench_v1_1`**
> (`query_utils.py:236`) **+** task JSON = **v1_3** (`all_tasks.yaml:6` = `valid_tasks_v1_3.json`). An executor
> must not assume a uniform v1_3 start.

### Phase 0.5 — Standalone overshoot bug-fix *(small, independent, land anytime)*
Fix the `axial_all_image` `i*0.1` overshoot (`vqa_dataset.py:198-205`); correct `02-ct-scans.md:45`. Lands
**before** the golden baseline, so the slice axis contributes zero to the Phase-1 diff. ⚠ **Not a
whole-refactor no-op:** Phase 1 also moves windowing dispatch (`is_gemma → by_model`); the golden is
**staged** (see Back-compat). The old buggy sampler is **dropped** (OQ-E).

### Phase 1 — Modality-adapter framework *(the core refactor)*
Introduce `src/context/`; make `PromptDataset` + the prompt builder **stop branching on `experiment`**. Build:
- **`ContextBlock`** + **`ADAPTER_REGISTRY`** + **`ModalityAdapter`** ABC.
- **CT adapter** — **slice** selector + windowing (config not `is_gemma`; `by_model` off `MODEL_REGISTRY`).
- **Pathology adapter** — **patch** selector lifted from `run_bq.py:461`; must not duplicate `tile_wsi.py`;
  folder resolution (`:486-498`) moves with it.
- **EHR adapter (LUMIA-in, live)** — `ingest` reads+parses the given LUMIA `.xml` (stdlib `xml.etree`) → event
  DataFrame; `contextualize` runs the filter chain (`window/note_type_filter/code_filter`, optional `summarize`;
  STANFORD skip = a `code_filter` row-drop) → **flat-render** to today's line format (vendored
  `get_llm_event_string` as the field-for-field spec) → `patient_string` → lifts `truncate_timeline`
  (verbatim). All inference-time; retires the `remove_imaging_report.py` DB-refetch fork. **Decide the
  summarize failure mode.**
- **`Assembler`** — `ordered` + the **typed content sequence** + the **`supports_inline` seam** (default
  False, fail-closed preflight). Actual interleaving = Phase 1.5.
- **`normalize_experiments`** + **`presets.py`** — resolve a block-list entry (or bare legacy name) to a
  typed spec; `name`/`display` are the only downstream tokens.
- **Split model-load out of `TaskOrchestrator.__init__`** (`run_bq.py:103`) so the viewer runs weight-free.

### Phase 1.5 — inline_by_timestamp interleaving *(deferred; when an experiment needs it)*
Wire real mid-prompt image interleaving: rewrite `create_template` for the inline-capable models and set
`supports_inline = True` for them. Additive on the Phase-1 seam. No consumer names it yet, so it waits.

### Phase 2 — Config-context viewer (HTML) *(new deliverable — detailed spec below)*
Runs the **same** ingest + select + preprocess + assemble path (no model weights) → one self-contained HTML:
assembled ContextBlocks (prompt text + selected images, in assembly order), paginated by batch, with a
**token-budget bar** (OQ-J). Zero drift with inference.

### Phase 3 — *(optional, deferrable)* cohort-source axis
Fold the 6 `_load_*` into a config `cohort.source`. Larger blast radius (resume/output-path). **Gated on OQ-F.**

## New package layout

```
src/context/
  __init__.py
  block.py           # ContextBlock {id, modality, representation, payload, metadata(+timestamp)}
  adapters/
    base.py          # ModalityAdapter ABC
    ct.py            # slice selector + windowing -> ContextBlock
    pathology.py     # patch selector -> ContextBlock
    ehr.py           # ingest+parse LUMIA .xml -> filter chain -> flat-render (vendored get_llm_event_string) -> truncate (all inference)
  selectors/
    ct_selectors.py  # evenly_spaced_k, every_n, center, center_k, slice_range, all
    path_selectors.py# random_n(seed), all  [+ future tissue_top_n]
    ehr_filters.py   # window, note_type_filter, code_filter  [+ summarize cluster lift]
  windowing.py       # multi_window_rgb (=window), grayscale (=normalize_slice)
  assembler.py       # assemble -> typed content sequence: ordered [inline in P1.5]; fail-closed preflight
  specs.py           # resolve config entry / legacy preset -> adapter instances
  presets.py         # legacy experiment-name -> block composition
  normalize.py       # normalize_experiments(cfg) -> [(name, display, spec)]
src/results/
  context_viewer.py  # Phase 2 generator (CLI) -> portable HTML
```

`supports_inline` lands on `src/models/base.py:BaseVLMAdapter`. No new third-party deps for the viewer.

## Config schema (before → after)

**After** (a named composition of adapter blocks + assembly; legacy names still accepted):
```yaml
experiments:
  - name: ct_every10_timeline
    cohort: subsampled
    assembly: ordered                             # inline_by_timestamp -> Phase 1.5
    blocks:
      - id: ehr
        modality: text
        adapter: ehr
        config:                                   # NOTE: select+serialize apply LIVE at inference over the given LUMIA
          select:
            - { fn: window, before: 365d, after: 0d }
            - { fn: note_type_filter, keep: [radiology, pathology] }
            - { fn: code_filter, exclude_stanford: true }
            # - { fn: summarize, model: <id> }     # optional cluster lift; failure mode must be decided
          serialize:  { style: flat_timeline }    # LUMIA -> today's flat line format (byte-for-byte spec = get_llm_event_string)
          truncation: { mode: last_k_events, k: 100 }   # applied at INFERENCE
      - id: ct
        modality: volume3d
        adapter: ct
        config:
          select: { fn: every_n, n: 10 }          # or evenly_spaced_k k=50, center_k k=10
          preprocess: { windowing: by_model, target_size: by_model }

  - name: path_full                                # back-compat preset
    cohort: path
    assembly: ordered
    blocks:
      - { id: ehr,  modality: text,      adapter: ehr,       config: { select: [], serialize: {style: flat_timeline, include_path_note: true}, truncation: {mode: last_k_events, k: 100} } }
      - { id: path, modality: patches2d, adapter: pathology, config: { select: { fn: random_n, n: 100, seed: 42 } } }

  - axial_all_image                                # bare string still works -> presets.py expands it
```

✅ Changing an EHR `select`/`serialize` value is a **live inference-time edit** (re-filters the given LUMIA in
memory) — **no re-materialization** — at full parity with CT slice / pathology patch / `truncation` / `assembly`.

### Normalized experiment contract (Codex + fresh-Claude — load-bearing)

Three result-discovery scripts read `experiments` straight from the YAML and break on non-string entries:

- `src/results/final_metrics.py:47` does `experiments = set(config.get("experiments", []))` →
  **`TypeError: unhashable type: 'dict'`** on any dict.
- `src/results/all_model_response.py:37,65,75` (`EXPERIMENT_DISPLAY_NAMES`, `load_config`, `collect_result_files`).
- `src/results/plot_code/ct_experiment_plot.py`: the `set()` is at **`:186`** (`:184` is `config.get`);
  `parse_experiment_comments` (`:53-66`) reads display names by **text-scanning the raw YAML file**
  (called at `:190` with `config_path`, not the parsed dict). ⇒ It must be **deleted and its call site
  rewired to `normalize_experiments`**, not adapted. (`check_image_usage.py` discovers by filename + a
  literal `no_image` check.)

**Contract:** `normalize_experiments(cfg) -> list[(name, display, spec)]` resolves a bare legacy string
**and** a block-list dict to `{name, display, cohort, assembly, blocks}`. `name` = filename/resume/metrics
token; `display` = plot label. Every reader consumes normalized names/labels, never raw dicts.

## Back-compat (YAGNI-respecting)

`presets.py` maps each legacy name to its exact block composition. Preserved: output path
`{task}_results_{name}.csv` (`run_bq.py:718`); resume-by-index (`:722-731`); the **opposite** retrieval
contract (`_setup_output_and_resume` deletes + no-resume, `:719`); the 6 loaders re-routed not removed;
**defaults = prior runtime (OQ-H): 100 tiles/slide, per-preset slice counts (30/50/10).**

A **staged golden-output test** (Phase 1 verification) — a single "pure no-op" claim isn't credible because
Phase 1 also moves windowing dispatch and switches the EHR source to LUMIA (+ vendors the flat renderer).
Per legacy preset, **sorted by
`(person_id, index)`** before diffing (fresh-Claude: `run_bq` writes as-processed + appends on resume — the
harness must impose ordering; `index` is a unique per-file join key):

1. **Legacy-equivalence (byte-identical) — imaging + structure.** Legacy windowing dispatch: full-string
   `dynamic_prompt` + selected slice/tile indices + image **hashes** identical. The true no-op gate for the
   CT/pathology/assembly/truncation surface. (Lifting `truncate_timeline` verbatim — incl. the
   `first_4_rows=''` quirk — keeps truncation byte-identical; the EHR *string* equivalence is gate 3.)
2. **`by_model` preprocessing delta.** Flip windowing `is_gemma → by_model`; image bytes identical *per model*.
3. **EHR LUMIA-render delta.** The LUMIA-ingest + live filter + flat-render vs the current `patient_string`
   for the same patient/window, within a **declared allowlist** — where **"ordering" means within-line field
   order ONLY; event/line order must be identical** (both render chronologically, so it holds; inline
   placement in P1.5 depends on it). ⚠ **Runs entirely on the live inference path — there is no CSV to
   regenerate.** Assert the LUMIA-derived `dynamic_prompt` equals the current one within the allowlist, and
   that a *filtered* config (e.g. `note_type_filter`) drops exactly the right events live. **Conditioned on
   the LUMIA field-coverage check** (Reuse ⚠): a missing field moves this to a declared-delta (OQ-K), not a failure.

> **OQ-K:** byte-equality is the target — gate 1 for imaging/structure, gate 3 for the EHR string — verified
> on one example first; a **declared-delta fallback** yielding equivalent results is acceptable (for the EHR
> string this is the LUMIA-field-coverage escape hatch); cross-version repro (pre-v1.5) is not a goal.

## Config-context viewer — detailed spec (Phase 2)

- **Generator:** `src/results/context_viewer.py`, CLI `--config --task --experiment --model [--limit]
  [--batches] --out`. Reuses the **same** ingest + select + preprocess + assemble path (imported) → zero
  drift. ⚠ `TaskOrchestrator.__init__` always loads the model (`run_bq.py:103`) → Phase 1 splits data/prompt
  setup from model loading. `model_type` resolves only `by_model` windowing/target — **no GPU, no weights.**
  ⚠ Model-backed `summarize` is **not weight-free** → viewer rejects-or-marks it at preflight.
- **Per example:** the **assembled ordered** context; composed `dynamic_prompt` (monospace, collapsible);
  selected images (CT slices post-window / pathology tiles) as base64 thumbnails; metadata chips (index,
  person_id, task, label, image count, a **token-budget bar** vs the model context window (OQ-J), selector
  params, assembly mode). ⚠ `used_image` is only **0/1** (`run_bq.py:831`) — read selected indices/counts
  off the assembled block.
- **Pan through batches** by `runtime.batch_size` (incl. Gemma image-count grouping). *(Side-by-side — OQ-J
  — optional, later.)* Self-contained one HTML file. Doubles as verification (0-image rows, empty prompts).

> **⚠ PHI (hard constraint).** The rendered HTML embeds real de-identified timelines + imagery → **VM-only,
> never committed.** Writes only to a **PHI-compliant su-vista mount** (OQ-J — e.g. `gs://su-vista-uscentral1/…`
> / mounted results dir); glob in `.gitignore`; `/phi-vet` gates every commit.

## Files to modify / add

- **New:** `src/context/**`. `path_selectors` operates on already-materialized tile-path lists — must **not**
  duplicate `tile_wsi.py`; folder resolution (`run_bq.py:486-498`) moves with the sampler.
- **New:** `src/results/context_viewer.py`. `.gitignore` (`configs/*`/`logs/*`/`figures/*` at `:46,60,63`) +
  viewer glob; write only to the su-vista PHI mount.
- `src/models/base.py` — add `supports_inline: bool = False` to `BaseVLMAdapter`; inline-capable models flip
  it in Phase 1.5.
- `src/vqa_dataset.py` — `__getitem__` consumes resolved adapters; remove the 3 inline CT branches;
  `window`/`normalize_slice` → `context/windowing` (shim).
- `src/vista_run/run_bq.py` — `_build_prompts_for_experiment` → EHR/assembler path; lift the path-tile
  sampler; consume `normalize_experiments`; **split model-load out of `TaskOrchestrator.__init__`**; the
  assembler receives the resolved adapter (for `supports_inline`), not a bare `model_type`.
- **Downstream `experiments` consumers (update together):** `final_metrics.py:47`, `all_model_response.py`
  (`:37,65,75`), `check_image_usage.py`, `ct_experiment_plot.py` (`:186` `set()` **and** delete
  `parse_experiment_comments` `:53-66` + rewire the `:190` call site).
- **EHR (LUMIA-in, live):** retire the DB-refetch fork `src/data_tools/OMOP_meds_query/remove_imaging_report.py`
  (its `get_described_events_window` fetch → LUMIA ingest+parse in the EHR adapter);
  `src/data_tools/utils/meds_timeline_utils.py` (vendor `get_llm_event_string` as the flat-render spec + lift
  `truncate_timeline` verbatim); `src/retrieval/iterative_retrieval.py` (lift the `_summarize…` cluster;
  decide the fallback); consolidate the third formatter copy in
  `src/data_tools/OMOP_meds_query/test_meds_tools.py:6`. STANFORD/report skip → a `code_filter` row-drop.
  **No `meds_tools`/`meds_reader`/ontology dependency.**
- `configs/all_tasks.yaml` — new schema. **Phase 0.5:** `vqa_dataset.py:198-205`, `docs/02-ct-scans.md:45`.
- `docs/02-ct-scans.md`, `docs/01-pathology-and-path-tools.md`, `docs/04-running-the-pipeline.md`,
  `docs/05-retrieval.md` — update to the adapter/selector/filter-chain/assembly model.

## Open questions — resolutions (2026-07-05)

**All 12 OQs + the fresh-Claude findings resolved. Phase 1 is unblocked** (the three items the plan had
wrongly called "no preconditions" — EHR layer, meds_tools install, inline scope — are now decided).

- **A** — leaf-only; study/series & specimen/block/slide upstream in VISTABench (code-verified). Re-add is a
  cross-repo vista_bench change.
- **B** — `by_model` off `MODEL_REGISTRY`, override allowed.
- **C** — generic selector signature; `summarize` = lift a small cluster (decide the fallback); learned
  slice-selector deferred.
- **D** — inline **fails closed** at preflight via `supports_inline` on `BaseVLMAdapter` (default False).
- **E** — drop the buggy `i*0.1` sampler.
- **F** — cohort axis stays Phase 3.
- **G** — **superseded (2026-07-06):** `meds_tools` is **not a dependency** — LUMIA-as-input deletes the
  `get_described_events_window` fetch, so the SHA-pin / `py>=3.14` floor / `uv.sources` concerns are moot.
- **H** — defaults = prior runtime: 100 tiles/slide, per-preset slice counts.
- **I** — canonical smoke → PFS `progression_recurrence_free_survival_1_yr` (`_2_yr` alt); no mortality task exists.
- **J** — viewer token-budget bar; output → su-vista PHI mount; side-by-side optional.
- **K** — gate-1 byte-identical target (verified on one example); declared-delta fallback OK.
- **L** — v1_5 substrate inlined; **mixed baseline** (BQ `v1_1` + task JSON `v1_3`).
- **EHR layer (fresh-Claude, superseded 2026-07-06)** — was "re-home the offline prep"; now **consume the
  given LUMIA and run filter + flat-render + truncate LIVE at inference** (no prep, no CSV, no DB/ontology).
- **Inline scope (fresh-Claude)** — **defer to Phase 1.5**; Phase 1 ships the `supports_inline` seam only.

**Small decisions still open (non-blocking):** (i) PFS **1yr vs 2yr** smoke; (ii) `summarize` on/off for the
first sweep **and its failure mode** (fail loud vs swallow); (iii) side-by-side in the first viewer cut.
**Verify on VM:** the LUMIA corpus carries the flat renderer's fields (`numeric_value`/`unit`/`text_value`/
`description`) and covers the eval cohorts — see below.

## Pre-implementation checklist

- [ ] `normalize_experiments` resolves bare-string **and** block-list dict to `{name, display, cohort, assembly, blocks}`.
- [ ] All **four** readers consume normalized names/labels; `ct_experiment_plot.parse_experiment_comments` deleted + `:190` rewired.
- [ ] Output path + resume + retrieval-overwrite preserved; defaults = prior runtime.
- [ ] Leaf selection only (CT slice, path patch, EHR filter chain); study/series & specimen/block/slide upstream (OQ-A); counts read off the block.
- [ ] EHR adapter ingests given LUMIA + filters LIVE at inference (`window/note_type_filter/code_filter`); STANFORD skip = a `code_filter` row-drop; flat-render matches `get_llm_event_string` field-for-field; `truncate_timeline` lifted **verbatim**; summarize fallback decided; **no third formatter copy** (incl. `test_meds_tools.py:6`).
- [ ] Assembler emits a typed content sequence; `ordered` ships; `supports_inline` on `BaseVLMAdapter` (default False); `inline_by_timestamp` **fails closed at preflight**; real interleaving = Phase 1.5.
- [ ] Viewer output → su-vista PHI mount + `.gitignore`; weight-free for deterministic selectors; model-backed step rejected-or-marked; model-load split out of `TaskOrchestrator.__init__`.
- [ ] LUMIA field-coverage confirmed on a real `.xml` (`numeric_value`/`unit`/`text_value`/`description`) + cohort coverage; **no `meds_tools`/`meds_reader`/ontology dependency**; equivalence fixture green within allowlist.
- [ ] Staged golden green (sorted by `(person_id, index)`): gate 1 byte-identical (imaging/structure), gate 2 `by_model` image-hash-identical, gate 3 EHR LUMIA-render within allowlist **on the live path (no CSV regen)**.

## Verification & VM handoff

Executed on the GCP VM (Mac is planner-only). Canonical smoke =
`progression_recurrence_free_survival_1_yr` (PFS — OQ-I; no mortality task) × `gemma3 medgemma-1.5-4b`.

- **Phase 0.5** — 30 `axial_all_image` indices evenly spaced across `[0, depth)`, no clamp; docs match code.
- **Phase 1 (scope)** — CT slice + pathology patch only; study/series & specimen/block/slide upstream (OQ-A).
- **Phase 1 (LUMIA field-coverage — likeliest breaker)** — confirm the corpus carries the flat renderer's
  fields and covers the eval cohorts. Pull one real `.xml` and diff its field set against
  `get_llm_event_string`'s inputs: `python -c "import xml.etree.ElementTree as ET; r=ET.parse('<one>.xml').getroot();
  print({a for e in r.iter('event') for a in e.attrib}); print([e.text for e in list(r.iter('event'))[:3]])"`.
  **Expected:** events expose (or the corpus otherwise carries) `code`, a `description`/`name`, lab
  `numeric_value`+`unit`, and note-body `text_value`; every eval `person_id` has an `.xml`. **Stop:** a field
  the flat renderer needs is absent (→ gate 3 becomes a declared-delta, OQ-K) or cohort coverage is partial
  (→ retain a DB-fetch fallback only for the gap). **No `meds_tools`/`meds_reader`/ontology import is
  attempted — if the design still needs one, the LUMIA-as-input premise failed; STOP and re-plan.**
- **Phase 1 (staged golden, sorted)** — harness writes `(index, person_id, dynamic_prompt, selected_indices,
  image_count, path_tile_count, image_hashes, adapter_prompt_string, assembly_mode)` before/after, **sorted by
  `(person_id, index)`**. Gate 1 byte-identical (imaging/structure); gate 2 `by_model` hashes identical per
  model; gate 3 EHR LUMIA-render within allowlist (**within-line field order only; event order identical**)
  **on the live inference path — no CSV to regenerate** — assert the LUMIA-derived `dynamic_prompt` matches
  the current one, and that a *filtered* config drops exactly the right events. **Stop:** any gate-1/2 drift;
  gate-3 outside allowlist without a declared LUMIA-field delta.
- **Phase 1 (downstream contract)** — all **four** readers over a config with a legacy string **and** a
  block-list dict; `ct_experiment_plot` shows a correct display label. **Stop:** TypeError on dicts / missing
  experiment / blank display label.
- **Phase 1 (assembler seam)** — a config requesting `inline_by_timestamp` against a model with
  `supports_inline = False` **raises a clear error at preflight** (before weights load). **Stop:** silent
  front-load / silent downgrade.
- **Phase 2 (viewer)** — offline; token-budget bar; a known `used_image==0` row flagged; CT slices match
  selector indices; a model-backed step rejected-or-marked; output on the su-vista PHI mount, `git status`
  does **not** list it. **Stop:** external refs / PHI staged / model-backed step silently rendered / shown
  context diverges from the inference CSV.
- **Phase 3** *(if taken)* — cohort `source` switch reproduces each loader's row counts; resume/output/retrieval
  unchanged. **Stop:** any rowcount or output-path drift.

`/vm-handoff` renders this section into a runnable `docs/vm-status/<date>-<sha>.md` on the VM.
