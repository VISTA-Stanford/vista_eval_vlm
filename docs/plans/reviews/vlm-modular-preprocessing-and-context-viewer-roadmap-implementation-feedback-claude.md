Reference: docs/claude_ops.md

# Implementation Feedback (fresh-Claude second pass): VLM modular preprocessing framework (foundation slice)

Scope reviewed: committed range `b94bf09..HEAD` (4 commits) — Phase 0.5 CT overshoot fix, the additive `src/context/` package, the three reader normalizations, and the `supports_inline` seam. Deferred items (hot-path wiring, Phase 2 viewer, Phase 1.5 inline) were treated as out of scope and are not flagged as missing.

## Verdict

Keep as-is for the foundation slice — no new blocker. The two Codex High/Medium blockers (reader import coupling, pathology row-drop) are **both confirmed-resolved** in the committed code, and every lifted function I checked is byte-identical to its legacy source. One **Medium fix-forward** item to resolve *with* the hot-path wiring pass (preset `variant` markers are wired to nothing — the plan's no_img_report filter chain is defined nowhere), plus a few Low doc-accuracy nits.

## New Findings (not in the Codex pass)

- **Medium | Preset `variant` markers are dead — the plan's no_img_report/report EHR filter chain is expressed nowhere, so gate-3 byte-identity is unwired.** `_ehr_block("no_img_report")` (and `report`, `passthrough`) only stamps `config={"variant": variant, "serialize": {...}}` with **no `select` chain** (`src/context/presets.py:26-35`, `:74-105`). The EHR adapter reads `self.config.get("select")` and never looks at `variant` (`src/context/adapters/ehr.py:155`), so for the `no_report` preset `self.filters == []` — no `window(6mo)`, no `code_filter(exclude_stanford)`. Yet `ehr.py:12-14` and `ehr_filters.py:8-14` assert "the STANFORD/report skip is a `code_filter` row-drop" as if it were wired. `grep variant src/context` confirms the token is referenced only in presets and never consumed. Result: the plan's gate-3 promise ("a *filtered* config drops exactly the right events") has no preset that exercises it, and when wiring lands this preset either silently produces an **unfiltered** timeline or falls back to the (plan-retired) `remove_imaging_report.py` loader. Required fix: decide now whether `variant`→filter-chain translation lives in `presets.py` (emit `select: [{fn: window, before: "6mo"}, {fn: code_filter, exclude_stanford: true}]` for `no_img_report`) or is explicitly deferred to the wiring pass — and, if deferred, soften the "handled by code_filter" claims in the docstrings so they don't read as already-wired.

- **Low | `Assembler.to_flat_item` docstring overclaims byte-identity and will mislead the wiring author.** `to_flat_item` sets `question = "\n".join(text-block payloads)` (`src/context/assembler.py:98-105`) and the docstring says "This is the exact shape `model.create_template` consumes today" (`:95-97`). But legacy `create_template` consumes `question = str(row['dynamic_prompt'])` — the *fully templated* prompt with `[PATIENT_TIMELINE]` already substituted (`src/vqa_dataset.py:116`), not the bare EHR timeline text. The assembler's own docstring correctly notes template stitching stays in the orchestrator (`:10-13`), which contradicts the `to_flat_item` claim. No runtime impact today (deferred), but the "exact shape consumed today" wording is wrong and load-bearing for whoever wires this. Required fix: reword to "the `{question, image}` container shape" and drop the byte-identity implication.

- **Low | Plot-label behavior silently changes; no migration for existing `# comment` labels.** Deleting `parse_experiment_comments` and routing to `display_names` (`src/results/plot_code/ct_experiment_plot.py:141-153`, `generate_pdf.py:50-58`) means labels now come from a config `display:` key, falling back to the raw experiment **name**. Legacy scraped the YAML inline comment (`- axial_all_image  # Axial CT + timeline`) as the label. Any existing config that carried pretty labels as `# comments` will now plot the bare name until a `display:` key is added. The plan mandated the deletion, but the comment→`display:` migration of existing configs is not part of this slice. Required fix: none code-side; note it in the config-migration checklist so plots don't regress to raw names on the next run.

- **Low | Reader hard-crashes (`ValueError`) on a dict experiment entry missing `name`.** `_normalize_entry` raises when a dict has no string `name` (`src/context/normalize.py:85-88`), and `experiment_names`/`display_names` call it eagerly, so `final_metrics.load_config` / `generate_pdf` / `ct_experiment_plot` now abort on a single malformed entry rather than skipping it during result *discovery*. This is arguably better than the old silent `TypeError`, but result-discovery readers scanning a partially-authored config will fail closed. Flagging for an explicit keep/soften decision (a discovery reader could `unknown=True`-passthrough instead of raising). Non-blocking.

- **Low (informational, not a defect) | The "four readers" are covered, one transitively.** `all_model_response.py` was listed in the plan as a break-risk reader but is untouched here; it imports `load_config` from `results.final_metrics` (`src/results/all_model_response.py:15-19`), so it inherits the normalized `experiment_names` path for free. `check_image_usage.py` does not exist in this worktree. So no reader gap — recording this so it isn't re-flagged.

## Codex Findings — Independent Re-verification

- **Reader import-coupling (Codex High / Contract Violation) — Confirmed-fixed.** `context/__init__.py` now imports only the pure-stdlib surface eagerly (`.block`, `.assembler`, `.normalize`, `.presets`) and defers every adapter/specs symbol behind a PEP 562 `__getattr__` lazy loader (`src/context/__init__.py:17-56`). Traced the eager graph: `block` → dataclasses/typing; `assembler` → `.block`; `normalize` → `.presets` → `copy`. None pull numpy/PIL/pandas or project modules. `from context.normalize import experiment_names` (the reader import at `src/results/final_metrics.py:19`) therefore no longer drags the imaging stack. Resolved.

- **Pathology empty-tile row-drop (Codex Medium / Byte-identity) — Confirmed-fixed.** `PathologyAdapter.materialize` now performs the drop itself: `df = df[[len(p) > 0 for p in path_tile_paths_list]].copy()` under `drop_empty=True` default (`src/context/adapters/pathology.py:96-97`), positionally identical to legacy `run_bq.py:527`. The `drop_empty=False` escape hatch for the viewer is a clean addition. Resolved.

- **prompt_style outside `.spec` (Codex Question / Defensible Deviation) — Confirmed as intended.** `NormalizedExperiment` carries `prompt_style` + `legacy_retrieval` as fields but `.spec` deliberately omits them, documented at `src/context/normalize.py:35-42`, matching the plan's "prompt_style threaded but NOT in .spec." Kept.

- **LUMIA chronological sort (Codex Low / Byte-identity) — Still present, VM-gated (unchanged).** `parse_lumia` still `sort_values("time", kind="stable")` (`src/context/adapters/ehr.py:137`). The stable sort keeps document order on ties, and the docstring flags the VM check. Legacy `get_llm_event_string` iterates rows without sorting, relying on `get_described_events_window`'s MultiIndex `(subject_id, time)` order. Defensible; correctly deferred to the gate-3 VM check. No regression.

- **summarize fail-loud (Codex Defensible Deviation) — Confirmed.** `summarize` raises `NotImplementedError` with a clear message (`src/context/selectors/ehr_filters.py:126-143`) and is registered in `MODEL_BACKED_FILTERS` (`:154`), matching "decide the failure mode → fail loud." Kept.

- **Canonical import over vendoring (Codex Defensible Deviation) — Confirmed.** EHR adapter imports `get_llm_event_string`/`truncate_timeline` from `data_tools.utils.meds_timeline_utils` rather than copying (`src/context/adapters/ehr.py:34`). Stricter than the plan's "vendor/lift" text and avoids a third formatter copy. Kept.

- **Unknown bare-string passthrough (Codex Defensible Deviation) — Confirmed.** `_normalize_entry` returns `unknown=True` for unknown bare strings (`src/context/normalize.py:78-81`). Kept.

## Byte-identity Risks

All lifted functions match their legacy sources exactly — verified line by line:

- **`windowing.norm` / `multi_window_rgb` == legacy `norm` / `window`.** `src/context/windowing.py:19-36` vs `src/vqa_dataset.py:42-57`: same clip order, same `float32` cast, same `[(-1024,1024),(-135,215),(0,80)]` clips, same `np.stack(..., axis=-1)`. Identical.
- **`windowing.grayscale` == `normalize_slice`.** `src/context/windowing.py:39-51` vs `src/vista_run/utils/utils_inference.py:43-57`: same NaN/inf guard, same min/max branch, same `uint8` cast. Identical.
- **`ct_selectors.evenly_spaced_k` == the legacy per-branch sampler.** `src/context/selectors/ct_selectors.py:16-36` reproduces `position = i/(k-1)` (with the `k==1 → 0.0` guard) and `index = int(position*(depth-1))` plus the never-firing `if index >= depth: index = depth-1` clamp, matching `src/vqa_dataset.py:201-212` (axial 30), `:220-231` (no_timeline 50), `:238-249` (no_report 10) exactly. Presets 30/50/10 (`src/context/presets.py:68,72,76,90`) map byte-identically onto post-Phase-0.5 `__getitem__`.
- **Phase 0.5 fix is correct and complete.** `src/vqa_dataset.py:193-212` now uses `num_slices=30` with `i/(num_slices-1)`; the buggy `i*0.1` is gone. The 50- and 10-slice branches already used even spacing (diff shows only the axial hunk changed). `docs/02-ct-scans.md:44-47` now correctly states 50 for `no_timeline`/`all_vb_image_only`, 30 for `axial_all_image` + retrieval+image, 10 for `no_report`, with the `int(i/(n-1)*(depth-1))` formula. Docs match code.
- **`path_selectors.random_n` + `PathologyAdapter.materialize` == legacy `_load_path_task_data`.** `src/context/selectors/path_selectors.py:26-35` + `src/context/adapters/pathology.py:74-98` reproduce: `Random(42)` (== module `random.seed(42)`, same Mersenne Twister) seeded once, `iterrows()` row order, all-if-`<=n` (no draw) else `rng.sample`, `str(p.resolve())` applied post-sample, `sorted(glob("*.jpg")) + sorted(glob("*.jpeg"))`, `folder.is_dir()` guard, and the positional empty-row drop — matching `src/vista_run/run_bq.py:507-527`. `folder_name_from_path_image_path` is a verbatim lift (`pathology.py:51-62` vs `run_bq.py:486-498`). Identical.
- **CT `preprocess_slice` == `_process_ct_slice`.** `src/context/adapters/ct.py:46-54`: gemma → `np.round(...,0).astype(uint8)` → `fromarray(mode="RGB")`; else `grayscale` → `fromarray(mode="L")`; then `pad_to_size(target)`. Matches `src/vqa_dataset.py:94-107`. `target_size`/`is_gemma` resolution (`src/context/adapters/base.py:58-62`) reproduces legacy 448/512 and windowing dispatch. Identical.
- **EHR STANFORD-skip predicate equivalence (mechanism correct; not wired via presets — see Medium above).** `code_filter(exclude_stanford=True)` drops `codes.str.contains("STANFORD", regex=False)` (`src/context/selectors/ehr_filters.py:115-116`), the same substring test as `get_llm_event_string`'s inline `if 'STANFORD' in code_val: continue` (`src/data_tools/utils/meds_timeline_utils.py:215-218`), and the adapter renders with `exclude_report=False` (`src/context/adapters/ehr.py:198`). The predicate is byte-equivalent; the risk is that no preset actually adds it.
- **LUMIA field-map is sound against `markup.md`.** `markup.md:62-68` documents `<event>` with only `note_id/provider_id/care_site_id/type/code/name` and element-text = a state token ("start"). `_lumia_event_to_row` (`src/context/adapters/ehr.py:53-93`) reads exactly those, defaults the four renderer-extra fields (`numeric_value/unit/text_value/description`) to `None`, records absences via `lumia_missing_fields` (`:141-147`, surfaced on block metadata at `:193`), and the `description=name` fallback is defensible because the example shows `name` carrying the ontology description (`markup.md:97`). The `text_value` = element-text-when-not-a-state-token logic (`:69-73`) correctly excludes "start"/"end"/empty. Provider `speciality` join is correctly encounter-scoped: `encounter.iter("provider")` builds the per-encounter `provider_id→speciality` map and events resolve within the same encounter (`:117-126`). All consistent with the plan's VM field-coverage precondition — gate 3 remains a declared-delta if the corpus lacks a field, exactly as designed.
- **`ehr_filters.window` calendar offset == legacy.** `_as_offset("6mo") → pd.DateOffset(months=6)` with integer coercion (`src/context/selectors/ehr_filters.py:55-58`), and `window` default `before="6mo"` (`:62`) reproduces `remove_imaging_report.py:46` `embed_time - pd.DateOffset(months=6)`. The duration regex `^\s*(\d+(?:\.\d+)?)\s*(mo|[dwyh])\s*$` handles `365d`/`6mo`/`1y`/`12h`; months/years force `int(qty)` (calendar-aware). Boundary inclusivity (`>= start & <= end`, both inclusive, `:78`) vs `get_described_events_window`'s semantics is the one VM-checkable open — but moot until the filter is actually wired (Medium finding).

## Contract / Correctness Issues

- **Import graph is clean; no cycles.** `specs.py` → `adapters` + `assembler` + `block`; `adapters/*` → `..block`, `..selectors.*`, `..windowing`, and project modules (`vista_run.utils.utils_inference`, `data_tools.utils.meds_timeline_utils`); nothing imports back into `context/__init__`. The heavy modules are reachable only via the lazy `__getattr__`, so the reader path stays pure-stdlib. Verified.
- **`ContextBlock` representation guard** fails closed on an invalid `representation` (`src/context/block.py:51-56`). `image_count` handles `None`/non-list payloads. Sound.
- **Assembler preflight fails closed correctly.** `preflight_assembly` raises `AssemblyError` for `inline_by_timestamp` unless `supports_inline` is True on the resolved adapter/caps (`src/context/assembler.py:38-55`), reading the capability off the model layer (not a bare `model_type`) to avoid the `context→models` cycle. `content_sequence` for `ORDERED` returns `image_entries + text_entries` (images-then-text), matching every model's `create_template`; the `INLINE_BY_TIMESTAMP` branch raises until Phase 1.5 (`:84-89`). `BaseVLMAdapter.supports_inline = False` class attr present (`src/models/base.py:34-36`). Correct and complete for the seam.
- **`resolve_model_caps` override precedence is correct for presets.** `by_model` string overrides are correctly ignored (`windowing != "by_model"` guard, `target_size` `isinstance int` guard at `src/context/adapters/base.py:72-75`), so `_ct_block`'s `{windowing: by_model, target_size: by_model}` resolves to the legacy 448/512 default. No drift.

## Test Gaps

(Consistent with Codex's list; no runtime here — these are for the VM.)

- A preset-composition parity test: assert `normalize_experiments` over each legacy name yields the CT slice count / path `n=100,seed=42` the legacy `__getitem__`/`_load_path_task_data` used — and, critically, assert what filter chain (if any) `no_report`/`report` resolve to, so the Medium `variant` gap is caught before wiring.
- Reader-only import smoke (`from context.normalize import experiment_names` with no numpy) — would regression-guard the resolved Codex High.
- `to_flat_item` / `content_sequence` ordering test (images-then-text) and the `inline_by_timestamp` preflight `AssemblyError` against a default adapter.
- Pathology `materialize` fixture (one missing-folder row, one oversize tile list) asserting drop + sample order + `str(resolve())` paths vs legacy.

## Defensible Deviations

- Canonical import of `get_llm_event_string`/`truncate_timeline` over vendoring (`src/context/adapters/ehr.py:34`) — stricter "no third copy" discipline; byte-identical by construction.
- `summarize` as a fail-loud `NotImplementedError` seam (`src/context/selectors/ehr_filters.py:126-143`) — matches "decide the failure mode."
- Unknown bare experiment names pass through with `unknown=True` (`src/context/normalize.py:78-81`) — tolerant result-discovery.
- `materialize(drop_empty=...)` param exposing the legacy drop as the default while giving the viewer a full-frame escape hatch (`src/context/adapters/pathology.py:74-98`) — sensible extension.
- Lazy `__getattr__` package surface (`src/context/__init__.py:34-56`) — clean solution to the reader import-footprint constraint.

## Questions For The Author

1. Is the preset `variant` marker (`no_img_report`/`report`/`passthrough`) meant to be translated into an actual EHR `select` filter chain in `presets.py` now, or is that translation deliberately deferred to the hot-path wiring pass? As committed nothing consumes it, and the docstrings read as if the code_filter skip is already wired. (Medium finding.)
2. For `no_report`, at wiring time will the STANFORD/6-month behavior come from the new `code_filter` + `window` chain (plan intent) or does the legacy `remove_imaging_report.py` loader still run to produce a pre-rendered `patient_string` that the EHR block just passes through? The two paths diverge and only one is gate-3-testable.
3. Should result-discovery readers fail closed (`ValueError`) on a dict entry missing `name` (`src/context/normalize.py:85-88`), or passthrough like unknown bare strings so a half-authored config doesn't abort a metrics scan?

## Audit Trail (files inspected, paths only)

- Plan: `docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md`
- Prior review: `docs/plans/reviews/vlm-modular-preprocessing-and-context-viewer-roadmap-implementation-feedback.md`
- New package (read in full): `src/context/__init__.py`, `block.py`, `windowing.py`, `normalize.py`, `presets.py`, `specs.py`, `assembler.py`, `adapters/__init__.py`, `adapters/base.py`, `adapters/ct.py`, `adapters/pathology.py`, `adapters/ehr.py`, `selectors/__init__.py`, `selectors/ct_selectors.py`, `selectors/path_selectors.py`, `selectors/ehr_filters.py`
- Edited files: `src/vqa_dataset.py`, `src/models/base.py`, `docs/02-ct-scans.md`, `src/results/final_metrics.py`, `src/results/plot_code/ct_experiment_plot.py`, `src/results/plot_code/generate_pdf.py`
- Legacy sources (byte-identity): `src/vqa_dataset.py`, `src/vista_run/utils/utils_inference.py`, `src/data_tools/utils/meds_timeline_utils.py`, `src/vista_run/run_bq.py` (`_load_path_task_data` 443-539; `_build_prompts_for_experiment` 732), `src/data_tools/OMOP_meds_query/remove_imaging_report.py`, `/Users/philadamson/Documents/Stanford/VISTA/code/meds2text/docs/markup.md`
- Cross-checks: `src/results/all_model_response.py` (transitive `load_config`); grep for `parse_experiment_comments` (0 refs), `context` imports in hot path (0), `variant` consumers (0 outside presets)
- `git diff --stat b94bf09..HEAD`, `git log --oneline`, `git status --short` (clean tree)

## Author Resolutions (2026-07-06)

- **Medium (variant markers / no_img_report chain unwired):** took the reviewer's *defer* option. Fixed the overclaiming docstrings so they read as mechanism-not-yet-wired (`ehr.py`, `ehr_filters.py`), and documented the deferred per-variant `select` chains + gate-3 gating explicitly in `presets._ehr_block`. The concrete `variant`→`select` encoding lands in the Task-3 hot-path pass where gate 3 verifies it on the VM.
- **Q1 (encode now vs defer):** deferred to the wiring pass (documented in `presets.py`).
- **Q2 (no_report: live-filter vs legacy passthrough):** answered by the plan — retire `remove_imaging_report.py`, go LUMIA-live with `window(6mo)` + `code_filter(exclude_stanford)`. Noted in `presets._ehr_block`.
- **Q3 (ValueError on nameless dict):** kept fail-closed — a block-dict with no `name` has no filename/metrics token and is genuinely malformed; a clear error beats silently dropping an experiment from a metrics scan.
- **Low (to_flat_item docstring):** reworded to "container shape," dropping the byte-identity implication.
- **Low (plot-label silent change):** intended per the plan (delete `parse_experiment_comments`); no code change — flagged for the config-migration checklist (add explicit `display:` keys where pretty labels were carried as `# comments`).
