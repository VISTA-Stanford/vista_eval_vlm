Reference: docs/claude_ops.md

# vista_eval_vlm — Step 5: wire the LUMIA-live EHR adapter into the real hot path

**Status: Draft** (2026-07-16)

## Context

vista_eval_vlm's ContextBlock modality-adapter roadmap already landed (2026-07-15/16): CT and
pathology both flow through typed adapters, and Phase 2 shipped a weight-free HTML viewer for
manual pre-run input-QA. But the EHR side is still a "seam only" — `EHRAdapter`
(`src/context/adapters/ehr.py`) and its filter chain (`src/context/selectors/ehr_filters.py`) are
**fully built and unit-clean but entirely unused**. The real inference hot path
(`src/vista_run/run_bq.py:_build_prompts_for_experiment`) still substitutes the pre-rendered
`patient_string` CSV/BQ column directly into the prompt template — it never parses a real LUMIA
`.xml` timeline, never applies the `window`/`code_filter` chain, never calls the adapter at all.
This was intentional sequencing (Step 5 was explicitly deferred behind Phase 2 in the roadmap), but
it's now the last piece of the "EHR consumes full LUMIA timelines live at inference" design that
was locked in back in the original roadmap review (2026-07-06 serialization interrogation).

Phil asked to build this now, then generate a few `context_viewer.py` HTML renders across
different configs so he can personally eyeball the live-rendered timeline text (not just
structural checks) before this becomes the default behavior.

## Goal

Replace the raw `patient_string` substitution with the real live path — parse each patient's
actual LUMIA `.xml`, run it through the already-built filter chain (`window` + `code_filter`),
render via the canonical `get_llm_event_string`/`truncate_timeline` — for the `timeline` preset
variant (`no_image`, `axial_all_image`) and the `no_img_report` variant (`no_report`,
`timeline_only`, `report`). Leave `passthrough` (`all_vb_timeline_only`, `path_full`) untouched —
it's deliberately staying pre-rendered — and leave retrieval experiments (`retrieved_timeline` and
its variants) untouched too, since they route through their own `build_retrieval_prompts` path and
never reach `_build_prompts_for_experiment`'s generic branch (`src/vista_run/run_bq.py:798`). Then
prove it's correct (byte-diff where provable, human eyeball where it isn't) and hand Phil a small
batch of real rendered HTML to review.

## Approach

**1. `presets.py` — `_ehr_block(variant)` emits the real `select` chain instead of just a label.**
Currently it only stamps `"variant": variant` into the block config; the concrete filter chains
documented in its own docstring were never encoded. Emit them for real:
```python
def _ehr_block(variant: str = "timeline"):
    if variant == "no_img_report":
        select = [{"fn": "window", "before": "6mo", "after": "0d"},
                  {"fn": "code_filter", "exclude_stanford": True}]
    elif variant == "timeline":
        select = []  # Phase 1 of verification below may flip this to add code_filter
    else:  # passthrough
        select = None
    config = {"variant": variant, "serialize": {"style": "flat_timeline"}}
    if select is not None:
        config["select"] = select
    return {"id": "ehr", "modality": "text", "adapter": "ehr", "config": config}
```
`passthrough` (`all_vb_timeline_only`, `path_full`) keeps `select` absent entirely — untouched.
The `timeline` chain is a single, clearly-commented line to flip once Phase 1's decision gate
resolves it — not a second runtime config toggle (one edit point, not two overlapping mechanisms).

**2. `run_bq.py` — one helper, called once at the top of `_build_prompts_for_experiment`.**
Analogous to (not identical to) the existing pattern where `PathologyAdapter` is imported and
constructed inline inside `_load_path_task_data` (no orchestrator-`__init__` singleton, config
differs per experiment) — `PathologyAdapter` there uses a cohort-level `materialize(...)` call,
while EHR is genuinely per-row (`ingest()`/`contextualize()` per patient), built once per call via
`.apply`. **Missing-LUMIA-file policy (Phil's call): fail-closed — exclude the row from the
experiment entirely, don't fall back to `patient_string`.** That means no per-row branching is
needed at render time; filter first, then render only the survivors:
```python
_EHR_ADAPTER_EXPERIMENTS = {"no_image", "axial_all_image", "no_report", "timeline_only", "report"}

def _apply_ehr_adapter(self, df_exp, experiment, timeline_col):
    if experiment not in _EHR_ADAPTER_EXPERIMENTS or timeline_col not in df_exp.columns:
        return df_exp
    if "person_id" not in df_exp.columns:
        raise ValueError(f"experiment={experiment!r} requires person_id to resolve LUMIA files, but it's missing")
    ehr_block = next(b for b in get_preset(experiment)["blocks"] if b["id"] == "ehr")
    adapter = EHRAdapter(config=copy.deepcopy(ehr_block["config"]))
    corpus_dir = self.cfg.get("retrieval", {}).get("corpus_dir")
    if not corpus_dir:
        raise ValueError(
            f"retrieval.corpus_dir is not configured (experiment={experiment!r}); "
            f"set it in your VM-local configs/all_tasks.vm.yaml overlay, e.g. pointing at a local "
            f"mount of gs://vista_bench/thoracic_cohort_lumia/"
        )

    def has_lumia(pid):
        xml_path = lumia_path_for(pid, corpus_dir)
        return xml_path is not None and xml_path.exists()

    mask = df_exp["person_id"].apply(has_lumia)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"[EHR] dropping {n_dropped}/{len(df_exp)} rows with no LUMIA file (experiment={experiment})")
    df_exp = df_exp[mask].copy()

    def render(row):
        ctx = {"embed_time": row.get("embed_time"), "model_type": self.model_type,
               "lumia_corpus_dir": corpus_dir,
               "timeline_truncation": self.cfg.get("timeline_truncation")}
        return adapter.contextualize(adapter.ingest(row["person_id"], ctx), ctx).payload

    df_exp[timeline_col] = df_exp.apply(render, axis=1)
    return df_exp
```
Called as the first line of `_build_prompts_for_experiment` (right after `df_exp = df.copy()`).
Every branch below it — the generic `[PATIENT_TIMELINE]` substitution and the `report` branch's
`row[timeline_col]` read — is unchanged; they now just consume already-rendered text over the
(possibly smaller) filtered `df_exp`. `path_full` stays outside the whitelist, so it keeps reading
raw `patient_string` exactly as today. The `report`/`path_full` branches' column-append logic runs
strictly *after* this single mutation point, so there's no double-substitution or ordering bug.
`context_capture.py` shares this same method (`src/vista_run/context_capture.py:190`, which calls
`_build_prompts_for_experiment` *before* its own sort+`head(limit)` step — see Verification Phase 1
below for why that ordering matters), so the golden-harness/viewer capture path gets the live
rendering (and the same row-exclusion) for free — no separate change needed there, just
re-verification once wired. **The `print(...)` drop-count line is load-bearing, not decoration**:
per Phil's fail-closed choice, a low-coverage cohort now directly shrinks eval N, so every run must
surface exactly how many rows were dropped — never a silent count.

*Truncation*: every surviving row's `timeline_col` is **fully replaced** by the adapter's freshly
rendered + truncated text (`adapter.contextualize(...).payload`), not derived from the old
`patient_string` value — so `_load_task_data`'s earlier truncation of `patient_string`
(`src/vista_run/run_bq.py:331`) on that now-discarded value is irrelevant, and there's no
double-truncation for the wired presets. (Note: `truncate_timeline` is *not* idempotent —
`meds_timeline_utils.py` unconditionally prepends a newline every call — but that's moot here since
no surviving row's timeline text is ever truncated twice.)

**3. Corpus-path config — reuse the existing `retrieval.corpus_dir` key; do NOT add a second one
(Phil, via `/explain-plan` feedback: "this should be one config path — no reason to be
independent").** An earlier version of this plan proposed a new `paths.lumia_corpus_dir` key
alongside the existing `retrieval.corpus_dir` (both pointing at the same physical LUMIA corpus).
Since there's no real reason for two config surfaces over one physical directory, `_apply_ehr_adapter`
reads `retrieval.corpus_dir` directly (Approach #2) — zero new config key, zero changes to
retrieval's own code (it already reads this key unchanged).
- `configs/all_tasks.yaml:93` already has
  `retrieval.corpus_dir: "/data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia"` — contributor
  Ryan D'Cunha's own machine-specific path, already documented as **broken** on the actual executor
  VM in `docs/vm-status/2026-07-06-golden-harness.md` (`phil-sllm-01` only has `/mnt/su-vista-hot`
  and `/mnt/su-vista-uscentral1`, no `/data/fries` at all). The fix is the same VM-local-overlay
  pattern used elsewhere in this config (`base_dir`/`ct_dir`/`results_dir`) — just applied to the
  existing key rather than a new one: each VM operator sets the real value in their git-ignored
  `configs/all_tasks.vm.yaml` overlay. No committed change to `all_tasks.yaml` needed.
- Phase 0 below confirms `phil-sllm-01`'s actual local path resolves — the 2026-07-06 doc already
  proved a GCS pull from `gs://vista_bench/thoracic_cohort_lumia/` works, so a resolvable local
  mount should exist; it just needs VM-side confirmation, not a Mac-side guess.
- `lumia_path_for` (already in `ehr.py`) keeps a one-line defensive guard: `Path(None)` raises, so
  return `None` instead if `corpus_dir` is `None`. `_apply_ehr_adapter` never hits this (it raises
  earlier), but other callers (`context_viewer.py`, future consumers) shouldn't get a confusing
  `TypeError`.

## Files to Modify

- `src/context/presets.py` — `_ehr_block(variant)` emits the real `select` chains (Approach #1).
- `src/vista_run/run_bq.py` — new `_apply_ehr_adapter` helper + one call at the top of
  `_build_prompts_for_experiment`; import `EHRAdapter`/`lumia_path_for` from `context.adapters.ehr`
  (Approach #2). No change needed in `src/vista_run/context_capture.py` itself (shares the same
  method) — but re-verify it post-wiring per the golden-harness gate.
- `src/context/adapters/ehr.py` — one-line `None`-guard in `lumia_path_for` (Approach #3).
- `src/vista_run/diff_golden.py` — fill in `normalize_text`'s field-order/formatting allowlist
  rules once Phase 2 of Verification shows what actually differs (can't be written from the Mac in
  advance of real VM output).
- `docs/04-running-the-pipeline.md` and `docs/00-data-setup.md` — both currently describe the EHR
  source as the pre-rendered `patient_string` CSV column; update to describe the live LUMIA path +
  fail-closed row exclusion, and note `retrieval.corpus_dir` is now dual-purpose (also read by the
  EHR adapter, not just retrieval) for the wired presets.

## Open Questions

- **`timeline` variant's `code_filter` scope**: whether the full-timeline preset should also drop
  STANFORD-tagged codes like `no_img_report` does was left explicitly unresolved in the original
  design (`presets.py` docstring: "gate-3-VM-gated, confirm before encoding"). Resolved via a
  self-resolving VM decision gate (Phase 1 below) rather than a Phil judgment call — it's an
  objective byte-diff question.
- **Full-run coverage threshold** (Codex review): Phase 0's ≥95%/80–95%/<80% bands below are sized
  for this plan's PFS-1yr smoke. What threshold should gate a full default (multi-task) run once
  this lands — same bands, or does a production run warrant a stricter bar given the cohort-size
  consequence is now permanent, not just a smoke artifact?
- **Where dropped-row/post-drop-denominator reporting should live long-term** (Codex review):
  this plan surfaces it via the `[EHR] dropping N/M rows` console log + the VM handoff doc readback
  (Phase 0), which is sufficient for this plan's scope. Should it *also* become a persisted
  per-run field (e.g. in the result CSV metadata or a run manifest) so future comparisons against
  historical full-cohort results can see the denominator without re-reading logs? Out of scope for
  this plan either way — flagging so it doesn't get silently forgotten.

## Verification & VM handoff

**What runs on the VM** — all steps on **Claude-Code CPU `phil-sllm-01`** (weight-free rendering +
BQ/GCS queries only; no GPU/hcpu split needed). Multi-phase because of a real precondition →
self-resolving decision gate → verification → human-QA sequence — a **complex-tier** handoff per
this repo's own classifier, phased below so round-trips are designed, not accidental.

### Phase 0 — cohort-vs-LUMIA coverage join (precondition)

**Authoritative source (Codex review): derive the cohort from the actual loader path, not a
hand-authored parallel query.** Don't independently re-derive the PFS-1yr cohort via a fresh BQ
query — that risks silently diverging from what the eval actually iterates. Instead, run
`_load_task_data`'s real BQ+CSV merge (`src/vista_run/run_bq.py:287-313`) for the PFS-1yr task and
use *its* resulting `person_id` set as ground truth — this is the exact set the golden harness and
real eval both iterate. If a lightweight script can't easily reuse that loader directly, a one-off
that imports and calls it beats a hand-written SQL approximation.

```bash
# Confirm and diff against the LUMIA corpus:
gcloud storage ls gs://vista_bench/thoracic_cohort_lumia/*.xml | xargs -n1 basename | sed 's/\.xml$//'
# read-only local set-diff against the loader's person_id set -> matched / missing / coverage_pct
```

**Also verify (Codex review):**
- **Local corpus reachability**, not just the GCS listing above — confirm `retrieval.corpus_dir`
  (once set in the VM-local overlay) resolves to a real local directory containing `.xml` files
  (e.g. `ls <corpus_dir>/*.xml | wc -l`), since `_apply_ehr_adapter` reads local paths, not
  GCS URIs directly.
- **LUMIA schema reconfirmation** — pull 1-2 real `.xml` files and confirm the `<encounter>` /
  `<entry timestamp>` / `<event>` shape `parse_lumia` assumes (`src/context/adapters/ehr.py:127-136`)
  still holds; the only prior check was a 12-patient sample back in 2026-07-06. Field names/counts
  only, no data.
- **Fail-closed guard actually fires** — a tiny negative check: confirm `_apply_ehr_adapter` raises
  when `retrieval.corpus_dir` is unset/misconfigured, proving the guard added in Approach #2 works
  before relying on it for the real run.

**Expected:** a coverage_pct + exact matched/missing counts (counts only — no person_ids reported
back to the Mac), plus confirmation of local reachability, schema shape, and the fail-closed guard.
**Decision gate (Phil's threshold, default proposed) — coverage now equals cohort size under
fail-closed, not just fallback quality, so this matters more than it would otherwise:**
`coverage_pct >= 95.0` → proceed, shrinkage negligible; `80.0 <= coverage_pct < 95.0` → proceed, but
call out the exact dropped-N wherever these presets' results get reported/compared (the eval cohort
for `no_image`/`axial_all_image`/`no_report`/`timeline_only`/`report` is genuinely smaller now, not
an invisible detail); `coverage_pct < 80.0` → **STOP**, re-plan — fail-closed at that level would
silently gut the eval cohort, a bigger problem than the LUMIA wiring itself.
**Stop:** the loader can't construct (bad config, precondition failure), the corpus directory
doesn't resolve locally, the schema check finds a shape mismatch, or the fail-closed guard doesn't
fire.
**Banks forward:** the confirmed-covered `person_id` subset feeds Phase 1 (see Phase 1's own note
on why "sampling from the covered set" needs a concrete mechanism, not just a description).
**Destructive:** no — read-only BQ/GCS queries and local existence checks.

### Phase 1 — decision gate: does the `timeline` variant need `code_filter`?

**Critical fix (Codex review, verified against code): "sample from Phase 0's covered set" isn't
automatic.** `iter_captured_examples` runs `_build_prompts_for_experiment` (where fail-closed
filtering happens) *before* its stable-sort + `head(limit)` step
(`src/vista_run/context_capture.py:190,195,198`). So the **after** bank's `--limit 20` selects the
first 20 *covered* patients, but the **before** (pre-wiring) bank has no such filtering — its
`--limit 20` selects the first 20 in sort order regardless of coverage. If even one differs, the
two banks' index sets diverge and `diff_golden` hard-fails on index-set mismatch
(`src/vista_run/diff_golden.py:121-133`) — a **precondition failure, not a `code_filter` verdict**,
which the original plan text didn't distinguish.

**Fix**: before banking, construct a restricted input containing *only* Phase-0-confirmed-covered
`person_id`s (a filtered local task CSV, or a `person_id` allowlist parameter if the loader
supports one — check for an existing filter flag before building new plumbing; if none exists, a
one-off filtered CSV copy is the simplest fix). Bank both before (worktree at the pre-wiring parent
SHA, `--tag legacy_small --limit 20`) and after (`select=[]`, `--tag lumia_live_nofilter --limit
20`) against this restricted input — every row is covered by construction, so the fail-closed drop
becomes a no-op and both banks' `head(20)` draw from the identical set.

```bash
diff_golden <legacy_small> <lumia_live_nofilter> --mode strict
```

**Expected/Stop:**
- Index-set mismatch even against the restricted input → **STOP** (a distinct, unexpected
  precondition failure — something other than coverage is causing divergence; hand back, don't
  proceed to interpreting it as a `code_filter` verdict).
- Strict PASS → `timeline` needs **no** `code_filter` — done.
- Strict FAIL (with index sets matching) → re-bank with `code_filter(exclude_stanford=True)` added
  (`--tag lumia_live_filtered`), re-diff strict — whichever configuration passes byte-identical is
  the answer, encode that into `presets.py`.
- **Both** configurations pass strict → the two are byte-identical on this sample (`code_filter` is
  a no-op here); default to `select=[]` (simpler, keeps `timeline` maximally permissive) and note
  this outcome explicitly rather than picking silently.
- **Neither** passes strict → STOP (class-3 — a third variable is at play; hand back for
  re-planning, don't guess).

**Clean up the worktree after** (mirrors the Phase-2-viewer precedent,
`docs/vm-status/2026-07-15-phase2-config-context-viewer.md:93`): `git worktree remove` the
pre-wiring parent-SHA checkout once both banks are done.
**Destructive:** no — local golden-bank writes under the git-ignored results tree only.

### Phase 2 — Gate-3 allowlist fill-in

**Bank-forward correctness (Codex review)**: this is a *new run at larger N*, not literally
extending the Phase-1 file — `golden_harness` writes one output per `--tag`/`--limit` pair
(`src/vista_run/golden_harness.py:139,161`), so "extend" means reusing Phase 1's already-validated
worktree/parent-SHA/config/restricted-input setup, then re-banking at N=20–50 with new tags across
both resolved variants (`timeline` per Phase 1's outcome, `no_img_report`'s already-known chain).
`diff_golden --mode allowlist` against the re-banked legacy comparator (same restricted-input
mechanism as Phase 1, so index sets stay aligned by construction).

Inspect residual `TEXT_FIELDS` diffs and classify — **normalizer guardrails (Codex review)**: keep
this objective, not open-ended VM-time invention. Any regex codified into `diff_golden.normalize_text`
must be field-local and event-line-local, preserve line count/event order exactly, preserve
CODE/description/note-text byte-for-byte, only normalize numeric VALUE-line equivalence within a
bounded tolerance, and never be a catch-all substitution:
- **PASS / declared-delta**: (a) VALUE-line numeric formatting differs but parses to the same
  float (LUMIA has no discrete `numeric_value` attr — parsed from element text/`text_value`,
  already a known delta per OQ-K); (b) UNIT token differs only in spelling for the same concept
  (LUMIA's `unit_source_value`, ~42% field coverage per the 2026-07-06/07 check, vs legacy's
  original key).
- **STOP (genuine bug)**: any event-order or event-count drift, any `STRUCTURE_FIELDS` drift
  (always hard, mode-independent), any CODE/description/note-text content difference, or a
  residual class outside (a)/(b) — e.g. truncated note text, a dropped patient.
- **STOP (explicit, not a soft flag)**: if the declared-delta fraction gets so pervasive that a
  human spot-check stops being credible, that's a hand-back to Phil for review, not a
  auto-proceed — report the % and the field breakdown.
- **`--lenient` is for local iteration only, never the committed gate**: the tool's own flag
  description says it "can exit 0 despite text drift" (`src/vista_run/diff_golden.py:206-208`) —
  use it while tuning `normalize_text`, but the final committed VM gate must run *without*
  `--lenient` and print `RESULT: ALL GATES PASS`.

**PHI-clean readback (Codex review)**: `diff_golden` prints raw BEFORE/AFTER text previews on
failure (`src/vista_run/diff_golden.py:57,183`) — these are real timelines. Report only counts,
affected field names, and index counts back to the Mac; never paste the printed preview text into
the vm-status doc.

**Scale caveat (Codex review)**: this smoke runs N=20–50 with per-row live XML parsing (a real
change from a CSV column read). Report wall-clock time for this phase so a timing/scale caveat can
be added before running the full (non-smoke) cohort — this plan doesn't require solving that now,
just not leaving it undiscovered.
**Destructive:** no.

### Phase 3 — human visual-QA render (`context_viewer.py`) — the step Phil asked for

Task = PFS 1yr (the canonical smoke task throughout this repo's prior VM gates). Render N≈5 each —
**all three are mandatory (Codex review)**, not optional: `axial_all_image` is the only wired
preset that exercises the CT+EHR composition, so skipping it would leave the multimodal path
completely unQA'd.
- `no_image` (full timeline, `timeline` variant — post Phase 1's resolved chain)
- `no_report` or `timeline_only` (`no_img_report` variant — windowed + STANFORD-filtered)
- `axial_all_image` (EHR + CT together)

Reuse the Phase-2 viewer's config conventions (`configs/all_tasks.viewer.vm.yaml`, extended
`experiments:` list) — the *mechanism* (glob patterns, `.gitignore` rules) is already validated by
the prior plan, but each new render must still be re-checked structurally below; "already
validated" refers to the tooling, not a license to skip re-running the checks on new output.

```bash
cd src
python -m results.context_viewer --config ../configs/all_tasks.viewer.vm.yaml --type gemma3 \
  --name google/medgemma-1.5-4b-it --task progression_recurrence_free_survival_1_yr \
  --experiment no_image --limit 5          # repeat for no_report/timeline_only, axial_all_image
```

**Expected (Codex review — expanded per the 2026-07-15 viewer precedent,
`docs/vm-status/2026-07-15-phase2-config-context-viewer.md:118`; "exists/non-empty" alone is too
thin for a step whose whole point is a human reading the content)**: each HTML exists, non-empty,
self-contained (no external URLs, no local filesystem paths in `<img src>`), exactly 5
cards; every card has non-empty rendered prompt/timeline text and a non-empty token-count/bar;
`axial_all_image` cards show the expected slice/thumbnail count, `no_image`/`no_report` cards show
0 images where expected; no `STOP:` or traceback text anywhere in the HTML.
**Report back:** file paths, existence, card counts, exit codes only — **never paste rendered
content** (the HTML embeds real timelines = PHI). Phil opens the files himself on `phil-sllm-01`
(or copies them locally) to actually read the rendered text — this is the manual QA step; no
agent can do it for him, since content can't cross the PHI boundary back to the Mac.
**Stop:** missing/empty HTML, non-zero exit, the self-containment grep fails, a card is missing
text/token content, or a `STOP:`/traceback string appears in output that should have rendered.
**Destructive:** no.

## Landing & cleanup

- **Branch**: `feat/lumia-live-ehr-adapter` off `main`. Plan-time state showed a single checkout
  with no parallel sessions active — **re-verify `git status`/`git worktree list` at implementation
  start rather than trusting this stale observation (Codex review)**, since this is a shared
  checkout other sessions may have touched since.
- **Landing gate**: all 4 Verification phases green (Phase 0 coverage measured + gated, Phase 1
  decision resolved with matching index sets, Phase 2 allowlist rules filled in and reviewed
  without `--lenient`, Phase 3 HTML rendered and **Phil has actually opened and read them** — this
  last part is the whole point of this plan and isn't satisfied by an agent's structural check
  alone).
- **Merge sequence**: single branch, `/land` at the end → `main`, prune branch (no worktree to
  remove).
- **Cleanup on land**: mark this plan doc `Status: Completed`; update `docs/next.md` (currently
  lists this as a "Live follow-up," flip to landed) and `docs/plans/README.md`; note the resolved
  `timeline`-variant `code_filter` decision directly in `presets.py`'s docstring so it stops
  reading as an open question for the next person who looks at that file.
- **Operational notes for the first full (non-smoke) eval after landing** (Codex review; out of
  scope to implement now, flagged so they aren't missed): (1) `_setup_output_and_resume` resumes by
  `index` (`src/vista_run/run_bq.py`) — a stale pre-wiring result CSV could silently resume against
  a mismatched row count now that fail-closed filtering changes N; use a fresh output tag/results
  dir for these presets on the first post-landing run. (2) Any report comparing these presets
  against historical full-cohort results should carry the post-drop denominator (see the Open
  Question above on where that should live long-term).
