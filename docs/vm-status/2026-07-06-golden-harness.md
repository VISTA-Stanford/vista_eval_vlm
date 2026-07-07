Reference: research-skills/claude_ops.md

# VM smoke — golden-harness legacy baseline + LUMIA field-coverage, v1_5 substrate

**Status: Handoff to VM** (2026-07-06)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` (authored this session, **never executed** — commit + push before pulling on the VM; SHA set at commit time)
**Machine posture:** authored on the planner Mac (`DNa82ce4d.SUNet`, no runtime/data/creds). Everything below has **never run** — execute on the GCP executor VM that holds the `/data/fries/...` vista_bench mount + BigQuery/GCS credentials for project `som-nero-plevriti-deidbdf`.
**Plans:** [`vlm-modular-preprocessing-and-context-viewer-roadmap.md`](../plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md#verification--vm-handoff) ← criteria source of truth (see also the *Golden harness — implementation* subsection)
**Prior handoffs:** none (first vm-status doc for this roadmap)

## Why this doc

Task 3b (dissolving `vqa_dataset.__getitem__`'s CT/pathology image branches + rerouting
`run_bq._build_prompts_for_experiment` through the new EHR/assembler path) must be *byte-identical*
on the imaging/structure surface. To prove that instead of asserting it, a **golden-output harness**
was authored (`src/vista_run/golden_harness.py` capture + `src/vista_run/diff_golden.py` gates +
a `selected_indices` instrumentation on `__getitem__`). None of it has run on real data.

This handoff does **two** things, both prerequisites to developing 3b:

1. **LUMIA field-coverage check** — the plan's "likeliest breaker": confirm the LUMIA corpus carries
   the flat renderer's fields so gate-3 (EHR-string byte-identity) is achievable, not a declared delta.
2. **Bank the legacy golden baseline** — run the harness against *today's* (pre-3b) code to capture the
   per-example tuple. This is the "before" snapshot 3b will be diffed against later.

**Not in scope here:** the before/after *diff* (`diff_golden.py`). There is no "after" until 3b is
implemented — that diff is a *later* handoff. This doc only banks the baseline and validates the
precondition. A clean result unblocks implementing 3b against a real golden instead of blind.

## Step 0 — get the artifacts onto the VM

```bash
cd <path-to>/vista_eval_vlm            # the checkout that tracks origin
git fetch origin
git checkout worktree-vlm-modular-preprocessing-roadmap
git pull --ff-only
uv sync --extra dev                    # or the repo's usual env sync
```

**Expected:** checkout on `worktree-vlm-modular-preprocessing-roadmap` at the just-pushed SHA (the one
`/commit-review` will stamp on the Mac); `src/vista_run/golden_harness.py` + `src/vista_run/diff_golden.py`
present; env resolves.
**STOP:** none — pure setup.

## Step 1 — LUMIA field-coverage check (likeliest breaker)

Pull one real per-patient LUMIA `.xml` from the corpus and inspect its field set. Corpus dir is the
config's `retrieval.corpus_dir` (`/data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia/`) or
`gs://vista_bench/thoracic_cohort_lumia/{person_id}.xml`.

```bash
ONE=$(ls /data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia/*.xml | head -1)
python -c "
import xml.etree.ElementTree as ET
r = ET.parse('$ONE').getroot()
attrs = {a for e in r.iter('event') for a in e.attrib}
print('event attrs:', sorted(attrs))
print('first 3 event texts:', [e.text for e in list(r.iter('event'))[:3]])
"
```

**Expected:** events expose (or the corpus otherwise carries) `code`, a `description`/`name`, lab
`numeric_value` + `unit`, and note-body `text_value`; and every eval `person_id` has an `.xml`. These
are the fields the vendored flat renderer (`get_llm_event_string`) consumes.
**STOP:**
- A field the flat renderer needs is **absent** → gate-3 EHR-string byte-identity is not achievable →
  record it; it becomes a **declared delta** (OQ-K), not a failure. (Class-2 decision gate — note which field.)
- Cohort coverage is **partial** (some eval `person_id` lack an `.xml`) → retain a DB-fetch fallback for
  the gap only; record the gap. (Class-2.)
- The design turns out to **need** a `meds_tools` / `meds_reader` / ontology import to render → the
  LUMIA-as-input premise failed → **STOP and hand back to the planner** (class-3 deviation, re-plan).

## Step 2 — golden harness one-example smoke (OQ-K: verify one first)

Capture a single example per experiment to confirm the harness produces sane, sorted output **without
loading model weights**. Canonical smoke = PFS × `gemma3 medgemma-1.5-4b`. Two experiments cover the
riskiest surfaces: `no_image` (EHR/timeline → `adapter_prompt_string`, `dynamic_prompt`) and
`axial_all_image` (CT → `selected_indices`, `image_hashes`).

```bash
cd src
python -m vista_run.golden_harness \
  --config configs/all_tasks.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image axial_all_image \
  --tag legacy --limit 1
```

**Expected:**
- Prints `>>> Wrote 1 golden rows -> .../progression_recurrence_free_survival_1_yr_no_image_legacy_golden.jsonl`
  (and the `axial_all_image` file); each `.jsonl` + `.meta.json` lands under
  `{results_dir}/golden/.../medgemma-1.5-4b-it/`.
- **No GPU allocation / no weights load** in the logs (the harness never calls `_ensure_model_loaded`).
- The `no_image` row has a non-null `adapter_prompt_string` and `dynamic_prompt`, `image_count: 0`,
  `image_hashes: []`, `selected_indices: null`, `assembly_mode: "ordered"`.
- The `axial_all_image` row (for a patient *with* a CT) has `image_count` > 0, a matching-length
  `image_hashes` list, and a `selected_indices` list of ints; a patient with no CT legitimately shows
  `image_count: 0` (CT is a nullable join — not an error).
- `git status` does **not** list the golden files (they land on the PHI mount / are git-ignored).
**STOP:**
- Any traceback, or a log line showing model weights loading → the weight-free contract broke, STOP.
- `adapter_prompt_string` is null for `no_image` (timeline should be present post-merge) → data/merge
  issue, STOP and report.
- Harness attempts a retrieval experiment / imports `meds_tools`/`meds_reader`/ontology → STOP (it
  should reject retrieval up front; if it didn't, that's a bug).

## Step 3 — bank the full legacy baseline

Drop `--limit` to capture every example, sorted by `(person_id, index)`. Add a non-gemma model for
gate-2 (per-model image identity) coverage if you want the grayscale windowing path banked too.

```bash
cd src
# gemma windowing (multi_window_rgb)
python -m vista_run.golden_harness \
  --config configs/all_tasks.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image axial_all_image \
  --tag legacy

# OPTIONAL gate-2 coverage — non-gemma (grayscale) windowing
python -m vista_run.golden_harness \
  --config configs/all_tasks.yaml \
  --type intern --name OpenGVLab/InternVL3_5-8B-hf \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image axial_all_image \
  --tag legacy
```

**Expected:** one `_legacy_golden.jsonl` + `.meta.json` per (experiment, model) under
`{results_dir}/golden/.../`; each non-empty; `.meta.json` `row_count` matches the `.jsonl` line count and
equals the loaded cohort size for that experiment; rows sorted by `(person_id, index)`. Files not listed
by `git status` (PHI).
**STOP:** any traceback; a `row_count` of 0 for an experiment the cohort should populate; weights loading.

**Class-2 decision gate — pathology surface:** if you also want to bank `path_full` (tiles + path_note +
timeline), add `--experiments path_full`. **But** PFS may not carry pathology tiles — if
`_load_path_full_task_data` returns `None`/empty (logged "No data loaded"), that experiment simply has no
golden for this task; skip it and note which. Do **not** treat a legitimately-empty pathology cohort as a
failure.

## Report back

In `## VM run results` below, per step: pass/fail vs the Expected block; for Step 1 the LUMIA field set
(field names only — **no** event text / patient data) and whether all four renderer fields are present +
cohort coverage; for Steps 2–3 the per-(experiment, model) `row_count`s and confirmation `git status`
stayed clean. Record any class-2 gate outcomes (missing LUMIA field → declared-delta; empty pathology →
skipped). **PHI: counts / field-names / pass-fail only — never paste golden rows, timelines, or `.xml`
contents.** A clean result = the legacy baseline is banked and 3b can be developed against it.

## VM run results
_(left empty by the planner; the executor fills this in readback mode)_
