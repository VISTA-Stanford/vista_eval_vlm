Reference: research-skills/claude_ops.md

# VM smoke — rungs 1–2 v1_5/feb26 pre-bank (Axis A/B: CT UID resolution + substrate cut, pre-3b)

**Status: Handoff to VM** (2026-07-13)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **uncommitted on the Mac; commit + push first, SHA set at commit time.** Everything below has **never executed.**
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **unmerged; `git fetch` first** · this doc `docs/vm-status/2026-07-13-rungs1-2-v1_5-feb26-prebank.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull` (shared / dirty checkout → `git worktree add ../vista_eval_vlm-rungs12 worktree-vlm-modular-preprocessing-roadmap`).
**Machine posture:** authored on the planner Mac (`DNa82cedc.SUNet`, no runtime/data/creds). Every step is the **weights-free golden harness** (`golden_harness.py`, never loads weights) → runs entirely on the **Claude-Code CPU** box `phil-sllm-01` (holds the `/mnt/su-vista-*` mounts + BQ creds; runs Claude Code → readback here directly). No GPU / high-throughput leg, no run-vs-readback split.
**Target machine:** Claude-Code CPU (`phil-sllm-01`) for all steps.
**Plans:** [`vlm-ct-feb26-v1_5-golden-rebaseline.md`](../plans/vlm-ct-feb26-v1_5-golden-rebaseline.md#verification--vm-handoff) ← criteria source of truth (Reviewed: Yes; Codex `/review-implementation` clean 2026-07-13, `reviews/vlm-ct-feb26-v1_5-golden-rebaseline-implementation-feedback.md`).
**Prior handoffs:** [`2026-07-13-rung0-0b-decoding-fix.md`](./2026-07-13-rung0-0b-decoding-fix.md) — rung-0 green (constrained decoding both arms, `predicted_label==-1 total: 0`); it introduced the `ct_snapshot_prefix` seam Axis A builds on. **This is a continuation, NOT a supersede** (OQ-Q5).

## Why this doc

Rungs 1–2 **Axis A** (CT resolution repointed off the deleted `nov25`/`nifti_path` heuristic onto the v1_5
`(image_study_uid, image_series_uid)` → feb26 blob via a shared `resolve_ct_blob` helper) + **Axis B** (substrate cut
`vista_bench_v1_1` → `vista_bench_v1_5`: dataset constant flipped, config cut to `valid_tasks.json` / `subsample:false`
/ `ct_snapshot_prefix: feb26`) have never run on real v1_5 data. This doc is **Doc 1 of 2** (OQ-Q5): it proves Axis A/B
read v1_5/feb26 (Steps 1, 1b), re-banks the EHR `no_image` golden baseline on the v1_5 cohort (Step 2), and banks the
axial **"before"** golden on feb26 (Step 3, C1). Then the Mac implements 3b (the CT-adapter dissolution — the Mac
interlude), and **Doc 2** (Phase 3) banks the "after" + diffs. A clean run here unblocks that 3b interlude.

**Scope:** Task 3b / Axis C is **NOT** in this branch by design (banking the byte-identity "before" must happen on the
un-refactored `__getitem__` slice-selection path). Do not expect a CT-adapter dissolution here.

## Step 0 — get the artifacts onto the VM + build the v1_5 config

```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull
uv sync --extra dev    # or the repo's provisioning step if the .venv isn't present

# Confirm the Axis A/B seams are present at the pushed SHA:
grep -n "def resolve_ct_blob" src/vqa_dataset.py
grep -n "image_study_uid\|image_series_uid" src/vqa_dataset.py | head
grep -n 'VISTA_BENCH_DATASET = "vista_bench_v1_5"' src/data_tools/utils/query_utils.py
grep -n "ct_snapshot_prefix\|valid_tasks: \"tasks/valid_tasks.json\"\|subsample: false" configs/all_tasks.yaml
```

**Build the localized v1_5 config** (git-ignored — do NOT commit; mirrors the rung-0 `.vm.yaml` pattern). Copy the
committed `configs/all_tasks.yaml` (which already carries `ct_snapshot_prefix: …/feb26`, `valid_tasks:
tasks/valid_tasks.json`, `subsample: false`, `ct_dir` unset → force-GCS) and override only the machine-local paths to
the **v1_5-staged** base on the mount:

```yaml
# configs/all_tasks.rungs12.vm.yaml  (git-ignored)
paths:
  base_dir: "<V15_BASE>"          # the v1_5-staged base: MUST contain tasks/valid_tasks.json,
                                  # tasks/prompts_by_task.json, and <PFS_SOURCE_CSV>/<PFS_TASK>.csv timelines.
                                  # (NOT the rung-0 v1_1 base — that carries v1_1 timelines.)
  results_dir: "/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12"   # golden -> results_dir/golden/ (PHI mount)
  ct_snapshot_prefix: "chaudhari_lab/ct_data/ct_scans/vista/feb26"
  valid_tasks: "tasks/valid_tasks.json"
  prompts: "tasks/prompts_by_task.json"
  # ct_dir intentionally UNSET -> force GCS (no local nov25 fallback)
model:
  device: "cpu"                   # golden harness is weight-free; device is irrelevant but must parse
runtime:
  cache_dir: "/mnt/su-vista-uscentral1/vistabench/vlm/model_cache"
  batch_size: 64
  max_new_tokens: 1024
subsample: false
timeline_truncation:
  mode: "last_k_events"
  k: 100
```

**Expected:** clean checkout at the pushed SHA; all four greps hit (`resolve_ct_blob`, the UID reads, the v1_5 constant,
the config keys); the v1_5 config parses and its `base_dir` resolves on the mount.
**STOP (precondition):** the v1_5 base is **not** staged — `<V15_BASE>/tasks/valid_tasks.json` or
`<V15_BASE>/<PFS_SOURCE_CSV>/<PFS_TASK>.csv` (PFS_TASK = `progression_recurrence_free_survival_1_yr`; PFS_SOURCE_CSV =
the v1_5 cohort table `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`) is missing → stage the v1_5 base first
(class-2 precondition), do not proceed. Any grep empty → the Mac hasn't pushed the branch (or wrong SHA); STOP.

## Step 1 — v1_5 CT resolution smoke (Phase 1) — Claude-Code CPU `phil-sllm-01`

Prove Axis A resolves a real v1_5 `(study, series)` → feb26 blob and loads it, and that the fail-closed schema guard
holds. Two parts:

**(a) Unit-exercise the resolver + schema guard (weight-free, no data):**
```bash
cd src && python - <<'PY'
from vqa_dataset import resolve_ct_blob, DEFAULT_CT_SNAPSHOT_PREFIX
import numpy as np
# CT-available row -> feb26 blob
b = resolve_ct_blob("1.2.840.STUDY", "1.2.840.SERIES")
assert b == (f"{DEFAULT_CT_SNAPSHOT_PREFIX}/1.2.840.STUDY__1.2.840.SERIES.nii.gz",
             "1.2.840.STUDY__1.2.840.SERIES.nii.gz"), b
# Missing / null UIDs -> None (fail closed to no-image; never reaches nifti_path)
assert resolve_ct_blob(None, "1.2.840.SERIES") is None
assert resolve_ct_blob("1.2.840.STUDY", None) is None
assert resolve_ct_blob(np.nan, np.nan) is None
assert resolve_ct_blob("", "  ") is None
print("[ok] resolve_ct_blob: UID pair -> feb26 blob; null/NaN/empty -> None (fail-closed)")
PY
```

**(b) Small live axial capture (a few CT rows) — resolution + load actually work on feb26:**
```bash
cd src && python -m vista_run.golden_harness \
  --config ../configs/all_tasks.rungs12.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments axial_all_image --tag smoke_feb26 --limit 3   2>&1 | tee /tmp/rungs12_step1.log
grep -c "\[CT\] source=gcs" /tmp/rungs12_step1.log
grep -c "\[CT\] source=local" /tmp/rungs12_step1.log
# inspect the captured rows (image_count / selected_indices) — PHI-safe fields only:
python - <<'PY'
import json, glob, os
p = sorted(glob.glob(os.path.expanduser("/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden/**/*smoke_feb26*_golden.jsonl"), recursive=True))[-1]
rows = [json.loads(l) for l in open(p)]
ct = [r for r in rows if (r.get("image_count") or 0) > 0]
print(f"rows={len(rows)}  ct_bearing={len(ct)}  image_counts={[r['image_count'] for r in ct]}")
print(f"selected_indices_len={[len(r['selected_indices']) for r in ct]}  (expect 30 each)")
PY
```

**Expected:** (a) all asserts pass, `[ok]` prints. (b) `[CT] source=gcs` count > 0 and `[CT] source=local` **== 0**
(force-GCS; the blob is built from the `(study, series)` path); at least one CT-bearing row with `image_count > 0` and
`selected_indices` length **30** (evenly-spaced); no traceback; no `nifti_path` read (there is no nov25 fall-through
possible — the heuristic is deleted).
**STOP:** any feb26 404 (prefix/UID wrong — check `ct_snapshot_prefix` + the v1_5 UID columns); any `[CT] source=local`
> 0 (ct_dir leaked in); `image_count == 0` across **all** limited rows (link/query broken → suspect the v1_5 `SELECT *`
isn't carrying `image_study_uid`/`image_series_uid`); the resolver unit asserts fail (contract regressed).

## Step 1b — reprocessing-coverage report (Phase 1; SOFT — a human decision gate before Step 3)

For the v1_5 PFS cohort, resolve each person's `(study_uid, series_uid)` via `resolve_ct_blob` and count how many map to
an **existing** feb26 blob. **Aggregate metrics only go in this doc (OQ-Q4); the detailed per-row / any UID-level output
is written to the GCP-mounted `su-vista-*` bucket (git-ignored), never the git tree.** Sketch (executor adapts the BQ /
GCS calls to the box's helpers):

```bash
cd src && python - <<'PY'
# Query the v1_5 cohort for PFS person_ids + (image_study_uid, image_series_uid); resolve via resolve_ct_blob;
# check blob existence in gs://su-vista-uscentral1/<feb26>/. Write AGGREGATE counts to stdout; write the DETAILED
# per-person (person_id, resolved-or-null, exists-or-not) table to a parquet/csv ON THE su-vista-* MOUNT (PHI).
# Aggregate to report here: cohort_size; n_resolved_to_existing_feb26_blob; n_null_or_missing_uid;
# (if derivable) n_series_differs_from_v1_1_selection. NO raw Study/Series UIDs in this doc.
PY
```

**Expected / report (aggregate counts only):** cohort size; count resolving to an existing feb26 blob; count with
null/missing UIDs; coverage % (= existing / cohort). **This never fails on divergence** — CT series-selection has
changed since v1.1 (Phil), so a shifted/reduced feb26 set is *expected*.
**Reference prior (OQ-Q3):** at/above **~87%** (≈ rung-0's 86.9%) = healthy (the native v1_5 UID link is at least as
good as rung-0's crude nov25-key reconstruction); **materially below (~<80%)** = suspect the v1_5 link/query, not
series drift. Link-health check, not an eval bar (the golden gate only compares CT rows present on *both* feb26 sides).
**DECISION GATE (class-2, Phil resolves in flight on the VM):** the executor **reports the coverage number**; **Phil
records an accept / revise decision before Step 3 banks C1.** accept → proceed to Step 3. not acceptable (cohort not
representative) → **STOP + hand back** (class-3). null/missing UIDs beyond the shape Phil agrees to → class-3 deviation.

## Step 2 — re-bank the EHR `no_image` golden baseline on v1_5 (Phase 1)

The current green `no_image` baseline is v1_1 (a different cohort). Re-bank on v1_5 — this is the new gate-3 "before".
```bash
cd src && python -m vista_run.golden_harness \
  --config ../configs/all_tasks.rungs12.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image --tag v1_5_baseline
# inspect meta + rows (PHI-safe fields):
python - <<'PY'
import json, glob, os
j = sorted(glob.glob("/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12/golden/**/*v1_5_baseline*_golden.jsonl", recursive=True))[-1]
m = j.replace("_golden.jsonl", "_golden.meta.json")
rows = [json.loads(l) for l in open(j)]; meta = json.load(open(m))
null_prompt = sum(1 for r in rows if not r.get("adapter_prompt_string") or not r.get("dynamic_prompt"))
img = sum(1 for r in rows if (r.get("image_count") or 0) != 0)
modes = set(r.get("assembly_mode") for r in rows)
srt = rows == sorted(rows, key=lambda r: (str(r.get("person_id")), r.get("index")))
print(f"row_count(meta)={meta['row_count']} jsonl_lines={len(rows)} sorted={srt}")
print(f"null_prompt_rows={null_prompt} nonzero_image_rows={img} assembly_modes={modes}")
PY
git status --short   # golden must NOT appear (lives on the PHI mount, git-ignored)
```

**Expected:** non-null `adapter_prompt_string` / `dynamic_prompt` (null_prompt_rows == 0); `image_count == 0` for all
rows; `assembly_mode == "ordered"`; `meta.row_count == jsonl line count`; rows sorted by `(person_id, index)`; `git
status` clean (golden stayed on the PHI mount). **Report cohort shape (counts):** total rows, unique `person_id`,
CT-bearing UID rows, timeline-non-null rows, task-filtered PFS count.
**STOP:** `row_count == 0`; `meta.row_count` ≠ jsonl lines; missing / zero-byte output; a dirty repo containing golden
output (leaked off the PHI mount); the v1_5 timeline / `embed_time` surface missing or shape-shifted (OQ-P: queryability
is release-note-confirmed, but *shape* is VM-checked here) → report shape (counts only) and hold.

## Step 3 — bank the axial "before" golden on feb26 (Phase 2, C1) — **only after Phil accepts Step 1b**

Two explicit harness invocations, both `--experiments axial_all_image --tag legacy_feb26` — gemma3 (multi-window RGB)
and intern (grayscale, the Gate-2 path):
```bash
cd src
python -m vista_run.golden_harness --config ../configs/all_tasks.rungs12.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr --experiments axial_all_image --tag legacy_feb26
python -m vista_run.golden_harness --config ../configs/all_tasks.rungs12.vm.yaml \
  --type intern  --name OpenGVLab/InternVL3_5-8B-hf \
  --task progression_recurrence_free_survival_1_yr --experiments axial_all_image --tag legacy_feb26
```

**Expected:** per (experiment, model) a `…_legacy_feb26_golden.jsonl` + `.meta.json`, non-empty, sorted by
`(person_id, index)`; `image_count > 0` for CT-bearing rows; `image_hashes` length matches `image_count`; weight-free
(no GPU / weights in logs); `git status` clean (PHI mount only). **Record the banked SHA** — C1 is only valid while its
inputs (this SHA + the v1_5 config) are frozen; 3b moves the SHA next, so Doc 2 banks the "after" and this "before"
must not be re-run under a changed input (un-bank + rerun if it is).
**STOP:** any traceback; `row_count == 0` for a CT-bearing cohort; a missing / zero-byte `.jsonl`/`.meta.json`; index
set not unique-and-sorted; an unexpected retrieval-skip / "no data" message (for `axial_all_image` a skip or zero-row
output is a **STOP**, not benign); weights loading.

## Report back

Under `## VM run results`, append per-step pass/fail against each **Expected** block:
- **Step 0:** checkout SHA; seams-grep hits; v1_5 config `base_dir` + staging confirmed.
- **Step 1:** (a) resolver asserts pass; (b) `[CT] source=gcs` count, `source=local` count (must be 0), CT-bearing
  `image_count` + `selected_indices` length (30).
- **Step 1b:** **aggregate coverage counts only** (cohort size, existing-feb26-blob count, null-UID count, coverage %) +
  **Phil's accept/revise decision**. Detailed per-row output path on the `su-vista-*` mount (not its contents).
- **Step 2:** null-prompt / image_count / assembly_mode / meta-vs-jsonl / sorted checks; cohort-shape counts; `git status` clean.
- **Step 3:** per (model, experiment) row_count, image_count>0 confirmation, hashes-length match; **the banked SHA**; `git status` clean.
- **Net:** Phase 1+2 clean → hand back to the **Mac** to implement 3b (the C2 interlude) → Doc 2 (Phase 3, Steps 4/5).

**PHI (per the plan's Verification PHI rule):** counts / field-names / affected-indices only — **never** paste golden
rows, timelines, `.xml` contents, or resolver output containing raw DICOM Study/Series UIDs. Raw UIDs are identifiers →
they stay on the `su-vista-*` PHI mount (Step 1b detail goes to the bucket, aggregate-only here). Golden output is
git-ignored on the mount; `/phi-vet` gates every commit.

## VM run results — readback on `phil-sllm-01`, 2026-07-14 · REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` @ `39b5b87` (pushed to `origin/<branch>`)

**Net: BLOCKED at Step 0 precondition — the v1_5 base is not staged on the VM.** The Axis A/B
*code* seams verify clean and the resolver contract passes weight-free, but every data-touching
step (1b, 2, 3) needs a v1_5-staged base that does not exist on the mount. This contradicts the
plan's assumption that "the v1_5 base is already staged by the VM" → **class-3 deviation, handing
back to the Mac planner.**

- **Step 0 — seams grep:** ✅ all four hit at `39b5b87`:
  `resolve_ct_blob` (`vqa_dataset.py:21`), UID reads `image_study_uid`/`image_series_uid`
  (`vqa_dataset.py:170-171`), `VISTA_BENCH_DATASET = "vista_bench_v1_5"`
  (`query_utils.py:257`), config keys `ct_snapshot_prefix: …/feb26` + `valid_tasks:
  tasks/valid_tasks.json` + `subsample: false` (`all_tasks.yaml`).
- **Step 0 — v1_5 config / base_dir staging:** ❌ **STOP (precondition fired).** The v1_5 base
  is **not** staged. Under `/mnt/su-vista-uscentral1/vistabench/vlm/base/` only the **v1_1**
  cohort dir exists (`progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1/…1_yr.csv`,
  ~1 GB) plus `tasks/`. No `…_1yr_2yr_3yr_4yr_5yr` (unsuffixed / v1_5) cohort dir anywhere on
  `su-vista-uscentral1` or `su-vista-hot`. The staged `tasks/valid_tasks.json` also maps PFS →
  `task_source_csv: …_v1_1` (the v1_1 table), not a v1_5 source. Config **not** written (would
  point at a non-existent base). ⇒ per the Step 0 STOP, do not proceed.
- **Step 1(a) — resolver + fail-closed schema guard (weight-free, no data):** ✅ PASS. UID pair
  → `chaudhari_lab/ct_data/ct_scans/vista/feb26/{study}__{series}.nii.gz`; `None`/NaN/empty →
  `None` (fail-closed, never reaches `nifti_path`). `DEFAULT_CT_SNAPSHOT_PREFIX` = feb26. Axis A
  contract holds.
- **Step 1(b) — live axial capture:** ⛔ NOT RUN — blocked by the missing v1_5 base (loader needs
  `base_path/<v1_5 source_csv>/<task>.csv` timelines to merge on `person_id`).
- **Step 1b — coverage report (soft gate):** ⛔ NOT RUN — same block; no coverage number to gate on.
- **Step 2 — re-bank `no_image` on v1_5:** ⛔ NOT RUN — same block.
- **Step 3 — bank axial "before" on feb26:** ⛔ NOT RUN — gated behind Step 1b + same base block.

**Root cause (data-flow confirmed by reading the loader):** `golden_harness` → `run_bq.TaskOrchestrator._load_task_data`
loads labels/UIDs from BQ table `{project}.vista_bench_v1_5.<source_csv>` (**exists** — the v1_5
cohort table `…_1yr_2yr_3yr_4yr_5yr` is present, 32 cols incl. `person_id`, `embed_time`,
`image_study_uid`, `image_series_uid`, deprecated `nifti_path`; **no timeline column**), then
**requires** a local timeline CSV `base_path/<source_csv>/<task>.csv` carrying `patient_string`
(the per-patient EHR timeline), merged on `person_id` — **missing local CSV ⇒ task returns
`None`.** The BQ side is v1_5-ready; the **local v1_5 timeline base is the gap.**

- **⚠️ DEVIATION (class 3):** **expected** — plan §"Axis B" states "the executor keeps a localized
  `configs/all_tasks.vm.yaml` pointing at the `su-vista-*` mounts (already staged by the VM)" and
  the doc's Step 0 lists `<V15_BASE>` as a precondition. **found** — no v1_5 timeline base is
  staged (only v1_1). **why it blocks** — re-using the v1_1 timeline CSV would inner-join-drop
  every v1_5-only person (v1_5 is a larger/different cohort: 374k rows / 249.6k with CT), silently
  shrinking + corrupting the "re-bank on v1_5" baseline the plan explicitly wants. Staging the
  v1_5 base is **not** a mechanical copy: it requires materializing a ~1 GB v1_5 EHR-timeline CSV
  via the OMOP/MEDS pipeline (`full_test_ehr_vb_download.py` — needs a `meds_reader` DB + ontology
  lookup + the `bigquery_data_2_3` cache, none verified present on this VM) **plus** a v1_5
  `valid_tasks.json` whose `task_source_csv` points at the unsuffixed v1_5 cohort. These carry
  design decisions (subsample policy, split assignment, dep availability) that are the planner's,
  not in-lane. **→ escalating to the Mac planner.**

**Mac decisions needed before a Doc-2 / superseding handoff:**
1. How is the v1_5 timeline base materialized — is `full_test_ehr_vb_download.py` (or the
   subsample path) the intended producer, and are its MEDS DB + ontology + `bigquery_data_2_3`
   inputs available to this VM (or must they be staged first)?
2. Subsample policy for the v1_5 base (the doc's config sets `subsample: false`, but the staged
   v1_1 base carries only the full `{task}.csv` — confirm the v1_5 producer emits the same shape).
3. The v1_5 `valid_tasks.json` `task_source_csv` value (unsuffixed `…_1yr_2yr_3yr_4yr_5yr` to match
   the existing v1_5 BQ table) and where it gets staged.

**PHI:** counts / field-names / dir-names only above; no raw Study/Series UIDs, no timeline rows,
no patient content. No golden output was produced (all data steps blocked).
