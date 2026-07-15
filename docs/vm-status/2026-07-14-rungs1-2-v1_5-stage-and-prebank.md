Reference: research-skills/claude_ops.md

# VM smoke — rungs 1–2 v1_5/feb26 stage + pre-bank (SUPERSEDES the 2026-07-13 prebank doc)

**Status: VM run complete — Steps 0–2 + 1b GREEN; Step 3 GREEN after an in-session CT-load fix (class-3 deviation resolved, Phil-directed); back to Mac for `/review-implementation`** (2026-07-15). See `## VM run results`.
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **uncommitted on the Mac; commit + push first, SHA set at commit time.** The only *new* code vs the superseded doc's SHA (`39b5b87`) is docs (this doc + the plan's Step-0/deviation edits).
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **unmerged; `git fetch` first** · this doc `docs/vm-status/2026-07-14-rungs1-2-v1_5-stage-and-prebank.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull` (shared / dirty checkout → `git worktree add ../vista_eval_vlm-rungs12 worktree-vlm-modular-preprocessing-roadmap`).
**Machine posture:** authored on the planner Mac (`DNa82ccae.SUNet`, no runtime/data/creds). Every step is the **weights-free golden harness** (`golden_harness.py`, never loads weights) → runs entirely on the **Claude-Code CPU** box `phil-sllm-01` (holds the `/mnt/su-vista-*` mounts + BQ creds; runs Claude Code → readback here directly). No GPU / high-throughput leg, no run-vs-readback split.
**Target machine:** Claude-Code CPU (`phil-sllm-01`) for all steps.
**Plan:** [`vlm-ct-feb26-v1_5-golden-rebaseline.md`](../plans/vlm-ct-feb26-v1_5-golden-rebaseline.md#verification--vm-handoff) ← criteria source of truth. See its **"Deviation & re-plan (2026-07-14)"** section for why this doc exists.
**Supersedes:** [`2026-07-13-rungs1-2-v1_5-feb26-prebank.md`](./2026-07-13-rungs1-2-v1_5-feb26-prebank.md) — that run **BLOCKED at Step 0** (no v1_5 timeline base staged; class-3 deviation → Mac). This doc adds the **Step 0 staging fix** and re-runs the same Steps 1/1b/2/3.
**Prior handoffs:** [`2026-07-13-rung0-0b-decoding-fix.md`](./2026-07-13-rung0-0b-decoding-fix.md) — rung-0 green (its `timeline_only` arm read the v1_1 `patient_string` CSV → 0.515; that same CSV is what Step 0 copies).

## Why this doc — what changed vs the superseded one

The superseded doc blocked because the loader (`run_bq._load_task_data`) needs a **local** timeline CSV
`base_dir/<source_csv>/<task>.csv` carrying `person_id` + `patient_string`, and only the **v1_1** base is on the mount —
the v1_5-named dir doesn't exist. **Decision (Phil, 2026-07-14):** `patient_string` is a **MEDS render, stable
v1_1→v1_5**, so **copy** the v1_1 CSV into the unsuffixed v1_5 cohort dir and continue — do **not** produce a fresh base
or chase Ryan's source CSVs. (The LUMIA-direct loader is the endgame but is a deferred follow-up — see the plan.)

Steps 1/1b/2/3 are **unchanged** from the superseded doc; only **Step 0** is new.

## Step 0 — stage the v1_5 timeline base (NEW — the deviation fix) — Claude-Code CPU `phil-sllm-01`

```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull
uv sync --extra dev    # or the repo's provisioning step if the .venv isn't present

BASE=/mnt/su-vista-uscentral1/vistabench/vlm/base           # config paths.base_dir
V11=progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1  # existing v1_1 cohort dir
V15=progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr       # unsuffixed v1_5 cohort table + dir
TASK=progression_recurrence_free_survival_1_yr

# 0a. Confirm the v1_1 base is present and the v1_5 dir is absent (baseline for the copy):
ls -la "$BASE/$V11/$TASK.csv"           # expect ~1 GB, has a patient_string column
ls -la "$BASE/$V15/" 2>&1 || echo "v1_5 dir absent (expected — about to create)"

# 0b. Copy the timeline CSV (dir only — NOT any bigquery_data_2_3 cache; v1_5 labels/UIDs query LIVE from BQ):
mkdir -p "$BASE/$V15"
cp -n "$BASE/$V11/$TASK.csv" "$BASE/$V15/$TASK.csv"
#   (copy the whole "$BASE/$V11/"*.csv if you want all tasks staged, not just PFS)
python - <<'PY'
import pandas as pd, glob, os
p = "/mnt/su-vista-uscentral1/vistabench/vlm/base/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr/progression_recurrence_free_survival_1_yr.csv"
# read only the header + a few rows (the file is ~1 GB) to confirm columns without loading it all:
head = pd.read_csv(p, nrows=5)
cols = [c for c in head.columns]
print(f"copied CSV columns include person_id={('person_id' in cols)} patient_string={('patient_string' in cols)}")
print(f"first cols: {cols[:8]}")
PY

# 0c. Flip the staged valid_tasks.json so PFS.task_source_csv is the UNSUFFIXED v1_5 table
#     (drives BOTH the BQ table `vista_bench_v1_5.<that>` AND the local dir just created).
python - <<'PY'
import json
p = "/mnt/su-vista-uscentral1/vistabench/vlm/base/tasks/valid_tasks.json"
d = json.load(open(p))
# structure: list of task entries with task_name / task_source_csv (adapt if it's a dict)
UNSUF = "progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
entries = d if isinstance(d, list) else d.get("tasks", d)
changed = 0
for e in (entries if isinstance(entries, list) else entries.values()):
    if isinstance(e, dict) and e.get("task_name") == "progression_recurrence_free_survival_1_yr":
        if e.get("task_source_csv") != UNSUF:
            print(f"  PFS task_source_csv: {e.get('task_source_csv')} -> {UNSUF}")
            e["task_source_csv"] = UNSUF; changed += 1
json.dump(d, open(p, "w"), indent=2)
print(f"valid_tasks.json PFS entries updated: {changed}")
PY
```

**0d. Build the localized v1_5 config** (git-ignored — do NOT commit). With the copy approach there is **one** base_dir
holding both `<V11>/` and the freshly-copied `<V15>/`; the `valid_tasks.json` flip makes `source_csv` unsuffixed so the
loader reads `$BASE/<V15>/`:

```yaml
# configs/all_tasks.rungs12.vm.yaml  (git-ignored — do NOT commit)
paths:
  base_dir: "/mnt/su-vista-uscentral1/vistabench/vlm/base"     # holds tasks/, <V11>/, and the copied <V15>/
  results_dir: "/mnt/su-vista-uscentral1/vistabench/vlm/results_rungs12"
  ct_snapshot_prefix: "chaudhari_lab/ct_data/ct_scans/vista/feb26"
  valid_tasks: "tasks/valid_tasks.json"
  prompts: "tasks/prompts_by_task.json"
  # ct_dir intentionally UNSET -> force GCS (no local nov25 fallback)
model:
  device: "cpu"                   # golden harness is weight-free; device must parse but is irrelevant
runtime:
  cache_dir: "/mnt/su-vista-uscentral1/vistabench/vlm/model_cache"
  batch_size: 64
  max_new_tokens: 1024
subsample: false
timeline_truncation:
  mode: "last_k_events"
  k: 100
```

**Expected:** `$BASE/$V15/$TASK.csv` exists, non-empty, has `person_id` + `patient_string` columns; `valid_tasks.json`
PFS `task_source_csv` = the unsuffixed `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`; the config parses and
`base_dir` resolves. No `bigquery_data_2_3/<V15>` cache was created (so the v1_5 BQ table is queried live).
**STOP:** the v1_1 `$TASK.csv` isn't present (nothing to copy — the base itself is missing); the copied CSV has no
`patient_string` column; `valid_tasks.json` has no PFS entry / a different structure than assumed (adapt the script,
don't blind-write). **Timeline-coverage guard** is measured on Steps 1b/2 below (`Matched N out of M`), not here.

## Step 1 — v1_5 CT resolution smoke (Phase 1) — Claude-Code CPU `phil-sllm-01`

Prove Axis A resolves a real v1_5 `(study, series)` → feb26 blob and loads it, and that the fail-closed schema guard holds.

**(a) Unit-exercise the resolver + schema guard (weight-free, no data):**
```bash
cd src && python - <<'PY'
from vqa_dataset import resolve_ct_blob, DEFAULT_CT_SNAPSHOT_PREFIX
import numpy as np
b = resolve_ct_blob("1.2.840.STUDY", "1.2.840.SERIES")
assert b == (f"{DEFAULT_CT_SNAPSHOT_PREFIX}/1.2.840.STUDY__1.2.840.SERIES.nii.gz",
             "1.2.840.STUDY__1.2.840.SERIES.nii.gz"), b
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
```

**Expected:** (a) all asserts pass, `[ok]` prints. (b) `[CT] source=gcs` count > 0 and `[CT] source=local` **== 0**
(force-GCS); at least one CT-bearing row with `image_count > 0` and `selected_indices` length **30**; no traceback; no
`nifti_path` read (the heuristic is deleted).
**STOP:** any feb26 404 (prefix/UID wrong); any `[CT] source=local` > 0 (ct_dir leaked in); `image_count == 0` across
**all** limited rows (link/query broken — suspect the v1_5 `SELECT *` isn't carrying `image_study_uid`/`image_series_uid`);
resolver unit asserts fail (contract regressed). *(Config note: use the same git-ignored `configs/all_tasks.rungs12.vm.yaml`
as the superseded doc — `base_dir=$BASE`, `ct_snapshot_prefix=…/feb26`, `valid_tasks: tasks/valid_tasks.json`,
`subsample: false`, `ct_dir` unset → force-GCS, `device: cpu`.)*

## Step 1b — reprocessing-coverage report (Phase 1; SOFT — a human decision gate before Step 3)

For the v1_5 PFS cohort, resolve each person's `(study_uid, series_uid)` via `resolve_ct_blob` and count how many map to
an **existing** feb26 blob. **Aggregate metrics only in this doc (OQ-Q4); detailed per-row / UID-level output → the
GCP-mounted `su-vista-*` bucket (git-ignored), never the git tree.**

**Report (aggregate counts only):** cohort size; count resolving to an existing feb26 blob; count with null/missing UIDs;
coverage % (= existing / cohort). **Never fails on divergence** — CT series-selection has changed since v1.1 (Phil), so a
shifted/reduced feb26 set is *expected*.
**Reference prior (OQ-Q3):** at/above ~87% (≈ rung-0's 86.9%) = healthy; materially below (~<80%) = suspect the v1_5
link/query, not series drift.
**Also report the timeline-coverage guard (NEW, from the Step-0 copy):** the loader's `Matched N out of M rows with
patient timelines` (`run_bq.py:318`). N≈M confirms the copied v1_1 CSV covers the v1_5 PFS person set; **materially N<M**
⇒ v1_5 grew persons the v1_1 CSV lacks → they silently inner-join-drop → **report the gap**.
**DECISION GATE (class-2, Phil resolves in flight):** report both coverage numbers; **Phil records accept / revise
before Step 3 banks C1.** accept → Step 3. not acceptable → STOP + hand back (class-3).

## Step 2 — re-bank the EHR `no_image` golden baseline on v1_5 (Phase 1)

```bash
cd src && python -m vista_run.golden_harness \
  --config ../configs/all_tasks.rungs12.vm.yaml \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image --tag v1_5_baseline   2>&1 | tee /tmp/rungs12_step2.log
grep "Matched .* out of .* rows with patient timelines" /tmp/rungs12_step2.log   # the timeline-coverage guard
python - <<'PY'
import json, glob
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

**Expected:** `Matched N out of M` recorded (guard); non-null `adapter_prompt_string`/`dynamic_prompt` (null_prompt_rows
== 0); `image_count == 0` all rows; `assembly_mode == "ordered"`; `meta.row_count == jsonl line count`; sorted by
`(person_id, index)`; `git status` clean. **Report cohort shape (counts):** total rows, unique `person_id`, CT-bearing
UID rows, timeline-non-null rows, task-filtered PFS count.
**STOP:** `row_count == 0`; `meta.row_count` ≠ jsonl lines; missing / zero-byte output; a dirty repo containing golden
output; the v1_5 timeline / `embed_time` surface missing or shape-shifted → report shape and hold. **Sanity vs
reference:** the v1_5 `no_image` cohort/prompts should look like the rung-0 v1_1 run (that landed timeline_only ≈ 0.515);
a wildly different row count/shape is a signal the copy or the BQ join is off.

## Step 3 — bank the axial "before" golden on feb26 (Phase 2, C1) — **only after Phil accepts Step 1b**

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
`(person_id, index)`; `image_count > 0` for CT-bearing rows; `image_hashes` length matches `image_count`; weight-free;
`git status` clean. **Record the banked SHA** — C1 is only valid while its inputs (this SHA + the v1_5 config + the
copied CSV) are frozen; 3b moves the SHA next.
**STOP:** any traceback; `row_count == 0` for a CT-bearing cohort; a missing / zero-byte `.jsonl`/`.meta.json`; index
set not unique-and-sorted; an unexpected retrieval-skip / "no data" (for `axial_all_image` a skip or zero-row is a
**STOP**); weights loading.

## Report back

Under `## VM run results`, append per-step pass/fail against each **Expected** block:
- **Step 0 (NEW):** v1_1 CSV present; copied to `<V15>/` dir; `patient_string` column confirmed; `valid_tasks.json` PFS
  `task_source_csv` = unsuffixed; no `bigquery_data_2_3/<V15>` cache created.
- **Step 1:** (a) resolver asserts pass; (b) `[CT] source=gcs` count, `source=local` count (must be 0), CT-bearing
  `image_count` + `selected_indices` length (30).
- **Step 1b:** **aggregate CT-coverage counts** (cohort size, existing-feb26-blob count, null-UID count, coverage %)
  **+ the timeline-coverage guard `Matched N out of M`** + **Phil's accept/revise decision**. Detailed per-row output
  path on the `su-vista-*` mount (not its contents).
- **Step 2:** the `Matched N/M` guard; null-prompt / image_count / assembly_mode / meta-vs-jsonl / sorted checks;
  cohort-shape counts; `git status` clean.
- **Step 3:** per (model, experiment) row_count, image_count>0, hashes-length match; **the banked SHA**; `git status` clean.
- **Net:** Phase 1+2 clean → hand back to the **Mac** to implement 3b (the C2 interlude) → Doc 2 (Phase 3, Steps 4/5).

**PHI (per the plan's Verification PHI rule):** counts / field-names / affected-indices only — **never** paste golden
rows, timelines, `.xml` contents, or resolver output containing raw DICOM Study/Series UIDs. Raw UIDs stay on the
`su-vista-*` PHI mount (Step 1b detail → the bucket, aggregate-only here). Golden output is git-ignored on the mount;
`/phi-vet` gates every commit.

## VM run results — readback on `phil-sllm-01`, 2026-07-15 · REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` @ 1b24507 (code fix) + docs this commit

All steps ran on the Claude-Code CPU box `phil-sllm-01` (weight-free golden harness). Started from doc SHA
`08bab06`; a CT-load fix (below) is committed on top — the banked C1 provenance SHA is `1b24507` (the CT-load fix commit); the readback lands in
this readback.

- **Step 0 (NEW) — stage v1_5 base:** ✅ v1_1 `PFS.csv` present (~1.0 GB); copied to unsuffixed `<V15>/` dir (byte-size
  match); copied CSV has `person_id` + `patient_string` columns; `valid_tasks.json` PFS `task_source_csv` flipped
  `…_v1_1` → unsuffixed (1 entry); git-ignored `configs/all_tasks.rungs12.vm.yaml` parses, `base_dir` resolves; **no**
  `bigquery_data_2_3/<V15>` cache created.
- **Step 1 — CT-resolution smoke:** ✅ (a) all resolver asserts pass (UID pair → feb26 blob; null/NaN/empty → None,
  fail-closed). (b) `--limit 3`: BQ v1_5 table → 9,350 rows; `source=gcs`=2, `source=local`=**0** (force-GCS held at
  smoke time); 2 CT-bearing rows `image_count`=30 with `selected_indices` length **30**; 1 null-UID row `image_count`=0;
  no traceback.
- **Step 1b — coverage + decision gate:** ✅ reported. Cohort 1,238 rows / 1,238 unique persons; **946 with CT UIDs**,
  292 null/missing. feb26 prefix = 77,713 blobs. **CT-key coverage 944/946 = 99.8%** (only 2 UIDs absent — series drift,
  not a link bug); coverage vs all cohort rows = 76.3% (depressed only by the 292 no-CT patients). **Timeline-coverage
  guard `Matched 1238 out of 1238` = 100%** (copied v1_1 CSV fully covers the v1_5 PFS person set; zero inner-join drop).
  Per-row UID detail → `results_rungs12/coverage/…` on the mount (git-ignored). **DECISION GATE (class-2): Phil ACCEPTED
  (2026-07-14)** — link/query healthy, proceed to Step 3.
- **Step 2 — re-bank `no_image` baseline:** ✅ `Matched 1238/1238` guard; 1,238 golden rows; null_prompt_rows=0;
  `image_count`=0 all rows; `assembly_mode`="ordered"; `meta.row_count`==jsonl lines (1,238); sorted by
  `(person_id, index)`; `git status` clean (golden on mount). Cohort shape: 1,238 rows / 1,238 unique persons / 946
  CT-bearing UID rows / 1,238 timeline-non-null / 1,238 PFS. Shape consistent with the rung-0 v1_1 `no_image` reference.
- **Step 3 — bank axial "before" golden (`legacy_feb26`):** ✅ both models, **1,238 rows each**, meta==jsonl, sorted by
  `(person_id, index)`, `image_hashes` length == `image_count` all rows, `selected_indices` length **30**, weight-free,
  `git status` shows no golden (mount, git-ignored):
  - `medgemma-1.5-4b-it`: 943 CT-bearing / 295 no-image.
  - `InternVL3_5-8B-hf`: 944 CT-bearing / 294 no-image.
  - 2 per-run CT-load failures (fail-closed to no-image, both caught): 1× 404 (a coverage-gap UID) + 1× degenerate
    volume ("height and width must be > 0"). The gemma/intern 943-vs-944 delta is that degenerate CT, which gemma's
    slice-processing rejected. **Banked C1 provenance SHA = `1b24507`** (the CT-load fix commit).

- **In-lane corrections (class 1):**
  - **Venv provisioning:** the doc's `uv sync --extra dev` is wrong for this repo (no `dev` extra; plain `uv sync` prunes
    the venv to the 9 base deps and drops torch/transformers). Restored via the repo's real provisioning —
    `uv pip install -e . && uv pip install -r requirements-default.txt` (per `scripts/setup.sh`). No design impact.

- **⚠️ DEVIATION (class 3) — RESOLVED IN-SESSION (Phil-directed):** Step 3 initially OOM-killed (SIGKILL/137) at the
  **same** cohort position (~row 293) across three attempts — *not* parallelism and *not* a bad file. Root cause: the
  CT cohort contains oversized volumes (6 CTs > 1 GB gzipped; max **512×512×8652**, whose `get_fdata()` float64 =
  **18.1 GB**), and the load path materialized the full float64 volume → OOM on the 15 GB box. **Fix (Phil-authorized,
  live):** read the feb26 NIfTI directly off the gcsfuse mount (`/mnt/<bucket>/<blob>`, Anywhere-Cache-accelerated;
  temp-file download only as fallback) **and** slice lazily via `img.dataobj[:, :, idx]` with `indexed_gzip` instead of
  `get_fdata()`. **Evidence:** on the 3 GB monster, peak RSS **18.1 GB → 0.16 GB**; on a `--limit 3` re-run the golden
  `image_count` + `selected_indices` + **`image_hashes` are byte-identical** to the pre-fix `get_fdata()` output, so the
  banked golden is unchanged. Files: `src/vqa_dataset.py` (load-path refactor + `[CT] source=mount` label),
  `requirements-default.txt` (`indexed_gzip==1.10.3`). This touches a shared load path (also used by real inference) —
  flagged for Mac `/review-implementation` before it rides further.

- **Net:** Steps 0–2 + 1b GREEN; Step 3 GREEN after the mount+lazy CT-load fix (C1 banked for both models). Hand back to
  the **Mac** to (1) `/review-implementation` the `vqa_dataset` load-path change, then (2) implement 3b (the C2
  interlude) → Doc 2 (Phase 3, Steps 4/5).
