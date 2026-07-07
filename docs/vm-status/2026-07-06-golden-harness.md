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

**Status: BLOCKED — class-3 deviation (data substrate not present on this VM).** Attempted on executor
`phil-sllm-01`, 2026-07-07. Nothing in Steps 1–3 executed; the harness cannot construct.

### Setup (Step 0)
- Branch `worktree-vlm-modular-preprocessing-roadmap` @ `6a9c134` checked out; both harness files present.
- `uv sync --extra dev` N/A — the repo defines **no** optional-dependencies/extras. Used the repo's real
  setup (`scripts/setup.sh`): default `.venv` built OK — Python 3.11.15, torch 2.8.0+cu128,
  transformers 4.57.1, vllm 0.11.0. (The optional `llava` env failed on a full root disk; not needed here.)
- BigQuery/GCS creds present and correct: `gcloud` project `som-nero-plevriti-deidbdf`, authed
  `padamson@stanford.edu`.
- Root disk was at 100% (723 MB free); freed ~26 GiB via `uv cache clean` → 3.2 GB free.

### DEVIATION — the `/data/fries/...` mount this handoff assumes does not exist on `phil-sllm-01`
This VM's data mounts are `/mnt/su-vista-hot` and `/mnt/su-vista-uscentral1`; there is no `/data/fries`.
Every `paths:` root in `configs/all_tasks.yaml` points at `/data/fries/users/rdcunha/...` — a different
contributor's machine layout — so none of the required data resolves here:

| config key | config value | on this VM |
| --- | --- | --- |
| `base_dir` | `/data/fries/users/rdcunha/vista_bench_cached/vista_bench` | **absent** (nearest local `…/vista/vista_bench_thoracic_r1_aug2025` holds only `meds/ omop/`, no `tasks/valid_tasks_v1_3.json`) |
| `results_dir` | `/data/fries/users/rdcunha/vb_results/path_test` | **absent** |
| `ct_dir` | `/data/fries/.../chaudhari_lab/ct_data/ct_scans/vista/nov25` | CT root exists at `/mnt/su-vista-*/chaudhari_lab/ct_data/ct_scans`, not this exact subpath |
| `path_tile_base` | `/data/fries/.../download_path` | **absent** (pathology at `/mnt/su-vista-uscentral1/vistabench/pathology_priority`) |
| `retrieval.corpus_dir` (LUMIA) | `/data/fries/.../thoracic_cohort_lumia` | only **tarballs** at `…/vista/vista_aug2025_lumia_xml_sample/*.tar.gz`, not extracted `.xml` |

`TaskOrchestrator.__init__` loads `base_dir/tasks/valid_tasks_v1_3.json` at construction
(`src/vista_run/run_bq.py:67`), so the harness `FileNotFoundError`s before Step 1's LUMIA check begins.

### Step 1 (LUMIA field-coverage) — NOT RUN
Precondition failed: the corpus is not present as extracted `.xml` on this VM (tarballs only), and the
config corpus_dir doesn't exist. No field-set inspected.

### Steps 2–3 (golden smoke + baseline bank) — NOT RUN
Blocked by the same missing `base_dir`/`ct_dir`/`path_tile_base`/`results_dir`.

### Resolution (agreed 2026-07-07)
Source data isn't on the VM yet. **Backlogged** until it's copied over; then the config must be localized
to this VM's `/mnt/su-vista-*` mounts and the handoff re-issued. No config remap was performed here
(remapping 5+ data roots + confirming the vista_bench cache shape + extracting LUMIA is planner work, not
an in-lane executor correction). Planning continues on the Mac. PHI: no patient data touched — paths /
package versions / status only.

---

## VM run results — RESUMED 2026-07-07 (data located in GCS)

**Status: HANDBACK → Mac planner (2026-07-07).** Step 1 done (class-2 declared delta); `no_image` legacy
baseline banked + green; **`axial_all_image` is a class-3 deviation** — the `nov25` CT snapshot is gone,
go-forward is the v1_5/feb26 dataset link (see the *DEVIATION → Mac planner* block at the end).
The substrate BLOCKED above turns out to live in the GCS bucket `gs://vista_bench/` (project
`som-nero-plevriti-deidbdf`), not just on the `/data/fries` layout. Relevant roots located:

| config root | GCS location | size |
| --- | --- | --- |
| `retrieval.corpus_dir` (LUMIA) | `gs://vista_bench/thoracic_cohort_lumia/*.xml` (9,350 files) | ~15 GB |
| `base_dir/tasks` | `gs://vista_bench/tasks/` (`valid_tasks.json`, `prompts_by_task.json`) | <1 MB |
| `base_dir` cohort cache | `gs://vista_bench/bigquery_v1_3/` (per-task dirs) | ~0.9 GB |
| (reference outputs) | `gs://vista_bench/vlm_result/vista_vlm_constrained_all_model_response_02142026.csv` | ~13 MB |
| `ct_dir` | on-mount `/mnt/su-vista-*/chaudhari_lab/ct_data/ct_scans` (subpath differs from config) | — |
| `path_tile_base` | on-mount `/mnt/su-vista-uscentral1/vistabench/pathology_priority` | — |

Note: GCS `tasks/` has `valid_tasks.json` (not `valid_tasks_v1_3.json`) and **no** `image_valid_tasks.json`
— the config's `base_dir/tasks` layout is not a 1:1 match; base_dir reconstruction still needs a decision.

### Step 1 — LUMIA field-coverage check — DONE (class-2 declared delta)
Pulled a 12-patient sample from `gs://vista_bench/thoracic_cohort_lumia/` and ran the real instrumentation
(`context.adapters.ehr.parse_lumia` + `lumia_missing_fields`) under the repo `.venv`. Import resolves
without data mounts. XML shape: `<eventstream>`→`<encounter>`→`<entry timestamp>`→`<event ...>`; the
timestamp lives on `<entry>`, not `<event>`. Pooled sample = 45,141 events. **Renderer-field coverage
(non-null %, values not shown):**

| renderer field | source in LUMIA | coverage | verdict |
| --- | --- | --- | --- |
| `time` | `<entry timestamp>` | 100% | ✓ present |
| `code` | `<event code>` | 100% | ✓ present |
| `description` | `<event name>` (fallback; no `description` attr) | 93.4% | ✓ present |
| `text_value` | `<event>` element text | 83.4% | ✓ present |
| `numeric_value` | **no attr; not carried discretely** — lab value is baked into element text | **0%** | ✗ **absent** |
| `unit` | corpus has `unit_source_value` (42% of events); adapter reads `attrib.get("unit")` | **0%** | ✗ **field-name mismatch** (data present, wrong key) |

Filter fields `type` and `speciality` are also 0% (no `type` attr; no `<provider speciality>` in sample) —
those affect EHR *filters*, not the renderer's byte-identity surface.

**Class-2 outcome (per Step 1 STOP): declared delta OQ-K.**
- `unit` — the data **is** in the corpus under `unit_source_value`; recoverable by a one-line adapter
  remap (`attrib.get("unit_source_value")`). Mechanical; corpus is not missing the field.
- `numeric_value` — **genuinely not a discrete field** in LUMIA (no `numeric_value`/`value_*` attr; the
  number is embedded in `<event>` element text → currently captured as `text_value`). The legacy renderer
  emitted `VALUE: {numeric_value}{unit}`; that exact line cannot be reconstructed from LUMIA discrete
  fields → **gate-3 EHR-string byte-identity on lab VALUE lines is at risk** unless (a) LUMIA's element
  text already equals the legacy text_value branch output (resolvable only by the later golden diff), or
  (b) the adapter parses the number out of element text.
- **LUMIA-as-input premise HOLDS** — rendering needs **no** `meds_tools`/`meds_reader`/ontology import
  (all data is in the `.xml`). So this is **class-2 (declared delta), not class-3** (no re-plan of the
  premise required). Open design fork for the Mac: accept the declared delta vs. parse numeric from text.

### base_dir reconciliation (Q2) — RESOLVED: bucket root **is** the base_dir
`gs://vista_bench/` maps 1:1 onto the code's expected `base_dir` layout:
- `tasks/valid_tasks.json` + `tasks/prompts_by_task.json` present (config `valid_tasks` repointed to
  `tasks/valid_tasks.json`; the bucket has **no** `_v1_3` variant and no `image_valid_tasks.json` — the
  latter is optional/guarded in `TaskOrchestrator.__init__`).
- `task_source_csv` for both PFS + has_recurrence = `progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1`;
  timeline CSVs live at `gs://vista_bench/<source_csv>/<task>.csv` — but **only base `.csv`, no
  `_subsampled.csv`** → config `subsample` flipped to **false** (`resolve_timeline_csv_filename` →
  `<task>.csv`). PFS timeline CSV = 961 MB.
- Cohort rows: local cache dir is `bigquery_data_2_3/<source_csv>/` (empty in bucket) → falls back to
  **BigQuery** (`vista_bench_v1_1.<source_csv>`), which works on this VM.
- Path tasks need `base_dir/v1_3/<source_csv>/…_path_subsampled.csv`; no `v1_3/` prefix in the bucket →
  `path_full` skipped for now (handoff explicitly allows an empty pathology cohort).

Staged onto `/mnt/su-vista-uscentral1/vistabench/vlm/` (root disk is only 3 GB free; mounts are 1 PB):
`base/tasks/*.json` + `base/<source_csv>/<pfs_task>.csv`, with `results/` + `model_cache/` alongside.
VM config: **`configs/all_tasks.vm.yaml`** (localized copy; the committed `all_tasks.yaml` keeps the
`/data/fries` layout — do not commit the `.vm` copy as default).

### Step 1 follow-up — `unit` remap applied (in-lane)
`src/context/adapters/ehr.py:_lumia_event_to_row` now reads `attrib.get("unit") or
attrib.get("unit_source_value")` → unit coverage 0% → **42.3%** on the sample. `numeric_value` stays the
declared delta (a code comment records why). Mechanical fix; corpus had the data under the OMOP source key.

### Steps 2–3 (no_image) — PASS on real data (first-ever harness run)
Ran `golden_harness … --experiments no_image` for PFS × `medgemma-1.5-4b-it` under the repo `.venv`
(weight-free confirmed: vLLM logged `libcuda.so.1: cannot open` — no GPU/weights touched).
BQ returned 9,350 cohort rows; local CSV 1,238 timelines; **matched 1,238/1,238**.

- **`--limit 1` smoke:** matches Expected exactly — `assembly_mode="ordered"`, `image_count=0`,
  `image_hashes=[]`, `selected_indices=null`, non-null `adapter_prompt_string` (24,563 chars) +
  `dynamic_prompt` (25,664 chars).
- **Full bank:** `row_count 1238 == jsonl lines 1238`; rows sorted by `(person_id, index)`; **0** null
  `adapter_prompt_string`; all `image_count==0` / `assembly_mode=="ordered"`. Golden files land under the
  `/mnt/su-vista-uscentral1` results_dir (git status clean of golden — PHI stays off the repo tree).

**no_image legacy baseline is banked.** 3b's EHR-passthrough surface can now be diffed against it.

### DEVIATION → Mac planner (2026-07-07): `axial_all_image` CT surface — snapshot gone + design coupling

**Class-3 (plan-level): the golden plan assumed the axial legacy baseline was bankable on the VM. It is
not — the CT snapshot the legacy loader is pinned to has been deleted. Handing back to the Mac planner.**

**How the legacy CT loader resolves (today's code, `src/vqa_dataset.py`):**
- `PromptDataset.__getitem__` (axial_all_image) reads the cohort `nifti_path`, then
  `_nifti_path_to_blob_and_filename()` **discards the path's directory** and re-derives a bucket blob under
  a **hard-coded constant** `DEFAULT_NIFTI_BUCKET_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/nov25"`
  (bucket `su-vista-uscentral1`), filename `{parts[-2]}__{parts[-1]}.nii.gz`.
- Load order: `ct_dir/{filename}` on disk → else GCS blob download → else `img_data=None` (silent no-image).

**Why it can't run here:**
- The `v1_1` cohort's own `nifti_path` values point at `…/vista/nov25/…`; the code constant is also `nov25`
  — internally consistent, BUT **`nov25` no longer exists** in `gs://su-vista-uscentral1/…/vista/`
  (zero objects). The bucket now holds `feb26/` (+ `Feb/`, `dicom_metadata/`) — the v1_5 "Feb paths"
  migration. **feb26 is the current snapshot.**
- Result: every CT download 404s → an axial "baseline" would bank `image_count=0` for **all** rows — an
  artifact of the missing snapshot, not a real "before." So the axial byte-identity arm of the golden test
  is **not achievable on this VM** unless a `nov25` archive is located.

**Go-forward (validated) — CTs linked in the vista-bench materialized dataset:**
- Current materialized table = `vista_bench_v1_5.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`
  (374,000 rows; 249,600 with CT). It **dropped** the string `nifti_path` (now a deprecated INTEGER) and
  links CTs via `image_study_uid` + `image_series_uid` (+ `local_path` = original DICOM provenance
  `gs://rit-shc-imaging-confidential…`, and `ct_accession_number`).
- Materialized NIfTI convention: `…/vista/feb26/{image_study_uid}__{image_series_uid}.nii.gz`. Verified a
  real v1_5 (study_uid, series_uid) pair → **the feb26 blob exists**.
- The design change (fits the roadmap's modular-preprocessing thesis): the CT modality adapter should
  resolve its file **from the materialized dataset link** (`image_study_uid`/`image_series_uid` → feb26),
  **not** from the hard-coded `nov25` prefix + `_nifti_path_to_blob_and_filename` heuristic. Then a
  re-materialization (nov25→feb26→…) is a data change, not a code edit.

**Mac decisions to make (re-plan):**
1. Backwards-compat golden — accept that the **axial arm is unbankable on this VM** (nov25 gone) and anchor
   the compat test on the `no_image`/EHR arm (already banked, green); or task someone to locate a `nov25`
   archive. Gate-1/Gate-2 (image-hash/windowing) would then be validated another way.
2. Go-forward CT — repoint CT resolution to the **v1_5 / feb26 dataset link** (study_uid+series_uid), drop
   `DEFAULT_NIFTI_BUCKET_PREFIX`. This also implies moving the eval substrate from `v1_1` → `v1_5`.
3. Roll decisions (1)+(2) plus the Step-1 `numeric_value` declared-delta OQ-K into a superseding plan doc.

**Executor state at handback:** `no_image` legacy baseline banked + green (the compat anchor that *does*
work here). Staging at `/mnt/su-vista-uscentral1/vistabench/vlm/`. Uncommitted on the VM: `ehr.py` unit
remap, `configs/all_tasks.vm.yaml`, this doc, `docs/next.md`. No axial run attempted (would have banked an
all-zero artifact). PHI: counts / field-names / UIDs-as-structure only; no patient rows touched.
