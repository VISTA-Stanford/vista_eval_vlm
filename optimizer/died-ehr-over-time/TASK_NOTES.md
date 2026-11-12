# died-ehr-over-time: task notes

Spec (approved 2026-11-11): `planning/vista_eval_vlm/2026-11-11-died-ehr-agent-over-time.md` on the mount
(`/mnt/su-vista-uscentral1/chaudhari_lab/phil/`). A Haiku 5.5 Claude Code session reads a patient's record cut at a
decision point and gives a 0-100 chance of death within 1 and 2 years of that day, with 3-5 reasons. Two arms: fresh
(each point alone) and memory (also shown its own earlier risks and reasons, with their dates). One pass, no
hill-climb, validation only.

## What it reuses from died-2yr-ehr

`common.py` sets that task's settings through the environment (probability answers, record only, Haiku, this task's
data folder) and imports its `config`, `agentic` and `adapters`. Reused as is: the engine pin, route and image; the
record whitelist and read-only staging (`agentic.record_files`); the answer format and parsing (`agentic.program_spec`'s
`summarize` and `read_stamp`); the AUROC (`adapters.auroc`) and failed-answer rule (scored 0.5); the work-folder layout
and forbidden list (`optimizer.Layout`, `optimizer.forbidden`); `scripts/build_data.py`'s index-tumour SQL, exporter
path, registry allow-list and `rehydrate`. New: decision-point rows instead of patients, the prompt wrapper (question
timed from the CT day, no cohort death rate, memory block), the record guide's first line, the driver, the
scorer's extra metrics.

## Decisions

| Decision | Answer |
|---|---|
| Patients | K29 phase table, benchmark VAL, first treatment `iv_chemotherapy` or `oral_targeted`: 355; none is in the died-2yr-ehr task's TRAIN or TEST (239 are in its VAL, 116 not in that task). 306 have a labelled decision point, 246 one after treatment start |
| Decision points | chest-containing CTs (K29 exam type chest, chest-abdomen-pelvis, skull-to-thigh body); day = days from first treatment. Treatment start: closest to day 0 in -60..+30 (ties: earlier); first restaging: first in 28..120; 6/12/24 months: latest in mark-91..mark (marks 183, 365, 730) |
| Cut | the decision CT's day, inclusive (the exporter keeps what is dated on or before it). Two points on the same CT share one record |
| Labels | K16 `per_ct_died_any_cause_{1,2}_yr` from `K16-PERCT-ehr/clinical__clinical_pretx_intersection_per_example.parquet`; a point with neither label is dropped; a missing horizon drops that horizon only. Every other K16 file holding the CT agrees (22,282 comparisons against 51 files, the source excluded, 0 disagree, 2026-11-11) |
| Prompt | died-2yr-ehr's probability wrapper, record only, with no cohort death rate in either arm (Phil 2026-11-11) |
| Memory at treatment start | no earlier answer exists, so the memory arm takes the fresh arm's answers there (one set of calls); memory minus fresh there is 0 by construction and the table says so |
| Probes | MOTOR (`K16-PERCT-ehr`), CT MerlinOnc (`K16-PERCT-ct`), clinical (`K16-PERCT-ehr`). The stacked model has TEST rows only in K16, so no VAL score; the table says so |
| Spend | passes of 25 calls; one starts only if spend + 25 x $0.50 (the per-call cap) stays within $250; stop if the first 100+ calls averaged over $0.10 |
| Leaks | before any call: each record's DATE/TIMESTAMP columns, its registry text date columns (from the allow-list: column X governed by X_date) and note headers against the row's cut; any violation stops the run. An earlier reason in a memory block naming a date after the current cut is dropped and counted per point, not a stop (Phil 2026-11-11) |
| Treatment-start check | after the fresh arm's treatment-start answers, before any later point: stop if its mean AUROC (1 and 2 yr) is more than 0.05 below the Haiku probability seed's, on the patients both tasks hold (226; the seed's per-patient VAL answers on sllm-02) if at least 40 have seed answers, else against the seed's VAL-200 0.816. `OVER_TIME_PAST_START_GATE=1` goes on past it after Phil's go |

Counts (2026-11-11 build):

| point | patients | 1-yr n | 1-yr deaths | 2-yr n | 2-yr deaths |
|---|---|---|---|---|---|
| start | 248 | 248 | 66 | 233 | 113 |
| restaging | 196 | 196 | 49 | 184 | 79 |
| m6 | 158 | 158 | 45 | 145 | 64 |
| m12 | 138 | 138 | 37 | 120 | 60 |
| m24 | 73 | 73 | 13 | 65 | 22 |

813 rows, 800 records, 1,378 paid calls (813 fresh + 565 memory).

## Commands

    export AGENTIC_LABEL_OPT=~/code/agentic-label-opt/.claude/worktrees/mortality-engine
    U="uv run -q --project $AGENTIC_LABEL_OPT --with pandas --with pyarrow --with duckdb<2 --with scikit-learn --with pyyaml python -B"
    $U build_points.py table                 phil-sllm-01 (+ google-cloud-bigquery): patient list, points, labels, probes
    $U build_points.py export --dry-run      price the export; --go runs it (PHI write: Phil's go)
    $U scripts/check_task.py synth /tmp/ot-synth && TASK_DATA_DIR=/tmp/ot-synth $U scripts/check_task.py check   $0
    $U run_over_time.py leak-check           $0, on the exported records
    bash scripts/run_on_vm.sh                the paid run on the run VM (see the hand-off); SETUP_ONLY=1 stops before it
    $U run_over_time.py score --run-id R     $0, the score tables

Private data: `/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-ehr-over-time-v1_7-private/`. Aggregates:
`.../died-ehr-over-time-v1_7-runs/<host>/<run id>/`.
