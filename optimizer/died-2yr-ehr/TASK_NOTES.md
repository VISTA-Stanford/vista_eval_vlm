# died-2yr-ehr: task notes

The EHR twin of `optimizer/died-2yr-ct/`, copied from it 2026-10-21. Spec (approved 2026-10-21):
`planning/vista_eval_vlm/2026-10-21-died-2yr-ehr-agentic.md` on the mount
(`/mnt/su-vista-uscentral1/chaudhari_lab/phil/`). Since 2026-10-27 it is the full-EHR arm of the agent in the eval-set
spec (`planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md`, work package 2): two forced yes/no answers, balanced
accuracy, $0.50 a patient, on the seat (Vertex deferred, 2026-10-28). Engine pin: agentic-label-opt main 9aee75f, shared with died-2yr-ct.

## The idea

A Claude Code session per patient answers vista_bench v1.7 `died_any_cause_1_yr` and `died_any_cause_2_yr`, yes or
no, from that patient's whole
EHR record up to the prediction time (first treatment - 7 days): every OMOP table, every note, and allow-listed
registry fields of the index tumour. Same patients, splits and pools as the CT task, so the two compare on
identical patients, and the published v1.7 EHR baselines are scored on the same VAL pool. Zero-shot first, then
5 optimizer rounds. Validation numbers only; TEST stays sealed.

## Decisions

| Decision | Answer |
|---|---|
| Patients | the CT task's exact list, ids, labels, splits and pools: TRAIN 3,886 (1,188 died, 30.6%), VAL 811 (31.7%), TEST 852; pools TRAIN 400 (129 died), VAL 200 (63 died) |
| Input | OMOP tables person, observation_period, visit_occurrence, visit_detail, condition_occurrence, procedure_occurrence, drug_exposure, drug_era, device_exposure, episode, image_occurrence, measurement, observation, note; `notes.txt`; registry tables of the index tumour; `meta.json`. All cut at the cutoff by the shared exporter; death table dropped; end dates clipped; registry-typed measurements (type 32879) dropped |
| Cutoff | vista_bench `embed_time` = first treatment date - 7 days |
| Registry | index tumour only, picked as the label picks it (all 5,549 found and matching the label's diagnosis date, 2026-10-21). Columns: `registry_allowlist.yaml`, from the 2026-10-21 audit with Phil's decisions |
| Answer | two forced yes/no answers in one session, death within 1 year and within 2 years of diagnosis; anything but exactly yes or no, or a failed session, is wrong at both (Phil 2026-10-27, eval-set spec) |
| Gold | `label` (2 years) and `label_1yr` (1 year, same patient and scan, added by the CT task's `build_data.py horizons` 2026-10-27; old files in `splits-2yr-only/`). TRAIN 718 died by 1 year, 470 in year 2; VAL pool 35 and 63 |
| Score | balanced accuracy at each horizon, averaged over the two (was AUROC on a probability until 2026-10-27) |
| Seed program | the plain questions only (`program_agentic/instructions.md`); the fixed wrapper (`agentic.py`) states both questions, the 7-day cutoff, the folder layout, the yes/no answer and the TRAIN death rates (18.5%, 30.6%). No clinical hints (Phil 2026-10-21). Since 2026-10-28 every session also gets `context/patient_data_guide.md` as `record_guide.md` beside the record: where each fact lives in the folder, adapted from the Cohort Explorer's guides, a data map with no prognostic hints (Phil 2026-10-28: the session must be told what it can read and where to look). Its hash is in the prompt tag |
| Per-patient model | `claude-sonnet-5-5`, $0.50 cap (Phil 2026-10-27) |
| Optimizer | Opus 5.5, effort high, $3 a session, may write tools; no WebSearch: it asks the engine's search proxy, which refuses patient-specific questions and logs every request (Phil 2026-10-22) |
| Route | one setting for all sessions, the seat by default (engine `isolation.route`; token at `~/.config/alo/oauth_token`). No Vertex-only service account can be made, so Vertex is future work (Phil 2026-10-28); the code keeps it: `ALO_ROUTE=vertex` mounts only a Vertex-only key at `~/.config/alo/vertex-sa.json` |
| Rounds | zero-shot (the seed on the 200 VAL pool) + 5 optimizer rounds (`ITERATIONS` 5), TRAIN 100 every round, VAL 200, run `yesno-r1` in `alo-tasks/died-2yr-ehr-yesno`; then the seed and the chosen version are scored once on all 811 VAL patients with `run.py score-val --tag <tag>` (Phil 2026-10-28, matching the CT task; was 2 rounds with TRAIN 30-90, 2026-10-21) |
| Image | `alo-died-2yr-ehr-optimizer:2.1.295`, its own tag: pandas, pyarrow, duckdb <2, grep/ripgrep; no CT packages |
| Where | records built on phil-sllm-01 (BigQuery) into an archive on the mount; the run on a new VM, records rehydrated to local disk (`scripts/run_on_vm.sh`) |

## Data

- `scripts/build_data.py splits` copies the CT task's splits and pools to
  `/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7/` (checked byte-identical). Done 2026-10-21.
- `scripts/build_data.py cohort` writes `.../died-2yr-ehr-v1_7-private/cohort.parquet` (id, person_id, cutoff,
  tumor_key: the exporter's input) and `.../died-2yr-ehr-v1_7/cutoffs.jsonl`. Done 2026-10-21: 5,549 patients.
- `scripts/build_data.py export --go` runs `vista-onc-labelers` `common/omop_export/export_omop.py` into the
  local copy (`~/data/died-2yr-ehr-v1_7`, `config.DATA_DIR`), leak-checks it and archives it to
  `.../died-2yr-ehr-v1_7/archive/`. Expensive step: waits for Phil's go. `--dry-run` priced it at 113.6 GB
  (about $0.71) on 2026-10-21.
- `scripts/build_data.py rehydrate` rebuilds the local copy from the archive on another VM, no BigQuery.
- The exporter also writes `OUT/registry_dates/` (raw governing dates, for its leak check). It is never in the
  archive, and staging copies only whitelisted record files (`agentic.record_files`), refusing a patient folder
  holding anything else; `run_on_vm.sh` fails if the folder exists in the task copy.

## Checks ($0)

    cd optimizer/died-2yr-ehr
    uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow python scripts/check_task.py synth /tmp/synth
    TASK_DATA_DIR=/tmp/synth uv run ... python scripts/check_task.py check     staging whitelist, scorer == sklearn, prompt
    TASK_DATA_DIR=/tmp/synth uv run --project $AGENTIC_LABEL_OPT python run.py dry-run
    python scripts/score_baselines.py reproduce      published clinical_pretx VAL AUROC, to 3 decimals
    python scripts/score_baselines.py score --ids pool
    bash scripts/cost_check.sh                       PAID (~$5, Phil's go): 5+5 patients, one iteration, judged

## Reference scores (v1.7 linear probes, VAL AUROC; not comparable with the task's balanced accuracy)

Reproduction: clinical_pretx 0.7766 published on all 1,124 VAL patients with a label, recomputed 0.7766.

| Baseline | All 811 task VAL | The 200 VAL pool |
|---|---|---|
| clinical_pretx | 0.796 | 0.804 [0.736, 0.864] |
| clinical_llm_pretx | 0.783 | 0.807 [0.742, 0.865] |
| run-vista-02 (MOTOR) | 0.802 | 0.797 [0.733, 0.856] |
| motor-arm-A-codes-notes | 0.797 | 0.780 [0.716, 0.840] |
| radiology text, Gemini (report-holding subset) | 0.806 (n=411) | 0.835 [0.756, 0.909] (n=94) |

## Open items

- `secondarydiagnosis1-10` are out of the allow-list for now: Phil kept them minus treatment-complication
  codes, which the audit identified by ICD-10 prefix (T80-T88, Y83-Y84, J95, I97, K91, N99, E89, G97, D78, H59,
  H95, L76, M96: 61 of 20,927 codes, 51 patients). That is a value rule; the exporter governs only by column and
  date. Comorbidities remain in the OMOP condition table.
- `yearfirstseen` is out: its governing date (`dateof1stcontact`) is in the outcomes table, and the exporter
  governs within a table only.
- Read as "outcome fields" only the audit's DROP bucket: the outcomes table's date-governed columns
  (`dateof1stcontact`, `datefirstpositivebx`, inpatient admission, `datenorxdecision`, COVID test) stay, nulled
  when after the cutoff. First-course start dates there are left out.
- `protocolparticipation` kept as the audit recommended without the "mostly not on trial" check it asked for.

## Hard cases and noise

Non-cancer deaths not foreshadowed in the record; long survivors with advanced disease on modern treatment;
patients referred in shortly before treatment with few notes here; registry death dates that lag.
