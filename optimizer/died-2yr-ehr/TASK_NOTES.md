# died-2yr-ehr: task notes

The EHR twin of `optimizer/died-2yr-ct/`, copied from it 2026-10-21. Spec (approved 2026-10-21):
`planning/vista_eval_vlm/2026-10-21-died-2yr-ehr-agentic.md` on the mount
(`/mnt/su-vista-uscentral1/chaudhari_lab/phil/`). Since 2026-10-27 it is the full-EHR arm of the agent in the eval-set
spec (`planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md`, work package 2): two forced yes/no answers, balanced
accuracy, $0.50 a patient, on the seat (Vertex an opt-in, 2026-10-28). Engine pin: agentic-label-opt `feat/vertex-login-plus-seat-pause` bb216c2 (main's Vertex login route merged with the seat-limit pause, 2026-11-01).

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
| Route | one setting for all sessions, the seat by default (engine `isolation.route`; token at `~/.config/alo/oauth_token`). `ALO_ROUTE=vertex` puts every session on Vertex holding the machine's `optimizer` login (no service account can be made); that login can read the labels, so the network block is the guard and the cost check runs the engine's after-run audit as the evidence (Phil 2026-10-28) |
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

## Probability version (`ANSWER=probability`, Phil 2026-10-31)

Spec: `planning/vista_eval_vlm/2026-10-31-died-2yr-ehr-probability.md` on the mount. One setting switches the task;
yes/no stays the default, and its prompt and prompt tag are byte-identical to before (checked 2026-10-31).

| Decision | Answer |
|---|---|
| Answer | a whole number 0-100 per horizon (stored 0..1), plus 3-5 `reasons` (`fact`, record `file`, `date`), never scored |
| Score | AUROC at each horizon (Mann-Whitney, ties half), averaged; a failed or unreadable answer scores 0.5. TRAIN breakdown adds Brier, mean probability by outcome, calibration by tenth, distinct values, share on the most common value |
| Mistake | a probability on the wrong side of the horizon's TRAIN death rate (18.5%, 30.6%): only picks what the optimizer studies |
| Seed, brief, instructions | `program_agentic_probability/`, `context/task_agentic_probability.md`, `prompts/optimizer_instructions_probability.md` |
| Schedule, rounds | TRAIN `linear(100, 400, over=10)`, VAL 200; up to 10 rounds; stop after 2 rounds with no new best VAL (`run.py` enforces it before the next pass; `run_on_vm.sh` reports it as finished) |
| Dollar stops | stop and ask past $200 a run or after a round over $50 (`RUN_BUDGET_USD`, `ROUND_BUDGET_USD`), from the run's receipts plus each round's optimizer budget, checked before each pass |
| Work folder | `alo-tasks/died-2yr-ehr-prob-summary` (`-prob` without the summary); run `prob-summary-r1` |

    ANSWER=probability TASK_DATA_DIR=/tmp/synth uv run ... python scripts/check_task.py check   AUROC/Brier == sklearn, 0-100 parsing
    ANSWER=probability TASK_DATA_DIR=/tmp/synth uv run --project $AGENTIC_LABEL_OPT python run.py dry-run --iterations 4
    python scripts/score_baselines.py score --ids val --horizons both     baselines incl. clinical XGBoost at 1 and 2 years
    python scripts/check_reasons.py --run <work folder>                   share of reasons in a real file, pre-cutoff, found on that date
    ANSWER=probability python run.py rescore --run <work folder>          $0: each VAL pass per horizon (AUROC, ties, Brier)

## Survival version (`ANSWER=survival`, Phil 2026-11-01)

Spec: `planning/vista_eval_vlm/2026-11-01-died-ehr-survival.md` on the mount. Extends the probability version's run
settings (rounds, TRAIN schedule, no-new-best and dollar stops); yes/no and probability prompts and tags are
byte-identical to before (checked 2026-11-01).

| Decision | Answer |
|---|---|
| Answer | a survival curve from the record's last day: `survival_curve` = points `[months, chance alive 0-1]`, months from 0 strictly increasing, never rising, to month 120 asked (unreadable only if it stops before TRAIN's window end, about 74.6 months), up to 2,000 points; read as a step curve (1 before the first point); plus 3-5 `reasons` |
| Gold | v1.7 `os_survival` (`time_to_event_days`, `event_observed`) in `survival.jsonl` (`scripts/build_data.py survival`), time counted from each record's own cut (`cutoffs.jsonl`): for 356 patients today's clock (first treatment - 7 days) is later than the record's cut (median 27 days, up to 475; first-treatment dates moved after the export), so their time is today's time plus that gap (Phil 2026-11-01). 5,482 of 5,549 patients have it: TRAIN 3,836, VAL 804, TEST 842; pools TRAIN 394, VAL 198 (summary-only: the same pools, all VAL 803). The rest are left out of survival mode. (The first key, 2026-11-01 morning, joined to the yes/no label rows and so dropped 73 censored patients and timed the 356 from today's clock; fixed the same day.) |
| Score | integrated Brier, lower is better (the climb score): censoring-weighted Brier on months 1-120 kept inside vista-eval's stable range (0.999 x 90th percentile of TRAIN death times, capped below the last death scored; months 1-74 here), trapezoid / span. C-index over all follow-up with risk = minus the curve's mean over the scoring months (minus restricted mean survival: the Cox order for Cox curves). Brier and time-dependent AUC at 12/24/36/60 months reported. Plain Python in `survival_metrics.py`, checked equal to scikit-survival |
| Reference | the average curve (TRAIN Kaplan-Meier) scored on the same patients in every breakdown |
| Unreadable | a failed session or unreadable curve stops the run right after that pass, before it is scored (`run.survival_after_pass`, the engine's after-pass hook, reason `unreadable_curve`; the last round's VAL pass included). `run.stop_rule` also refuses a resumed run with one in its finished passes. `score-val` records no score when any curve is unreadable. Network stops never count: the engine halts those passes as resumable |
| Mistake | a curve that did worse on that patient than the average curve (its own censoring-weighted Brier over the scoring months) |
| Seed, brief, instructions | `program_agentic_survival/`, `context/task_agentic_survival.md`, `prompts/optimizer_instructions_survival.md` |
| Work folder | `alo-tasks/died-2yr-ehr-surv-summary` (`-surv` without the summary); run `surv-summary-r1` |
| Baselines | vista-eval `configs/runs/v1_7/survival_monthly_grid/` (clinical Cox and XGBoost survival, clinical LLM Cox, patient-summary Cox, radiology-text and MOTOR probes re-fitted) write per-patient VAL and TEST curves to `results/v1_7/survival_monthly_grid/`; `score_baselines.py survival` scores them with `survival_metrics.py` |

    ANSWER=survival TASK_DATA_DIR=/tmp/synth uv run -p 3.12 --project $AGENTIC_LABEL_OPT --with scikit-learn --with scikit-survival --with pandas --with pyarrow python scripts/check_task.py check
    ANSWER=survival TASK_DATA_DIR=/tmp/synth uv run --project $AGENTIC_LABEL_OPT python run.py dry-run --iterations 4
    python scripts/score_baselines.py survival --ids pool|val --out <file>     needs survival.jsonl and the grid rows

Python 3.12 for the check: scikit-survival does not build on the engine's 3.14 here (its `ecos` dependency needs a
compiler). The engine itself runs survival mode on 3.14: the scorer needs no new package.

### Survival, summary-only bare (`ANSWER=survival SUMMARY_ONLY=1`, Phil 2026-11-01)

On the shared summary-only foundation (2413631). Each session gets only `summary.txt`, under the eval set's Astra/Argon
summary prompt verbatim (opening + PATIENT SUMMARY + Question, `src/evalset/cards/death.py` on main) with the survival
question and curve format in place of yes/no; no record, record guide, cohort survival numbers, persona or hints. The
optimizer's brief and instructions (`context/task_agentic_survival_bare.md`, `prompts/optimizer_instructions_survival_bare.md`)
keep the task, what the session sees, the curve format and the score, nothing else; its round report is the integrated
Brier and counts only. Mistake sampling is kept (worse than the average curve on that patient), and every TRAIN patient,
right or wrong, comes with its summary, findings note and transcript; no prebuilt tools. Pools TRAIN 394, VAL 198; all VAL
803. Work folder `alo-tasks/died-2yr-ehr-surv-summaryonly`; run `surv-summaryonly-r1` (`run_on_vm.sh` names it from
`SUMMARY_ONLY=1`). `check_task.py check` adds: the prompt carries the eval-set text verbatim (read from origin/main), and
no cohort number or hint is in the prompt, seed, brief, instructions or per-round message.

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
