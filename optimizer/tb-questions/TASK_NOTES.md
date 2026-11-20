# tb-questions: task notes

Spec: `planning/vista_eval_vlm/2026-11-20-tb-agent-questions.md` on the mount (Approved 2026-11-20); hand-off
`session/2026-11-20-tb-handoff-3-agent-task.md`. Built 2026-11-20 from ct-findings-5 (multi-question runner,
probability scoring, audit, cost check, launcher) with died-2yr-ehr's summary staging, on branch
`feat/tb-agent-questions` off `feat/ct-findings-agentic`. Engine pin agentic-label-opt bb216c2, Claude Code 2.1.307.
Every command needs `TB_AGENT=treatment` or `TB_AGENT=imaging`: one code base, two work folders, data folders and
images.

## The idea

One sealed Claude Code session per patient (Sonnet 5.5) answers the thoracic tumor board's own question(s) as
probabilities, and agentic-label-opt's optimizer (Opus 5.5) rewrites the instructions it follows, climbing on AUROC
from a zero-shot start. Question: does tuning an agent's instructions beat our embedding models and frontier models
asked directly?

## Decisions

| Decision | Answer |
|---|---|
| Treatment questions | `early_stage_management_answer` (5 options) and `optimal_systemic_therapy_answer` (5 options), vista_bench v1.7 `tb_classification_tasks`; a patient is asked only the question(s) the tumor board answered (spec) |
| Treatment input | the tumor board-time `compact_summary` only (cut one week before the first tumor board), quoted in the prompt and staged as `summary.txt` (spec) |
| Imaging question | `progression_assessment_answer` (yes/no), one 0-100 probability (spec) |
| Imaging input | the most recent CT before the tumor board + its prior (step-4 list, from `per_ct_prior_ct_link`), no report, no EHR; volumes as ct-findings-5 converts them (spec) |
| Question text | the eval-set cards word for word (feat/tb-question-cards 1888097: `board_questions.py`, step 4's `board.py` with the probability answer line), in the fixed per-call wrapper and in the seed; `check_task.py` compares them. Always "tumor board" (Phil 2026-11-20) |
| Answer | several-option: a number 0-100 per letter A-E, rescaled to sum to 100 whatever the total; only all zeros, a missing letter, a word or a number outside 0-100 is unreadable (Phil 2026-11-20). Yes/no: one number 0-100. A question asked but not answered fails the whole answer |
| Failed answer | scored as no information: 0.5 (yes/no), an equal share on every option (several-option) |
| Answer keys | `early_stage_plan`, `first_line_regimen`, `progression` (renamed from board_* 2026-11-20 so the sessions never read a bare "board") |
| Score | treatment: per question, mean one-vs-rest AUROC over the options present (vista-eval's rule; `check_task.py` matches `evalset.score._option_metrics` to 4 dp), averaged over the two questions; imaging: AUROC (spec) |
| Mistake | treatment: top letter not the tumor board's; imaging: probability on the wrong side of the TRAIN rate. Only picks what the optimizer studies |
| Pools | treatment: official split, TRAIN 492 patients (train), VAL 111 (valid). Imaging: the 288 train + valid pairs redrawn, stratified by answer, seed `tb-questions-v1:imaging:redraw`: TRAIN 188 (113 progression), VAL 100 (60). TEST = step6_test (119) / step4_test (45), built only on Phil's go (`build_data.py test`) |
| Optimizer briefs | TRAIN shares and pool sizes only; no VAL answer counts (evaluator 2026-11-20) |
| Pneumonitis | not built yet (the hand-off's build list leaves it out; the spec has it as zero-shot only): waiting on Phil |
| Seed | zero-shot: the card's opening and question(s) verbatim; imaging also the CT pair viewer (lung / soft tissue / bone) |
| Models, route | Sonnet 5.5 per patient, Opus 5.5 optimizer at effort high, $3 per optimizer session, Vertex global (default here), optimizer web search off and no search proxy (spec "optimizer web search off"; ct-findings-5 had the proxy) |
| Caps | treatment $0.25/session, $40/round, $150/run; imaging $0.75/session, $60/round, $250/run; stop after 2 rounds with no new best or 10 rounds (spec). The cost check fails (needs Phil's go) if a full round passes the round cap or fewer than 4 rounds fit in the run cap |
| Schedule | VAL whole every round; TRAIN `linear(200, 492, over=5)` / `linear(90, 188, over=5)`: the whole feedback set by round 5 (build decision: past climbs stopped by rounds 4-7); the engine retires patients answered right several rounds in a row, so later batches are smaller |
| Zero-shot | `run.py zero-shot`: the seed on every VAL patient once as its own run, with bootstrap intervals and an always-most-common row; exit 3 when the spec's stop rule fires (failures over 10%; treatment at chance on both questions). Not shared with the climb (pass reuse is keyed on the run), so the climb's round 1 scores the seed on VAL a second time: about $6 / $20 more, and a free run-to-run noise estimate. Its table reports AUROC both ways: failed = no information (the climb's rule) and failed left out (`auroc_readable_only`, the eval set's rule for the Argon / Opus rows); progression also gets balanced accuracy at 50. The verdict is saved in the file and `run_on_vm.sh` refuses to climb unless it passed |
| Machines | treatment climb phil-sllm-02, imaging climb phil-sllm-03 (spec); cost checks and zero-shot may run on any box with Docker and the optimizer login |
| Work folder | `~/.cache/alo-tasks/tb-questions-<agent>`, run `r1`; results to `chaudhari_lab/phil/tb-questions-v1_7-runs/<host>/<agent>/<run id>/` |
| Data | `chaudhari_lab/phil/tb-questions-v1_7/<agent>/` (`splits/{train,val}.jsonl`, `pools.json`, `patients/<id>/`); id map in `.../tb-questions-v1_7-private/<agent>/`. Ids are salted hashes of the patient |

## Data (built 2026-11-20)

- Treatment TRAIN 492 patients (18 asked both questions): early-stage A/B/C/D/E 79/22/84/61/109; metastatic 66/29/11/37/12.
- Treatment VAL 111 (3 both): early-stage 26/6/9/10/21; metastatic 18/9/1/13/1. Two metastatic options have one VAL
  patient each, so their AUROCs (and the metastatic score) are mostly noise.
- Imaging: step-4 list 334 pairs; 288 train + valid pooled; 1 with no split left out; 45 TEST untouched, 0 overlap.

## Commands

    TB_AGENT=treatment uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow --with pyyaml python -B scripts/check_task.py check
    TB_AGENT=treatment uv run --project $AGENTIC_LABEL_OPT python run.py dry-run --iterations 4       $0; ends STOPPED (no_new_best)
    TB_AGENT=treatment bash scripts/cost_check.sh                                                     PAID 5+5 (approved)
    TB_AGENT=treatment uv run --project $AGENTIC_LABEL_OPT python run.py zero-shot                    PAID VAL pass (approved)
    TB_AGENT=treatment uv run --project $AGENTIC_LABEL_OPT python run.py score-pass <answers.json>    $0
    TB_AGENT=treatment bash scripts/run_on_vm.sh                                                      PAID climb (approved, sllm-02)

## Reference points (in the task briefs)

Argon / Opus asked directly on TEST: treatment mean AUROC 0.81 / 0.82; progression 0.59 / 0.70. MerlinOnc single-CT
progression probe about 0.66 on the benchmark's valid split. The summary-embedding probe and the MerlinOnc pair row
are added to the briefs when hand-offs 1-2 land.

## Hard cases and noise

The answer key is the tumor board note as a language model read it. Treatment: several plans can be acceptable, the
summary may already state the plan (fair, Phil 2026-11-20). Progression: the tumor board used more than these two CTs
(PET, markers, symptoms); the two scans cover different lengths of the body, so the viewer's matched levels are rough.
