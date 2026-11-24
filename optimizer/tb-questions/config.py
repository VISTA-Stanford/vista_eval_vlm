"""Every decision for the tb-questions task, in one place: two Claude Code agents answer the thoracic tumor board's
own questions, each as probabilities, one sealed session per patient, and agentic-label-opt's optimizer (Opus 5.5)
rewrites the instructions they follow, climbing on AUROC from a zero-shot start.

    TB_AGENT=treatment   the tumor board-time summary (compact_summary, cut one week before the first thoracic tumor
                         board) -> the early-stage curative plan and/or the first-line metastatic regimen, five options
                         each, one probability per option; climb score = mean one-vs-rest AUROC over the two questions
    TB_AGENT=treatment-record   the same questions and scoring, from the same summary plus the patient's full EHR record
                         (OMOP tables, notes, allow-listed registry fields) cut at the same day, with a notes reader
                         tool from round 0 (spec planning/vista_eval_vlm/2026-11-21-tb-treatment-full-record.md)
    TB_AGENT=imaging     the most recent chest CT before the tumor board and its prior CT -> progression yes/no as one
                         0-100 probability; climb score = AUROC

Spec: planning/vista_eval_vlm/2026-11-20-tb-agent-questions.md (Approved 2026-11-20); hand-off
session/2026-11-20-tb-handoff-3-agent-task.md. Built from ct-findings-5 (multi-question runner, probability scoring,
audit, cost check, launcher) with died-2yr-ehr's summary staging. Lines marked ``DECIDE`` name who decided;
TASK_NOTES.md has the table. Always "tumor board", never the bare word (Phil 2026-11-20).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Which agent this process is. Required: the two agents share code, never a work folder, data folder or image.
AGENT = os.environ.get("TB_AGENT", "")
if AGENT not in ("treatment", "imaging", "treatment-record", "boards-record"):
    raise SystemExit("set TB_AGENT=treatment, imaging, treatment-record or boards-record")
#: The treatment questions (summary only, or summary + record). RECORD: the full-record arm.
TREATMENT = AGENT in ("treatment", "treatment-record")
RECORD = AGENT == "treatment-record"
#: BOARDS: one session per tumor board (not per patient), the record cut 7 days before that board, no summary; each
#: board's routed questions (spec planning/vista_eval_vlm/2026-11-22-tb-board-questions-agent.md, Approved 2026-11-22).
BOARDS = AGENT == "boards-record"
#: Either record mode: a record/ folder is staged and its local export is forbidden to the optimizer.
HAS_RECORD = RECORD or BOARDS
#: Boards only: which model answers (spec: Sonnet and Haiku arms, same data). Names the work folder, images and
#: results folder (``RUN_LABEL``), never the data folder.
TB_MODEL = os.environ.get("TB_MODEL", "sonnet")
if TB_MODEL not in ("sonnet", "haiku") or (TB_MODEL != "sonnet" and not BOARDS):
    raise SystemExit("TB_MODEL=haiku is for TB_AGENT=boards-record only")
#: Boards only: which question set the climb scores. "board" (spec 2026-11-22: T1/T2/T3) or "treatment" (spec
#: planning/vista_eval_vlm/2026-11-23-tb-board-treatment-climb.md: six yes/no modality questions + T3, every board).
TB_QUESTIONS = os.environ.get("TB_QUESTIONS", "board")
if TB_QUESTIONS not in ("board", "treatment") or (TB_QUESTIONS != "board" and not BOARDS):
    raise SystemExit("TB_QUESTIONS=treatment is for TB_AGENT=boards-record only")
TX = TB_QUESTIONS == "treatment"
#: Treatment only: which seed. "v1" (default; Sonnet board climb r1's program-v1 + the six questions) or "untuned" (the
#: plain board seed + the six questions; Phil 2026-11-24: untuned Sonnet first, then Opus).
TB_SEED = os.environ.get("TB_SEED", "v1")
if TB_SEED not in ("v1", "untuned") or (TB_SEED != "v1" and not TX):
    raise SystemExit("TB_SEED=untuned is for TB_QUESTIONS=treatment only")
RUN_LABEL = AGENT + ("-tx" if TX else "") + ("-untuned" if TB_SEED == "untuned" else "") + \
    ("-haiku" if TB_MODEL == "haiku" else "")
# --------------------------------------------------------------------------------------------------
# Where the engine is. The engine checks that it runs from a git checkout that contains ENGINE_PIN.
# --------------------------------------------------------------------------------------------------
#: A checkout of agentic-label-opt at ENGINE_PIN (shared with died-2yr-ct and died-2yr-ehr).
ENGINE_ROOT = Path(os.environ.get("AGENTIC_LABEL_OPT") or Path.home() / "code" / "agentic-label-opt" / ".claude" / "worktrees" / "mortality-engine").resolve()
if not (ENGINE_ROOT / "engine" / "driver.py").is_file():
    raise SystemExit(f"no engine at {ENGINE_ROOT}: set AGENTIC_LABEL_OPT to your agentic-label-opt checkout")
#: The full 40-character engine commit this task was built and tested against. Bump it on purpose, after the dry run.
ENGINE_PIN = "bb216c2489ff63bebd8ad3f1acdcb9f969bb0e2f"  # DECIDE (spec: feat/vertex-login-plus-seat-pause, as died-2yr-ct)
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

# --------------------------------------------------------------------------------------------------
# The task
# --------------------------------------------------------------------------------------------------
TASK_NAME = f"tb-questions-{RUN_LABEL}"  # names the work folder and images

#: Phil's summary opening (11-20), verbatim from the eval-set card (vista_eval_vlm feat/tb-question-cards 1888097,
#: src/evalset/cards/board_questions.py SUMMARY_OPENING). scripts/check_task.py compares every quoted text here with
#: that file when the cards are readable.
SUMMARY_OPENING = ("You are given a summary of this patient's longitudinal electronic health record, summarized from a "
                   "point cut off one week before their first thoracic tumor board encounter.")
#: The card's several-option answer line and yes/no probability line (src/evalset/cards/base.py ANSWER_OPTIONS,
#: ANSWER_PROB), verbatim.
ANSWER_OPTIONS = "Give each letter a probability from 0 to 100. These probabilities must add up to 100."
ANSWER_PROB = "Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number."

from board_options import *  # noqa: E402,F403  (the option tables: EARLY_STAGE_OPTIONS, ..., BOARD_ANSWER_CODES)
from board_options import (AFTER_SURGERY_OPTIONS, BOARD_EARLY_OPTIONS, BOARD_FIRST_LINE_OPTIONS,  # noqa: E402
                           DECISION_OPTIONS, EARLY_STAGE_OPTIONS, INTENT_OPTIONS, SYSTEMIC_OPTIONS)


def _options_block(options) -> str:
    return "\n".join(f"{k}. {text}" for k, _c, text in options)


class Question(tuple):
    """(key, vista_bench task, question text as the card asks it, options). ``options`` is empty for a yes/no question
    (answer: one 0-100 probability of yes) and ((letter, code, text), ...) for a several-option question (answer: one
    0-100 probability per letter, rescaled to sum to 100 whatever the total; only all zeros is unreadable)."""

    def __new__(cls, key, task, text, options=()):
        return super().__new__(cls, (key, task, text, tuple(options)))

    key = property(lambda self: self[0])
    task = property(lambda self: self[1])
    text = property(lambda self: self[2])
    options = property(lambda self: self[3])


#: The card's question bodies (from "Question:" on, the answer line included), verbatim. DECIDE: Phil 2026-11-20 (card
#: page v3 approved).
TREATMENT_QUESTIONS = (
    Question("early_stage_plan", "early_stage_management_answer",
             "Question: The tumor board discussed curative-intent treatment for this patient's stage I-III non-small "
             "cell lung cancer. Which treatment do you predict the tumor board will recommend?\n"
             + _options_block(EARLY_STAGE_OPTIONS) + "\n" + ANSWER_OPTIONS, EARLY_STAGE_OPTIONS),
    Question("first_line_regimen", "optimal_systemic_therapy_answer",
             "Question: The tumor board discussed first-line systemic treatment for this patient's metastatic "
             "non-small cell lung cancer. Which regimen do you predict the tumor board will recommend?\n"
             + _options_block(SYSTEMIC_OPTIONS) + "\n" + ANSWER_OPTIONS, SYSTEMIC_OPTIONS),
)
#: Step 4's progression question (src/evalset/cards/board.py), its yes/no answer line swapped for the eval set's
#: probability line (base.ANSWER_PROB), as the probability recipes do.
IMAGING_QUESTIONS = (
    Question("progression", "progression_assessment_answer",
             "Question: As of one week before the tumor board, does the most recent imaging show cancer "
             "progression?\n- Yes: progression.\n- No: response or stable disease.\n" + ANSWER_PROB),
)
#: The imaging agent's opening, in place of the card's "These are axial slices ..." (the agent views the volumes
#: itself; the scans named as the card names them).
IMAGING_OPENING = ("You are given this patient's most recent chest CT before their thoracic tumor board and the prior "
                   "CT, as two CT volumes: PRIOR and MOST RECENT.")

# --------------------------------------------------------------------------------------------------
# Boards (TB_AGENT=boards-record): the board questions; option tables in board_options.py.
# --------------------------------------------------------------------------------------------------
def _board_q(key, task, text, options):
    opts = tuple((k, c, t) for k, c, t, _v in options)
    return Question(key, task, f"Question: {text}\n" + _options_block(opts) + "\n" + ANSWER_OPTIONS, opts)


BOARD_QUESTIONS = (
    _board_q("decision_type", "T1", "What will the tumor board decide for this patient at this encounter?", DECISION_OPTIONS),
    _board_q("intent", "T2", "What will the intent of the tumor board's plan be?", INTENT_OPTIONS),
    _board_q("early_stage_plan", "T3", "This patient has untreated stage I-III non-small cell lung cancer. What will "
             "the tumor board recommend?", BOARD_EARLY_OPTIONS),
    _board_q("after_surgery_plan", "T3b", "This patient has stage I-III non-small cell lung cancer and has had surgery "
             "or chemoradiation. What will the tumor board recommend next?", AFTER_SURGERY_OPTIONS),
    _board_q("first_line_regimen", "T4", "This patient has untreated stage IV or recurrent non-small cell lung "
             "cancer. Which first-line plan will the tumor board recommend?", BOARD_FIRST_LINE_OPTIONS),
)
#: The climb's score: the mean over these questions only. T3b and T4 are scored and reported, flagged too small to
#: read alone (VAL 19 and 36 boards; Phil 2026-11-22).
CLIMB_KEYS = ("decision_type", "intent", "early_stage_plan")
#: The A-E-only score for T3 / T4 (Phil 2026-11-22): boards whose answer is A-E, probabilities renormalised over A-E.
A_TO_E_KEYS = ("early_stage_plan", "first_line_regimen")
SMALL_KEYS = ("after_surgery_plan", "first_line_regimen")

from board_options import MODALITY_QUESTIONS as _MODALITIES  # noqa: E402

#: The treatment climb's yes/no questions, in the step-4 progression question's form (two bullets + ANSWER_PROB).
MODALITY_QUESTION_LIST = tuple(Question(k, k, f"Question: {t}\n- {y}\n- {n}\n" + ANSWER_PROB) for k, t, y, n in _MODALITIES)
if TX:
    CLIMB_KEYS = tuple(q.key for q in MODALITY_QUESTION_LIST) + ("early_stage_plan",)
QUESTIONS = (BOARD_QUESTIONS + MODALITY_QUESTION_LIST if TX else BOARD_QUESTIONS) if BOARDS else \
    TREATMENT_QUESTIONS if TREATMENT else IMAGING_QUESTIONS
#: The questions a TRAIN answer must get right to count as right (mistakes file, retirement). Treatment climb: only the
#: score's questions (spec 2026-11-23); otherwise every question asked.
MISTAKE_KEYS = CLIMB_KEYS if TX else tuple(q.key for q in QUESTIONS)
if not BOARDS:
    CLIMB_KEYS = tuple(q.key for q in QUESTIONS)
    A_TO_E_KEYS = SMALL_KEYS = ()
LABELS = tuple(sorted({c for q in QUESTIONS for _k, c, _t in q.options} or {0, 1}))

#: Built by scripts/build_data.py: splits/{train,val}.jsonl ({"id", <question keys>: answer code | null}; null = the
#: tumor board did not answer that question for this patient, so it is not asked or scored), pools.json (the fixed
#: TRAIN = feedback and VAL = picking pools, with the draw recorded under "_draw") and patients/<id>/ (treatment:
#: summary.txt; imaging: prior/ and current/, each ct.npz + meta.json). splits/test.jsonl only on Phil's go
#: (build_data.py test); run.py test is its only reader. On the shared mount (secure environment).
#: The optimizer never sees this folder. It sees TRAIN only through its mistakes, and VAL only as a score.
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "tb-questions-v1_7" / ("boards-tx" if TX else AGENT)))
#: The id -> person_id (and scans) map: never mounted, listed in the forbidden set.
#: The record agent shares the treatment agent's patients and so its id map (its own private folder holds only the
#: exporter's cohort file, build_data.py record-cohort).
ID_MAP_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "tb-questions-v1_7-private" / ("treatment" if AGENT == "treatment-record" else AGENT)))
#: Boards: each board's subgroup facts (patient hash for the patient-level bootstrap, first board, already treated),
#: kept out of the splits so nothing but answers is in them.
#: Record mode: the summaries stay in the treatment agent's folder (same patients, same ids); DATA_DIR holds the same
#: splits and pools plus cutoffs.jsonl and the record archive. The record's patient folders (exporter output,
#: build_data.py record-export) are on local disk: thousands of small files are slow on the mount.
SUMMARY_PATIENTS = MOUNT / "tb-questions-v1_7" / "treatment" / "patients"
RECORD_PATIENTS = Path(os.environ.get("TASK_RECORD_DIR", Path.home() / "data" / f"tb-questions-{'boards' if BOARDS else 'treatment'}-record")) / "patients"
PROGRAM_MODE = "agentic"
AGENTIC = True
ANSWER = "probability"
PROBABILITY = True

#: The labeling program the optimizer starts from (program-v0), and the files it may edit. Zero-shot (spec): the
#: card's opening and question(s) verbatim, the folder layout, and (imaging) the CT pair viewer.
SEED_PROGRAM_DIR = HERE / {"treatment": "program_treatment", "imaging": "program_imaging",
                           "treatment-record": "program_treatment_record",
                           "boards-record": ("program_boards_tx_untuned" if TB_SEED == "untuned" else "program_boards_tx")
                           if TX else "program_boards_record"}[AGENT]
EDITABLE_FILES = ("instructions.md", "tools")
CONTRACT_FILES: tuple[str, ...] = ()
#: The optimizer's task brief and its standing instructions (its system prompt).
TASK_BRIEF = HERE / "context" / f"task_{AGENT}{'-tx' if TX else ''}.md"
OPTIMIZER_INSTRUCTIONS = HERE / "prompts" / f"optimizer_instructions_{AGENT}{'-tx' if TX else ''}.md"

#: The model each patient's session runs on, how many run at once, and each one's cap (spec, Decided).
PROGRAM_MODEL = os.environ.get("PROGRAM_MODEL", "claude-haiku-5-5" if TB_MODEL == "haiku" else "claude-sonnet-5-5")  # DECIDE: Phil 2026-11-20
PROGRAM_MAX_CONCURRENT = int(os.environ.get("PROGRAM_MAX_CONCURRENT", "4" if TREATMENT else "2"))  # as died-2yr-ehr / ct-findings-5
PROGRAM_CALL_BUDGET_USD = float(os.environ.get("PROGRAM_CALL_BUDGET_USD",
                                               {"treatment": "0.25", "imaging": "0.75", "treatment-record": "0.50",
                                                "boards-record": "0.50"}[AGENT]))  # DECIDE: specs (record: Phil 2026-11-21)
PROGRAM_CALL_TIMEOUT_SECONDS = {"treatment": 600, "imaging": 1200, "treatment-record": 900, "boards-record": 900}[AGENT]  # two CTs or a record take longer

# --------------------------------------------------------------------------------------------------
# The metric: one number, higher is better. adapters.TaskScorer computes it.
# --------------------------------------------------------------------------------------------------
#: Treatment: per question, the one-vs-rest AUROC of each option over the patients asked it, averaged over the options
#: that have both a yes and a no patient (vista-eval's and evalset.score.option_table's multi-class rule: sklearn
#: roc_auc_score(multi_class="ovr", average="macro") over the answers present); the climb score is the mean of the two
#: questions. Imaging: the progression AUROC. A failed or unreadable answer counts as no information: 0.5 on a yes/no
#: question, 20 on every option of a five-option one (Phil 2026-11-20).
METRIC_NAME = "auroc"  # DECIDE

# --------------------------------------------------------------------------------------------------
# The optimizer and the climb's stop rules (spec: the death climb's rules)
# --------------------------------------------------------------------------------------------------
OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "claude-opus-5-5")  # DECIDE: Phil 2026-11-20
#: The climb's round limit (spec). ITERATIONS may be overridden for one run (the cost check sets 1); the cost report
#: extrapolates over CLIMB_ITERATIONS.
CLIMB_ITERATIONS = 8 if BOARDS else 10  # DECIDE: specs (boards: Phil 2026-11-22)
ITERATIONS = int(os.environ.get("ITERATIONS", str(CLIMB_ITERATIONS)))
#: Stop once this many rounds in a row set no new best VAL score (the seed's zero-shot VAL is the first best).
#: Enforced by run.py before the next pass, so a resume counts the same rounds. 0 = off.
STOP_AFTER_ROUNDS_WITHOUT_BEST = int(os.environ.get("STOP_AFTER_ROUNDS_WITHOUT_BEST", "3" if RECORD else "2"))  # boards: 2 (Phil 2026-11-22)  # DECIDE: specs (record: Phil 2026-11-21)
#: Dollar stops, enforced by run.py before each pass (0 = off): the run total counts every scan session this run id
#: paid for plus OPTIMIZER_SESSION_BUDGET_USD for each round started; a round = its TRAIN and VAL passes' sessions.
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", {"treatment": "150", "imaging": "250", "treatment-record": "250", "boards-record": "600"}[AGENT]))  # DECIDE: specs 2026-11-20 / 11-21
ROUND_BUDGET_USD = float(os.environ.get("ROUND_BUDGET_USD", {"treatment": "40", "imaging": "60", "treatment-record": "60", "boards-record": "100"}[AGENT]))  # DECIDE: specs 2026-11-20 / 11-21
OPTIMIZER_SESSION_BUDGET_USD = float(os.environ.get("OPTIMIZER_SESSION_BUDGET_USD", "3.0"))  # as died-2yr-ct
OPTIMIZER_EFFORT = os.environ.get("OPTIMIZER_EFFORT", "high")  # as died-2yr-ct
OPTIMIZER_TIMEOUT_SECONDS = 2700
#: Optimizer web search off (spec, Decided 2026-11-20): no search of its own and no search proxy either.
OPTIMIZER_WEB_SEARCH = False
SEARCH_PROXY = False  # DECIDE: spec "optimizer web search off"; ct-findings-5 had the proxy on
SEARCH_REQUESTS_PER_ROUND = 0
STEP_BACK_REVERT_TOLERANCE = 0.0
TOKEN_BUDGET: int | None = None

# --------------------------------------------------------------------------------------------------
# Which scans each iteration sees (docs/reference/engine-internals.md, "Data schedule").
# --------------------------------------------------------------------------------------------------
#: The pools are fixed by build_data.py (one row per patient). Treatment: TRAIN = the official train split (492
#: patients), VAL = the official valid split (111). Imaging: the 288 train + valid CT pairs redrawn into 188 feedback
#: (TRAIN) and 100 picking (VAL). VAL is scored whole every round; TRAIN grows to the whole feedback set by round 5
#: (spec: "feedback grows to the whole feedback set"; round 5 because past climbs stopped by rounds 4-7).
CLIMB_SCHEDULE: dict = ({"train_size": "linear(300, 500, over=4)", "val_size": "constant(637)"} if TX else
                        {"train_size": "linear(300, 600, over=4)", "val_size": "constant(637)"} if BOARDS else
                        {"train_size": "linear(200, 492, over=5)", "val_size": "constant(111)"} if TREATMENT else
                        {"train_size": "linear(90, 188, over=5)", "val_size": "constant(100)"})  # DECIDE: build 2026-11-20
# Boards (build 2026-11-22): all 637 VAL boards each round; TRAIN 300 growing to 600 of the 2,916 by round 4, so a
# round (about 1,240 sessions at the spec's ~$0.07) stays under the $100 round cap; the cost check measures it.
# Treatment climb (build 2026-11-23): TRAIN to 500, because its cost check measured $0.073 a session from v1 (a
# 600-board round $93, too close to the $100 cap); about $86 a round.
#: Patients answered right several rounds in a row are retired from TRAIN by the engine, so later batches are smaller.
SCHEDULE: dict | None = CLIMB_SCHEDULE
if os.environ.get("TASK_SCHEDULE"):  # the same, as JSON, for one run: the cost check uses it
    import json as _json

    SCHEDULE = _json.loads(os.environ["TASK_SCHEDULE"])

# --------------------------------------------------------------------------------------------------
# Where everything a run writes goes. Outside the repo, so nothing a run writes is ever committed.
# --------------------------------------------------------------------------------------------------
WORKDIR = Path(os.environ.get("TASK_WORKDIR", Path.home() / ".cache" / "alo-tasks" / TASK_NAME)).resolve()

# --------------------------------------------------------------------------------------------------
# Sealing: which model route and which images. docs/guide/decisions.md, "Which models it can use".
# --------------------------------------------------------------------------------------------------
#: The optimizer runs in a container that cannot see your Claude Code login, so it needs its own token.
#: `claude setup-token` mints a one-year token on the account you sign into (Pro, Max, Team or Enterprise):
#: your existing seat, not a new account or an API key. Pick your organization's account at the browser
#: login. run.py reads the token from TOKEN_ENV_NAME, else from TOKEN_FILE (keep it mode 600).
#: `run.py check-token` tests it alone before you spend anything. docs/guide/decisions.md, "The Claude token".
TOKEN_ENV_NAME = "CLAUDE_CODE_OAUTH_TOKEN"
TOKEN_FILE = Path(os.environ.get("ALO_TOKEN_FILE", Path.home() / ".config" / "alo" / "oauth_token"))
#: The Claude Code version baked into the optimizer image; its tag must name it.
CLAUDE_CODE_VERSION = "2.1.307"  # Vertex global verified on this version 2026-10-21 (the run `first` used 2.1.295)
OPTIMIZER_IMAGE_TAG = f"alo-{TASK_NAME}-optimizer:{CLAUDE_CODE_VERSION}"
WORKER_IMAGE_TAG = f"alo-{TASK_NAME}-worker:3.11.0"
#: The route, one setting for the optimizer, the patient sessions and the search sessions (engine isolation.route).
#: Vertex is the default here (spec, Decided 2026-11-20); ALO_ROUTE=anthropic runs one run on the seat instead. Vertex bills
#: the GCP project, not the seat, and reaches Google's API addresses, not api.anthropic.com, so it neither spends
#: the seat's session limit nor shares the Cloud NAT's 64 ports (Phil 2026-10-21). No service account can be made,
#: so on Vertex every session holds the machine's `optimizer` login (engine OPTIMIZER_LOGIN, the file crc and
#: rad-eval mount) and runs as its owner. That login can read the BigQuery labels: the network block (only the
#: Vertex model endpoint and Google sign-in) is the guard, and the cost check runs the engine's after-run audit
#: (scripts/audit_vertex_run.py) as the evidence (Phil 2026-10-28).
#: Both routes are approved for this de-identified data, CT slice images included (Phil 2026-10-17, 10-20, 10-22).
from isolation.route import OPTIMIZER_LOGIN, Route, route_name  # noqa: E402  (the engine is on sys.path above)

ROUTE = Route(route_name("vertex"), vertex_project="som-nero-plevriti-deidbdf",
              vertex_region="global",  # Claude 5.5 is served on global only in this project (probed 2026-10-17)
              token_env_name=TOKEN_ENV_NAME, vertex_credential=OPTIMIZER_LOGIN)  # DECIDE
#: Patient data: turns on the engine's patient-data guards (fail-closed run store, Vertex only).
PATIENT_DATA = False  # DECIDE: de-identified, approved for the subscription route (Phil 2026-10-17, 10-20)
