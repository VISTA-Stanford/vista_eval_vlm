"""Every decision for the radprog-ct task, in one place: a Claude Code agent reads one scan pair (the current chest CT
and the prior CT its report was read against) and gives the probability that the cancer progressed, one sealed session
per pair, and agentic-label-opt's optimizer (Opus 5.5) rewrites the instructions it follows, climbing on AUROC from a
zero-shot start.

Spec: planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md (Approved 2026-11-21). Copied 2026-11-21 from
optimizer/tb-questions (the tumor board imaging agent, vista_eval_vlm feat/tb-agent-questions 716bf60), hard-wired to
its imaging path: the treatment branches the copied modules still carry never run (TREATMENT is False). Lines marked
``DECIDE`` name who decided; TASK_NOTES.md has the table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: One agent only. The copied tb-questions modules branch on these; radprog-ct is always the CT-pair agent.
AGENT = "imaging"
TREATMENT = False
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
TASK_NAME = "radprog-ct"  # names the work folder and images

#: The eval set's probability answer line (src/evalset/cards/base.py ANSWER_PROB), verbatim.
ANSWER_PROB = "Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number."
#: Kept for the copied modules' imports; the treatment path never runs here.
SUMMARY_OPENING = ""
ANSWER_OPTIONS = ""


class Question(tuple):
    """(key, vista_bench task, question text as the card asks it, options). ``options`` is empty: a yes/no question,
    answered with one 0-100 probability of yes."""

    def __new__(cls, key, task, text, options=()):
        return super().__new__(cls, (key, task, text, tuple(options)))

    key = property(lambda self: self[0])
    task = property(lambda self: self[1])
    text = property(lambda self: self[2])
    options = property(lambda self: self[3])


#: Step 3's question (src/evalset/cards/progression.py), word for word, its answer line swapped for the probability
#: line as every probability recipe does. scripts/check_task.py compares it with the card. DECIDE: same question text
#: for every VLM and as the agent's seed instructions (Phil 2026-11-21).
CARD_QUESTION = ("Question: Comparing the CURRENT scan with the PRIOR scan, has the cancer progressed?\n"
                 "- Yes: progression or mixed response: clear growth of known malignant lesions, or new lesions "
                 "suspicious for cancer, even if other sites improved.\n- No: stable disease or response.\n"
                 "Do not count non-cancer changes (infection, aspiration, organizing pneumonia, drug reaction, "
                 "evolving post-radiation change).\n")
IMAGING_QUESTIONS = (Question("progression", "radiology_progression", CARD_QUESTION + ANSWER_PROB),)
TREATMENT_QUESTIONS: tuple = ()
#: The agent's opening, in place of the card's "These are axial slices ..." (the agent views the volumes itself; the
#: scans named as the card names them). DECIDE: input = both CTs and their dates (Phil 2026-10-26, 2026-11-21).
IMAGING_OPENING = ("You are given two chest CTs of the same patient as two CT volumes: the PRIOR scan and the CURRENT "
                   "scan, with the number of days between them.")
QUESTIONS = IMAGING_QUESTIONS
LABELS = (0, 1)

#: Built by scripts/build_data.py: splits/{train,val}.jsonl ({"id", "progression": 0 | 1}; 1 = progression or mixed),
#: pools.json (the fixed TRAIN = feedback and VAL = picking pools, with the draw recorded under "_draw") and
#: patients/<id>/ (one pair: prior/ and current/, each ct.npz + meta.json, plus pair.json with the days between the
#: scans). splits/test.jsonl only on Phil's go (build_data.py test); run.py test is its only reader. On the shared
#: mount (secure environment).
#: The optimizer never sees this folder. It sees TRAIN only through its mistakes, and VAL only as a score.
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "radprog-ct-v1_7"))
#: The id -> person_id (and scans) map: never mounted, listed in the forbidden set.
ID_MAP_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "radprog-ct-v1_7-private"))
PROGRAM_MODE = "agentic"
AGENTIC = True
ANSWER = "probability"
PROBABILITY = True

#: The labeling program the optimizer starts from (program-v0), and the files it may edit. Zero-shot (spec): the
#: card's opening and question(s) verbatim, the folder layout, and (imaging) the CT pair viewer.
SEED_PROGRAM_DIR = HERE / "program_imaging"
EDITABLE_FILES = ("instructions.md", "tools")
CONTRACT_FILES: tuple[str, ...] = ()
#: The optimizer's task brief and its standing instructions (its system prompt).
TASK_BRIEF = HERE / "context" / "task_imaging.md"
OPTIMIZER_INSTRUCTIONS = HERE / "prompts" / "optimizer_instructions_imaging.md"

#: The model each patient's session runs on, how many run at once, and each one's cap (spec, Decided).
PROGRAM_MODEL = os.environ.get("PROGRAM_MODEL", "claude-sonnet-5-5")  # DECIDE: Phil 2026-10-26, 2026-11-21
PROGRAM_MAX_CONCURRENT = int(os.environ.get("PROGRAM_MAX_CONCURRENT", "2"))  # as the tumor board imaging climb
PROGRAM_CALL_BUDGET_USD = float(os.environ.get("PROGRAM_CALL_BUDGET_USD", "0.75"))  # DECIDE: spec 2026-11-21
PROGRAM_CALL_TIMEOUT_SECONDS = 1200  # two CTs

# --------------------------------------------------------------------------------------------------
# The metric: one number, higher is better. adapters.TaskScorer computes it.
# --------------------------------------------------------------------------------------------------
#: The progression AUROC over the pool's pairs. A failed or unreadable answer counts as no information: 0.5 (the
#: tumor board climb's rule, Phil 2026-11-20).
METRIC_NAME = "auroc"  # DECIDE

# --------------------------------------------------------------------------------------------------
# The optimizer and the climb's stop rules (spec: the death climb's rules)
# --------------------------------------------------------------------------------------------------
OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "claude-opus-5-5")  # DECIDE: Phil 2026-11-20
#: The climb's round limit (spec). ITERATIONS may be overridden for one run (the cost check sets 1); the cost report
#: extrapolates over CLIMB_ITERATIONS.
CLIMB_ITERATIONS = 10  # DECIDE: spec
ITERATIONS = int(os.environ.get("ITERATIONS", str(CLIMB_ITERATIONS)))
#: Stop once this many rounds in a row set no new best VAL score (the seed's zero-shot VAL is the first best).
#: Enforced by run.py before the next pass, so a resume counts the same rounds. 0 = off.
STOP_AFTER_ROUNDS_WITHOUT_BEST = int(os.environ.get("STOP_AFTER_ROUNDS_WITHOUT_BEST", "2"))  # DECIDE: spec
#: Dollar stops, enforced by run.py before each pass (0 = off): the run total counts every scan session this run id
#: paid for plus OPTIMIZER_SESSION_BUDGET_USD for each round started; a round = its TRAIN and VAL passes' sessions.
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "250"))  # DECIDE: spec 2026-11-21
ROUND_BUDGET_USD = float(os.environ.get("ROUND_BUDGET_USD", "60"))  # DECIDE: spec 2026-11-21
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
#: The pools are fixed by build_data.py: TRAIN = 400 feedback pairs (table train split), VAL = 200 picking pairs
#: (table valid split), natural rate. VAL is scored whole every round, as the tumor board climb does, so every round's
#: score is on the same 200 pairs (build decision 2026-11-21; Phil's 10-26 answer had VAL grow 100 -> 200, but a growing
#: VAL makes rounds incomparable). TRAIN feedback grows 40 -> 100 a round over 4 rounds (Phil 2026-10-26).
CLIMB_SCHEDULE: dict = {"train_size": "linear(40, 100, over=4)", "val_size": "constant(200)"}  # DECIDE: see above
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
