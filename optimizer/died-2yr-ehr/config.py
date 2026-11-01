"""Every decision for the died-2yr-ehr task, in one place: predict vista_bench's ``died_any_cause_1_yr`` and
``died_any_cause_2_yr`` (two forced yes/no answers, one session) from
the patient's EHR (OMOP tables, notes, allow-listed registry fields) cut at first treatment - 7 days, one sealed
Claude Code session per patient. The EHR twin of died-2yr-ct: same patients, splits, pools, scorer and engine.

Lines marked ``DECIDE`` were settled in the spec interview (Phil, 2026-10-21; spec
planning/vista_eval_vlm/2026-10-21-died-2yr-ehr-agentic.md on the mount); TASK_NOTES.md has each answer and why.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------------------
# Where the engine is. The engine checks that it runs from a git checkout that contains ENGINE_PIN.
# --------------------------------------------------------------------------------------------------
#: A checkout of agentic-label-opt main at ENGINE_PIN (shared with died-2yr-ct).
ENGINE_ROOT = Path(os.environ.get("AGENTIC_LABEL_OPT") or Path.home() / "code" / "agentic-label-opt" / ".claude" / "worktrees" / "mortality-engine").resolve()
if not (ENGINE_ROOT / "engine" / "driver.py").is_file():
    raise SystemExit(f"no engine at {ENGINE_ROOT}: set AGENTIC_LABEL_OPT to your agentic-label-opt checkout")
#: The full 40-character engine commit this task was built and tested against
#: (``git -C $AGENTIC_LABEL_OPT rev-parse HEAD``). Bump it on purpose, after re-running the dry run.
ENGINE_PIN = "bb216c2489ff63bebd8ad3f1acdcb9f969bb0e2f"  # DECIDE (feat/vertex-login-plus-seat-pause: main's Vertex optimizer-login route, after-run audit and background-traffic-off merged with fix/seat-limit-pause's seat-limit pause and search-answer fixes, 2026-11-01)
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

# --------------------------------------------------------------------------------------------------
# The task
# --------------------------------------------------------------------------------------------------
TASK_NAME = "died-2yr-ehr"  # DECIDE: short, lowercase, dashes; names the work folder and images
#: The gold: vista_bench v1.7 died_any_cause_2_yr (``label``: 1 died, 0 alive at 2 years; -1, too little follow-up,
#: dropped) and died_any_cause_1_yr (``label_1yr``, same patient and scan; scripts/build_data.py splits). The program
#: answers yes or no for each horizon (work package 2, Phil 2026-10-27: forced answers, scored like the VLMs).
LABELS = (0, 1)  # DECIDE
#: The two questions: (answer key in output.json, gold field in splits/*.jsonl, words in the prompt).
HORIZONS = (("died_within_1_year", "label_1yr", "1 year"), ("died_within_2_years", "label", "2 years"))
#: What each patient's session answers. "yesno" (the default: two forced answers, balanced accuracy, every run up to
#: 2026-10-31) or "probability": a 0-100 chance of death at each horizon plus the few record facts that moved it
#: most, scored by AUROC averaged over the two horizons (Phil 2026-10-31, spec
#: planning/vista_eval_vlm/2026-10-31-died-2yr-ehr-probability.md). Chooses the seed, the optimizer's brief and
#: instructions, the scorer, the default work folder, the rounds and the TRAIN schedule below.
#: "survival" (Phil 2026-11-01, spec planning/vista_eval_vlm/2026-11-01-died-ehr-survival.md): a whole survival curve
#: from the record's last day (first treatment - 7 days), scored like the benchmark's survival models against v1.7
#: os_survival: integrated Brier on a monthly grid (lower is better; the climb score), C-index over all follow-up.
ANSWER = os.environ.get("ANSWER", "yesno")  # DECIDE
if ANSWER not in ("yesno", "probability", "survival"):
    raise SystemExit(f"ANSWER must be yesno, probability or survival, not {ANSWER!r}")
#: Survival mode extends the probability mode's setting (Phil 2026-11-01: everything else as the probability run):
#: PROBABILITY stays True for the run settings below (rounds, TRAIN schedule, stops, dollar limits); SURVIVAL picks
#: the seed, brief, instructions, scorer and work folder.
SURVIVAL = ANSWER == "survival"
PROBABILITY = ANSWER in ("probability", "survival")
#: The task data on the shared mount (inside the secure environment): splits/{train,val,test}.jsonl
#: ({"id", "label"}), pools.json (the fixed TRAIN and VAL pools, the CT task's own) and the shared exporter's
#: archive of every patient's record. scripts/build_data.py writes it.
MOUNT_DATA_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7")
#: The working copy the task reads: the same splits and pools plus patients/<id>/ (one parquet per OMOP table,
#: notes.txt, registry_<table>.parquet, meta.json), rebuilt on local disk from the mount archive because tens
#: of thousands of small files are hours on the gcsfuse mount and seconds locally (scripts/run_on_vm.sh does it).
#: The optimizer never sees this folder (optimizer.forbidden lists it). It sees TRAIN only through its
#: mistakes, and VAL only as a score.
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", Path.home() / "data" / "died-2yr-ehr-v1_7"))
#: Tuomo's patient summary as an extra input beside the record (Phil 2026-10-29: test v1 + summary once on the
#: 200-patient VAL pool against v1's 0.739; a full run only if it helps). One LLM summary per patient dated exactly
#: at this task's cutoff (llm_labels.text_llm_summary, rich text), pulled by scripts/pull_summaries.py into
#: SUMMARY_DIR/<id>.txt. Off by default, so the yesno-r1 setup is unchanged. Under DATA_DIR, so the optimizer never
#: sees it.
WITH_SUMMARY = os.environ.get("WITH_SUMMARY", "0") == "1"  # DECIDE
#: Bare summary-only arm (Phil 2026-11-01): give each session only what the frozen VLMs (Astra/Argon/MedGemma) got in
#: the eval-set summary arm -- the eval-set summary prompt verbatim, the answer changed from yes/no to a probability
#: (or a curve, for survival) -- and strip the optimizer of every hint, to test whether agent-based optimization can
#: discover the base rate, the spread and what predicts death by sampling TRAIN itself. SUMMARY_ONLY stages only the
#: summary (no record, notes or record guide) as the session's whole input; BARE strips the hints from the prompt and
#: the brief and cuts the per-round scorecard to the headline AUROC and counts. SUMMARY_ONLY implies WITH_SUMMARY and
#: BARE. Patients without a summary are left out. The survival session shares this machinery and adds the curve
#: question and the Brier scorer branch on top (its WIP patch on the mount).
SUMMARY_ONLY = os.environ.get("SUMMARY_ONLY", "0") == "1"
#: BARE (the stripped seed, brief, optimizer instructions and scorecard) is implied by SUMMARY_ONLY and keyed to it
#: everywhere today; it is a named alias only so a later full-record bare arm can wire it as its own lever then. Do
#: not set it on its own yet: nothing but SUMMARY_ONLY drives the bare staging/scorecard, so a standalone BARE run
#: would pick the bare assets but still stage the full record.
BARE = SUMMARY_ONLY
if SUMMARY_ONLY:
    WITH_SUMMARY = True
if SUMMARY_ONLY and not PROBABILITY:
    raise SystemExit("SUMMARY_ONLY is built for ANSWER=probability or survival")
SUMMARY_DIR = DATA_DIR / "summaries"
#: The CT task's data and id map (id -> person_id): this task reuses its ids, labels, splits and pools.
CT_DATA_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7")
ID_MAP_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7-private")
#: This task's own private folder: the exporter's cohort file (id, person_id, cutoff, index tumour). Never mounted.
PRIVATE_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ehr-v1_7-private")
#: The registry columns a session may see (index tumour only), from the 2026-10-21 registry audit.
REGISTRY_ALLOWLIST = HERE / "registry_allowlist.yaml"
PROGRAM_MODE = os.environ.get("PROGRAM_MODE", "agentic")  # DECIDE: as died-2yr-ct
AGENTIC = PROGRAM_MODE == "agentic"

#: The labeling program the optimizer starts from (program-v0), and the files it may edit.
#: The seed is the plain question and the folder layout, no clinical hints (Phil 2026-10-21); tools/ starts empty.
_ANSWER_SUFFIX = ((("_survival" if SURVIVAL else "_probability") + ("_bare" if BARE else "")) if PROBABILITY else "")
SEED_PROGRAM_DIR = HERE / (f"program_agentic{_ANSWER_SUFFIX}" if AGENTIC else "program")
EDITABLE_FILES = ("instructions.md", "tools") if AGENTIC else ("label.py",)  # DECIDE: what the optimizer may change
CONTRACT_FILES: tuple[str, ...] = ()  # files shipped with the program that it must never change
#: The optimizer's task brief (prompts: docs/guide/prompts.md).
TASK_BRIEF = HERE / "context" / (f"task_agentic{_ANSWER_SUFFIX}.md" if AGENTIC else "task.md")
#: The optimizer's standing instructions (its system prompt), one per answer type.
OPTIMIZER_INSTRUCTIONS = HERE / "prompts" / f"optimizer_instructions{_ANSWER_SUFFIX}.md"

#: agentic only: the model each patient's session runs on, how many run at once, and each one's cap.
PROGRAM_MODEL = os.environ.get("PROGRAM_MODEL", "claude-sonnet-5-5")  # DECIDE: pinned id, same as CT (Phil 2026-10-21)
#: The Squid network gate stalls above about 2 concurrent sessions on phil-sllm-01 (recurrence run, 10-19).
PROGRAM_MAX_CONCURRENT = int(os.environ.get("PROGRAM_MAX_CONCURRENT", "2"))
#: $0.50 a patient, one session for both horizons (Phil 2026-10-27, eval-set spec). A patient has a median of
#: about 55 notes before the cutoff; the 5+5-patient cost check (an Expensive step) confirms under 1 in 20 hit it.
PROGRAM_CALL_BUDGET_USD = float(os.environ.get("PROGRAM_CALL_BUDGET_USD", "0.50"))
PROGRAM_CALL_TIMEOUT_SECONDS = 900

# --------------------------------------------------------------------------------------------------
# The metric: one number, higher is better. adapters.TaskScorer computes it.
# --------------------------------------------------------------------------------------------------
#: Balanced accuracy at each horizon, averaged over the two: forced answers, macro balanced accuracy, as the
#: eval set scores every model (spec planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md, Decided: Score).
#: Survival: integrated Brier, LOWER is better (Phil 2026-11-01); the engine reads the direction from the score, and
#: run.py's no-new-best rule from HIGHER_IS_BETTER.
METRIC_NAME = "integrated_brier" if SURVIVAL else "auroc" if PROBABILITY else "balanced_accuracy"  # DECIDE (probability: Phil 2026-10-31)
HIGHER_IS_BETTER = not SURVIVAL
#: Survival mode's answer key: {"id", "time_days", "event"} per patient, vista_bench v1.7 os_survival with time counted
#: from each record's own cut (cutoffs.jsonl; for 356 patients today's first-treatment clock is later, Phil 2026-11-01),
#: written by scripts/build_data.py survival. A patient with no
#: os_survival row (67 of 5,549) is left out of survival mode: not staged, not scored, not in the pools.
SURVIVAL_KEY = DATA_DIR / "survival.jsonl"

# --------------------------------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------------------------------
OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "claude-opus-5-5")  # DECIDE: Phil 2026-10-20
ITERATIONS = int(os.environ.get("ITERATIONS", "10" if PROBABILITY else "5"))  # DECIDE: zero-shot + 5 rounds, as the CT task (Phil 2026-10-28; was 2, 2026-10-21)
#: Stop the run once this many rounds in a row set no new best VAL score (the seed's zero-shot VAL counts as the
#: first best). Enforced by run.py before the next pass, so a resume counts the same rounds. 0 = off (the yes/no
#: default: those runs stopped by hand). Probability: 2 (Phil 2026-10-31; summary-r3 ran 2 rounds past this rule
#: because nothing enforced it).
STOP_AFTER_ROUNDS_WITHOUT_BEST = int(os.environ.get("STOP_AFTER_ROUNDS_WITHOUT_BEST", "2" if PROBABILITY else "0"))  # DECIDE
#: Dollar stops, enforced by run.py before each pass (0 = off; the yes/no runs were watched by hand). The run total
#: counts every patient session this run id paid for (the receipts) plus OPTIMIZER_SESSION_BUDGET_USD for each round
#: started (an upper bound for the optimizer). A round = its TRAIN and VAL passes' patient sessions. Probability:
#: stop past $200 a run or after a round over $50 (Phil 2026-10-31, spec).
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "200" if PROBABILITY else "0"))  # DECIDE
ROUND_BUDGET_USD = float(os.environ.get("ROUND_BUDGET_USD", "50" if PROBABILITY else "0"))  # DECIDE
OPTIMIZER_SESSION_BUDGET_USD = float(os.environ.get("OPTIMIZER_SESSION_BUDGET_USD", "3.0"))  # DECIDE
#: Claude Code --effort: at the default a Sonnet optimizer quit in 4-6 turns on recurrence (10-19).
OPTIMIZER_EFFORT = os.environ.get("OPTIMIZER_EFFORT", "high")  # DECIDE: Phil 2026-10-20
OPTIMIZER_TIMEOUT_SECONDS = 2700
#: The optimizer reads TRAIN records, so it never searches itself (Phil 2026-10-22): it asks the engine's search
#: proxy, a separate sealed session that sees only the request and the findings file and refuses anything
#: patient-specific. Spec planning/agentic-label-opt/2026-10-22-search-proxy.md.
OPTIMIZER_WEB_SEARCH = False  # DECIDE
SEARCH_PROXY = True  # DECIDE
SEARCH_REQUESTS_PER_ROUND = 1000  # DECIDE: no practical cap (Phil 2026-10-30; was 6, and the optimizer asked only 4 in 6 rounds)
#: Pass ``attempting_step_back: true`` in the change note to try a big change that is undone if VAL drops
#: by more than this. 0.0 means any drop.
STEP_BACK_REVERT_TOLERANCE = 0.0
TOKEN_BUDGET: int | None = None

# --------------------------------------------------------------------------------------------------
# Which examples each iteration sees (docs/reference/engine-internals.md, "Data schedule").
# --------------------------------------------------------------------------------------------------
#: The pools are the CT task's, copied by build_data.py: 400 TRAIN and 200 VAL patients at the natural death rate.
#: VAL 200 puts AUROC noise at about +-0.035 (Phil 2026-10-20). Stop and ask before an iteration
#: projected over $100 (Phil 2026-10-20).
#: TRAIN 100 every round, not a growing schedule: VAL is most of each round's cost, so a bigger TRAIN costs little and
#: gives the optimizer more cases to learn from. The chosen version is then scored once on all 811 VAL patients
#: (`run.py score-val`) for the reported number. The same as the CT task (Phil 2026-10-28).
SCHEDULE: dict | None = {"train_size": "constant(100)", "val_size": "constant(200)"}  # DECIDE
if PROBABILITY:  # TRAIN 100, 133, 167, ... 400 at round 10; VAL the same 200 (Phil 2026-10-31)
    SCHEDULE = {"train_size": "linear(100, 400, over=10)", "val_size": "constant(200)"}
if os.environ.get("TASK_SCHEDULE"):  # the same, as JSON, for one run: handy for a cheap live check
    import json as _json

    SCHEDULE = _json.loads(os.environ["TASK_SCHEDULE"])

# --------------------------------------------------------------------------------------------------
# Where everything a run writes goes. Outside the repo, so nothing a run writes is ever committed.
# --------------------------------------------------------------------------------------------------
#: Its own folder for the yes/no version: the old folder (alo-tasks/<task>) holds the probability-era runs and its
#: program-v0 is the probability seed, so reusing it would start a new run from the wrong instructions (2026-10-28).
#: Probability runs get their own folders too (program-v0 there is the probability seed), one per input arm.
_DEFAULT_WORKDIR = ((f"{TASK_NAME}-surv" if SURVIVAL else f"{TASK_NAME}-prob")
                    + ("-summaryonly" if SUMMARY_ONLY else "-summary" if WITH_SUMMARY else "")
                    if PROBABILITY else f"{TASK_NAME}-yesno")
WORKDIR = Path(os.environ.get("TASK_WORKDIR", Path.home() / ".cache" / "alo-tasks" / _DEFAULT_WORKDIR)).resolve()

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
CLAUDE_CODE_VERSION = "2.1.307"  # Vertex global verified from this image 2026-10-21
OPTIMIZER_IMAGE_TAG = f"alo-{TASK_NAME}-optimizer:{CLAUDE_CODE_VERSION}"
WORKER_IMAGE_TAG = f"alo-{TASK_NAME}-worker:3.11.0"
#: The route, one setting for the optimizer, the patient sessions and the search sessions (engine isolation.route).
#: The seat ("anthropic", the token above) is the default. ALO_ROUTE=vertex runs one run on Vertex. Vertex bills
#: the GCP project, not the seat, and reaches Google's API addresses, not api.anthropic.com, so it neither spends
#: the seat's session limit nor shares the Cloud NAT's 64 ports (Phil 2026-10-21). No service account can be made,
#: so on Vertex every session holds the machine's `optimizer` login (engine OPTIMIZER_LOGIN, the file crc and
#: rad-eval mount) and runs as its owner. That login can read the BigQuery labels: the network block (only the
#: Vertex model endpoint and Google sign-in) is the guard, and the cost check runs the engine's after-run audit
#: (scripts/audit_vertex_run.py) as the evidence (Phil 2026-10-28).
#: Both routes are approved for this de-identified data (Phil 2026-10-17, 10-22).
from isolation.route import OPTIMIZER_LOGIN, Route, route_name  # noqa: E402  (the engine is on sys.path above)

ROUTE = Route(route_name("anthropic"), vertex_project="som-nero-plevriti-deidbdf",
              vertex_region="global",  # Claude 5.5 is served on global only in this project (probed 2026-10-17)
              token_env_name=TOKEN_ENV_NAME, vertex_credential=OPTIMIZER_LOGIN)  # DECIDE
#: Patient data: turns on the engine's patient-data guards (fail-closed run store, Vertex only).
PATIENT_DATA = False  # DECIDE: de-identified, approved for the subscription route (Phil 2026-10-17), as died-2yr-ct
