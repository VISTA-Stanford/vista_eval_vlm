"""Every decision for the pneumonitis-variants task, in one place: from one chest CT whose lungs show pneumonitis-like
change, a 0-100 probability for each of five causes (radiation, immunotherapy, other drug, organizing pneumonia,
infection), one sealed Claude Code session (the reader, Sonnet) per scan. A builder session (Opus, with the engine's
search proxy) writes the reader's guide and example library over a 5-round loop on TRAIN (scripts/run_loop.py); the
reader's instructions and tools never change. program-v0 has an empty guide and library: the zero-shot arm.

Spec: planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md (Approved 2026-11-24). Copied from ct-findings-5
(feat/ct-organ-zoom @6c1d699): its scan staging, organ-zoom tool, knowledge-base builder loop, pre-drawn examples and
launcher, with the arm switches removed. Lines marked ``DECIDE`` name who decided.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
import paths  # noqa: E402

# --------------------------------------------------------------------------------------------------
# The engine (as ct-findings-5)
# --------------------------------------------------------------------------------------------------
ENGINE_ROOT = Path(os.environ.get("AGENTIC_LABEL_OPT") or Path.home() / "code" / "agentic-label-opt" / ".claude" / "worktrees" / "mortality-engine").resolve()
if not (ENGINE_ROOT / "engine" / "driver.py").is_file():
    raise SystemExit(f"no engine at {ENGINE_ROOT}: set AGENTIC_LABEL_OPT to your agentic-label-opt checkout")
ENGINE_PIN = "bb216c2489ff63bebd8ad3f1acdcb9f969bb0e2f"  # DECIDE: as ct-findings-5
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

# --------------------------------------------------------------------------------------------------
# The task
# --------------------------------------------------------------------------------------------------
TASK_NAME = "pneumonitis-variants"
LABELS = (0, 1)
#: The five variants (spec, Decided): (answer key and gold field in splits/*.jsonl, benchmark source, the question).
#: Every scan has exactly one yes. Wording repeated in program_agentic/instructions.md.
QUESTIONS = (
    ("radiation", "raw_subtype radiation", "Is the lung change radiation pneumonitis or radiation fibrosis?"),
    ("immunotherapy", "raw_subtype drug_induced_ici", "Is the lung change pneumonitis from an immune checkpoint "
                                                      "inhibitor (immunotherapy)?"),
    ("other_drug", "raw_subtype drug_induced_other", "Is the lung change drug-induced pneumonitis from a drug other "
                                                     "than immunotherapy?"),
    ("cop", "raw_subtype organizing_pneumonia_cop", "Is the lung change organizing pneumonia?"),
    ("infection", "infection pass yes or model label infectious", "Is the lung change an infection (pneumonia)?"),
)
#: scripts/build_data.py: splits/{train,val}.jsonl, pools.json, rounds.json, groups.json, embeddings.npz,
#: patients/<id>/{ct.npz, meta.json}, masks/<id>.npz, organs/<id>.json, train_reports/<id>.txt, probe/. On the mount.
DATA_DIR = paths.DATA_DIR
ID_MAP_DIR = paths.PRIVATE_DIR
PROGRAM_MODE = "agentic"
AGENTIC = True
ANSWER = "probability"
PROBABILITY = True

SEED_PROGRAM_DIR = HERE / "program_agentic"
#: The builder edits the guide and the example library only; the reader's instructions and tools are fixed for every
#: version (the engine refuses an edit to a contract file), so every version differs only in what it knows.
EDITABLE_FILES = ("guide", "examples")
CONTRACT_FILES = ("instructions.md", "tools")
#: The atlas arm (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-atlas-arm.md, Approved 2026-11-25): ATLAS=1
#: runs program_agentic_atlas/: the same reader tools plus tools/atlas.py, and in place of the similarity-picked
#: examples a fixed atlas of ATLAS_PER_VARIANT marked examples per cause (atlas/), every one staged beside every scan
#: (stage_examples.stage_atlas). A literature review comes first (scripts/run_loop.py, round 0). Own work folder.
ATLAS = os.environ.get("ATLAS", "0") == "1"  # DECIDE: spec, an on/off switch
ATLAS_PER_VARIANT = 10  # DECIDE: Phil 2026-11-25
#: The literature review seeded into the atlas arm's program-v0 as guide/literature.md (run.init_layout copies it in):
#: an agent with working web search wrote it, 52 sources, PubMed ids checked against their titles (2026-11-25). The
#: in-loop search proxy is off for this arm: on Vertex every search fails (VPCSC), and the builder took model memory
#: for literature (Phil 2026-11-25: "this is fine for now"; a deeper review is handed to the Mac).
LITERATURE_SEED = paths.MOUNT / "planning" / "vista_eval_vlm" / "session" / "2026-11-25-pneumonitis-literature-review.md"
SEED_EXTRA: tuple = ()
if ATLAS:
    SEED_PROGRAM_DIR = HERE / "program_agentic_atlas"
    EDITABLE_FILES = ("guide", "atlas")
    SEED_EXTRA = (("guide/literature.md", LITERATURE_SEED),)
#: Examples drawn beside each scan before its session (stage_examples.py): per variant, this many entries.
EXAMPLES_PER_VARIANT = 2  # DECIDE: spec, 2 per variant, 10 images
import kb_materialize  # noqa: E402  the library holds PNGs; the pinned engine writes versions out as text

kb_materialize.install()

#: The builder's brief and standing instructions (scripts/run_loop.py stages the brief).
TASK_BRIEF = HERE / "context" / ("task_atlas.md" if ATLAS else "task.md")
OPTIMIZER_INSTRUCTIONS = HERE / "prompts" / ("builder_instructions_atlas.md" if ATLAS else "builder_instructions.md")

#: The reader: Sonnet, 3 at a time, $0.75 a scan (spec, Decided), 30 minutes (10 example images plus the zoom tool).
PROGRAM_MODEL = os.environ.get("PROGRAM_MODEL", "claude-sonnet-5-5")  # DECIDE: spec
PROGRAM_MAX_CONCURRENT = int(os.environ.get("PROGRAM_MAX_CONCURRENT", "3"))
PROGRAM_CALL_BUDGET_USD = float(os.environ.get("PROGRAM_CALL_BUDGET_USD", "0.75"))  # DECIDE: spec
PROGRAM_CALL_TIMEOUT_SECONDS = 1800

#: AUROC per variant, one-vs-rest, over every scored scan (a failed answer counts 0.5), averaged over the five.
METRIC_NAME = "auroc"

# --------------------------------------------------------------------------------------------------
# The builder (the engine's optimizer session, driven by scripts/run_loop.py, not the engine's climb)
# --------------------------------------------------------------------------------------------------
OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "claude-opus-5-5")  # DECIDE: spec, Opus builder
ITERATIONS = int(os.environ.get("ITERATIONS", "5"))  # DECIDE: spec, 5 rounds
STOP_AFTER_ROUNDS_WITHOUT_BEST = 0  # the loop runs all 5 rounds; only round 5 is scored (spec)
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "300" if ATLAS else "250"))  # DECIDE: specs, $250 / atlas $300
ROUND_BUDGET_USD = float(os.environ.get("ROUND_BUDGET_USD", "60"))
OPTIMIZER_SESSION_BUDGET_USD = float(os.environ.get("OPTIMIZER_SESSION_BUDGET_USD", "15"))
OPTIMIZER_EFFORT = os.environ.get("OPTIMIZER_EFFORT", "high")
#: The atlas builder writes 10 marked write-ups a cause and the literature review: 2 hours (build decision).
OPTIMIZER_TIMEOUT_SECONDS = 7200 if ATLAS else 3600
#: The builder reads TRAIN scans, so it never searches itself: the search proxy, a separate sealed session that sees
#: only the request (Radiopaedia and the literature as text and links; spec, Decided).
OPTIMIZER_WEB_SEARCH = False
SEARCH_PROXY = not ATLAS  # off for the atlas arm (see LITERATURE_SEED)
#: Atlas: up to 80 every round (the spec's round-0 literature budget; a ceiling, kept the same every round so the
#: sealed setup does not change between rounds).
SEARCH_REQUESTS_PER_ROUND = 80 if ATLAS else 40  # DECIDE: build
STEP_BACK_REVERT_TOLERANCE = 0.0
TOKEN_BUDGET: int | None = None
#: TRAIN rounds come from rounds.json (80 a round); VAL is every VAL scan. The engine's schedule is not used.
SCHEDULE: dict | None = None

WORKDIR = Path(os.environ.get("TASK_WORKDIR", Path.home() / ".cache" / "alo-tasks" /
                              (f"{TASK_NAME}-atlas" if ATLAS else TASK_NAME))).resolve()

# --------------------------------------------------------------------------------------------------
# Sealing (as ct-findings-5): route, images
# --------------------------------------------------------------------------------------------------
TOKEN_ENV_NAME = "CLAUDE_CODE_OAUTH_TOKEN"
TOKEN_FILE = Path(os.environ.get("ALO_TOKEN_FILE", Path.home() / ".config" / "alo" / "oauth_token"))
CLAUDE_CODE_VERSION = "2.1.307"
#: ct-findings-5's optimizer image (same Dockerfile, already built on phil-sllm-02): numpy, scipy, pillow, nibabel.
OPTIMIZER_IMAGE_TAG = f"alo-ct-findings-5-optimizer:{CLAUDE_CODE_VERSION}"
WORKER_IMAGE_TAG = f"alo-{TASK_NAME}-worker:3.11.0"
from isolation.route import OPTIMIZER_LOGIN, Route, route_name  # noqa: E402

ROUTE = Route(route_name("anthropic"), vertex_project="som-nero-plevriti-deidbdf", vertex_region="global",
              token_env_name=TOKEN_ENV_NAME, vertex_credential=OPTIMIZER_LOGIN)
#: De-identified data, approved for both routes (as ct-findings-5); this task runs on Vertex (ALO_ROUTE=vertex).
PATIENT_DATA = False
