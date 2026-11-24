"""Every decision for the ct-findings-5 task, in one place: from one CT, a 0-100 probability for each of the five
CT findings questions the paper's Fig 5 asked every VLM (vista_bench v1.7 ``per_ct_diagnostic_tasks``: a non-benign
finding in the liver, the lungs and the lymph nodes; a malignant or possibly malignant lesion in the right upper
lobe; non-infectious pneumonitis), one sealed Claude Code session per scan answering all five.

Spec: planning/vista_eval_vlm/2026-11-10-ct-agent-probability-climbs.md (Approved 2026-11-10). Built from
died-2yr-ct's probability mode (its scan staging, slice viewer, route, stop rules and launcher); probability only,
there is no yes/no version of this task. Lines marked ``DECIDE`` name who decided; TASK_NOTES.md has the table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

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
TASK_NAME = "ct-findings-5"  # names the work folder and images
LABELS = (0, 1)
#: The five questions (Phil 2026-11-10): (answer key in output.json and gold field in splits/*.jsonl, vista_bench task,
#: the question as the VLM eval-set card asks it: vista_eval_vlm main src/evalset/cards/lesions.py). A scan whose
#: label is -1 or missing for a question has gold None there: unscored for that question, never negative.
QUESTIONS = (
    ("lesion_liver", "per_ct_lesion_liver", "Is there a non-benign finding in the liver?"),
    ("lesion_lungs", "per_ct_lesion_lungs", "Is there a non-benign finding in the lungs?"),
    ("lesion_lymph_nodes", "per_ct_lesion_lymph_nodes", "Is there a non-benign finding in the lymph nodes?"),
    ("lung_lobe_rul", "per_ct_lung_lobe_rul", "Is there a malignant or possibly malignant lung lesion in the right "
                                              "upper lobe? (The right apex counts as the right upper lobe.)"),
    ("pneumonitis", "per_ct_pneumonitis", "Does this scan show non-infectious pneumonitis?"),
)
#: Built by scripts/build_data.py: splits/{train,val}.jsonl ({"id", <five keys>: 0 | 1 | null}, the pool scans only),
#: pools.json (the fixed enriched TRAIN 400 and VAL 200, with the draw seed and enrichment under "_draw") and
#: patients/<id>/{ct.npz, meta.json}. splits/test.jsonl (the VLMs' 1,000 TEST scans) only after Phil's go 2026-11-16
#: (build_data.py test); run.py test is its only reader. On the shared mount (secure environment).
#: The optimizer never sees this folder. It sees TRAIN only through its mistakes, and VAL only as a score.
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7"))
#: The id -> person_id, accession and NIfTI map: never mounted, listed in the forbidden set.
ID_MAP_DIR = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-private")
PROGRAM_MODE = "agentic"
AGENTIC = True
#: Kept for the code shared with died-2yr-ct: this task only has the probability answer.
ANSWER = "probability"
PROBABILITY = True

#: The labeling program the optimizer starts from (program-v0), and the files it may edit. Pure zero-shot (spec,
#: Decided): the five questions as the VLM cards ask them, the folder layout and the starter slice viewer; no clinical
#: hints, no tools from the death climb.
SEED_PROGRAM_DIR = HERE / "program_agentic"
#: The retrieval arm (spec planning/vista_eval_vlm/2026-11-17-ct-findings-retrieval-arm.md, Approved 2026-11-17):
#: NEIGHBOURS=1 stages each scan's 5 MerlinOnc-nearest TRAIN patients (scripts/build_neighbours.py) beside its CT and
#: starts from program_agentic_neighbours/, the untuned seed plus one paragraph. Its own work folder (below), so its
#: passes never mix with the climbs'. The optimizer brief is unchanged: this arm is one untuned VAL pass, no climb.
NEIGHBOURS = os.environ.get("NEIGHBOURS", "0") == "1"  # DECIDE: spec, an on/off switch
NEIGHBOURS_DIR = DATA_DIR / "neighbours"  # per VAL id: README.md, neighbours.json, n1..n5/{answers.json, report.txt}
NEIGHBOUR_SCANS_DIR = DATA_DIR / "neighbour_scans"  # per neighbour id: ct.npz, meta.json
NEIGHBOUR_PICKS = ID_MAP_DIR / "neighbour_picks.json"  # private: which neighbour id fills which folder
if NEIGHBOURS:
    SEED_PROGRAM_DIR = HERE / "program_agentic_neighbours"
#: The example-CT arms (spec planning/vista_eval_vlm/2026-11-22-ct-findings-example-ct-arms.md, Approved 2026-11-22):
#: EXAMPLE_ARM=A|B|C uses the same 5 neighbours as the retrieval arm. A: must view at least one slice of each
#: neighbour's CT; B: the same folders without the CTs (the image control); C: A plus TotalSegmentator organ ranges
#: for the scan and each neighbour (scripts/build_organ_ranges.py), scored only on the VAL scans that have them.
#: D (Phil 2026-11-23): C plus pre-rendered organ-targeted grids (lungs, liver, chest soft tissue) of the scan and of
#: every neighbour (scripts/build_views.py); must open all of them and learn from the neighbours' images and reports.
EXAMPLE_ARM = os.environ.get("EXAMPLE_ARM", "").upper()  # DECIDE: spec, the arms
if EXAMPLE_ARM not in ("", "A", "B", "C", "D"):
    raise SystemExit(f"EXAMPLE_ARM must be A, B, C or D, not {EXAMPLE_ARM!r}")
ORGAN_ARMS = ("C", "D")  # stage organs.json and score only the VAL scans with organ ranges
if EXAMPLE_ARM:
    NEIGHBOURS = True
    SEED_PROGRAM_DIR = HERE / f"program_agentic_examples_{EXAMPLE_ARM.lower()}"
ORGAN_DIR = DATA_DIR / "organ_ranges"  # per VAL id: target.json, n1.json .. n5.json
#: Each neighbour CT is copied here once and hard-linked into every call folder that uses it, so a pass holds one
#: copy per distinct neighbour (~48 GB) instead of five per call (the 11-17 pass used ~300 MB a call). Must be on the
#: same filesystem as WORKDIR.
NEIGHBOUR_CT_CACHE = Path(os.environ.get("NEIGHBOUR_CT_CACHE", Path.home() / ".cache" / "alo-tasks" / "neighbour-ct-cache"))
#: Arm D's grids (scripts/build_views.py), hard-linked into call folders like the CT cache, so the same disk.
EXAMPLE_VIEWS = Path(os.environ.get("EXAMPLE_VIEWS", NEIGHBOUR_CT_CACHE.parent / "example-views"))
VIEW_GRIDS = ("lungs.png", "liver.png", "chest_soft.png", "views.json")
EDITABLE_FILES = ("instructions.md", "tools")
CONTRACT_FILES: tuple[str, ...] = ()
#: The knowledge-base pilot (spec planning/vista_eval_vlm/2026-11-23-ct-knowledge-base-pilot.md, Approved 2026-11-23).
#: KB_ARM=1 runs the tester program program_agentic_kb/ (the untuned seed's words, the same viewer, the knowledge-base
#: tool tools/kb.py and library/). Its builder (scripts/run_kb_loop.py, a Sonnet session) may edit library/ only: the
#: tester's instructions and tools are contract files, fixed for every version. Its program-v0 has an empty library, so
#: it is also the paired empty-library arm (spec, Decided: same tester, same tools, no entries). Each scan is staged with
#: organs.json and kb_similarity.json (scripts/build_kb_data.py); scored only on VAL scans with organ ranges (the 190).
KB_ARM = os.environ.get("KB_ARM", "0") == "1"  # DECIDE: spec, an on/off switch
KB_DIR = DATA_DIR / "kb"  # rounds.json, source.json, splits/kb_train.jsonl, organs/, similarity/, source_reports/
if KB_ARM:
    if EXAMPLE_ARM or NEIGHBOURS:
        raise SystemExit("KB_ARM runs alone: unset EXAMPLE_ARM and NEIGHBOURS")
    SEED_PROGRAM_DIR = HERE / "program_agentic_kb"
    EDITABLE_FILES = ("library",)
    CONTRACT_FILES = ("instructions.md", "tools")
#: Forced-image re-test (Phil 2026-11-24): KB_PRESTAGE=1 draws, before each session, the 2 most similar entries for each
#: target beside the same region of the scan into examples/ (kb_prestage.py), from the library of the version being
#: scored, KB_PRESTAGE_TAG (run.py val-pass writes it out to versions/val-pass-<tag>). The pilot's tester opened the
#: saved image in only 61 of 381 opens.
KB_PRESTAGE = KB_ARM and os.environ.get("KB_PRESTAGE", "0") == "1"  # DECIDE: Phil 2026-11-24
KB_PRESTAGE_TAG = os.environ.get("KB_PRESTAGE_TAG", "")
if KB_PRESTAGE and not KB_PRESTAGE_TAG:
    raise SystemExit("KB_PRESTAGE=1 needs KB_PRESTAGE_TAG, the program version whose library is drawn")
#: Oracle-examples Half A (spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md, Approved 2026-11-24):
#: KB_ORACLE=grid | zoom draws each scan's report-matched sets (kb_oracle.py; built once into the private folder) in
#: place of the similarity picks; needs KB_PRESTAGE=1 and its KB_PRESTAGE_TAG (the version whose library is used).
KB_ORACLE = os.environ.get("KB_ORACLE", "") if KB_PRESTAGE else ""  # DECIDE: Phil 2026-11-24, two arms
if KB_ORACLE not in ("", "grid", "zoom"):
    raise SystemExit(f"KB_ORACLE must be grid or zoom, not {KB_ORACLE!r}")
if KB_ARM:
    import kb_materialize  # the library holds PNGs; the pinned engine writes versions out as text (2026-11-23)

    kb_materialize.install()
#: The optimizer's task brief and its standing instructions (its system prompt).
TASK_BRIEF = HERE / "context" / "task.md"
OPTIMIZER_INSTRUCTIONS = HERE / "prompts" / "optimizer_instructions.md"

#: The model each scan's session runs on, how many run at once, and each one's cap (spec: as the death climb).
PROGRAM_MODEL = os.environ.get("PROGRAM_MODEL", "claude-sonnet-5-5")  # DECIDE: Phil 2026-11-10
PROGRAM_MAX_CONCURRENT = int(os.environ.get("PROGRAM_MAX_CONCURRENT", "2"))  # DECIDE: spec, 2 at a time
PROGRAM_CALL_BUDGET_USD = float(os.environ.get("PROGRAM_CALL_BUDGET_USD", "0.50"))  # DECIDE: spec, $0.50 a scan
PROGRAM_CALL_TIMEOUT_SECONDS = 900

# --------------------------------------------------------------------------------------------------
# The metric: one number, higher is better. adapters.TaskScorer computes it.
# --------------------------------------------------------------------------------------------------
#: AUROC per question over the scans scored for it (a failed or unreadable answer counts 0.5), averaged over the five
#: (Phil 2026-11-10: climb score = mean AUROC; each also reported alone).
METRIC_NAME = "auroc"  # DECIDE

# --------------------------------------------------------------------------------------------------
# The optimizer and the climb's stop rules (spec: the death climb's rules)
# --------------------------------------------------------------------------------------------------
OPTIMIZER_MODEL = os.environ.get("OPTIMIZER_MODEL", "claude-opus-5-5")  # DECIDE: Phil 2026-11-10
ITERATIONS = int(os.environ.get("ITERATIONS", "10"))  # DECIDE: spec
#: Stop once this many rounds in a row set no new best VAL score (the seed's zero-shot VAL is the first best).
#: Enforced by run.py before the next pass, so a resume counts the same rounds. 0 = off.
STOP_AFTER_ROUNDS_WITHOUT_BEST = int(os.environ.get("STOP_AFTER_ROUNDS_WITHOUT_BEST", "2"))  # DECIDE: spec
#: Dollar stops, enforced by run.py before each pass (0 = off): the run total counts every scan session this run id
#: paid for plus OPTIMIZER_SESSION_BUDGET_USD for each round started; a round = its TRAIN and VAL passes' sessions.
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "200"))  # DECIDE: spec
ROUND_BUDGET_USD = float(os.environ.get("ROUND_BUDGET_USD", "50"))  # DECIDE: spec
OPTIMIZER_SESSION_BUDGET_USD = float(os.environ.get("OPTIMIZER_SESSION_BUDGET_USD", "3.0"))  # as died-2yr-ct
OPTIMIZER_EFFORT = os.environ.get("OPTIMIZER_EFFORT", "high")  # as died-2yr-ct
OPTIMIZER_TIMEOUT_SECONDS = 2700
#: The optimizer reads TRAIN scans, so it never searches itself: it asks the engine's search proxy, a separate sealed
#: session that sees only the request and the findings file (as died-2yr-ct).
OPTIMIZER_WEB_SEARCH = False
SEARCH_PROXY = True
SEARCH_REQUESTS_PER_ROUND = 1000  # DECIDE: spec, no practical cap
STEP_BACK_REVERT_TOLERANCE = 0.0
TOKEN_BUDGET: int | None = None

# --------------------------------------------------------------------------------------------------
# Which scans each iteration sees (docs/reference/engine-internals.md, "Data schedule").
# --------------------------------------------------------------------------------------------------
#: The pools are fixed by build_data.py: TRAIN 400 and VAL 200 scans, one per patient, enriched for positives.
#: TRAIN 100, 133, 167, ... 400 at round 10; VAL the same 200 every round (spec).
SCHEDULE: dict | None = {"train_size": "linear(100, 400, over=10)", "val_size": "constant(200)"}  # DECIDE: spec
if os.environ.get("TASK_SCHEDULE"):  # the same, as JSON, for one run: the cost check uses it
    import json as _json

    SCHEDULE = _json.loads(os.environ["TASK_SCHEDULE"])

# --------------------------------------------------------------------------------------------------
# Where everything a run writes goes. Outside the repo, so nothing a run writes is ever committed.
# --------------------------------------------------------------------------------------------------
WORKDIR = Path(os.environ.get("TASK_WORKDIR", Path.home() / ".cache" / "alo-tasks" /
                              (f"{TASK_NAME}-kb" if KB_ARM else f"{TASK_NAME}-examples-{EXAMPLE_ARM.lower()}" if EXAMPLE_ARM
                               else f"{TASK_NAME}-neighbours" if NEIGHBOURS else TASK_NAME))).resolve()

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
#: The seat ("anthropic", the token above) is the default. ALO_ROUTE=vertex runs one run on Vertex. Vertex bills
#: the GCP project, not the seat, and reaches Google's API addresses, not api.anthropic.com, so it neither spends
#: the seat's session limit nor shares the Cloud NAT's 64 ports (Phil 2026-10-21). No service account can be made,
#: so on Vertex every session holds the machine's `optimizer` login (engine OPTIMIZER_LOGIN, the file crc and
#: rad-eval mount) and runs as its owner. That login can read the BigQuery labels: the network block (only the
#: Vertex model endpoint and Google sign-in) is the guard, and the cost check runs the engine's after-run audit
#: (scripts/audit_vertex_run.py) as the evidence (Phil 2026-10-28).
#: Both routes are approved for this de-identified data, CT slice images included (Phil 2026-10-17, 10-20, 10-22).
from isolation.route import OPTIMIZER_LOGIN, Route, route_name  # noqa: E402  (the engine is on sys.path above)

ROUTE = Route(route_name("anthropic"), vertex_project="som-nero-plevriti-deidbdf",
              vertex_region="global",  # Claude 5.5 is served on global only in this project (probed 2026-10-17)
              token_env_name=TOKEN_ENV_NAME, vertex_credential=OPTIMIZER_LOGIN)  # DECIDE
#: Patient data: turns on the engine's patient-data guards (fail-closed run store, Vertex only).
PATIENT_DATA = False  # DECIDE: de-identified, approved for the subscription route (Phil 2026-10-17, 10-20)
