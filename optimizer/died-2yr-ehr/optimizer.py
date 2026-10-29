"""The work folder, what the optimizer may never see, and the sealed sessions built from them.

Two containers run in a real run, both sealed by the engine:

    the optimizer   a Claude Code session per iteration that edits a copy of the program
    the worker      runs the program over one batch: no network, user nobody, nothing writable

plus, with ``config.SEARCH_PROXY``, the engine's search sessions: one per request the optimizer drops, seeing only
that request and the findings file (engine/search_proxy).

The forbidden list is the heart of the seal: every folder whose contents would let either container
see the answers. The engine refuses to mount anything on or under these paths, and the setup check
fingerprints both containers so a change to either stops the run before it spends anything.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import config
from config import DATA_DIR, HERE, TASK_NAME, WORKDIR

from engine.loop import AgentOutcome
from isolation.session_scope import Forbidden

import optimizer_prompt as op


@dataclass(frozen=True)
class Layout:
    """Everything a run writes, under one work folder outside the repo. ``private/`` is never mounted
    into either container; the rest is mounted read-only, or not at all."""

    root: Path = WORKDIR

    @property
    def repo(self) -> Path:  # the git repo the engine commits each program version to
        return self.root / "repo"

    @property
    def versions(self) -> Path:  # frozen versions, materialized as plain folders (read-only)
        return self.root / "versions"

    @property
    def feedback(self) -> Path:  # TRAIN mistake files; copied into the optimizer's feedback folder
        return self.root / "feedback"

    @property
    def notes(self) -> Path:  # every change note and notebook the optimizer wrote, per run and iteration
        return self.root / "notes"

    @property
    def notebook(self) -> Path:  # the optimizer's running notebook
        return self.root / "notebook.md"

    @property
    def worker_dir(self) -> Path:  # a copy of worker.py, mounted read-only into the worker
        return self.root / "worker"

    @property
    def inputs(self) -> Path:  # copies of the files the optimizer reads (the task brief)
        return self.root / "inputs"

    @property
    def stubs(self) -> Path:  # what the run-start reset puts back after archiving
        return self.root / "stubs"

    @property
    def program_outputs(self) -> Path:  # agentic mode: each pass's sessions and answers (ProgramRunner)
        return self.root / "program-outputs"

    @property
    def search(self) -> Path:  # the search proxy's drop folder, findings file (kept across runs) and requests
        return self.root / "search"

    @property
    def private(self) -> Path:
        return self.root / "private"

    @property
    def search_logs(self) -> Path:  # every search request, its queries, reply and cost, per round
        return self.private / "search-logs"

    @property
    def outputs(self) -> Path:  # the program's answers, per pass
        return self.private / "outputs"

    @property
    def batches(self) -> Path:
        return self.private / "batches"

    @property
    def store(self) -> Path:  # program snapshots and which versions failed
        return self.private / "store"

    @property
    def archive(self) -> Path:  # earlier runs' state, moved aside at each fresh start
        return self.private / "archive"

    @property
    def transcripts(self) -> Path:
        return self.private / "transcripts"

    @property
    def test_ledger(self) -> Path:  # every time TEST was opened, and for which version
        return self.private / "test_ledger.jsonl"

    def summary(self, run_id: str) -> Path:
        return self.private / "runs" / run_id / "run_summary.json"


def optimizer_forbidden(layout: Layout) -> tuple[Forbidden, ...]:
    """The optimizer's list: everything in ``forbidden``, plus the agentic program's output folder, which
    holds its answers. Not on the program's own list: each call's writable folder is inside it, and the
    engine's runner already denies every call the rest of it."""
    return (*forbidden(layout), Forbidden(layout.program_outputs, representatives=("archive",)))


def forbidden(layout: Layout) -> tuple[Forbidden, ...]:
    """What neither container may see. Each entry names a representative file inside it, which the
    engine's seal proof plants a marker beside. Add an entry for anything else that holds answers."""
    entries = [
        # This task's code and data: the scorer, the gold labels, TEST. Nothing in here is mounted, not even
        # the task brief: stage_files copies the two files the containers need into the work folder.
        Forbidden(HERE, representatives=("adapters.py", "TASK_NOTES.md")),
        Forbidden(layout.private, representatives=("README.txt",)),
        Forbidden(layout.repo / ".git", representatives=("HEAD",)),
        Forbidden(layout.repo / "iter", representatives=("README.txt",)),
        Forbidden(Path.home() / ".claude", representatives=("settings.json",)),
        # Every vista_bench label table is one BigQuery query away with this credential.
        Forbidden(Path.home() / ".config" / "gcloud", representatives=("application_default_credentials.json",)),
        # The whole shared mount: this task's data (splits with labels, pools, the record archive), the CT task's
        # id map and this task's cohort file, and every label snapshot and result.
        Forbidden(Path("/mnt/su-vista-uscentral1"), representatives=(
            str((config.MOUNT_DATA_DIR / "splits" / "test.jsonl").relative_to("/mnt/su-vista-uscentral1")),
            str((config.ID_MAP_DIR / "id_map.jsonl").relative_to("/mnt/su-vista-uscentral1")))),
    ]
    if not DATA_DIR.is_relative_to("/mnt/su-vista-uscentral1"):
        # The local working copy (splits with labels, every patient's record): records reach a session only by
        # copy (agentic.stage), TRAIN mistakes only through the feedback folder.
        entries.append(Forbidden(DATA_DIR, representatives=("splits/test.jsonl",)))
    return tuple(entries)


# ==================================================================================================
# The sealed worker (program mode rules_based)
# ==================================================================================================


def worker_spec_for(layout: Layout, *, image: str, pin: str):
    from isolation.program_mode import ProgramMode
    from isolation.sealed_worker import SealedWorkerSpec

    def spec_for(tree: Path):
        return SealedWorkerSpec(
            mode=ProgramMode(config.PROGRAM_MODE), image=image, program_dir=Path(tree), forbidden=forbidden(layout),
            python="python", args=("/workspace/worker/worker.py",),
            ro_mounts=((layout.worker_dir, "/workspace/worker"),), expected_fingerprint=pin,
        )

    return spec_for


# ==================================================================================================
# The optimizer
# ==================================================================================================


class ChangeNoteAgent:
    """Reads ``attempting_step_back`` from the change note the session filed, so the loop knows a
    declared big change from an ordinary one. Everything else passes straight through."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def setup_description(self):
        return self.inner.setup_description()

    def run(self, *, iter_n: int, materialized_path: Path | None = None) -> AgentOutcome:
        outcome = self.inner.run(iter_n=iter_n, materialized_path=materialized_path)
        if outcome.exit_code != 0 or not outcome.notes_dir:
            return outcome
        try:
            note = json.loads((Path(outcome.notes_dir) / op.CHANGE_NOTE_NAME).read_text(encoding="utf-8"))
            step_back = bool(note.get("attempting_step_back")) if isinstance(note, dict) else False
        except (OSError, json.JSONDecodeError):
            step_back = False
        return dataclasses.replace(outcome, attempting_step_back=step_back)


#: The optimizer's container variables, shared with the search sessions (the same route, Decided 2026-10-22).
#: Running the program writes __pycache__ into the program folder, and any file outside the editable list refuses
#: the whole edit and stops the run. This keeps Python from writing it.
ENV = (("PYTHONDONTWRITEBYTECODE", "1"),)


def search_proxy(layout: Layout, *, image: str, pin: str, token: str | None):
    """The engine's search proxy on the optimizer's route, or None when ``config.SEARCH_PROXY`` is off. ``pin`` is
    the committed ``search`` pin (or ``UNPINNED`` while printing one). Its first pass reads the staged task brief
    (no patient data) to pick literature topics before round 1."""
    if not config.SEARCH_PROXY:
        return None
    from engine.search_proxy import SearchLimits, SearchProxy

    prof, credential = config.ROUTE.session_auth()
    return SearchProxy(
        work_root=layout.search, log_root=layout.search_logs, forbidden=optimizer_forbidden(layout),
        profile=prof, credential=credential, image=image, expected_fingerprint=pin,
        model=config.PROGRAM_MODEL,  # Sonnet 5.5, the same id on both routes
        host_env={config.TOKEN_ENV_NAME: token} if token and not config.ROUTE.is_vertex else None, env=ENV,
        limits=SearchLimits(requests_per_round=config.SEARCH_REQUESTS_PER_ROUND),
        brief_path=layout.inputs / op.TASK_BRIEF_NAME,
    )


def build_agent(layout: Layout, store, *, run_id: str, image: str, pin: str, token: str | None,
                search_pin: str):
    """The sealed optimizer, wrapped in the search proxy when it is on (the wrapper's ``proxy`` attribute is the
    ``search`` setup owner). ``pin`` and ``search_pin`` are the committed setup pins (or ``UNPINNED``)."""
    from engine.search_proxy import ProxiedOptimizer
    from isolation.contained_agent import ContainedOptimizerAgent, DeclaredOutput

    prof, credential = config.ROUTE.session_auth()
    proxy = search_proxy(layout, image=image, pin=search_pin, token=token)
    agent = ContainedOptimizerAgent(
        model=config.OPTIMIZER_MODEL,
        system_prompt=op.system_prompt(),
        prompt_builder=op.build_prompt,
        manifest=store.manifest(),
        manifest_for_version=lambda sha: store.manifest(),
        repo_root=layout.repo,
        materialize_root=layout.versions,
        iter_root=layout.repo / "iter",
        forbidden=optimizer_forbidden(layout),
        profile=prof,
        credential=credential,
        image=image,
        expected_fingerprint=pin,
        transcript_dir=layout.transcripts / "optimizer",
        readable_inputs=((op.TASK_BRIEF_NAME, layout.inputs / op.TASK_BRIEF_NAME), (op.NOTEBOOK_NAME, layout.notebook)),
        declared_outputs=(DeclaredOutput(op.CHANGE_NOTE_NAME), DeclaredOutput(op.NOTEBOOK_NAME, destination=layout.notebook)),
        notes_root=layout.notes,
        run_id=run_id,
        train_feedback_roots=(layout.feedback,),
        validation_roots=(layout.private, DATA_DIR),
        run_summary_path=layout.summary(run_id),
        max_budget_usd=config.OPTIMIZER_SESSION_BUDGET_USD,
        effort=config.OPTIMIZER_EFFORT,
        web_search=config.OPTIMIZER_WEB_SEARCH,
        search_grant=proxy.grant if proxy else None,
        timeout_seconds=config.OPTIMIZER_TIMEOUT_SECONDS,
        host_env={config.TOKEN_ENV_NAME: token} if token and not config.ROUTE.is_vertex else None,
        env=ENV,
        patient_data=config.PATIENT_DATA,
    )
    return ProxiedOptimizer(agent, proxy) if proxy else agent


def stage_files(layout: Layout) -> None:
    """Copy the two files the containers need out of the task folder, which is forbidden to both:
    worker.py for the worker, the task brief for the optimizer. Run before every run, so an edit to
    either is picked up. The engine refuses to mount anything from inside a forbidden folder."""
    layout.worker_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "worker.py", layout.worker_dir / "worker.py")
    layout.inputs.mkdir(parents=True, exist_ok=True)
    brief = config.TASK_BRIEF.read_text(encoding="utf-8")
    if config.AGENTIC:
        # The optimizer reads the same data map every session gets, verbatim, at the end of its brief.
        from agentic import GUIDE, GUIDE_SRC
        brief += f"\n\n---\n\n# Appendix: `{GUIDE}`, as every session sees it\n\n" + GUIDE_SRC.read_text(encoding="utf-8")
    if config.WITH_SUMMARY:
        brief += (
            "\n\n---\n\n# This run: each record also holds `summary.txt`\n\n"
            "Every session's record folder also holds `summary.txt`: a summary of the same record, written earlier "
            "by a language model and dated at the same cutoff. It can be wrong or leave things out. Your program may "
            "use it however helps (read it first, trust it, check it against the record, or ignore it). A TRAIN "
            "mistake's copy is `<id>-summary.txt`, beside its other record files.\n")
    (layout.inputs / op.TASK_BRIEF_NAME).write_text(brief, encoding="utf-8")

