"""The four adapters the engine calls, plus the data source. This is the task-specific code.

    TaskStore          which files make up the program, and which the optimizer may edit
    LocalRunner        runs a frozen program version over one batch, in this process (dry run only)
    SealedRunner       the same, inside the sealed worker container (every real run)
    TaskScorer         turns the program's answers into one number, plus a breakdown
    mistake_corpus     writes the TRAIN mistakes the optimizer reads
    TaskReleaseBuilder what the optimizer is shown each iteration: TRAIN in detail, VAL as a score only
    TaskSource         the data schedule's view of your data (only when config.SCHEDULE is set)

What a new task changes: ``read_split`` (your data format), ``TaskScorer`` (your metric), and what
``mistake_corpus`` writes per mistake. The runners and the release builder rarely change.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import config  # puts the engine on sys.path
from engine.schemas import (
    ArtifactRef,
    ErrorInfo,
    ManifestEntry,
    MistakeCorpus,
    PrimaryMetric,
    ProgramManifest,
    ReleasePayloadTrain,
    ReleasePayloadVal,
    RunArtifacts,
    RunInputs,
    ScoreResult,
)

import worker as worker_module
from config import CONTRACT_FILES, DATA_DIR, EDITABLE_FILES, METRIC_NAME, TASK_NAME

SPLITS = ("train", "val", "test")


# ==================================================================================================
# Data
# ==================================================================================================


@lru_cache(maxsize=None)
def read_split(split: str) -> list[dict]:
    """Every patient in one split, as {"id", "label"} (1 died within 2 years, 0 alive). The id is an opaque
    hash; the patient's CT is the folder ``patient_dir(id)``. Cached; callers must not mutate the rows."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = DATA_DIR / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def patient_dir(pid: str) -> Path:
    return DATA_DIR / "patients" / pid


@lru_cache(maxsize=None)
def pools() -> dict[str, frozenset]:
    """The fixed TRAIN (400) and VAL (200) pools scripts/build_data.py chose, at the natural death rate.
    Only these patients' CTs are converted, so the schedule draws from nothing else."""
    return {k: frozenset(v) for k, v in json.loads((DATA_DIR / "pools.json").read_text(encoding="utf-8")).items()}


def rows_for(inputs: RunInputs) -> list[dict]:
    """The examples a pass runs: a staged batch file (scheduled runs), or the whole split."""
    rows = read_split(inputs.split)
    ref = Path(inputs.input_ref)
    if ref.suffix == ".json" and ref.is_file():
        wanted = set(json.loads(ref.read_text(encoding="utf-8"))["ids"])
        rows = [r for r in rows if r["id"] in wanted]
    return rows


def whole_split(split: str) -> RunInputs:
    return RunInputs(input_ref=f"{TASK_NAME}:{split}:all", batch_id=f"{split}-all", split=split)


class TaskSource:
    """The data schedule's view of the data (engine.schedule.ScheduleSource): which ids exist in a
    split, and how to stage a batch of them as RunInputs. TEST is never offered."""

    def __init__(self, batch_dir: Path) -> None:
        self.batch_dir = Path(batch_dir)

    def pool(self, split: str) -> list[str]:
        if split == "test":
            raise ValueError("the schedule never draws TEST")
        return sorted(pools()[split])

    def stage(self, ids, split: str, label: str) -> RunInputs:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        path = self.batch_dir / f"{label}.json"
        path.write_text(json.dumps({"split": split, "ids": list(ids)}), encoding="utf-8")
        return RunInputs(input_ref=str(path), batch_id=label, split=split)


# ==================================================================================================
# The program
# ==================================================================================================


class TaskStore:
    """The program's files. ``committed`` means each version is a git commit the engine tags
    ``program-v<n>``. A contract file ships with the program and must never change; the engine
    refuses a version that edits it."""

    def manifest(self) -> ProgramManifest:
        entries = [ManifestEntry(path=p, freeze_policy="committed") for p in EDITABLE_FILES]
        entries += [ManifestEntry(path=p, freeze_policy="committed", contract_file=True) for p in CONTRACT_FILES]
        return ProgramManifest(entries=tuple(entries), combined_hash=f"{TASK_NAME}:program")


# ==================================================================================================
# Runners: one frozen version over one batch, answers written to a file the scorer reads
# ==================================================================================================


class _Runner:
    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)

    def _open(self, tree: Path):  # -> an object with request(dict) -> dict and close()
        raise NotImplementedError

    def run(self, materialized_path: Path, inputs: RunInputs) -> RunArtifacts:
        rows = rows_for(inputs)
        try:
            worker = self._open(Path(materialized_path))
        except Exception as exc:  # noqa: BLE001 - the program not loading makes a failed version
            return RunArtifacts(batch_id=inputs.batch_id, status="program_error", artifact_refs=(),
                                error=ErrorInfo(code="load_error", message_redacted=type(exc).__name__))
        answers = {}
        try:
            for row in rows:  # the label never leaves the host
                reply = worker.request({"id": row["id"]})
                answers[row["id"]] = {"probability": reply.get("probability"), "reason": reply.get("reason"),
                                   "error": reply.get("error"), "findings_log": None}
        except Exception as exc:  # noqa: BLE001 - a hang or a dead worker is the version's fault: a failed version
            hung = "timed out" in str(exc)
            return RunArtifacts(batch_id=inputs.batch_id, status="timeout" if hung else "program_error", artifact_refs=(),
                                error=ErrorInfo(code="worker_timeout" if hung else "worker_died", message_redacted=type(exc).__name__))
        finally:
            worker.close()
        if rows and all(a["error"] for a in answers.values()):
            # Every example crashed: report the pass as failed, so the engine records a failed version (and
            # tells the optimizer) instead of scoring it 0.0. Some crashes are scored: those examples are wrong.
            return RunArtifacts(batch_id=inputs.batch_id, status="program_error", artifact_refs=(),
                                error=ErrorInfo(code="all_examples_failed", message_redacted="all_examples_failed"))
        out_dir = self.output_root / f"{inputs.split}-{inputs.batch_id}-{Path(materialized_path).name}-{uuid.uuid4().hex[:6]}"
        out_dir.mkdir(parents=True)
        out = out_dir / "answers.json"
        out.write_text(json.dumps({"split": inputs.split, "answers": answers}), encoding="utf-8")
        return RunArtifacts(
            batch_id=inputs.batch_id, status="ok",
            artifact_refs=(ArtifactRef(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest()),),
            sub_invocation_count=len(rows),
        )


class _InProcessWorker:
    """The worker's own request handling, called directly: the dry run's stand-in for the container."""

    def __init__(self, tree: Path) -> None:
        self.fn = worker_module.load(tree)

    def request(self, payload: dict) -> dict:
        return worker_module.answer(self.fn, payload)

    def close(self) -> None:
        pass


class LocalRunner(_Runner):
    """Runs the program in this process. Dry runs only: it is not sealed, so a program could read the
    answer files. A real run always uses SealedRunner."""

    def _open(self, tree: Path):
        return _InProcessWorker(tree)


class SealedRunner(_Runner):
    """Runs the program inside the sealed worker: no network, user ``nobody``, the program and worker.py
    read-only, nothing writable. ``spec_for`` is also what the run-start setup check fingerprints."""

    def __init__(self, output_root: Path, *, spec_for: Callable[[Path], object], stderr_dir: Path,
                 versions_root: Path) -> None:
        super().__init__(output_root)
        self.spec_for = spec_for
        self.stderr_dir = Path(stderr_dir)
        self.versions_root = Path(versions_root)

    def setup_description(self):
        # A placeholder where real program versions live: the fingerprint ignores host folders, but the
        # mount checks still run, and a folder under a forbidden root would be refused.
        return self.spec_for(self.versions_root / "setup-check")

    def _open(self, tree: Path):
        from isolation.sealed_worker import open_sealed_worker

        self.stderr_dir.mkdir(parents=True, exist_ok=True)
        return open_sealed_worker(self.spec_for(tree), label=f"{TASK_NAME} worker",
                                  stderr_log_path=self.stderr_dir / f"{tree.name}.log")


# ==================================================================================================
# Scoring
# ==================================================================================================


def _answers(artifacts: RunArtifacts) -> dict:
    return json.loads(Path(artifacts.artifact_refs[0].path).read_text(encoding="utf-8"))["answers"]


#: A patient whose session failed (no output, unreadable, a probability outside 0..1) is scored at this
#: probability: the middle of the ranking, so a crash neither helps nor hides. Counted in the breakdown.
FAILED_PROBABILITY = 0.5
#: "Right" for one patient, used only to pick which TRAIN patients the optimizer studies closely and which the
#: schedule retires: a probability on the correct side of the TRAIN death rate (about 31%). AUROC, the score,
#: does not use it.
RIGHT_SIDE_THRESHOLD = 0.31


def prob(answer: dict) -> float:
    p = answer.get("probability")
    return FAILED_PROBABILITY if answer.get("error") or p is None else float(p)


def auroc(pairs: list[tuple[int, float]]) -> float:
    """Area under the ROC curve for (gold, probability) pairs: the chance a patient who died is ranked above
    one who lived, ties counted half (the Mann-Whitney form). 0.5 when either class is missing."""
    pos = [p for g, p in pairs if g == 1]
    neg = [p for g, p in pairs if g == 0]
    if not pos or not neg:
        return 0.5
    ranked = sorted((p, g) for g, p in pairs)
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid = (i + 1 + j) / 2  # average 1-based rank of the tie block
        rank_sum += mid * sum(1 for _p, g in ranked[i:j] if g == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def right_side(gold: int, p: float) -> bool:
    return (p >= RIGHT_SIDE_THRESHOLD) == (gold == 1)


class TaskScorer:
    """One number for the engine (``primary_metric``: AUROC) plus a breakdown the optimizer reads on TRAIN.
    Everything in a breakdown must be counts and short strings: the engine refuses report text and
    identifier-shaped keys there."""

    def score(self, artifacts: RunArtifacts, split: str, task_config: dict) -> ScoreResult:
        gold = {r["id"]: r["label"] for r in read_split(split)}
        answers = _answers(artifacts)
        pairs = [(gold[i], prob(a)) for i, a in answers.items()]
        died = [p for g, p in pairs if g == 1]
        lived = [p for g, p in pairs if g == 0]
        bins = Counter()
        for g, p in pairs:
            b = f"{min(int(p * 10), 9) / 10:.1f}"
            bins[f"p{b}:n"] += 1
            bins[f"p{b}:died"] += g
        return ScoreResult(
            primary_metric=PrimaryMetric(name=METRIC_NAME, value=auroc(pairs), higher_is_better=True),
            breakdown={"n": len(pairs), "died": len(died), "lived": len(lived),
                       "program_errors": sum(1 for a in answers.values() if a["error"]),
                       "mean_probability_died": round(sum(died) / len(died), 3) if died else None,
                       "mean_probability_lived": round(sum(lived) / len(lived), 3) if lived else None,
                       "brier": round(sum((p - g) ** 2 for g, p in pairs) / len(pairs), 4) if pairs else None,
                       "right_side_of_threshold": sum(right_side(g, p) for g, p in pairs),
                       "calibration_by_decile": dict(sorted(bins.items()))},
            task_config={**task_config, "_split": split},
        )

    def per_example_correct(self, artifacts: RunArtifacts, split: str) -> dict:
        """Right or wrong per example on a TRAIN pass (the data schedule retires examples answered right
        N times in a row). ``None`` for an example whose call failed."""
        gold = {r["id"]: r["label"] for r in read_split(split)}
        return {i: (None if a["error"] else right_side(gold[i], prob(a))) for i, a in _answers(artifacts).items()}


# ==================================================================================================
# What the optimizer sees
# ==================================================================================================


def _trace_files(artifacts: RunArtifacts, call_id: str) -> list[Path]:
    """The labeling session's transcript for one TRAIN example, latest attempt: events.jsonl (every turn, its
    reasoning, each command and its output), response.md (its final message) and meta.json (cost, turns).
    The ProgramRunner keeps them beside the pass's answers; none exist in a dry run."""
    root = Path(artifacts.artifact_refs[0].path).parent / "transcripts" / call_id
    attempts = sorted(root.glob("attempt*"), key=lambda p: int(p.name.removeprefix("attempt") or 0))
    if not attempts:
        return []
    runs = sorted(attempts[-1].glob("*/*"))
    return [f for f in (runs[-1] / n for n in ("events.jsonl", "response.md", "meta.json")) if f.is_file()] if runs else []




def mistake_corpus(feedback_root: Path) -> Callable[[RunArtifacts, ScoreResult], MistakeCorpus]:
    """The loop calls this after each TRAIN pass. It writes every TRAIN patient of the pass (gold, the
    session's probability and one-line reason, and where that patient ranks), marking as a mistake each one
    on the wrong side of RIGHT_SIDE_THRESHOLD. For each mistake it copies the patient's CT, the session's
    findings note and its transcript beside the file as ``<id>-<file>``, listed in the entry. AUROC is about
    ranking, so the patients it got right are listed too (without files): they show what a good ranking
    looks like. VAL and TEST patients are never written anywhere it can reach."""

    def build(artifacts: RunArtifacts, score: ScoreResult) -> MistakeCorpus:
        n = score.task_config.get("_iter", 0)
        rows = {r["id"]: r for r in read_split("train")}
        out = Path(feedback_root) / "iter" / str(n)
        records = out / "patients"
        records.mkdir(parents=True, exist_ok=True)
        answers = _answers(artifacts)
        scored = {i: prob(a) for i, a in answers.items()}
        died = [p for i, p in scored.items() if rows[i]["label"] == 1]
        lived = [p for i, p in scored.items() if rows[i]["label"] == 0]
        entries = []
        for i, a in sorted(answers.items()):
            g, p = rows[i]["label"], scored[i]
            # The share of the other class this patient is ranked wrongly against (ties half).
            other = lived if g == 1 else died
            wrong = sum((q > p) if g == 1 else (q < p) for q in other) + 0.5 * sum(q == p for q in other)
            mistake = bool(a["error"]) or not right_side(g, p)
            files, traces = [], []
            if mistake:
                for f in sorted(patient_dir(i).iterdir()):
                    dest = records / f"{i}-{f.name}"
                    shutil.copy2(f, dest)
                    files.append(str(dest))
                log = Path(a["findings_log"]) if a.get("findings_log") else None
                for f in _trace_files(artifacts, i) + ([log] if log and log.is_file() else []):
                    dest = records / f"{i}-trace-{f.name}"
                    shutil.copy2(f, dest)
                    traces.append(str(dest))
            entries.append({"id": i, "gold_died": g, "probability": a.get("probability"), "error": a["error"],
                            "reason": a.get("reason"), "mistake": mistake,
                            "ranked_wrong_vs_other_class": round(wrong / len(other), 3) if other else None,
                            "ct_files": files, "trace_files": traces})
        path = out / "train_mistakes.json"
        path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        counts = Counter(("died" if e["gold_died"] else "lived") + ("_mistake" if e["mistake"] else "_right")
                         for e in entries)
        return MistakeCorpus(ref=str(path), counts={"total": sum(e["mistake"] for e in entries), **counts})

    return build


class TaskReleaseBuilder:
    """Builds the two files the optimizer reads each iteration, ``release_train.json`` (the TRAIN score,
    its breakdown, a pointer to the mistakes file, the best version so far and the spend) and
    ``release_val.json`` (the VAL score and the number of examples, nothing else).

    ``stamp`` is the run's setup value; the engine checks every release carries it, which catches a
    builder left stamping a placeholder. ``None`` in a dry run, where nothing is sealed."""

    def __init__(self, stamp: str | None) -> None:
        self.stamp = stamp or "unsealed-dry-run"

    def build_release(self, score: ScoreResult, corpus: MistakeCorpus, *, frontier: dict | None = None,
                      budget: dict | None = None):
        split = score.task_config.get("_split", "train")
        n = score.task_config.get("_iter", 0)
        now = datetime.now(timezone.utc).isoformat()
        metric = {"name": score.primary_metric.name, "value": score.primary_metric.value}
        if split == "val":
            val_only = ScoreResult(primary_metric=score.primary_metric, breakdown={"n": score.breakdown.get("n")},
                                   task_config={"_split": "val", "_iter": n})
            return ReleasePayloadVal(schema_version="1", phase="val", metrics=val_only, iter=n,
                                     produced_at_utc=now, optimizer_isolation_hash=self.stamp)
        return ReleasePayloadTrain(
            schema_version="1", phase="train", iter=n, produced_at_utc=now, optimizer_isolation_hash=self.stamp,
            corpus={"metric": metric, "breakdown": score.breakdown, "mistakes_file": corpus.ref,
                    "mistake_counts": corpus.counts, "frontier": frontier or {}, "budget": budget or {}},
        )
