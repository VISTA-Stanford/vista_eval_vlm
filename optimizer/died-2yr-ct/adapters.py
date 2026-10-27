"""The four adapters the engine calls, plus the data source. This is the task-specific code.

    TaskStore          which files make up the program, and which the optimizer may edit
    LocalRunner        runs a frozen program version over one batch, in this process (dry run only)
    SealedRunner       the same, inside the sealed worker container (every real run)
    TaskScorer         turns the program's answers into one number, plus a breakdown
    mistake_corpus     writes the TRAIN mistakes the optimizer reads
    TaskReleaseBuilder what the optimizer is shown each iteration: TRAIN in detail, VAL as a score only
    TaskSource         the data schedule's view of your data (only when config.SCHEDULE is set)

What a new task changes: ``read_split`` (your data format), ``TaskScorer`` (your metric), and what
``mistake_corpus`` writes per mistake. The runners and the release builder rarely change. Since 2026-10-27 each
patient gets two forced yes/no answers (death within 1 and within 2 years), scored by balanced accuracy averaged
over the two.
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
from config import CONTRACT_FILES, DATA_DIR, EDITABLE_FILES, HORIZONS, METRIC_NAME, TASK_NAME

SPLITS = ("train", "val", "test")


# ==================================================================================================
# Data
# ==================================================================================================


@lru_cache(maxsize=None)
def read_split(split: str) -> list[dict]:
    """Every patient in one split, as {"id", "label", "label_1yr"} (1 died within 2 years / 1 year, 0 alive). The id is an opaque
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
                answers[row["id"]] = {**{key: reply.get(key) for key, _g, _w in HORIZONS},
                                      "reason": reply.get("reason"), "error": reply.get("error"), "findings_log": None}
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


def said(answer: dict, key: str) -> bool | None:
    """The session's yes (True) or no (False) for one horizon; None when its call failed or the answer was not
    exactly yes or no."""
    v = answer.get(key)
    return None if answer.get("error") or not isinstance(v, bool) else v


def right(answer: dict, key: str, gold: int) -> bool:
    """A failed or unreadable answer is wrong (eval-set spec, Decided: Score: unparseable or refused = wrong)."""
    v = said(answer, key)
    return v is not None and v == bool(gold)


def balanced_accuracy(pairs: list[tuple[int, bool]]) -> float:
    """Mean of the recall of each class present, for (gold, right) pairs: (sensitivity + specificity) / 2 when
    both classes are present, as sklearn's balanced_accuracy_score. 0.0 for no pairs."""
    recalls = [sum(r for g, r in pairs if g == c) / n for c in (0, 1) if (n := sum(1 for g, _r in pairs if g == c))]
    return sum(recalls) / len(recalls) if recalls else 0.0


def horizon_scores(gold_rows: dict, answers: dict) -> dict:
    """Per horizon: balanced accuracy and its counts (counts and short strings only, as the engine requires)."""
    out = {}
    for key, field, words in HORIZONS:
        pairs = [(gold_rows[i][field], right(a, key, gold_rows[i][field])) for i, a in answers.items()]
        died = [r for g, r in pairs if g == 1]
        lived = [r for g, r in pairs if g == 0]
        tag = words.replace(" ", "")
        out[tag] = {"balanced_accuracy": balanced_accuracy(pairs), "died": len(died), "lived": len(lived),
                    "died_said_yes": sum(died), "lived_said_no": sum(lived),
                    "said_yes": sum(1 for a in answers.values() if said(a, key) is True)}
    return out


class TaskScorer:
    """One number for the engine (``primary_metric``: balanced accuracy at 1 and at 2 years, averaged) plus a
    breakdown the optimizer reads on TRAIN. Everything in a breakdown must be counts and short strings: the engine
    refuses report text and identifier-shaped keys there."""

    def score(self, artifacts: RunArtifacts, split: str, task_config: dict) -> ScoreResult:
        gold = {r["id"]: r for r in read_split(split)}
        answers = _answers(artifacts)
        per = horizon_scores(gold, answers)
        value = sum(h["balanced_accuracy"] for h in per.values()) / len(per) if answers else 0.0
        for h in per.values():  # rounded for reading only, after the score is taken
            h["balanced_accuracy"] = round(h["balanced_accuracy"], 4)
        (k1, *_), (k2, *_) = HORIZONS
        return ScoreResult(
            primary_metric=PrimaryMetric(name=METRIC_NAME, value=value, higher_is_better=True),
            breakdown={"n": len(answers),
                       "program_errors": sum(1 for a in answers.values() if a["error"]),
                       # yes at 1 year but no at 2 years: impossible, so one of the two is wrong
                       "said_dead_by_1yr_alive_at_2yr": sum(1 for a in answers.values()
                                                            if said(a, k1) is True and said(a, k2) is False),
                       **per},
            task_config={**task_config, "_split": split},
        )

    def per_example_correct(self, artifacts: RunArtifacts, split: str) -> dict:
        """Right or wrong per example on a TRAIN pass (the data schedule retires examples answered right
        N times in a row): right only when both horizons are right. ``None`` for an example whose call failed."""
        gold = {r["id"]: r for r in read_split(split)}
        return {i: (None if a["error"] else all(right(a, key, gold[i][field]) for key, field, _w in HORIZONS))
                for i, a in _answers(artifacts).items()}


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
    """The loop calls this after each TRAIN pass. It writes every TRAIN patient of the pass (gold at each horizon,
    the session's yes/no answers and one-line reason), marking as a mistake each one wrong at either horizon or
    whose call failed. For each mistake it copies the patient's CT, the session's findings note and its
    transcript beside the file as ``<id>-<file>``, listed in the entry. The patients it got right are listed too
    (without files): balanced accuracy weighs both classes, so they show what the program already gets right.
    VAL and TEST patients are never written anywhere it can reach."""

    def build(artifacts: RunArtifacts, score: ScoreResult) -> MistakeCorpus:
        n = score.task_config.get("_iter", 0)
        rows = {r["id"]: r for r in read_split("train")}
        out = Path(feedback_root) / "iter" / str(n)
        records = out / "patients"
        records.mkdir(parents=True, exist_ok=True)
        answers = _answers(artifacts)
        entries = []
        for i, a in sorted(answers.items()):
            wrong_at = [words for key, field, words in HORIZONS if not right(a, key, rows[i][field])]
            mistake = bool(a["error"]) or bool(wrong_at)
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
            entries.append({"id": i,
                            **{f"gold_{key}": rows[i][field] for key, field, _w in HORIZONS},
                            **{f"said_{key}": ({True: "yes", False: "no"}.get(said(a, key))) for key, _f, _w in HORIZONS},
                            "error": a["error"], "reason": a.get("reason"), "mistake": mistake,
                            "wrong_at": wrong_at, "ct_files": files, "trace_files": traces})
        path = out / "train_mistakes.json"
        path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        last = HORIZONS[-1][0]  # grouped by the 2-year outcome: died by 1 year, died in year 2, alive at 2 years
        counts = Counter(("died_1yr" if e[f"gold_{HORIZONS[0][0]}"] else "died_2yr" if e[f"gold_{last}"] else "lived")
                         + ("_mistake" if e["mistake"] else "_right") for e in entries)
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
