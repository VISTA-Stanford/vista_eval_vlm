"""The four adapters the engine calls, plus the data source. This is the task-specific code.

    TaskStore          which files make up the program, and which the optimizer may edit
    LocalRunner        runs a frozen program version over one batch, in this process (dry run only)
    SealedRunner       the same, inside the sealed worker container (rules_based only; unused here)
    TaskScorer         turns the program's answers into one number, plus a breakdown
    mistake_corpus     writes the TRAIN mistakes the optimizer reads
    TaskReleaseBuilder what the optimizer is shown each iteration: TRAIN in detail, VAL as a score only
    TaskSource         the data schedule's view of the data

Copied from died-2yr-ct's probability mode (2026-11-10). Each scan gets five 0..1 probabilities, one per findings
question (config.QUESTIONS); each question is scored by AUROC over the scans that have a label for it (a failed or
unreadable answer counts FAILED_PROBABILITY), and the task score is the mean over the five.
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
from config import CONTRACT_FILES, DATA_DIR, EDITABLE_FILES, METRIC_NAME, QUESTIONS, TASK_NAME

#: TEST is sealed: build_data.py writes no test file, so reading it fails (spec: TEST waits for its own go).
SPLITS = ("train", "val", "test")


# ==================================================================================================
# Data
# ==================================================================================================


@lru_cache(maxsize=None)
def read_split(split: str) -> list[dict]:
    """Every scan in one split's pool, as {"id", <five question keys>: 1 yes, 0 no, None unscored}. The id is an
    opaque hash of one scan (one per patient); the scan is the folder ``patient_dir(id)``. Cached; do not mutate."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = DATA_DIR / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def patient_dir(pid: str) -> Path:
    return DATA_DIR / "patients" / pid


@lru_cache(maxsize=None)
def pools() -> dict[str, frozenset]:
    """The fixed TRAIN (400) and VAL (200) pools scripts/build_data.py drew (enriched for positives). Only these
    scans are converted, so the schedule draws from nothing else. The "_draw" record is not a pool."""
    data = json.loads((DATA_DIR / "pools.json").read_text(encoding="utf-8"))
    return {k: frozenset(v) for k, v in data.items() if not k.startswith("_")}


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
                answers[row["id"]] = {**{key: reply.get(key) for key, _t, _q in QUESTIONS},
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


#: A scan whose session failed (no output, unreadable, a number outside 0-100) is scored at this probability on every
#: question: the middle of the ranking, so a crash neither helps nor hides. Counted in the breakdown (spec).
FAILED_PROBABILITY = 0.5


@lru_cache(maxsize=None)
def train_rates() -> dict[str, float]:
    """Each question's share of yes among the TRAIN pool's scored scans. Used only for "right" (which TRAIN scans the
    optimizer studies closely and which the schedule retires); AUROC, the score, does not use it."""
    rows = read_split("train")
    out = {}
    for key, _t, _q in QUESTIONS:
        scored = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(scored) / len(scored) if scored else 0.5
    return out


def prob(answer: dict, key: str) -> float:
    """The stated probability for one question, or FAILED_PROBABILITY when the call failed."""
    v = answer.get(key)
    if answer.get("error") or isinstance(v, bool) or not isinstance(v, (int, float)):
        return FAILED_PROBABILITY
    return float(v)


def right(answer: dict, key: str, gold: int) -> bool:
    """Right when the call worked and the probability is on the correct side of the question's TRAIN-pool rate."""
    return not answer.get("error") and (prob(answer, key) >= train_rates()[key]) == (gold == 1)


def auroc(pairs: list[tuple[int, float]]) -> float:
    """Area under the ROC curve for (gold, probability) pairs: the chance a yes scan is ranked above a no scan, ties
    counted half (the Mann-Whitney form). 0.5 when either class is missing."""
    pos = sum(1 for g, _p in pairs if g == 1)
    neg = len(pairs) - pos
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
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def question_scores(gold_rows: dict, answers: dict) -> dict:
    """Per question, over the scans with a label for it: AUROC, Brier, the mean probability for yes and no scans, a
    10-bin calibration table, and how bunched the answers are (distinct values, share on the most common one). Counts
    and short strings only, as the engine requires. The optimizer sees all of it on TRAIN; VAL releases carry only
    the headline AUROC and n (TaskReleaseBuilder)."""
    out = {}
    for key, _t, _q in QUESTIONS:
        pairs = [(gold_rows[i][key], prob(a, key)) for i, a in answers.items() if gold_rows[i].get(key) is not None]
        yes = [p for g, p in pairs if g == 1]
        no = [p for g, p in pairs if g == 0]
        bins = Counter()
        for g, p in pairs:
            b = f"{min(int(p * 10), 9) / 10:.1f}"
            bins[f"p{b}:n"] += 1
            bins[f"p{b}:yes"] += g
        values = Counter(p for _g, p in pairs)
        out[key] = {
            "auroc": auroc(pairs), "yes": len(yes), "no": len(no), "unscored": len(answers) - len(pairs),
            "mean_probability_yes": round(sum(yes) / len(yes), 3) if yes else None,
            "mean_probability_no": round(sum(no) / len(no), 3) if no else None,
            "brier": round(sum((p - g) ** 2 for g, p in pairs) / len(pairs), 4) if pairs else None,
            "right_side_of_train_rate": sum(right(answers[i], key, gold_rows[i][key]) for i in answers
                                            if gold_rows[i].get(key) is not None),
            "distinct_values": len(values),
            "share_on_most_common_value": round(values.most_common(1)[0][1] / len(pairs), 3) if pairs else None,
            "calibration_by_decile": dict(sorted(bins.items())),
        }
    return out


class TaskScorer:
    """One number for the engine (``primary_metric``: AUROC averaged over the five questions) plus a breakdown the
    optimizer reads on TRAIN. Everything in a breakdown must be counts and short strings: the engine refuses report
    text and identifier-shaped keys there."""

    def score(self, artifacts: RunArtifacts, split: str, task_config: dict) -> ScoreResult:
        gold = {r["id"]: r for r in read_split(split)}
        return self.score_answers(gold, _answers(artifacts), task_config, split)

    def score_answers(self, gold: dict, answers: dict, task_config: dict, split: str) -> ScoreResult:
        per = question_scores(gold, answers)
        value = sum(q["auroc"] for q in per.values()) / len(per) if answers else 0.0
        for q in per.values():  # rounded for reading only, after the score is taken
            q["auroc"] = round(q["auroc"], 4)
        return ScoreResult(
            primary_metric=PrimaryMetric(name=METRIC_NAME, value=value, higher_is_better=True),
            breakdown={"n": len(answers), "program_errors": sum(1 for a in answers.values() if a["error"]), **per},
            task_config={**task_config, "_split": split},
        )

    def per_example_correct(self, artifacts: RunArtifacts, split: str) -> dict:
        """Right or wrong per scan on a TRAIN pass (the data schedule retires scans answered right N times in a
        row): right only when every scored question is right. ``None`` for a scan whose call failed."""
        gold = {r["id"]: r for r in read_split(split)}
        return {i: (None if a["error"] else all(right(a, key, gold[i][key]) for key, _t, _q in QUESTIONS
                                                if gold[i].get(key) is not None))
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




def _copy_over(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest``, overwriting ``dest`` even if a prior run left it read-only.
    A resume re-runs the TRAIN pass and re-copies the session trace files, which are mode 0444:
    ``shutil.copy2`` cannot reopen an existing 0444 ``dest`` for writing, so remove it first."""
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)


def mistake_corpus(feedback_root: Path) -> Callable[[RunArtifacts, ScoreResult], MistakeCorpus]:
    """The loop calls this after each TRAIN pass. It writes every TRAIN scan of the pass (gold per question, None when
    unscored; the session's probabilities and reasons), marking as a mistake each one on the wrong side of a question's
    TRAIN rate, or whose call failed. For each mistake it copies the scan, the session's findings note and its
    transcript beside the file as ``<id>-<file>``, listed in the entry. VAL scans are never written anywhere it can
    reach."""

    def build(artifacts: RunArtifacts, score: ScoreResult) -> MistakeCorpus:
        n = score.task_config.get("_iter", 0)
        rows = {r["id"]: r for r in read_split("train")}
        out = Path(feedback_root) / "iter" / str(n)
        records = out / "patients"
        records.mkdir(parents=True, exist_ok=True)
        answers = _answers(artifacts)
        entries = []
        for i, a in sorted(answers.items()):
            wrong_at = [key for key, _t, _q in QUESTIONS if rows[i].get(key) is not None and not right(a, key, rows[i][key])]
            mistake = bool(a["error"]) or bool(wrong_at)
            files, traces = [], []
            if mistake:
                for f in sorted(patient_dir(i).iterdir()):
                    dest = records / f"{i}-{f.name}"
                    _copy_over(f, dest)
                    files.append(str(dest))
                log = Path(a["findings_log"]) if a.get("findings_log") else None
                for f in _trace_files(artifacts, i) + ([log] if log and log.is_file() else []):
                    dest = records / f"{i}-trace-{f.name}"
                    _copy_over(f, dest)
                    traces.append(str(dest))
            entries.append({"id": i,
                            **{f"gold_{key}": rows[i].get(key) for key, _t, _q in QUESTIONS},
                            **{f"probability_{key}": (None if a["error"] else round(prob(a, key), 3))
                               for key, _t, _q in QUESTIONS},
                            "error": a["error"], "reason": a.get("reason"), "reasons": a.get("reasons") or [],
                            "mistake": mistake, "wrong_at": wrong_at, "ct_files": files, "trace_files": traces})
        path = out / "train_mistakes.json"
        path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        counts = Counter(f"wrong_{key}" for e in entries for key in e["wrong_at"])
        counts["failed"] = sum(1 for e in entries if e["error"])
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
