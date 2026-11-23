"""The four adapters the engine calls, plus the data source. This is the task-specific code.

    TaskStore          which files make up the program, and which the optimizer may edit
    LocalRunner        runs a frozen program version over one batch, in this process (dry run only)
    SealedRunner       the same, inside the sealed worker container (rules_based only; unused here)
    TaskScorer         turns the program's answers into one number, plus a breakdown
    mistake_corpus     writes the TRAIN mistakes the optimizer reads
    TaskReleaseBuilder what the optimizer is shown each iteration: TRAIN in detail, VAL as a score only
    TaskSource         the data schedule's view of the data

Copied from ct-findings-5 (2026-11-20). A patient's answer holds, for each question the tumor board answered for
them, either one 0..1 probability of yes (yes/no question) or {label code: 0..1} over the five options, rescaled to
sum to 1 (several-option question). Each question is scored by AUROC over the patients asked it (several-option:
the mean one-vs-rest AUROC over the options present); the task score is the mean over the questions.
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
from config import (BOARDS, CLIMB_KEYS, CONTRACT_FILES, DATA_DIR, EDITABLE_FILES, METRIC_NAME, QUESTIONS, RECORD,
                    RECORD_PATIENTS, SUMMARY_PATIENTS, TASK_NAME)

#: TEST: built only on Phil's go (2026-11-16) by build_data.py test; until then reading it fails. Never scheduled.
SPLITS = ("train", "val", "test")


# ==================================================================================================
# Data
# ==================================================================================================


@lru_cache(maxsize=None)
def read_split(split: str) -> list[dict]:
    """Every patient in one split's pool, as {"id", <question key>: the tumor board's answer code, or None when the
    tumor board did not answer that question for this patient (not asked, not scored)}. The id is an opaque hash of
    the patient; their inputs are the folder ``patient_dir(id)``. Cached; do not mutate."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = DATA_DIR / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def asked(row: dict) -> list[str]:
    """The question keys a patient is asked: those the tumor board answered for them, in config order."""
    return [q.key for q in QUESTIONS if row.get(q.key) is not None]


def asked_of(pid: str) -> list[str]:
    """The questions one patient was asked, looked up in whichever built split holds them (TEST only once built)."""
    for split in SPLITS:
        if (DATA_DIR / "splits" / f"{split}.jsonl").is_file():
            row = next((r for r in read_split(split) if r["id"] == pid), None)
            if row is not None:
                return asked(row)
    raise KeyError("a call id that is in no split")


def patient_dir(pid: str) -> Path:
    """The patient's own inputs folder: summary.txt (treatment; the record agent reads the treatment agent's copy) or
    the CT pair (imaging). The record agent's record folder is ``record_dir``. Boards: the board's record folder (no
    summary)."""
    if BOARDS:
        return record_dir(pid)
    return (SUMMARY_PATIENTS if RECORD else DATA_DIR / "patients") / pid


@lru_cache(maxsize=None)
def read_strata() -> dict[str, dict]:
    """Boards: {id: {"patient": hash, "first_board": bool, "already_treated": bool}} for TRAIN + VAL (TEST's are in
    the private folder until the TEST look). Subgroup tables and the patient-level bootstrap; never staged."""
    path = DATA_DIR / "strata.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def record_dir(pid: str) -> Path:
    """Record mode: the exporter's folder for this patient (OMOP parquet, notes.txt, registry files, meta.json)."""
    return RECORD_PATIENTS / pid


@lru_cache(maxsize=None)
def pools() -> dict[str, frozenset]:
    """The fixed TRAIN (feedback) and VAL (picking) pools scripts/build_data.py wrote. The schedule draws from
    nothing else. The "_draw" record is not a pool."""
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
                answers[row["id"]] = {**{q.key: reply.get(q.key) for q in QUESTIONS},
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


#: A patient whose session failed (no output, unreadable, a number outside 0-100, all-zero options) is scored as no
#: information on every question asked: 0.5 on a yes/no question, an equal share on every option of a several-option
#: one. The middle of the ranking, so a crash neither helps nor hides. Counted in the breakdown.
FAILED_PROBABILITY = 0.5

QUESTION = {q.key: q for q in QUESTIONS}


@lru_cache(maxsize=None)
def train_rates() -> dict[str, float]:
    """Each yes/no question's share of yes among the TRAIN pool's patients asked it. Used only for "right" (which
    TRAIN patients the optimizer studies closely and which the schedule retires); AUROC, the score, does not use it."""
    rows = read_split("train")
    out = {}
    for q in QUESTIONS:
        if not q.options:
            scored = [r[q.key] for r in rows if r.get(q.key) is not None]
            out[q.key] = sum(scored) / len(scored) if scored else 0.5
    return out


def prob(answer: dict, key: str) -> float:
    """A yes/no question's stated probability, or FAILED_PROBABILITY when the call failed."""
    v = answer.get(key)
    if answer.get("error") or isinstance(v, bool) or not isinstance(v, (int, float)):
        return FAILED_PROBABILITY
    return float(v)


def option_probs(answer: dict, key: str) -> dict[int, float]:
    """A several-option question's stated probabilities {code: 0..1} (rescaled to sum to 1 when read), or an equal
    share on every option when the call failed."""
    codes = [c for _k, c, _t in QUESTION[key].options]
    v = answer.get(key)
    if answer.get("error") or not isinstance(v, dict):
        return {c: 1 / len(codes) for c in codes}
    return {c: float(v[str(c)]) for c in codes}


def top_pick(probs: dict[int, float]) -> int:
    """The option with the highest probability; a tie goes to the first option listed (evalset.score's rule)."""
    return max(probs, key=lambda c: (probs[c], -list(probs).index(c)))


def right(answer: dict, key: str, gold: int) -> bool:
    """Right when the call worked and: yes/no, the probability is on the correct side of the question's TRAIN-pool
    rate; several-option, the top pick is the tumor board's answer."""
    if answer.get("error"):
        return False
    if QUESTION[key].options:
        return top_pick(option_probs(answer, key)) == gold
    return (prob(answer, key) >= train_rates()[key]) == (gold == 1)


def auroc(pairs: list[tuple[int, float]]) -> float:
    """Area under the ROC curve for (gold, probability) pairs: the chance a yes is ranked above a no, ties counted
    half (the Mann-Whitney form). 0.5 when either class is missing."""
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


def option_scores(golds: list[int], probs: list[dict[int, float]], codes: list[int]) -> dict:
    """One several-option question over its patients: per option the one-vs-rest AUROC (None when the option has no
    yes or no patient), their mean over the options present ("auroc", the climb's number), top-pick balanced
    accuracy (mean over the answers present of the share picked right) and the share with the tumor board's answer in
    the top two. Failed answers are already an equal share on every option, so they rank in the middle and their top
    pick is the first option (evalset.score counts an unreadable answer wrong; here only "right" uses the pick)."""
    per = {}
    for c in codes:
        ys = [int(g == c) for g in golds]
        per[c] = auroc(list(zip(ys, (p[c] for p in probs)))) if 0 < sum(ys) < len(ys) else None
    present = [v for v in per.values() if v is not None]
    picks = [top_pick(p) for p in probs]
    top2 = [g in sorted(p, key=lambda c: (-p[c], list(p).index(c)))[:2] for g, p in zip(golds, probs)]
    recalls = [sum(pk == c for g, pk in zip(golds, picks) if g == c) / golds.count(c) for c in codes if c in golds]
    return {"auroc": sum(present) / len(present) if present else 0.5,
            "option_auroc": {str(c): (None if v is None else round(v, 4)) for c, v in per.items()},
            "top_pick_balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "tumor_board_answer_in_top_two": round(sum(top2) / len(top2), 4) if top2 else None,
            "n_per_answer": {str(c): golds.count(c) for c in codes},
            "top_picks": {str(c): picks.count(c) for c in codes},
            "mean_probability_on_tumor_board_answer": round(sum(p[g] for g, p in zip(golds, probs)) / len(golds), 3)
            if golds else None}


def question_scores(gold_rows: dict, answers: dict) -> dict:
    """Per question, over the patients asked it. Yes/no: AUROC, Brier, the mean probability for yes and no patients,
    a 10-bin calibration table and how bunched the answers are. Several-option: option_scores. Both: how many were
    asked and how many answers failed. Counts and short strings only, as the engine requires. The optimizer sees all
    of it on TRAIN; VAL releases carry only the headline AUROC and n (TaskReleaseBuilder)."""
    out = {}
    for q in QUESTIONS:
        ids = [i for i in answers if gold_rows[i].get(q.key) is not None]
        failed = sum(1 for i in ids if answers[i].get("error"))
        if q.options:
            codes = [c for _k, c, _t in q.options]
            out[q.key] = {**option_scores([gold_rows[i][q.key] for i in ids], [option_probs(answers[i], q.key) for i in ids],
                                          codes), "asked": len(ids), "failed": failed}
            continue
        pairs = [(gold_rows[i][q.key], prob(answers[i], q.key)) for i in ids]
        yes = [p for g, p in pairs if g == 1]
        no = [p for g, p in pairs if g == 0]
        bins = Counter()
        for g, p in pairs:
            b = f"{min(int(p * 10), 9) / 10:.1f}"
            bins[f"p{b}:n"] += 1
            bins[f"p{b}:yes"] += g
        values = Counter(p for _g, p in pairs)
        out[q.key] = {
            "auroc": auroc(pairs), "yes": len(yes), "no": len(no), "asked": len(ids), "failed": failed,
            "mean_probability_yes": round(sum(yes) / len(yes), 3) if yes else None,
            "mean_probability_no": round(sum(no) / len(no), 3) if no else None,
            "brier": round(sum((p - g) ** 2 for g, p in pairs) / len(pairs), 4) if pairs else None,
            "right_side_of_train_rate": sum(right(answers[i], q.key, gold_rows[i][q.key]) for i in ids),
            "distinct_values": len(values),
            "share_on_most_common_value": round(values.most_common(1)[0][1] / len(pairs), 3) if pairs else None,
            "calibration_by_decile": dict(sorted(bins.items())),
        }
    return out


class TaskScorer:
    """One number for the engine (``primary_metric``: AUROC averaged over the questions asked in the pass) plus a
    breakdown the optimizer reads on TRAIN. Everything in a breakdown must be counts and short strings: the engine refuses report
    text and identifier-shaped keys there."""

    def score(self, artifacts: RunArtifacts, split: str, task_config: dict) -> ScoreResult:
        gold = {r["id"]: r for r in read_split(split)}
        return self.score_answers(gold, _answers(artifacts), task_config, split)

    def score_answers(self, gold: dict, answers: dict, task_config: dict, split: str) -> ScoreResult:
        per = question_scores(gold, answers)
        # The climb's number: the mean over CLIMB_KEYS (boards: T1, T2, T3; T3b and T4 too small, Phil 2026-11-22).
        live = [q["auroc"] for k, q in per.items() if q["asked"] and k in CLIMB_KEYS]
        value = sum(live) / len(live) if live else 0.0
        for q in per.values():  # rounded for reading only, after the score is taken
            q["auroc"] = round(q["auroc"], 4)
        return ScoreResult(
            primary_metric=PrimaryMetric(name=METRIC_NAME, value=value, higher_is_better=True),
            breakdown={"n": len(answers), "program_errors": sum(1 for a in answers.values() if a["error"]), **per},
            task_config={**task_config, "_split": split},
        )

    def per_example_correct(self, artifacts: RunArtifacts, split: str) -> dict:
        """Right or wrong per patient on a TRAIN pass (the data schedule retires patients answered right N times in
        a row): right only when every question asked is right. ``None`` for a patient whose call failed."""
        gold = {r["id"]: r for r in read_split(split)}
        return {i: (None if a["error"] else all(right(a, key, gold[i][key]) for key in asked(gold[i])))
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


#: Boards: a mistake's record is copied for the optimizer whole unless its notes are over this size; then the two
#: notes files are left out (the session's transcript still shows what it read). Build 2026-11-22.
FEEDBACK_NOTES_MAX_BYTES = 2_000_000


def patient_files(pid: str) -> list[tuple[Path, str]]:
    """Every input file of one patient and its flat name in the optimizer's feedback folder: summary.txt, or
    prior/ct.npz -> prior-ct.npz, current/meta.json -> current-meta.json. Boards: the record's files, without
    notes.txt and note.parquet when notes.txt is over FEEDBACK_NOTES_MAX_BYTES."""
    root = patient_dir(pid)
    files = [f for f in sorted(root.rglob("*")) if f.is_file()]
    if BOARDS and (root / "notes.txt").is_file() and (root / "notes.txt").stat().st_size > FEEDBACK_NOTES_MAX_BYTES:
        files = [f for f in files if f.name not in ("notes.txt", "note.parquet")]
    return [(f, "-".join(f.relative_to(root).parts)) for f in files]


def tumor_board_year(pid: str) -> int | None:
    """Boards: the year of the tumor board (record cutoff + 7 days, from the record's meta.json; dates are shifted by at
    most about a month, so the year is right), for the optimizer to study mistakes by era (Phil 2026-11-23). Year only."""
    from datetime import date, timedelta

    try:
        cut = json.loads((record_dir(pid) / "meta.json").read_text(encoding="utf-8"))["cutoff"]
        return (date.fromisoformat(cut[:10]) + timedelta(days=7)).year
    except (OSError, ValueError, KeyError):
        return None


def mistake_corpus(feedback_root: Path) -> Callable[[RunArtifacts, ScoreResult], MistakeCorpus]:
    """The loop calls this after each TRAIN pass. It writes every TRAIN patient of the pass (the tumor board's answer
    per question asked, None when not asked; the session's probabilities and reasons), marking as a mistake each one
    whose call failed or that was wrong on a question asked (``right``). For each mistake it copies the patient's
    inputs (``patient_files``), the session's findings note and its transcript beside the file as ``<id>-<file>``,
    listed in the entry. VAL patients are never written anywhere it can reach."""

    def show(a: dict, key: str):
        if a["error"]:
            return None
        if QUESTION[key].options:
            return {letter: round(100 * option_probs(a, key)[c]) for letter, c, _t in QUESTION[key].options}
        return round(100 * prob(a, key))

    def build(artifacts: RunArtifacts, score: ScoreResult) -> MistakeCorpus:
        n = score.task_config.get("_iter", 0)
        rows = {r["id"]: r for r in read_split("train")}
        out = Path(feedback_root) / "iter" / str(n)
        records = out / "patients"
        records.mkdir(parents=True, exist_ok=True)
        answers = _answers(artifacts)
        entries = []
        for i, a in sorted(answers.items()):
            keys = asked(rows[i])
            wrong_at = [key for key in keys if not right(a, key, rows[i][key])]
            mistake = bool(a["error"]) or bool(wrong_at)
            files, traces = [], []
            if mistake:
                for f, name in patient_files(i):
                    dest = records / f"{i}-{name}"
                    _copy_over(f, dest)
                    files.append(str(dest))
                log = Path(a["findings_log"]) if a.get("findings_log") else None
                for f in _trace_files(artifacts, i) + ([log] if log and log.is_file() else []):
                    dest = records / f"{i}-trace-{f.name}"
                    _copy_over(f, dest)
                    traces.append(str(dest))
            gold = {}
            for key in keys:
                q = QUESTION[key]
                gold[key] = next(letter for letter, c, _t in q.options if c == rows[i][key]) if q.options else rows[i][key]
            extra = {"tumor_board_year": tumor_board_year(i)} if BOARDS else {}
            entries.append({"id": i, **extra, "asked": keys, "tumor_board_answer": gold,
                            "answer": {key: show(a, key) for key in keys},
                            "error": a["error"], "reason": a.get("reason"), "reasons": a.get("reasons") or [],
                            "mistake": mistake, "wrong_at": wrong_at, "input_files": files, "trace_files": traces})
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
