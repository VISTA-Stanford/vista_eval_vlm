#!/usr/bin/env python3
"""Hill-climb the death agent over time (spec addendum, approved 2026-11-12): Opus 5.5 edits ONE shared
instructions.md used by both arms at every decision point; each round runs 100 TRAIN patients (all their points, both
arms, in point order) and then every VAL row (1,378 calls); score = mean AUROC over 5 points x 2 horizons x 2 arms.
10 rounds, no early stop, best round kept. Hard stop $300 in code; the $0.10-a-call early stop.

    climb.py init                         the work folder and program-v0 (the first pass's fresh instructions)
    climb.py pins [--write]               fingerprint the optimizer, program and search sessions; commit the pins
    climb.py dry-run [--iterations 2]     $0: the real loop over fake sessions and a scripted optimizer
    climb.py run --run-id R [--resume]    the paid climb
    climb.py report --run-id R            per-round VAL table (aggregates)

What it reuses, unchanged: the engine's loop, driver, schedule and ProgramRunner; died-2yr-ehr's work-folder layout
and init, optimizer prompt builder (``optimizer_prompt.build_prompt``), release builder, ScriptedAgent, change-note
wrapper, program store, transcript reader and pins; this task's program spec, leak checks, spend counter and fake
sessions. What is new: ``ClimbRunner`` (a ProgramRunner whose pass is every decision point of a batch of patients,
both arms, in order, in passes of common.CHUNK calls: the first pass's driver applied to one program version), the
over-time scorer and mistake corpus, the optimizer's brief and standing instructions (``context/task_over_time.md``,
``prompts/optimizer_instructions.md``: died-2yr-ehr's probability text adapted), and the optimizer's forbidden list,
which adds this task's folder (died-2yr-ehr's ``build_agent`` hard-codes its own list, so its 30 lines are mirrored
here rather than imported).

Round 0: the seed's VAL pass is the first pass (run over-time-r1): when the version being scored on VAL is the seed,
its answers are read from OVER_TIME_ROUND0_DIR (default the first pass's private answers) and nothing is paid. The
first pass's memory arm used program_memory/instructions.md, which words the earlier-answers paragraph slightly
differently from program.MEMORY_PARAGRAPH; every later round uses the shared text plus that paragraph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as c  # noqa: E402  (first: died-2yr-ehr's settings, then its config)
import config  # noqa: E402
import adapters as ad  # noqa: E402
import optimizer as opt  # noqa: E402
import optimizer_prompt as op  # noqa: E402
import program  # noqa: E402
import run as d2  # noqa: E402  (died-2yr-ehr's run.py)
import run_over_time as rot  # noqa: E402
import score as sc  # noqa: E402

from engine.driver import DriverConfig, StartPoint, exit_code, run_driver  # noqa: E402
from engine.loop import LoopPause, LoopStop  # noqa: E402
from engine.loop_ops import LocalLoopOps  # noqa: E402
from engine.run_start import Carried  # noqa: E402
from engine.run_store import CheckedRunStore  # noqa: E402
from engine.schemas import ArtifactRef, MistakeCorpus, PrimaryMetric, RunArtifacts, RunInputs, ScoreResult  # noqa: E402
from isolation.program_runner import ProgramRunner  # noqa: E402
from isolation.session_scope import Forbidden  # noqa: E402

TASK = "died-ehr-over-time"
ITERATIONS = int(os.environ.get("ITERATIONS", "10"))
#: Hard stop over the whole climb (spec addendum: $300), patient sessions + the optimizer's allowance.
CLIMB_BUDGET_USD = float(os.environ.get("CLIMB_BUDGET_USD", "300"))
#: The optimizer's search assistant, capped for this task (died-2yr-ehr allows 1,000 requests a round): at most this
#: many requests a round, each at most this much.
SEARCH_REQUESTS_PER_ROUND = 100
SEARCH_BUDGET_USD = 0.20
#: Kept back, before every pass, for the next optimizer round not yet paid: its session cap plus a full round of
#: search at the caps. What the optimizer and its search already spent is counted for real (spend_total).
OPTIMIZER_RESERVE_USD = config.OPTIMIZER_SESSION_BUDGET_USD + SEARCH_REQUESTS_PER_ROUND * SEARCH_BUDGET_USD
TRAIN_SIZE = int(os.environ.get("CLIMB_TRAIN_SIZE", "100"))
SCHEDULE_SEED = 20261112
ROUND0_DIR = Path(os.environ.get("OVER_TIME_ROUND0_DIR", c.PRIVATE_DIR / "runs" / "over-time-r1" / "answers"))
WORKDIR = Path(os.environ.get("CLIMB_WORKDIR", Path.home() / ".cache" / "alo-tasks" / "died-ehr-over-time-climb")).resolve()
BRIEF = HERE / "context" / "task_over_time.md"
INSTRUCTIONS = HERE / "prompts" / "optimizer_instructions.md"
SEED_DIR = program.PROGRAM_DIRS["fresh"]


def layout() -> "opt.Layout":
    return opt.Layout(root=WORKDIR)


def forbidden(lay) -> tuple:
    return rot.forbidden(lay)


def optimizer_forbidden(lay) -> tuple:
    """died-2yr-ehr's optimizer list (its own + the program outputs) plus this task's folder and the first pass's work
    folder (its VAL answers and transcripts), when that folder is on this machine."""
    first = c.WORKDIR
    extra = ((Forbidden(first, representatives=("private/README.txt",)),)
             if (first / "private" / "README.txt").is_file() and first != WORKDIR else ())
    return (*forbidden(lay), Forbidden(lay.program_outputs, representatives=("archive",)), *extra)


def optimizer_spend(lay) -> tuple[float, float]:
    """(optimizer sessions, search sessions) dollars spent in this work folder: each optimizer session's meta.json
    (transcripts/optimizer) and every request in the search logs (priming included)."""
    opt_usd = 0.0
    for f in (lay.transcripts / "optimizer").rglob("meta.json"):
        try:
            cost = json.loads(f.read_text(encoding="utf-8")).get("cost_usd")
        except (OSError, ValueError):
            cost = None
        # A session cut off (time limit, crash) records no cost: count it at its cap, as patient sessions are.
        opt_usd += float(cost) if isinstance(cost, (int, float)) else config.OPTIMIZER_SESSION_BUDGET_USD
    search_usd = 0.0
    for f in lay.search_logs.rglob("search_log.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                cost = json.loads(line).get("cost_usd")
            except ValueError:
                cost = None
            if not line.strip():
                continue
            # A failed or cut-off search records no cost: count it at its cap.
            search_usd += float(cost) if isinstance(cost, (int, float)) else SEARCH_BUDGET_USD
    return opt_usd, search_usd


# ==================================================================================================
# The runner: one program version over every decision point of a batch of patients, both arms
# ==================================================================================================


def is_seed(tree: Path) -> bool:
    """The tree is the seed program (the first pass's fresh instructions, no tools)."""
    files = sorted(p.relative_to(tree).as_posix() for p in Path(tree).rglob("*") if p.is_file())
    want = sorted(p.relative_to(SEED_DIR).as_posix() for p in SEED_DIR.rglob("*") if p.is_file())
    return files == want and all((tree / f).read_bytes() == (SEED_DIR / f).read_bytes() for f in files)


def memory_tree(tree: Path) -> Path:
    """The memory arm's copy of a version: the same files, MEMORY_PARAGRAPH after instructions.md. Under the
    runner's materialize_root, read-only, rebuilt when the version's content changes."""
    tree = Path(tree)
    dest = tree.parent / f"{tree.name}-memory"
    text = (tree / program.agentic.ENTRY).read_text(encoding="utf-8").rstrip() + "\n\n" + program.MEMORY_PARAGRAPH + "\n"
    marker = hashlib.sha256((text + "".join(sorted(str(p.relative_to(tree)) + p.read_bytes().hex()
                                                 for p in tree.rglob("*") if p.is_file()))).encode()).hexdigest()
    stamp = dest.parent / f".{dest.name}.sha"
    if dest.exists() and stamp.is_file() and stamp.read_text() == marker:
        return dest
    if dest.exists():
        for p in [dest, *dest.rglob("*")]:
            p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(dest)
    shutil.copytree(tree, dest)
    (dest / program.agentic.ENTRY).chmod(0o644)
    (dest / program.agentic.ENTRY).write_text(text, encoding="utf-8")
    for p in [*dest.rglob("*"), dest]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    stamp.write_text(marker)
    return dest


class ClimbRunner(ProgramRunner):
    """A ProgramRunner (so the engine's agentic setup check holds) whose bound ``run`` is a whole over-time pass."""

    def as_runner(self, *, run_id: str):
        return BoundClimb(self, super().as_runner(run_id=run_id), run_id)


class BoundClimb:
    def __init__(self, outer: ClimbRunner, inner, run_id: str) -> None:
        self.outer, self.inner, self.run_id = outer, inner, run_id
        self.lay = layout()
        self.state: dict = {}

    def cache_identity(self, tree: Path, inputs: RunInputs) -> str:
        return f"{program.PROMPT_TAG}:{program.MEMORY_PARAGRAPH}"

    def _cache_key(self, tree: Path, inputs: RunInputs) -> str:
        """The same program content over the same patients and split is the same pass, whatever its folder is
        called: so a resumed run (whose in-memory pass cache is gone) never pays twice for a finished pass."""
        from engine.materialize import tree_fingerprint

        ids = sorted(json.loads(Path(inputs.input_ref).read_text(encoding="utf-8"))["ids"])
        return hashlib.sha256(json.dumps([tree_fingerprint(tree), inputs.split, ids, program.PROMPT_TAG,
                                          program.MEMORY_PARAGRAPH]).encode()).hexdigest()[:20]

    def run(self, tree: Path, inputs: RunInputs) -> RunArtifacts:
        cache = self.lay.private / "climb" / self.run_id / "done" / f"{self._cache_key(Path(tree), inputs)}.json"
        if cache.is_file():
            path = Path(json.loads(cache.read_text())["answers"])
            print(f"[{inputs.split}-{Path(tree).name}] finished earlier for the same program and patients: served "
                  f"from {path.parent.name}, 0 calls", flush=True)
            return RunArtifacts(batch_id=inputs.batch_id, status="ok", cost_usd=0.0,
                                artifact_refs=(ArtifactRef(path=str(path),
                                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()),))
        try:
            answers, paid = self._answers(Path(tree), inputs)
        except rot.Stop as exc:
            raise LoopStop(str(exc), reason="over_time_rule") from exc
        out = self.lay.private / "climb" / self.run_id / f"{inputs.batch_id}-{Path(tree).name}"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "answers.json"
        path.write_text(json.dumps({"answers": answers, "batch_id": inputs.batch_id, "tree": Path(tree).name,
                                    "split": inputs.split}), encoding="utf-8")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"answers": str(path)}))
        return RunArtifacts(batch_id=inputs.batch_id, status="ok", cost_usd=paid,
                            artifact_refs=(ArtifactRef(path=str(path),
                                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()),))

    def _answers(self, tree: Path, inputs: RunInputs) -> tuple[dict, float]:
        from isolation.pass_outputs import PassStopped  # noqa: F401  (a stop propagates to the loop as is)

        patients = set(json.loads(Path(inputs.input_ref).read_text(encoding="utf-8"))["ids"])
        split = inputs.split
        if split == "val" and is_seed(tree) and ROUND0_DIR.is_dir():
            merged = {}
            for point in c.POINT_KEYS:
                for arm in program.ARMS:
                    got = json.loads((ROUND0_DIR / f"{point}-{arm}.json").read_text(encoding="utf-8"))["answers"]
                    merged.update({f"{r}.{arm}": {**a, "pass_dir": None} for r, a in got.items()
                                   if r in program.rows() and program.rows()[r]["patient"] in patients})
            print(f"round 0: the seed's VAL answers reused from the first pass ({len(merged)} answers, 0 calls)",
                  flush=True)
            return merged, 0.0
        trees = {"fresh": tree, "memory": memory_tree(tree)}
        before, _ = rot.spend(self.lay, self.run_id)
        merged: dict[str, dict] = {}
        answers: dict[str, dict[str, dict]] = {}
        tag = f"{split}-{tree.name}"
        for point in c.POINT_KEYS:
            pr = program.point_rows(point, split, patients)
            n, v = rot.record_leaks(point, split, patients)
            print(f"[{tag} {point}] record leak check: {n} records, {sum(v.values())} violations {v or ''}", flush=True)
            if v:
                raise rot.Stop(f"a {split} record at {point} holds something dated after its cut: stop")
            for arm in program.ARMS:
                name = f"{point}-{arm}"
                if arm == "memory" and point == c.POINT_KEYS[0]:
                    answers[name] = answers[f"{point}-fresh"]  # no earlier answer: the fresh arm's (as the first pass)
                    continue
                earlier = {p: answers[f"{p}-{arm}"] for p in c.POINT_KEYS[: c.POINT_KEYS.index(point)]}
                calls, leaks, dropped = [], 0, 0
                for r in pr:
                    mem, gone = program.memory_for(r, earlier) if arm == "memory" else ([], 0)
                    dropped += gone
                    leaks += program.memory_violations(r, mem)
                    calls.append({"row": r["row"], "record": r["record"], "memory": mem})
                if arm == "memory":
                    print(f"[{tag} {name}] memory: {len(calls)} blocks, {dropped} earlier reasons dropped for a date "
                          f"after the cut, {leaks} violations", flush=True)
                    if leaks:
                        raise rot.Stop(f"a memory block at {point} names a date after its row's cut: stop")
                got: dict[str, dict] = {}
                for i in range(0, len(calls), c.CHUNK):
                    part = calls[i:i + c.CHUNK]
                    o_usd, s_usd = optimizer_spend(self.lay)
                    rot.check_spend(self.lay, self.run_id, len(part), self.state, cap=CLIMB_BUDGET_USD,
                                    reserve=o_usd + s_usd + OPTIMIZER_RESERVE_USD)
                    label = f"{inputs.batch_id}.{name}.c{i // c.CHUNK:03d}"
                    path = self.lay.batches / f"{self.run_id}-{tree.name}-{label}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"arm": arm, "point": point, "calls": part}), encoding="utf-8")
                    arts = self.inner.run(trees[arm], RunInputs(input_ref=str(path), batch_id=label, split=split))
                    ref = Path(arts.artifact_refs[0].path)
                    part_answers = json.loads(ref.read_text(encoding="utf-8"))["answers"]
                    got.update(part_answers)
                    merged.update({f"{r}.{arm}": {**a, "pass_dir": str(ref.parent),
                                                  "memory_block": program.memory_text(x["memory"])}
                                   for x in part for r, a in [(x["row"], part_answers.get(x["row"], {"error": "missing"}))]})
                answers[name] = got
        for r in program.point_rows(c.POINT_KEYS[0], split, patients):  # the shared treatment-start answers
            merged[f"{r['row']}.memory"] = {**merged[f"{r['row']}.fresh"], "shared_with": "fresh"}
        after, calls = rot.spend(self.lay, self.run_id)
        print(f"[{tag}] {len(merged)} answers, {sum(1 for a in merged.values() if a.get('error'))} failed; this pass "
              f"${after - before:.2f}; climb so far ${after:.2f} over {calls} calls", flush=True)
        return merged, after - before


# ==================================================================================================
# Scorer and mistakes
# ==================================================================================================


def _merged(artifacts: RunArtifacts) -> dict:
    return json.loads(Path(artifacts.artifact_refs[0].path).read_text(encoding="utf-8"))["answers"]


def cells(answers: dict, split: str) -> dict[tuple, dict]:
    """(point, arm, horizon words) -> metrics over this pass's labelled rows (failed answers at 0.5)."""
    rows, labels = program.split_rows(split), program.labels()
    out = {}
    for point in c.POINT_KEYS:
        for arm in program.ARMS:
            for key, field, words in c.HORIZONS:
                pairs = [(labels[r][field], sc.risk(answers.get(f"{r}.{arm}"), key)) for r, x in rows.items()
                         if x["point"] == point and f"{r}.{arm}" in answers and labels[r][field] is not None]
                if pairs and 0 < sum(g for g, _p in pairs) < len(pairs):
                    out[(point, arm, words)] = sc.metrics(pairs)
    return out


def train_threshold() -> dict[str, float]:
    """The TRAIN death rate per horizon over every TRAIN row: only picks which rows are marked mistakes."""
    labels = program.labels()
    out = {}
    for key, field, _w in c.HORIZONS:
        g = [labels[r][field] for r in program.split_rows("train") if labels[r][field] is not None]
        out[key] = sum(g) / len(g) if g else 0.5
    return out


class ClimbScorer:
    """Mean AUROC over the 20 point x arm x horizon cells; the breakdown has every cell's AUROC, Brier,
    observed/expected and calibration slope (counts and short keys only). Each VAL round is also printed and kept in
    the work folder's private/ (never the optimizer's)."""

    def score(self, artifacts: RunArtifacts, split: str, task_config: dict) -> ScoreResult:
        answers = _merged(artifacts)
        cs = cells(answers, split)
        value = sum(m["auroc"] for m in cs.values()) / len(cs) if cs else 0.5
        breakdown = {"n_answers": len(answers), "program_errors": sum(1 for a in answers.values() if a.get("error")),
                     "cells": len(cs)}
        for (point, arm, words), m in cs.items():
            breakdown[f"{point}_{arm}_{words.replace(' ', '')}"] = {
                "n": m["n"], "deaths": m["deaths"], "auroc": round(m["auroc"], 4), "brier": round(m["brier"], 4),
                "obs_exp": None if m["obs_exp"] is None else round(m["obs_exp"], 3),
                "cal_slope": None if m["cal_slope"] is None else round(m["cal_slope"], 3)}
        if split == "val":
            n = task_config.get("_iter")
            print(f"VAL round {n}: mean AUROC {value:.4f} over {len(cs)} cells")
            print(f"  {'point':<10}{'arm':<8}{'horizon':<9}{'n':>5}{'AUROC':>8}{'Brier':>8}{'O/E':>7}{'slope':>7}")
            for (point, arm, words), m in cs.items():
                print(f"  {point:<10}{arm:<8}{words:<9}{m['n']:>5}{m['auroc']:>8.3f}{m['brier']:>8.4f}"
                      f"{sc.fmt(m['obs_exp'], 2):>7}{sc.fmt(m['cal_slope'], 2):>7}")
            log = WORKDIR / "private" / "climb" / "val_rounds.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as f:
                f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "iter": n, "auroc": value,
                                    "answers": artifacts.artifact_refs[0].path,
                                    "cells": {"|".join(k): v for k, v in cs.items()}}) + "\n")
        return ScoreResult(primary_metric=PrimaryMetric(name="auroc", value=value, higher_is_better=True),
                           breakdown=breakdown, task_config={**task_config, "_split": split})

    def per_example_correct(self, artifacts: RunArtifacts, split: str) -> dict:
        """Per TRAIN patient (the schedule's example id): right only when every row, arm and labelled horizon is on
        the right side of the TRAIN death rate; None when any answer failed."""
        answers, labels, rows, thr = _merged(artifacts), program.labels(), program.rows(), train_threshold()
        by: dict[str, list] = {}
        for k, a in answers.items():
            row = k.rsplit(".", 1)[0]
            by.setdefault(rows[row]["patient"], []).append(
                None if a.get("error") else all((a[key] >= thr[key]) == (labels[row][field] == 1)
                                                for key, field, _w in c.HORIZONS if labels[row][field] is not None))
        return {p: (None if None in v else all(v)) for p, v in by.items()}


def mistake_corpus(feedback_root: Path):
    """After each TRAIN pass: one entry per row and arm (died-2yr-ehr's probability entry, plus point, arm and the
    memory block), mistakes marked; each mistake row's record copied once as <row>-<file>, each mistake's findings
    note and transcript as <row>.<arm>-trace-<file>. Only TRAIN reaches here."""

    def build(artifacts: RunArtifacts, score: ScoreResult) -> MistakeCorpus:
        if score.task_config.get("_split", "train") != "train":
            raise LoopStop("the mistake corpus was asked to write a non-TRAIN pass", reason="corpus_split")
        n = score.task_config.get("_iter", 0)
        out = Path(feedback_root) / "iter" / str(n)
        records = out / "patients"
        records.mkdir(parents=True, exist_ok=True)
        rows, labels, thr = program.rows(), program.labels(), train_threshold()
        entries, copied = [], {}
        for k, a in sorted(_merged(artifacts).items()):
            row, arm = k.rsplit(".", 1)
            if rows[row]["split"] != "train":
                raise LoopStop("a non-TRAIN row reached the mistake corpus", reason="corpus_split")
            wrong_at = [words for key, field, words in c.HORIZONS if labels[row][field] is not None and not a.get("error")
                        and (a[key] >= thr[key]) != (labels[row][field] == 1)]
            mistake = bool(a.get("error")) or bool(wrong_at)
            files, traces = [], []
            if mistake:
                if row not in copied:
                    copied[row] = []
                    for f in program.agentic.record_files(config.DATA_DIR / "patients" / rows[row]["record"]):
                        dest = records / f"{row}-{f.name}"
                        ad._copy_over(f, dest)
                        copied[row].append(str(dest))
                files = copied[row]
                if a.get("pass_dir"):
                    fake = RunArtifacts(batch_id="x", status="ok",
                                        artifact_refs=(ArtifactRef(path=str(Path(a["pass_dir"]) / "answers.json"),
                                                                   sha256="0" * 64),))
                    log = Path(a["findings_log"]) if a.get("findings_log") else None
                    for f in ad._trace_files(fake, row) + ([log] if log and log.is_file() else []):
                        dest = records / f"{k}-trace-{f.name}"
                        ad._copy_over(f, dest)
                        traces.append(str(dest))
            entries.append({"id": k, "row": row, "point": rows[row]["point"], "arm": arm,
                            **{f"gold_{key}": labels[row][field] for key, field, _w in c.HORIZONS},
                            **{f"probability_{key}": (None if a.get("error") else round(sc.risk(a, key), 3))
                               for key, _f, _w in c.HORIZONS},
                            "error": a.get("error"), "reasons": a.get("reasons") or [],
                            "memory_block": a.get("memory_block") or "", "mistake": mistake, "wrong_at": wrong_at,
                            "record_files": files, "trace_files": traces})
        path = out / "train_mistakes.json"
        path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        counts = Counter(f"{e['point']}_{e['arm']}_" + ("mistake" if e["mistake"] else "right") for e in entries)
        return MistakeCorpus(ref=str(path), counts={"total": sum(e["mistake"] for e in entries), **counts})

    return build


# ==================================================================================================
# Data schedule: patients are the examples
# ==================================================================================================


class PatientSource:
    def __init__(self, batch_dir: Path) -> None:
        self.batch_dir = Path(batch_dir)

    def pool(self, split: str) -> list[str]:
        if split not in ("train", "val"):
            raise ValueError("the schedule never draws TEST")
        return sorted({r["patient"] for r in program.split_rows(split).values()})

    def stage(self, ids, split: str, label: str) -> RunInputs:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        path = self.batch_dir / f"{label}.json"
        path.write_text(json.dumps({"split": split, "ids": list(ids)}), encoding="utf-8")
        return RunInputs(input_ref=str(path), batch_id=label, split=split)


def schedule(lay):
    from engine.schedule import DataSchedule, ScheduleConfig

    n_val = len(PatientSource(lay.batches).pool("val"))
    cfg = ScheduleConfig(train_size=f"constant({TRAIN_SIZE})", val_size=f"constant({n_val})", train_mode="fresh",
                         seed=SCHEDULE_SEED, name="over-time")
    return DataSchedule(cfg, PatientSource(lay.batches))


# ==================================================================================================
# The optimizer (died-2yr-ehr's, with this task's brief, standing instructions and forbidden list)
# ==================================================================================================


def system_prompt() -> str:
    text = INSTRUCTIONS.read_text(encoding="utf-8")
    if config.SEARCH_PROXY:
        from engine.search_proxy import optimizer_instructions

        text = f"{text.rstrip()}\n\n{optimizer_instructions(SEARCH_REQUESTS_PER_ROUND)}"
    return text


def stage_brief(lay) -> None:
    lay.inputs.mkdir(parents=True, exist_ok=True)
    brief = (BRIEF.read_text(encoding="utf-8") + f"\n\n---\n\n# Appendix: `{program.agentic.GUIDE}`, as every session "
             "sees it\n\n" + program.GUIDE_TEXT)
    (lay.inputs / op.TASK_BRIEF_NAME).write_text(brief, encoding="utf-8")


def build_agent(lay, *, run_id: str, image: str, pin: str, search_pin: str, token: str | None):
    """Mirrors died-2yr-ehr's optimizer.build_agent and search_proxy with this task's forbidden list, brief and
    standing instructions; the model, effort, budget, route, prompt builder and search limits are that task's."""
    from engine.search_proxy import ProxiedOptimizer, SearchLimits, SearchProxy
    from isolation.contained_agent import ContainedOptimizerAgent, DeclaredOutput

    prof, credential = config.ROUTE.session_auth()
    host_env = {config.TOKEN_ENV_NAME: token} if token and not config.ROUTE.is_vertex else None
    proxy = SearchProxy(
        work_root=lay.search, log_root=lay.search_logs, forbidden=optimizer_forbidden(lay), profile=prof,
        credential=credential, image=image, expected_fingerprint=search_pin, model=config.PROGRAM_MODEL,
        host_env=host_env, env=opt.ENV, limits=SearchLimits(requests_per_round=SEARCH_REQUESTS_PER_ROUND, budget_usd=SEARCH_BUDGET_USD),
        brief_path=lay.inputs / op.TASK_BRIEF_NAME) if config.SEARCH_PROXY else None
    store = ad.TaskStore()
    agent = ContainedOptimizerAgent(
        model=config.OPTIMIZER_MODEL, system_prompt=system_prompt(), prompt_builder=op.build_prompt,
        manifest=store.manifest(), manifest_for_version=lambda sha: store.manifest(), repo_root=lay.repo,
        materialize_root=lay.versions, iter_root=lay.repo / "iter", forbidden=optimizer_forbidden(lay),
        profile=prof, credential=credential, image=image, expected_fingerprint=pin,
        transcript_dir=lay.transcripts / "optimizer",
        readable_inputs=((op.TASK_BRIEF_NAME, lay.inputs / op.TASK_BRIEF_NAME), (op.NOTEBOOK_NAME, lay.notebook)),
        declared_outputs=(DeclaredOutput(op.CHANGE_NOTE_NAME), DeclaredOutput(op.NOTEBOOK_NAME, destination=lay.notebook)),
        notes_root=lay.notes, run_id=run_id, train_feedback_roots=(lay.feedback,),
        validation_roots=(lay.private, c.DATA_DIR), run_summary_path=lay.summary(run_id),
        max_budget_usd=config.OPTIMIZER_SESSION_BUDGET_USD, effort=config.OPTIMIZER_EFFORT,
        web_search=config.OPTIMIZER_WEB_SEARCH, search_grant=proxy.grant if proxy else None,
        timeout_seconds=config.OPTIMIZER_TIMEOUT_SECONDS, host_env=host_env, env=opt.ENV,
        patient_data=config.PATIENT_DATA)
    return ProxiedOptimizer(agent, proxy) if proxy else agent


# ==================================================================================================
# Driver
# ==================================================================================================


def driver_config(lay, *, run_id, agent, runner, release_builder, setup_owners, unsealed=(), program_mode=None,
                  resume=False, start=StartPoint.NEWEST_GOOD) -> DriverConfig:
    """died-2yr-ehr's run.driver_config with this task's scorer, corpus and schedule, and no stop for lack of a new
    best (spec addendum: 10 rounds, no early stop; the dollar stops live in the runner)."""
    store = ad.TaskStore()
    sched = schedule(lay)
    lay.summary(run_id).parent.mkdir(parents=True, exist_ok=True)
    return DriverConfig(
        run_id=run_id, repo_root=lay.repo, iterations=ITERATIONS, program_store=store, scorer=ClimbScorer(),
        release_builder=release_builder, agent=agent, runner=runner, train_inputs=None, val_inputs=None,
        schedule=sched, materialize_root=lay.versions, manifest_for_version=lambda sha: store.manifest(),
        carried=(Carried(lay.repo / "iter", "run", stub=lay.stubs / "iter"),
                 Carried(lay.feedback, "run", stub=lay.stubs / "feedback"),
                 Carried(lay.notebook, "lineage", stub=lay.stubs / "notebook.md"),
                 Carried(lay.notes, "lineage", stub=lay.stubs / "notes")),
        archive_root=lay.archive, run_summary_path=lay.summary(run_id), run_store=CheckedRunStore(lay.store),
        patient_data=config.PATIENT_DATA, engine_pin=config.ENGINE_PIN, engine_root=config.ENGINE_ROOT,
        engine_capabilities=d2.CAPABILITIES, setup_owners=setup_owners, unsealed=unsealed, program_mode=program_mode,
        pin_command=f"python {HERE / 'climb.py'} pins --write", start=start, resume=resume,
        loop_options={"loop_ops": LocalLoopOps(lay.repo), "build_mistake_corpus": mistake_corpus(lay.feedback),
                      "metric_field": "auroc",
                      "hard_fields": {"task": TASK, "metric": "auroc", "schedule": {"train_size": TRAIN_SIZE}},
                      "soft_fields": {"optimizer_model": config.OPTIMIZER_MODEL},
                      "step_back_revert_tolerance": config.STEP_BACK_REVERT_TOLERANCE,
                      "token_budget": config.TOKEN_BUDGET})


def init(lay) -> None:
    """died-2yr-ehr's init_layout with this task's seed (the first pass's fresh instructions), then this brief."""
    config.SEED_PROGRAM_DIR = SEED_DIR  # read by d2.init_layout for this process only
    d2.init_layout(lay)
    stage_brief(lay)


def drive(cfg) -> int:
    try:
        result = run_driver(cfg)
    except (LoopStop, LoopPause) as exc:
        print(f"\nSTOPPED ({getattr(exc, 'reason', None) or type(exc).__name__}): {exc}")
        return exit_code(exc)
    print(f"\nFINISHED {len(result.results)} iterations. best {result.best_tag} = {result.best_metric_value}")
    return exit_code(result)


def cmd_init(_args) -> int:
    lay = layout()
    if lay.repo.exists():
        print(f"{lay.repo} already exists")
        return 1
    init(lay)
    return 0


def sealed(lay, *, run_id: str, pins: dict, token):
    image = d2.images()[0]
    agent = build_agent(lay, run_id=run_id, image=image, pin=pins["optimizer"], search_pin=pins.get("search"),
                        token=token)
    runner = rot.make_runner(lay, pin=pins["program"], image=d2.images()[1], cls=ClimbRunner)
    owners = {"optimizer": agent, "program": runner}
    if config.SEARCH_PROXY:
        owners["search"] = agent.proxy
    return agent, runner, owners


def cmd_pins(args) -> int:
    """died-2yr-ehr's run.py pins over this task's sessions."""
    from isolation.setup_fingerprint import PIN_KEY, UNPINNED, fingerprint_hash, platform_key, setup_fingerprint

    lay = layout()
    _a, _r, owners = sealed(lay, run_id="pins", pins=dict.fromkeys(d2.PIN_KINDS, UNPINNED), token=None)
    hashes = {k: fingerprint_hash(setup_fingerprint(o.setup_description())) for k, o in owners.items()}
    print(f"platform {platform_key()}: {json.dumps(hashes, indent=2)}")
    if not args.write:
        return 0
    path = lay.repo / d2.PIN_FILE
    data = json.loads(path.read_text()) if path.exists() else {}
    pins = data.setdefault("runtime_pins", {}).setdefault(PIN_KEY, {})
    for kind, value in hashes.items():
        pins.setdefault(kind, {})[platform_key()] = value
    path.write_text(json.dumps(data, indent=2) + "\n")
    d2.git(lay.repo, "add", d2.PIN_FILE)
    d2.git(lay.repo, "commit", "-m", f"setup pins for {platform_key()}")
    print(f"committed {path}")
    return 0


def cmd_run(args) -> int:
    from isolation.program_mode import ProgramMode
    from isolation.setup_gate import check_setup

    lay = layout()
    op.ITERATIONS = ITERATIONS
    token, _ = d2.load_token()
    if not config.ROUTE.is_vertex and not token:
        print("no Claude token for the seat route (or set ALO_ROUTE=vertex)")
        return 1
    stage_brief(lay)
    agent, runner, owners = sealed(lay, run_id=args.run_id, pins=d2.committed_pins(lay),
                                   token=None if config.ROUTE.is_vertex else token)
    try:
        stamp = check_setup(owners, program_mode=ProgramMode.AGENTIC, runner=runner).value
    except LoopStop as exc:
        print(f"STOPPED ({exc.reason}): {exc}")
        return exit_code(exc)
    print(f"climb {args.run_id}: {ITERATIONS} rounds, TRAIN {TRAIN_SIZE} patients a round, hard stop "
          f"${CLIMB_BUDGET_USD:.0f} (the optimizer's spend counted, ${OPTIMIZER_RESERVE_USD:.0f} kept for its next round), Haiku {config.PROGRAM_MODEL}, "
          f"optimizer {config.OPTIMIZER_MODEL}; round 0 from {ROUND0_DIR if ROUND0_DIR.is_dir() else 'nowhere (paid)'}")
    cfg = driver_config(lay, run_id=args.run_id, agent=opt.ChangeNoteAgent(agent), runner=runner,
                        release_builder=ad.TaskReleaseBuilder(stamp), setup_owners=owners,
                        program_mode=ProgramMode.AGENTIC, resume=args.resume)
    return drive(cfg)


def cmd_dry_run(args) -> int:
    """$0: the real loop, schedule, runner, scorer, corpus and release builder; fake patient sessions
    (fake_sessions) and died-2yr-ehr's ScriptedAgent for the optimizer. Its own throwaway work folder."""
    import fake_sessions
    from isolation.setup_fingerprint import UNPINNED

    global ITERATIONS
    ITERATIONS = args.iterations
    lay = layout()
    if args.resume:  # resume a stopped dry run in place (the resume test)
        if not (lay.root / d2.DRY_RUN_MARKER).is_file():
            print(f"{lay.root} is not a dry-run folder")
            return 1
    else:
        if lay.root.exists():
            if not (lay.root / d2.DRY_RUN_MARKER).is_file():
                print(f"{lay.root} exists and is not a dry-run folder; not deleting it")
                return 1
            d2.remove_tree(lay.root)
        lay.root.mkdir(parents=True)
        (lay.root / d2.DRY_RUN_MARKER).write_text("climb dry run\n")
        init(lay)
    op.ITERATIONS = args.iterations
    runner = rot.make_runner(lay, pin=UNPINNED, image="alo-dry-run:0.0.0@sha256:" + "0" * 64,
                             fake=fake_sessions.World(), cls=ClimbRunner)
    agent = d2.ScriptedAgent(lay)
    cfg = driver_config(lay, run_id=args.run_id, agent=agent, runner=runner,
                        release_builder=ad.TaskReleaseBuilder(None), setup_owners={},
                        unsealed=("optimizer", "program"), start=StartPoint.SEED, resume=args.resume)
    code = drive(cfg)
    for name, good in agent.checks:
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
    dollars, calls = rot.spend(lay, args.run_id)
    print(f"dry run paid calls: {calls} (${dollars:.2f})")
    return code if all(g for _n, g in agent.checks) else 1


def round0_check(recorded: float) -> str:
    """The first pass's answers re-scored by the first pass's own scorer (score.score: per point x horizon x arm on
    every labelled VAL row), averaged over the 20 cells, beside the climb's recorded round-0 score."""
    ans = {f.stem: json.loads(f.read_text())["answers"] for f in ROUND0_DIR.glob("*.json")}
    out = sc.score(ans, program.split_rows("val"), program.labels(), {}, n_boot=1)
    cells_ = [t["auroc"] for t in out["table"] if t["rows"] == "all labelled"]
    mean = sum(cells_) / len(cells_)
    return (f"round 0 check: first-pass answers re-scored {mean:.6f} over {len(cells_)} cells, climb recorded "
            f"{recorded:.6f} -> {'MATCH' if abs(mean - recorded) < 1e-9 else 'DIFFER'}")


def cmd_report(args) -> int:
    log = WORKDIR / "private" / "climb" / "val_rounds.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    for r in rows:
        print(f"round {r['iter']}: VAL mean AUROC {r['auroc']:.4f}")
    if rows and ROUND0_DIR.is_dir():
        print(round0_check(rows[0]["auroc"]))
    o_usd, s_usd = optimizer_spend(layout())
    print(f"optimizer sessions ${o_usd:.2f}, search ${s_usd:.2f}")
    summary = layout().summary(args.run_id)
    if summary.exists():
        d = json.loads(summary.read_text())
        for row in d.get("iters", []):
            print(f"  iter {row.get('iter')} {row.get('tag')} {row.get('status')} {row.get('auroc')}")
        print(f"stop: {d.get('stop')}")
    dollars, calls = rot.spend(layout(), args.run_id)
    print(f"patient sessions: ${dollars:.2f} over {calls} calls")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("pins")
    p.add_argument("--write", action="store_true")
    d = sub.add_parser("dry-run")
    d.add_argument("--run-id", default="climb-dry")
    d.add_argument("--iterations", type=int, default=2)
    d.add_argument("--resume", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--resume", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    return {"init": cmd_init, "pins": cmd_pins, "dry-run": cmd_dry_run, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
