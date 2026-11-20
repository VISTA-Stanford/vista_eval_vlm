"""Export agent and LLM answer runs to the paper's one-table-per-run layout (paper-code stage 2, 2026-11-20).

The layout is vista-eval's ``configs/paper/paper_v1_7.json`` -> ``agent_answers_layout`` and is read by
``vistaeval.results.scores.load_agent_answers``:

    <out_root>/<run_id>/answers.parquet   one row per (patient, task[, case])
      person_id, task, program_version, run_id, split, label_table     required
      probability (0-1) and/or answer ('yes' / 'no' / class)          at least one
      case_id (the scan or decision-point row, when a patient has several), question_id, step,
      label (the run's own answer key), horizon_years, failed (agents: errored / unreadable answer, scored 0.5),
      refused (general LLMs: the model declined)                      optional
    <out_root>/<run_id>/answers.provenance.json   input file hashes, the program version's source, git commit

Two kinds of run:
- general-purpose LLM runs of the shared eval set (``evalset_v1/runs/<set>/<recipe>/step<N>_<arm>.csv``):
  ``export_evalset_file``; program_version = the recipe (its prompt and model), run_id names the file.
- hill-climbed agents: ``export_agent_answers``; program_version is ``<climb run>/<tag>``, checked against the
  climb's run_summary.json by ``check_climb_version`` (winner = the best validation score it measured, ``winner_tag``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from evalset.score import prepare

OUT_ROOT = Path("/mnt/su-vista-uscentral1/vistabench/results/paper_agent_answers_v1_7")
EVALSET_RUNS = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/runs")
BENCH = "som-nero-plevriti-deidbdf.vista_bench_v1_7"
#: The table each eval-set step's answer key was drawn from (src/evalset/samples.py).
STEP_LABEL_TABLE = {
    1: f"{BENCH}.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr",
    2: f"{BENCH}.per_ct_diagnostic_tasks",
    3: f"{BENCH}.radiology_progression_tasks",
}
REQUIRED = ["person_id", "task", "program_version", "run_id", "split", "label_table"]
_REPO = Path(__file__).resolve().parents[2]


def horizon_years(task: str) -> int | None:
    m = re.search(r"_(\d+)_yr", task or "")
    return int(m.group(1)) if m else None


def sample_note(step: int, root: Path = EVALSET_RUNS.parent) -> str:
    """'built <date>, code <commit>' of the step's whole-TEST sample, for the label_table column."""
    f = root / f"step{step}_test.json"
    if not f.exists():
        return ""
    meta = json.loads(f.read_text())
    return f" (evalset_v1 step {step} test sample built {meta.get('built')}, code {meta.get('code_commit')})"


def export_evalset_file(
    path: Path, runs_root: Path = EVALSET_RUNS, splits: Iterable[str] = ("test",)
) -> pd.DataFrame:
    """One eval-set answers CSV -> layout rows. Probability runs give ``probability``; yes/no runs give ``answer``."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={"case_id": str, "person_id": str})
    a = prepare(raw)
    a = a[a["split"].isin(list(splits))]
    rel = path.relative_to(runs_root).with_suffix("")
    yesno = bool(a["yesno"].all()) if len(a) else True
    step = int(a["step"].iloc[0]) if len(a) else None
    out = pd.DataFrame(
        {
            "person_id": a["person_id"].astype(str),
            "task": a["task"],
            "case_id": a["case_id"].astype(str),
            "question_id": a["question_id"],
            "step": a["step"].astype(int),
            "probability": np.nan if yesno else a["score"].astype(float),
            "answer": a["answer"].where(a["answer"].isin(["yes", "no"]), None)
            if yesno
            else None,
            "refused": a["refused"].astype(bool),
            "label": a["label"].astype(int),
            "program_version": a["recipe"],
            "run_id": "evalset_v1/" + str(rel),
            "split": a["split"],
            "label_table": (STEP_LABEL_TABLE.get(step, "") + sample_note(step))
            if step
            else "",
        }
    )
    out["horizon_years"] = [horizon_years(t) for t in out["task"]]
    out.attrs["sources"] = [str(path)]
    out.attrs["version_source"] = (
        "recipe name in the answers file (a general-purpose LLM run has no climb)"
    )
    return out.reset_index(drop=True)


def climb_scores(run_summary: Path) -> list[tuple[str, float]]:
    """(version tag, validation metric) for every version a climb measured, in the order it measured them: the
    starting version's score (the engine records it under ``regrades``) first, then each round in ``iters``."""
    s = json.loads(Path(run_summary).read_text())
    out = []
    for r in s.get("regrades", []):
        if isinstance(r.get("metric"), (int, float)):
            out.append((str(r["tag"]), float(r["metric"])))
    for it in s.get("iters", []):
        v = it.get("auroc", it.get("metric"))
        if isinstance(v, (int, float)) and it.get("tag"):
            out.append((str(it["tag"]), float(v)))
    return out


def winner_tag(run_summary: Path) -> tuple[str, float]:
    """The version a climb picked: the best validation score among everything it measured, the earliest on a tie.
    The engine's run_summary.json has no winner field, so this is the one rule every export uses (never typed)."""
    scores = climb_scores(run_summary)
    if not scores:
        raise ValueError(f"{run_summary} records no measured version")
    best = max(m for _, m in scores)
    return next((t, m) for t, m in scores if m == best)


def version_sha(run_summary: Path, tag: str) -> str | None:
    """The climb record's version hash for a tag (start, regrades or iters), for a later check against the copied
    program bundles. The answers files carry no version metadata, so this is recorded, not compared."""
    s = json.loads(Path(run_summary).read_text())
    for r in [s.get("start", {}), *s.get("regrades", []), *s.get("iters", [])]:
        if r.get("tag") == tag and r.get("version_sha"):
            return str(r["version_sha"])
    return None


def check_climb_version(run_summary: Path, tag: str, role: str) -> str:
    """Refuse when a run's version is not what its role claims: ``winner`` must be the climb's winner, ``seed`` its
    starting version, ``other`` a version the climb measured. Returns the provenance sentence. The link from an
    answers file to its version rests on the file or folder name (the answers carry no version metadata)."""
    rs = Path(run_summary)
    measured = dict(climb_scores(rs))
    start = json.loads(rs.read_text())["start"]["tag"]
    if role == "winner":
        win, score = winner_tag(rs)
        if win != tag:
            raise ValueError(f"{rs}: the climb's winner is {win}, the run is {tag}")
        how = f"winner of {rs} (validation {score:.4f})"
    elif role == "seed":
        if start != tag:
            raise ValueError(f"{rs}: the climb started from {start}, the run is {tag}")
        how = f"starting version of {rs}" + (
            f" (validation {measured[tag]:.4f})" if tag in measured else ""
        )
    elif role == "other":
        if tag not in measured and tag != start:
            raise ValueError(f"{rs}: the climb never measured {tag}")
        how = f"a measured version of {rs}, not its winner" + (
            f" (validation {measured[tag]:.4f})" if tag in measured else ""
        )
    else:
        raise ValueError(f"unknown role {role!r}")
    sha = version_sha(rs, tag)
    return (
        how + f"; climb version hash {sha}; file-to-version link by file name"
        if sha
        else how
    )


def export_agent_answers(
    rows: pd.DataFrame,
    run_id: str,
    program_version: str,
    label_table: str,
    sources: list[str],
    version_source: str,
) -> pd.DataFrame:
    """Hill-climbed agent answers already flattened to (person_id, task, probability[, case_id, split, label])."""
    out = rows.copy()
    out["person_id"] = out["person_id"].astype(str)
    out["program_version"] = program_version
    out["run_id"] = run_id
    out["label_table"] = label_table
    if "split" not in out:
        out["split"] = "test"
    out["horizon_years"] = [horizon_years(t) for t in out["task"]]
    out.attrs["sources"] = list(sources)
    out.attrs["version_source"] = version_source
    return out.reset_index(drop=True)


def check_layout(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if not any(c in df and df[c].notna().any() for c in ("probability", "answer")):
        raise ValueError("neither probability nor answer is filled")
    if df["run_id"].nunique() != 1:
        raise ValueError("one run per table")
    key = ["person_id", "task"] + (["case_id"] if "case_id" in df else [])
    if df.duplicated(key).any():
        raise ValueError(f"duplicate rows on {key}")
    p = df["probability"].dropna() if "probability" in df else pd.Series(dtype=float)
    if len(p) and not p.between(0, 1).all():
        raise ValueError("probability outside 0-1")


def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git() -> dict:
    def run(*a):
        return subprocess.run(
            ["git", "-C", str(_REPO), *a], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "repo": "vista_eval_vlm",
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(
                run("status", "--porcelain", "--", "src", "scripts", "optimizer")
            ),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"repo": "vista_eval_vlm", "commit": None, "dirty": None}


def write_run(df: pd.DataFrame, out_root: Path = OUT_ROOT) -> Path:
    """Write ``<out_root>/<run_id>/answers.parquet`` and its provenance record; returns the parquet path."""
    check_layout(df)
    d = Path(out_root) / df["run_id"].iloc[0]
    d.mkdir(parents=True, exist_ok=True)
    f = d / "answers.parquet"
    tmp = d / "answers.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(f)
    prov = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": _git(),
        "inputs": [
            {"path": s, "sha256": _sha256(s)} for s in df.attrs.get("sources", [])
        ],
        "program_version": df["program_version"].iloc[0],
        "version_source": df.attrs.get("version_source"),
        "rows": len(df),
        "patients": int(df["person_id"].nunique()),
    }
    (d / "answers.provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    return f


# ---------------------------------------------------------------------------- agent answer readers
EHR_KEYS = {
    "died_within_1_year": "died_any_cause_1_yr",
    "died_within_2_years": "died_any_cause_2_yr",
}
QUESTION_TASK = {"death_1yr": "died_any_cause_1_yr", "death_2yr": "died_any_cause_2_yr"}


def agent_prob(a, key):
    """The paper's rule for one agent answer: a failed, missing or yes/no-typed answer scores 0.5."""
    v = (a or {}).get(key)
    failed = (
        (not a)
        or bool(a.get("error"))
        or isinstance(v, bool)
        or not isinstance(v, (int, float))
    )
    return (0.5, True) if failed else (float(v), False)


def ehr_agent_rows(answers_json: Path, idmap: Path, split_jsonl: Path) -> pd.DataFrame:
    """died-2yr-ehr agent answers ({"answers": {case_id: {...}}}) -> one row per (patient, horizon)."""
    ids = {
        json.loads(x)["id"]: str(json.loads(x)["person_id"])
        for x in Path(idmap).read_text().splitlines()
    }
    gold = {}
    for x in Path(split_jsonl).read_text().splitlines():
        r = json.loads(x)
        gold[ids[r["id"]]] = {
            "died_any_cause_1_yr": r["label_1yr"],
            "died_any_cause_2_yr": r["label"],
        }
    rows = []
    for cid, a in json.loads(Path(answers_json).read_text())["answers"].items():
        pid = ids[cid]
        for key, task in EHR_KEYS.items():
            p, failed = agent_prob(a, key)
            rows.append(
                {
                    "person_id": pid,
                    "task": task,
                    "case_id": cid,
                    "probability": p,
                    "failed": failed,
                    "label": gold.get(pid, {}).get(task),
                }
            )
    return pd.DataFrame(rows)


def over_time_rows(
    private_dir: Path,
    answers_dir: Path,
    split: str = "test",
    points=("start", "restaging", "m6", "m12", "m24"),
) -> pd.DataFrame:
    """Over-time agent answers (<point>-fresh.json keyed by decision-point row) -> one row per (row, horizon)."""
    pts = pd.read_parquet(Path(private_dir) / "points.parquet")
    pts = pts[pts["split"] == split].set_index("row")
    lab = pd.DataFrame(
        [
            json.loads(x)
            for x in (Path(private_dir) / "labels.jsonl").read_text().splitlines()
        ]
    ).set_index("row")
    ans = {}
    for p in points:
        f = Path(answers_dir) / f"{p}-fresh.json"
        if f.exists():
            ans.update(json.loads(f.read_text())["answers"])
    rows = []
    for row, a in ans.items():
        if row not in pts.index:
            continue
        for key, task in EHR_KEYS.items():
            p, failed = agent_prob(a, key)
            rows.append(
                {
                    "person_id": str(pts.at[row, "person_id"]),
                    "task": task,
                    "case_id": str(row),
                    "point": pts.at[row, "point"],
                    "probability": p,
                    "failed": failed,
                    "label": lab.at[
                        row, "label_1yr" if task.endswith("1_yr") else "label_2yr"
                    ]
                    if row in lab.index
                    else None,
                }
            )
    return pd.DataFrame(rows)


def ct_rows_file(csv: Path) -> pd.DataFrame:
    """The mount's derived CT agent rows (one row per patient and question, prob already 0-1)."""
    d = pd.read_csv(csv)
    if d.duplicated(["person_id", "question_id"]).any():
        raise ValueError(
            f"{csv} has several rows per patient and question (scan number missing); export from raw answers"
        )
    return pd.DataFrame(
        {
            "person_id": d["person_id"].astype(str),
            "task": d["question_id"].map(QUESTION_TASK),
            "question_id": d["question_id"],
            "probability": d["prob"].astype(float),
            "failed": d["refused"].astype(bool),
            "label": d["label"].astype(int),
        }
    )
