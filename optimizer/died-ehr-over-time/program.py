"""What one patient session is, at one decision point, in either arm: the call list, staging, the prompt (with the
memory arm's block of earlier answers), and the date checks that guard both against anything after the cut.

Reused from died-2yr-ehr (``agentic``): the record whitelist and staging rules (``record_files``), the read-only
copies, the answer format and its parsing (``program_spec(...).summarize`` and ``read_stamp`` under
ANSWER=probability: two 0-100 numbers plus 3-5 reasons), the session tools and file names. New here: calls are
decision-point rows (``<patient>-<point>``) whose record is ``patients/<record id>/`` (one per patient and cut day),
the prompt says the record stops on the decision point's CT day and asks about death within 1 and 2 years of that
day, the record guide's first line names that cut, and the memory arm's prompt carries its own earlier answers.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import common as c
import agentic  # died-2yr-ehr's (common put it on sys.path)
import config

ARMS = ("fresh", "memory")
#: The hill-climb (spec addendum, 2026-11-12): one shared instructions.md, edited by the optimizer, runs both arms;
#: the memory arm's copy gets this paragraph added after it, so it knows what the earlier-answers block is.
MEMORY_PARAGRAPH = ("You have assessed this patient before, at earlier points in their care; the prompt lists your "
                    "earlier probabilities and reasons, if any. Read what has happened in the record since your last "
                    "assessment, then give your updated probabilities.")
PROGRAM_DIRS = {"fresh": c.HERE / "program_fresh", "memory": c.HERE / "program_memory"}

# The died-2yr-ehr record guide, with its one sentence about the cut changed to this task's cut.
_GUIDE_FROM = "cut 7 days before their first cancer treatment"
_GUIDE_TO = "cut at the end of the day of a chest CT taken at a decision point in their care"
GUIDE_TEXT = agentic.GUIDE_SRC.read_text(encoding="utf-8")
if _GUIDE_FROM not in GUIDE_TEXT:
    raise SystemExit("died-2yr-ehr's record guide no longer names its cut as expected: re-check program.GUIDE_TEXT")
GUIDE_TEXT = GUIDE_TEXT.replace(_GUIDE_FROM, _GUIDE_TO)

#: Bump when the prompt wording changes, so a finished pass under the old wording is not served again. The guide's
#: hash is part of it.
PROMPT_TAG = f"died-ehr-over-time-v2-guide-{hashlib.sha256(GUIDE_TEXT.encode()).hexdigest()[:10]}"


# ==================================================================================================
# Data on the local working copy
# ==================================================================================================


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


@lru_cache(maxsize=None)
def rows() -> dict[str, dict]:
    """Every decision-point row: row id -> {"row", "patient", "point", "record", "cut", "split"} (cut = YYYY-MM-DD;
    split "val", or "train" for the hill-climb's TRAIN pool; rows written before the TRAIN pool existed are VAL)."""
    return {r["row"]: {"split": "val", **r} for r in _jsonl(c.DATA_DIR / "rows.jsonl")}


def split_rows(split: str = "val") -> dict[str, dict]:
    return {k: r for k, r in rows().items() if r["split"] == split}


@lru_cache(maxsize=None)
def labels() -> dict[str, dict]:
    """row id -> {"label_1yr", "label_2yr"} (None where not followed long enough). Never reaches a session."""
    return {r["row"]: r for r in _jsonl(c.DATA_DIR / "labels.jsonl")}


def point_rows(point: str, split: str = "val", patients: set | None = None) -> list[dict]:
    """One point's rows of one split (optionally only these patients), in a fixed order."""
    return sorted((r for r in rows().values() if r["point"] == point and r["split"] == split
                   and (patients is None or r["patient"] in patients)), key=lambda r: r["row"])


# ==================================================================================================
# Memory: the arm's own earlier answers for this patient
# ==================================================================================================


def memory_for(row: dict, earlier: dict[str, dict[str, dict]]) -> tuple[list[dict], int]:
    """The memory arm's entries for one row, and how many earlier reasons were dropped: for each earlier point this
    patient had (oldest first), its date and the arm's own parsed answer there. ``earlier`` maps point -> row id ->
    the answers.json entry. Points after this one, and other patients, never enter. A reason that names or carries a
    date after this row's cut is dropped from the block and counted, not a stop (Phil 2026-11-11)."""
    out, dropped = [], 0
    for point in c.POINT_KEYS[: c.POINT_KEYS.index(row["point"])]:
        prev = rows().get(f"{row['patient']}-{point}")
        if prev is None:
            continue
        a = earlier.get(point, {}).get(prev["row"])
        ok = a is not None and not a.get("error")
        out.append({
            "date": prev["cut"],
            "risk_1yr": round(100 * a["died_within_1_year"]) if ok else None,
            "risk_2yr": round(100 * a["died_within_2_years"]) if ok else None,
            "reasons": [],
        })
        for r in (a.get("reasons") or []) if ok else []:
            r = {k: r.get(k, "") for k in ("fact", "file", "date")}
            if dates_after(" ".join(r.values()), row["cut"]):
                dropped += 1
            else:
                out[-1]["reasons"].append(r)
    return out, dropped


def memory_text(entries: list[dict]) -> str:
    """The memory block the memory arm's prompt carries (empty when there is no earlier point)."""
    if not entries:
        return ""
    lines = ["Your own earlier assessments of this patient, oldest first. Each was made from the record as it stood "
             "on that day:"]
    for e in entries:
        if e["risk_1yr"] is None:
            lines.append(f"- On {e['date']}: no answer was recorded.")
            continue
        lines.append(f"- On {e['date']}: chance of death within 1 year {e['risk_1yr']}, within 2 years "
                     f"{e['risk_2yr']}. Your reasons then:")
        for r in e["reasons"]:
            lines.append(f"    - {r['fact']} ({r['file']}, {r['date']})")
    return "\n".join(lines)


# ==================================================================================================
# Leak checks: nothing dated after the row's cut, in the memory block or in the staged record
# ==================================================================================================

DATE = re.compile(r"(?<!\d)(\d{4})[-/](\d{2})[-/](\d{2})(?!\d)")


def dates_after(text: str, cut: str) -> int:
    """How many YYYY-MM-DD (or YYYY/MM/DD) dates in ``text`` fall after ``cut``."""
    return sum(1 for y, m, d in DATE.findall(text) if f"{y}-{m}-{d}" > cut)


def memory_violations(row: dict, entries: list[dict]) -> int:
    """Dates after this row's cut anywhere in its memory block as staged: the earlier points' dates and every date
    the kept reasons carry or mention. memory_for drops such reasons, so anything here is a leak and stops the run."""
    return dates_after(memory_text(entries), row["cut"]) + sum(e["date"] > row["cut"] for e in entries)


@lru_cache(maxsize=None)
def registry_text_date_columns() -> dict[str, frozenset]:
    """Registry file -> the allow-listed columns holding a date as text, read off died-2yr-ehr's registry allow-list
    (config.REGISTRY_ALLOWLIST, the exporter's input): a column X whose governing date (or the table's row_date) is
    X_date. The *_date columns themselves are DATE-typed and checked as such."""
    import yaml

    out = {}
    for table, spec in (yaml.safe_load(config.REGISTRY_ALLOWLIST.read_text()) or {}).items():
        cols = (spec or {}).get("columns") or {}
        govs = {g for g in cols.values() if g} | ({spec["row_date"]} if spec.get("row_date") else set())
        out["registry_" + table.removeprefix("registry_case_")] = frozenset(
            g.removesuffix("_date") for g in govs if g.endswith("_date") and g.removesuffix("_date") in cols)
    return out


def text_date(value) -> str | None:
    """A registry text date as YYYY-MM-DD: YYYYMMDD, YYYYMM (read as the 1st, as died-2yr-ehr's cohort SQL reads it)
    or YYYY-MM-DD; None when it is none of these or not a real date (blank, unknown codes)."""
    import datetime as dt

    v = str(value or "").strip()
    for fmt, size in (("%Y%m%d", 8), ("%Y%m", 6), ("%Y-%m-%d", 10)):
        if len(v) == size:
            try:
                return dt.datetime.strptime(v, fmt).date().isoformat()
            except ValueError:
                return None
    return None


def record_violations(record_dir: Path, cut: str) -> dict[str, int]:
    """Items dated after ``cut`` in one staged record, by kind: every DATE or TIMESTAMP column of every parquet, the
    registry files' text date columns (registry_text_date_columns), the note headers in notes.txt, and meta.json's
    cutoff (must equal the cut). Counts only."""
    import duckdb

    bad: dict[str, int] = {}
    meta = json.loads((record_dir / "meta.json").read_text())
    if meta.get("cutoff") != cut:
        bad["meta_cutoff_differs"] = 1
    con = duckdb.connect()
    for f in sorted(record_dir.glob("*.parquet")):
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{f}')").fetchall()
        for name, typ, *_ in cols:
            t = str(typ)
            if t == "DATE" or t.startswith("TIMESTAMP"):
                n = con.execute(f"SELECT count(*) FROM read_parquet('{f}') WHERE CAST(\"{name}\" AS DATE) > DATE '{cut}'"
                                ).fetchone()[0]
                if n:
                    bad["date_after_cut"] = bad.get("date_after_cut", 0) + n
            elif name in registry_text_date_columns().get(f.stem, ()):
                vals = con.execute(f"SELECT CAST(\"{name}\" AS VARCHAR) FROM read_parquet('{f}')").fetchall()
                n = sum(1 for (v,) in vals if (d := text_date(v)) and d > cut)
                if n:
                    bad["registry_text_date_after_cut"] = bad.get("registry_text_date_after_cut", 0) + n
    notes = record_dir / "notes.txt"
    if notes.exists():
        head = re.compile(r"^===== NOTE [^|]* \| (\d{4}-\d{2}-\d{2})")
        with notes.open(encoding="utf-8") as fh:
            n = sum(1 for line in fh if (m := head.match(line)) and m.group(1) > cut)
        if n:
            bad["note_after_cut"] = n
    con.close()
    return bad


# ==================================================================================================
# The program spec
# ==================================================================================================


def batch(inputs) -> dict:
    """The staged batch a pass runs: {"arm", "point", "calls": [{"row", "record", "memory"}]} (run_over_time.py)."""
    return json.loads(Path(inputs.input_ref).read_text(encoding="utf-8"))


def prompt_text(call, paths, pass_id: str) -> str:
    """The fixed wrapper around the arm's instructions.md: the died-2yr-ehr probability wrapper, with the question
    timed from the decision point's CT day (the record's last day), no cohort death rate (Phil 2026-11-11: no
    number about the cohort in either arm), and the memory arm's block added. Container paths only."""
    rec = f"{paths.call}/{agentic.RECORD}"
    keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key, _gold, _words in c.HORIZONS)
    memory = memory_text(call.payload.get("memory") or [])
    return (
        f"Follow the instructions in {paths.program}/{agentic.ENTRY}; any tools are in {paths.program}/tools/ (run "
        f"Python with python3 -B).\n\n"
        f"The questions: will this patient die, of any cause, within 1 year of the last day of their record? And "
        f"within 2 years? Give each as the chance, from 0 to 100, that it happens. The patient has a thoracic cancer "
        f"and is being treated for it. Their medical record is the folder {rec}/ and it runs through the day of "
        f"a chest CT taken at a decision point in their care: nothing dated after that day is in it.\n\n"
        + (f"{memory}\n\n" if memory else "")
        + f"The folder (read-only):\n"
        f"- notes.txt: every clinical note in date order, each starting with a header line that begins "
        f"'===== NOTE'. Search it with grep.\n"
        f"- one parquet file per table of the OMOP common data model: {', '.join(agentic.OMOP_TABLES)}. A table with "
        f"no rows for this patient may be absent.\n"
        f"- {', '.join(f'registry_{t}.parquet' for t in agentic.REGISTRY_TABLES)}: the cancer registry's record of "
        f"this cancer, limited to the fields allowed at that day.\n"
        f"- meta.json: the patient's opaque id and the record's last day (cutoff).\n"
        f"python3 has pandas, pyarrow and duckdb for the parquet files.\n\n"
        f"Read {paths.call}/{agentic.GUIDE} before the record: it says what each file holds, where each kind of fact "
        f"(stage, histology, biomarkers, treatment, imaging and reports, labs, smoking, visits) lives, how the "
        f"codes read, and which sources mislead.\n\n"
        f"Scratch files go in /tmp. Before answering, write {paths.call}/{agentic.FINDINGS}: "
        f"a short note of what you looked at and what you found that bears on the answer. Use no names, dates "
        f"or other identifiers in it. Then write {paths.call}/{agentic.OUTPUT} containing exactly one JSON object on "
        f'one line: {{{keys}, "reasons": [{{"fact": "<one fact from the record>", "file": "<the record file it is in, '
        f'e.g. notes.txt or measurement.parquet>", "date": "<its date, YYYY-MM-DD>"}}, ...], "pass_id": "{pass_id}"}}. '
        f"Each number is your probability in percent, 0 to 100; a number outside that range, or a word, counts as no "
        f"answer. In \"reasons\" list the 3 to 5 facts from the record that moved your probabilities most, most "
        f"important first, each with the file you read it in and the date it carries (every one is on or before the "
        f"record's last day)."
    )


def stage(call, folder, pass_id) -> None:
    """The record for this row's cut, read-only, through died-2yr-ehr's whitelist, plus this task's record guide.
    The call folder's own permissions stay as the runner set them (Vertex needs it writable by the container)."""
    dest = Path(folder) / agentic.RECORD
    dest.mkdir(parents=True)
    dest.chmod(0o755)
    for f in agentic.record_files(config.DATA_DIR / "patients" / call.payload["record"]):
        shutil.copy2(f, dest / f.name)
        (dest / f.name).chmod(0o444)
    (Path(folder) / agentic.GUIDE).write_text(GUIDE_TEXT, encoding="utf-8")
    (Path(folder) / agentic.GUIDE).chmod(0o444)


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    base = agentic.program_spec(forbidden)  # its answer parsing, stamp reader and run settings

    def calls_for(inputs):
        b = batch(inputs)
        # Never a label: the payload is the record id, the point and the memory block (hashed into the pass key, so
        # a pass is only reused for the very same earlier answers).
        return [Call(x["row"], {"record": x["record"], "point": b["point"], "arm": b["arm"], "memory": x["memory"]})
                for x in b["calls"]]

    return ProgramSpec(
        calls_for=calls_for, stage=stage, prompt=lambda call, _stage, paths, pass_id: prompt_text(call, paths, pass_id),
        read_stamp=base.read_stamp, summarize=base.summarize, stages=base.stages,
        output_patterns=base.output_patterns, prompt_tag=PROMPT_TAG, max_concurrent=base.max_concurrent,
        timeout_seconds=base.timeout_seconds, max_budget_usd=base.max_budget_usd, forbidden=forbidden,
        token_env_name=base.token_env_name,
    )
