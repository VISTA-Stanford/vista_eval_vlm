"""$0 check of the reasons a probability-mode session gives (Phil 2026-10-31, spec
planning/vista_eval_vlm/2026-10-31-died-2yr-ehr-probability.md): for each stated reason {"fact", "file", "date"},
does it hold up against that patient's record?

    file_ok    the named file is one the session was given (a record file, or summary.txt in the summary arm)
    date_ok    the date reads as YYYY-MM-DD and is on or before the record's cutoff (meta.json)
    found      the named file has something on that date: a note whose header carries it (notes.txt), a row whose
               date or datetime column falls on it (a parquet table), or the date written in it (summary.txt)

It cannot check that the fact itself says what the record says on that day; "found" is the strongest claim it
makes. Prints counts only, never an id, a fact or a date.

    python scripts/check_reasons.py <answers.json> [<answers.json> ...] [--out <file.json>]
    python scripts/check_reasons.py --run <work folder> [--out <file.json>]   every pass's answers.json in a run

Reads DATA_DIR (TASK_DATA_DIR) for the records, like the task.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NOTE_HEADER = re.compile(r"^===== NOTE [^|]*\| (\d{4}-\d{2}-\d{2})")


def as_date(text: str) -> date | None:
    text = (text or "").strip()[:10]
    if not DATE.match(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


@lru_cache(maxsize=4096)
def dates_in(path: str) -> frozenset:
    """Every calendar date a record file carries, as ISO strings."""
    p = Path(path)
    if p.name == "notes.txt":
        return frozenset(m.group(1) for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
                         if (m := NOTE_HEADER.match(line)))
    if p.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
        out = set()
        for col in df.columns:
            if col.endswith(("_date", "_datetime")) or str(df[col].dtype).startswith("datetime"):
                out |= {str(v)[:10] for v in pd.to_datetime(df[col], errors="coerce").dropna()}
        return frozenset(out)
    return frozenset(re.findall(r"\d{4}-\d{2}-\d{2}", p.read_text(encoding="utf-8", errors="replace")))


def check_answers(paths: list[Path]) -> dict:
    import config
    import agentic
    from adapters import patient_dir

    allowed = set(agentic.RECORD_FILES) | ({agentic.SUMMARY} if config.WITH_SUMMARY else set())
    c = {"answers": 0, "answers_with_reasons": 0, "reasons": 0, "file_ok": 0, "date_ok": 0, "found": 0,
         "all_three": 0, "after_cutoff": 0}
    for path in paths:
        for pid, a in json.loads(Path(path).read_text(encoding="utf-8"))["answers"].items():
            c["answers"] += 1
            reasons = a.get("reasons") or []
            c["answers_with_reasons"] += bool(reasons)
            if not reasons:
                continue
            folder = patient_dir(pid)
            cutoff = as_date(json.loads((folder / "meta.json").read_text(encoding="utf-8")).get("cutoff", ""))
            for r in reasons:
                c["reasons"] += 1
                name = Path(r.get("file", "")).name
                file_ok = name in allowed
                src = (config.SUMMARY_DIR / f"{pid}.txt") if name == agentic.SUMMARY else folder / name
                file_ok = file_ok and src.is_file()
                d = as_date(r.get("date", ""))
                date_ok = d is not None and cutoff is not None and d <= cutoff
                c["after_cutoff"] += d is not None and cutoff is not None and d > cutoff
                found = file_ok and d is not None and d.isoformat() in dates_in(str(src))
                c["file_ok"] += file_ok
                c["date_ok"] += date_ok
                c["found"] += found
                c["all_three"] += file_ok and date_ok and found
    n = c["reasons"] or 1
    return {**c, **{f"share_{k}": round(c[k] / n, 3) for k in ("file_ok", "date_ok", "found", "all_three")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="*", type=Path)
    ap.add_argument("--run", type=Path, help="a work folder: every answers.json under its program outputs")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    paths = list(args.answers)
    if args.run:
        paths += sorted((args.run / "program-outputs" / "archive").glob("*/answers.json"))  # where passes land
    if not paths:
        sys.exit("no answers.json given or found")
    result = {"files": len(paths), **check_answers(paths)}
    if len(paths) > 1:  # one row per pass (its folder name), so the share can be read round by round
        result["per_pass"] = {p.parent.name: check_answers([p]) for p in paths}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
