"""How many patients' records already show a treatment plan before the cutoff (spec
planning/vista_eval_vlm/2026-11-21-tb-treatment-full-record.md, Done when: "Plan-note count"). $0, no model call.

Pre-board notes that state the plan stay in the record (Phil 2026-11-21, the same rule as the summary); this counts
them so a gain from the record can be read against it. For each patient of a split, over the last WINDOW_DAYS days of
the record (up to and including the cutoff), it marks:
    order     a notes.txt entry typed ``ORDER: ...`` whose text names a lung cancer treatment (TREATMENT_RE)
    consent   a note whose type or first 300 characters say consent, naming a treatment
    plan      a note with an assessment/plan section (the notes reader's pattern) naming a treatment in that section
    any       any of the three
and prints, per split and per question, how many patients have each. A name in a plan section can be an option
discussed, not the decision: this is an upper bound on "the record states the plan", not a count of leaked answers.

    cd optimizer/tb-questions && TB_AGENT=treatment-record uv run -q --no-project --with pandas python -B scripts/count_plan_notes.py --split val

Prints counts only: never an id, a date or note text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapters as ad  # noqa: E402
import config  # noqa: E402

WINDOW_DAYS = 60
TREATMENT_RE = re.compile(
    r"(?i)\b(lobectomy|segmentectomy|wedge resection|pneumonectomy|thoracotomy|vats|robotic|resection|sbrt|sabr|"
    r"stereotactic|chemoradiation|chemo-?radiation|radiation therapy|radiotherapy|carboplatin|cisplatin|pemetrexed|"
    r"paclitaxel|docetaxel|etoposide|pembrolizumab|nivolumab|atezolizumab|durvalumab|cemiplimab|ipilimumab|"
    r"osimertinib|erlotinib|gefitinib|afatinib|alectinib|lorlatinib|brigatinib|crizotinib|capmatinib|tepotinib|"
    r"sotorasib|adagrasib|selpercatinib|pralsetinib|dabrafenib|amivantamab|neoadjuvant|adjuvant)\b")
AP_RE = re.compile(r"(?i)(assessment\s*(?:and|&|/)\s*\n?\s*plan|impression\s*(?:and|&|/)\s*plan|"
                   r"assessment and recommendations|\ba/p\b|\bplan\s*:|\brecommendations?\s*:)")
HEADER = re.compile(r"===== NOTE (\S+) \| (\S+)[^|]*\| (.*) =====")


def notes(path: Path):
    """(date, type, text) per note of a notes.txt."""
    cur = None
    for line in path.open(encoding="utf-8", errors="replace"):
        m = HEADER.match(line)
        if m:
            if cur:
                yield cur
            cur = [m.group(2)[:10], m.group(3).strip(), []]
        elif cur is not None:
            cur[2].append(line)
    if cur:
        yield cur


def marks(pid: str) -> set[str]:
    d = ad.record_dir(pid)
    cutoff = pd.Timestamp(json.loads((d / "meta.json").read_text())["cutoff"]).normalize()
    lo = cutoff - pd.Timedelta(days=WINDOW_DAYS)
    out: set[str] = set()
    if not (d / "notes.txt").is_file():
        return out
    for date, typ, lines in notes(d / "notes.txt"):
        t = pd.to_datetime(date, errors="coerce")
        if pd.isna(t) or t < lo or t >= cutoff + pd.Timedelta(days=1):
            continue
        text = "".join(lines)
        if typ.upper().startswith("ORDER:") and TREATMENT_RE.search(typ + " " + text[:2000]):
            out.add("order")
        if re.search(r"(?i)consent", typ + " " + text[:300]) and TREATMENT_RE.search(text):
            out.add("consent")
        m = None
        for m in AP_RE.finditer(text):
            pass
        if m and TREATMENT_RE.search(text[m.start():m.start() + 2500]):
            out.add("plan")
    return out | ({"any"} if out else set())


def main() -> int:
    if not config.RECORD:
        raise SystemExit("set TB_AGENT=treatment-record")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val"), default="val")
    args = ap.parse_args()
    rows = ad.read_split(args.split)
    keys = ("order", "consent", "plan", "any")
    total, by_q, n_q = Counter(), {q.key: Counter() for q in config.QUESTIONS}, Counter()
    for r in rows:
        m = marks(r["id"])
        total.update(m)
        for k in ad.asked(r):
            by_q[k].update(m)
            n_q[k] += 1
    print(f"{args.split}: {len(rows)} patients; last {WINDOW_DAYS} days before the cutoff, patients with a note that "
          "names a lung cancer treatment as ...")
    print("  all patients: " + ", ".join(f"{k} {total[k]} ({total[k] / len(rows):.0%})" for k in keys))
    for q, c in by_q.items():
        print(f"  {q} ({n_q[q]} asked): " + ", ".join(f"{k} {c[k]} ({c[k] / max(n_q[q], 1):.0%})" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
