"""What the study has spent, from the work folder: every reader session's cost (the program runner's receipts, live
and archived passes), every builder session's (scripts/run_loop.py state), plus what was spent before the work folder
existed (--before: the infection pass). With --run-id, also one pass judged as a cost check: sessions, mean and
highest cost, how many reached the per-scan cap (95% of it or more) or timed out, and how many answers were not five
readable 0-100 numbers. Standard library only (reads config for the cap). Prints dollars and counts only.

    python -B scripts/spend.py WORK_FOLDER [--before 3.74] [--run-id val-pass-program-v0-look1] [--out F]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def receipts(work: Path):
    for f in sorted((work / "program-outputs").glob("*/*/receipts/*.json")):
        try:
            yield f, json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path)
    ap.add_argument("--before", type=float, default=0.0)
    ap.add_argument("--run-id")
    ap.add_argument("--cap", type=float, default=0.75)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    reader, sessions, one, loop_reader = 0.0, 0, [], 0.0
    answers_files = set()
    for f, r in receipts(args.work):
        stages = r.get("stages", [])
        cost = sum(float(s.get("cost_usd") or 0.0) for s in stages)
        reader += cost
        sessions += 1
        if str(r.get("run_id", "")).startswith("loop-"):
            loop_reader += cost
        if args.run_id and r.get("run_id") == args.run_id:
            one.append((cost, any(s.get("timed_out") for s in stages)))
            answers_files.add(f.parent.parent / "answers.json")
    builder = 0.0
    for st in (args.work / "private" / "loop").glob("*/state.json"):
        d = json.loads(st.read_text())
        builder += sum(float(x.get("builder_usd") or 0.0) for x in d.get("rounds", {}).values())
        builder += sum(float(x.get("builder_usd") or 0.0) for x in d.get("discovery", {}).values())  # library arm
    out = {"reader_usd": round(reader, 2), "reader_sessions": sessions, "builder_usd": round(builder, 2),
           "loop_usd": round(builder + loop_reader, 2),
           "before_usd": args.before, "total_usd": round(reader + builder + args.before, 2)}
    if args.run_id:
        unreadable = 0
        for a in answers_files:
            if a.is_file():
                unreadable += sum(1 for v in json.loads(a.read_text())["answers"].values() if v.get("error"))
        n = len(one)
        out["pass"] = {"run_id": args.run_id, "sessions": n,
                       "mean_usd": round(sum(c for c, _t in one) / n, 4) if n else None,
                       "max_usd": round(max((c for c, _t in one), default=0.0), 4),
                       "at_cap": sum(c >= 0.95 * args.cap for c, _t in one), "timed_out": sum(t for _c, t in one),
                       "unreadable": unreadable, "finished": all(a.is_file() for a in answers_files) and bool(answers_files)}
    print(json.dumps(out))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
