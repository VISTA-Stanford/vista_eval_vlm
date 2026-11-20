"""Judge a cost check: what each patient session cost, per pass, read from the program runner's receipts.

    python scripts/cost_report.py WORK_FOLDER [OUT_JSON]

Per pass (the seed's TRAIN pass, the first version's VAL pass, ...): patients, mean and highest cost, how many
reached the per-patient cap (cost at 95% of it or more) or timed out, and how many answers were not a readable
yes/no. PASS when fewer than 1 in 20 sessions reached the cap or timed out, no answer was unreadable, and every pass
finished. Prints counts and dollars only, never an id or a record. Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> int:
    import config

    work = Path(argv[1])
    cap = config.PROGRAM_CALL_BUDGET_USD
    passes = sorted((work / "program-outputs" / "archive").glob("*/receipts"))
    if not passes:
        print(f"FAIL: no finished pass under {work}/program-outputs/archive")
        return 1
    rows, sessions, at_cap, unreadable, total = [], 0, 0, 0, 0.0
    for receipts in passes:
        costs, capped, timed_out = [], 0, 0
        for f in sorted(receipts.glob("*.json")):
            stages = json.loads(f.read_text(encoding="utf-8")).get("stages", [])
            cost = sum(float(s.get("cost_usd") or 0.0) for s in stages)
            costs.append(cost)
            capped += cost >= 0.95 * cap
            timed_out += any(s.get("timed_out") for s in stages)
        answers_file = receipts.parent / "answers.json"
        answers = json.loads(answers_file.read_text())["answers"] if answers_file.is_file() else {}
        bad = sum(1 for a in answers.values() if a.get("error"))
        row = {"pass": receipts.parent.name.rsplit("-", 1)[0], "patients": len(costs),
               "mean_usd": round(sum(costs) / len(costs), 4) if costs else None,
               "max_usd": round(max(costs), 4) if costs else None, "at_cap": capped, "timed_out": timed_out,
               "unreadable_or_failed": bad, "pass_usd": round(sum(costs), 4)}
        rows.append(row)
        sessions += len(costs)
        at_cap += capped + timed_out
        unreadable += bad
        total += sum(costs)
    for r in rows:
        print(f"  {r['pass']:<28} {r['patients']:>3} patients  mean ${r['mean_usd']}  max ${r['max_usd']}  "
              f"at cap {r['at_cap']}  timed out {r['timed_out']}  unreadable/failed {r['unreadable_or_failed']}")
    summary_file = next((work / "private" / "runs").glob("*/run_summary.json"), None)
    stop = json.loads(summary_file.read_text()).get("stop") if summary_file else "no run summary"
    ok = at_cap * 20 < sessions and unreadable == 0 and not stop
    verdict = {"route": config.ROUTE.name, "cap_usd": cap, "sessions": sessions, "at_cap_or_timed_out": at_cap,
               "unreadable_or_failed": unreadable, "patient_sessions_usd": round(total, 2), "run_stop": stop,
               "passes": rows, "ok": ok}
    print(f"patient sessions ${total:.2f} over {sessions}; run stop: {stop or 'none (finished)'}")
    print(f"{'PASS' if ok else 'FAIL'}: {at_cap} of {sessions} at the ${cap:.2f} cap or timed out (pass: under 1 in 20), "
          f"{unreadable} unreadable or failed (pass: 0)")
    if len(argv) > 2:
        Path(argv[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[2]).write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
        print(f"written {argv[2]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
