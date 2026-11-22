"""Judge a cost check: what each patient session cost, per pass, read from the program runner's receipts.

    python scripts/cost_report.py WORK_FOLDER [OUT_JSON]

Per pass (the seed's TRAIN pass, the first version's VAL pass, ...): patients, mean and highest cost, how many
reached the per-session cap (cost at 95% of it or more) or timed out, and how many answers were unreadable. Then the
extrapolation the hand-off asks for: a full round (the largest TRAIN pass plus VAL, at the mean session cost) and the
zero-shot VAL pass, against the round cap. PASS when fewer than 1 in 20 sessions reached the cap or timed out, no answer
was unreadable, every pass finished, and the extrapolated round is within the round cap. Prints counts and dollars
only, never an id or a record. tb-questions: copied from ct-findings-5 (2026-11-20).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

#: Past climbs stopped by the no-new-best rule at rounds 4-7; a run cap that ends the climb sooner needs Phil's go.
MIN_ROUNDS = 4


def main(argv: list[str]) -> int:
    import config
    import adapters as ad

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
    mean = total / sessions if sessions else 0.0
    n_train, n_val = len(ad.pools()["train"]), len(ad.pools()["val"])
    m = re.fullmatch(r"linear\((\d+),\s*(\d+),\s*over=(\d+)\)", str(config.CLIMB_SCHEDULE["train_size"]))
    a0, a1, over = (int(x) for x in m.groups())
    # A full round: the schedule's largest TRAIN batch (all of TRAIN for the patient agents; boards 600 of 2,916).
    n_train = min(n_train, a1)
    round_usd = mean * (n_train + n_val) + config.OPTIMIZER_SESSION_BUDGET_USD
    zero_shot_usd = mean * n_val
    within = round_usd <= config.ROUND_BUDGET_USD
    print(f"extrapolated at ${mean:.3f} a session: zero-shot VAL ({n_val}) ${zero_shot_usd:.2f}; a full round (TRAIN "
          f"{n_train} + VAL {n_val} + the ${config.OPTIMIZER_SESSION_BUDGET_USD:.0f} optimizer budget) ${round_usd:.2f} "
          f"against the ${config.ROUND_BUDGET_USD:.0f} round cap: {'within' if within else 'OVER: needs its own go'}")
    # Rounds that fit in the run cap on the climb's real schedule (TRAIN growing, VAL whole), ignoring retirement
    # (which only shrinks TRAIN, so this is a floor). Fewer than MIN_ROUNDS means the dollar cap, not the no-new-best
    # rule, would likely end the climb: that changes what it can conclude, so it needs Phil's go (hand-off 3).
    spent, fit = 0.0, 0
    for k in range(1, config.CLIMB_ITERATIONS + 1):
        t = min(a1, round(a0 + (a1 - a0) * (k - 1) / (over - 1)))
        spent += mean * (t + n_val) + config.OPTIMIZER_SESSION_BUDGET_USD
        if spent > config.RUN_BUDGET_USD:
            break
        fit = k
    enough = fit >= MIN_ROUNDS
    print(f"rounds that fit in the ${config.RUN_BUDGET_USD:.0f} run cap at that rate: {fit} of {config.CLIMB_ITERATIONS} "
          f"({'enough' if enough else f'FEWER THAN {MIN_ROUNDS}: the cap would end the climb, needs Phil' + chr(39) + 's go'})")
    ok = at_cap * 20 < sessions and unreadable == 0 and not stop and within and enough
    verdict = {"route": config.ROUTE.name, "answer": config.ANSWER, "cap_usd": cap, "sessions": sessions, "at_cap_or_timed_out": at_cap,
               "unreadable_or_failed": unreadable, "patient_sessions_usd": round(total, 2), "run_stop": stop,
               "mean_session_usd": round(mean, 4), "extrapolated_round_usd": round(round_usd, 2),
               "rounds_within_run_cap": fit, "run_cap_usd": config.RUN_BUDGET_USD,
               "extrapolated_zero_shot_val_usd": round(zero_shot_usd, 2), "round_cap_usd": config.ROUND_BUDGET_USD,
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
