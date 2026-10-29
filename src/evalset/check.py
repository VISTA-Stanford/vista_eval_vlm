"""Judge a check run: every (case, question, arm) answered and parsed. Exit 1 otherwise.

    python -m evalset.check --recipe astra --run check5 --steps 1 2 3 4 --limit 5

Prints counts, errors by message (no identifiers), refusals and token totals per step and arm.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from evalset import packs, samples
from evalset.cards import CARDS
from evalset.recipes import get
from evalset.run import CHOICES, out_path


def check(recipe_name: str, run: str, steps, limit: int | None, questions=None, root=None, arms=None, variant=None,
          which: str = "sample") -> bool:
    """Every (case, question) this invocation asked has a parsed answer. Only those rows are counted, so
    earlier rows in the same answers file (another limit or question set) neither help nor hurt."""
    from evalset.run import recipe_for
    recipe = recipe_for(recipe_name, arms, variant)[0]  # refuses --variant without --arms
    ok = True
    for step in steps:
        card = CARDS[step]
        qids = {q.id for q in card.questions if not questions or q.id in questions}
        if not qids:
            continue
        cases = samples.load(step, root, which)
        if limit:
            cases = packs.check_cases(cases, step, limit)
        case_ids = set(cases["case_id"].astype(str))
        for arm in card.arms:
            want = len(case_ids) * len(qids)
            p = out_path(run, recipe, step, arm, root)
            if not p.exists():
                print(f"FAIL step {step} {arm}: no answers file")
                ok = False
                continue
            df = pd.read_csv(p, dtype={"case_id": str})
            df = df[df["case_id"].isin(case_ids) & df["question_id"].isin(qids)]
            parsed = int(df["answer"].isin(CHOICES).sum())
            tok = {c.replace("usage_", ""): int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
                   for c in df.columns if c.startswith("usage_") and "tokens" in c}
            errs = Counter(str(e)[:120] for e in df["error"].dropna())
            line = (f"step {step} {arm}: {len(df)}/{want} rows, {parsed} parsed, {int(df['refused'].sum())} refused; "
                    f"tokens {tok}")
            good = len(df) == want and parsed == want
            print(("PASS " if good else "FAIL ") + line)
            for msg, n in errs.most_common(5):
                print(f"    error x{n}: {msg}")
            ok &= good
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=None, help="first N cases per step (default: whole sample)")
    ap.add_argument("--questions", nargs="+", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--which", choices=["sample", "test"], default="sample")
    a = ap.parse_args()
    ok = check(a.recipe, a.run, a.steps, a.limit, a.questions, arms=a.arms, variant=a.variant, which=a.which)
    print("ALL PARSED" if ok else "NOT ALL PARSED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
