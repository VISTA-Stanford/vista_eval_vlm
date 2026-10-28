"""Judge a check run: every (case, question, arm) answered and parsed. Exit 1 otherwise.

    python -m evalset.check --recipe astra --run check5 --steps 1 2 3 4 --limit 5

Prints counts, errors by message (no identifiers), refusals and token totals per step and arm.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import pandas as pd

from evalset.cards import CARDS
from evalset.recipes import get
from evalset.run import CHOICES, out_path


def check(recipe_name: str, run: str, steps, limit: int, questions=None) -> bool:
    recipe = get(recipe_name)
    ok = True
    for step in steps:
        card = CARDS[step]
        n_q = len([q for q in card.questions if not questions or q.id in questions])
        if n_q == 0:
            continue
        for arm in card.arms:
            want = limit * n_q
            p = out_path(run, recipe, step, arm)
            if not p.exists():
                print(f"FAIL step {step} {arm}: no answers file")
                ok = False
                continue
            df = pd.read_csv(p, dtype={"case_id": str})
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
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--questions", nargs="+", default=None)
    a = ap.parse_args()
    ok = check(a.recipe, a.run, a.steps, a.limit, a.questions)
    print("ALL PARSED" if ok else "NOT ALL PARSED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
