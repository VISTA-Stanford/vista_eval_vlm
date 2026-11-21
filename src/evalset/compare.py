"""Step 3 on the fixed 1,000 TEST pairs: every arm and the CT probes side by side (spec
planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md, Done when).

Rows: the three comparison probes (evalset.samples.PROBE_FILES, vista-eval's v1.7 current + prior pair per-example
files; their four-way probabilities collapsed to P(progression) + P(mixed), the card's two-way rule) and one row per
answers file given (an eval-set answers CSV with ``case_id`` and ``p_yes``, or the agent's export in the same shape).
Columns: AUROC with a 95% patient-bootstrap interval, the paired difference from Pillar0 on the same bootstrap draws,
answered / unreadable counts, distinct values and the share on the most common one. An unreadable or missing answer
counts as no information (0.5), the rule the agent climb uses, so every row scores all 1,000 pairs; the AUROC over
readable answers only is a column too. A second table breaks each row out by how the prior was found
(``prior_source``) and whether the link was ambiguous (``prior_ambiguous``).

    cd src && python -m evalset.compare --answers Argon=<runs>/radprog-t1000/argon__argon_rp/step3_ct.csv \\
        Opus=<...>/opus__opus_rp/step3_ct.csv --out <runs>/radprog-t1000/compare.csv

Prints aggregate numbers only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalset import paths, samples
from evalset.cards.progression import TWO_WAY
from evalset.score import _weighted_auroc, auroc

REFERENCE = "Pillar0"
YES_CODES = tuple(k for k, v in TWO_WAY.items() if v == 1)


def probe_scores(name: str, cases: pd.DataFrame) -> pd.Series:
    """P(yes) per case_id for one probe: the four-way probabilities of the yes codes summed, TEST rows only."""
    pe = pd.read_parquet(paths.mount_root() / paths.BUCKET / samples.PROBE_DIR / samples.PROBE_FILES[name])
    pe = pe[(pe["task"] == "radiology_progression") & (pe["split"] == "test")]
    pe = pe.drop_duplicates("accession_number").set_index(pe["accession_number"].astype(str))
    classes = [int(c) for c in pe["classes"].iloc[0]]
    proba = np.stack(pe["proba"].to_numpy())
    p_yes = sum(proba[:, classes.index(c)] for c in YES_CODES if c in classes)
    s = pd.Series(p_yes, index=pe.index)
    lab = pe["label"].astype(int).map(TWO_WAY)
    want = cases.set_index("case_id")
    if not (lab.reindex(want.index) == want["label__radiology_progression"]).all():
        raise RuntimeError(f"{name}: the probe file's labels disagree with the 1,000 list's")
    return s.reindex(want.index)


def answer_scores(path: str, cases: pd.DataFrame) -> pd.Series:
    """P(yes) per case_id from an answers CSV (``case_id``, ``p_yes``); NaN where unreadable or missing."""
    a = pd.read_csv(path, dtype={"case_id": str})
    if a["case_id"].duplicated().any():
        raise RuntimeError(f"{path}: more than one answer per case")
    extra = set(a["case_id"]) - set(cases["case_id"])
    if extra:
        raise RuntimeError(f"{path}: {len(extra)} answers are not in the 1,000 list")
    return pd.to_numeric(a.set_index("case_id")["p_yes"], errors="coerce").reindex(cases["case_id"].values)


def table(scores: dict[str, pd.Series], cases: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    y = cases["label__radiology_progression"].astype(int).to_numpy(dtype=float)
    pcode, _ = pd.factorize(cases["person_id"].astype(str))
    n_p = int(pcode.max()) + 1
    rng = np.random.default_rng(seed)
    weights = [np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float)[pcode] for _ in range(n_boot)]
    filled = {k: v.fillna(0.5).to_numpy(dtype=float) for k, v in scores.items()}
    draws = {}
    for k, s in filled.items():
        u, inv = np.unique(s, return_inverse=True)
        draws[k] = np.array([_weighted_auroc(inv, len(u), y, w) for w in weights])
    rows = []
    for k, s in scores.items():
        ok = s.notna().to_numpy()
        vals = s[ok].round(6)
        rec = {"row": k, "n": len(s), "answered": int(ok.sum()), "unreadable_or_missing": int((~ok).sum()),
               "auroc": auroc(filled[k], y), "auroc_lo": float(np.percentile(draws[k], 2.5)),
               "auroc_hi": float(np.percentile(draws[k], 97.5)),
               "auroc_readable_only": auroc(s[ok].to_numpy(dtype=float), y[ok]) if ok.any() else float("nan"),
               "distinct_values": int(vals.nunique()),
               "top_value_share": float(vals.value_counts(normalize=True).iloc[0]) if len(vals) else float("nan")}
        if REFERENCE in draws and k != REFERENCE:
            d = draws[k] - draws[REFERENCE]
            rec.update({"diff_vs_pillar0": rec["auroc"] - auroc(filled[REFERENCE], y),
                        "diff_lo": float(np.percentile(d, 2.5)), "diff_hi": float(np.percentile(d, 97.5))})
        rows.append(rec)
    return pd.DataFrame(rows)


def breakout(scores: dict[str, pd.Series], cases: pd.DataFrame) -> pd.DataFrame:
    """AUROC per row within each prior-link group (point estimates; groups with one answer class are blank)."""
    y = cases["label__radiology_progression"].astype(int).to_numpy(dtype=float)
    out = []
    for col in ("prior_source", "prior_ambiguous"):
        if col not in cases.columns:
            continue
        for g, idx in cases.groupby(cases[col].astype(str)).indices.items():
            yy = y[idx]
            rec = {"by": col, "group": g, "n": len(idx), "n_yes": int(yy.sum())}
            for k, s in scores.items():
                ss = s.fillna(0.5).to_numpy(dtype=float)[idx]
                rec[k] = auroc(ss, yy) if 0 < yy.sum() < len(yy) else float("nan")
            out.append(rec)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="*", default=[], help="NAME=answers.csv, one per row")
    ap.add_argument("--out", required=True, help="CSV path; the breakout goes beside it as <stem>_by_prior.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cases = samples.load(3, which="test1000")
    cases["case_id"] = cases["case_id"].astype(str)
    scores = {name: probe_scores(name, cases) for name in samples.PROBE_FILES}
    for spec in args.answers:
        name, path = spec.split("=", 1)
        scores[name] = answer_scores(path, cases)
    t = table(scores, cases, args.n_boot)
    b = breakout(scores, cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    b.to_csv(out.with_name(out.stem + "_by_prior.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t.round(3).to_string(index=False))
        print(b.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
