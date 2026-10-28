"""Scoring for the eval set: rule-based answer parsing, macro balanced accuracy, patient bootstrap, and
probes cut at a validation-chosen point. Extends ``results.final_metrics`` (its forced-answer metrics
are reused, so an unreadable answer counts as wrong the same way everywhere).

Rules (spec, Decided): balanced accuracy per question; a step's headline is the mean over its
questions (macro); unparseable or refused = wrong, with the refusal rate reported; intervals from a
bootstrap over patients (all of a patient's rows move together, across questions too); chance is
1 / number of answers (0.5 for yes/no) for any answer-blind rule; probes are cut at the threshold that
maximizes balanced accuracy on VALIDATION, then frozen and applied unchanged to the scored rows.

    python -m evalset.score --results <answers.csv> [<answers.csv> ...] --out <table.csv> [--split test]

The probe side (``validation_cut``, ``probe_predictions``) is a library for now; the model-vs-probe table
is built when the pilot is scored (work package 4).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from results.final_metrics import forced_answer_metrics

_FINAL = re.compile(r"final\s*answer\s*[:：]?\s*[\*\"'`_]*\s*(yes|no)\b", re.IGNORECASE)
_REFUSAL_TEXT = re.compile(
    r"\b(i\s*(?:am|'m)\s*(?:not able|unable)|i\s*can(?:not|'t)\s*(?:provide|determine|answer|interpret|make)"
    r"|not (?:able|qualified) to (?:provide|interpret|diagnose)|consult (?:a|your) (?:radiologist|physician|doctor)"
    r"|i\s*(?:am|'m)\s*sorry)", re.IGNORECASE)
_REFUSAL_ERROR = re.compile(r"content.?filter|responsibleaipolicy|safety|blocked|prohibited|refus", re.IGNORECASE)


def parse_answer(text, form: str):
    """'yes', 'no' or None. ``form``: 'final_answer' (the last "Final Answer: yes/no" in free text) or
    'schema' (a forced answer: the bare word, or JSON {"answer": ...})."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).strip()
    if form == "final_answer":
        hits = _FINAL.findall(t)
        return hits[-1].lower() if hits else None
    if form == "schema":
        if t.lower() in ("yes", "no"):
            return t.lower()
        try:
            v = json.loads(t)
        except (json.JSONDecodeError, TypeError):
            return None
        a = str(v.get("answer", "")).strip().lower() if isinstance(v, dict) else ""
        return a if a in ("yes", "no") else None
    raise ValueError(form)


def is_refusal(text, error) -> bool:
    """A declined answer (not just an unreadable one): a filter/safety error, or refusal wording."""
    if isinstance(error, str) and error and _REFUSAL_ERROR.search(error):
        return True
    return isinstance(text, str) and bool(_REFUSAL_TEXT.search(text))


def to_label(answer, yes_label: int = 1):
    if answer is None:
        return None
    return yes_label if answer == "yes" else 1 - yes_label


def balanced_accuracy(y_true, y_pred) -> float:
    return forced_answer_metrics(list(y_true), list(y_pred), [0, 1]).get("balanced_accuracy", float("nan"))


def macro_balanced_accuracy(rows: pd.DataFrame) -> float:
    """Mean over questions of each question's balanced accuracy (rows: question_id, label, pred)."""
    vals = [balanced_accuracy(g["label"].astype(int), g["pred"]) for _, g in rows.groupby("question_id")]
    return float(np.nanmean(vals)) if vals else float("nan")


def _weighted_macro_ba(qcode, label, correct, w, nq):
    """Mean over questions of balanced accuracy, each row weighted by ``w`` (classes absent -> skipped)."""
    idx = qcode * 2 + label
    den = np.bincount(idx, weights=w, minlength=2 * nq).reshape(nq, 2)
    num = np.bincount(idx, weights=w * correct, minlength=2 * nq).reshape(nq, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(den > 0, num / den, np.nan)
        ba = np.nanmean(rec, axis=1)
    return float(np.nanmean(ba)) if np.isfinite(ba).any() else float("nan")


def patient_bootstrap(rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile interval of the macro balanced accuracy over ``rows``' questions (one question: its
    balanced accuracy), resampling patients (``person_id``) with replacement: all of a patient's rows
    move together. Each draw reweights rows by how often their patient was drawn."""
    pcode, _ = pd.factorize(rows["person_id"])
    qcode, q_uni = pd.factorize(rows["question_id"])
    label = rows["label"].astype(int).to_numpy()
    pred = rows["pred"].to_numpy(dtype=object)
    correct = np.array([p is not None and not (isinstance(p, float) and np.isnan(p)) and int(p) == t
                        for p, t in zip(pred, label)], dtype=float)
    n_p, nq = int(pcode.max()) + 1 if len(pcode) else 0, len(q_uni)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float)
        draws.append(_weighted_macro_ba(qcode, label, correct, counts[pcode], nq))
    return float(np.nanpercentile(draws, 100 * alpha / 2)), float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))


# -- probes ---------------------------------------------------------------------------------------

def validation_cut(scores, labels) -> float:
    """The threshold (predict yes when score >= t) with the best balanced accuracy on validation rows.

    Ties go to the threshold nearest 0.5. Candidates are the midpoints between sorted unique scores,
    plus one below the lowest and one above the highest."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]]) if len(u) else np.array([0.5])
    best, best_t = -1.0, 0.5
    for t in cands:
        ba = balanced_accuracy(y, (s >= t).astype(int))
        if ba > best + 1e-12 or (abs(ba - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = ba, float(t)
    return best_t


def probe_predictions(per_example: pd.DataFrame, task: str, key: str = "accession_number"):
    """(threshold, frame of key, split, label, pred) for one task from a vista-eval per-example file.

    The cut is chosen on the file's validation rows ('val' or 'valid') and applied to every row."""
    pe = per_example[per_example["task"] == task].copy()
    pe = pe[pe["label"].astype(int).isin([0, 1])]
    pe["split"] = pe["split"].replace({"val": "valid"})
    val = pe[pe["split"] == "valid"]
    if val.empty:
        raise ValueError(f"{task}: probe file has no validation rows to choose a cut on")
    t = validation_cut(val["score"], val["label"])
    pe["pred"] = (pe["score"].astype(float) >= t).astype(int)
    return t, pe[[key, "split", "label", "pred"]].rename(columns={key: "key"})


# -- tables ---------------------------------------------------------------------------------------

def score_rows(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per (step, recipe, arm, question) plus a macro row per (step, recipe, arm) when the step
    has 2+ questions: each step's headline averages its own questions only.

    ``df`` columns: step, recipe, arm, question_id, person_id, label (0/1), pred (0/1/None), refused (bool)."""
    out = []
    if "step" not in df.columns:
        df = df.assign(step=0)
    for (step, recipe, arm), g in df.groupby(["step", "recipe", "arm"], sort=True):
        for qid, q in g.groupby("question_id", sort=True):
            m = forced_answer_metrics(q["label"].astype(int).tolist(), list(q["pred"]), [0, 1])
            lo, hi = patient_bootstrap(q, n_boot, seed)
            out.append({"step": step, "recipe": recipe, "arm": arm, "question_id": qid, "n": m["n"],
                        "n_yes": int((q["label"] == 1).sum()), "n_no": int((q["label"] == 0).sum()),
                        "balanced_accuracy": m.get("balanced_accuracy"), "ba_lo": lo, "ba_hi": hi,
                        "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
                        "unparseable_rate": float(q["pred"].isna().mean()),
                        "refusal_rate": float(q["refused"].astype(bool).mean()), "chance": 0.5})
        if g["question_id"].nunique() > 1:
            lo, hi = patient_bootstrap(g, n_boot, seed)
            out.append({"step": step, "recipe": recipe, "arm": arm, "question_id": "MACRO", "n": len(g),
                        "balanced_accuracy": macro_balanced_accuracy(g), "ba_lo": lo, "ba_hi": hi,
                        "unparseable_rate": float(g["pred"].isna().mean()),
                        "refusal_rate": float(g["refused"].astype(bool).mean()), "chance": 0.5})
    return pd.DataFrame(out)


def prepare(answers: pd.DataFrame) -> pd.DataFrame:
    """Answers file rows -> scoring rows (parsed answer, label, refusal), dropping rows with no scored label."""
    a = answers[answers["label"].notna()].copy()
    a["label"] = a["label"].astype(int)
    a["pred"] = [to_label(x) if isinstance(x, str) else None for x in a["answer"]]
    a["pred"] = a["pred"].astype(object).where(a["pred"].notna(), None)
    a["refused"] = [bool(r) for r in a["refused"]]
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split", default=None, help="score only this split (step 4: test reported separately)")
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(p, dtype={"case_id": str, "person_id": str}) for p in args.results], ignore_index=True)
    if args.split:
        df = df[df["split"] == args.split]
    table = score_rows(prepare(df), n_boot=args.n_boot)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
