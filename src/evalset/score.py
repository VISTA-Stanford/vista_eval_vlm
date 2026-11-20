"""Scoring for the eval set: rule-based answer parsing, macro balanced accuracy, patient bootstrap, and
probes cut at a validation-chosen point. Extends ``results.final_metrics`` (its forced-answer metrics
are reused, so an unreadable answer counts as wrong the same way everywhere).

Probability runs (recipe ``answer_form: "probability"``, Phil 11-17): the parsed 0-100 answer, as 0-1, is the
answers file's ``p_yes`` (open models fill the same column from token probabilities). AUROC per question and the
mean over a step's questions, patient bootstrap; a row with no readable probability is left out of the AUROC and
counted (``prob_missing_rate``). Balanced accuracy is reported only for rows with a yes/no answer.

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
_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%?")
_REFUSAL_ERROR = re.compile(r"content.?filter|responsibleaipolicy|safety|blocked|prohibited|refus", re.IGNORECASE)


def parse_probability(t: str):
    """P(yes) in 0-1 from a probability reply, or None: JSON {"probability": n}, else the one number in the text
    ("42", "42%", "Probability: 42"). Two or more numbers, none, a number outside 0-100, or refusal wording -> None."""
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        v = None
    if isinstance(v, dict):
        n = v.get("probability")
        if isinstance(n, bool) or not isinstance(n, (int, float)):
            return None
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        n = v
    else:
        if _REFUSAL_TEXT.search(t):
            return None
        hits = _NUMBER.findall(t)
        if len(hits) != 1:
            return None
        n = float(hits[0])
    n = float(n)
    return n / 100.0 if np.isfinite(n) and 0 <= n <= 100 else None


#: Same rule as the adapters (``models.base.DISTRIBUTION_SUM_TOLERANCE``): option probabilities adding up to within
#: this much of 100 are read and rescaled to 100; anything else is unreadable.
OPTION_SUM_TOLERANCE = 10


def parse_options(text, options):
    """JSON {label code: share in 0-1, ...} (shares rescaled to add up to 1, codes in option order) from a
    several-option reply, or None. ``options``: the question's ((key, label code, text), ...). The reply must be JSON
    with every key, each a whole number 0-100, adding up to 100 within ``OPTION_SUM_TOLERANCE``, and no other keys."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    try:
        v = json.loads(str(text))
    except (json.JSONDecodeError, TypeError):
        return None
    keys = [k for k, _, _ in options]
    if not isinstance(v, dict) or set(v) != set(keys):
        return None
    vals = []
    for k in keys:
        n = v[k]
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not np.isfinite(n) or n != int(n) or not 0 <= n <= 100:
            return None
        vals.append(float(n))
    total = sum(vals)
    if total <= 0 or abs(total - 100) > OPTION_SUM_TOLERANCE:
        return None
    return json.dumps({str(code): x / total for (_, code, _), x in zip(options, vals)})


def parse_answer(text, form: str):
    """'yes', 'no' or None; for ``form`` 'probability', P(yes) in 0-1 or None. ``form``: 'final_answer' (the last
    "Final Answer: yes/no" in free text), 'schema' (a forced answer: the bare word, or JSON {"answer": ...}) or
    'probability' (``parse_probability``)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).strip()
    if form == "probability":
        return parse_probability(t)
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
    if rows["person_id"].isna().any():
        # A row without a patient cannot be resampled with its patient; better to stop than skew the interval.
        raise ValueError(f"{int(rows['person_id'].isna().sum())} rows have no person_id; the patient bootstrap needs one")
    if rows.empty:
        return float("nan"), float("nan")
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


def _weighted_auroc(inv, n_u, label, w) -> float:
    """AUROC with ties counted half, each row weighted by ``w``; ``inv`` indexes each row's score among the ``n_u``
    sorted unique scores. NaN when a class has no weight."""
    pos = np.bincount(inv, weights=w * label, minlength=n_u)
    neg = np.bincount(inv, weights=w * (1 - label), minlength=n_u)
    p_tot, n_tot = pos.sum(), neg.sum()
    if p_tot <= 0 or n_tot <= 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (p_tot * n_tot))


def auroc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    u, inv = np.unique(s, return_inverse=True)
    return _weighted_auroc(inv, len(u), y, np.ones(len(s)))


def auroc_bootstrap(rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """(point, lo, hi): mean over ``rows``' questions of each question's AUROC from ``score``, with a percentile
    interval from resampling patients (all of a patient's rows move together, across questions too). ``rows`` must
    already exclude rows without a score."""
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    if rows["person_id"].isna().any():
        raise ValueError(f"{int(rows['person_id'].isna().sum())} rows have no person_id; the patient bootstrap needs one")
    pcode, _ = pd.factorize(rows["person_id"])
    qcode, q_uni = pd.factorize(rows["question_id"])
    label = rows["label"].astype(int).to_numpy(dtype=float)
    score = rows["score"].astype(float).to_numpy()
    per_q = []
    for qi in range(len(q_uni)):
        m = qcode == qi
        u, inv = np.unique(score[m], return_inverse=True)
        per_q.append((m, inv, len(u)))

    def macro(w):
        vals = [_weighted_auroc(inv, n_u, label[m], w[m]) for m, inv, n_u in per_q]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    point = macro(np.ones(len(rows)))
    n_p = int(pcode.max()) + 1
    rng = np.random.default_rng(seed)
    draws = [macro(np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float)[pcode]) for _ in range(n_boot)]
    return point, float(np.nanpercentile(draws, 100 * alpha / 2)), float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))


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

TABLE_COLS = ["step", "recipe", "arm", "question_id", "n", "n_yes", "n_no", "balanced_accuracy", "ba_lo", "ba_hi",
              "sensitivity", "specificity", "unparseable_rate", "refusal_rate", "chance",
              "auroc", "auroc_lo", "auroc_hi", "n_prob", "prob_missing_rate"]


def score_rows(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per (step, recipe, arm, question) plus a macro row per (step, recipe, arm) when the step
    has 2+ questions: each step's headline averages its own questions only.

    ``df`` columns: step, recipe, arm, question_id, person_id, label (0/1), pred (0/1/None), refused (bool), and
    optionally score (P(yes), NaN when missing). Balanced accuracy is filled for a (step, recipe, arm) with yes/no
    answers ("yesno" True); AUROC for one with any score."""
    out = []
    if "step" not in df.columns:
        df = df.assign(step=0)
    if "score" not in df.columns:
        df = df.assign(score=np.nan)
    if "yesno" not in df.columns:
        df = df.assign(yesno=True)

    def auc_cols(rows):
        s = rows[rows["score"].notna()]
        if not rows["score"].notna().any():
            return {}
        a, lo, hi = auroc_bootstrap(s, n_boot, seed)
        return {"auroc": a, "auroc_lo": lo, "auroc_hi": hi, "n_prob": len(s),
                "prob_missing_rate": float(rows["score"].isna().mean())}

    for (step, recipe, arm), g in df.groupby(["step", "recipe", "arm"], sort=True):
        yesno = bool(g["yesno"].any())
        for qid, q in g.groupby("question_id", sort=True):
            rec = {"step": step, "recipe": recipe, "arm": arm, "question_id": qid, "n": len(q),
                   "n_yes": int((q["label"] == 1).sum()), "n_no": int((q["label"] == 0).sum())}
            if yesno:
                m = forced_answer_metrics(q["label"].astype(int).tolist(), list(q["pred"]), [0, 1])
                lo, hi = patient_bootstrap(q, n_boot, seed)
                rec.update({"balanced_accuracy": m.get("balanced_accuracy"), "ba_lo": lo, "ba_hi": hi,
                            "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
                            "unparseable_rate": float(q["pred"].isna().mean())})
            rec.update({"refusal_rate": float(q["refused"].astype(bool).mean()), "chance": 0.5, **auc_cols(q)})
            out.append(rec)
        if g["question_id"].nunique() > 1:
            rec = {"step": step, "recipe": recipe, "arm": arm, "question_id": "MACRO", "n": len(g)}
            if yesno:
                lo, hi = patient_bootstrap(g, n_boot, seed)
                rec.update({"balanced_accuracy": macro_balanced_accuracy(g), "ba_lo": lo, "ba_hi": hi,
                            "unparseable_rate": float(g["pred"].isna().mean())})
            rec.update({"refusal_rate": float(g["refused"].astype(bool).mean()), "chance": 0.5, **auc_cols(g)})
            out.append(rec)
    table = pd.DataFrame(out)
    for c in TABLE_COLS:  # one column set whatever mix of runs is scored (blank where it does not apply)
        if c not in table.columns:
            table[c] = np.nan
    return table[TABLE_COLS + [c for c in table.columns if c not in TABLE_COLS]]


# -- several-option questions (board treatment and pneumonitis cards, Phil 11-20) ------------------------------------

OPTION_COLS = ["top2", "top2_lo", "top2_hi", "top2_chance", "n_per_answer", "option_auroc"]


def option_questions() -> dict:
    """{(step, question_id): Question} for every several-option question on the cards."""
    from evalset.cards import CARDS
    return {(c.step, q.id): q for c in CARDS.values() for q in c.questions if q.options}


def is_option_row(df: pd.DataFrame) -> pd.Series:
    oq = option_questions()
    return pd.Series([(int(st), str(qid)) in oq for st, qid in zip(df["step"], df["question_id"])], index=df.index)


def _option_metrics(P, readable, label, codes, w):
    """(macro one-vs-rest AUROC, {code: AUROC}, top-pick balanced accuracy, top-two rate) with row weights ``w``.

    AUROC: over readable rows, one per answer that has both yes and no weight, then their mean (vista-eval's multi-class
    rule: ``roc_auc_score(multi_class="ovr", average="macro")`` over the answers present). Top pick and top two: an
    unreadable row counts as wrong (the eval set's rule for forced answers); a top-pick tie goes to the first option
    listed. Balanced accuracy = mean over the answers present of the share of their rows picked correctly."""
    r = readable
    per = {}
    for j, c in enumerate(codes):
        y = (label[r] == c).astype(float)
        u, inv = np.unique(P[r, j], return_inverse=True)
        per[c] = _weighted_auroc(inv, len(u), y, w[r]) if r.any() else float("nan")
    vals = [v for v in per.values() if np.isfinite(v)]
    macro = float(np.mean(vals)) if vals else float("nan")
    Pf = np.where(r[:, None], P, -1.0)
    pick = np.array(codes)[np.argmax(Pf, axis=1)]
    order = np.argsort(-Pf, axis=1, kind="stable")[:, :2]
    top2 = np.array([label[i] in (codes[order[i, 0]], codes[order[i, 1]]) for i in range(len(label))]) & r
    correct = (pick == label) & r
    recalls = []
    for c in codes:
        m = label == c
        if w[m].sum() > 0:
            recalls.append((w[m] * correct[m]).sum() / w[m].sum())
    ba = float(np.mean(recalls)) if recalls else float("nan")
    t2 = float((w * top2).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return macro, per, ba, t2


def option_table(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """One row per (step, recipe, arm, several-option question), plus a MACRO row per (step, recipe, arm) with 2+
    questions (mean of the questions' values). Intervals from one patient bootstrap per (step, recipe, arm): all of a
    patient's rows move together, across questions too. ``df``: answers-file rows of several-option questions with a
    scored label (step, recipe, arm, question_id, person_id, label, answer, refused)."""
    oq = option_questions()
    out = []
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for (step, recipe, arm), g in df.groupby(["step", "recipe", "arm"], sort=True):
        if g["person_id"].isna().any():
            raise ValueError("rows without a person_id; the patient bootstrap needs one")
        pcode, _ = pd.factorize(g["person_id"])
        n_p = int(pcode.max()) + 1
        qs = []
        for qid, q in g.groupby("question_id", sort=True):
            qq = oq[(int(step), str(qid))]
            codes = [c for _, c, _ in qq.options]
            parsed = [json.loads(a) if isinstance(a, str) and a.startswith("{") else None for a in q["answer"]]
            readable = np.array([p is not None for p in parsed])
            P = np.array([[p[str(c)] for c in codes] if p is not None else [np.nan] * len(codes) for p in parsed])
            label = q["label"].astype(int).to_numpy()
            qs.append((qid, q, codes, P, readable, label, pcode[g.index.get_indexer(q.index)]))
        rng = np.random.default_rng(seed)
        draws = [rng.integers(0, n_p, n_p) for _ in range(n_boot)]
        per_q = []
        for qid, q, codes, P, readable, label, pc in qs:
            point = _option_metrics(P, readable, label, codes, np.ones(len(q)))
            boot = []
            for d in draws:
                w = np.bincount(d, minlength=n_p).astype(float)[pc]
                m, _, ba, t2 = _option_metrics(P, readable, label, codes, w)
                boot.append((m, ba, t2))
            boot = np.array(boot)
            per_q.append((point, boot))
            k = len(codes)
            out.append({"step": step, "recipe": recipe, "arm": arm, "question_id": qid, "n": len(q),
                        "balanced_accuracy": point[2], "ba_lo": np.nanpercentile(boot[:, 1], lo_q),
                        "ba_hi": np.nanpercentile(boot[:, 1], hi_q),
                        "unparseable_rate": float((~readable).mean()),
                        "refusal_rate": float(q["refused"].astype(bool).mean()), "chance": 1.0 / k,
                        "auroc": point[0], "auroc_lo": np.nanpercentile(boot[:, 0], lo_q),
                        "auroc_hi": np.nanpercentile(boot[:, 0], hi_q), "n_prob": int(readable.sum()),
                        "prob_missing_rate": float((~readable).mean()),
                        "top2": point[3], "top2_lo": np.nanpercentile(boot[:, 2], lo_q),
                        "top2_hi": np.nanpercentile(boot[:, 2], hi_q), "top2_chance": min(1.0, 2.0 / k),
                        "n_per_answer": json.dumps({str(c): int((label == c).sum()) for c in codes}),
                        "option_auroc": json.dumps({str(c): (None if not np.isfinite(v) else round(v, 4))
                                                    for c, v in point[1].items()})})
        if len(qs) > 1:
            pts = np.array([[p[0], p[2], p[3]] for p, _ in per_q])
            bs = np.nanmean(np.stack([b for _, b in per_q]), axis=0)
            out.append({"step": step, "recipe": recipe, "arm": arm, "question_id": "MACRO", "n": len(g),
                        "auroc": float(np.nanmean(pts[:, 0])), "auroc_lo": np.nanpercentile(bs[:, 0], lo_q),
                        "auroc_hi": np.nanpercentile(bs[:, 0], hi_q),
                        "balanced_accuracy": float(np.nanmean(pts[:, 1])), "ba_lo": np.nanpercentile(bs[:, 1], lo_q),
                        "ba_hi": np.nanpercentile(bs[:, 1], hi_q),
                        "top2": float(np.nanmean(pts[:, 2])), "top2_lo": np.nanpercentile(bs[:, 2], lo_q),
                        "top2_hi": np.nanpercentile(bs[:, 2], hi_q),
                        "refusal_rate": float(g["refused"].astype(bool).mean())})
    table = pd.DataFrame(out)
    for c in TABLE_COLS + OPTION_COLS:
        if c not in table.columns:
            table[c] = np.nan
    return table[TABLE_COLS + OPTION_COLS]


def prepare(answers: pd.DataFrame) -> pd.DataFrame:
    """Answers file rows -> scoring rows (parsed answer, label, refusal), dropping rows with no scored label."""
    a = answers[answers["label"].notna()].copy()
    a["label"] = a["label"].astype(int)
    # A probability run's answer column holds numbers (P(yes)); a yes/no run's holds 'yes' / 'no' / blank.
    # A run is scored as yes/no unless its answers are numbers and none is yes/no.
    is_yn = pd.Series([isinstance(x, str) and x in ("yes", "no") for x in a["answer"]], index=a.index)
    is_num = pd.to_numeric(a["answer"], errors="coerce").notna()
    keys = [c for c in ("step", "recipe", "arm") if c in a.columns]
    a["yesno"] = (is_yn.groupby([a[k] for k in keys]).transform("any")
                  | ~is_num.groupby([a[k] for k in keys]).transform("any")) if len(a) else True
    a["score"] = pd.to_numeric(a["p_yes"], errors="coerce") if "p_yes" in a.columns else np.nan
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
    opt = is_option_row(df) if len(df) else pd.Series([], dtype=bool)
    parts = []
    if (~opt).any():
        parts.append(score_rows(prepare(df[~opt]), n_boot=args.n_boot))
    if opt.any():
        o = df[opt & df["label"].notna()].copy()
        o["label"] = o["label"].astype(int)
        o["refused"] = [bool(r) for r in o["refused"]]
        parts.append(option_table(o.reset_index(drop=True), n_boot=args.n_boot))
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TABLE_COLS)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
