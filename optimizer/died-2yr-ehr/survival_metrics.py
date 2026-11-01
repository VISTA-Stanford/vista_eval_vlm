"""Survival scoring for died-2yr-ehr's survival mode, in plain Python (no numpy, no scikit-survival, no engine), so
the engine runs it with no new package and scripts/score_baselines.py scores the benchmark's survival curves with the
very same code. It re-implements vista-eval's survival scoring (src/vistaeval/metrics/survival.py: the stable window
and monthly grid) on scikit-survival's formulas (concordance_index_censored, brier_score, integrated_brier_score,
cumulative_dynamic_auc, CensoringDistributionEstimator); scripts/check_task.py checks each against scikit-survival.

A row is {"id", "time_days", "event"} (v1.7 os_survival); a curve is a list of survival values on GRID_MONTHS.
"""

from __future__ import annotations

import bisect
import math

#: vista-eval's month (src/vistaeval/eval/survival.py DAYS_PER_MONTH).
DAYS_PER_MONTH = 30.44
#: The monthly grid the scorer asks for (months after the record's last day); the scorer keeps the months inside the
#: stable range (scoring_months), about months 1-74 on this data. 120 = the most a curve is asked to cover.
GRID_MONTHS = tuple(range(1, 121))
#: Brier and time-dependent AUC reported at these months (not scored; the optimizer sees them).
REPORT_MONTHS = (12, 24, 36, 60)
#: A curve "near the average" stays within this of the TRAIN Kaplan-Meier curve at every grid month: the spec's
#: stop sign when most seed curves are one shape.
NEAR_AVERAGE = 0.05
#: sksurv concordance_index_censored's tie tolerance.
TIED_TOL = 1e-8


def kaplan_meier(times, events) -> tuple[list[float], list[float]]:
    """(unique times, survival just after each), as sksurv kaplan_meier_estimator: every distinct observed time,
    S *= 1 - deaths / at risk (time >= t)."""
    pairs = sorted(zip(times, events))
    uniq, probs, s, n, i = [], [], 1.0, len(pairs), 0
    while i < n:
        t, d, j = pairs[i][0], 0, i
        while j < n and pairs[j][0] == t:
            d += 1 if pairs[j][1] else 0
            j += 1
        s *= 1.0 - d / (n - i)
        uniq.append(t)
        probs.append(s)
        i = j
    return uniq, probs


def step_value(xs: list[float], ys: list[float], t: float, before: float = 1.0) -> float:
    """A right-continuous step function at t: the y of the last x <= t, ``before`` ahead of the first x."""
    k = bisect.bisect_right(xs, t) - 1
    return before if k < 0 else ys[k]


class Censoring:
    """sksurv CensoringDistributionEstimator: the Kaplan-Meier curve of being still under follow-up (censoring as the
    event), fitted on TRAIN. ``g(t)`` is P(not yet censored at t); 0 is read as infinity by the callers, as sksurv.
    As sksurv's reverse Kaplan-Meier, deaths at a tied time come first and leave the at-risk count before that
    time's censorings are counted; with no censoring at all, G is 1."""

    def __init__(self, times, events) -> None:
        pairs = sorted(zip(times, events))
        self.xs, self.ys, g, n, i = [], [], 1.0, len(pairs), 0
        if all(e for _t, e in pairs):
            self.xs, self.ys = [], []
            return
        while i < n:
            t, deaths, censored, j = pairs[i][0], 0, 0, i
            while j < n and pairs[j][0] == t:
                deaths += 1 if pairs[j][1] else 0
                censored += 0 if pairs[j][1] else 1
                j += 1
            at_risk = n - i - deaths
            if censored:
                g *= 1.0 - censored / at_risk
            self.xs.append(t)
            self.ys.append(g)
            i = j

    def g(self, t: float) -> float:
        return step_value(self.xs, self.ys, t)


def percentile(values: list[float], q: float) -> float:
    """numpy.percentile's default (linear) interpolation."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo = math.floor(pos)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)


def window_upper_days(train_rows: list[dict], eval_rows: list[dict] | None = None) -> float:
    """vista-eval's upper bound: 0.999 x the 90th percentile of TRAIN death times, and below 0.999 x the earlier of
    the last TRAIN and last evaluated death (src/vistaeval/metrics/survival.py)."""
    tev = [r["time_days"] for r in train_rows if r["event"]]
    eev = [r["time_days"] for r in (eval_rows or []) if r["event"]]
    max_event = min(max(tev), max(eev) if eev else max(tev))
    return min(percentile(tev, 90) * 0.999, max_event * 0.999)


def scoring_months(train_rows: list[dict], eval_rows: list[dict]) -> list[int]:
    """The grid months the score integrates over: those inside vista-eval's stable range for these patients."""
    upper = window_upper_days(train_rows, eval_rows)
    lower = max(min(r["time_days"] for r in train_rows), min(r["time_days"] for r in eval_rows)) + 1e-6
    return [m for m in GRID_MONTHS if lower <= m * DAYS_PER_MONTH <= upper]


def curve_at(curve, months) -> list[float]:
    """A curve (a list of [month, survival] points, read as a step curve, 1.0 before its first point) on a grid."""
    xs = [float(p[0]) for p in curve]
    ys = [float(p[1]) for p in curve]
    return [step_value(xs, ys, m) for m in months]


def c_index(times, events, risks) -> float:
    """Harrell's C, as sksurv concordance_index_censored: a pair is comparable when the earlier time is a death (a
    death and a censoring at the same time compare too); higher risk should die first; risks within TIED_TOL count
    half."""
    order = sorted(range(len(times)), key=lambda k: times[k])
    num = den = 0.0
    n, i = len(order), 0
    while i < n:
        end = i
        while end < n and times[order[end]] == times[order[i]]:
            end += 1
        for a in range(i, end):
            if not events[order[a]]:
                continue
            r = risks[order[a]]
            others = [order[b] for b in range(end, n)] + [order[b] for b in range(i, end) if not events[order[b]]]
            for b in others:
                den += 1
                diff = risks[b] - r
                num += 0.5 if abs(diff) <= TIED_TOL else (1.0 if diff < 0 else 0.0)
        i = end
    return num / den if den else float("nan")


def brier_at(times, events, surv: list[float], t: float, cens: Censoring) -> float:
    """sksurv brier_score at one time: deaths by t weigh (S)^2 / G(their time), patients still followed past t weigh
    (1 - S)^2 / G(t), those censored before t drop out; averaged over every patient."""
    gt = cens.g(t) or math.inf
    total = 0.0
    for T, e, sv in zip(times, events, surv):
        if T <= t and e:
            total += sv * sv / (cens.g(T) or math.inf)
        elif T > t:
            total += (1 - sv) ** 2 / gt
    return total / len(times)


def brier_contributions(times, events, surv: list[float], t: float, cens: Censoring) -> list[float]:
    """Each patient's term of brier_at (for the per-patient mistake flag)."""
    gt = cens.g(t) or math.inf
    return [(sv * sv / (cens.g(T) or math.inf)) if (T <= t and e) else ((1 - sv) ** 2 / gt) if T > t else 0.0
            for T, e, sv in zip(times, events, surv)]


def dynamic_auc_at(times, events, risk: list[float], t: float, cens: Censoring) -> float:
    """sksurv cumulative_dynamic_auc at one time: deaths by t (weighted 1 / G(their time)) against everyone still
    followed past t, higher risk should be the death; ties half."""
    cases = [(risk[k], 1 / (cens.g(times[k]) or math.inf)) for k in range(len(times)) if times[k] <= t and events[k]]
    controls = sorted(risk[k] for k in range(len(times)) if times[k] > t)
    if not cases or not controls:
        return float("nan")
    num = sum(w * (bisect.bisect_left(controls, r) + 0.5 * (bisect.bisect_right(controls, r) - bisect.bisect_left(controls, r)))
              for r, w in cases)
    return num / (sum(w for _r, w in cases) * len(controls))


def trapezoid(ys: list[float], xs: list[float]) -> float:
    return sum((xs[k + 1] - xs[k]) * (ys[k] + ys[k + 1]) / 2 for k in range(len(xs) - 1))


def survival_metrics(rows: list[dict], curves: dict, train_rows: list[dict]) -> dict:
    """vista-eval's survival trio for one set of patients: C-index over all follow-up (risk = minus the mean of the
    curve over the scoring months, i.e. minus restricted mean survival: for Cox-type curves the same order as the Cox
    risk), Brier on every scoring month and integrated Brier over them (trapezoid / span, as sksurv), and Brier and
    time-dependent AUC (risk 1 - S(t)) at REPORT_MONTHS. ``curves`` maps id -> survival at each of GRID_MONTHS."""
    months = scoring_months(train_rows, rows)
    cens = Censoring([r["time_days"] for r in train_rows], [bool(r["event"]) for r in train_rows])
    times = [r["time_days"] for r in rows]
    events = [bool(r["event"]) for r in rows]
    at = {m: GRID_MONTHS.index(m) for m in GRID_MONTHS}
    out = {"n": len(rows), "deaths": sum(events), "grid_first_month": months[0] if months else None,
           "grid_last_month": months[-1] if months else None}
    if not months:
        return out
    risk = [-sum(curves[r["id"]][at[m]] for m in months) / len(months) for r in rows]
    out["c_index"] = c_index(times, events, risk)
    days = [m * DAYS_PER_MONTH for m in months]
    briers = [brier_at(times, events, [curves[r["id"]][at[m]] for r in rows], d, cens) for m, d in zip(months, days)]
    out["integrated_brier"] = trapezoid(briers, days) / (days[-1] - days[0]) if len(days) > 1 else float("nan")
    for m in REPORT_MONTHS:
        if m in months:
            out[f"brier_{m}m"] = briers[months.index(m)]
            out[f"auc_{m}m"] = dynamic_auc_at(times, events, [1 - curves[r["id"]][at[m]] for r in rows],
                                              m * DAYS_PER_MONTH, cens)
    return out


def per_patient_brier(rows: list[dict], curves: dict, train_rows: list[dict], months: list[int]) -> dict[str, float]:
    """Each patient's mean Brier term over the scoring months (censoring-weighted, as the score)."""
    cens = Censoring([r["time_days"] for r in train_rows], [bool(r["event"]) for r in train_rows])
    times = [r["time_days"] for r in rows]
    events = [bool(r["event"]) for r in rows]
    tot = [0.0] * len(rows)
    for m in months:
        terms = brier_contributions(times, events, [curves[r["id"]][GRID_MONTHS.index(m)] for r in rows],
                                    m * DAYS_PER_MONTH, cens)
        tot = [a + b for a, b in zip(tot, terms)]
    return {r["id"]: t / max(len(months), 1) for r, t in zip(rows, tot)}
