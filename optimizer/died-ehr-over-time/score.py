"""The death-agent-over-time scorer, in plain Python (no numpy), checked equal to scikit-learn in
scripts/check_task.py.

Per decision point x horizon x model: AUROC (died-2yr-ehr's Mann-Whitney ``adapters.auroc``, ties half) with a
patient-bootstrap 95% interval, Brier, observed/expected deaths, and the calibration slope (logistic regression of the
outcome on the logit of the stated risk). Models: the two arms on every labelled row (``fresh``, ``memory``), and,
on the rows every listed probe scores (``common`` rows), the two arms beside the K16 probes. Memory minus fresh:
AUROC and Brier differences per point with a paired patient-bootstrap interval. Under-updating: the median absolute
change in each arm's stated risk from a patient's previous decision point, with an interval.

A failed or unreadable answer is scored at 0.5, as died-2yr-ehr scores it (``adapters.FAILED_PROBABILITY``), and is
counted. Each patient has at most one row per point, so a patient bootstrap of one point resamples its rows.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from statistics import median

import common as c
import adapters as ad  # died-2yr-ehr's

#: The spec's stop rule: the fresh arm at treatment start more than this far below the Haiku probability seed.
GATE_MARGIN = 0.05
#: Compare on the patients both tasks hold when there are at least this many; else against the seed's VAL-200 AUROC.
MIN_OVERLAP = 40
#: The Haiku probability seed's zero-shot mean AUROC on the died-2yr-ehr VAL-200 pool (prob-summary-haiku-r1,
#: readback planning/vista_eval_vlm/session/2026-11-10-prob-summary-haiku-sllm-02-readback.md: 0.789 / 0.843).
#: The environment override is for the synthetic checks only.
SEED_VAL200_AUROC = float(os.environ.get("OVER_TIME_SEED_VAL200_AUROC", "0.816"))
N_BOOT = 2000
SEED = 20261111
EPS = 1e-6


def brier(pairs: list[tuple[int, float]]) -> float:
    return sum((p - g) ** 2 for g, p in pairs) / len(pairs)


def observed_expected(pairs: list[tuple[int, float]]) -> float | None:
    expected = sum(p for _g, p in pairs)
    return sum(g for g, _p in pairs) / expected if expected > 0 else None


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def calibration_slope(pairs: list[tuple[int, float]], iters: int = 100) -> float | None:
    """Slope b of the unpenalized logistic regression y ~ a + b * logit(p), by Newton-Raphson. None when it does not
    exist (one class, a single stated value, or separation)."""
    xs = [logit(p) for _g, p in pairs]
    ys = [g for g, _p in pairs]
    if len(set(ys)) < 2 or len(set(xs)) < 2:
        return None
    a = b = 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            z = a + b * x
            m = 1 / (1 + math.exp(-z)) if z > -700 else 0.0
            w = m * (1 - m)
            ga += y - m
            gb += (y - m) * x
            haa += w
            hab += w * x
            hbb += w * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            return None
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            return b
        if abs(b) > 1e6:
            return None
    return b


def metrics(pairs: list[tuple[int, float]]) -> dict:
    return {"n": len(pairs), "deaths": sum(g for g, _p in pairs), "auroc": ad.auroc(pairs), "brier": brier(pairs),
            "obs_exp": observed_expected(pairs), "cal_slope": calibration_slope(pairs)}


def interval(values: list[float]) -> tuple[float | None, float | None]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None, None
    return v[int(0.025 * (len(v) - 1))], v[int(math.ceil(0.975 * (len(v) - 1)))]


def boot(units: list, stat, n_boot: int, rng: random.Random) -> tuple:
    """Percentile 95% interval of stat(resampled units): units are one per patient."""
    vals = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(len(units))] for _ in units]
        try:
            vals.append(stat(sample))
        except (ZeroDivisionError, ValueError):
            continue
    return interval(vals)


def risk(answer: dict | None, key: str) -> float:
    return ad.prob(answer or {"error": "missing"}, key)


def score(answers: dict[str, dict[str, dict]], rows: dict[str, dict], labels: dict[str, dict],
          probes: dict[tuple[str, str, str], float], *, n_boot: int = N_BOOT, seed: int = SEED,
          probe_names: tuple[str, ...] = tuple(c.K16_PROBES)) -> dict:
    """answers: "<point>-<arm>" -> row -> answers.json entry. rows/labels: program.rows()/labels(). probes:
    (row, probe, label field) -> score. Returns {"table": [...], "memory_minus_fresh": [...], "risk_change": [...]}."""
    rng = random.Random(seed)
    table, diffs, change = [], [], []
    for point in c.POINT_KEYS:
        pr = sorted(r for r, x in rows.items() if x["point"] == point)
        for key, field, words in c.HORIZONS:
            lab = [r for r in pr if labels[r][field] is not None]
            if not lab:
                continue
            arms = {arm: answers.get(f"{point}-{arm}", {}) for arm in ("fresh", "memory")}
            for arm, ans in arms.items():
                pairs = {r: (labels[r][field], risk(ans.get(r), key)) for r in lab}
                m = metrics(list(pairs.values()))
                m["auroc_lo"], m["auroc_hi"] = boot(lab, lambda s: ad.auroc([pairs[r] for r in s]), n_boot, rng)
                m["failed"] = sum(1 for r in lab if (ans.get(r) or {"error": 1}).get("error"))
                table.append({"point": point, "horizon": words, "model": arm, "rows": "all labelled", **m})
            # the probes and both arms on the rows every probe that has VAL scores covers
            have = [p for p in probe_names if any((r, p, field) in probes for r in lab)]
            common = [r for r in lab if all((r, p, field) in probes for p in have)]
            for model in ("fresh", "memory", *probe_names):
                if model in probe_names and model not in have:
                    table.append({"point": point, "horizon": words, "model": model, "rows": "common", "n": 0,
                                  "note": "no VAL score in K16"})
                    continue
                if not common:
                    continue
                get = ((lambda r, m=model: probes[(r, m, field)]) if model in probe_names
                       else (lambda r, m=model: risk(arms[m].get(r), key)))
                pairs = {r: (labels[r][field], get(r)) for r in common}
                m = metrics(list(pairs.values()))
                m["auroc_lo"], m["auroc_hi"] = boot(common, lambda s: ad.auroc([pairs[r] for r in s]), n_boot, rng)
                table.append({"point": point, "horizon": words, "model": model, "rows": "common", **m})
            # memory minus fresh, paired on the same rows. At treatment start the memory arm takes the fresh arm's
            # answers (no earlier answer exists), so the difference there is 0 by construction.
            if point == c.POINT_KEYS[0]:
                diffs.append({"point": point, "horizon": words, "n": len(lab), "auroc_diff": 0.0, "auroc_diff_lo": 0.0,
                              "auroc_diff_hi": 0.0, "brier_diff": 0.0, "brier_diff_lo": 0.0, "brier_diff_hi": 0.0,
                              "note": "0 by construction: the memory arm shares the fresh arm's answers here"})
            elif all(arms.values()):
                pf = {r: (labels[r][field], risk(arms["fresh"].get(r), key)) for r in lab}
                pm = {r: (labels[r][field], risk(arms["memory"].get(r), key)) for r in lab}

                def d_auc(s):
                    return ad.auroc([pm[r] for r in s]) - ad.auroc([pf[r] for r in s])

                def d_brier(s):
                    return brier([pm[r] for r in s]) - brier([pf[r] for r in s])

                lo, hi = boot(lab, d_auc, n_boot, rng)
                blo, bhi = boot(lab, d_brier, n_boot, rng)
                diffs.append({"point": point, "horizon": words, "n": len(lab), "auroc_diff": d_auc(lab),
                              "auroc_diff_lo": lo, "auroc_diff_hi": hi, "brier_diff": d_brier(lab),
                              "brier_diff_lo": blo, "brier_diff_hi": bhi})
        # under-updating: |risk change| from each patient's previous decision point, per arm and horizon
        if point == c.POINT_KEYS[0]:
            continue
        for arm in ("fresh", "memory"):
            for key, _field, words in c.HORIZONS:
                deltas = []
                for r in pr:
                    prev = previous_row(rows, r)
                    if prev is None:
                        continue
                    a_now = answers.get(f"{point}-{arm}", {}).get(r)
                    a_prev = answers.get(f"{rows[prev]['point']}-{arm}", {}).get(prev)
                    if not a_now or not a_prev or a_now.get("error") or a_prev.get("error"):
                        continue
                    deltas.append(100 * abs(a_now[key] - a_prev[key]))
                if deltas:
                    lo, hi = boot(deltas, median, n_boot, rng)
                    change.append({"point": point, "arm": arm, "horizon": words, "pairs": len(deltas),
                                   "median_abs_change": median(deltas), "lo": lo, "hi": hi})
    return {"table": table, "memory_minus_fresh": diffs, "risk_change": change}


def mean_auroc(answers: dict, gold: dict, ids, fields) -> float:
    """AUROC averaged over the two horizons for ``ids`` (rows without a label at a horizon left out there):
    answers id -> answers.json entry, gold id -> {field: 0/1/None}; fields = the gold field per horizon key."""
    out = []
    for (key, _f, _w), field in zip(c.HORIZONS, fields):
        pairs = [(gold[i][field], risk(answers.get(i), key)) for i in ids if gold[i].get(field) is not None]
        out.append(ad.auroc(pairs))
    return sum(out) / len(out)


def start_gate(fresh_start: dict, rows: dict, labels: dict, seed_answers: dict | None, seed_gold: dict | None,
               seed_ids: dict) -> dict:
    """The spec's stop rule, after the treatment-start fresh arm: stop and ask if its mean AUROC (1 and 2 years) is
    more than GATE_MARGIN below the Haiku probability seed. On the patients both tasks hold (seed_ids: patient ->
    died-2yr-ehr id, with a seed answer and label) when there are at least MIN_OVERLAP, each scored on its own labels;
    else this arm on all its treatment-start rows against SEED_VAL200_AUROC. Says which it used."""
    start = sorted(r for r, x in rows.items() if x["point"] == c.POINT_KEYS[0])
    pairs = [(r, seed_ids[rows[r]["patient"]]) for r in start if rows[r]["patient"] in seed_ids]
    if seed_answers and seed_gold:
        pairs = [(r, t) for r, t in pairs if t in seed_answers and t in seed_gold]
    else:
        pairs = []
    if len(pairs) >= MIN_OVERLAP:
        ours = mean_auroc(fresh_start, labels, [r for r, _t in pairs], ("label_1yr", "label_2yr"))
        ref = mean_auroc(seed_answers, seed_gold, [t for _r, t in pairs], ("label_1yr", "label"))
        used = f"the {len(pairs)} patients both tasks hold, each on its own labels"
    else:
        ours = mean_auroc(fresh_start, labels, start, ("label_1yr", "label_2yr"))
        ref = SEED_VAL200_AUROC
        used = (f"the seed's VAL-200 AUROC (only {len(pairs)} overlapping patients with seed answers, under "
                f"{MIN_OVERLAP})")
    return {"fresh_start_auroc": ours, "seed_auroc": ref, "compared_on": used, "overlap": len(pairs),
            "margin": GATE_MARGIN, "pass": ours >= ref - GATE_MARGIN}


def load_seed(answers_path: Path | None, labels_path: Path) -> tuple[dict | None, dict | None, str]:
    """The Haiku seed's per-patient VAL answers (an engine answers.json) and the died-2yr-ehr VAL labels, re-scored
    against the readback's VAL-200 number as a check that the right pass was found. (None, None, why) otherwise."""
    if answers_path is None or not Path(answers_path).is_file():
        return None, None, "no seed answers file found"
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))["answers"]
    gold = {r["id"]: r for r in (json.loads(x) for x in Path(labels_path).read_text().splitlines() if x.strip())}
    ids = [i for i in answers if i in gold]
    got = mean_auroc(answers, gold, ids, ("label_1yr", "label")) if ids else None
    if got is None or abs(got - SEED_VAL200_AUROC) > 0.005:
        return None, None, (f"the seed answers re-score to {fmt(got)} on their {len(ids)} patients, not the readback's "
                            f"{SEED_VAL200_AUROC:.3f}: not the seed's zero-shot pass, so not used")
    return answers, gold, f"seed answers re-scored {got:.3f} on {len(ids)} patients (readback {SEED_VAL200_AUROC:.3f})"


def previous_row(rows: dict[str, dict], row: str) -> str | None:
    """The same patient's row at the latest earlier decision point they had, or None."""
    x = rows[row]
    for point in reversed(c.POINT_KEYS[: c.POINT_KEYS.index(x["point"])]):
        prev = f"{x['patient']}-{point}"
        if prev in rows:
            return prev
    return None


def fmt(v, nd=3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def print_gate(g: dict) -> None:
    print(f"treatment-start check: fresh arm mean AUROC {g['fresh_start_auroc']:.3f} vs Haiku seed "
          f"{g['seed_auroc']:.3f}, compared on {g['compared_on']}; stop if more than {g['margin']:.2f} below -> "
          f"{'PASS' if g['pass'] else 'STOP'}")


def print_tables(out: dict) -> None:
    print(f"{'point':<10}{'horizon':<9}{'model':<15}{'rows':<14}{'n':>5}{'deaths':>7}  {'AUROC [95%]':<22}"
          f"{'Brier':>7}{'O/E':>7}{'slope':>7}{'failed':>7}")
    for t in out["table"]:
        if t.get("note"):
            print(f"{t['point']:<10}{t['horizon']:<9}{t['model']:<15}{t['rows']:<14}  {t['note']}")
            continue
        ci = f"{fmt(t['auroc'])} [{fmt(t['auroc_lo'])}, {fmt(t['auroc_hi'])}]"
        print(f"{t['point']:<10}{t['horizon']:<9}{t['model']:<15}{t['rows']:<14}{t['n']:>5}{t['deaths']:>7}  {ci:<22}"
              f"{fmt(t['brier']):>7}{fmt(t['obs_exp'], 2):>7}{fmt(t['cal_slope'], 2):>7}{t.get('failed', ''):>7}")
    print("\nmemory minus fresh (all labelled rows, paired patient bootstrap)")
    for d in out["memory_minus_fresh"]:
        print(f"{d['point']:<10}{d['horizon']:<9} n={d['n']:<5} AUROC {d['auroc_diff']:+.3f} "
              f"[{fmt(d['auroc_diff_lo'])}, {fmt(d['auroc_diff_hi'])}]  Brier {d['brier_diff']:+.4f} "
              f"[{fmt(d['brier_diff_lo'], 4)}, {fmt(d['brier_diff_hi'], 4)}]"
              + (f"  ({d['note']})" if d.get("note") else ""))
    print("\nmedian |risk change| from the patient's previous decision point (percentage points)")
    for d in out["risk_change"]:
        print(f"{d['point']:<10}{d['arm']:<8}{d['horizon']:<9} pairs={d['pairs']:<5} {d['median_abs_change']:.1f} "
              f"[{fmt(d['lo'], 1)}, {fmt(d['hi'], 1)}]")
