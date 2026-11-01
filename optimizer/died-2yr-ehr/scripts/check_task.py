"""$0 checks of the task code that need no real patient: synthetic patients, staging and the scorer.

    synth DIR    write 8 SYNTHETIC patients (made-up values, no real data) in the exporter's layout to DIR:
                 splits, pools, patients/<id>/{person,condition_occurrence,measurement}.parquet, notes.txt,
                 registry_diagnoses.parquet, meta.json, plus a registry_dates/ side folder beside patients/ as the
                 exporter writes it. Then run the engine's dry run on it with TASK_DATA_DIR=DIR.
    check        with TASK_DATA_DIR pointing at a synth DIR (or the real working copy, read-only use):
                   1. staging copies exactly a patient's record files, read-only, and nothing from registry_dates/
                   2. staging refuses a patient folder holding a stray file or subfolder (planted in a copy)
                   3. the scorer's balanced accuracy equals scikit-learn's at each horizon, failed answers wrong,
                      on random answer sets and on the VAL pool
                   4. the session's output.json is read as yes/no exactly: anything else is an error
                 With ANSWER=probability, 3 and 4 instead check AUROC and Brier against scikit-learn on random
                 tie-heavy answer sets (failed answers at 0.5) and the VAL pool, the right-side-of-death-rate flag,
                 and that output.json is read as a 0-100 number per horizon (anything else an error) with its
                 reasons kept, capped and cleaned.
                 With ANSWER=survival, 3 and 4 instead check C-index, Brier, integrated Brier and time-dependent AUC
                 against scikit-survival computed directly (random tie-heavy sets and the VAL pool), the scoring months
                 against vista-eval's rule worked out with numpy, the average curve against scikit-survival's
                 Kaplan-Meier, the mistakes file, the stated TRAIN survival against the key, and that output.json's
                 curve is read only when it is well formed (anything else an error that stops the run).
                   5. prints one patient's per-call prompt, with placeholder container paths

    cd optimizer/died-2yr-ehr && uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow \\
        python scripts/check_task.py synth /tmp/died2yr-ehr-synth
    TASK_DATA_DIR=/tmp/died2yr-ehr-synth uv run ... python scripts/check_task.py check
    ANSWER=survival TASK_DATA_DIR=... uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with scikit-survival \\
        --with pandas --with pyarrow python scripts/check_task.py check
"""

from __future__ import annotations

import json
import os
import random
import shutil
import stat
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def cmd_synth(root: Path) -> int:
    import pandas as pd

    if root.exists():
        shutil.rmtree(root)
    # Ids: 16 hex characters like the real ones; last digits mixed odd/even so the dry run's stand-in answers
    # both ways. Labels: 2 died within 2 years of 4 in each pool, 1 of them within 1 year.
    pts = [("synth0000000000a2", "train", 1, 1), ("synth0000000000b3", "train", 0, 0), ("synth0000000000c4", "train", 0, 0),
           ("synth0000000000d6", "train", 1, 0), ("synth0000000000e8", "val", 1, 1), ("synth0000000000f1", "val", 0, 0),
           ("synth000000000102", "val", 0, 0), ("synth000000000114", "val", 1, 0)]
    (root / "splits").mkdir(parents=True)
    for s in ("train", "val", "test"):
        (root / "splits" / f"{s}.jsonl").write_text("".join(json.dumps({"id": i, "label": y, "label_1yr": y1}) + "\n"
                                                             for i, sp, y, y1 in pts if sp == s))
    (root / "pools.json").write_text(json.dumps({s: sorted(i for i, sp, *_ in pts if sp == s) for s in ("train", "val")}))
    # Survival mode's answer key (made-up times in days; 1 death, 0 censored). TRAIN deaths at 200 and 600 days put
    # the scoring months at about 10-18.
    times = {"synth0000000000a2": (200, 1), "synth0000000000b3": (2000, 0), "synth0000000000c4": (1500, 0),
             "synth0000000000d6": (600, 1), "synth0000000000e8": (300, 1), "synth0000000000f1": (1800, 0),
             "synth000000000102": (900, 0), "synth000000000114": (700, 1)}
    (root / "survival.jsonl").write_text("".join(json.dumps({"id": i, "time_days": float(t), "event": e}) + "\n"
                                                 for i, (t, e) in times.items()))
    rng = random.Random(0)
    for n, (i, _s, y, _y1) in enumerate(pts):
        d = root / "patients" / i
        d.mkdir(parents=True)
        cutoff = f"2020-0{1 + n % 9}-15"
        (d / "meta.json").write_text(json.dumps({"id": i, "cutoff": cutoff}) + "\n")
        pd.DataFrame({"gender_concept_id": [8507], "gender_concept_name": ["MALE"], "year_of_birth": [1950 + n]}
                     ).to_parquet(d / "person.parquet", index=False)
        pd.DataFrame({"condition_concept_id": [4112853], "condition_concept_name": ["Synthetic condition"],
                      "condition_start_date": [pd.Timestamp(cutoff) - pd.Timedelta(days=30)]}
                     ).to_parquet(d / "condition_occurrence.parquet", index=False)
        pd.DataFrame({"measurement_concept_id": [3013682], "measurement_concept_name": ["Synthetic lab"],
                      "value_as_number": [round(rng.uniform(1, 9), 1)],
                      "measurement_date": [pd.Timestamp(cutoff) - pd.Timedelta(days=3)]}
                     ).to_parquet(d / "measurement.parquet", index=False)
        pd.DataFrame({"row": [1], "primarysite": ["C341"], "ajcctnmclinstagegroup": ["2B" if y else "1A"]}
                     ).to_parquet(d / "registry_diagnoses.parquet", index=False)
        (d / "notes.txt").write_text(f"===== NOTE 1 | {cutoff} 09:00:00 | Synthetic note =====\n"
                                     "SYNTHETIC TEST NOTE. Not a real patient.\n\n")
    # The exporter's side file: raw governing dates. Beside patients/, never to be staged.
    (root / "registry_dates").mkdir()
    pd.DataFrame({"id": [p[0] for p in pts], "row": 1, "dateofdiagnosis_date": pd.Timestamp("2020-01-01")}
                 ).to_parquet(root / "registry_dates" / "registry_diagnoses.parquet", index=False)
    print(f"{len(pts)} synthetic patients in {root}")
    return 0


def cmd_check() -> int:
    import config
    import adapters as ad
    import agentic
    from isolation.pass_outputs import Call
    from engine.schemas import ArtifactRef, RunArtifacts
    from sklearn.metrics import balanced_accuracy_score

    bad = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")

    spec = agentic.program_spec(forbidden=())
    pid = sorted(ad.pools()["val"])[0]
    src = ad.patient_dir(pid)
    side = {p.name for p in (config.DATA_DIR / "registry_dates").glob("*")} if (config.DATA_DIR / "registry_dates").exists() else set()

    # 1. staging copies the record files, read-only, and nothing else
    with tempfile.TemporaryDirectory() as tmp:
        spec.stage(Call(pid, {"id": pid}), Path(tmp), "p0")
        staged = Path(tmp) / agentic.RECORD
        names = {p.name for p in staged.iterdir()}
        want = {p.name for p in src.iterdir()} | ({agentic.SUMMARY} if config.WITH_SUMMARY else set())
        ro = all(not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) for p in staged.iterdir())
        everything = {p.name for p in Path(tmp).rglob("*")}
        report("staging copies exactly the patient's record files", names == want, f"{len(names)} files")
        report("staged files are read-only", ro)
        guide = Path(tmp) / agentic.GUIDE
        report("the data guide sits beside the record, read-only, byte-identical to context/patient_data_guide.md",
               guide.is_file() and guide.read_bytes() == agentic.GUIDE_SRC.read_bytes()
               and not guide.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
               and {p.name for p in Path(tmp).iterdir()} == {agentic.RECORD, agentic.GUIDE})
        report("nothing named registry_dates, and no side-file content path, in the session folder",
               "registry_dates" not in everything and not any("registry_dates" in str(p) for p in Path(tmp).rglob("*")),
               f"side folder present beside patients/: {bool(side)}")

    # 2. a stray file or folder in a patient folder stops staging (planted in a copy of the data, never in place)
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "patients" / pid
        shutil.copytree(src, fake)
        orig = config.DATA_DIR
        try:
            config.DATA_DIR = Path(tmp)
            for plant in ("registry_dates", "stray.txt"):
                target = fake / plant
                target.mkdir() if plant == "registry_dates" else target.write_text("x")
                try:
                    agentic.record_files(fake)
                    report(f"refuses a patient folder holding {plant}", False)
                except RuntimeError:
                    report(f"refuses a patient folder holding {plant}", True)
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            try:
                agentic.record_files(Path(tmp) / "elsewhere")
                report("refuses a folder outside DATA_DIR/patients", False)
            except (RuntimeError, FileNotFoundError):
                report("refuses a folder outside DATA_DIR/patients", True)
        finally:
            config.DATA_DIR = orig

    if config.SURVIVAL:
        check_survival(report, spec)
    elif config.PROBABILITY:
        check_probability(report, spec)
    else:
        check_yes_no(report, spec)
    check_run_rules(report)

    # 5. the per-call prompt, as a session gets it (placeholder container paths)
    from isolation.program_runner import CallPaths

    print("\n--- per-call prompt ---")
    print(spec.prompt(Call(pid, {"id": pid}), "label", CallPaths(program="/workspace/program", call="/workspace/call", ro={}), "PASS_ID"))
    print("--- end ---\n")
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def _score_file(answers: dict, split: str):
    import adapters as ad
    from engine.schemas import ArtifactRef, RunArtifacts

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": answers}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        return ad.TaskScorer().score(arts, split, {}), ad.TaskScorer().per_example_correct(arts, split)


def check_probability(report, spec) -> None:
    """Checks 3 and 4 for ANSWER=probability, worked out independently with scikit-learn."""
    import config
    import adapters as ad
    import agentic
    from sklearn.metrics import brier_score_loss, roc_auc_score

    # 3a. AUROC == sklearn on random answer sets drawn from few values (many ties), failures replaced by 0.5 here
    #     before sklearn sees them.
    rng = random.Random(1)
    worst = 0.0
    for _ in range(300):
        n = rng.randint(5, 300)
        y = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y)) < 2:
            continue
        levels = [rng.choice((0.05, 0.1, 0.3, 0.5, 0.9)) if rng.random() < 0.7 else rng.random() for _ in range(n)]
        ans = [{"died_within_1_year": p, "error": "x" if rng.random() < 0.1 else None} for p in levels]
        ours = ad.auroc([(g, ad.prob(a, "died_within_1_year")) for g, a in zip(y, ans)])
        theirs = roc_auc_score(y, [0.5 if a["error"] else a["died_within_1_year"] for a in ans])
        worst = max(worst, abs(ours - theirs))
    report("AUROC == sklearn on 300 random tie-heavy answer sets with failures at 0.5", worst < 1e-12,
           f"max |diff| {worst:.1e}")
    report("AUROC is 0.5 when one class is missing", ad.auroc([(1, 0.2), (1, 0.9)]) == 0.5)

    # 3b. the task score on the VAL pool, Brier, and the right-side flag, against sklearn and the death rate
    pool = sorted(ad.pools()["val"])
    rows = {r["id"]: r for r in ad.read_split("val")}
    answers = {}
    for n, i in enumerate(pool):  # mixed informative, constant, failed answers
        a = {key: (0.7 if rows[i][field] else 0.2) if n % 3 else 0.4 for key, field, _w in config.HORIZONS}
        answers[i] = {**a, "error": "not_a_probability" if n % 7 == 0 else None, "reason": "", "findings_log": None,
                      "reasons": []}
    score, correct = _score_file(answers, "val")

    def probs(key):
        return [0.5 if answers[i]["error"] else answers[i][key] for i in pool]

    want = sum(roc_auc_score([rows[i][field] for i in pool], probs(key)) for key, field, _w in config.HORIZONS) / 2
    report("task score on the VAL pool == mean of sklearn's AUROC at 1 and 2 years",
           abs(score.primary_metric.value - want) < 1e-9, f"{score.primary_metric.value:.4f} vs {want:.4f}")
    briers = [(score.breakdown[w.replace(" ", "")]["brier"],
               round(brier_score_loss([rows[i][field] for i in pool], probs(key)), 4)) for key, field, w in config.HORIZONS]
    report("Brier per horizon == sklearn's", all(a == b for a, b in briers), str(briers))
    thr = {"died_within_1_year": 718 / 3886, "died_within_2_years": 1188 / 3886}
    want_right = {i: None if answers[i]["error"] else all(
        (answers[i][key] >= thr[key]) == bool(rows[i][field]) for key, field, _w in config.HORIZONS) for i in pool}
    report("per-example right = on the correct side of each horizon's TRAIN death rate, failed = None",
           correct == want_right, f"{sum(v is True for v in correct.values())} right of {len(pool)}")
    one = score.breakdown["1year"]
    report("breakdown counts ties: distinct values and share on the most common value",
           one["distinct_values"] == len({p for p in probs("died_within_1_year")})
           and abs(one["share_on_most_common_value"] - max(probs("died_within_1_year").count(v) for v in
                                                           set(probs("died_within_1_year"))) / len(pool)) < 1e-3)

    # 3c. calibration bins (0.0-0.1 ... 0.9-1.0, with p = 1.0 in the top bin) and the impossible 1yr > 2yr count,
    #     on hand-built answers over the first ten VAL patients
    ten = [f"hand{k:02d}" for k in range(10)]  # made-up patients with their own gold, scored directly
    p1 = [0.0, 0.05, 0.1, 0.35, 0.5, 0.99, 1.0, 1.0, 0.3, 0.2]
    p2 = [0.1, 0.05, 0.2, 0.30, 0.6, 1.0, 1.0, 0.9, 0.3, 0.1]  # 1yr above 2yr at positions 3, 7, 9
    hand = {i: {"died_within_1_year": a, "died_within_2_years": b, "error": None, "reason": "", "reasons": [],
                "findings_log": None} for i, a, b in zip(ten, p1, p2)}
    hgold = {i: {"label_1yr": k % 2, "label": 1 if k % 2 else k % 3 == 0} for k, i in enumerate(ten)}
    hs = ad.TaskScorer()._score_probability(hgold, hand, {}, "val")
    bins = hs.breakdown["1year"]["calibration_by_decile"]
    want_bins = {"p0.0:n": 2, "p0.1:n": 1, "p0.2:n": 1, "p0.3:n": 2, "p0.5:n": 1, "p0.9:n": 3}
    report("calibration bins: tenths, p = 1.0 in the top bin; impossible 1yr > 2yr pairs counted",
           {k: v for k, v in bins.items() if k.endswith(":n")} == want_bins and hs.breakdown["p_1yr_above_p_2yr"] == 3,
           f"{hs.breakdown['p_1yr_above_p_2yr']} impossible pairs")

    # 3d. the optimizer's mistakes file carries each TRAIN patient's probabilities and reasons
    from engine.schemas import ArtifactRef, RunArtifacts

    train = sorted(ad.pools()["train"])[:4]
    trows = {r["id"]: r for r in ad.read_split("train")}
    tans = {i: {"died_within_1_year": 0.9, "died_within_2_years": 0.95, "error": None, "reason": "x",
                "reasons": [{"fact": "f", "file": "notes.txt", "date": "2020-01-01"}], "findings_log": None}
            for i in train}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": tans}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        tscore = ad.TaskScorer().score(arts, "train", {"_iter": 1})
        corpus = ad.mistake_corpus(Path(tmp) / "fb")(arts, tscore)
        entries = json.loads(Path(corpus.ref).read_text())
    good = all(e["probability_died_within_1_year"] == 0.9 and e["probability_died_within_2_years"] == 0.95
               and e["reasons"] and "said_died_within_1_year" not in e
               and e["mistake"] == (not trows[e["id"]]["label_1yr"] or not trows[e["id"]]["label"]) for e in entries)
    report("mistakes file: probabilities and reasons per TRAIN patient; a survivor at 90% is a mistake", good and
           len(entries) == len(train), f"{sum(e['mistake'] for e in entries)} of {len(entries)} mistakes")

    # 4. reading output.json: a number 0..100 per horizon, stored as 0..1; anything else an error (scored at 0.5)
    r = [{"fact": f"f{k}", "file": "notes.txt", "date": "2020-01-01", "extra": "dropped"} for k in range(7)]
    cases = [({"died_within_1_year": 37, "died_within_2_years": 61.5, "reasons": r[:2]}, (0.37, 0.615), None, 2),
             ({"died_within_1_year": 0, "died_within_2_years": 100, "reasons": r}, (0.0, 1.0), None, agentic.MAX_REASONS),
             ({"died_within_1_year": 101, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": -1, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": "40", "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": True, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": "yes", "died_within_2_years": "no"}, (None, None), "not_a_probability", 0),
             ({"died_within_2_years": 30, "reasons": ["not an object", r[0]]}, (None, 0.3), "not_a_probability", 1)]
    ok, why = True, ""
    with tempfile.TemporaryDirectory() as tmp:
        for n, (body, want_said, want_err, n_reasons) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps(body))
            manifest = spec.summarize(tmp, [{"call_id": "c", "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(manifest).read_text())["answers"]["c"]
            good = ((got["died_within_1_year"], got["died_within_2_years"]) == want_said and got["error"] == want_err
                    and len(got["reasons"]) == n_reasons and all(set(x) == {"fact", "file", "date"} for x in got["reasons"]))
            ok &= good
            why += "" if good else f" case {n}: {got}"
    report("output.json read as 0-100 per horizon; anything else is an error; reasons kept, capped, cleaned", ok,
           f"{len(cases)} cases{why}")
    report("a failed answer scores 0.5 at both horizons",
           ad.prob({"died_within_1_year": 0.9, "error": "x"}, "died_within_1_year") == 0.5)


def check_survival(report, spec) -> None:
    """Checks 3 and 4 for ANSWER=survival, worked out independently with scikit-survival and numpy."""
    import numpy as np
    import config
    import adapters as ad
    import agentic
    from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.util import Surv

    rng = np.random.default_rng(3)

    # 3a. Harrell's C == sksurv on random sets with tied times, tied risks and ties between a death and a censoring
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(5, 200))
        t = rng.integers(1, 40, n).astype(float)
        e = rng.random(n) < 0.6
        if not e.any():
            continue
        r = np.round(rng.normal(size=n), 1)
        worst = max(worst, abs(ad.c_index(list(t), list(e), list(r)) - concordance_index_censored(e, t, r)[0]))
    report("C-index == sksurv concordance_index_censored on 200 random tie-heavy sets", worst < 1e-12,
           f"max |diff| {worst:.1e}")

    # 3b. Brier at each time, integrated Brier and time-dependent AUC == sksurv, on random step curves
    worst_b = worst_i = worst_a = 0.0
    sets = 0
    for _ in range(100):
        ntr, nte = int(rng.integers(30, 300)), int(rng.integers(20, 200))
        ttr = rng.integers(1, 3000, ntr).astype(float)
        etr = rng.random(ntr) < 0.5
        tte = np.minimum(rng.integers(1, 3000, nte).astype(float), ttr.max() - 1)
        ete = rng.random(nte) < 0.5
        lo, hi = max(ttr.min(), tte.min()) + 1, min(np.percentile(ttr[etr], 90), tte.max() - 1)
        if hi <= lo + 10 or ete.sum() < 2:
            continue
        times = np.unique(np.round(np.linspace(lo, hi, int(rng.integers(2, 12)))))
        surv = np.sort(np.round(rng.random((nte, len(times))), 1), axis=1)[:, ::-1]  # non-increasing, many ties
        ytr, yte = Surv.from_arrays(etr, ttr), Surv.from_arrays(ete, tte)
        cens = ad.Censoring(list(ttr), list(etr))
        _, sk_b = brier_score(ytr, yte, surv, times)
        ours_b = [ad.brier_at(list(tte), list(ete), list(surv[:, k]), float(tm), cens) for k, tm in enumerate(times)]
        worst_b = max(worst_b, float(np.max(np.abs(np.array(ours_b) - sk_b))))
        if len(times) >= 2:
            sk_i = integrated_brier_score(ytr, yte, surv, times)
            worst_i = max(worst_i, abs(ad.trapezoid(ours_b, list(times)) / (times[-1] - times[0]) - sk_i))
        for k, tm in enumerate(times):
            risk = 1 - surv[:, k]
            if not ((tte <= tm) & ete).any() or not (tte > tm).any():
                continue
            sk_a, _ = cumulative_dynamic_auc(ytr, yte, risk, [tm])
            worst_a = max(worst_a, abs(ad.dynamic_auc_at(list(tte), list(ete), list(risk), float(tm), cens) - float(sk_a[0])))
        sets += 1
    report(f"Brier at each time == sksurv brier_score on {sets} random curve sets", sets > 50 and worst_b < 1e-10,
           f"max |diff| {worst_b:.1e}")
    report("integrated Brier == sksurv integrated_brier_score", worst_i < 1e-10, f"max |diff| {worst_i:.1e}")
    report("time-dependent AUC == sksurv cumulative_dynamic_auc", worst_a < 1e-10, f"max |diff| {worst_a:.1e}")

    # 3c. the scoring months follow vista-eval's rule (numpy), and the task score on the VAL pool == sksurv directly
    train = ad.read_split("train")
    pool = sorted(ad.pools()["val"])
    rows = {r["id"]: r for r in ad.read_split("val")}
    ttr = np.array([r["time_days"] for r in train])
    etr = np.array([bool(r["event"]) for r in train])
    tev = np.array([rows[i]["time_days"] for i in pool])
    eev = np.array([bool(rows[i]["event"]) for i in pool])
    upper = min(float(np.percentile(ttr[etr], 90)) * 0.999,
                min(ttr[etr].max(), tev[eev].max() if eev.any() else ttr[etr].max()) * 0.999)
    lower = max(ttr.min(), tev.min()) + 1e-6
    want_months = [m for m in range(1, 121) if lower <= m * 30.44 <= upper]
    months = ad.scoring_months(train, [rows[i] for i in pool])
    report("scoring months == vista-eval's rule worked out with numpy", months == want_months and len(months) >= 2,
           f"months {months[0]}-{months[-1]} ({len(months)})")
    km_x, km_p = kaplan_meier_estimator(etr, ttr / ad.DAYS_PER_MONTH)
    ax, ay = ad.average_curve()
    report("average curve == sksurv Kaplan-Meier on TRAIN", np.allclose(ax[1:], km_x) and np.allclose(ay[1:], km_p)
           and ax[0] == 0.0 and ay[0] == 1.0, f"{len(km_x)} steps")
    answers = {}
    for n, i in enumerate(pool):  # informative, flat, average; a few unreadable
        m = rows[i]["time_days"] / ad.DAYS_PER_MONTH
        curve = ([[0, 1.0], [m * 0.8, 0.3 if rows[i]["event"] else 0.8], [130, 0.2]] if n % 3 == 0 else
                 [[0, 1.0], [6, 0.9], [24, 0.7], [60, 0.5], [130, 0.3]] if n % 3 == 1 else
                 [list(p) for p in zip(ax, ay)] + [[130.0, ay[-1]]])
        curve = [p for k, p in enumerate(curve) if k == 0 or p[0] > curve[k - 1][0]]
        bad = n % 11 == 5
        answers[i] = {"curve": None if bad else curve, "error": "rising_curve" if bad else None, "reasons": [],
                      "reason": "", "findings_log": None}
    score, correct = _score_file(answers, "val")
    days = np.array(months) * ad.DAYS_PER_MONTH
    grid = {i: (ad.curve_at(answers[i]["curve"], months) if answers[i]["curve"] else
                [float(ay[np.searchsorted(ax, mm, side="right") - 1]) for mm in months]) for i in pool}
    S = np.array([grid[i] for i in pool])
    want_ibs = integrated_brier_score(Surv.from_arrays(etr, ttr), Surv.from_arrays(eev, tev), S, days)
    want_c = concordance_index_censored(eev, tev, -S.mean(axis=1))[0]
    report("task score on the VAL pool == sksurv integrated Brier (unreadable answers at the average curve)",
           abs(score.primary_metric.value - want_ibs) < 1e-10 and score.primary_metric.higher_is_better is False,
           f"{score.primary_metric.value:.4f} vs {want_ibs:.4f}; lower is better")
    report("C-index on the VAL pool == sksurv with risk = minus the curve's mean over the scoring months",
           abs(score.breakdown["c_index"] - round(want_c, 4)) < 1e-9, f"{score.breakdown['c_index']} vs {want_c:.4f}")
    nbad = sum(1 for a in answers.values() if a["error"])
    report("breakdown counts unreadable answers and gives the average curve's own score",
           score.breakdown["program_errors"] == nbad and score.breakdown.get("error_rising_curve", 0) == nbad
           and "integrated_brier" in score.breakdown["average_curve"], f"{nbad} unreadable")
    report("per-example: failed = None, others right or wrong",
           all((correct[i] is None) == bool(answers[i]["error"]) for i in pool))

    # 3d. the mistakes file: gold death and follow-up, the curve at 12/24/36/60 months, reasons; a confident wrong-way
    #     curve is a mistake (worse than the average curve on that patient)
    from engine.schemas import ArtifactRef, RunArtifacts

    trows = {r["id"]: r for r in train}
    real = len(train) > 1000
    if real:  # deaths between months 6 and 48 and censorings past month 90, so the window sees every one; alternating
        mo = {i: trows[i]["time_days"] / ad.DAYS_PER_MONTH for i in ad.pools()["train"]}
        died = [i for i in sorted(mo) if trows[i]["event"] and 6 < mo[i] < 48][:2]
        alive = [i for i in sorted(mo) if not trows[i]["event"] and mo[i] > 90][:2]
        died.sort(key=lambda i: mo[i])
        # k even right-way, k odd wrong-way, one death and one censoring each way. The wrong-way death is the earlier:
        # the window ends just before the last death scored (vista-eval's rule), so the later one is outside it.
        tids = [died[1], alive[0], alive[1], died[0]]
    else:  # synthetic: too few patients for a scoring window; only the fields are checked
        tids = sorted(ad.pools()["train"])[:4]
    tans = {}
    for k, i in enumerate(tids):  # even: the right way (drops just after a death, stays high past a censoring); odd: wrong way
        m = trows[i]["time_days"] / ad.DAYS_PER_MONTH
        died = bool(trows[i]["event"])
        if k % 2 == 0:
            curve = [[0, 1.0], [m + 0.01, 0.02], [130, 0.01]] if died else [[0, 1.0], [130, 0.98]]
        else:
            curve = [[0, 0.98], [130, 0.98]] if died else [[0, 0.02], [130, 0.01]]
        tans[i] = {"curve": curve, "error": None, "reason": "x", "findings_log": None,
                   "reasons": [{"fact": "f", "file": "notes.txt", "date": "2020-01-01"}]}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": tans}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        tscore = ad.TaskScorer().score(arts, "train", {"_iter": 1})
        corpus = ad.mistake_corpus(Path(tmp) / "fb")(arts, tscore)
        entries = {e["id"]: e for e in json.loads(Path(corpus.ref).read_text())}
    ok = all("gold_died" in e and "gold_months_followed" in e and "survival_at_12m" in e and e["reasons"]
             and "said_died_within_1_year" not in e for e in entries.values())
    wrong_way = [entries[i]["mistake"] for k, i in enumerate(tids) if k % 2 == 1]
    right_way = [entries[i]["mistake"] for k, i in enumerate(tids) if k % 2 == 0]
    report("mistakes file: gold, follow-up, curve at 12-60 months, reasons"
           + ("; a confident wrong-way curve is a mistake, a right-way one is not" if real else " (synthetic: fields only)"),
           ok and len(entries) == len(tids) and (not real or (all(wrong_way) and not any(right_way))),
           f"{sum(e['mistake'] for e in entries.values())} of {len(entries)} mistakes")

    # 3e. the cohort survival the prompt states == Kaplan-Meier of the real key's TRAIN (skipped on synthetic data)
    if len(train) > 1000:
        got = {k: f"{100 * float(km_p[np.searchsorted(km_x, m, side='right') - 1]):.1f}%" for k, m in
               (("1 year", 12), ("2 years", 24), ("3 years", 36), ("5 years", 60), ("10 years", 120))}
        report("prompt's TRAIN survival == Kaplan-Meier of the key", got == agentic.TRAIN_SURVIVAL, str(got))
        report("cover months = TRAIN's window end", 70 < ad.cover_months() < 80, f"{ad.cover_months():.1f}")

    # 4. reading output.json's curve
    cover = ad.cover_months()
    far = max(cover + 1, 130.0)
    cases = [([[0, 1], [12, 0.8], [far, 0.4]], None), ([{"month": 0, "survival": 1}, {"month": far, "survival": 0.5}], None),
             ([[0, 1], [12, 0.8], [24, 0.8], [far, 0.0]], None),                       # flat stretches and 0 are fine
             ([[0, 1], [12, 0.8], [24, 0.81], [far, 0.5]], "rising_curve"),
             ([[0, 1], [12, 1.2], [far, 0.5]], "outside_0_1"), ([[0, 1], [12, -0.1], [far, 0]], "outside_0_1"),
             ([[0, 1], [12, 0.8], [cover - 1, 0.5]], "too_short"),
             ([[0, 1], [12, 0.8], [12, 0.7], [far, 0.5]], "not_a_curve"),              # months must increase
             ([[-1, 1], [far, 0.5]], "not_a_curve"), ([[0, True], [far, 0.5]], "not_a_curve"),
             ([[0, "1"], [far, 0.5]], "not_a_curve"), ([[0, float("nan")], [far, 0.5]], "not_a_curve"),
             ([], "not_a_curve"), ("[[0,1]]", "not_a_curve"), (None, "not_a_curve"),
             ([[k * far / 2000, 1 - k / 2001] for k in range(2001)], "not_a_curve")]   # over MAX_POINTS
    ok, why = True, ""
    with tempfile.TemporaryDirectory() as tmp:
        for n, (curve, want_err) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps({"survival_curve": curve, "reasons": [{"fact": "f", "file": "notes.txt",
                                                                               "date": "2020-01-01"}]}))
            manifest = spec.summarize(tmp, [{"call_id": "c", "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(manifest).read_text())["answers"]["c"]
            good = (got["error"] == want_err and (got["curve"] is None) == (want_err is not None)
                    and len(got["reasons"]) == 1)
            ok &= good
            why += "" if good else f" case {n}: {got['error']}"
    report("output.json's curve read only when well formed; rising, outside 0-1, short or malformed = error", ok,
           f"{len(cases)} cases{why}")
    report("a curve is read as a step curve, 1 before its first point",
           ad.curve_at([[2, 0.9], [5, 0.5]], [1, 2, 4.9, 5, 9]) == [1.0, 0.9, 0.9, 0.5, 0.5])


def check_run_rules(report) -> None:
    """6. run.py's stops, on hand-built run-summary rows and receipts: the no-new-best count and the dollar limits."""
    import config
    import run
    from engine.loop import LoopStop

    def row(it, st, v, base=0.60):
        return {"iter": it, "status": st, config.METRIC_NAME: v, "base_metric": base}

    m = "measured"
    cases = [  # (rows, rounds in a row with no new best)
        ([], 0),
        ([row(1, m, 0.65)], 0),                                    # beats the seed's 0.60
        ([row(1, m, 0.60)], 1),                                    # ties the seed: not a new best
        ([row(1, m, 0.55), row(2, m, 0.58)], 2),                   # both below the seed
        ([row(1, m, 0.55), row(2, m, 0.62)], 0),                   # a new best resets the count
        ([row(1, m, 0.62), row(2, m, 0.61), row(3, "failed", None)], 2),  # a failed round counts
        ([row(1, m, 0.62), row(2, m, 0.61), row(3, "saved_unmeasured", None)], 1),  # an unscored round is skipped
        ([row(1, m, 0.70, base=0.75), row(2, m, 0.74, base=0.70)], 2),  # the start version's score is the first best
    ]
    got = [run.rounds_without_best(r, config.METRIC_NAME) for r, _ in cases]
    report("no-new-best count on 8 hand-built run summaries", got == [w for _, w in cases], f"{got}")
    low = [  # lower is better (survival's integrated Brier): the same shapes with the direction flipped
        ([row(1, m, 0.17, base=0.18)], 0), ([row(1, m, 0.18, base=0.18)], 1),
        ([row(1, m, 0.19, base=0.18), row(2, m, 0.185, base=0.18)], 2),
        ([row(1, m, 0.19, base=0.18), row(2, m, 0.17, base=0.18)], 0)]
    got = [run.rounds_without_best(r, config.METRIC_NAME, higher_is_better=False) for r, _ in low]
    report("no-new-best count with lower-is-better scores", got == [w for _, w in low], f"{got}")

    saved = (config.STOP_AFTER_ROUNDS_WITHOUT_BEST, config.RUN_BUDGET_USD, config.ROUND_BUDGET_USD,
             config.OPTIMIZER_SESSION_BUDGET_USD)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        summary, archive = tmp / "run_summary.json", tmp / "archive"

        def receipt(pass_name, n, cost, run_id="r1"):
            d = archive / pass_name / "receipts"
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                (d / f"c{len(list(d.iterdir()))}.json").write_text(json.dumps(
                    {"run_id": run_id, "stages": [{"cost_usd": cost}]}))

        def stops(rows, rule=(2, 200.0, 50.0, 3.0)):
            (config.STOP_AFTER_ROUNDS_WITHOUT_BEST, config.RUN_BUDGET_USD, config.ROUND_BUDGET_USD,
             config.OPTIMIZER_SESSION_BUDGET_USD) = rule
            summary.write_text(json.dumps({"iters": rows}))
            try:
                run.stop_rule(summary, archive, "r1")(None)
                return None
            except LoopStop as exc:
                return exc.reason

        try:
            receipt("iter-1-train-aa", 100, 0.10)   # $10
            receipt("iter-1-val-bb", 200, 0.10)     # $20 -> round 1 $30
            receipt("iter-1-val-cc", 300, 0.10, run_id="other")  # another run's sessions: not counted
            # lower-is-better (survival): the same rows mirrored, so round 1 beats the start and round 2 beats round 1
            hb = config.HIGHER_IS_BETTER
            r1 = [row(1, m, 0.70 if hb else 0.30, base=0.60 if hb else 0.40)]
            ok = stops(r1) is None
            receipt("iter-2-train-dd", 400, 0.10)   # round 2: $40 so far, not yet scored
            ok &= stops(r1 + [row(2, "saved_unmeasured", None)]) is None
            receipt("iter-2-val-ee", 200, 0.10)     # round 2 now $60, scored -> over $50
            ok &= stops(r1 + [row(2, m, 0.72 if hb else 0.28, base=0.60 if hb else 0.40)]) == "round_over_budget"
            ok &= stops(r1 + [row(2, m, 0.72 if hb else 0.28, base=0.60 if hb else 0.40)], rule=(2, 90.0, 0.0, 3.0)) == "run_over_budget"  # $90 + 2 x $3 > $90
            ok &= stops(r1 + [row(2, m, 0.72 if hb else 0.28, base=0.60 if hb else 0.40)], rule=(2, 0.0, 0.0, 3.0)) is None  # all off
            spend = run.run_spend(archive, "r1")
            ok &= {k: round(v, 6) for k, v in spend.items()} == {1: 30.0, 2: 60.0}
            report("dollar stops: round over $50 and run over its limit stop; other runs and unscored rounds don't count",
                   ok, f"spend by round {spend}")
            if config.SURVIVAL:  # any failed or unreadable answer in this run's passes stops the next pass
                def answers(pass_name, n_bad):
                    (archive / pass_name / "answers.json").write_text(json.dumps({"answers": {
                        f"p{k}": {"curve": None if k < n_bad else [[0, 1]], "error": "too_short" if k < n_bad else None}
                        for k in range(5)}}))
                answers("iter-1-train-aa", 0)
                answers("iter-1-val-cc", 2)  # another run's pass: not counted
                ok2 = stops(r1, rule=(2, 0.0, 0.0, 3.0)) is None
                answers("iter-2-val-ee", 1)
                ok2 &= stops(r1, rule=(2, 0.0, 0.0, 3.0)) == "unreadable_curve"
                ok2 &= run.unreadable_answers(archive, "r1") == {"iter-2-val-ee": 1}
                report("an unreadable curve in this run's passes stops the run; another run's doesn't", ok2)

        finally:
            (config.STOP_AFTER_ROUNDS_WITHOUT_BEST, config.RUN_BUDGET_USD, config.ROUND_BUDGET_USD,
             config.OPTIMIZER_SESSION_BUDGET_USD) = saved

    if config.SURVIVAL:  # 7. through the real driver: one rising curve in the first pass stops the run before scoring it
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DRY_RUN_UNREADABLE": "1", "TASK_WORKDIR": str(Path(tmp) / "wd")}
            proc = subprocess.run([sys.executable, "-B", "run.py", "dry-run", "--iterations", "2"], cwd=HERE, env=env,
                                  capture_output=True, text=True, timeout=900)
            out = proc.stdout + proc.stderr
            summaries = list(Path(tmp).glob("wd-dryrun/private/runs/dry-run/run_summary.json"))
            rows = json.loads(summaries[0].read_text()).get("iters", []) if summaries else []
            scored = [r for r in rows if r.get(config.METRIC_NAME) is not None]
            report("dry run through the driver: one unreadable curve stops the run before that pass is scored",
                   proc.returncode == 3 and "STOPPED (unreadable_curve)" in out and not scored,
                   f"exit {proc.returncode}; {len(scored)} scored rounds; "
                   + next((l for l in out.splitlines() if l.startswith("STOPPED")), "no STOPPED line"))


def check_yes_no(report, spec) -> None:
    """Checks 3 and 4 for the default yes/no answers."""
    import config
    import adapters as ad
    import agentic
    from engine.schemas import ArtifactRef, RunArtifacts
    from sklearn.metrics import balanced_accuracy_score

    # 3. the scorer's balanced accuracy equals scikit-learn's, worked out independently: a failed or unreadable answer
    #    becomes the opposite of gold before sklearn sees it, and the task score is the mean over the two horizons.
    def sk(golds, answers, key):
        preds = [int(a[key]) if isinstance(a.get(key), bool) and not a.get("error") else 1 - g
                 for g, a in zip(golds, answers)]
        return balanced_accuracy_score(golds, preds)

    rng = random.Random(1)
    worst = 0.0
    for _ in range(200):
        n = rng.randint(5, 300)
        y = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y)) < 2:
            continue
        ans = [{"died_within_1_year": rng.random() < 0.4, "error": "x" if rng.random() < 0.1 else None} for _ in range(n)]
        ours = ad.balanced_accuracy([(g, ad.right(a, "died_within_1_year", g)) for g, a in zip(y, ans)])
        worst = max(worst, abs(ours - sk(y, ans, "died_within_1_year")))
    report("balanced accuracy == sklearn on 200 random answer sets with failures", worst < 1e-12, f"max |diff| {worst:.1e}")

    pool = sorted(ad.pools()["val"])
    rows = {r["id"]: r for r in ad.read_split("val")}
    answers = {}
    for n, i in enumerate(pool):  # mixed right, wrong, unreadable and failed answers
        a = {key: bool(rows[i][field]) if n % 3 else n % 2 == 0 for key, field, _w in config.HORIZONS}
        answers[i] = {**a, "error": "not_yes_or_no" if n % 7 == 0 else None, "reason": "", "findings_log": None}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": answers}))
        score = ad.TaskScorer().score(RunArtifacts(batch_id="b", status="ok",
                                                   artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),)), "val", {})
    want = sum(sk([rows[i][field] for i in pool], [answers[i] for i in pool], key)
               for key, field, _w in config.HORIZONS) / len(config.HORIZONS)
    report("task score on the VAL pool == mean of sklearn's balanced accuracy at 1 and 2 years",
           abs(score.primary_metric.value - want) < 1e-9, f"{score.primary_metric.value:.4f} vs {want:.4f}")

    # 4. reading output.json: exactly yes or no (any case, spaces), anything else an error and scored wrong
    cases = [({"died_within_1_year": "Yes", "died_within_2_years": " no "}, (True, False), None),
             ({"died_within_1_year": "no", "died_within_2_years": "yes"}, (False, True), None),
             ({"died_within_1_year": "maybe", "died_within_2_years": "yes"}, (None, True), "not_yes_or_no"),
             ({"died_within_1_year": 0.7, "died_within_2_years": True}, (None, None), "not_yes_or_no"),
             ({"probability": 0.4}, (None, None), "not_yes_or_no")]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for n, (body, want_said, want_err) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps(body))
            manifest = spec.summarize(tmp, [{"call_id": "c", "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(manifest).read_text())["answers"]["c"]
            ok &= (got["died_within_1_year"], got["died_within_2_years"]) == want_said and got["error"] == want_err
    report("output.json read as yes/no exactly; anything else is an error", ok, f"{len(cases)} cases")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "synth" and len(sys.argv) == 3:
        sys.exit(cmd_synth(Path(sys.argv[2])))
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    sys.exit(__doc__)
