"""$0 checks of the tb-questions task code and data, for one agent (TB_AGENT=treatment or imaging): staging, the
scorer, reading a session's output.json, the pools, the run's stop rules, the quoted card text and the "tumor board"
wording. No model call, no Docker. Reads the task data read-only (TASK_DATA_DIR, default the mount), the private id
map and the eval set's TEST lists' patient column, and prints counts only, never an id, a row or summary text.

    check   1. staging: treatment copies exactly summary.txt; imaging exactly ct/prior/ and ct/current/ (ct.npz and
               meta.json each), for a VAL patient; nothing else
            2. scorer: yes/no AUROC == scikit-learn on random tie-heavy sets (failures at 0.5); several-option AUROC ==
               the mean of scikit-learn's one-vs-rest AUROCs over the answers present, and AUROC, per-option AUROC,
               top-pick balanced accuracy and top two == the eval set's own option scorer (evalset.score._option_metrics,
               read from the tb-question-cards checkout) on random answer sets; the task score on VAL == the mean over
               the questions asked; the dry run's stand-in scores what it should; the mistakes file
            3. output.json: yes/no read as 0-100; options rescaled to sum to 1 whatever the total; a missing letter,
               out of range, a word, a bool, NaN, all zeros, or a question asked but not answered = failed; reasons kept
            4. pools: the counts the build printed, splits == pools, one row per patient, no patient in TRAIN and VAL,
               none of the TEST list's patients in either, no TEST file yet; treatment: every patient has a summary;
               imaging: the redraw kept the progression share within one pair, every pair has both volumes
            5. run.py's stops: the no-new-best count and the dollar limits, on hand-built summaries and receipts
            6. the question texts == the eval-set cards (feat/tb-question-cards), and no bare "board" anywhere the
               sessions or the optimizer read
            7. prints the per-call prompt, with placeholder container paths and the summary replaced

    cd optimizer/tb-questions && TB_AGENT=treatment uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow --with pyyaml python -B scripts/check_task.py check
"""

from __future__ import annotations

import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SAMPLES = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/samples")
#: The eval-set cards (feat/tb-question-cards 1888097), for checks 2 and 6. Skipped, and said so, when absent.
CARDS_SRC = Path.home() / "code" / "vista_eval_vlm-tb-cards" / "src"
#: What the build printed (2026-11-20): pool sizes per agent.
SIZES = {"treatment": {"train": 492, "val": 111}, "imaging": {"train": 188, "val": 100},
         "treatment-record": {"train": 492, "val": 111}}


def cmd_check() -> int:
    import config
    import adapters as ad
    import agentic
    from isolation.pass_outputs import Call

    bad = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")

    print(f"task={config.TASK_NAME}  metric={config.METRIC_NAME}  questions={[q.key for q in config.QUESTIONS]}  "
          f"schedule={config.SCHEDULE}  iterations={config.ITERATIONS}  stop after {config.STOP_AFTER_ROUNDS_WITHOUT_BEST} "
          f"rounds without best, ${config.ROUND_BUDGET_USD:.0f}/round, ${config.RUN_BUDGET_USD:.0f}/run, "
          f"${config.PROGRAM_CALL_BUDGET_USD:.2f}/session, {config.PROGRAM_MAX_CONCURRENT} at a time, route "
          f"{config.ROUTE.name}, search proxy {config.SEARCH_PROXY}  workdir={config.WORKDIR.name}")
    spec = agentic.program_spec(forbidden=())

    # 1. staging
    rows = {r["id"]: r for r in ad.read_split("val")}
    pid = next(i for i in sorted(ad.pools()["val"])
               if config.TREATMENT or (ad.patient_dir(i) / "current" / "meta.json").is_file())
    want = (["summary.txt"] if config.TREATMENT else
            ["ct/current/ct.npz", "ct/current/meta.json", "ct/prior/ct.npz", "ct/prior/meta.json"])
    with tempfile.TemporaryDirectory() as tmp:
        spec.stage(Call(pid, {"id": pid, "asked": ad.asked(rows[pid])}), Path(tmp), "p0")
        got = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file())
        same = all((Path(tmp) / w).read_bytes() == (ad.patient_dir(pid) / w.removeprefix("ct/")).read_bytes() for w in want)
        readonly = all(not ((Path(tmp) / w).stat().st_mode & 0o222) for w in want)
    if config.RECORD:  # the summary as above, plus record/ (exactly the record folder's files) and the guide
        rec_files = sorted(f.name for f in ad.record_dir(pid).iterdir())
        want = sorted(want + [agentic.GUIDE] + [f"{agentic.RECORD}/{n}" for n in rec_files])
        with tempfile.TemporaryDirectory() as tmp:
            spec.stage(Call(pid, {"id": pid, "asked": ad.asked(rows[pid])}), Path(tmp), "p0")
            got = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file())
            src = {"summary.txt": ad.patient_dir(pid) / "summary.txt", agentic.GUIDE: agentic.GUIDE_SRC,
                   **{f"{agentic.RECORD}/{n}": ad.record_dir(pid) / n for n in rec_files}}
            same = got == want and all((Path(tmp) / w).read_bytes() == src[w].read_bytes() for w in want)
            readonly = all(not ((Path(tmp) / w).stat().st_mode & 0o222) for w in want)
    report("staging copies exactly the patient's inputs, byte for byte, read-only", got == want and same and readonly,
           f"{len(got)} files")
    if config.RECORD:
        check_record(report, spec)

    check_scorer(report)
    check_output(report, spec)
    check_pools(report)
    check_run_rules(report)
    check_wording(report)

    from isolation.program_runner import CallPaths

    print("\n--- per-call prompt (summary replaced) ---")
    text = spec.prompt(Call(pid, {"id": pid, "asked": ad.asked(rows[pid])}), "label",
                       CallPaths(program="/workspace/program", call="/workspace/call", ro={}), "PASS_ID")
    if config.TREATMENT:
        summary = (ad.patient_dir(pid) / "summary.txt").read_text(encoding="utf-8").strip()
        report("the prompt quotes the patient's summary whole", summary in text)
        text = text.replace(summary, "[the patient's tumor board-time summary]")
    print(text)
    print("--- end ---\n")
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def check_record(report, spec) -> None:
    """Record mode: the staging whitelist refuses a stray file or a folder outside the record root, and every VAL and
    TRAIN record holds nothing dated after its cutoff (notes.txt headers and every date column of every table; end
    dates are clipped to the cutoff by the exporter, so they count too). Counts only."""
    import shutil

    import pandas as pd

    import adapters as ad
    import agentic
    import config

    pid = sorted(ad.pools()["val"])[0]
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = Path(tmp) / "patients"
        shutil.copytree(ad.record_dir(pid), fake_root / pid)
        (fake_root / pid / "registry_dates.parquet").write_bytes(b"x")  # the exporter's side file must never pass
        saved, config.RECORD_PATIENTS = config.RECORD_PATIENTS, fake_root
        try:
            stray = _raises(lambda: agentic.record_files(fake_root / pid))
            (fake_root / pid / "registry_dates.parquet").unlink()
            outside = _raises(lambda: agentic.record_files(Path(tmp)))
            clean = not _raises(lambda: agentic.record_files(fake_root / pid))
        finally:
            config.RECORD_PATIENTS = saved
    report("record staging refuses a stray file and a folder outside the record root, and accepts a clean folder",
           stray and outside and clean)

    ids = sorted(ad.pools()["val"]) + sorted(ad.pools()["train"])
    have = [i for i in ids if ad.record_dir(i).is_dir()]
    late, tables_seen, no_meta = 0, 0, 0
    for i in have:
        d = ad.record_dir(i)
        try:
            cutoff = pd.Timestamp(json.loads((d / "meta.json").read_text())["cutoff"]).normalize()
        except (OSError, ValueError, KeyError):
            no_meta += 1
            continue
        limit = cutoff + pd.Timedelta(days=1)  # anything on the cutoff day itself is allowed (inclusive)
        for f in d.glob("*.parquet"):
            df = pd.read_parquet(f)
            tables_seen += 1
            for c in df.columns:
                if c.endswith(("_date", "_datetime")) or str(df[c].dtype).startswith(("datetime", "date")):
                    v = pd.to_datetime(df[c], errors="coerce")
                    if v.dt.tz is not None:
                        v = v.dt.tz_localize(None)
                    late += int((v >= limit).sum())
        notes = d / "notes.txt"
        if notes.is_file():
            for line in notes.open(encoding="utf-8", errors="replace"):
                if line.startswith("===== NOTE "):
                    parts = line.split(" | ")
                    t = pd.to_datetime(parts[1].strip()[:10], errors="coerce") if len(parts) > 2 else pd.NaT
                    late += int(pd.notna(t) and t >= limit)
    report("every TRAIN and VAL patient has a record folder with meta.json", len(have) == len(ids) and not no_meta,
           f"{len(have)} of {len(ids)} folders, {no_meta} without a readable meta.json")
    report("no record holds anything dated after its cutoff (notes and every date column)", late == 0,
           f"{late} late values over {tables_seen} tables")


def _raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def _score(answers: dict, split: str, gold: dict | None = None):
    import adapters as ad

    if gold is not None:
        return ad.TaskScorer().score_answers(gold, answers, {}, split)
    from engine.schemas import ArtifactRef, RunArtifacts

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": answers}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        return ad.TaskScorer().score(arts, split, {})


def _cards_option_metrics():
    """The eval set's own option scorer, or None when the tb-question-cards checkout is not here."""
    if not (CARDS_SRC / "evalset" / "score.py").is_file():
        return None
    sys.path.insert(0, str(CARDS_SRC))
    try:
        from evalset.score import _option_metrics  # noqa: PLC0415
        return _option_metrics
    except Exception as exc:  # noqa: BLE001 - reported as a skipped comparison, not a pass
        print(f"  note: could not import the eval set's option scorer ({type(exc).__name__})")
        return None
    finally:
        sys.path.remove(str(CARDS_SRC))


def check_scorer(report) -> None:
    """2. The scorer against scikit-learn and the eval set's option scorer, worked out independently."""
    import numpy as np
    import config
    import adapters as ad
    import run
    from sklearn.metrics import roc_auc_score

    rng = random.Random(1)
    # 2a. yes/no AUROC
    worst, sets = 0.0, 0
    for _ in range(300):
        n = rng.randint(5, 200)
        gold = [rng.randint(0, 1) for _ in range(n)]
        ans = [{"q": rng.choice((0.0, 0.05, 0.3, 0.5, 0.9, 1.0)) if rng.random() < 0.7 else rng.random(),
                "error": "x" if rng.random() < 0.1 else None} for _ in range(n)]
        if len(set(gold)) < 2:
            continue
        sets += 1
        ours = ad.auroc([(g, ad.prob(a, "q")) for g, a in zip(gold, ans)])
        theirs = roc_auc_score(gold, [0.5 if a["error"] else a["q"] for a in ans])
        worst = max(worst, abs(ours - theirs))
    report(f"yes/no AUROC == sklearn on {sets} random tie-heavy sets, failures at 0.5", worst < 1e-12,
           f"max |diff| {worst:.1e}")

    # 2b. several-option scores
    codes = [0, 1, 2, 3, 4]
    om = _cards_option_metrics()
    worst_sk, worst_ev, sets, ev_sets = 0.0, 0.0, 0, 0
    for _ in range(300):
        n = rng.randint(6, 120)
        present = rng.sample(codes, rng.randint(2, 5))
        golds = [rng.choice(present) for _ in range(n)]
        probs = []
        for _ in range(n):
            raw = [rng.choice((0, 0, 5, 10, 20, 50, 80)) for _ in codes]
            raw[rng.randrange(5)] += 1
            probs.append({c: v / sum(raw) for c, v in zip(codes, raw)})
        ours = ad.option_scores(golds, probs, codes)
        sk = [roc_auc_score([int(g == c) for g in golds], [p[c] for p in probs]) for c in codes
              if 0 < sum(g == c for g in golds) < n]
        worst_sk = max(worst_sk, abs(ours["auroc"] - sum(sk) / len(sk)))
        sets += 1
        if om is not None:
            P = np.array([[p[c] for c in codes] for p in probs])
            m, per, ba, t2 = om(P, np.ones(n, bool), np.array(golds), codes, np.ones(n))
            per_ok = all((ours["option_auroc"][str(c)] is None) == (not np.isfinite(per[c])) and
                         (ours["option_auroc"][str(c)] is None or abs(ours["option_auroc"][str(c)] - per[c]) < 6e-5)
                         for c in codes)
            worst_ev = max(worst_ev, abs(ours["auroc"] - m), abs(ours["top_pick_balanced_accuracy"] - ba),
                           abs(ours["tumor_board_answer_in_top_two"] - t2), 0.0 if per_ok else 1.0)
            ev_sets += 1
    report(f"several-option AUROC == mean of sklearn's one-vs-rest AUROCs over the answers present, {sets} random sets",
           worst_sk < 1e-12, f"max |diff| {worst_sk:.1e}")
    if om is None:
        report("several-option scores == the eval set's option scorer (skipped: tb-question-cards checkout not found)",
               False, str(CARDS_SRC))
    else:
        report(f"several-option AUROC, per-option AUROC, top-pick BA and top two == the eval set's option scorer, "
               f"{ev_sets} random sets", worst_ev < 1e-4, f"max |diff| {worst_ev:.1e} (4 dp rounding)")

    # 2c. the task score on VAL: mixed informative, flat, random and failed answers
    rows = {r["id"]: r for r in ad.read_split("val")}
    pool = sorted(ad.pools()["val"])
    r2 = random.Random(2)
    answers = {}
    for n, i in enumerate(pool):
        a = {}
        for q in config.QUESTIONS:
            g = rows[i].get(q.key)
            if g is None:
                a[q.key] = None
            elif q.options:
                cs = [c for _k, c, _t in q.options]
                raw = ([5 if c != g else 60 for c in cs] if n % 3 == 1 else [20] * 5 if n % 3 == 0
                       else [r2.randint(0, 9) + 1 for _ in cs])
                a[q.key] = {str(c): v / sum(raw) for c, v in zip(cs, raw)}
            else:
                a[q.key] = (0.7 if g == 1 else 0.2) if n % 3 == 1 else 0.4 if n % 3 == 0 else round(r2.random(), 2)
        answers[i] = {**a, "error": "not_a_probability" if n % 7 == 0 else None, "reason": "", "findings_log": None,
                      "reasons": []}
    score = _score(answers, "val")
    per = {}
    for q in config.QUESTIONS:
        ids = [i for i in pool if rows[i].get(q.key) is not None]
        if q.options:
            cs = [c for _k, c, _t in q.options]
            pr = lambda i, c: 0.2 if answers[i]["error"] else answers[i][q.key][str(c)]  # noqa: E731
            sk = [roc_auc_score([int(rows[i][q.key] == c) for i in ids], [pr(i, c) for i in ids]) for c in cs
                  if 0 < sum(rows[i][q.key] == c for i in ids) < len(ids)]
            per[q.key] = sum(sk) / len(sk)
        else:
            per[q.key] = roc_auc_score([rows[i][q.key] for i in ids],
                                       [0.5 if answers[i]["error"] else answers[i][q.key] for i in ids])
    want = sum(per.values()) / len(per)
    report("task score on VAL == mean over the questions of sklearn's AUROC (failures as no information)",
           abs(score.primary_metric.value - want) < 1e-9, f"{score.primary_metric.value:.4f} vs {want:.4f}; "
           + " ".join(f"{k}={v:.3f}" for k, v in per.items()))
    asked_ok = all(score.breakdown[q.key]["asked"] == sum(rows[i].get(q.key) is not None for i in pool)
                   for q in config.QUESTIONS)
    report("asked counts per question match the VAL split", asked_ok,
           " ".join(f"{q.key} {score.breakdown[q.key]['asked']}" for q in config.QUESTIONS))

    # 2d. the dry run's stand-in: even ids right at 0.8, odd ids no information, "f" ids failed
    stub = {i: {**{k: v for k, v in run.stub_answer(i).items() if k not in ("status", "id")},
                "findings_log": None, "reasons": [], "error": run.stub_answer(i).get("error")} for i in pool}
    for i in stub:
        for q in config.QUESTIONS:
            stub[i].setdefault(q.key, None)
    s = _score(stub, "val")
    exp = {}
    for q in config.QUESTIONS:
        ids = [i for i in pool if rows[i].get(q.key) is not None]
        good = lambda i: i[-1] != "f" and int(i[-1], 16) % 2 == 0  # noqa: E731
        if q.options:
            cs = [c for _k, c, _t in q.options]
            pr = lambda i, c: (0.8 if rows[i][q.key] == c else 0.05) if good(i) else 0.2  # noqa: E731
            sk = [roc_auc_score([int(rows[i][q.key] == c) for i in ids], [pr(i, c) for i in ids]) for c in cs
                  if 0 < sum(rows[i][q.key] == c for i in ids) < len(ids)]
            exp[q.key] = sum(sk) / len(sk)
        else:
            exp[q.key] = roc_auc_score([rows[i][q.key] for i in ids],
                                       [(0.8 if rows[i][q.key] == 1 else 0.2) if good(i) else
                                        (0.5 if i[-1] == "f" else 0.3) for i in ids])
    report("the dry run's stand-in scores what sklearn says it should", abs(s.primary_metric.value
           - sum(exp.values()) / len(exp)) < 1e-9, f"{s.primary_metric.value:.4f}")

    # 2e. per-example right and the mistakes file, on TRAIN patients answered 99% on the first option / yes
    from engine.schemas import ArtifactRef, RunArtifacts

    trows = {r["id"]: r for r in ad.read_split("train")}
    train = sorted(i for i in ad.pools()["train"] if len(ad.asked(trows[i])) > 1)[:2] + sorted(ad.pools()["train"])[:6]

    def first(q):
        if q.options:
            return {str(c): (0.96 if n == 0 else 0.01) for n, (_k, c, _t) in enumerate(q.options)}
        return 0.99
    tans = {i: {**{q.key: (first(q) if trows[i].get(q.key) is not None else None) for q in config.QUESTIONS},
                "error": None, "reason": "x", "reasons": [{"fact": "f", "where": "w"}], "findings_log": None}
            for i in train}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": tans}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        correct = ad.TaskScorer().per_example_correct(arts, "train")
        tscore = ad.TaskScorer().score(arts, "train", {"_iter": 1})
        corpus = ad.mistake_corpus(Path(tmp) / "fb")(arts, tscore)
        entries = json.loads(Path(corpus.ref).read_text())
        n_files = 1 if config.TREATMENT else 4
        copied = all(len(e["input_files"]) == (n_files if e["mistake"] else 0)
                     and all(Path(c).is_file() for c in e["input_files"]) for e in entries)

    def wrong(i, q):
        g = trows[i][q.key]
        return g != q.options[0][1] if q.options else g == 0
    good = all(set(e["wrong_at"]) == {q.key for q in config.QUESTIONS if trows[e["id"]].get(q.key) is not None
                                      and wrong(e["id"], q)}
               and e["mistake"] == bool(e["wrong_at"]) and e["asked"] == ad.asked(trows[e["id"]])
               and all((e["tumor_board_answer"][k] in "ABCDE") if ad.QUESTION[k].options else
                       e["tumor_board_answer"][k] == trows[e["id"]][k] for k in e["asked"])
               for e in entries)
    right_ok = correct == {e["id"]: not e["wrong_at"] for e in entries}
    report("mistakes file: asked questions, the tumor board's answer as a letter, wrong exactly where the first "
           "option / yes is wrong; inputs copied only for mistakes; per-example right agrees",
           good and copied and right_ok and len(entries) == len(train),
           f"{sum(e['mistake'] for e in entries)} of {len(entries)} mistakes; wrong-at logic {good}, inputs copied "
           f"{copied}, per-example right {right_ok}")


def check_output(report, spec) -> None:
    """3. Reading output.json, on patients built by hand."""
    import config
    import adapters as ad
    import agentic

    rows = ad.read_split("val")
    if config.TREATMENT:
        one = next(r for r in rows if ad.asked(r) == ["early_stage_plan"])
        both = next((r for r in rows if len(ad.asked(r)) == 2), None)
        A = {"A": 10, "B": 20, "C": 30, "D": 20, "E": 20}
        cases = [  # (row, body, expected answer for the asked keys or None, error)
            (one, {"early_stage_plan": A}, {"early_stage_plan": {"0": .1, "1": .2, "2": .3, "3": .2, "4": .2}}, None),
            (one, {"early_stage_plan": {"a": 1, "B ": 1, "C": 0, "D": 0, "E": 2}},  # rescaled, case and spaces
             {"early_stage_plan": {"0": .25, "1": .25, "2": 0.0, "3": 0.0, "4": .5}}, None),
            (one, {"early_stage_plan": {**A, "A": 70}}, {"early_stage_plan": {"0": .7 / 1.6, "1": .2 / 1.6,
                                                                               "2": .3 / 1.6, "3": .2 / 1.6, "4": .2 / 1.6}}, None),
            (one, {"early_stage_plan": {"A": 50, "B": 50, "C": 0, "D": 0}}, None, "not_a_probability"),  # missing E
            (one, {"early_stage_plan": dict.fromkeys("ABCDE", 0)}, None, "not_a_probability"),  # all zero
            (one, {"early_stage_plan": {**A, "C": 101}}, None, "not_a_probability"),
            (one, {"early_stage_plan": {**A, "C": "30"}}, None, "not_a_probability"),
            (one, {"early_stage_plan": {**A, "C": True}}, None, "not_a_probability"),
            (one, {"early_stage_plan": {**A, "C": float("nan")}}, None, "not_a_probability"),
            (one, {"early_stage_plan": 40}, None, "not_a_probability"),
            (one, {"first_line_regimen": A}, None, "not_a_probability"),  # answered the question not asked
        ]
        if both is not None:
            cases.append((both, {"early_stage_plan": A}, None, "not_a_probability"))  # one of two asked missing
            cases.append((both, {"early_stage_plan": A, "first_line_regimen": A},
                          {k: {"0": .1, "1": .2, "2": .3, "3": .2, "4": .2} for k in ("early_stage_plan", "first_line_regimen")}, None))
    else:
        one = rows[0]
        cases = [(one, {"progression": 37}, {"progression": 0.37}, None),
                 (one, {"progression": 61.5}, {"progression": 0.615}, None),
                 (one, {"progression": 0}, {"progression": 0.0}, None),
                 (one, {"progression": 101}, None, "not_a_probability"),
                 (one, {"progression": "40"}, None, "not_a_probability"),
                 (one, {"progression": True}, None, "not_a_probability"),
                 (one, {"progression": float("nan")}, None, "not_a_probability"),
                 (one, {}, None, "not_a_probability")]
    r = [{"fact": f"f{k}", "where": "w", "extra": "dropped"} for k in range(7)]
    ok, why = True, ""
    with tempfile.TemporaryDirectory() as tmp:
        for n, (row, body, want, want_err) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps({**body, "reasons": r}))
            m = spec.summarize(tmp, [{"call_id": row["id"], "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(m).read_text())["answers"][row["id"]]

            def close(a, b):
                if isinstance(a, dict) and isinstance(b, dict):
                    return a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)
                return a == b if not isinstance(a, float) else isinstance(b, float) and abs(a - b) < 1e-12
            good = got["error"] == want_err and len(got["reasons"]) == agentic.MAX_REASONS and \
                all(set(x) == {"fact", "where"} for x in got["reasons"])
            if want is not None:
                good &= all(close(got[k], want[k]) for k in want)
            good &= all(got[q.key] is None for q in config.QUESTIONS if q.key not in ad.asked(row))
            ok &= good
            why += "" if good else f" case {n}"
        bad_json = Path(tmp) / "bad.json"
        bad_json.write_text("{not json")
        m = spec.summarize(tmp, [{"call_id": rows[0]["id"], "outputs": []},
                                 {"call_id": rows[1]["id"], "outputs": [{"rel": agentic.OUTPUT, "path": str(bad_json)}]}])
        got = json.loads(Path(m).read_text())["answers"]
        ok &= got[rows[0]["id"]]["error"] == "no_output" and got[rows[1]["id"]]["error"] == "unreadable_output"
    report("output.json: numbers read as 0-100; options rescaled to 100 whatever the total; missing letter, all zero, "
           "out of range, word, bool, NaN, a question asked but not answered, no file, bad JSON = failed; reasons capped",
           ok, f"{len(cases) + 2} cases{why}")
    failed = {q.key: ({"0": 1.0} if q.options else 0.9) for q in config.QUESTIONS} | {"error": "x"}
    flat = all((set(ad.option_probs(failed, q.key).values()) == {1 / len(q.options)}) if q.options else
               ad.prob(failed, q.key) == 0.5 for q in config.QUESTIONS)
    report("a failed answer is no information: 0.5, or an equal share on every option", flat)


def check_pools(report) -> None:
    """4. The pools and the split rule, from the task data, the private id map and the TEST lists' patients."""
    import config
    import adapters as ad
    import pandas as pd

    pools = ad.pools()
    want = SIZES[config.AGENT]
    split_ids = {s: [r["id"] for r in ad.read_split(s)] for s in ("train", "val")}
    report(f"pools: TRAIN {want['train']} and VAL {want['val']}, each split file == its pool",
           all(len(pools[s]) == want[s] and set(split_ids[s]) == pools[s] and len(split_ids[s]) == want[s]
               for s in ("train", "val")), f"{len(pools['train'])} / {len(pools['val'])}")
    id_map = [json.loads(x) for x in (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines() if x]
    by = {s: [m for m in id_map if m["split"] == s] for s in ("train", "val")}
    one_each = all(len({m["person_id"] for m in by[s]}) == len(by[s]) == len(pools[s])
                   and {m["id"] for m in by[s]} == pools[s] for s in ("train", "val"))
    disjoint = not ({m["person_id"] for m in by["train"]} & {m["person_id"] for m in by["val"]})
    report("one row per patient in each pool, and no patient in both TRAIN and VAL", one_each and disjoint)
    test_file = "step6_test.parquet" if config.TREATMENT else "step4_test.parquet"
    test = set(pd.read_parquet(SAMPLES / test_file, columns=["person_id"])["person_id"].astype(int))
    overlap = len(test & {int(m["person_id"]) for m in id_map})
    report(f"none of the TEST list's patients ({test_file}) in TRAIN or VAL", len(test) > 0 and overlap == 0,
           f"{len(test)} TEST patients, {overlap} in the pools")
    report("no TEST answers in the task data yet (TEST sealed until Phil's go)",
           not (config.DATA_DIR / "splits" / "test.jsonl").exists())
    every = all(ad.asked(r) for s in ("train", "val") for r in ad.read_split(s))
    report("every patient is asked at least one question", every)
    if config.TREATMENT:
        have = sum((ad.patient_dir(i) / "summary.txt").is_file() and (ad.patient_dir(i) / "summary.txt").stat().st_size > 50
                   for s in ("train", "val") for i in pools[s])
        report("every pool patient has a non-empty summary", have == sum(want.values()), f"{have} of {sum(want.values())}")
    else:
        yes = {s: sum(r["progression"] for r in ad.read_split(s)) for s in ("train", "val")}
        tot = sum(yes.values())
        n = sum(want.values())
        report("the redraw kept the progression share: VAL within one pair of its share",
               abs(yes["val"] - tot * want["val"] / n) <= 1, f"TRAIN {yes['train']}/{want['train']}, VAL {yes['val']}/{want['val']}")
        vols = sum(all((ad.patient_dir(i) / s / f).is_file() for s in ("prior", "current") for f in ("ct.npz", "meta.json"))
                   for s2 in ("train", "val") for i in pools[s2])
        report("every pair has both converted volumes", vols == n, f"{vols} of {n}")


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
            r1 = [row(1, m, 0.70)]
            ok = stops(r1) is None
            receipt("iter-2-train-dd", 400, 0.10)   # round 2: $40 so far, not yet scored
            ok &= stops(r1 + [row(2, "saved_unmeasured", None)]) is None
            receipt("iter-2-val-ee", 200, 0.10)     # round 2 now $60, scored -> over $50
            ok &= stops(r1 + [row(2, m, 0.72)]) == "round_over_budget"
            ok &= stops(r1 + [row(2, m, 0.72)], rule=(2, 90.0, 0.0, 3.0)) == "run_over_budget"  # $90 + 2 x $3 > $90
            ok &= stops(r1 + [row(2, m, 0.72)], rule=(2, 0.0, 0.0, 3.0)) is None  # all off
            spend = run.run_spend(archive, "r1")
            ok &= {k: round(v, 6) for k, v in spend.items()} == {1: 30.0, 2: 60.0}
            report("dollar stops: round over $50 and run over its limit stop; other runs and unscored rounds don't count",
                   ok, f"spend by round {spend}")
        finally:
            (config.STOP_AFTER_ROUNDS_WITHOUT_BEST, config.RUN_BUDGET_USD, config.ROUND_BUDGET_USD,
             config.OPTIMIZER_SESSION_BUDGET_USD) = saved


def check_wording(report) -> None:
    """6. The quoted question text equals the eval-set cards, and nothing the sessions or the optimizer read says
    "board" without "tumor" (Phil 2026-11-20)."""
    import config

    if (CARDS_SRC / "evalset" / "cards" / "board_questions.py").is_file():
        sys.path.insert(0, str(CARDS_SRC))
        try:
            from evalset.cards import base, board, board_questions as bq  # noqa: PLC0415
            ok = (bq.SUMMARY_OPENING == config.SUMMARY_OPENING and base.ANSWER_OPTIONS == config.ANSWER_OPTIONS
                  and base.ANSWER_PROB == config.ANSWER_PROB
                  and [q.body for q in bq.TREATMENT.questions] == [q.text for q in config.TREATMENT_QUESTIONS]
                  and [q.options for q in bq.TREATMENT.questions] == [q.options for q in config.TREATMENT_QUESTIONS]
                  and [q.task for q in bq.TREATMENT.questions] == [q.task for q in config.TREATMENT_QUESTIONS]
                  and board.CARD.questions[0].body.replace(base.ANSWER, base.ANSWER_PROB)
                  == config.IMAGING_QUESTIONS[0].text)
            report("question text, options, codes and opening == the eval-set cards, word for word", ok)
        finally:
            sys.path.remove(str(CARDS_SRC))
    else:
        report("question text == the eval-set cards (skipped: tb-question-cards checkout not found)", False, str(CARDS_SRC))
    files = [config.SEED_PROGRAM_DIR / "instructions.md", config.TASK_BRIEF, config.OPTIMIZER_INSTRUCTIONS,
             HERE / "agentic.py", HERE / "optimizer_prompt.py"]
    bare = re.compile(r"(?<!tumor )\bboards?\b", re.IGNORECASE)
    hits = {f.name: len(bare.findall(re.sub(r"\s+", " ", f.read_text()))) for f in files}  # "tumor" may end a line
    report('no bare "board" in the seed, the briefs, the wrapper or the optimizer prompt (always "tumor board")',
           not any(hits.values()), str({k: v for k, v in hits.items() if v}))
    seed = (config.SEED_PROGRAM_DIR / "instructions.md").read_text()
    quoted = all(q.text in seed for q in config.QUESTIONS) and (
        config.SUMMARY_OPENING in seed if config.TREATMENT else config.IMAGING_OPENING in seed)
    report("the seed instructions quote the opening and every question word for word", quoted)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    sys.exit(__doc__)
