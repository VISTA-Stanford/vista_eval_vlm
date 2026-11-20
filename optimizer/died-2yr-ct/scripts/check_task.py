"""$0 checks of the died-2yr-ct task code: the scorer, reading a session's output.json, the run's stop rules and
staging. No model call, no Docker. Reads the task data read-only (TASK_DATA_DIR, default the mount) and prints
counts only, never a patient id or row. Copied from died-2yr-ehr's scripts/check_task.py (2026-11-10), minus its
record-staging checks, which are about the EHR export.

    check        1. staging copies exactly ct.npz and meta.json for a VAL-pool patient
                 2. the default (yes/no) answers: balanced accuracy == scikit-learn on random answer sets with
                    failures wrong and on the VAL pool, and output.json read as yes/no exactly
                 With ANSWER=probability, 2 instead checks AUROC and Brier against scikit-learn on random tie-heavy
                 answer sets (failed answers at 0.5) and the VAL pool, the right-side-of-death-rate flag, the
                 calibration bins, the mistakes file, and that output.json is read as a 0-100 number per horizon
                 (anything else, NaN and out of range included, an error) with its reasons kept, capped and cleaned.
                 3. run.py's stops: the no-new-best count and the dollar limits, on hand-built summaries and receipts
                 4. prints the per-call prompt, with placeholder container paths (no id in it)

    cd optimizer/died-2yr-ct && uv run --project $AGENTIC_LABEL_OPT --with scikit-learn python -B scripts/check_task.py check
    ANSWER=probability uv run --project $AGENTIC_LABEL_OPT --with scikit-learn python -B scripts/check_task.py check
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


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

    print(f"ANSWER={config.ANSWER}  metric={config.METRIC_NAME}  seed={config.SEED_PROGRAM_DIR.name}  "
          f"brief={config.TASK_BRIEF.name}  instructions={config.OPTIMIZER_INSTRUCTIONS.name}  "
          f"schedule={config.SCHEDULE}  iterations={config.ITERATIONS}  workdir={config.WORKDIR.name}")
    spec = agentic.program_spec(forbidden=())
    pid = sorted(ad.pools()["val"])[0]

    # 1. staging copies the CT files and nothing else
    with tempfile.TemporaryDirectory() as tmp:
        spec.stage(Call(pid, {"id": pid}), Path(tmp), "p0")
        names = {p.name for p in (Path(tmp) / "ct").iterdir()}
        report("staging copies exactly ct.npz and meta.json", names == {"ct.npz", "meta.json"}
               and {p.name for p in Path(tmp).iterdir()} == {"ct"}, f"{len(names)} files")

    if config.PROBABILITY:
        check_probability(report, spec)
    else:
        check_yes_no(report, spec)
    check_run_rules(report)

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
                "reasons": [{"finding": "f", "where": "right upper lobe"}], "findings_log": None}
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
    r = [{"finding": f"f{k}", "where": "liver", "extra": "dropped"} for k in range(7)]
    cases = [({"died_within_1_year": 37, "died_within_2_years": 61.5, "reasons": r[:2]}, (0.37, 0.615), None, 2),
             ({"died_within_1_year": 0, "died_within_2_years": 100, "reasons": r}, (0.0, 1.0), None, agentic.MAX_REASONS),
             ({"died_within_1_year": 101, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": -1, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": "40", "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": True, "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_1_year": "yes", "died_within_2_years": "no"}, (None, None), "not_a_probability", 0),
             ({"died_within_1_year": float("nan"), "died_within_2_years": 50}, (None, 0.5), "not_a_probability", 0),
             ({"died_within_2_years": 30, "reasons": ["not an object", r[0]]}, (None, 0.3), "not_a_probability", 1)]
    ok, why = True, ""
    with tempfile.TemporaryDirectory() as tmp:
        for n, (body, want_said, want_err, n_reasons) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps(body))
            manifest = spec.summarize(tmp, [{"call_id": "c", "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(manifest).read_text())["answers"]["c"]
            good = ((got["died_within_1_year"], got["died_within_2_years"]) == want_said and got["error"] == want_err
                    and len(got["reasons"]) == n_reasons and all(set(x) == {"finding", "where"} for x in got["reasons"]))
            ok &= good
            why += "" if good else f" case {n}: {got}"
    report("output.json read as 0-100 per horizon; anything else is an error; reasons kept, capped, cleaned", ok,
           f"{len(cases)} cases{why}")
    report("a failed answer scores 0.5 at both horizons",
           ad.prob({"died_within_1_year": 0.9, "error": "x"}, "died_within_1_year") == 0.5)


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
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    sys.exit(__doc__)
