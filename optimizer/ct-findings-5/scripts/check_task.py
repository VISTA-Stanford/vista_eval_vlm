"""$0 checks of the ct-findings-5 task code and data: the scorer, reading a session's output.json, the pools, the run's
stop rules and staging. No model call, no Docker. Reads the task data read-only (TASK_DATA_DIR, default the mount),
the private id map and the VLMs' TEST sample's patient column, and prints counts only, never an id or a row. Built
from died-2yr-ct's probability checks (2026-11-10).

    check   1. staging copies exactly ct.npz and meta.json for a VAL-pool scan
            2. AUROC per question == scikit-learn on random tie-heavy answer sets with failures at 0.5 and unscored
               labels left out; the task score on the VAL pool == the mean of sklearn's five; Brier == sklearn's;
               ties counted; calibration bins; per-example right ignores unscored questions; the mistakes file
            3. output.json read as 0-100 per question; out of range, a word, a bool, NaN or a missing key = failed,
               scored 0.5 on every question; reasons kept, capped and cleaned
            4. the pools: VAL 200 and TRAIN 400, splits == pools, one scan per patient, no patient in both, none of
               the VLMs' TEST-sample patients in TRAIN or VAL, every question >= 20 positives and >= 20 negatives on
               VAL-200 (TRAIN: >= 40 of 400)
            5. run.py's stops: the no-new-best count and the dollar limits, on hand-built summaries and receipts
            6. prints the per-call prompt, with placeholder container paths (no id in it)

    cd optimizer/ct-findings-5 && uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow python -B scripts/check_task.py check
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

TEST_SAMPLE = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/samples/step2.parquet")
FLOOR = 20  # spec: every question >= 20 positives on VAL-200


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

    print(f"task={config.TASK_NAME}  metric={config.METRIC_NAME}  questions={len(config.QUESTIONS)}  "
          f"schedule={config.SCHEDULE}  iterations={config.ITERATIONS}  stop after {config.STOP_AFTER_ROUNDS_WITHOUT_BEST} "
          f"rounds without best, ${config.ROUND_BUDGET_USD:.0f}/round, ${config.RUN_BUDGET_USD:.0f}/run, "
          f"${config.PROGRAM_CALL_BUDGET_USD:.2f}/scan, {config.PROGRAM_MAX_CONCURRENT} at a time  workdir={config.WORKDIR.name}")
    spec = agentic.program_spec(forbidden=())

    # 1. staging copies the CT files and nothing else (a VAL-pool scan whose volume is converted)
    pid = next(i for i in sorted(ad.pools()["val"]) if (ad.patient_dir(i) / "meta.json").is_file())
    with tempfile.TemporaryDirectory() as tmp:
        spec.stage(Call(pid, {"id": pid}), Path(tmp), "p0")
        names = {p.name for p in (Path(tmp) / "ct").iterdir()}
        report("staging copies exactly ct.npz and meta.json", names == {"ct.npz", "meta.json"}
               and {p.name for p in Path(tmp).iterdir()} == {"ct"}, f"{len(names)} files")

    check_scorer(report)
    check_output(report, spec)
    check_pools(report)
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


def check_scorer(report) -> None:
    """2. The scorer against scikit-learn, worked out independently: sklearn sees failures as 0.5 and never sees an
    unscored scan."""
    import config
    import adapters as ad
    from sklearn.metrics import brier_score_loss, roc_auc_score

    keys = [k for k, _t, _q in config.QUESTIONS]
    # 2a. one question, random tie-heavy sets with failures and unscored labels
    rng = random.Random(1)
    worst, sets = 0.0, 0
    for _ in range(300):
        n = rng.randint(5, 300)
        gold = [rng.choice((0, 1, None)) if rng.random() < 0.3 else rng.randint(0, 1) for _ in range(n)]
        ans = [{"q": rng.choice((0.0, 0.05, 0.3, 0.5, 0.9, 1.0)) if rng.random() < 0.7 else rng.random(),
                "error": "x" if rng.random() < 0.1 else None} for _ in range(n)]
        y = [g for g in gold if g is not None]
        if len(set(y)) < 2:
            continue
        sets += 1
        ours = ad.auroc([(g, ad.prob(a, "q")) for g, a in zip(gold, ans) if g is not None])
        theirs = roc_auc_score(y, [0.5 if a["error"] else a["q"] for g, a in zip(gold, ans) if g is not None])
        worst = max(worst, abs(ours - theirs))
    report(f"AUROC == sklearn on {sets} random tie-heavy sets, failures at 0.5, unscored left out", worst < 1e-12,
           f"max |diff| {worst:.1e}")
    report("AUROC is 0.5 when one class is missing", ad.auroc([(1, 0.2), (1, 0.9)]) == 0.5)

    # 2b. the task score on the VAL pool: mixed informative, constant, random and failed answers
    pool = sorted(ad.pools()["val"])
    rows = {r["id"]: r for r in ad.read_split("val")}
    r2 = random.Random(2)
    answers = {}
    for n, i in enumerate(pool):
        a = {k: (0.7 if rows[i][k] == 1 else 0.2) if n % 3 == 1 else 0.4 if n % 3 == 0 else round(r2.random(), 2)
             for k in keys}
        answers[i] = {**a, "error": "not_a_probability" if n % 7 == 0 else None, "reason": "", "findings_log": None,
                      "reasons": []}
    score, correct = _score_file(answers, "val")

    def sk(k):
        ids = [i for i in pool if rows[i][k] is not None]
        return [rows[i][k] for i in ids], [0.5 if answers[i]["error"] else answers[i][k] for i in ids]

    per = {k: roc_auc_score(*sk(k)) for k in keys}
    want = sum(per.values()) / len(keys)
    report("task score on the VAL pool == mean of sklearn's AUROC over the five questions",
           abs(score.primary_metric.value - want) < 1e-9, f"{score.primary_metric.value:.4f} vs {want:.4f}")
    report("per-question AUROC in the breakdown == sklearn's (4 dp)",
           all(score.breakdown[k]["auroc"] == round(per[k], 4) for k in keys),
           " ".join(f"{k}={per[k]:.3f}" for k in keys))
    briers = [(score.breakdown[k]["brier"], round(brier_score_loss(*sk(k)), 4)) for k in keys]
    report("Brier per question == sklearn's", all(a == b for a, b in briers))
    counts_ok = all(score.breakdown[k]["yes"] + score.breakdown[k]["no"] == len(sk(k)[0])
                    and score.breakdown[k]["unscored"] == sum(rows[i][k] is None for i in pool) for k in keys)
    report("yes/no/unscored counts per question match the VAL split", counts_ok)
    k0 = keys[0]
    vals = sk(k0)[1]
    b = score.breakdown[k0]
    report("breakdown counts ties: distinct values and share on the most common value",
           b["distinct_values"] == len(set(vals))
           and abs(b["share_on_most_common_value"] - max(vals.count(v) for v in set(vals)) / len(vals)) < 1e-3)
    rates = ad.train_rates()
    want_right = {i: None if answers[i]["error"] else all((answers[i][k] >= rates[k]) == (rows[i][k] == 1)
                                                          for k in keys if rows[i][k] is not None) for i in pool}
    report("per-example right = correct side of each scored question's TRAIN rate (unscored ignored), failed = None",
           correct == want_right, f"{sum(v is True for v in correct.values())} right of {len(pool)}")

    # 2c. calibration bins (tenths, p = 1.0 in the top bin), on hand-built scans with one unscored
    ten = [f"hand{k:02d}" for k in range(10)]
    p = [0.0, 0.05, 0.1, 0.35, 0.5, 0.99, 1.0, 1.0, 0.3, 0.2]
    hgold = {i: {k: (k2 % 2 if k2 != 9 else None) for k in keys} for k2, i in enumerate(ten)}
    hand = {i: {**{k: v for k in keys}, "error": None, "reason": "", "reasons": [], "findings_log": None}
            for i, v in zip(ten, p)}
    hs = ad.TaskScorer().score_answers(hgold, hand, {}, "val")
    bins = hs.breakdown[k0]["calibration_by_decile"]
    want_bins = {"p0.0:n": 2, "p0.1:n": 1, "p0.3:n": 2, "p0.5:n": 1, "p0.9:n": 3}  # the 0.2 scan is unscored
    report("calibration bins: tenths, p = 1.0 in the top bin, unscored scans left out",
           {k: v for k, v in bins.items() if k.endswith(":n")} == want_bins and hs.breakdown[k0]["unscored"] == 1,
           str({k: v for k, v in bins.items() if k.endswith(':n')}))

    # 2d. the mistakes file: probabilities and reasons per TRAIN scan; unscored questions never count as wrong
    from engine.schemas import ArtifactRef, RunArtifacts

    trows = {r["id"]: r for r in ad.read_split("train")}
    train = sorted(i for i in ad.pools()["train"] if any(trows[i][k] is None for k in keys))[:3] + \
        sorted(i for i in ad.pools()["train"] if all(trows[i][k] is not None for k in keys))[:3]
    tans = {i: {**{k: 0.99 for k in keys}, "error": None, "reason": "x",
                "reasons": [{"finding": "f", "where": "liver"}], "findings_log": None} for i in train}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "answers.json"
        f.write_text(json.dumps({"answers": tans}))
        arts = RunArtifacts(batch_id="b", status="ok", artifact_refs=(ArtifactRef(path=str(f), sha256="0" * 64),))
        tscore = ad.TaskScorer().score(arts, "train", {"_iter": 1})
        corpus = ad.mistake_corpus(Path(tmp) / "fb")(arts, tscore)
        entries = json.loads(Path(corpus.ref).read_text())
        copied = all((Path(c).is_file() for c in e["ct_files"]) for e in entries)
    good = all(set(e["wrong_at"]) == {k for k in keys if trows[e["id"]][k] == 0}
               and e["mistake"] == bool(e["wrong_at"]) and e["reasons"]
               and all(e[f"probability_{k}"] == 0.99 and e[f"gold_{k}"] == trows[e["id"]][k] for k in keys)
               and (len(e["ct_files"]) == 2) == e["mistake"] for e in entries)
    report("mistakes file: gold (null when unscored), probabilities, reasons; at 99% wrong exactly on the no questions",
           good and copied and len(entries) == len(train), f"{sum(e['mistake'] for e in entries)} of {len(entries)} mistakes")


def check_output(report, spec) -> None:
    """3. Reading output.json."""
    import config
    import adapters as ad
    import agentic

    keys = [k for k, _t, _q in config.QUESTIONS]
    r = [{"finding": f"f{k}", "where": "liver", "extra": "dropped"} for k in range(7)]
    full = dict.fromkeys(keys, 50)

    def body(**over):
        return {**full, **over}

    cases = [(body(lesion_liver=37, pneumonitis=61.5, reasons=r[:2]), {"lesion_liver": 0.37, "pneumonitis": 0.615}, None, 2),
             (body(lesion_liver=0, lesion_lungs=100, reasons=r), {"lesion_liver": 0.0, "lesion_lungs": 1.0}, None, agentic.MAX_REASONS),
             (body(lung_lobe_rul=101), {"lung_lobe_rul": None}, "not_a_probability", 0),
             (body(lung_lobe_rul=-1), {"lung_lobe_rul": None}, "not_a_probability", 0),
             (body(lesion_lymph_nodes="40"), {"lesion_lymph_nodes": None}, "not_a_probability", 0),
             (body(pneumonitis=True), {"pneumonitis": None}, "not_a_probability", 0),
             (body(lesion_liver="yes"), {"lesion_liver": None}, "not_a_probability", 0),
             (body(lesion_lungs=float("nan")), {"lesion_lungs": None}, "not_a_probability", 0),
             ({k: 20 for k in keys[:4]} | {"reasons": ["not an object", r[0]]}, {"pneumonitis": None}, "not_a_probability", 1)]
    ok, why = True, ""
    with tempfile.TemporaryDirectory() as tmp:
        for n, (b, want, want_err, n_reasons) in enumerate(cases):
            out = Path(tmp) / f"o{n}.json"
            out.write_text(json.dumps(b))
            manifest = spec.summarize(tmp, [{"call_id": "c", "outputs": [{"rel": agentic.OUTPUT, "path": str(out)}]}])
            got = json.loads(Path(manifest).read_text())["answers"]["c"]
            expect = {k: (want[k] if k in want else (b[k] / 100 if isinstance(b.get(k), int) else None)) for k in keys}
            good = ({k: got[k] for k in keys} == expect and got["error"] == want_err and len(got["reasons"]) == n_reasons
                    and all(set(x) == {"finding", "where"} for x in got["reasons"]))
            ok &= good
            why += "" if good else f" case {n}"
        # a session that wrote nothing, and one whose output is not JSON
        bad_json = Path(tmp) / "bad.json"
        bad_json.write_text("{not json")
        m = spec.summarize(tmp, [{"call_id": "a", "outputs": []},
                                 {"call_id": "b", "outputs": [{"rel": agentic.OUTPUT, "path": str(bad_json)}]}])
        got = json.loads(Path(m).read_text())["answers"]
        ok &= got["a"]["error"] == "no_output" and got["b"]["error"] == "unreadable_output"
    report("output.json read as 0-100 per question; out of range, word, bool, NaN, missing key, no file, bad JSON = "
           "failed; reasons kept, capped, cleaned", ok, f"{len(cases) + 2} cases{why}")
    failed = {**dict.fromkeys(keys, 0.9), "error": "x"}
    report("a failed answer scores 0.5 on every question", all(ad.prob(failed, k) == 0.5 for k in keys))


def check_pools(report) -> None:
    """4. The pools and the split rule, from the task data, the private id map and the TEST sample's patients."""
    import config
    import adapters as ad
    import pandas as pd

    keys = [k for k, _t, _q in config.QUESTIONS]
    pools = ad.pools()
    split_ids = {s: [r["id"] for r in ad.read_split(s)] for s in ("train", "val")}
    report("pools: VAL 200 and TRAIN 400 scans, each split file == its pool",
           len(pools["val"]) == 200 and len(pools["train"]) == 400
           and all(set(split_ids[s]) == pools[s] and len(split_ids[s]) == len(pools[s]) for s in ("train", "val")))
    id_map = [json.loads(x) for x in (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines() if x]
    by_split = {s: [m for m in id_map if m["split"] == s] for s in ("train", "val")}
    one_each = all(len({m["person_id"] for m in by_split[s]}) == len(by_split[s]) == len(pools[s])
                   and {m["id"] for m in by_split[s]} == pools[s] for s in ("train", "val"))
    disjoint = not ({m["person_id"] for m in by_split["train"]} & {m["person_id"] for m in by_split["val"]})
    report("one scan per patient in each pool, and no patient in both TRAIN and VAL", one_each and disjoint)
    test_patients = set(pd.read_parquet(TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    overlap = len(test_patients & {int(m["person_id"]) for m in id_map})
    report("none of the VLMs' TEST-sample patients in TRAIN or VAL", len(test_patients) > 0 and overlap == 0,
           f"{len(test_patients)} TEST-sample patients, {overlap} in the pools")
    no_test_file = not (config.DATA_DIR / "splits" / "test.jsonl").exists()
    report("no TEST labels in the task data (TEST sealed)", no_test_file)
    vols = sum((ad.patient_dir(i) / "meta.json").is_file() and (ad.patient_dir(i) / "ct.npz").is_file()
               for s in ("train", "val") for i in pools[s])
    report("every pool scan has its converted volume", vols == 600, f"{vols} of 600")
    for s, floor in (("val", FLOOR), ("train", 2 * FLOOR)):
        rows = ad.read_split(s)
        c = {k: (sum(r[k] == 1 for r in rows), sum(r[k] == 0 for r in rows)) for k in keys}
        report(f"{s.upper()}: every question >= {floor} positives and >= {floor} negatives",
               all(p >= floor and n >= floor for p, n in c.values()),
               " ".join(f"{k} {p}/{n}" for k, (p, n) in c.items()))


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


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    sys.exit(__doc__)
