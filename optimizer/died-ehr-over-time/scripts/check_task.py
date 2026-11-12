"""$0 checks for the death-agent-over-time task. Synthetic data only: fake ids (p000000000001...), fake dates (2099),
made-up labels. Nothing here reads a real record.

    python scripts/check_task.py synth /tmp/ot-synth          write a synthetic working copy (60 fake patients)
    TASK_DATA_DIR=/tmp/ot-synth python scripts/check_task.py check
        decision-point rules on fake scans; scorer == scikit-learn; leak checks catch a planted later date;
        prompts and instructions carry no cohort number, result or label; the treatment-start check against the Haiku
        seed equals scikit-learn and falls back as specified; the full dry run (5 points x 2 arms, fake
        sessions through the engine's real pass machinery) answers every row, feeds the memory arm its own earlier
        answers, and scores to the AUROC worked out here from the fake answer rule; the $250 hard stop and the
        first-100-calls cost stop both fire.

Run with the engine's environment plus pandas, pyarrow, duckdb, scikit-learn (see TASK_NOTES.md).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool]] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")


# ==================================================================================================
# synth
# ==================================================================================================


def cmd_synth(out: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(7)
    points = ("start", "restaging", "m6", "m12", "m24")
    days = {"start": 0, "restaging": 60, "m6": 170, "m12": 350, "m24": 720}
    base = dt.date(2099, 1, 1)
    (out / "patients").mkdir(parents=True)
    (out / "splits").mkdir()
    rows, labels, probes = [], [], []
    for i in range(60):
        pat = f"p{i + 1:012d}"
        dead_day = rng.choice([None, None, rng.randint(30, 1500)])
        have = [p for p in points if p == "start" or rng.random() < 0.7]
        for p in have:
            cut = base + dt.timedelta(days=days[p] + i)
            rec = f"r{hashlib.sha256((pat + p).encode()).hexdigest()[:12]}"
            row = f"{pat}-{p}"
            d = out / "patients" / rec
            d.mkdir()
            (d / "meta.json").write_text(json.dumps({"id": rec, "cutoff": cut.isoformat()}))
            (d / "notes.txt").write_text(f"===== NOTE 1 | {cut - dt.timedelta(days=1)} 09:00:00 | Progress Notes =====\n"
                                         "Made-up note.\n\n")
            pq.write_table(pa.table({"measurement_date": [cut - dt.timedelta(days=5), cut],
                                     "value_as_number": [1.0, 2.0]}), d / "measurement.parquet")
            rows.append({"row": row, "patient": pat, "point": p, "record": rec, "cut": cut.isoformat()})
            t = days[p] + i
            l1 = int(dead_day is not None and dead_day - t <= 365 and dead_day > t)
            l2 = None if rng.random() < 0.1 else int(dead_day is not None and dead_day - t <= 730 and dead_day > t)
            labels.append({"row": row, "label_1yr": l1, "label_2yr": l2})
            for name in ("motor", "ct_merlinonc", "clinical_pretx"):
                for field, g in (("label_1yr", l1), ("label_2yr", l2)):
                    if g is not None and rng.random() < 0.95:
                        probes.append({"row": row, "probe": name, "field": field,
                                       "score": min(1.0, max(0.0, 0.3 * g + rng.random() * 0.7))})
    w = lambda p, xs: p.write_text("".join(json.dumps(x) + "\n" for x in xs))  # noqa: E731
    w(out / "rows.jsonl", rows)
    w(out / "labels.jsonl", labels)
    w(out / "splits" / "val.jsonl", [{"id": r["record"]} for r in rows])
    (out / "splits" / "test.jsonl").write_text("")
    (out / "splits" / "train.jsonl").write_text("")
    (out / "pools.json").write_text(json.dumps({"train": [], "val": []}))
    # The Haiku seed's side: a fake died-2yr-ehr id per patient, its answers and labels (for the treatment-start check)
    w(out / "seed_ids.jsonl", [{"patient": f"p{i + 1:012d}", "task_id": f"t{i + 1:012d}"} for i in range(60)])
    seed = {}
    for i in range(60):
        tid = f"t{i + 1:012d}"
        g1, g2 = int(rng.random() < 0.2), None
        g2 = max(g1, int(rng.random() < 0.3))
        seed[tid] = {"gold": {"id": tid, "label_1yr": g1, "label": g2},
                     "answer": {"died_within_1_year": round(0.2 + 0.5 * g1 * rng.random() + 0.3 * rng.random(), 2),
                                "died_within_2_years": round(0.3 + 0.5 * g2 * rng.random() + 0.2 * rng.random(), 2),
                                "error": None}}
    w(out / "seed_val.jsonl", [v["gold"] for v in seed.values()])
    (out / "seed_answers.json").write_text(json.dumps({"answers": {k: v["answer"] for k, v in seed.items()}}))
    pq.write_table(pa.Table.from_pylist(probes), out / "probes.parquet")
    print(f"synthetic copy in {out}: {len({r['patient'] for r in rows})} fake patients, {len(rows)} rows")
    return 0


# ==================================================================================================
# check
# ==================================================================================================


def check_points() -> None:
    import build_points as bp

    s = lambda day, acc=None: {"day": day, "accession_number": acc or f"a{day}"}  # noqa: E731
    got = bp.pick_points([s(-90), s(-20), s(10), s(25), s(40), s(100), s(130), s(170), s(300), s(366), s(640), s(731)])
    ok("treatment start = CT closest to day 0 in -60..+30", got["start"]["day"] == 10)
    ok("first restaging = first CT 28-120 days", got["restaging"]["day"] == 40)
    ok("6 months = latest CT in 92..183", got["m6"]["day"] == 170)
    ok("12 months = latest CT in 274..365", got["m12"]["day"] == 300)
    ok("24 months = latest CT in 639..730", got["m24"]["day"] == 640)
    got = bp.pick_points([s(-61), s(184), s(250)])
    ok("no CT in a window -> no point", got == {}, str(sorted(got)))
    got = bp.pick_points([s(-10), s(10)])
    ok("start tie goes to the earlier CT", got["start"]["day"] == -10)


def check_scorer() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    import score as sc

    rng = random.Random(3)
    for trial in range(5):
        n = rng.randint(40, 300)
        ys = [rng.random() < 0.3 for _ in range(n)]
        ps = [round(min(0.99, max(0.01, 0.25 * y + rng.random() * 0.75)), 2) for y in ys]  # ties on purpose
        pairs = [(int(y), p) for y, p in zip(ys, ps)]
        m = sc.metrics(pairs)
        x = np.array([sc.logit(p) for p in ps]).reshape(-1, 1)
        lr = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(x, np.array(ys, int))
        ok(f"trial {trial}: AUROC == sklearn", abs(m["auroc"] - roc_auc_score(ys, ps)) < 1e-12)
        ok(f"trial {trial}: Brier == sklearn", abs(m["brier"] - brier_score_loss(ys, ps)) < 1e-12)
        ok(f"trial {trial}: calibration slope == sklearn", abs(m["cal_slope"] - lr.coef_[0][0]) < 1e-5,
           f"{m['cal_slope']:.6f} vs {lr.coef_[0][0]:.6f}")
        ok(f"trial {trial}: observed/expected", abs(m["obs_exp"] - sum(ys) / sum(ps)) < 1e-12)


def check_leaks() -> None:
    import program

    row = {"cut": "2099-06-01"}
    clean = [{"date": "2099-01-01", "risk_1yr": 20, "risk_2yr": 30,
              "reasons": [{"fact": "made-up", "file": "notes.txt", "date": "2098-12-01"}]}]
    ok("memory leak check: clean block -> 0", program.memory_violations(row, clean) == 0)
    late = json.loads(json.dumps(clean))
    late[0]["reasons"][0]["date"] = "2099-06-02"
    ok("memory leak check: a reason dated after the cut -> 1", program.memory_violations(row, late) == 1)
    late = json.loads(json.dumps(clean))
    late[0]["reasons"][0]["fact"] = "scan on 2099/07/01 showed made-up"
    ok("memory leak check: a later date inside a reason's text -> 1", program.memory_violations(row, late) == 1)
    # memory_for drops a reason dated after the cut, counts it, and keeps the rest (Phil 2026-11-11)
    rows = program.rows()
    pat = next(r for r in rows.values() if r["point"] == "m6" and f"{r['patient']}-start" in rows)
    prev = rows[f"{pat['patient']}-start"]
    earlier = {"start": {prev["row"]: {"died_within_1_year": 0.2, "died_within_2_years": 0.3, "error": None, "reasons": [
        {"fact": "kept made-up fact", "file": "notes.txt", "date": prev["cut"]},
        {"fact": "made-up fact", "file": "notes.txt", "date": "2199-01-01"},
        {"fact": "follow-up planned 2199/02/01", "file": "notes.txt", "date": prev["cut"]}]}}}
    mem, gone = program.memory_for(pat, earlier)
    ok("memory_for drops the 2 reasons dated after the cut, keeps 1, block then clean",
       gone == 2 and len(mem[0]["reasons"]) == 1 and program.memory_violations(pat, mem) == 0, f"{gone} dropped")
    cols = program.registry_text_date_columns()
    ok("registry text date columns read off the allow-list", "dateofdiagnosis" in cols.get("registry_diagnoses", ())
       and "eventdate" in cols.get("registry_events", ()), str({k: len(v) for k, v in cols.items()}))
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "meta.json").write_text(json.dumps({"id": "r0", "cutoff": "2099-06-01"}))
        (d / "notes.txt").write_text("===== NOTE 1 | 2099-05-01 09:00:00 | x =====\nmade-up\n\n")
        pq.write_table(pa.table({"visit_start_date": [dt.date(2099, 5, 1)]}), d / "visit_occurrence.parquet")
        ok("record leak check: clean record -> 0", program.record_violations(d, "2099-06-01") == {})
        pq.write_table(pa.table({"visit_start_date": [dt.date(2099, 6, 2)]}), d / "visit_occurrence.parquet")
        with (d / "notes.txt").open("a") as fh:
            fh.write("===== NOTE 2 | 2099-06-03 09:00:00 | x =====\nmade-up\n\n")
        v = program.record_violations(d, "2099-06-01")
        ok("record leak check: a later row and a later note are caught", v == {"date_after_cut": 1, "note_after_cut": 1},
           str(v))
        pq.write_table(pa.table({"visit_start_date": [dt.date(2099, 5, 1)]}), d / "visit_occurrence.parquet")
        (d / "notes.txt").write_text("===== NOTE 1 | 2099-05-01 09:00:00 | x =====\nmade-up\n\n")
        pq.write_table(pa.table({"row": [0], "dateofdiagnosis": ["20990501"], "primarysite": ["C341"]}),
                       d / "registry_diagnoses.parquet")
        pq.write_table(pa.table({"row": [0, 1], "eventdate": ["20990520", "209905"]}), d / "registry_events.parquet")
        ok("record leak check: registry text dates on or before the cut -> 0",
           program.record_violations(d, "2099-06-01") == {})
        pq.write_table(pa.table({"row": [0], "dateofdiagnosis": ["20990602"], "primarysite": ["C341"]}),
                       d / "registry_diagnoses.parquet")
        pq.write_table(pa.table({"row": [0, 1], "eventdate": ["20990520", "209907"]}), d / "registry_events.parquet")
        v = program.record_violations(d, "2099-06-01")
        ok("record leak check: a post-cut text date in a registry column is caught (YYYYMMDD and YYYYMM)",
           v == {"registry_text_date_after_cut": 2}, str(v))


BANNED = re.compile(r"auroc|brier|probe|motor|merlin|stacked|k16|k29|validation|calibrat|label|readback|result",
                    re.IGNORECASE)


def check_prompts() -> None:
    import common as c
    import program
    from isolation.pass_outputs import Call
    from isolation.program_runner import CallPaths

    paths = CallPaths(program="/workspace/program", call="/workspace/call", ro={})
    allowed = {"0", "1", "2", "3", "5", "100"}  # 0 to 100, 1 and 2 years, "3 to 5 facts": no cohort number at all
    for arm in program.ARMS:
        text = (program.PROGRAM_DIRS[arm] / "instructions.md").read_text()
        ok(f"{arm} instructions: no cohort numbers or results", set(re.findall(r"\d+(?:\.\d+)?", text)) <= {"0", "1", "2", "100"}
           and not BANNED.search(text))
        for point in c.POINT_KEYS:
            mem = [{"date": "2099-01-01", "risk_1yr": 11, "risk_2yr": 22,
                    "reasons": [{"fact": "made-up", "file": "notes.txt", "date": "2098-12-31"}]}] if arm == "memory" else []
            call = Call(f"p000000000001-{point}", {"record": "r0", "point": point, "arm": arm, "memory": mem})
            p = program.prompt_text(call, paths, "PASSID")
            body = p.replace(program.memory_text(mem), "")
            nums = set(re.findall(r"\d+(?:\.\d+)?", body))
            ok(f"{arm} {point} prompt: no cohort number (only 0-100, 1/2 years, 3-5 facts)", nums <= allowed and
               "%" not in body and "died within" not in body, str(nums - allowed))
            ok(f"{arm} {point} prompt: no result words", not BANNED.search(body), str(BANNED.findall(body)))
            if arm == "memory":
                ok(f"memory {point} prompt carries the earlier answer", "within 1 year 11, within 2 years 22" in p)


def check_gate() -> None:
    import common as c
    import program
    import score as sc
    from sklearn.metrics import roc_auc_score

    d = c.DATA_DIR
    seed, gold, note = sc.load_seed(d / "seed_answers.json", d / "seed_val.jsonl")
    ok("seed answers that do not re-score to the readback are refused", seed is None and "not used" in note, note)
    old = sc.SEED_VAL200_AUROC
    sc.SEED_VAL200_AUROC = sc.mean_auroc(json.loads((d / "seed_answers.json").read_text())["answers"],
                                         {r["id"]: r for r in map(json.loads, (d / "seed_val.jsonl").read_text().splitlines())},
                                         [f"t{i + 1:012d}" for i in range(60)], ("label_1yr", "label"))
    seed, gold, note = sc.load_seed(d / "seed_answers.json", d / "seed_val.jsonl")
    ok("seed answers that re-score to the readback are used", seed is not None, note)
    rows, labels = program.rows(), program.labels()
    start = [r for r, x in rows.items() if x["point"] == "start"]
    rng = random.Random(5)
    fresh = {r: {"died_within_1_year": rng.random(), "died_within_2_years": rng.random(), "error": None} for r in start}
    ids = {r["patient"]: r["task_id"] for r in map(json.loads, (d / "seed_ids.jsonl").read_text().splitlines())}
    g = sc.start_gate(fresh, rows, labels, seed, gold, ids)

    def sk(ans, gl, keys, fields):
        out = []
        for (key, _f, _w), f in zip(c.HORIZONS, fields):
            ys = [gl[i][f] for i in keys if gl[i].get(f) is not None]
            ps = [ans[i][key] for i in keys if gl[i].get(f) is not None]
            out.append(roc_auc_score(ys, ps))
        return sum(out) / 2
    want_ours = sk(fresh, labels, start, ("label_1yr", "label_2yr"))
    want_ref = sk(seed, gold, [ids[rows[r]["patient"]] for r in start], ("label_1yr", "label"))
    ok("treatment-start check on overlapping patients == sklearn", g["overlap"] == len(start) >= 40
       and abs(g["fresh_start_auroc"] - want_ours) < 1e-12 and abs(g["seed_auroc"] - want_ref) < 1e-12
       and g["pass"] == (want_ours >= want_ref - 0.05) and "both tasks hold" in g["compared_on"], g["compared_on"])
    few = dict(list(ids.items())[:39])
    g = sc.start_gate(fresh, rows, labels, seed, gold, few)
    ok("treatment-start check under 40 overlapping falls back to the VAL-200 AUROC and says so",
       g["seed_auroc"] == sc.SEED_VAL200_AUROC and "VAL-200" in g["compared_on"], g["compared_on"])
    sc.SEED_VAL200_AUROC = old


NO_SEED = {"OVER_TIME_SEED_ANSWERS": "/nonexistent", "OVER_TIME_SEED_VAL200_AUROC": "0.5"}


def run_cmd(*args: str, env: dict | None = None) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-B", str(HERE / "run_over_time.py"), *args], capture_output=True, text=True,
                          env={**os.environ, **NO_SEED, **(env or {})}, cwd=HERE)
    return proc.returncode, proc.stdout + proc.stderr


def check_dry_run() -> None:
    import common as c
    import fake_sessions
    import program
    from sklearn.metrics import roc_auc_score

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        code, out = run_cmd("run", "--run-id", "dry", "--dry-run", env={"TASK_WORKDIR": str(work)})
        ok("dry run: all 5 points x 2 arms finish", code == 0 and "FINISHED" in out, out[-600:] if code else "")
        if code:
            return
        for point in c.POINT_KEYS:
            m = re.search(rf"\[{point}\] record leak check: (\d+) records, (\d+) violations", out)
            ok(f"dry run {point}: record leak check printed, 0 violations", bool(m) and m.group(2) == "0")
            if point != "start":
                m = re.search(rf"\[{point}-memory\] memory leak check: (\d+) blocks, (\d+) with earlier answers, "
                              rf"(\d+) earlier reasons dropped for a date after the cut, (\d+) violations", out)
                ok(f"dry run {point}: memory leak check printed, dropped count, 0 violations", bool(m) and m.group(4) == "0")
        ok("dry run: treatment-start check printed and passed", "treatment-start check:" in out and "-> PASS" in out)
        ans = {f.stem: json.loads(f.read_text())["answers"] for f in (work / "private" / "runs" / "dry" / "answers").glob("*.json")}
        labels = program.labels()
        ok("dry run: every row answered in both arms", all(set(ans[f"{p}-{a}"]) == {r["row"] for r in program.point_rows(p)}
                                                          for p in c.POINT_KEYS for a in program.ARMS))
        # the memory arm saw its own earlier answer: its 1-year risk is the last answered earlier one + 1
        good = bad = 0
        for p in c.POINT_KEYS[1:]:
            for r in program.point_rows(p):
                a = ans[f"{p}-memory"][r["row"]]
                prior = [ans[f"{q}-memory"].get(f"{r['patient']}-{q}") for q in c.POINT_KEYS[: c.POINT_KEYS.index(p)]]
                prior = [x for x in prior if x and not x.get("error")]
                if a.get("error") or not prior:
                    continue
                if abs(a["died_within_1_year"] - (min(100, round(100 * prior[-1]["died_within_1_year"]) + 1)) / 100) < 1e-9:
                    good += 1
                else:
                    bad += 1
        ok("dry run: memory arm prompts carried its own earlier risks", good > 20 and bad == 0, f"{good} right, {bad} wrong")
        code, out = run_cmd("score", "--run-id", "dry", "--n-boot", "200", "--out", str(Path(tmp) / "score.json"),
                            env={"TASK_WORKDIR": str(work)})
        ok("dry run: score runs", code == 0, out[-400:] if code else "")
        table = json.loads((Path(tmp) / "score.json").read_text())
        # the fresh arm's AUROC from the fake answer rule, worked out here with sklearn
        for p in c.POINT_KEYS:
            for key, field, words in c.HORIZONS:
                ys, ps = [], []
                for r in program.point_rows(p):
                    g = labels[r["row"]][field]
                    if g is None:
                        continue
                    k = fake_sessions.kind(r["row"])
                    ps.append(0.5 if k == 9 else ((0.8 if g else 0.2) if k <= 4 else 0.3))
                    ys.append(g)
                got = next(t for t in table["table"] if t["point"] == p and t["horizon"] == words
                           and t["model"] == "fresh" and t["rows"] == "all labelled")
                want = roc_auc_score(ys, ps) if len(set(ys)) == 2 else 0.5
                ok(f"dry run {p} {words}: fresh AUROC == the fake rule's", abs(got["auroc"] - want) < 1e-12,
                   f"{got['auroc']:.4f} vs {want:.4f}")
        ch = [d for d in table["risk_change"] if d["arm"] == "memory" and d["horizon"] == "1 year"]
        ok("dry run: memory arm's median 1-year change is 1 point (the fake +1 rule)",
           ch and all(abs(d["median_abs_change"] - 1.0) < 1e-9 for d in ch), str([d["median_abs_change"] for d in ch]))
        stacked = [t for t in table["table"] if t["model"] == "stacked"]
        ok("score table says the stacked probe has no VAL score", stacked and all(t.get("note") for t in stacked))
        ok("score table holds the probes on common rows",
           any(t["model"] == "motor" and t["rows"] == "common" and t.get("n") for t in table["table"]))
        # the treatment-start stop: a seed far above the fake arm stops before any later point; the override goes on
        w4 = str(Path(tmp) / "w4")
        code, out = run_cmd("run", "--run-id", "dry-gate", "--dry-run",
                            env={"TASK_WORKDIR": w4, "OVER_TIME_SEED_VAL200_AUROC": "0.99"})
        ok("treatment-start check: a seed 0.99 stops before the first restaging pass",
           code == 3 and "-> STOP" in out and "[restaging-fresh]" not in out,
           out[-300:])
        code, out = run_cmd("run", "--run-id", "dry-gate", "--dry-run",
                            env={"TASK_WORKDIR": w4, "OVER_TIME_SEED_VAL200_AUROC": "0.99", "OVER_TIME_PAST_START_GATE": "1"})
        ok("treatment-start check: OVER_TIME_PAST_START_GATE=1 resumes and finishes", code == 0 and "FINISHED" in out
           and "going on because" in out, out[-300:])
        # dollar stops
        code, out = run_cmd("run", "--run-id", "dry-cap", "--dry-run",
                            env={"TASK_WORKDIR": str(Path(tmp) / "w2"), "RUN_BUDGET_USD": "16", "FAKE_COST_USD": "0.05"})
        spent = re.search(r"spent \$([\d.]+)", out)
        ok("hard stop: a $16 cap stops the run before a pass could take it past $16", code == 3 and "hard stop" in out
           and spent and 16 - 25 * 0.5 < float(spent.group(1)) <= 16, out[-300:])
        code, out = run_cmd("run", "--run-id", "dry-dear", "--dry-run",
                            env={"TASK_WORKDIR": str(Path(tmp) / "w3"), "FAKE_COST_USD": "0.2"})
        ok("cost stop: $0.20 a call stops after the first 100 calls", code == 3 and "over $0.10" in out
           and re.search(r"first (\d+) calls", out) and int(re.search(r"first (\d+) calls", out).group(1)) <= 125,
           out[-300:])


def cmd_check() -> int:
    sys.path.insert(0, str(HERE))
    import common as c  # noqa: F401  (sets the died-2yr-ehr settings first)

    print("decision points"); check_points()
    print("scorer vs scikit-learn"); check_scorer()
    print("leak checks"); check_leaks()
    print("prompts and instructions"); check_prompts()
    print("treatment-start check"); check_gate()
    print("dry run (fake sessions)"); check_dry_run()
    bad = [n for n, good in RESULTS if not good]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks PASS" + (f"; FAIL: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "synth":
        sys.exit(cmd_synth(Path(sys.argv[2])))
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    print(__doc__)
    sys.exit(2)
