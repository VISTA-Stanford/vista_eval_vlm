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
                   5. prints one patient's per-call prompt, with placeholder container paths

    cd optimizer/died-2yr-ehr && uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow \\
        python scripts/check_task.py synth /tmp/died2yr-ehr-synth
    TASK_DATA_DIR=/tmp/died2yr-ehr-synth uv run ... python scripts/check_task.py check
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
        want = {p.name for p in src.iterdir()}
        ro = all(not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)) for p in staged.iterdir())
        everything = {p.name for p in Path(tmp).rglob("*")}
        report("staging copies exactly the patient's record files", names == want, f"{len(names)} files")
        report("staged files are read-only", ro)
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

    # 5. the per-call prompt, as a session gets it (placeholder container paths)
    from isolation.program_runner import CallPaths

    print("\n--- per-call prompt ---")
    print(spec.prompt(Call(pid, {"id": pid}), "label", CallPaths(program="/workspace/program", call="/workspace/call", ro={}), "PASS_ID"))
    print("--- end ---\n")
    print("ALL PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "synth" and len(sys.argv) == 3:
        sys.exit(cmd_synth(Path(sys.argv[2])))
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(cmd_check())
    sys.exit(__doc__)
