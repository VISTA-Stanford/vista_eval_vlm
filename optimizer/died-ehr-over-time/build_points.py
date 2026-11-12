"""Build the patient list, the decision-point table and the record export for the death-agent-over-time task.

    table      the patients (validation, first treatment IV chemotherapy or an oral targeted drug, K29 subsetting),
               their decision points (common.POINTS rules on K29's chest-containing CTs), the K16 per-CT death
               labels at 1 and 2 years for each point's CT (checked against every other K16 file holding that CT),
               the K16 probe scores on the same rows, the died-2yr-ehr task ids of the patients it shares (for the
               treatment-start check against the Haiku seed), and the exporter's cohort: one record per patient and
               cut day (two points on the same CT share a record). A tiny BigQuery read finds the index tumour of
               patients the died-2yr-ehr cohort does not hold. Writes PRIVATE_DIR; prints counts only.
    export     run the shared exporter (vista-onc-labelers common/omop_export, several cuts per patient) into the
               local copy and archive it to PRIVATE_DIR/archive. The real export is a PHI write: it refuses without
               --go. --dry-run prices the BigQuery queries and downloads nothing.
    local      copy rows.jsonl, labels.jsonl, probes.parquet and seed_ids.jsonl from PRIVATE_DIR into DATA_DIR (the
               run VM's working copy; died-2yr-ehr's build_data.py rehydrate rebuilds the records beside them).

    uv run --project $AGENTIC_LABEL_OPT --with pandas --with pyarrow --with google-cloud-bigquery python build_points.py table
    uv run --project $AGENTIC_LABEL_OPT python build_points.py export --dry-run          (then, with Phil's go: --go)

Prints counts only: never an id, a date or a row.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import common as c

SCRIPTS = c.DIED_2YR_EHR / "scripts"


def _build_data():
    """died-2yr-ehr's build_data module (its index-tumour SQL, exporter path and allow-list)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import build_data

    return build_data


# ==================================================================================================
# Decision points (pure: unit-tested on fake scans in scripts/check_task.py)
# ==================================================================================================


def pick_points(scans: list[dict]) -> dict[str, dict]:
    """One patient's decision-point CTs from their chest-containing scans ({"accession_number", "day", ...}; day =
    days from first treatment). Returns {point key: scan} for the points the patient has."""
    out: dict[str, dict] = {}
    lo, hi = c.START_WINDOW
    start = [s for s in scans if lo <= s["day"] <= hi]
    if start:
        out["start"] = min(start, key=lambda s: (abs(s["day"]), s["day"], s["accession_number"]))
    lo, hi = c.RESTAGING_WINDOW
    rest = [s for s in scans if lo <= s["day"] <= hi]
    if rest:
        out["restaging"] = min(rest, key=lambda s: (s["day"], s["accession_number"]))
    for key, mark in c.MONTH_MARKS.items():
        win = [s for s in scans if mark - c.MONTH_LOOKBACK <= s["day"] <= mark]
        if win:
            out[key] = max(win, key=lambda s: (s["day"], s["accession_number"]))
    return out


def point_rows(scans, labels: dict) -> list[dict]:
    """Decision-point rows for a set of patients: scans is a DataFrame of K29 chest-containing scans
    (person_id, accession_number, scan_date, day); labels maps (person_id, accession) -> {"label_1yr", "label_2yr"}
    (either may be missing). A point whose CT has neither label is dropped (nothing to score)."""
    rows = []
    for pid, g in scans.groupby("person_id", sort=True):
        picked = pick_points(g.to_dict("records"))
        for key in c.POINT_KEYS:
            s = picked.get(key)
            if s is None:
                continue
            lab = labels.get((str(pid), s["accession_number"]), {})
            if lab.get("label_1yr") is None and lab.get("label_2yr") is None:
                continue
            rows.append({"person_id": str(pid), "point": key, "accession_number": s["accession_number"],
                         "cut": s["scan_date"].date(), "day": int(s["day"]),
                         "label_1yr": lab.get("label_1yr"), "label_2yr": lab.get("label_2yr")})
    return rows


# ==================================================================================================
# table
# ==================================================================================================


def _k16_labels(path: Path, persons: set[str]):
    import pandas as pd

    t = pd.read_parquet(path, columns=["task", "split", "person_id", "accession_number", "label", "score"])
    t = t[t.task.isin(list(c.K16_TASKS.values())) & t.person_id.isin(persons)]
    return t


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_table(args) -> int:
    import pandas as pd

    k29 = pd.read_parquet(c.K29_PHASE_TABLE, columns=["person_id", "accession_number", "scan_date", "split", "day",
                                                      "exam_type", "subsetting"])
    k29 = k29[k29.subsetting.isin(c.SUBSETTINGS)]
    chest = k29[k29.exam_type.isin(c.CHEST_EXAMS) & k29.day.notna()].copy()
    chest["day"] = chest["day"].astype(int)

    # -------- split membership: K29's benchmark split, checked against the died-2yr-ehr task's splits
    id_map = {m["id"]: str(m["person_id"]) for m in _read_jsonl(c.DIED_2YR_EHR_ID_MAP)}
    task_split = {id_map[r["id"]]: s for s in ("train", "val", "test")
                  for r in _read_jsonl(c.DIED_2YR_EHR_SPLITS / f"{s}.jsonl")}
    val_persons = set(k29[k29.split == "val"].person_id)
    clash = sum(1 for p in val_persons if task_split.get(p) not in (None, "val"))
    print(f"validation patients, first treatment {' or '.join(c.SUBSETTINGS)}: {len(val_persons)}; "
          f"in the died-2yr-ehr task's VAL {sum(task_split.get(p) == 'val' for p in val_persons)}, in its TRAIN or "
          f"TEST {clash}, not in that task {sum(p not in task_split for p in val_persons)}")
    if clash:
        raise SystemExit("a K29 validation patient is in the died-2yr-ehr task's TRAIN or TEST: stop")

    def rows_for(split: str):
        persons = set(k29[k29.split == split].person_id)
        lab = _k16_labels(c.K16_LABELS, persons)
        if set(lab.split.unique()) - {split}:
            raise SystemExit(f"K16 labels for {split} patients carry another split")
        labels: dict = {}
        for r in lab.itertuples():
            field = next(k for k, v in c.K16_TASKS.items() if v == r.task)
            labels.setdefault((r.person_id, r.accession_number), {})[field] = int(r.label)
        return point_rows(chest[chest.person_id.isin(persons)], labels), lab

    rows, _lab = rows_for("val")
    after = {r["person_id"] for r in rows if r["point"] != "start"}
    print(f"patients with a labelled decision point: {len({r['person_id'] for r in rows})}; with one after treatment "
          f"start: {len(after)}")
    print(f"{'point':<11}{'patients':>9}{'1-yr n':>8}{'1-yr deaths':>12}{'2-yr n':>8}{'2-yr deaths':>12}")
    for key in c.POINT_KEYS:
        pr = [r for r in rows if r["point"] == key]
        n1 = [r["label_1yr"] for r in pr if r["label_1yr"] is not None]
        n2 = [r["label_2yr"] for r in pr if r["label_2yr"] is not None]
        print(f"{key:<11}{len(pr):>9}{len(n1):>8}{sum(n1):>12}{len(n2):>8}{sum(n2):>12}")
    print(f"total rows {len(rows)}, calls for both arms (the memory arm shares the fresh arm's treatment-start "
          f"answers) {2 * len(rows) - sum(r['point'] == 'start' for r in rows)}")

    # -------- outcome check: every other K16 file holding these CTs gives the same labels
    want = {(r["person_id"], r["accession_number"], f): r[f] for r in rows for f in c.K16_TASKS if r[f] is not None}
    compared = mismatched = files = 0
    for f in sorted(c.K16_DIR.glob("*/*per_example.parquet")):
        if f.resolve() == c.K16_LABELS.resolve():
            continue  # the label source itself: comparing it with itself cannot fail
        files += 1
        other = _k16_labels(f, {r["person_id"] for r in rows})
        for r in other.itertuples():
            field = next(k for k, v in c.K16_TASKS.items() if v == r.task)
            key = (r.person_id, r.accession_number, field)
            if key in want:
                compared += 1
                mismatched += int(r.label) != want[key]
    print(f"outcome check: {compared} label comparisons against {files} other K16 files (the label source "
          f"excluded), {mismatched} disagree "
          f"-> {'PASS' if not mismatched else 'FAIL'}")
    if mismatched:
        raise SystemExit("K16 files disagree on a death label: stop")

    # -------- opaque ids: kept stable across rebuilds (the private patient list holds the map)
    c.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    old_p = (pd.read_parquet(c.PRIVATE_DIR / "patients.parquet") if (c.PRIVATE_DIR / "patients.parquet").exists()
             else pd.DataFrame(columns=["oid", "person_id"]))
    oid = dict(zip(old_p.person_id.astype(str), old_p.oid))
    old_r = (pd.read_parquet(c.PRIVATE_DIR / "points.parquet") if (c.PRIVATE_DIR / "points.parquet").exists()
             else pd.DataFrame(columns=["record", "person_id", "cut"]))
    rec = {(str(p), str(d)): r for r, p, d in zip(old_r.record, old_r.person_id, old_r.cut)}
    for r in rows:
        oid.setdefault(r["person_id"], "p" + secrets.token_hex(6))
        r["patient"] = oid[r["person_id"]]
        r["row"] = f"{r['patient']}-{r['point']}"
        r["record"] = rec.setdefault((r["person_id"], str(r["cut"])), "r" + secrets.token_hex(6))
    persons = sorted({r["person_id"] for r in rows})

    # -------- index tumour: the died-2yr-ehr cohort's, else the same rule read from BigQuery
    known = pd.read_parquet(c.DIED_2YR_EHR_COHORT)
    tumor = {str(p): (None if pd.isna(t) else int(t)) for p, t in zip(known.person_id, known.tumor_key)}
    missing = [p for p in persons if p not in tumor]
    if missing:
        tumor.update(index_tumours(missing))
    print(f"index tumour: {sum(tumor.get(p) is not None for p in persons)} of {len(persons)} patients "
          f"({len(missing)} read from BigQuery)")

    # -------- write (private)
    write_parquet(pd.DataFrame([{"oid": oid[p], "person_id": int(p), "tumor_key": tumor.get(p)} for p in persons])
                  .astype({"tumor_key": "Int64"}), c.PRIVATE_DIR / "patients.parquet")
    pts = pd.DataFrame(rows)
    write_parquet(pts.astype({"label_1yr": "Int64", "label_2yr": "Int64"}), c.PRIVATE_DIR / "points.parquet")
    records = pts.drop_duplicates("record")
    cohort = pd.DataFrame({"id": records.record, "person_id": records.person_id.astype("int64"),
                           "cutoff": records.cut, "tumor_key": [tumor.get(p) for p in records.person_id]})
    write_parquet(cohort.astype({"tumor_key": "Int64"}), c.PRIVATE_DIR / "cohort.parquet")
    print(f"records to export: {len(cohort)} ({len(rows) - len(cohort)} points share a CT with an earlier point)")
    write_jsonl(c.PRIVATE_DIR / "rows.jsonl", [{"row": r["row"], "patient": r["patient"], "point": r["point"],
                                                "record": r["record"], "cut": str(r["cut"])} for r in rows])
    write_jsonl(c.PRIVATE_DIR / "labels.jsonl", [{"row": r["row"], "label_1yr": r["label_1yr"],
                                                  "label_2yr": r["label_2yr"]} for r in rows])
    # The died-2yr-ehr task id of each patient that task holds (its VAL): for the treatment-start check against the
    # Haiku probability seed's answers (run_over_time.start_gate).
    to_task = {p: i for i, p in id_map.items() if task_split.get(p) == "val"}
    write_jsonl(c.PRIVATE_DIR / "seed_ids.jsonl", [{"patient": oid[p], "task_id": to_task[p]} for p in persons
                                                   if p in to_task])
    print(f"patients also in the died-2yr-ehr task's VAL (for the treatment-start check): "
          f"{sum(p in to_task for p in persons)}")
    # What died-2yr-ehr's build_data.py rehydrate reads: splits (record ids, in VAL), pools, cutoffs.
    (c.PRIVATE_DIR / "splits").mkdir(exist_ok=True)
    write_jsonl(c.PRIVATE_DIR / "splits" / "val.jsonl", [{"id": i} for i in cohort.id])
    write_jsonl(c.PRIVATE_DIR / "splits" / "train.jsonl", [])
    write_jsonl(c.PRIVATE_DIR / "splits" / "test.jsonl", [])
    (c.PRIVATE_DIR / "pools.json").write_text(json.dumps({"train": [], "val": []}) + "\n")
    write_jsonl(c.PRIVATE_DIR / "cutoffs.jsonl", [{"id": i, "cutoff": str(d)} for i, d in zip(cohort.id, cohort.cutoff)])
    probe_scores(rows)
    print(f"written {c.PRIVATE_DIR}")
    return 0


def probe_scores(rows: list[dict]) -> None:
    """K16 probe scores on the decision-point rows: PRIVATE_DIR/probes.parquet (row, probe, horizon field, score)."""
    import pandas as pd

    by_key: dict[tuple, list[str]] = {}  # two points on the same CT share its score
    for r in rows:
        by_key.setdefault((r["person_id"], r["accession_number"]), []).append(r["row"])
    out = []
    for name, path in c.K16_PROBES.items():
        t = _k16_labels(path, {r["person_id"] for r in rows})
        t = t[t.split == "val"]
        n0 = len(out)
        for r in t.itertuples():
            field = next(k for k, v in c.K16_TASKS.items() if v == r.task)
            for row in by_key.get((r.person_id, r.accession_number), []):
                out.append({"row": row, "probe": name, "field": field, "score": float(r.score)})
        got = {(x["row"], x["field"]) for x in out[n0:]}
        print(f"probe {name}: " + ", ".join(f"{f} {sum((r['row'], f) in got for r in rows)} of "
                                            f"{sum(r[f] is not None for r in rows)} labelled rows" for f in c.K16_TASKS)
              + ("" if len(out) > n0 else " (none on VAL: this K16 file holds TEST rows only)"))
    write_parquet(pd.DataFrame(out, columns=["row", "probe", "field", "score"]), c.PRIVATE_DIR / "probes.parquet")


def index_tumours(persons: list[str]) -> dict[str, int | None]:
    """The index tumour as the died-2yr-ehr task picks it (its COHORT_SQL), for patients outside that task: the
    same SQL without its CT requirement (these patients may have no CT at the pre-treatment cutoff). Read-only."""
    from google.cloud import bigquery

    bd = _build_data()
    sql = bd.COHORT_SQL.replace(" AND image_series_uid IS NOT NULL", "")
    if sql == bd.COHORT_SQL:
        raise SystemExit("died-2yr-ehr's COHORT_SQL changed: re-check the index-tumour query")
    client = bigquery.Client(project=bd.PROJECT)
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("task", "STRING", bd.TASK),
                                                    bigquery.ArrayQueryParameter("pids", "INT64", [int(p) for p in persons])])
    out: dict[str, set] = {}
    for r in client.query(sql, job_config=cfg).result():
        out.setdefault(str(r["person_id"]), set()).add(r["tumor_key"])
    if any(len(v) > 1 for v in out.values()):
        raise SystemExit("more than one index tumour for a patient")
    return {p: next(iter(out[p])) if p in out else None for p in persons}


def write_parquet(df, path: Path) -> None:
    """Write locally, then copy whole onto the mount (gcsfuse: no partial file, no rename)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / path.name
        df.to_parquet(local, index=False)
        shutil.copyfile(local, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")


# ==================================================================================================
# export and local
# ==================================================================================================


def exporter_cmd(*extra: str) -> list[str]:
    bd = _build_data()
    if not bd.EXPORTER.is_file():
        raise SystemExit(f"no exporter at {bd.EXPORTER}: set OMOP_EXPORT_REPO to a vista-onc-labelers checkout "
                         "on feat/shared-omop-export with the several-cuts change")
    stage = Path.home() / ".cache" / "died-ehr-over-time-export-stage"
    return ["uv", "run", str(bd.EXPORTER), "--cohort", str(c.PRIVATE_DIR / "cohort.parquet"),
            "--registry-allowlist", str(bd.ALLOWLIST), "--out", str(c.DATA_DIR), "--stage", str(stage), *extra]


def cmd_export(args) -> int:
    if args.dry_run:
        return subprocess.run(exporter_cmd("--dry-run")).returncode
    if not args.go:
        raise SystemExit("the export is a PHI write and an Expensive step: rerun with --go once Phil has approved it, "
                         "or use --dry-run to price it")
    code = subprocess.run(["nice", "-n", "19", *exporter_cmd("--archive", str(c.PRIVATE_DIR / "archive"))],
                          env={**__import__("os").environ, "OMP_NUM_THREADS": "2"}).returncode
    if code == 0:
        cmd_local(args)
    return code


LOCAL_FILES = ("rows.jsonl", "labels.jsonl", "probes.parquet", "seed_ids.jsonl")


def cmd_local(_args) -> int:
    c.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in LOCAL_FILES:
        shutil.copyfile(c.PRIVATE_DIR / name, c.DATA_DIR / name)
    print(f"copied {', '.join(LOCAL_FILES)} to {c.DATA_DIR}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    e = sub.add_parser("export")
    e.add_argument("--go", action="store_true", help="Phil approved the export")
    e.add_argument("--dry-run", action="store_true")
    sub.add_parser("local")
    args = ap.parse_args(argv)
    return {"table": cmd_table, "export": cmd_export, "local": cmd_local}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
