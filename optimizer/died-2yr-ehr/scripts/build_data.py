"""Build the died-2yr-ehr task data: the CT task's patients, splits and pools, and each patient's EHR record cut at
first treatment - 7 days, written by the shared exporter (vista-onc-labelers common/omop_export/export_omop.py).

    splits     copy the CT task's splits/{train,val,test}.jsonl and pools.json to the mount data folder, checked
               byte-identical: same ids, labels, split membership and pools. Prints counts per split.
    cohort     query BigQuery for each patient's cutoff (vista_bench embed_time = first treatment - 7 days) and
               index tumour (picked as the label picks it; the 2026-10-21 registry audit's SQL) and write
               PRIVATE/cohort.parquet (id, person_id, cutoff, tumor_key: the exporter's input; private because it
               holds person_id) and DATA/cutoffs.jsonl ({"id", "cutoff"}, what rehydrate puts in meta.json).
    export     run the exporter into the local working copy, leak-check it, and archive it to DATA/archive/.
               The first PHI write: an Expensive step that needs Phil's go, so it refuses without --go.
               --dry-run prices the queries instead (no download).
    rehydrate  rebuild the local working copy (DATA_DIR) from the mount: splits, pools and every patient folder
               from DATA/archive/tables/, notes.txt in the exporter's format, meta.json from cutoffs.jsonl. No
               BigQuery: for the run VM. The exporter's registry_dates/ side file (raw governing dates, for its leak
               check) is never in the archive and is never written here.

    OMP_NUM_THREADS=2 nice -n 19 ~/code/vista_eval_vlm-radprog/.venv/bin/python scripts/build_data.py splits
    OMP_NUM_THREADS=2 nice -n 19 ~/code/vista_eval_vlm-radprog/.venv/bin/python scripts/build_data.py cohort
    python scripts/build_data.py export --dry-run          (then, with Phil's go: export --go)
    uv run --with duckdb==1.5.6 --with pandas python scripts/build_data.py rehydrate      (run VM)

Prints counts only: never an id, a date or a row.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = "som-nero-plevriti-deidbdf"
DS = f"{PROJECT}.vista_cohort_omop_aug2026_v1_7"
LABEL_TABLE = f"{PROJECT}.vista_bench_v1_7.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
TASK = "died_any_cause_2_yr"
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
CT_DATA_DIR = MOUNT / "died-2yr-ct-v1_7"
ID_MAP = MOUNT / "died-2yr-ct-v1_7-private" / "id_map.jsonl"
MOUNT_DATA_DIR = Path(os.environ.get("TASK_MOUNT_DATA_DIR", MOUNT / "died-2yr-ehr-v1_7"))
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "died-2yr-ehr-v1_7-private"))
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", Path.home() / "data" / "died-2yr-ehr-v1_7"))
STAGE = Path(os.environ.get("TASK_EXPORT_STAGE", Path.home() / ".cache" / "died-2yr-ehr-export-stage"))
EXPORTER_REPO = Path(os.environ.get("OMOP_EXPORT_REPO", Path.home() / "code" / "vista-onc-labelers-omop-export"))
EXPORTER = EXPORTER_REPO / "common" / "omop_export" / "export_omop.py"
ALLOWLIST = HERE / "registry_allowlist.yaml"
SPLITS = ("train", "val", "test")

# The index tumour, as the label picks it: a thoracic tumour (vista_bench's primary-site / histology filter) whose
# diagnosis date equals the label's diagnosis_date; ties to the one whose first-course date is cutoff + 7 days,
# then the latest diagnosis, then tumor_key. Copied from the 2026-10-21 registry audit (common.py), which found all
# 5,549 patients with an index tumour matching the label's diagnosis date.
THORACIC = """(d.primarysite IN ('C150','C151','C152','C153','C154','C155','C158','C159',
    'C339','C340','C341','C342','C343','C348','C349','C379','C380','C381','C382','C383','C384','C388','C390','C398',
    'C399','C412','C413'))
  OR (d.primarysite = 'C493' AND d.morphtypebehavicdo3 NOT IN ('9220','9240'))
  OR (d.morphtypebehavicdo3 IN ('9050','9051','9052','9053','9054','9055'))"""
COHORT_SQL = f"""
WITH cohort AS (
  SELECT person_id, diagnosis_date, embed_time
  FROM `{LABEL_TABLE}`
  WHERE task = @task AND label IN (0, 1) AND split IS NOT NULL AND image_series_uid IS NOT NULL
    AND person_id IN UNNEST(@pids)
),
dx AS (
  SELECT d.person_id, d.tumor_key,
    CASE
      WHEN REGEXP_CONTAINS(d.dateofdiagnosis, r'^\\d{{8}}$') THEN SAFE.PARSE_DATE('%Y%m%d', d.dateofdiagnosis)
      WHEN REGEXP_CONTAINS(d.dateofdiagnosis, r'^\\d{{6}}$') AND SAFE_CAST(SUBSTR(d.dateofdiagnosis,5,2) AS INT64) BETWEEN 1 AND 12
        THEN DATE(SAFE_CAST(SUBSTR(d.dateofdiagnosis,1,4) AS INT64), SAFE_CAST(SUBSTR(d.dateofdiagnosis,5,2) AS INT64), 1)
    END AS dxd,
    ({THORACIC}) AS in_filter
  FROM `{DS}.registry_case_diagnoses` d
  WHERE d.person_id IN (SELECT person_id FROM cohort)
)
SELECT c.person_id, DATE(c.embed_time) AS cutoff, x.tumor_key, IFNULL(x.dxd = c.diagnosis_date, FALSE) AS dx_match,
       IFNULL(x.in_filter, FALSE) AS in_filter
FROM cohort c
LEFT JOIN (
  SELECT * EXCEPT (rn) FROM (
    SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.person_id ORDER BY
             IF(IFNULL(x.in_filter, FALSE), 0, 1),
             IF(x.dxd = c.diagnosis_date, 0, 1),
             IF(t.date1stcrsrxcoc_date = DATE_ADD(DATE(c.embed_time), INTERVAL 7 DAY), 0, 1),
             x.dxd DESC, x.tumor_key) AS rn
    FROM dx x JOIN cohort c USING (person_id)
    LEFT JOIN `{DS}.registry_case_treatments` t ON t.person_id = x.person_id AND t.tumor_key = x.tumor_key)
  WHERE rn = 1) x USING (person_id)
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_counts(root: Path) -> None:
    pools = json.loads((root / "pools.json").read_text())
    for s in SPLITS:
        rows = read_jsonl(root / "splits" / f"{s}.jsonl")
        died = sum(r["label"] for r in rows)
        line = f"{s}: {len(rows)} patients, {died} died ({died / len(rows):.1%})"
        if s in pools:
            lab = {r["id"]: r["label"] for r in rows}
            pd_ = sum(lab[i] for i in pools[s])
            line += f"; pool {len(pools[s])}, {pd_} died"
        print(line)


def cmd_splits(_args) -> int:
    (MOUNT_DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)
    names = [f"splits/{s}.jsonl" for s in SPLITS] + ["pools.json"]
    for n in names:
        shutil.copyfile(CT_DATA_DIR / n, MOUNT_DATA_DIR / n)  # gcsfuse: plain copy, no rename
        if not filecmp.cmp(CT_DATA_DIR / n, MOUNT_DATA_DIR / n, shallow=False):
            raise SystemExit(f"{n}: the copy differs from the CT task's")
    print(f"copied {len(names)} files from the CT task, byte-identical")
    split_counts(MOUNT_DATA_DIR)
    return 0


def cmd_cohort(_args) -> int:
    import pandas as pd
    from google.cloud import bigquery

    id_map = read_jsonl(ID_MAP)
    task_ids = {r["id"] for s in SPLITS for r in read_jsonl(CT_DATA_DIR / "splits" / f"{s}.jsonl")}
    # The id map also lists patients the CT task dropped for a missing NIfTI: keep only the task's patients.
    id_map = [m for m in id_map if m["id"] in task_ids]
    if len(id_map) != len(task_ids):
        raise SystemExit("the CT id map does not cover every task patient")
    client = bigquery.Client(project=PROJECT)
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("task", "STRING", TASK),
        bigquery.ArrayQueryParameter("pids", "INT64", [int(m["person_id"]) for m in id_map])])
    rows = {r["person_id"]: dict(r) for r in client.query(COHORT_SQL, job_config=cfg).result()}
    missing = sum(int(m["person_id"]) not in rows for m in id_map)
    if missing or len(rows) != len(id_map):
        raise SystemExit(f"{missing} patients have no label row; {len(rows)} rows for {len(id_map)} patients")
    out = pd.DataFrame([{"id": m["id"], "person_id": int(m["person_id"]), "cutoff": rows[int(m["person_id"])]["cutoff"],
                         "tumor_key": rows[int(m["person_id"])]["tumor_key"]} for m in id_map])
    out["tumor_key"] = out["tumor_key"].astype("Int64")
    out["cutoff"] = pd.to_datetime(out["cutoff"]).dt.date
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "died2yr_ehr_cohort.parquet"
    out.to_parquet(tmp, index=False)
    shutil.copyfile(tmp, PRIVATE_DIR / "cohort.parquet")
    tmp.unlink()
    MOUNT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (MOUNT_DATA_DIR / "cutoffs.jsonl").open("w", encoding="utf-8") as f:
        for r in out.itertuples():
            f.write(json.dumps({"id": r.id, "cutoff": r.cutoff.isoformat()}) + "\n")
    c = Counter((bool(v["in_filter"]), bool(v["dx_match"])) for v in rows.values())
    print(f"cohort: {len(out)} patients; index tumour found {int(out['tumor_key'].notna().sum())}; "
          f"thoracic and matching the label's diagnosis date {c[(True, True)]}; other {len(out) - c[(True, True)]}")
    print(f"written {PRIVATE_DIR / 'cohort.parquet'} and {MOUNT_DATA_DIR / 'cutoffs.jsonl'}")
    return 0


def exporter_cmd(*extra: str) -> list[str]:
    if not EXPORTER.is_file():
        raise SystemExit(f"no exporter at {EXPORTER}: set OMOP_EXPORT_REPO to a vista-onc-labelers checkout with it")
    return ["uv", "run", str(EXPORTER), "--cohort", str(PRIVATE_DIR / "cohort.parquet"),
            "--registry-allowlist", str(ALLOWLIST), "--out", str(DATA_DIR), "--stage", str(STAGE), *extra]


def copy_answer_key(src: Path, dest: Path) -> None:
    (dest / "splits").mkdir(parents=True, exist_ok=True)
    for n in [f"splits/{s}.jsonl" for s in SPLITS] + ["pools.json"]:
        shutil.copyfile(src / n, dest / n)


def cmd_export(args) -> int:
    if args.dry_run:
        return subprocess.run(exporter_cmd("--dry-run")).returncode
    if not args.go:
        raise SystemExit("export is the first PHI write (an Expensive step in the spec): rerun with --go once Phil "
                         "has approved it, or use --dry-run to price it")
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    code = subprocess.run(["nice", "-n", "19", *exporter_cmd("--archive", str(MOUNT_DATA_DIR / "archive"))],
                          env=env).returncode
    if code == 0:
        copy_answer_key(MOUNT_DATA_DIR, DATA_DIR)
        n = sum(1 for _ in (DATA_DIR / "patients").iterdir())
        print(f"export done: {n} patient folders in {DATA_DIR}; archive in {MOUNT_DATA_DIR / 'archive'}")
    return code


def cmd_rehydrate(_args) -> int:
    import duckdb

    archive = MOUNT_DATA_DIR / "archive"
    tables = sorted(p.stem for p in (archive / "tables").glob("*.parquet"))
    if not tables:
        raise SystemExit(f"no archive at {archive / 'tables'}")
    if (DATA_DIR / "patients").exists():
        raise SystemExit(f"{DATA_DIR / 'patients'} exists: delete it to rebuild")
    copy_answer_key(MOUNT_DATA_DIR, DATA_DIR)
    cutoffs = {r["id"]: r["cutoff"] for r in read_jsonl(MOUNT_DATA_DIR / "cutoffs.jsonl")}
    ids = {r["id"] for s in SPLITS for r in read_jsonl(DATA_DIR / "splits" / f"{s}.jsonl")}
    if ids != set(cutoffs):
        raise SystemExit("cutoffs.jsonl and the splits list different patients")
    pdir, part = DATA_DIR / "patients", STAGE / "rehydrate"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    con = duckdb.connect()
    # Insertion order off: the partitioned COPY of the measurement table ran out of memory at 4 GB with it on
    # (phil-sllm-02, 2026-10-27), and row order within a patient's table carries no meaning.
    con.execute("SET memory_limit = '4GB'; SET threads = 2; SET partitioned_write_max_open_files = 20000; "
                "SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{STAGE / 'spill'}'")
    rows = {}
    for t in tables:
        src = archive / "tables" / f"{t}.parquet"
        rows[t] = con.execute(f"SELECT count(*) FROM read_parquet('{src}')").fetchone()[0]
        con.execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{part / t}' (FORMAT parquet, PARTITION_BY (id))")
        for d in (part / t).glob("id=*"):
            files = list(d.glob("*.parquet"))
            assert len(files) == 1, f"{t}: {len(files)} files for one patient"
            (pdir / d.name.removeprefix("id=")).mkdir(parents=True, exist_ok=True)
            shutil.move(str(files[0]), pdir / d.name.removeprefix("id=") / f"{t}.parquet")
        shutil.rmtree(part / t)
        print(f"{t}: {rows[t]:,} rows", flush=True)
    if "note" in tables:  # notes.txt exactly as export_omop.write_notes_txt writes it
        cur = con.execute("SELECT id, note_id, note_datetime, note_date, COALESCE(note_type, 'ORDER: ' || order_type), "
                          f"note_text FROM read_parquet('{archive / 'tables' / 'note.parquet'}') "
                          "ORDER BY id, note_datetime, note_id")
        id_now, fh = None, None
        while batch := cur.fetchmany(20000):
            for pid, nid, ndt, nd, ntype, text in batch:
                if pid != id_now:
                    if fh:
                        fh.close()
                    fh, id_now = (pdir / pid / "notes.txt").open("w", encoding="utf-8"), pid
                fh.write(f"===== NOTE {nid} | {ndt or nd} | {ntype} =====\n{text or ''}\n\n")
        if fh:
            fh.close()
    for i in ids:  # every patient has a folder and a meta.json, as the exporter writes them
        (pdir / i).mkdir(parents=True, exist_ok=True)
        (pdir / i / "meta.json").write_text(json.dumps({"id": i, "cutoff": cutoffs[i]}) + "\n")
    # The check: every archived row reached a patient file, the archive's counts agree, nothing but patients.
    bad = 0
    want = json.loads((archive / "counts.json").read_text()).get("rows_per_table", {})
    for t in tables:
        got = con.execute(f"SELECT count(*) FROM read_parquet('{pdir}/*/{t}.parquet')").fetchone()[0]
        if got != rows[t] or (t in want and want[t] != got):
            print(f"rows: {t} patient files {got:,}, archive {rows[t]:,}, export counts {want.get(t)}")
            bad += 1
    stray = {p.name for p in pdir.iterdir()} - ids
    print(f"rehydrated {len(ids)} patients ({len(stray)} stray folders), {len(tables)} tables; "
          f"{'PASS' if not bad and not stray else 'FAIL'}")
    return 1 if bad or stray else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("splits")
    sub.add_parser("cohort")
    e = sub.add_parser("export")
    e.add_argument("--go", action="store_true", help="Phil approved the export")
    e.add_argument("--dry-run", action="store_true")
    sub.add_parser("rehydrate")
    args = ap.parse_args()
    return {"splits": cmd_splits, "cohort": cmd_cohort, "export": cmd_export, "rehydrate": cmd_rehydrate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
