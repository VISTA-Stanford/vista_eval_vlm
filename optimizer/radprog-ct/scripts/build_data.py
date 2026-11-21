"""Build the radprog-ct task data. Prints counts only: never an id, a UID, a date, a row or any report text.

Spec planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md (Approved 2026-11-21). Pairs come from vista_bench
v1.7 ``radiology_progression_tasks`` (task ``radiology_progression``, v2 labels): each scored CT with the prior CT the
report was read against (the table's own prior columns, as the eval set's step-3 lists use), both NIfTIs on the mount,
and scored by every comparison probe (evalset.samples.PROBE_FILES), so the agent, the VLMs and the probes all answer
about the same kind of pair. Answer = two-way, following MSK-CHORD: progression or mixed (labels 1, 3) = yes, stable
or response (0, 2) = no (evalset.cards.progression.TWO_WAY).

    select   the agent's pools at the natural rate, seeded: TRAIN (feedback) TRAIN_N pairs from the table's train split,
             VAL (picking) VAL_N from its valid split. Writes splits/{train,val}.jsonl, pools.json, the private id map
    volumes  convert each pair's NIfTIs into patients/<id>/{prior,current}/ct.npz + meta.json (ct-findings-5's converter:
             HU int16, RAS, thin scans averaged into ~2.5 mm slabs, no header, date or UID kept)
    test     the TEST split = the eval set's fixed 1,000 (samples/step3_test1000.parquet), with its own private id
             map; then ``volumes --split test``. Only on Phil's go (spec: each test run needs its own go)

    cd optimizer/radprog-ct
    uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with db-dtypes python -B scripts/build_data.py select
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_data.py volumes --workers 4

BigQuery runs on this host's application-default login. CTs are written to the shared mount (secure environment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "som-nero-plevriti-deidbdf"
TABLE = f"{PROJECT}.vista_bench_v1_7.radiology_progression_tasks"
KEY = "progression"  # the answer key in splits/*.jsonl (config.QUESTIONS)
TWO_WAY = {0: 0, 1: 1, 2: 0, 3: 1}  # = evalset.cards.progression.TWO_WAY (check_task.py compares them)
BUCKET = Path("/mnt/su-vista-uscentral1")
NIFTI_ROOT = BUCKET / "chaudhari_lab/ct_data/ct_scans/vista/feb26"  # context.nifti's snapshot
SAMPLES = BUCKET / "vistabench/vlm/evalset_v1/samples"
#: = evalset.samples.PROBE_DIR / PROBE_FILES (check_task.py compares them).
PROBE_DIR = BUCKET / "vistabench/results/v1_7/ct/radiology_progression_tasks"
PROBE_FILES = ("pillar0_prior_pair_per_example.parquet", "colipri_prior_pair_per_example.parquet",
               "merlinonc_top_centered_v2.2_prior_pair_per_example.parquet")
MOUNT = BUCKET / "chaudhari_lab/phil"
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "radprog-ct-v1_7"))
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "radprog-ct-v1_7-private"))
SALT = "radprog-ct-v1"
TRAIN_N, VAL_N = 400, 200  # DECIDE: Phil 2026-10-26, re-confirmed 2026-11-21
SEED = "radprog-ct-v1:pools"
SLAB_MM = 2.5


def opaque_id(accession) -> str:
    return hashlib.sha1(f"{SALT}:{accession}".encode()).hexdigest()[:16]


def nifti(study: str, series: str) -> Path:
    return NIFTI_ROOT / f"{study}__{series}.nii.gz"


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def probe_accessions(split: str) -> set[str]:
    import pandas as pd

    sets = []
    for f in PROBE_FILES:
        pe = pd.read_parquet(PROBE_DIR / f, columns=["task", "split", "accession_number"])
        pe = pe[(pe["task"] == "radiology_progression") & (pe["split"].replace({"valid": "val"}) == split)]
        sets.append(set(pe["accession_number"].astype(str)))
    return set.intersection(*sets)


def table_pairs(split: str) -> list[dict]:
    """Every scored pair of one table split with both NIfTIs on the mount and every probe's score."""
    from google.cloud import bigquery

    sql = f"""select person_id, ct_accession_number, label, image_study_uid, image_series_uid,
          prior_image_study_uid, prior_image_series_uid, prior_gap_days
        from `{TABLE}`
        where task = 'radiology_progression' and split = @split and label in (0, 1, 2, 3)
          and image_series_uid is not null and prior_image_series_uid is not null"""
    job = bigquery.Client(project=PROJECT).query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("split", "STRING", split)]))
    rows = [dict(r) for r in job.result()]
    if len({str(r["ct_accession_number"]) for r in rows}) != len(rows):
        raise SystemExit("more than one row per accession: the table's contract changed")
    keep = probe_accessions("val" if split == "valid" else split)
    out, no_probe = [], 0
    for r in rows:
        acc = str(r["ct_accession_number"])
        if acc not in keep:
            no_probe += 1
            continue
        gap = r["prior_gap_days"]
        out.append({"accession": acc, "person_id": int(r["person_id"]), "label4": int(r["label"]),
                    "gap_days": None if gap is None else int(gap),
                    "current": str(nifti(r["image_study_uid"], r["image_series_uid"])),
                    "prior": str(nifti(r["prior_image_study_uid"], r["prior_image_series_uid"]))})
    print(f"{split}: {len(rows)} scored pairs; {no_probe} not scored by every probe (dropped); {len(out)} eligible "
          "before the NIfTI check")
    return sorted(out, key=lambda e: opaque_id(e["accession"]))


def entry(e: dict, split: str) -> tuple[dict, dict]:
    sid = opaque_id(e["accession"])
    row = {"id": sid, KEY: TWO_WAY[e["label4"]]}
    return row, {"id": sid, "split": split, **e}


def counts(rows: list[dict], id_map: list[dict]) -> dict:
    return {"pairs": len(rows), "patients": len({m["person_id"] for m in id_map}), "yes": sum(r[KEY] for r in rows),
            "four_way": dict(sorted(Counter(str(m["label4"]) for m in id_map).items()))}


def cmd_select(_args) -> int:
    rng = random.Random(SEED)
    rows, id_map, cnt = {}, [], {}
    for split, table_split, n in (("train", "train", TRAIN_N), ("val", "valid", VAL_N)):
        pool = table_pairs(table_split)
        # Natural rate: a simple random draw (Phil 2026-10-26). The NIfTI check runs on the shuffled order until n pairs
        # with both files are found, so the draw is random among pairs on the mount without statting all ~21k.
        order = rng.sample(pool, len(pool))
        picked, missing = [], 0
        for e in order:
            if len(picked) == n:
                break
            if Path(e["current"]).is_file() and Path(e["prior"]).is_file():
                picked.append(e)
            else:
                missing += 1
        if len(picked) < n:
            raise SystemExit(f"{split}: only {len(picked)} pairs with both NIfTIs on the mount, need {n}")
        print(f"{split}: drew {n}; skipped {missing} with a NIfTI missing on the way")
        pairs = [entry(e, split) for e in picked]
        rows[split] = sorted((r for r, _m in pairs), key=lambda r: r["id"])
        id_map += [m for _r, m in pairs]
        cnt[split] = counts(rows[split], [m for _r, m in pairs])
    if {m["person_id"] for m in id_map if m["split"] == "train"} & {m["person_id"] for m in id_map if m["split"] == "val"}:
        raise SystemExit("STOP: a patient in both TRAIN and VAL")
    test = SAMPLES / "step3_test1000.parquet"
    if test.is_file():
        import pandas as pd

        tp = set(pd.read_parquet(test, columns=["person_id"])["person_id"].astype(int))
        if tp & {m["person_id"] for m in id_map}:
            raise SystemExit("STOP: a pool patient is in the 1,000 TEST pairs")
    if len({m["id"] for m in id_map}) != len(id_map):
        raise SystemExit("two pairs share an opaque id")
    for split, items in rows.items():
        write_jsonl(DATA_DIR / "splits" / f"{split}.jsonl", items)
    write_jsonl(PRIVATE_DIR / "id_map.jsonl", id_map)
    (DATA_DIR / "pools.json").write_text(json.dumps({
        **{s: [r["id"] for r in items] for s, items in rows.items()},
        "_draw": {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "table": TABLE,
                  "seed": SEED, "salt": SALT, "rule": ("simple random draw at the natural rate from the table's train "
                  "(TRAIN, feedback) and valid (VAL, picking) splits, pairs with both NIfTIs on the mount and every "
                  "comparison probe's score; answer = progression or mixed"), "counts": cnt}}, indent=1),
        encoding="utf-8")
    for s, c in cnt.items():
        print(f"{s}: {c['pairs']} pairs from {c['patients']} patients, {c['yes']} yes ({c['yes'] / c['pairs']:.0%}); "
              f"four-way {c['four_way']}")
    return 0


def cmd_test(_args) -> int:
    import pandas as pd

    d = pd.read_parquet(SAMPLES / "step3_test1000.parquet")
    pool = {m["person_id"] for m in read_jsonl(PRIVATE_DIR / "id_map.jsonl")}
    if pool & set(d["person_id"].astype(int)):
        raise SystemExit("STOP: a TEST patient is in the TRAIN/VAL pools")
    pairs = []
    for r in d.itertuples(index=False):
        e = {"accession": str(r.ct_accession_number), "person_id": int(r.person_id),
             "label4": int(r.label__radiology_progression_4way),
             "gap_days": None if pd.isna(r.prior_gap_days) else int(r.prior_gap_days), "current": str(nifti(r.study, r.series)),
             "prior": str(nifti(r.prior_study, r.prior_series))}
        if TWO_WAY[e["label4"]] != int(r.label__radiology_progression):
            raise SystemExit("the 1,000's two-way answer disagrees with its four-way label")
        pairs.append(entry(e, "test"))
    rows = sorted((r for r, _m in pairs), key=lambda r: r["id"])
    write_jsonl(PRIVATE_DIR / "id_map_test.jsonl", [m for _r, m in pairs])
    write_jsonl(DATA_DIR / "splits" / "test.jsonl", rows)
    print(f"test: {len(rows)} pairs, {sum(r[KEY] for r in rows)} yes; next: volumes --split test")
    return 0


def convert(item: tuple[str, str]) -> tuple[str, str]:
    """One NIfTI -> <out>/ct.npz + meta.json (ct-findings-5's converter). Returns (status, detail), no ids."""
    import nibabel as nib
    import numpy as np

    out_dir, src = item
    out = Path(out_dir)
    if (out / "meta.json").is_file():
        return "kept", ""
    try:
        img = nib.as_closest_canonical(nib.load(src))
        hu = np.asarray(img.dataobj, dtype=np.float32)
        if hu.ndim == 4:
            hu = hu[..., 0]
        sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
        k = max(1, int(round(SLAB_MM / sz))) if sz < SLAB_MM else 1
        if k > 1:
            n = hu.shape[2] // k * k
            hu = hu[:, :, :n].reshape(hu.shape[0], hu.shape[1], n // k, k).mean(axis=3)
            sz *= k
        hu = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "ct.tmp.npz"
        np.savez_compressed(tmp, hu=hu)
        tmp.replace(out / "ct.npz")
        meta = {
            "array": "ct.npz, key 'hu': Hounsfield units, int16, shape (x, y, z)",
            "shape": list(hu.shape),
            "spacing_mm": [round(sx, 4), round(sy, 4), round(sz, 4)],
            "orientation": ("RAS: index increases toward the patient's Right along axis 0 (x), Anterior along "
                            "axis 1 (y), Superior along axis 2 (z); z = 0 is the lowest (most inferior) slice"),
            "slab_averaged": k,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return "converted", ""
    except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the batch; reported as a count
        return "failed", type(exc).__name__


def cmd_volumes(args) -> int:
    entries = read_jsonl(PRIVATE_DIR / ("id_map_test.jsonl" if args.split == "test" else "id_map.jsonl"))
    if args.pool:  # one pool only. Never run two converters at once: on 2026-11-21 two overlapped and 52 of 1,200
        # volumes came out unreadable (both wrote the same ct.tmp.npz); found by loading every volume, then rebuilt
        entries = [e for e in entries if e["split"] == args.pool]
    if args.limit:
        entries = entries[: args.limit]
    for e in entries:  # the days between the scans: the only fact besides the volumes (spec: both CTs and their dates)
        d = DATA_DIR / "patients" / e["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "pair.json").write_text(json.dumps({"days_between_scans": e["gap_days"]}), encoding="utf-8")
    items = [(str(DATA_DIR / "patients" / e["id"] / scan), e[scan]) for e in entries for scan in ("prior", "current")]
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, detail) in enumerate(pool.map(convert, items), 1):
            status[st] += 1
            if detail:
                errors[detail] += 1
            if n % 50 == 0:
                print(f"{n}/{len(items)} {dict(status)}", flush=True)
    print(f"done {len(items)} volumes ({len(entries)} pairs): {dict(status)}; errors by type {dict(errors)}")
    return 1 if status["failed"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    v = sub.add_parser("volumes")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--split", choices=("pools", "test"), default="pools")
    v.add_argument("--pool", choices=("train", "val"), default=None, help="convert one pool only")
    sub.add_parser("test")
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
