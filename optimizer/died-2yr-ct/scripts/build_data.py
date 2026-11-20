"""Build the died-2yr-ct task data: one CT volume per patient, the answer key, and the fixed pools.

The scan is the one vista_bench v1.7 already pairs with each patient for ``died_any_cause_2_yr`` (latest
qualifying chest CT on or before first treatment - 7 days), so the agent sees the same scan the CT probes
(MerlinOnc and the rest) were scored on. Patients without 2 years of follow-up (label -1) or without a split
are dropped, as vista-eval drops them.

    select   query the label table, write splits/{train,val,test}.jsonl ({"id", "label"}), the fixed pools,
             and the private id map (id -> person_id and NIfTI path), and print counts per split and label
    volumes  convert each pool patient's NIfTI (or --split test: every TEST patient) into
             patients/<id>/ct.npz + meta.json: Hounsfield units, int16, reoriented to RAS, thin scans
             averaged into slabs of about SLAB_MM, and no header, date or UID carried over

Run with the vista_eval_vlm radprog environment (it has google-cloud-bigquery, nibabel and numpy), threads capped:

    OMP_NUM_THREADS=2 nice -n 19 ~/code/vista_eval_vlm-radprog/.venv/bin/python scripts/build_data.py select
    OMP_NUM_THREADS=1 nice -n 19 ~/code/vista_eval_vlm-radprog/.venv/bin/python scripts/build_data.py volumes --workers 4

Prints counts only: never an id, a UID or a row.
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
from pathlib import Path

PROJECT = "som-nero-plevriti-deidbdf"
LABEL_TABLE = f"{PROJECT}.vista_bench_v1_7.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
TASK = "died_any_cause_2_yr"
#: vista-ct's NIfTI store; the task table's own nifti_path column is a placeholder since v1.5.
NIFTI_ROOT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/feb26")
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "died-2yr-ct-v1_7"))
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "died-2yr-ct-v1_7-private"))
SALT = "died-2yr-ct-v1"
#: The table's split names -> the engine's.
SPLIT_NAMES = {"train": "train", "valid": "val", "test": "test"}
#: Fixed pools the data schedule draws from (Phil 2026-10-20: VAL 200). Seeded, natural death rate.
POOL_SIZES = {"train": 400, "val": 200}
#: Thin scans are averaged into slabs no thinner than this; thicker scans are kept as they are.
SLAB_MM = 2.5


def opaque_id(person_id: int) -> str:
    return hashlib.sha1(f"{SALT}:{person_id}".encode()).hexdigest()[:16]


def cmd_select(_args) -> int:
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT)
    sql = f"""
        select person_id, split, label, image_study_uid, image_series_uid
        from `{LABEL_TABLE}`
        where task = @task and label in (0, 1) and split is not null and image_series_uid is not null"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("task", "STRING", TASK)]))
    rows = [dict(r) for r in job.result()]
    if len({r["person_id"] for r in rows}) != len(rows):
        raise SystemExit("more than one CT row for a patient: the table's contract changed")

    by_split: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES.values()}
    id_map, missing = [], Counter()
    for r in rows:
        path = NIFTI_ROOT / f"{r['image_study_uid']}__{r['image_series_uid']}.nii.gz"
        split = SPLIT_NAMES[r["split"]]
        if not path.is_file():
            missing[split] += 1
            continue
        pid = opaque_id(r["person_id"])
        by_split[split].append({"id": pid, "label": int(r["label"])})
        id_map.append({"id": pid, "person_id": r["person_id"], "split": split, "nifti": str(path)})

    (DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    pools = {}
    for split, items in by_split.items():
        items.sort(key=lambda x: x["id"])
        with (DATA_DIR / "splits" / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x) + "\n")
        if split in POOL_SIZES:
            ids = [x["id"] for x in items]
            random.Random(f"{SALT}:pool:{split}").shuffle(ids)
            pools[split] = sorted(ids[: POOL_SIZES[split]])
    (DATA_DIR / "pools.json").write_text(json.dumps(pools), encoding="utf-8")
    with (PRIVATE_DIR / "id_map.jsonl").open("w", encoding="utf-8") as f:
        for m in id_map:
            f.write(json.dumps(m) + "\n")

    labels = {s: {x["id"]: x["label"] for x in items} for s, items in by_split.items()}
    for split, items in by_split.items():
        c = Counter(x["label"] for x in items)
        line = f"{split}: {len(items)} patients, {c[1]} died ({c[1] / max(len(items), 1):.1%}); NIfTI missing {missing[split]}"
        if split in pools:
            pc = Counter(labels[split][i] for i in pools[split])
            line += f"; pool {len(pools[split])}, {pc[1]} died ({pc[1] / len(pools[split]):.1%})"
        print(line)
    return 0


def convert(item: tuple[str, str]) -> tuple[str, str]:
    """One NIfTI -> patients/<id>/ct.npz + meta.json. Returns (status, detail) with no identifiers."""
    import nibabel as nib
    import numpy as np

    pid, src = item
    out = DATA_DIR / "patients" / pid
    if (out / "meta.json").is_file():
        return "kept", ""
    try:
        img = nib.as_closest_canonical(nib.load(src))  # RAS+, checked on a rendered slice
        hu = np.asarray(img.dataobj, dtype=np.float32)  # dataobj applies the scale and intercept
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
    id_map = {}
    with (PRIVATE_DIR / "id_map.jsonl").open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            id_map[m["id"]] = m["nifti"]
    if args.split == "test":
        wanted = [json.loads(l)["id"] for l in (DATA_DIR / "splits" / "test.jsonl").read_text().splitlines() if l]
    else:
        pools = json.loads((DATA_DIR / "pools.json").read_text())
        wanted = [i for s in ("train", "val") for i in pools[s]]
    if args.limit:
        wanted = wanted[: args.limit]
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, detail) in enumerate(pool.map(convert, [(i, id_map[i]) for i in wanted]), 1):
            status[st] += 1
            if detail:
                errors[detail] += 1
            if n % 50 == 0:
                print(f"{n}/{len(wanted)} {dict(status)}", flush=True)
    print(f"done {len(wanted)}: {dict(status)}; errors by type {dict(errors)}")
    return 1 if status["failed"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    v = sub.add_parser("volumes")
    v.add_argument("--split", choices=("pools", "test"), default="pools")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
