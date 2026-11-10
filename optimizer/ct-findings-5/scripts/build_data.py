"""Build the ct-findings-5 task data: one CT per patient from the benchmark's train and valid splits, the five
answers for each scan, the fixed enriched pools, and each pool scan's volume.

The five questions are vista_bench v1.7 ``per_ct_diagnostic_tasks`` (report-derived, one row per scan and task):
``per_ct_lesion_liver``, ``per_ct_lesion_lungs``, ``per_ct_lesion_lymph_nodes``, ``per_ct_lung_lobe_rul`` and
``per_ct_pneumonitis`` (spec planning/vista_eval_vlm/2026-11-10-ct-agent-probability-climbs.md). Only the table's
train and valid rows are ever read: TEST is sealed. A label of -1, or no row for a task, leaves that question
unscored for that scan (``None``), never negative.

    select   query the train/valid rows of the five tasks, draw the pools (VAL 200 from valid, TRAIN 400 from train,
             one scan per patient, positives over-sampled so each question has at least FLOOR positives and FLOOR
             negatives per 200 scans), check each drawn scan's volume exists (a scan without one is dropped,
             counted and replaced; over 10% dropped stops the build), check no patient of the VLMs' TEST sample is
             in either pool or anywhere in the table's train/valid rows, and write splits/{train,val}.jsonl,
             pools.json and the private id map. Prints counts only.
    volumes  convert each pool scan's NIfTI into patients/<id>/ct.npz + meta.json, exactly as died-2yr-ct does
             (Hounsfield units, int16, RAS, thin scans averaged into ~2.5 mm slabs, no header, date or UID kept)

    cd optimizer/ct-findings-5
    OMP_NUM_THREADS=2 nice -n 19 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow python -B scripts/build_data.py select
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_data.py volumes --workers 4 [--limit 10]

BigQuery runs on this host's application-default login. Prints counts only: never an id, a UID, a date or a row.
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
TABLE = f"{PROJECT}.vista_bench_v1_7.per_ct_diagnostic_tasks"
#: (answer key, vista_bench task), in config.QUESTIONS order.
QUESTIONS = (("lesion_liver", "per_ct_lesion_liver"), ("lesion_lungs", "per_ct_lesion_lungs"),
             ("lesion_lymph_nodes", "per_ct_lesion_lymph_nodes"), ("lung_lobe_rul", "per_ct_lung_lobe_rul"),
             ("pneumonitis", "per_ct_pneumonitis"))
#: The same mount folder the VLMs' TEST sample's volumes come from (spec, Decided).
NIFTI_ROOT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/feb26")
#: The VLMs' TEST sample (1,000 scans): none of its patients may be in TRAIN or VAL. Only person_id is read.
TEST_SAMPLE = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/samples/step2.parquet")
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "ct-findings-5-v1_7"))
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "ct-findings-5-v1_7-private"))
SALT = "ct-findings-5-v1"
#: The table's split names -> the engine's. TEST is never queried.
SPLIT_NAMES = {"train": "train", "valid": "val"}
#: Fixed pools the data schedule draws from: VAL the 200 every round scores, TRAIN up to 400 (the schedule's last round).
POOL_SIZES = {"val": 200, "train": 400}
#: Enrichment target, per 200 scans of a pool, for each question and each class: the spec's floor is 20 positives on
#: VAL-200; the draw aims at 30 for a margin, and also guards the negatives (lungs is mostly positive).
TARGET_PER_200 = 30
FLOOR_PER_200 = 20
#: Stop if more than this share of the drawn scans have no volume (spec, Decided).
MAX_MISSING_SHARE = 0.10
SLAB_MM = 2.5


def opaque_id(person_id: int, study: str, series: str) -> str:
    return hashlib.sha1(f"{SALT}:{person_id}:{study}:{series}".encode()).hexdigest()[:16]


def nifti(study: str, series: str) -> Path:
    return NIFTI_ROOT / f"{study}__{series}.nii.gz"


def load_scans(client) -> dict[str, dict]:
    """Every train/valid scan with a row for at least one of the five tasks: {scan key: {person_id, split, study,
    series, accession, labels: {key: 0 | 1 | None}}}. Stops if one scan carries two labels for a task."""
    from google.cloud import bigquery

    sql = f"""
        select person_id, split, task, label, ct_accession_number, image_study_uid, image_series_uid
        from `{TABLE}`
        where task in unnest(@tasks) and split in ('train', 'valid')
          and image_study_uid is not null and image_series_uid is not null"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tasks", "STRING", [t for _k, t in QUESTIONS])]))
    key_of = {t: k for k, t in QUESTIONS}
    scans: dict[str, dict] = {}
    for r in job.result():
        sk = f"{r['image_study_uid']}__{r['image_series_uid']}"
        s = scans.setdefault(sk, {"person_id": r["person_id"], "split": SPLIT_NAMES[r["split"]],
                                  "study": r["image_study_uid"], "series": r["image_series_uid"],
                                  "accession": r["ct_accession_number"], "labels": dict.fromkeys(key_of.values())})
        if s["person_id"] != r["person_id"] or s["split"] != SPLIT_NAMES[r["split"]]:
            raise SystemExit("one scan under two patients or two splits: the table's contract changed")
        q = key_of[r["task"]]
        value = int(r["label"]) if r["label"] in (0, 1) else None
        if s["labels"][q] is not None and value is not None and s["labels"][q] != value:
            raise SystemExit("one scan with two different labels for one task: the table's contract changed")
        s["labels"][q] = value if value is not None else s["labels"][q]
    return {k: s for k, s in scans.items() if any(v is not None for v in s["labels"].values())}


def draw_pool(scans: list[dict], size: int, rng: random.Random, have_volume) -> tuple[list[dict], Counter]:
    """One scan per patient, ``size`` scans. First, while some (question, class) is under its target, take the next
    patient in a seeded shuffle who has a scan with that answer, and one such scan of theirs at random; then fill
    with the next patients in the shuffle, one random scan each. A drawn scan with no volume is dropped, counted and
    replaced by the patient's next candidate (or the next patient). Returns (picked scans, counts)."""
    target = TARGET_PER_200 * size // 200
    by_patient: dict[int, list[dict]] = {}
    for s in scans:
        by_patient.setdefault(s["person_id"], []).append(s)
    order = sorted(by_patient)
    rng.shuffle(order)
    for p in order:
        by_patient[p].sort(key=lambda s: (s["study"], s["series"]))
        rng.shuffle(by_patient[p])
    picked: list[dict] = []
    taken: set[int] = set()
    stats = Counter()

    def take(p: int, ok) -> bool:
        for s in by_patient[p]:
            if not ok(s):
                continue
            stats["drawn"] += 1
            if not have_volume(s):
                stats["no_volume"] += 1
                continue
            picked.append(s)
            taken.add(p)
            return True
        return False

    def counts() -> Counter:
        return Counter((q, s["labels"][q]) for s in picked for q, _t in QUESTIONS if s["labels"][q] is not None)

    needs = [(q, c) for q, _t in reversed(QUESTIONS) for c in (1, 0)]  # pneumonitis (rarest) first
    progress = True
    while progress and len(picked) < size:
        progress = False
        have = counts()
        for q, c in sorted(needs, key=lambda n: have[n]):  # the scarcest need first
            if have[(q, c)] >= target:
                continue
            for p in order:
                if p not in taken and take(p, lambda s, q=q, c=c: s["labels"][q] == c):
                    progress = True
                    stats[f"enriched_{q}_{c}"] += 1
                    break
            if progress:
                break
    stats["enrichment_scans"] = len(picked)
    for p in order:
        if len(picked) >= size:
            break
        if p not in taken:
            take(p, lambda s: True)
    return picked, stats


def cmd_select(_args) -> int:
    import pandas as pd
    from google.cloud import bigquery

    scans = load_scans(bigquery.Client(project=PROJECT))
    test_patients = set(pd.read_parquet(TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    tv_patients = {s["person_id"] for s in scans.values()}
    by_split: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES.values()}
    for s in scans.values():
        by_split[s["split"]].append(s)
    if {s["person_id"] for s in by_split["train"]} & {s["person_id"] for s in by_split["val"]}:
        raise SystemExit("a patient is in both the train and valid splits: the split rule is broken")

    checked: dict[str, bool] = {}

    def have_volume(s: dict) -> bool:
        k = f"{s['study']}__{s['series']}"
        if k not in checked:
            checked[k] = nifti(s["study"], s["series"]).is_file()
        return checked[k]

    rows, pools, id_map, stats_out = {}, {}, [], {}
    for split in ("val", "train"):  # fixed order: each split's draw has its own seeded generator
        seed = f"{SALT}:pool:{split}"
        picked, stats = draw_pool(by_split[split], POOL_SIZES[split], random.Random(seed), have_volume)
        if stats["drawn"] and stats["no_volume"] / stats["drawn"] > MAX_MISSING_SHARE:
            raise SystemExit(f"STOP: {split}: {stats['no_volume']} of {stats['drawn']} drawn scans have no volume "
                             f"(over {MAX_MISSING_SHARE:.0%}); nothing written")
        if len(picked) < POOL_SIZES[split]:
            raise SystemExit(f"STOP: {split}: only {len(picked)} scans could be drawn; nothing written")
        if len({s["person_id"] for s in picked}) != len(picked):
            raise SystemExit(f"{split}: two scans of one patient in the pool")
        rows[split] = []
        for s in picked:
            sid = opaque_id(s["person_id"], s["study"], s["series"])
            rows[split].append({"id": sid, **s["labels"]})
            id_map.append({"id": sid, "person_id": s["person_id"], "accession_number": s["accession"],
                           "split": split, "nifti": str(nifti(s["study"], s["series"]))})
        rows[split].sort(key=lambda r: r["id"])
        pools[split] = [r["id"] for r in rows[split]]
        stats_out[split] = {"seed": seed, "scans_in_split": len(by_split[split]),
                            "patients_in_split": len({s["person_id"] for s in by_split[split]}), "no_volume": 0, **stats}
    pool_patients = {m["person_id"] for m in id_map}
    if pool_patients & test_patients:
        raise SystemExit(f"STOP: {len(pool_patients & test_patients)} pool patients are in the VLMs' TEST sample")
    if tv_patients & test_patients:
        raise SystemExit(f"STOP: {len(tv_patients & test_patients)} train/valid patients are in the VLMs' TEST sample")

    (DATA_DIR / "splits").mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (DATA_DIR / "splits" / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items),
                                                            encoding="utf-8")
    counts = {split: {q: {"positive": sum(r[q] == 1 for r in items), "negative": sum(r[q] == 0 for r in items),
                          "unscored": sum(r[q] is None for r in items)} for q, _t in QUESTIONS}
              for split, items in rows.items()}
    (DATA_DIR / "pools.json").write_text(json.dumps({
        **pools,
        "_draw": {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "table": TABLE,
                  "salt": SALT, "one_scan_per_patient": True,
                  "enrichment": (f"per (question, class), in a seeded patient shuffle, take the next patient with a "
                                 f"scan having that answer until each has {TARGET_PER_200} per 200 scans (scarcest "
                                 f"first), then fill with the next patients, one random scan each"),
                  "target_per_200": TARGET_PER_200, "floor_per_200": FLOOR_PER_200,
                  "test_sample_patients": len(test_patients), "test_sample_overlap": 0,
                  "per_split": stats_out, "counts": counts}}, indent=1), encoding="utf-8")
    (PRIVATE_DIR / "id_map.jsonl").write_text("".join(json.dumps(m) + "\n" for m in id_map), encoding="utf-8")

    print(f"VLM TEST sample: {len(test_patients)} patients, 0 in the train/valid rows or the pools")
    for split, items in rows.items():
        st = stats_out[split]
        print(f"{split}: {len(items)} scans of {st['scans_in_split']} ({st['patients_in_split']} patients); seed "
              f"{st['seed']!r}; drawn {st['drawn']}, no volume {st['no_volume']} ({st['no_volume'] / st['drawn']:.1%}); "
              f"{st['enrichment_scans']} picked by enrichment")
        floor = FLOOR_PER_200 * len(items) // 200
        for q, _t in QUESTIONS:
            c = counts[split][q]
            flag = "" if min(c["positive"], c["negative"]) >= floor else f"  BELOW FLOOR {floor}"
            print(f"  {q:<20} positive {c['positive']:>3}  negative {c['negative']:>3}  unscored {c['unscored']:>3}{flag}")
    return 0


def convert(item: tuple[str, str]) -> tuple[str, str]:
    """One NIfTI -> patients/<id>/ct.npz + meta.json (died-2yr-ct's converter). Returns (status, detail), no ids."""
    import nibabel as nib
    import numpy as np

    pid, src = item
    out = DATA_DIR / "patients" / pid
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
    id_map = {m["id"]: m["nifti"] for m in map(json.loads, (PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    pools = json.loads((DATA_DIR / "pools.json").read_text())
    wanted = [i for s in ("val", "train") for i in pools[s]]
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
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
