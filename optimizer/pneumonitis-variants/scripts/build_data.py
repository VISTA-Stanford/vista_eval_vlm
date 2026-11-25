"""Build the pneumonitis-variants task data (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md).

Scans (spec, Decided): vista_bench v1.7 ``per_ct_pneumonitis`` positives (label 1) whose ``raw_subtype`` is one of the
four powered variants, plus the confirmed infection scans (scripts/infection_pass.py ``yes``, or the existing
``llm_labels.pneumonitis`` model label ``infectious`` strong/weak; a benchmark pneumonitis positive is never an
infection scan), each with a MerlinOnc ep99 embedding. Train and valid splits only: TEST is sealed (own go).

    select   every eligible train/valid scan -> its variant (one of five), split, opaque id and patient group;
             which have a CT volume and an organ mask; the TRAIN draw (source scans for the builder's first round, 12
             per variant, by patient; then 5 rounds of 80 fresh scans, 16 per variant, from other patients, a variant
             reused from earlier rounds once its fresh scans run out); VAL = every eligible valid scan with a volume
             and a mask. Writes splits/{train,val}.jsonl (labels), probe/{train,val}.jsonl (the probe's rows: every
             eligible scan with an embedding), rounds.json, pools.json, groups.json (id -> opaque patient),
             embeddings.npz, and the private id map. Prints counts only.
    volumes  convert each pool scan's NIfTI to patients/<id>/{ct.npz, meta.json} with ct-findings-5's converter
    masks    each pool scan's TotalSegmentator mask on ct.npz's slab grid: masks/<id>.npz and organs/<id>.json, as
             ct-findings-5 scripts/build_masks.py builds them for the organ-zoom arms
    drop-empty-masks  remove from the pools, rounds and splits any pool scan whose organ mask has no lungs (a failed
             segmentation: the zoom tool and the drawn examples need the lungs); recorded in pools.json _draw. Not
             redrawn: a redraw would reshuffle every round (2026-11-25: 1 TRAIN round-2 scan, round 2 has 79)
    reports  each TRAIN pool scan's report to train_reports/<id>.txt, for the builder only (never a VAL scan)
    check    $0, judges itself: every pool scan has its CT, mask (same shape) and organ ranges with lungs; splits,
             rounds and patients are consistent; prints counts by variant; PASS or FAIL

    cd optimizer/pneumonitis-variants
    uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with db-dtypes --with numpy python -B scripts/build_data.py select
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_data.py volumes --workers 4
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_data.py masks --workers 4
    uv run -q --no-project --with google-cloud-bigquery python -B scripts/build_data.py reports
    uv run -q --no-project --with numpy python -B scripts/build_data.py check

Prints counts only: never an id, a UID, a date or a row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

PROJECT = paths.PROJECT
DATA_DIR, PRIVATE_DIR = paths.DATA_DIR, paths.PRIVATE_DIR
TABLE = f"{PROJECT}.vista_bench_v1_7.per_ct_diagnostic_tasks"
LLM_LABELS = f"{PROJECT}.llm_labels.pneumonitis"
REPORT_TABLE = f"{PROJECT}.radiology.thoracic_ct_june26_clinical_notes"
EMBEDDINGS = Path("/mnt/su-vista-uscentral1/vistabench/embeddings/v1_7/ct/per_ct_diagnostic_tasks/"
                  "merlinonc_top_centered_pancancer_postdx_ep99_embeddings.parquet")
NIFTI_ROOT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/feb26")
MASKS = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/organ_localization/masks_thoracic_june26")
#: The five variants (answer keys), in config.QUESTIONS order, and the benchmark subtype behind each of the four.
VARIANTS = ("radiation", "immunotherapy", "other_drug", "cop", "infection")
SUBTYPE = {"radiation": "radiation", "drug_induced_ici": "immunotherapy", "drug_induced_other": "other_drug",
           "organizing_pneumonia_cop": "cop"}
SPLITS = {"train": "train", "valid": "val"}
SALT = "pneumonitis-variants-v1"
SOURCE_PER_VARIANT = 12
ROUNDS = 5
ROUND_PER_VARIANT = 16
MAX_MISSING_SHARE = 0.10


def opaque(*parts) -> str:
    return hashlib.sha1(":".join([SALT, *map(str, parts)]).encode()).hexdigest()[:16]


def nifti_name(study: str, series: str) -> str:
    return f"{study}__{series}.nii.gz"


# --------------------------------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------------------------------


def load_rows():
    """Every train/valid per_ct_pneumonitis scan that is a variant scan or an infection scan, with an embedding."""
    import pandas as pd
    import pyarrow.parquet as pq
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT)
    df = client.query(f"""
        select t.person_id, t.split, t.label, t.raw_subtype, t.ct_accession_number acc, t.image_study_uid study,
               t.image_series_uid series, l.primary_subtype_v5 llm_sub, l.pneumonitis_label llm_lab
        from `{TABLE}` t left join `{LLM_LABELS}` l on l.accession_number = t.ct_accession_number
        where t.task = 'per_ct_pneumonitis' and t.split in ('train', 'valid')
          and t.image_study_uid is not null and t.image_series_uid is not null""").to_dataframe()
    if df.duplicated(["study", "series"]).any():
        raise SystemExit("one scan has two per_ct_pneumonitis rows (or two model labels): the contract changed")
    emb = pq.read_table(EMBEDDINGS, columns=["image_study_uid", "image_series_uid", "embedding"]).to_pandas()
    vec = dict(zip(zip(emb.image_study_uid, emb.image_series_uid), emb.embedding))
    df = df[[k in vec for k in zip(df.study, df.series)]].copy()
    confirmed = {a for a, v in json.loads((PRIVATE_DIR / "infection_pass.json").read_text()).items()
                 if v.get("answer") == "yes"}
    infection = (df.acc.isin(confirmed) | ((df.llm_sub == "infectious") & df.llm_lab.isin(["strong", "weak"]))) & (df.label != 1)
    df["variant"] = None
    pos = df.label == 1
    df.loc[pos, "variant"] = df.loc[pos, "raw_subtype"].map(SUBTYPE)
    df.loc[infection, "variant"] = "infection"
    df = df[df.variant.notna()].reset_index(drop=True)
    df["split"] = df.split.map(SPLITS)
    if set(df.person_id[df.split == "train"]) & set(df.person_id[df.split == "val"]):
        raise SystemExit("a patient is in both train and valid: the split rule is broken")
    df["id"] = [opaque(p, s, r) for p, s, r in zip(df.person_id, df.study, df.series)]
    df["group"] = [opaque("patient", p) for p in df.person_id]
    return df, vec, pd


def draw_train(train, rng: random.Random) -> tuple[dict, Counter]:
    """{"source": ids, "1".."5": ids} from TRAIN scans with a volume and a mask. Source: patients in a seeded shuffle,
    one scan each, of a variant still under SOURCE_PER_VARIANT; a source patient's other scans never enter a round. Rounds:
    16 per variant, fresh first; once a variant has no fresh scan left, reused from earlier rounds, oldest first."""
    patients = sorted(set(train.group))
    rng.shuffle(patients)
    source, source_groups, have = [], set(), Counter()
    by_group = {g: train[train.group == g] for g in patients}
    for g in patients:
        rows = by_group[g]
        need = [r for r in rows.itertuples() if have[r.variant] < SOURCE_PER_VARIANT]
        if not need:
            continue
        r = need[0]
        source.append(r.id)
        have[r.variant] += 1
        source_groups.add(g)
        if all(have[v] >= SOURCE_PER_VARIANT for v in VARIANTS):
            break
    fresh = {v: [] for v in VARIANTS}
    for g in patients:
        if g in source_groups:
            continue
        for r in by_group[g].sort_values("id").itertuples():
            fresh[r.variant].append(r.id)
    for v in VARIANTS:
        rng.shuffle(fresh[v])
    rounds, used, stats = {"source": sorted(source)}, {v: [] for v in VARIANTS}, Counter()
    for k in range(1, ROUNDS + 1):
        ids = []
        for v in VARIANTS:
            take = fresh[v][:ROUND_PER_VARIANT]
            fresh[v] = fresh[v][ROUND_PER_VARIANT:]
            short = ROUND_PER_VARIANT - len(take)
            reuse = [i for i in used[v] if i not in take][:short]
            used[v] = used[v][len(reuse):] + take + reuse  # oldest first stays at the front
            stats[f"reused_{v}"] += len(reuse)
            stats[f"short_{v}"] += short - len(reuse)  # too few scans of this variant to fill the round
            ids += take + reuse
        rounds[str(k)] = sorted(ids)
    return rounds, stats


def cmd_select(_args) -> int:
    import numpy as np

    df, vec, pd = load_rows()
    df["has_volume"] = [(NIFTI_ROOT / nifti_name(s, r)).is_file() for s, r in zip(df.study, df.series)]
    df["has_mask"] = [(MASKS / nifti_name(s, r)).is_file() for s, r in zip(df.study, df.series)]
    usable = df.has_volume & df.has_mask
    for split in ("train", "val"):
        part = df[df.split == split]
        missing = 1 - usable[df.split == split].mean()
        print(f"{split}: {len(part)} eligible scans, {part.group.nunique()} patients; no volume {(~part.has_volume).sum()}, "
              f"no mask {(part.has_volume & ~part.has_mask).sum()} ({missing:.1%} unusable)")
        if missing > MAX_MISSING_SHARE:
            raise SystemExit(f"STOP: {split}: {missing:.1%} of the scans have no volume or mask; nothing written")
    train = df[(df.split == "train") & usable]
    val = df[(df.split == "val") & usable]
    rounds, stats = draw_train(train, random.Random(f"{SALT}:rounds"))
    train_pool = sorted({i for ids in rounds.values() for i in ids})

    def label_row(r) -> dict:
        return {"id": r.id, **{v: int(r.variant == v) for v in VARIANTS}}

    for sub in ("splits", "probe"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    for split, part in (("train", train[train.id.isin(train_pool)]), ("val", val)):
        rows = [label_row(r) for r in part.sort_values("id").itertuples()]
        (DATA_DIR / "splits" / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    for split in ("train", "val"):  # the probe: every eligible scan with an embedding (volume not needed)
        part = df[df.split == split] if split == "train" else val  # VAL: exactly the scans the agent reads
        rows = [{"id": r.id, "group": r.group, "variant": r.variant} for r in part.sort_values("id").itertuples()]
        (DATA_DIR / "probe" / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    ids = sorted(set(df.id))
    key = dict(zip(df.id, zip(df.study, df.series)))
    np.savez_compressed(DATA_DIR / "embeddings.npz", ids=np.array(ids),
                        emb=np.stack([np.asarray(vec[key[i]], dtype=np.float32) for i in ids]))
    (DATA_DIR / "groups.json").write_text(json.dumps(dict(sorted(zip(df.id, df.group))), indent=0))
    (DATA_DIR / "rounds.json").write_text(json.dumps(rounds, indent=1))
    (DATA_DIR / "pools.json").write_text(json.dumps({
        "val": sorted(val.id), "train": train_pool,
        "_draw": {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "salt": SALT,
                  "source_per_variant": SOURCE_PER_VARIANT, "rounds": ROUNDS, "round_per_variant": ROUND_PER_VARIANT,
                  "reused": dict(stats)}}, indent=1))
    (PRIVATE_DIR / "id_map.jsonl").write_text("".join(json.dumps(
        {"id": r.id, "person_id": int(r.person_id), "accession_number": r.acc, "split": r.split,
         "nifti": str(NIFTI_ROOT / nifti_name(r.study, r.series))}) + "\n" for r in df.sort_values("id").itertuples()))

    def by_variant(part) -> dict:
        c = Counter(part.variant)
        return {v: c[v] for v in VARIANTS}

    print(f"VAL scans {len(val)} ({val.group.nunique()} patients): {by_variant(val)}")
    print(f"TRAIN usable {len(train)}: {by_variant(train)}; probe TRAIN rows {int((df.split == 'train').sum())}")
    src = train[train.id.isin(rounds['source'])]
    print(f"source {len(src)}: {by_variant(src)}")
    for k in range(1, ROUNDS + 1):
        print(f"round {k}: {len(rounds[str(k)])} scans: {by_variant(train[train.id.isin(rounds[str(k)])])}")
    print(f"reused from earlier rounds: {dict(stats)}; TRAIN pool {len(train_pool)} distinct scans")
    return 0


# --------------------------------------------------------------------------------------------------
# volumes, masks, reports
# --------------------------------------------------------------------------------------------------


def _ct5_build_data():
    """ct-findings-5's build_data module (its converter), loaded under its own name."""
    src = Path(__file__).resolve().parents[2] / "ct-findings-5" / "scripts" / "build_data.py"
    spec = importlib.util.spec_from_file_location("ct5_build_data", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def id_map() -> dict[str, dict]:
    return {m["id"]: m for m in map(json.loads, (PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}


def pool_ids() -> list[str]:
    p = json.loads((DATA_DIR / "pools.json").read_text())
    return sorted(set(p["val"]) | set(p["train"]))


def _convert(item):
    return _ct5_build_data().convert(item)


def cmd_volumes(args) -> int:
    m = id_map()
    wanted = pool_ids()[: args.limit or None]
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, detail) in enumerate(pool.map(_convert, [(i, m[i]["nifti"], str(DATA_DIR / "patients"))
                                                              for i in wanted]), 1):
            status[st] += 1
            errors[detail] += bool(detail)
            if n % 50 == 0:
                print(f"{n}/{len(wanted)} {dict(status)}", flush=True)
    print(f"done {len(wanted)}: {dict(status)}; errors by type {dict(e for e in errors.items() if e[1])}")
    return 1 if status["failed"] else 0


#: From ct-findings-5 scripts/build_masks.py (2026-11-24, organ-zoom arms): TotalSegmentator v2 label -> bit, and the
#: organ ranges' names -> labels. Copied, not imported: that script imports ct-findings-5's config.
LABEL_BIT = {5: 1, 10: 2, 11: 4, 12: 8, 13: 16, 14: 32}
ORGAN_LABELS = {"heart": (51,), "liver": (5,), "lung_lower_lobe_left": (11,), "lung_lower_lobe_right": (14,),
                "lung_middle_lobe_right": (13,), "lung_upper_lobe_left": (10,), "lung_upper_lobe_right": (12,),
                "lungs": (10, 11, 12, 13, 14)}


def ranges_json(present: dict[str, list[int]], n: int) -> dict:
    organs = []
    for name in sorted(ORGAN_LABELS):
        zs = present.get(name) or []
        o = {"organ": name, "present": bool(zs)}
        if zs:
            lo, hi = zs[0], zs[-1]
            mid = round((lo + hi) / 2)
            f = lambda z: round(z / max(n - 1, 1), 3)  # noqa: E731
            o.update({"from_fraction": f(lo), "centre_fraction": f(mid), "to_fraction": f(hi),
                      "from_slice": lo, "centre_slice": mid, "to_slice": hi})
        organs.append(o)
    return {"available": True, "source": "TotalSegmentator organ mask, automatic; may be wrong on unusual scans",
            "axis": "superior-inferior; slice 0 and fraction 0 are the lowest (most inferior) slice", "slices": n,
            "organs": organs}


def slab_labels(labels, k: int, n_slabs: int):
    import numpy as np

    x, y, _ = labels.shape
    bits = np.zeros((x, y, n_slabs), dtype=np.uint8)
    present: dict[str, list[int]] = {name: [] for name in ORGAN_LABELS}
    for z in range(n_slabs):
        block = labels[:, :, z * k:(z + 1) * k]
        for lab, bit in LABEL_BIT.items():
            bits[:, :, z] |= (block == lab).any(axis=2).astype(np.uint8) * bit
        here = set(np.unique(block).tolist())
        for name, labs in ORGAN_LABELS.items():
            if here.intersection(labs):
                present[name].append(z)
    return bits, present


def _mask_one(job) -> str:
    import nibabel as nib
    import numpy as np

    sid, name = job
    out_mask, out_org = DATA_DIR / "masks" / f"{sid}.npz", DATA_DIR / "organs" / f"{sid}.json"
    if out_mask.is_file() and out_org.is_file():
        return "kept"
    meta_path = DATA_DIR / "patients" / sid / "meta.json"
    if not meta_path.is_file():
        return "no volume"
    meta = json.loads(meta_path.read_text())
    k, shape = int(meta.get("slab_averaged", 1)), tuple(int(v) for v in meta["shape"])
    labels = np.asarray(nib.as_closest_canonical(nib.load(str(MASKS / name))).dataobj)
    if labels.shape[:2] != shape[:2] or labels.shape[2] // k != shape[2]:
        return "shape differs"
    bits, present = slab_labels(labels, k, shape[2])
    tmp = out_mask.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, bits=bits)
    tmp.replace(out_mask)
    out_org.write_text(json.dumps(ranges_json(present, shape[2]), indent=1))
    return "built"


def cmd_masks(args) -> int:
    (DATA_DIR / "masks").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "organs").mkdir(parents=True, exist_ok=True)
    m = id_map()
    jobs = [(i, Path(m[i]["nifti"]).name) for i in pool_ids()]
    c = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, r in enumerate(pool.map(_mask_one, jobs), 1):
            c[r] += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} {dict(c)}", flush=True)
    print(f"done {len(jobs)}: {dict(c)}")
    return 0 if c["built"] + c["kept"] == len(jobs) else 1


def cmd_drop_empty_masks(_args) -> int:
    p = json.loads((DATA_DIR / "pools.json").read_text())
    rounds = json.loads((DATA_DIR / "rounds.json").read_text())
    bad = set()
    for i in pool_ids():
        org = json.loads((DATA_DIR / "organs" / f"{i}.json").read_text())
        if not any(o["organ"] == "lungs" and o["present"] for o in org["organs"]):
            bad.add(i)
    for split in ("train", "val"):
        p[split] = [i for i in p[split] if i not in bad]
        f = DATA_DIR / "splits" / f"{split}.jsonl"
        rows = [x for x in f.read_text().splitlines() if x and json.loads(x)["id"] not in bad]
        f.write_text("".join(r + "\n" for r in rows))
    by_round = {k: sum(i in bad for i in v) for k, v in rounds.items()}
    rounds = {k: [i for i in v if i not in bad] for k, v in rounds.items()}
    p["_draw"].setdefault("dropped_no_lungs_in_mask", {})
    for k, n in by_round.items():
        if n:
            p["_draw"]["dropped_no_lungs_in_mask"][k] = p["_draw"]["dropped_no_lungs_in_mask"].get(k, 0) + n
    (DATA_DIR / "pools.json").write_text(json.dumps(p, indent=1))
    (DATA_DIR / "rounds.json").write_text(json.dumps(rounds, indent=1))
    print(f"dropped {len(bad)} scans without lungs in the mask; by round {by_round}; "
          f"VAL {len(p['val'])}, TRAIN pool {len(p['train'])}")
    return 0


def cmd_reports(_args) -> int:
    from google.cloud import bigquery

    p = json.loads((DATA_DIR / "pools.json").read_text())
    train, val = set(p["train"]), set(p["val"])
    m = id_map()
    acc = {m[i]["accession_number"]: i for i in train}
    if {m[i]["accession_number"] for i in val} & set(acc):
        raise SystemExit("a VAL scan shares an accession with a TRAIN scan")
    job = bigquery.Client(project=PROJECT).query(
        f"""select accession_number, array_agg(distinct report_note_text ignore nulls) texts from `{REPORT_TABLE}`
            where accession_number in unnest(@acc) group by accession_number""",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("acc", "STRING", sorted(acc))]))
    out = DATA_DIR / "train_reports"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in job.result():
        text = "\n\n".join(t for t in r["texts"] if t and t.strip())
        if text and acc[r["accession_number"]] in train:
            (out / f"{acc[r['accession_number']]}.txt").write_text(text)
            n += 1
    print(f"TRAIN pool scans {len(train)}; reports written {n}; without a report {len(train) - n}")
    return 0


# --------------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------------


def cmd_check(_args) -> int:
    import numpy as np

    ok = []
    p = json.loads((DATA_DIR / "pools.json").read_text())
    rounds = json.loads((DATA_DIR / "rounds.json").read_text())
    groups = json.loads((DATA_DIR / "groups.json").read_text())
    rows = {s: [json.loads(x) for x in (DATA_DIR / "splits" / f"{s}.jsonl").read_text().splitlines() if x]
            for s in ("train", "val")}
    ok.append(all(sum(r[v] for v in VARIANTS) == 1 for s in rows.values() for r in s))
    print(f"every scan has exactly one variant: {ok[-1]}")
    tg, vg = {groups[r['id']] for r in rows['train']}, {groups[r['id']] for r in rows['val']}
    ok.append(not tg & vg and {r["id"] for r in rows["val"]} == set(p["val"])
              and {r["id"] for r in rows["train"]} == set(p["train"]))
    print(f"no patient in both TRAIN and VAL, splits match the pools: {ok[-1]}")
    src_groups = {groups[i] for i in rounds["source"]}
    ok.append(not any(groups[i] in src_groups for k in range(1, ROUNDS + 1) for i in rounds[str(k)]))
    print(f"no round scan shares a patient with a source scan: {ok[-1]}")
    bad = Counter()
    for i in pool_ids():
        try:
            hu_shape = tuple(json.loads((DATA_DIR / "patients" / i / "meta.json").read_text())["shape"])
            with np.load(DATA_DIR / "masks" / f"{i}.npz") as z:
                if z["bits"].shape != hu_shape:
                    bad["mask shape differs"] += 1
            org = json.loads((DATA_DIR / "organs" / f"{i}.json").read_text())
            if not any(o["organ"] == "lungs" and o["present"] for o in org["organs"]):
                bad["no lungs in organ ranges"] += 1
            if not (DATA_DIR / "patients" / i / "ct.npz").is_file():
                bad["no ct.npz"] += 1
        except (OSError, KeyError, ValueError):
            bad["missing a file"] += 1
    ok.append(not bad)
    print(f"pool scans {len(pool_ids())}; problems {dict(bad)}")
    reports = sum((DATA_DIR / "train_reports" / f"{i}.txt").is_file() for i in p["train"])
    leaked = sum((DATA_DIR / "train_reports" / f"{i}.txt").is_file() for i in p["val"])
    ok.append(leaked == 0)
    print(f"TRAIN reports {reports}/{len(p['train'])}; VAL reports {leaked} (must be 0)")
    for s, rs in rows.items():
        print(f"{s}: {len(rs)} scans " + str({v: sum(r[v] for r in rs) for v in VARIANTS}))
    print("PASS" if all(ok) else "FAIL")
    return 0 if all(ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    for name in ("volumes", "masks"):
        b = sub.add_parser(name)
        b.add_argument("--workers", type=int, default=4)
        b.add_argument("--limit", type=int, default=0)
    sub.add_parser("reports")
    sub.add_parser("drop-empty-masks")
    sub.add_parser("check")
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes, "masks": cmd_masks, "reports": cmd_reports, "drop-empty-masks": cmd_drop_empty_masks,
            "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
