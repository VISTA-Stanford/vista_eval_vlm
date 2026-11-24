"""Stage the organ-zoom arms' masks (spec planning/vista_eval_vlm/2026-11-24-agent-organ-zoom.md).

    select-train  pick the 50 TRAIN find-the-slice check scans: small-finding study positives (5-9 and 10-19 mm, the
                  four tracks, round robin, seeded, one lesion per scan) whose CT and mask exist. Writes
                  zoom/splits/train_check.jsonl (ids only, no answers), zoom/train_check_dirs.json (id -> CT folder)
                  and, private, ct-findings-5-v1_7-private/zoom/train_check_items.jsonl (each scan's lesion track and
                  slab, for the find rate; never staged).
    build         for the 190 VAL scans (those with organ ranges) and the TRAIN check scans: the TotalSegmentator mask
                  on ct.npz's slab grid (a slab has an organ when any NIfTI slice averaged into it does) as
                  zoom/masks/<id>.npz ("bits": liver 1, left upper lobe 2, left lower 4, right upper 8, right middle
                  16, right lower 32; same shape as ct.npz "hu"), and zoom/organs/<id>.json (each organ's slab range,
                  the format of organ_ranges/<id>/target.json). Resumes: a built scan is skipped.
    check         $0, judges itself: every scan built, mask shape = CT shape; the CT inside the mask looks like the
                  organ on the organ's middle slab (liver median HU 0..150, lungs median below -500) on at least 95%
                  of scans; on VAL the mask's liver, lungs and right upper lobe ranges agree with organ_ranges (the
                  localization table) within 3 slabs at both ends on at least 90%; each TRAIN check lesion's slab
                  lies in its organ's range on at least 90%. Prints counts only. PASS or FAIL, exit 0 or 1.

    cd optimizer/ct-findings-5
    uv run -q --no-project --with numpy --with nibabel python -B scripts/build_masks.py select-train|build|check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

MASKS = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/organ_localization/masks_thoracic_june26")
SF_PRIVATE = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/small-finding-perception-private")
SEED = 20261124
N_TRAIN = 50
TRAIN_TRACKS = ("lung_solid", "lung_ground_glass", "liver", "lymph_node")
TRAIN_BINS = ("5-9", "10-19")
#: TotalSegmentator v2 label -> bit in "bits" (matched 2026-11-24 against the organ table: liver 5, lobes 10-14).
LABEL_BIT = {5: 1, 10: 2, 11: 4, 12: 8, 13: 16, 14: 32}
#: organs.json entries: name (as organ_ranges' target.json) -> TotalSegmentator labels.
ORGAN_LABELS = {"heart": (51,), "liver": (5,), "lung_lower_lobe_left": (11,), "lung_lower_lobe_right": (14,),
                "lung_middle_lobe_right": (13,), "lung_upper_lobe_left": (10,), "lung_upper_lobe_right": (12,),
                "lungs": (10, 11, 12, 13, 14)}

ZOOM = config.ZOOM_DIR
PRIVATE = config.ZOOM_PRIVATE


def ranges_json(present: dict[str, list[int]], n: int) -> dict:
    """target.json's format from per-organ slab lists (slab 0 = most inferior)."""
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
    """(bits array x*y*n_slabs, {organ: [slabs]}) from a canonical label volume with k NIfTI slices per slab."""
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


def _build_one(job) -> str:
    import nibabel as nib
    import numpy as np

    sid, ct_dir, nifti_name = job
    out_mask, out_org = ZOOM / "masks" / f"{sid}.npz", ZOOM / "organs" / f"{sid}.json"
    if out_mask.is_file() and out_org.is_file():
        return "kept"
    src = MASKS / nifti_name
    if not src.is_file():
        return "no mask"
    meta = json.loads((Path(ct_dir) / "meta.json").read_text())
    k = int(meta.get("slab_averaged", 1))
    shape = tuple(int(v) for v in meta["shape"])
    labels = np.asarray(nib.as_closest_canonical(nib.load(str(src))).dataobj)
    if labels.shape[:2] != shape[:2] or labels.shape[2] // k != shape[2]:
        return "shape differs"
    bits, present = slab_labels(labels, k, shape[2])
    tmp = out_mask.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, bits=bits)
    tmp.replace(out_mask)
    out_org.write_text(json.dumps(ranges_json(present, shape[2]), indent=1), encoding="utf-8")
    return "built"


def val_ids() -> list[str]:
    import adapters as ad
    return [i for i in sorted(ad.pools()["val"]) if (config.ORGAN_DIR / i / "target.json").is_file()]


def jobs() -> list[tuple[str, str, str]]:
    idm = {m["id"]: m for m in map(json.loads, (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    out = [(i, str(config.DATA_DIR / "patients" / i), Path(idm[i]["nifti"]).name) for i in val_ids()]
    dirs = json.loads((ZOOM / "train_check_dirs.json").read_text())
    names = {r["id"]: r["nifti"] for r in map(json.loads, (PRIVATE / "train_check_items.jsonl").read_text().splitlines()) if r}
    out += [(i, d, names[i]) for i, d in sorted(dirs.items())]
    return out


def cmd_select_train(_args) -> int:
    items = [json.loads(l) for l in (SF_PRIVATE / "items.jsonl").read_text().splitlines() if l]
    scans = {r["scan"]: r for r in map(json.loads, (SF_PRIVATE / "scans.jsonl").read_text().splitlines()) if r}
    cells = {(t, b): [] for t in TRAIN_TRACKS for b in TRAIN_BINS}
    for i in items:
        s = scans.get(i["scan"])
        if i["role"] != "positive" or (i["track"], i["bin"]) not in cells or not s:
            continue
        if (Path(s["ct_dir"]) / "ct.npz").is_file() and (MASKS / i["name"]).is_file():
            cells[(i["track"], i["bin"])].append(i)
    rng = random.Random(SEED)
    for v in cells.values():
        v.sort(key=lambda i: i["item"])
        rng.shuffle(v)
    picked, seen = [], set()
    while len(picked) < N_TRAIN and any(cells.values()):
        for key in sorted(cells):
            while cells[key] and cells[key][0]["scan"] in seen:
                cells[key].pop(0)
            if cells[key] and len(picked) < N_TRAIN:
                i = cells[key].pop(0)
                seen.add(i["scan"])
                picked.append(i)
    (ZOOM / "splits").mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _t, _q in config.QUESTIONS]
    (ZOOM / "splits" / "train_check.jsonl").write_text(
        "".join(json.dumps({"id": i["scan"], **{k: None for k in keys}}) + "\n" for i in picked), encoding="utf-8")
    (ZOOM / "train_check_dirs.json").write_text(
        json.dumps({i["scan"]: scans[i["scan"]]["ct_dir"] for i in picked}, indent=1), encoding="utf-8")
    (PRIVATE / "train_check_items.jsonl").write_text("".join(json.dumps(
        {"id": i["scan"], "track": i["track"], "bin": i["bin"], "slab": i["z"], "nifti": i["name"]}) + "\n"
        for i in picked), encoding="utf-8")
    print(f"TRAIN check scans: {len(picked)}; by track and bin {dict(Counter((i['track'], i['bin']) for i in picked))}")
    return 0 if len(picked) == N_TRAIN else 1


def cmd_build(args) -> int:
    (ZOOM / "masks").mkdir(parents=True, exist_ok=True)
    (ZOOM / "organs").mkdir(parents=True, exist_ok=True)
    js = jobs()
    c = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, r in enumerate(pool.map(_build_one, js), 1):
            c[r] += 1
            if n % 50 == 0:
                print(f"{n}/{len(js)} {dict(c)}", flush=True)
    print(f"done: {len(js)} scans {dict(c)}")
    return 0 if c["built"] + c["kept"] == len(js) else 1


def _check_one(job) -> dict:
    import numpy as np

    sid, ct_dir, _name = job
    r = {"id": sid, "built": False}
    try:
        bits = np.load(ZOOM / "masks" / f"{sid}.npz")["bits"]
        org = json.loads((ZOOM / "organs" / f"{sid}.json").read_text())
    except OSError:
        return r
    hu = np.load(Path(ct_dir) / "ct.npz")["hu"]
    r.update(built=True, shape_ok=bits.shape == hu.shape)
    if not r["shape_ok"]:
        return r
    for name, bit, ok in (("liver", 1, lambda v: 0 <= v <= 150), ("lungs", 62, lambda v: v < -500)):
        zs = np.nonzero(((bits & bit) > 0).any(axis=(0, 1)))[0]
        if not len(zs):
            r[f"{name}_hu_ok"] = None
            continue
        z = int(zs[len(zs) // 2])
        r[f"{name}_hu_ok"] = bool(ok(float(np.median(hu[:, :, z][(bits[:, :, z] & bit) > 0]))))
    r["ranges"] = {o["organ"]: (o["from_slice"], o["to_slice"]) for o in org["organs"] if o["present"]}
    return r


def cmd_check(args) -> int:
    js = jobs()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(_check_one, js))
    verdicts = []
    built = sum(r["built"] for r in res)
    shape = sum(r.get("shape_ok", False) for r in res)
    verdicts.append(built == len(js) and shape == len(js))
    print(f"built {built}/{len(js)}; mask shape = CT shape {shape}/{len(js)}")
    for name in ("liver", "lungs"):
        vals = [r.get(f"{name}_hu_ok") for r in res if r.get(f"{name}_hu_ok") is not None]
        share = sum(vals) / len(vals) if vals else 0.0
        verdicts.append(share >= 0.95)
        print(f"{name}: CT inside the mask looks like {name} on its middle slab: {sum(vals)}/{len(vals)} ({share:.1%})")
    vals = set(val_ids())
    agree, total = Counter(), Counter()
    for r in res:
        if r["id"] not in vals or not r.get("ranges"):
            continue
        tj = json.loads((config.ORGAN_DIR / r["id"] / "target.json").read_text())
        ref = {o["organ"]: (o["from_slice"], o["to_slice"]) for o in tj["organs"] if o.get("present")}
        for organ in ("liver", "lungs", "lung_upper_lobe_right"):
            if organ in ref and organ in r["ranges"]:
                total[organ] += 1
                agree[organ] += all(abs(a - b) <= 3 for a, b in zip(ref[organ], r["ranges"][organ]))
    for organ in ("liver", "lungs", "lung_upper_lobe_right"):
        share = agree[organ] / total[organ] if total[organ] else 0.0
        verdicts.append(share >= 0.90)
        print(f"VAL {organ} range within 3 slabs of the organ table at both ends: {agree[organ]}/{total[organ]} ({share:.1%})")
    items = {r["id"]: r for r in map(json.loads, (PRIVATE / "train_check_items.jsonl").read_text().splitlines()) if r}
    inside = 0
    by_id = {r["id"]: r for r in res}
    for sid, it in items.items():
        rg = by_id.get(sid, {}).get("ranges", {})
        organ = "liver" if it["track"] == "liver" else "lungs"
        inside += organ in rg and rg[organ][0] <= it["slab"] <= rg[organ][1]
    share = inside / len(items) if items else 0.0
    verdicts.append(share >= 0.90)
    print(f"TRAIN check lesion slab inside its organ's mask range: {inside}/{len(items)} ({share:.1%})")
    ok = all(verdicts)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select-train")
    for name in ("build", "check"):
        b = sub.add_parser(name)
        b.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    return {"select-train": cmd_select_train, "build": cmd_build, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
