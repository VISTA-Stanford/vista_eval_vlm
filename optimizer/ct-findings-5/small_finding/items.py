"""Build the small-finding perception items (spec planning/vista_eval_vlm/2026-11-23-small-finding-perception.md).

Positives come from the existing Gemini findings extraction (radiology-findings-v1, flash38-off_ct-v1-prog-cite): per
lesion its organ, current size (kept apart from the prior size), size axis and cited slice. A lesion is an item when
it is in one of four tracks (solid lung nodule, ground-glass nodule, liver lesion, thoracic lymph node), states one
current size, cites an image in the series we have a volume for, and that image lands inside the organ's slice range.
Negatives are other TRAIN patients whose report says the organ is clear and whose extraction lists no lesion there,
shown at the same height inside the organ as the positive they are matched to. TRAIN only; VAL and TEST untouched.

    cd optimizer/ct-findings-5
    nice -n 19 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pydicom --with nibabel \
        --with numpy python -B small_finding/items.py build

Writes PRIVATE/items.jsonl and PRIVATE/scans.jsonl (ids, paths and report-derived wording: private folder only) and
OUT/item_counts.json (counts only). Prints counts only: never an id, a date or report text.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
OUT = Path(os.environ.get("SF_OUT", MOUNT / "small-finding-perception"))
PRIVATE = Path(os.environ.get("SF_PRIVATE", MOUNT / "small-finding-perception-private"))
SCANS_DIR = OUT / "scans"  # volumes this study converts: scans/<scan id>/{ct.npz, meta.json}
ANSWERS = MOUNT / "radiology-findings-v1" / "flash38-off_ct-v1-prog-cite" / "answers.jsonl"
DICOM_META = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/dicom_metadata")
LOCALIZATION = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/organ_localization/thoracic_june26_localization.csv")
SALT = "small-finding-perception-v1"
SEED = 20261123

TRACKS = ("lung_solid", "lung_ground_glass", "liver", "lymph_node")
BINS = ("<5", "5-9", "10-19", ">=20")
TRACK_BINS = {t: BINS[1:] if t == "lymph_node" else BINS for t in TRACKS}  # no <5 mm node bin (spec)
PER_BIN = 40       # positives per track and bin, each with one matched negative (spec)
EXAMPLES_PER_BIN = 10  # layer 2 reference pool per track and bin; each target is shown three of them
SLAB_MM = 2.5

# --- pure helpers (tested in tests/test_small_finding.py) -------------------------------------------------------

_CITE = (
    re.compile(r"^\s*\(?\s*(\d{1,4})\s*[/:]\s*(\d{1,4})\s*\)?"),                      # "4/56", "4:56", "(4/56)"
    re.compile(r"series\s*#?\s*(\d{1,4})\s*[,;]?\s*(?:image|img|im)\s*#?\s*(\d{1,4})", re.I),
)
_CITE_REV = re.compile(r"(?:image|img|im)\s*#?\s*(\d{1,4})\s*(?:of|on|in|,)?\s*series\s*#?\s*(\d{1,4})", re.I)


def parse_cite(text) -> tuple[int, int] | None:
    """(series, image) from the extraction's current_slice_number, or None when it names no series (we keep only
    citations that name the series, so the image can be placed on the volume we have)."""
    s = str(text or "")
    for rx in _CITE:
        m = rx.search(s)
        if m:
            return int(m.group(1)), int(m.group(2))
    m = _CITE_REV.search(s)
    return (int(m.group(2)), int(m.group(1))) if m else None


_NUM = r"\d{1,3}(?:\.\d+)?"
_SIZE = re.compile(rf"^\s*({_NUM})(?:\s*(mm|cm))?(?:\s*(?:x|by|×)\s*({_NUM})(?:\s*(mm|cm))?)?"
                   rf"(?:\s*(?:x|by|×)\s*({_NUM})(?:\s*(mm|cm))?)?\s*[-‐]?\s*(mm|cm)?\s*$", re.I)


def parse_size(text) -> list[float] | None:
    """The stated dimensions in mm for ONE current measurement ("8 mm", "1.2 x 0.9 cm", "4-mm"); None for anything
    else: no size, a range ("3-4 mm"), a bound ("up to 6 mm", "<5 mm"), words ("subcentimeter"), or two sizes."""
    s = str(text or "").strip().lower()
    if not s or re.search(r"up to|less than|<|>|sub|tiny|small|several|multiple|and|,|;|\d\s*-\s*\d", s):
        return None
    m = _SIZE.match(s)
    if not m:
        return None
    nums = [m.group(i) for i in (1, 3, 5) if m.group(i)]
    units = [u for u in (m.group(2), m.group(4), m.group(6), m.group(7)) if u]
    if not units:
        return None
    if len(set(units)) > 1:
        return None
    k = 10.0 if units[0] == "cm" else 1.0
    return [float(n) * k for n in nums]


def lesion_mm(dims: list[float], track: str, axis: str) -> float:
    """Size used for binning: the largest dimension; for a lymph node the short axis when the sentence gives one
    (two dimensions read as long x short, or the extraction says short), else the largest (spec)."""
    if track == "lymph_node" and (axis == "short" or (axis == "long_x_short" and len(dims) >= 2)):
        return min(dims)
    return max(dims)


def bin_of(mm: float) -> str:
    return "<5" if mm < 5 else "5-9" if mm < 10 else "10-19" if mm < 20 else ">=20"


def nifti_z(n_slices: int, image: int, inst_mid: int) -> float:
    """The cited image's slice on the NIfTI's canonical z axis (RAS, 0 = most inferior). The middle slice's DICOM
    header sits on the volume's centre slice; image numbers run head to foot (measured 2026-11-23: liver-citing
    sentences land in the liver 93% of the time this way, 6% the other way)."""
    return (n_slices - 1) / 2 - (image - inst_mid)


def slab_factor(z_spacing_mm: float) -> int:
    """How many NIfTI slices build_data.convert averages into one ct.npz slab."""
    return max(1, int(round(SLAB_MM / z_spacing_mm))) if z_spacing_mm < SLAB_MM else 1


def npz_z(z: float, k: int) -> int:
    return int(round(z)) // k


_GG = re.compile(r"ground[- ]?glass|\bggo\b|\bggn\b")
_SOLIDISH = re.compile(r"part[- ]?solid|solid component|mixed")
_NOT_NODULE = re.compile(r"cyst|bulla|bleb|scar|atelecta|consolidat|effusion|bronchiectas|emphysema|fibros|thicken")
_THORACIC_NODE = re.compile(r"mediastin|hil(ar|um)|paratrache|subcarin|prevascular|aortopulmonary|\bap window|"
                            r"precarinal|paraesophag|axill|supraclavic|internal mammary|cardiophrenic|"
                            r"pretrache|retrocrural|station|\b(2|4|5|6|7|8|10|11)[rl]?\b")


def track_of(lesion: dict) -> str | None:
    organ = str(lesion.get("organ") or "").lower()
    desc = " ".join(str(lesion.get(k) or "") for k in ("ct_descriptor", "location")).lower()
    morph = " ".join(map(str, lesion.get("ct_morphology") or [])).lower()
    if organ == "lungs":
        if _NOT_NODULE.search(desc):
            return None
        if _GG.search(desc + " " + morph) and not _SOLIDISH.search(desc + " " + morph):
            return "lung_ground_glass"
        return "lung_solid"
    if organ == "liver":
        return "liver"
    if organ in ("lymph nodes", "lymph_nodes"):
        return "lymph_node" if _THORACIC_NODE.search(desc) else None
    return None


_LUNG_NEG = re.compile(r"\bno (?:(?:suspicious|discrete|focal|definite|dominant|significant|concerning|worrisome|new)\s+)?"
                       r"(?:pulmonary|lung|parenchymal)\s+(?:nodules?|masses?)\b")
_LIVER_NEG = re.compile(r"\bliver\s+(?:is\s+|appears\s+)?(?:unremarkable|normal|within normal limits)\b|"
                        r"\bno (?:(?:focal|suspicious|discrete)\s+)?(?:hepatic|liver)\s+(?:lesions?|masses?)\b|"
                        r"\bunremarkable liver\b")
_NODE_NEG = re.compile(r"\bno (?:(?:thoracic|mediastinal|hilar|axillary|intrathoracic|supraclavicular|significant|"
                       r"pathologic(?:al)?|enlarged)\s+(?:or\s+\w+\s+)?)*(?:lymphadenopathy|adenopathy|lymph nodes)\b")
NEG_RX = {"lung_solid": _LUNG_NEG, "lung_ground_glass": _LUNG_NEG, "liver": _LIVER_NEG, "lymph_node": _NODE_NEG}
NEG_ORGAN = {"lung_solid": "lungs", "lung_ground_glass": "lungs", "liver": "liver", "lymph_node": "lymph nodes"}


def side_of(location: str) -> str | None:
    """'right' | 'left' from a lesion's location wording, for the secondary where-agreement score."""
    s = str(location or "").lower()
    r = bool(re.search(r"\bright\b|\brul\b|\brml\b|\brll\b", s))
    l = bool(re.search(r"\bleft\b|\blul\b|\blll\b|lingul", s))
    return "right" if r and not l else "left" if l and not r else None


def opaque(*parts) -> str:
    import hashlib
    return hashlib.sha1(":".join([SALT, *map(str, parts)]).encode()).hexdigest()[:16]


# --- organ ranges (TotalSegmentator localization, canonical z fractions) -----------------------------------------

def organ_ranges(names: set[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """{nifti file name: {'lungs' | 'liver' | 'chest_soft': (lo, hi)}} as fractions of the z axis (0 = inferior).
    chest_soft is bottom of the heart to top of the lungs (arm D's soft-tissue grid: the node question)."""
    import pandas as pd

    loc = pd.read_csv(LOCALIZATION, usecols=["nifti_filename", "structure", "dim_2_min_fraction",
                                             "dim_2_max_fraction", "present"])
    loc["name"] = loc.nifti_filename.map(lambda x: Path(str(x)).name)
    loc = loc[loc.name.isin(names) & loc.structure.isin(["lungs", "liver", "heart"])]
    loc = loc[loc.present.astype(str).str.lower().isin(["true", "1"])].dropna(subset=["dim_2_min_fraction"])
    out: dict[str, dict] = defaultdict(dict)
    for r in loc.itertuples():
        out[r.name][r.structure] = (float(r.dim_2_min_fraction), float(r.dim_2_max_fraction))
    for d in out.values():
        if "heart" in d and "lungs" in d:
            d["chest_soft"] = (d["heart"][0], d["lungs"][1])
    return out


RANGE_OF = {"lung_solid": "lungs", "lung_ground_glass": "lungs", "liver": "liver", "lymph_node": "chest_soft"}


# --- build ------------------------------------------------------------------------------------------------------

def _header(study: str, series: str):
    import io
    import pydicom
    try:
        b = (DICOM_META / f"{study}__{series}.dcm").read_bytes()[:65536]
        d = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True, force=True,
                            specific_tags=["SeriesNumber", "InstanceNumber", "ImagePositionPatient"])
        sn, inn, ipp = d.get("SeriesNumber"), d.get("InstanceNumber"), d.get("ImagePositionPatient")
        return ((None if sn is None else int(sn)), (None if inn is None else int(inn)),
                (None if ipp is None else float(ipp[2])))
    except Exception:  # noqa: BLE001 - a missing header drops the scan; counted
        return None, None, None


def _nifti_geom(path: str):
    """(slice count along the superior axis, z spacing in mm, world z in mm of the volume's centre) without
    reading voxels."""
    import nibabel as nib
    import numpy as np
    try:
        img = nib.load(path)
        ax = int(np.where(nib.io_orientation(img.affine)[:, 0] == 2)[0][0])
        shape = np.array(img.header.get_data_shape()[:3], dtype=float)
        zc = float(img.affine[2, :3] @ ((shape - 1) / 2) + img.affine[2, 3])
        return int(shape[ax]), float(img.header.get_zooms()[ax]), zc
    except Exception:  # noqa: BLE001
        return None, None, None


def centre_offset(z_mm_mid: float | None, zc_mm: float | None, dz: float) -> float | None:
    """Where the middle slice's DICOM header sits relative to the volume's centre, in slices (measured 2026-11-23 on
    150 scans: exactly 0 when the slice count is odd, +0.5 when it is even). None when it is more than one slice
    off (slices missing from one end: counting images from the centre would land on the wrong slice), and such
    scans are dropped as positives."""
    if z_mm_mid is None or zc_mm is None or not dz:
        return None
    off = (z_mm_mid - zc_mm) / dz
    return off if abs(off) <= 1.0 else None


def _existing_volumes() -> dict[str, Path]:
    """NIfTI file name -> folder of an existing ct.npz (the ct-findings-5 TRAIN pool and TRAIN neighbours)."""
    import build_data as bd
    out = {}
    for m in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()):
        if m and m["split"] == "train":
            out[Path(m["nifti"]).name] = bd.DATA_DIR / "patients" / m["id"]
    for ps in json.loads((bd.PRIVATE_DIR / "neighbour_picks.json").read_text()).values():
        for p in ps:
            out[Path(p["nifti"]).name] = bd.DATA_DIR / "neighbour_scans" / p["nid"]
    return {k: v for k, v in out.items() if (v / "meta.json").is_file()}


def cmd_build(_args) -> int:
    from concurrent.futures import ThreadPoolExecutor
    from google.cloud import bigquery
    import build_data as bd
    import build_neighbours as bn

    client = bigquery.Client(project=bd.PROJECT)
    train, val_patients, _ = bn.the_split(client)
    counts = bn.report_counts(client, [s["accession"] for s in train if s["accession"]])
    single = [s for s in train if s["accession"] and counts.get(s["accession"], 0) == 1
              and s["person_id"] not in val_patients]
    with ThreadPoolExecutor(32) as ex:
        have = list(ex.map(lambda s: bd.nifti(s["study"], s["series"]).is_file(), single))
    scans = {s["accession"]: s for s, h in zip(single, have) if h}
    print(f"TRAIN single-report scans with a NIfTI: {len(scans)}")

    # 1. lesions from the extraction
    c = Counter()
    raw, lesion_free = [], defaultdict(set)  # lesion_free[track] = accessions with no lesion of that organ
    for line in ANSWERS.open(encoding="utf-8"):
        r = json.loads(line)
        acc = r.get("accession_number")
        if r.get("status") != "ok" or acc not in scans:
            continue
        c["reports"] += 1
        try:
            les = r["data"]["lesions"]["lesions_json"]
            les = json.loads(les) if isinstance(les, str) else (les or [])
        except Exception:  # noqa: BLE001
            c["unreadable lesion list"] += 1
            continue
        # a negative lists no lesion of the kind tested (a report with "no pulmonary nodules" and some scarring is
        # still a clean nodule negative); both lung tracks share the lung-nodule test
        present = {track_of(l) for l in les}
        lung_nodule = bool(present & {"lung_solid", "lung_ground_glass"})
        for t in TRACKS:
            if not (lung_nodule if t.startswith("lung") else t in present):
                lesion_free[t].add(acc)
        for l in les:
            t = track_of(l)
            if not t:
                continue
            c[f"{t}: lesions"] += 1
            dims = parse_size(l.get("size_current"))
            cite = parse_cite(l.get("current_slice_number"))
            if not dims:
                c[f"{t}: no single current size"] += 1
                continue
            if not cite:
                c[f"{t}: no series/image citation"] += 1
                continue
            mm = lesion_mm(dims, t, str(l.get("size_axis") or ""))
            if bin_of(mm) not in TRACK_BINS[t]:
                c[f"{t}: outside bins"] += 1
                continue
            raw.append({"acc": acc, "track": t, "mm": mm, "bin": bin_of(mm), "series_no": cite[0], "image": cite[1],
                        "desc": f"{l.get('size_current')} {l.get('ct_descriptor') or ''}".strip(),
                        "location": str(l.get("location") or ""), "side": side_of(l.get("location"))})
    print(f"reports joined to the extraction: {c['reports']}")

    # 2. headers and geometry for the scans that matter
    need = sorted({x["acc"] for x in raw})
    with ThreadPoolExecutor(64) as ex:
        H = dict(zip(need, ex.map(lambda a: _header(scans[a]["study"], scans[a]["series"]), need)))
    keep = [x for x in raw if H[x["acc"]][0] is not None and H[x["acc"]][0] == x["series_no"] and H[x["acc"]][1]]
    for x in raw:
        if x not in keep:
            c[f"{x['track']}: cites another series (or no header)"] += 1
    geo_need = sorted({x["acc"] for x in keep})
    with ThreadPoolExecutor(32) as ex:
        G = dict(zip(geo_need, ex.map(lambda a: _nifti_geom(str(bd.nifti(scans[a]["study"], scans[a]["series"]))),
                                      geo_need)))
    names = {bd.nifti(s["study"], s["series"]).name: a for a, s in scans.items()}
    R = organ_ranges(set(names))

    pos = []
    liver_hit = Counter()
    for x in keep:
        a = x["acc"]
        n, dz, zc = G.get(a, (None, None, None))
        name = bd.nifti(scans[a]["study"], scans[a]["series"]).name
        if not n:
            c[f"{x['track']}: unreadable NIfTI"] += 1
            continue
        off = centre_offset(H[a][2], zc, dz)
        if off is None:
            c[f"{x['track']}: middle header more than a slice off the centre"] += 1
            continue
        z = nifti_z(n, x["image"], H[a][1]) + off
        if not 0 <= z <= n - 1:
            c[f"{x['track']}: image off the volume"] += 1
            continue
        rng = R.get(name, {}).get(RANGE_OF[x["track"]])
        if not rng:
            c[f"{x['track']}: no organ range"] += 1
            continue
        frac = z / (n - 1)
        inside = rng[0] <= frac <= rng[1]
        if x["track"] == "liver":
            liver_hit["inside" if inside else "outside"] += 1
            zf = (n - 1) / 2 + (x["image"] - H[a][1])  # the other direction, as a check on the sign
            liver_hit["other direction inside"] += int(0 <= zf <= n - 1 and rng[0] <= zf / (n - 1) <= rng[1])
        if not inside:
            c[f"{x['track']}: image outside the organ"] += 1
            continue
        k = slab_factor(dz)
        if npz_z(z, k) >= n // k:  # the converter drops a last partial slab
            c[f"{x['track']}: image in the dropped last slab"] += 1
            continue
        pos.append({**x, "name": name, "n": n, "k": k, "z": npz_z(z, k),
                    "organ_frac": (frac - rng[0]) / max(rng[1] - rng[0], 1e-6)})

    # 3. negatives: report says the organ is clear AND the extraction lists no lesion there
    neg_accs = {a for t in TRACKS for a in lesion_free[t]}
    sql = f"""select accession_number a, any_value(lower(report_note_text)) t from `{bd.REPORT_TABLE}`
      where accession_number in unnest(@acc) and trim(report_note_text) != '' group by a"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted(neg_accs))]))
    negs = defaultdict(list)
    for row in job.result():  # report text stays in memory, never written or printed
        for t in TRACKS:
            if row["a"] in lesion_free[t] and NEG_RX[t].search(row["t"] or ""):
                negs[t].append(row["a"])
    neg_geo = sorted({a for t in TRACKS for a in negs[t]} - set(G))
    with ThreadPoolExecutor(32) as ex:
        G.update(zip(neg_geo, ex.map(lambda a: _nifti_geom(str(bd.nifti(scans[a]["study"], scans[a]["series"]))),
                                     neg_geo)))

    # 4. draw: per track and bin, 40 test positives + 10 layer 2 examples, one lesion per patient; each test
    #    positive matched to one negative patient at the same height inside the organ
    rng_ = random.Random(SEED)
    pid = {a: scans[a]["person_id"] for a in scans}
    items, used_neg = [], defaultdict(set)
    short = {}
    for t in TRACKS:
        neg_pool = [a for a in sorted(negs[t]) if G.get(a, (None,))[0]
                    and R.get(bd.nifti(scans[a]["study"], scans[a]["series"]).name, {}).get(RANGE_OF[t])]
        rng_.shuffle(neg_pool)
        test_patients = set()
        for b in TRACK_BINS[t]:
            cands = [x for x in pos if x["track"] == t and x["bin"] == b]
            rng_.shuffle(cands)
            seen, picked = set(), []
            for x in cands:
                if pid[x["acc"]] in seen:
                    continue
                seen.add(pid[x["acc"]])
                picked.append(x)
            test, examples = picked[:PER_BIN], picked[PER_BIN:PER_BIN + EXAMPLES_PER_BIN]
            if len(test) < PER_BIN or len(examples) < 3:
                short[f"{t} {b}"] = (len(test), len(examples))
            test_patients |= {pid[x["acc"]] for x in test}
            for x in test:
                items.append(_item(x, t, b, "positive", scans, pid))
            for x in examples:
                items.append(_item(x, t, b, "example", scans, pid))
            for x in test:  # matched negative
                for a in neg_pool:
                    if a in used_neg[t] or pid[a] in test_patients or pid[a] in seen:
                        continue
                    n, dz, _zc = G[a]
                    name = bd.nifti(scans[a]["study"], scans[a]["series"]).name
                    lo, hi = R[name][RANGE_OF[t]]
                    frac = lo + x["organ_frac"] * (hi - lo)
                    k = slab_factor(dz)
                    z = min(npz_z(frac * (n - 1), k), n // k - 1)
                    used_neg[t].add(a)
                    items.append({"item": opaque("neg", t, b, a, x["acc"]), "track": t, "bin": b, "role": "negative",
                                  "matched_to": opaque("pos", t, b, x["acc"], x["image"]), "scan": opaque("scan", a),
                                  "person": opaque("person", pid[a]), "name": name, "acc": a,
                                  "z": z, "n_npz": n // k})
                    break
                else:
                    short[f"{t} {b} negatives"] = short.get(f"{t} {b} negatives", 0) + 1

    # 5. scans: reuse an existing ct.npz, else convert into OUT/scans/<scan id>
    existing = _existing_volumes()
    scan_rows = {}
    for it in items:
        s = scans[it["acc"]]
        d = existing.get(it["name"])
        scan_rows[it["scan"]] = {"scan": it["scan"], "nifti": str(bd.nifti(s["study"], s["series"])),
                                 "ct_dir": str(d if d else SCANS_DIR / it["scan"]), "convert": d is None}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    (PRIVATE / "scans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scan_rows.values()), encoding="utf-8")

    by = Counter((i["track"], i["bin"], i["role"]) for i in items)
    summary = {"funnel": dict(sorted(c.items())), "liver_inside_organ": dict(liver_hit),
               "items": {f"{t} {b}": {r: by[(t, b, r)] for r in ("positive", "negative", "example")}
                         for t in TRACKS for b in TRACK_BINS[t]},
               "eligible_negative_reports": {t: len(negs[t]) for t in TRACKS},
               "short": short, "scans": len(scan_rows), "scans_to_convert": sum(r["convert"] for r in scan_rows.values())}
    (OUT / "item_counts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    n_liver = liver_hit["inside"] + liver_hit["outside"]
    if n_liver:
        print(f"liver citations inside the liver: {liver_hit['inside'] / n_liver:.1%} of {n_liver} "
              f"(the other direction: {liver_hit['other direction inside'] / n_liver:.1%})")
    return 0


def _item(x, t, b, role, scans, pid) -> dict:
    return {"item": opaque("pos" if role == "positive" else "ex", t, b, x["acc"], x["image"]), "track": t, "bin": b,
            "role": role, "scan": opaque("scan", x["acc"]), "person": opaque("person", pid[x["acc"]]),
            "name": x["name"], "acc": x["acc"], "z": x["z"], "n_npz": x["n"] // x["k"],
            "mm": x["mm"], "side": x["side"], "desc": x["desc"], "location": x["location"], "image": x["image"],
            "organ_frac": round(x["organ_frac"], 4)}


def cmd_check(args) -> int:
    """Re-derive 20 random positives' slabs another way: in millimetres through the NIfTI's own affine and the middle
    slice's DICOM position, instead of slice-index arithmetic. The middle header's position must land on the
    volume's centre slice (within half a slice), and the cited image, stepped from there by the slice spacing toward
    the feet, must land on the item's slab. PASS when all 20 match."""
    import io
    import nibabel as nib
    import numpy as np
    import pydicom

    pos = [i for i in load_items() if i["role"] == "positive"]
    pick = random.Random(SEED + 9).sample(pos, min(args.n, len(pos)))
    nifti_root = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/feb26")
    ok, why = 0, Counter()
    for i in pick:
        stem = i["name"][: -len(".nii.gz")]
        d = pydicom.dcmread(io.BytesIO((DICOM_META / f"{stem}.dcm").read_bytes()), stop_before_pixels=True, force=True)
        img = nib.as_closest_canonical(nib.load(str(nifti_root / i["name"])))
        A, n = img.affine, img.shape[2]
        z_mm_mid = float(d.ImagePositionPatient[2])  # DICOM LPS z = RAS z (both increase toward the head)
        idx = lambda zmm: (zmm - A[2, 3]) / A[2, 2]  # canonical: z index from world z
        centre = idx(z_mm_mid)
        if abs(centre - (n - 1) / 2) > 1.0:
            why["middle header more than a slice off the centre"] += 1
            continue
        z_mm = z_mm_mid - (i["image"] - int(d.InstanceNumber)) * abs(A[2, 2])
        k = slab_factor(abs(A[2, 2]))
        slab = int(round(idx(z_mm))) // k
        if slab == i["z"]:
            ok += 1
        else:
            why[f"slab differs by {slab - i['z']}"] += 1
    print(f"slab re-derived in mm: {ok}/{len(pick)} match; mismatches {dict(why)}")
    print("PASS" if ok == len(pick) else "FAIL")
    return 0 if ok == len(pick) else 1


def load_items() -> list[dict]:
    return [json.loads(l) for l in (PRIVATE / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]


def load_scans() -> dict[str, dict]:
    return {r["scan"]: r for r in map(json.loads, (PRIVATE / "scans.jsonl").read_text(encoding="utf-8").splitlines()) if r}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    return {"build": cmd_build, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
