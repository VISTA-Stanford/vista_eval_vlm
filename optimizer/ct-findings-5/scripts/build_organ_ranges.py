"""Organ ranges for the example-CT arm C (spec planning/vista_eval_vlm/2026-11-22-ct-findings-example-ct-arms.md):
for each VAL pool scan and each of its 5 neighbours, where each organ lies along the superior-inferior axis, from the
TotalSegmentator localization table (organ_localization/thoracic_june26_localization.csv, vista-ct
vista_ct/processing/organ_localization.py). The table's dim_2 fractions are RAS bounding-box fractions, 0 = inferior,
1 = superior (vista-ct docs/organ-localization.md), the same frame as the task's volumes (as_closest_canonical; z = 0
is the lowest slice), so slice = round(fraction * (slices - 1)).

    build   write organ_ranges/<VAL id>/{target.json, n1.json .. n5.json} on the mount (a scan the table lacks gets
            {"available": false, ...}), then run the check
    check   190-ish targets covered (prints the count), every slice inside its volume, from <= centre <= to, files
            hold no identifier, and the orientation test: the lungs' centre lies above the liver's on nearly every
            scan with both (a frame flipped upside down would put nearly all of them below). Prints PASS.

    cd optimizer/ct-findings-5
    python3 -B scripts/build_organ_ranges.py build

Prints counts only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402

TABLE = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/organ_localization/thoracic_june26_localization.csv")
OUT = bd.DATA_DIR / "organ_ranges"
NEIGHBOUR_SCANS = bd.DATA_DIR / "neighbour_scans"
NOT_AVAILABLE = {"available": False, "note": "organ ranges not available for this scan"}
#: The orientation test: share of scans with both organs whose lung centre is above the liver centre.
ORIENT_MIN = 0.95


def scans():
    """[(VAL id, target nifti name, target meta path, [(rank, nifti name, meta path)])] for the VAL pool."""
    pool = sorted(json.loads((bd.DATA_DIR / "pools.json").read_text())["val"])
    id_map = {m["id"]: m for m in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    picks = json.loads((bd.PRIVATE_DIR / "neighbour_picks.json").read_text())
    smap = json.loads((bd.DATA_DIR / "neighbours_scan_map.json").read_text())
    out = []
    for vid in pool:
        nbrs = [(r, Path(p["nifti"]).name, NEIGHBOUR_SCANS / nid / "meta.json")
                for r, (p, nid) in enumerate(zip(picks[vid], smap[vid]), 1)]
        out.append((vid, Path(id_map[vid]["nifti"]).name, bd.DATA_DIR / "patients" / vid / "meta.json", nbrs))
    return out


def load_table(names: set[str]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    with TABLE.open(newline="") as f:
        for r in csv.DictReader(f):
            n = Path(r["nifti_filename"]).name
            if n in names:
                rows.setdefault(n, []).append(r)
    return rows


def num(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def ranges(rows: list[dict] | None, meta: Path) -> dict:
    if not rows:
        return dict(NOT_AVAILABLE)
    nz = int(json.loads(meta.read_text())["shape"][2])
    organs = []
    for r in sorted(rows, key=lambda r: r["structure"]):
        lo, hi, mid = num(r["dim_2_min_fraction"]), num(r["dim_2_max_fraction"]), num(r["dim_2_fraction"])
        present = str(r.get("present", "")).strip().lower() in ("true", "1") and None not in (lo, hi, mid)
        o = {"organ": r["structure"], "present": present}
        if present:
            o.update({"from_fraction": round(lo, 3), "centre_fraction": round(mid, 3), "to_fraction": round(hi, 3),
                      "from_slice": round(lo * (nz - 1)), "centre_slice": round(mid * (nz - 1)),
                      "to_slice": round(hi * (nz - 1))})
        organs.append(o)
    return {"available": True, "source": "TotalSegmentator, automatic; may be wrong on unusual scans",
            "axis": "superior-inferior; slice 0 and fraction 0 are the lowest (most inferior) slice", "slices": nz,
            "organs": organs}


def cmd_build() -> int:
    items = scans()
    names = {t for _v, t, _m, _n in items} | {n for *_x, nb in items for _r, n, _m in nb}
    table = load_table(names)
    for vid, tname, tmeta, nbrs in items:
        d = OUT / vid
        d.mkdir(parents=True, exist_ok=True)
        t = ranges(table.get(tname), tmeta)
        if t["available"]:  # arm C runs only on covered targets: an uncovered target gets no target.json
            (d / "target.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
        elif (d / "target.json").exists():
            (d / "target.json").unlink()
        for rank, nname, nmeta in nbrs:
            (d / f"n{rank}.json").write_text(json.dumps(ranges(table.get(nname), nmeta), indent=1), encoding="utf-8")
    print(f"built organ ranges for {len(items)} VAL scans")
    return cmd_check()


def cmd_check() -> int:
    items = scans()
    bad, targets, nb_ok, nb_na, above, both = [], 0, 0, 0, 0, 0
    for vid, _t, _m, nbrs in items:
        d = OUT / vid
        files = ([d / "target.json"] if (d / "target.json").is_file() else []) + [d / f"n{r}.json" for r, _n, _m in nbrs]
        targets += (d / "target.json").is_file()
        for f in files:
            if not f.is_file():
                bad.append(f"missing {f.name}")
                continue
            text = f.read_text()
            if vid in text or ".nii" in text:
                bad.append(f"{f.name} holds an identifier")
            j = json.loads(text)
            if f.name != "target.json":
                nb_ok += j["available"]
                nb_na += not j["available"]
            if not j["available"]:
                continue
            pos = {}
            for o in j["organs"]:
                if not o["present"]:
                    continue
                if not (0 <= o["from_slice"] <= o["centre_slice"] <= o["to_slice"] <= j["slices"] - 1):
                    bad.append(f"{f.name}: {o['organ']} slices out of order or outside the volume")
                pos[o["organ"]] = o["centre_fraction"]
            lung = [v for k, v in pos.items() if k.startswith("lung")]
            if lung and "liver" in pos:
                both += 1
                above += sum(lung) / len(lung) > pos["liver"]
    share = above / both if both else 0.0
    print(f"targets with organ ranges: {targets} of {len(items)}; neighbour files available {nb_ok}, "
          f"not available {nb_na}")
    print(f"orientation: lungs above liver on {above} of {both} scans with both ({share:.1%})")
    if share < ORIENT_MIN:
        bad.append(f"orientation test {share:.1%} < {ORIENT_MIN:.0%}")
    if bad:
        print(f"FAIL: {len(bad)} problems, first: {bad[:5]}")
        return 1
    print("PASS: organ ranges ready")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        return cmd_build()
    if cmd == "check":
        return cmd_check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
