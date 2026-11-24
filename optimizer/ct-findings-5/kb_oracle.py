"""Half A of the oracle-examples study (spec planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md): each VAL
scan's report-matched examples for lymph nodes and pneumonitis, drawn beside the same level of its own scan.

    build   the sets -> ct-findings-5-v1_7-private/kb_oracle/sets.json (private: built from VAL reports), and prints how
            close each pick was (the spec's coverage check: stop if under half are close)
    stage   (from agentic.stage, config.KB_ORACLE = grid | zoom) examples/ for one call folder

Per target, three examples with the same structure whatever the scan's answer (oracle_match): a confirmed finding at
the scan's site, the same site reported clear in another patient, and a confirmed finding at a different site.
- Lymph nodes: examples are the small-finding study's node items (TRAIN; a positive at a station, or a clean slice at a
  positive's level); a VAL scan's site and size bin come from the findings extraction (its largest thoracic node); a
  scan with none takes a site and bin drawn from the VAL positives' (seeded), so clear scans look the same.
- Pneumonitis: examples are the round-4 library's pneumonitis entries, lobes read from their text; a VAL scan's lobe is
  one of the lobes the report pass found (scripts/pneumonitis_pass.py) for answer-key positives, else drawn from the
  positives' lobes.
Two arms: grid = three whole slices per side in the finding's window; zoom = the same three slices cropped to the
organ's TotalSegmentator box + 10% (lungs for pneumonitis; the middle half of the lungs' box for nodes) and enlarged.
Each image is the example's three slices above the same three levels of the tester's scan.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "small_finding"))

import oracle_match as om  # noqa: E402

TARGETS = ("lesion_lymph_nodes", "pneumonitis")
SEED = 20261124
TILE = 512


def _paths():
    import config

    priv = config.ID_MAP_DIR / "kb_oracle"
    return config, priv / "sets.json", priv / "pneumonitis_val.json"


# --------------------------------------------------------------------------------------------------
# Sets
# --------------------------------------------------------------------------------------------------


def node_bin(mm: float | None) -> str | None:
    if mm is None:
        return None
    return "5-9" if mm < 10 else "10-19" if mm < 20 else ">=20"


def val_node_sites(config) -> dict[str, tuple[str, str]]:
    """{VAL id: (station, size bin)} from the findings extraction: the largest thoracic node the report measures."""
    import items as it

    idm = {r["id"]: r for r in map(json.loads, (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines()) if r}
    acc = {idm[i]["accession_number"]: i for i in val_ids(config)}
    out = {}
    for line in it.ANSWERS.open(encoding="utf-8"):
        d = json.loads(line)
        i = acc.get(d.get("accession_number"))
        if not i:
            continue
        try:
            les = json.loads(d["data"]["lesions"]["lesions_json"] or "[]")
        except Exception:  # noqa: BLE001
            continue
        nodes = [l for l in les if str(l.get("organ", "")).lower() == "lymph nodes"
                 and om.site("lymph_node", l.get("location", "")) != "other"]
        if nodes:
            l = max(nodes, key=lambda l: node_mm(l) or 0)
            out[i] = (om.site("lymph_node", l.get("location", "")), node_bin(node_mm(l)) or "10-19")
    return out


def node_mm(lesion: dict) -> float | None:
    """A node's size by the small-finding study's own rule (items.parse_size + lesion_mm: the short axis when given),
    so VAL scans and the node examples are binned the same way."""
    import items as it

    dims = it.parse_size(lesion.get("size_current"))
    return it.lesion_mm(dims, "lymph_node", str(lesion.get("size_axis") or "")) if dims else None


def val_ids(config) -> list[str]:
    return sorted(i for i in json.loads((config.DATA_DIR / "pools.json").read_text())["val"]
                  if (config.ORGAN_DIR / i / "target.json").is_file())


LOBE_RX = dict(om.LOBES)


def entry_lobes(e: dict) -> set[str]:
    t = " ".join([e.get("title", ""), " ".join(e.get("words", [])), e.get("report_sentence", ""), e.get("note", "")]).lower()
    if re.search(r"bilateral|both lungs|diffuse", t):
        return set(LOBE_RX)
    return {k for k, rx in om.LOBES if re.search(rx, t)}


def build(library: Path) -> dict:
    """The sets for every VAL scan, and how close each pick was."""
    import items as it

    config, out_path, pn_path = _paths()
    gold = {json.loads(l)["id"]: json.loads(l) for l in (config.DATA_DIR / "splits" / "val.jsonl").read_text().splitlines() if l}
    vals = val_ids(config)
    close = Counter()
    sets: dict[str, dict] = {v: {} for v in vals}

    # lymph nodes, from the perception study's node items
    items = [i for i in it.load_items() if i["track"] == "lymph_node"]
    by = {i["item"]: i for i in items}
    scans = it.load_scans()
    def nsite(i):
        src = by[i["matched_to"]] if i["role"] == "negative" else i
        return om.site("lymph_node", src.get("location", ""))
    lesions = [i for i in items if i["role"] in ("positive", "example")]
    clear = [i for i in items if i["role"] == "negative"]
    sites = val_node_sites(config)
    pos_sites = [sites[v] for v in vals if gold[v].get("lesion_lymph_nodes") == 1 and v in sites]
    for v in vals:
        rng = random.Random(f"{SEED}:nodes:{v}")
        want = sites[v] if (gold[v].get("lesion_lymph_nodes") == 1 and v in sites) else rng.choice(pos_sites)
        st, b = want
        def pick(pool, at_site: bool):
            cand = [i for i in pool if (nsite(i) == st) == at_site]
            same_bin = [i for i in cand if i["bin"] == b]
            rng.shuffle(same_bin)
            if same_bin:
                return same_bin[0], "exact"
            rng.shuffle(cand)
            return (cand[0], "other bin") if cand else (None, "none")
        yes, hy = pick(lesions, True)
        no, hn = pick(clear, True)
        decoy, _hd = pick(lesions, False)
        close[f"nodes yes {hy}"] += 1
        close[f"nodes no {hn}"] += 1
        if yes and no and decoy:
            exs = [dict(kind=k, type="node", item=x["item"], ct_dir=scans[x["scan"]]["ct_dir"], nifti=x["name"], z=x["z"],
                        frac=x.get("organ_frac", by[x["matched_to"]]["organ_frac"] if x["role"] == "negative" else 0.5), desc=x.get("desc", ""), location=x.get("location", ""), site=nsite(x))
                   for k, x in (("yes", yes), ("no", no), ("decoy", decoy))]
            rng.shuffle(exs)
            sets[v]["lesion_lymph_nodes"] = exs

    # pneumonitis, from the library's entries
    pn = json.loads(pn_path.read_text())
    entries = []
    for f in sorted((library / "entries").glob("*/entry.json")):
        e = json.loads(f.read_text())
        if e.get("finding") == "pneumonitis" and e.get("views"):
            entries.append((f.parent.name, e, entry_lobes(e)))
    pos_lobes = [l for v in vals if gold[v].get("pneumonitis") == 1 for l in pn.get(v, {}).get("lobes", [])]
    for v in vals:
        rng = random.Random(f"{SEED}:pneu:{v}")
        own = pn.get(v, {}).get("lobes", []) if gold[v].get("pneumonitis") == 1 else []
        lobe = rng.choice(sorted(own)) if own else rng.choice(pos_lobes)
        def pick(answer: str, at: bool):
            cand = [x for x in entries if x[1].get("answer") == answer and ((lobe in x[2]) == at) and (x[2] or not at)]
            rng.shuffle(cand)
            return cand[0] if cand else None
        yes, no, decoy = pick("yes", True), pick("no", True), pick("yes", False)
        close[f"pneumonitis yes {'exact' if yes else 'none'}"] += 1
        close[f"pneumonitis no {'exact' if no else 'none'}"] += 1
        if yes and no and decoy:
            exs = [dict(kind=k, type="entry", entry=x[0], site=lobe if k != "decoy" else ",".join(sorted(x[2])) or "unstated",
                        title=x[1].get("title", ""), report_sentence=x[1].get("report_sentence", ""), note=x[1].get("note", ""))
                   for k, x in (("yes", yes), ("no", no), ("decoy", decoy))]
            rng.shuffle(exs)
            sets[v]["pneumonitis"] = exs
    full = sum(all(t in sets[v] for t in TARGETS) for v in vals)
    summary = {"close": dict(close), "scans": len(vals), "scans_with_both_sets": full,
               "target_questions_with_a_set": sum(len(sets[v]) for v in vals)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"sets": sets, "_summary": summary}, indent=1))
    return summary


# --------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------


def _vol(ct_dir: Path):
    import numpy as np

    meta = json.loads((Path(ct_dir) / "meta.json").read_text())
    return np.load(Path(ct_dir) / "ct.npz")["hu"], meta


def _mask(nifti_name: str):
    import nibabel as nib
    import numpy as np
    import render as rd

    return np.asarray(nib.as_closest_canonical(nib.load(str(rd.MASKS / nifti_name))).dataobj)


def _range(organs: dict, track: str) -> tuple[float, float] | None:
    by = {o["organ"]: o for o in organs.get("organs", []) if o.get("present")}
    if track == "lymph_node":
        if "heart" in by and "lungs" in by:
            return by["heart"]["from_fraction"], by["lungs"]["to_fraction"]
        return None
    return (by["lungs"]["from_fraction"], by["lungs"]["to_fraction"]) if "lungs" in by else None


def _slices(hu, meta, mask, zs, track: str, arm: str):
    """The three slice images of one scan, whole (grid) or cropped to the organ box (zoom), each TILE px long side."""
    import render as rd

    win = "soft" if track == "lymph_node" else "lung"
    k = int(meta.get("slab_averaged", 1))
    out = []
    for z in zs:
        z = min(max(int(z), 0), hu.shape[2] - 1)
        full = rd.slice_image(hu, meta["spacing_mm"], z, win)
        if arm == "zoom" and mask is not None:
            dm = rd.display_mask(mask, z, k, rd.ZOOM_LABELS["lymph_node" if track == "lymph_node" else "lung_solid"],
                                 meta["spacing_mm"])
            box = rd.organ_box(dm, track) if dm.shape == (full.height, full.width) else None
            if box:
                full = full.crop(box)
        s = TILE / max(full.size)
        out.append(full.resize((max(1, round(full.width * s)), max(1, round(full.height * s)))))
    return out


def _pair(top, bottom, top_label: str, bottom_label: str):
    from PIL import Image, ImageDraw

    w = max(sum(t.width for t in top), sum(t.width for t in bottom)) + 8
    h1, h2 = max(t.height for t in top), max(t.height for t in bottom)
    img = Image.new("L", (w, 18 + h1 + 22 + h2), 0)
    d = ImageDraw.Draw(img)
    d.text((4, 2), top_label, fill=255)
    x = 0
    for t in top:
        img.paste(t, (x, 18)); x += t.width + 4
    d.text((4, 18 + h1 + 4), bottom_label, fill=255)
    x = 0
    for t in bottom:
        img.paste(t, (x, 18 + h1 + 22)); x += t.width + 4
    return img


def stage(scan_id: str, folder: Path, arm: str, library: Path) -> list[str]:
    """examples/ for one call folder (ct/, organs.json beside it): the scan's oracle sets drawn for this arm."""
    config, sets_path, _ = _paths()
    sets = json.loads(sets_path.read_text())["sets"].get(scan_id, {})
    folder = Path(folder)
    out = folder / "examples"
    out.mkdir(parents=True, exist_ok=True)
    organs = json.loads((folder / "organs.json").read_text())
    hu, meta = _vol(folder / "ct")
    nz = hu.shape[2]
    idm = {r["id"]: r for r in map(json.loads, (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines()) if r}
    own_mask = _mask(Path(idm[scan_id]["nifti"]).name) if arm == "zoom" else None
    written, lines, n = [], ["# Examples for your scan", "",
                             "Each image: TOP three slices of the example from another patient, BOTTOM the same levels of "
                             "YOUR scan. Open every image below with Read.", ""], 0
    for finding in TARGETS:
        for ex in sets.get(finding, []):
            n += 1
            name = f"{n}-{finding}.png"
            if ex["type"] == "node":
                ehu, emeta = _vol(Path(ex["ct_dir"]))
                ezs = [ex["z"] - 1, ex["z"], ex["z"] + 1]
                top = _slices(ehu, emeta, _mask(ex["nifti"]) if arm == "zoom" else None, ezs, "lymph_node", arm)
                rng = _range(organs, "lymph_node")
                oz = round((rng[0] + ex["frac"] * (rng[1] - rng[0])) * (nz - 1)) if rng else nz // 2
                bottom = _slices(hu, meta, own_mask, [oz - 1, oz, oz + 1], "lymph_node", arm)
                cap = (f"no enlarged or abnormal lymph node on these slices (the report says this region is clear); site: {ex['site']}"
                       if ex["kind"] == "no" else f"a confirmed lymph node: {ex['desc']}; location: {ex['location']}")
            else:
                import kb_prestage

                kb = kb_prestage.load_kb(library.parent)
                e = dict(kb.entries())[ex["entry"]]
                v = e["views"][0]
                src = config.WORKDIR / "kb-source" / e["source"]
                ehu, emeta = _vol(src / "ct")
                eorg = json.loads((src / "organs.json").read_text())
                er = kb.axis_range(v, eorg) or (0.3, 0.9)
                fr = [er[0] + (er[1] - er[0]) * q for q in (0.25, 0.5, 0.75)]
                ezs = [round(f * (ehu.shape[2] - 1)) for f in fr]
                enifti = None
                if arm == "zoom":
                    kbmap = {r["id"]: r for r in map(json.loads, (config.ID_MAP_DIR / "kb_id_map.jsonl").read_text().splitlines()) if r}
                    enifti = Path(kbmap[e["source"]]["nifti"]).name
                top = _slices(ehu, emeta, _mask(enifti) if enifti else None, ezs, "lung", arm)
                orr = kb.axis_range(v, organs) or (0.3, 0.9)
                ofr = [orr[0] + (orr[1] - orr[0]) * q for q in (0.25, 0.5, 0.75)]
                bottom = _slices(hu, meta, own_mask, [round(f * (nz - 1)) for f in ofr], "lung", arm)
                cap = (f"no pneumonitis (the answer key says no on this scan): {e.get('title', '')}; note: {e.get('note', '')}"
                       if ex["kind"] == "no" else f"{e.get('title', '')}; radiologist's words: {e.get('report_sentence', '')}; note: {e.get('note', '')}")
            _pair(top, bottom, "EXAMPLE (another patient)", "YOUR SCAN, same levels").save(out / name)
            written.append(name)
            lines += [f"## {name}", f"- question: {finding}", f"- {cap}", ""]
    if not written:
        lines += ["No examples could be drawn for this scan. Answer from your scan alone.", ""]
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for f in out.iterdir():
        f.chmod(0o444)
    return written


if __name__ == "__main__":
    import os

    os.environ.setdefault("KB_ARM", "1")
    lib = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print(json.dumps(build(lib), indent=1))
