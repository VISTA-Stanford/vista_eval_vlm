"""The examples drawn beside each scan before its session (spec, Decided: 2 builder-chosen examples per variant, most
similar by MerlinOnc cosine, drawn beside the same lung region, 10 images, each required). Modelled on ct-findings-5's
kb_prestage.py (Phil 2026-11-24: drawn ahead, so looking at them is one Read each and can be required).

For each variant, the ``k`` library entries of that variant whose source scan is most similar to this scan (cosine of
the MerlinOnc ep99 whole-scan embeddings, DATA_DIR/embeddings.npz) and is not from this scan's patient
(DATA_DIR/groups.json) are drawn with the library's own tool (program tools/examples.py: the entry's first view on the
left, the same region of this scan on the right) as drawn_examples/<n>-<variant>-<entry>.png; drawn_examples/index.md
lists each with its variant, the radiologist's words and the note. An empty library gives an index that says so and
no images (the zero-shot arm). The pick reads only embeddings, patient groups and the library: never a label.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import config

OUT_DIR = "drawn_examples"


@lru_cache(maxsize=1)
def _embeddings() -> dict:
    import numpy as np

    with np.load(config.DATA_DIR / "embeddings.npz") as z:
        ids, emb = [str(i) for i in z["ids"]], z["emb"].astype("float64")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return dict(zip(ids, emb))


@lru_cache(maxsize=1)
def _groups() -> dict:
    return json.loads((config.DATA_DIR / "groups.json").read_text(encoding="utf-8"))


def similarities(case_id: str, sources: set[str], emb: dict | None = None) -> dict[str, float]:
    """Cosine of the case's embedding with each source scan's (sources without an embedding are left out)."""
    emb = _embeddings() if emb is None else emb
    me = emb[case_id]
    return {s: float(me @ emb[s]) for s in sources if s in emb}


def pick(entries: list[tuple[str, dict]], sims: dict[str, float], groups: dict[str, str], case_id: str,
         variant: str, k: int) -> list[str]:
    """The ``k`` entries of ``variant`` with views, most similar source first (ties by entry name), never one whose
    source is this scan or another scan of its patient."""
    mine = groups.get(case_id)
    rows = [(eid, e) for eid, e in entries if e.get("variant") == variant and e.get("views")
            and e.get("source") != case_id and (mine is None or groups.get(e.get("source")) != mine)
            and e.get("source") in sims]
    rows.sort(key=lambda r: (-sims[r[1]["source"]], r[0]))
    return [eid for eid, _e in rows[:k]]


def load_tool(tree: Path):
    """The reader's own tools/examples.py from a program version, pointed at that version's library."""
    spec = importlib.util.spec_from_file_location(f"examples_{abs(hash(str(tree)))}", Path(tree) / "tools" / "examples.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(tree) / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(Path(tree) / "tools"))
    mod.LIBRARY = Path(tree) / "examples"
    mod.ENTRIES = mod.LIBRARY / "entries"
    return mod


def planned(tree: Path, case_id: str, k: int = config.EXAMPLES_PER_VARIANT) -> list[str]:
    """The entries ``stage`` draws for this scan from version ``tree``, in order, without drawing them."""
    tool = load_tool(tree)
    entries = tool.entries()
    sims = similarities(case_id, {e.get("source") for _eid, e in entries if e.get("source")})
    return [eid for variant, _t, _q in config.QUESTIONS for eid in pick(entries, sims, _groups(), case_id, variant, k)]


def stage(tree: Path, folder: Path, case_id: str, k: int = config.EXAMPLES_PER_VARIANT) -> list[str]:
    """Write drawn_examples/ in the call folder ``folder`` (holding ct/ and organs.json) from the library of program
    version ``tree``. Returns the image names written, in order. Every file is made read-only."""
    from PIL import Image

    tool = load_tool(tree)
    folder = Path(folder)
    out = folder / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    organs = json.loads((folder / "organs.json").read_text(encoding="utf-8"))
    entries = tool.entries()
    by = dict(entries)
    sims = similarities(case_id, {e.get("source") for _eid, e in entries if e.get("source")})
    written, lines = [], ["# Examples drawn for your scan", "",
                          "Each image: LEFT the example from another patient's scan, RIGHT the same region of YOUR scan, "
                          "same window and slices. Open every image below with Read.", ""]
    n = 0
    for variant, _t, _q in config.QUESTIONS:
        for eid in pick(entries, sims, _groups(), case_id, variant, k):
            e, v = by[eid], by[eid]["views"][0]
            example = Image.open(tool.ENTRIES / eid / v["image"]).convert("L")
            mine, why = tool.render(folder / "ct", organs, v)
            n += 1
            name = f"{n:02d}-{variant}-{eid}.png"
            if mine is None:
                example.save(out / name)
                where = f"example only: {why}"
            else:
                tool.side_by_side(example, mine, f"EXAMPLE ({variant})", "YOUR SCAN, same region").save(out / name)
                where = "left: example; right: your scan"
            written.append(name)
            lines += [f"## {name}", f"- cause on the example's scan (its answer key): {variant}", f"- {e.get('title', '')}",
                      f"- radiologist's words: {e.get('report_sentence', '')}", f"- note: {e.get('note', '')}",
                      f"- image: {where}", ""]
    if not written:
        lines += ["The library has no examples, so there are no images here. Answer from your scan alone.", ""]
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for f in out.iterdir():
        f.chmod(0o444)
    return written


# --------------------------------------------------------------------------------------------------
# The atlas arm (config.ATLAS; spec 2026-11-25-ct-pneumonitis-atlas-arm.md): every entry for every scan
# --------------------------------------------------------------------------------------------------
ATLAS_DIR = "atlas_images"


def atlas_entries(tree: Path) -> list[tuple[str, dict]]:
    rows = []
    for f in sorted((Path(tree) / "atlas" / "entries").glob("*/entry.json")):
        try:
            rows.append((f.parent.name, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    order = {v: n for n, (v, _t, _q) in enumerate(config.QUESTIONS)}
    rows.sort(key=lambda r: (order.get(r[1].get("variant"), 99), r[0]))
    return rows


def planned_atlas(tree: Path, case_id: str) -> list[str]:
    """The atlas entries staged for this scan, in order: all of them except any from this scan or its patient. Reads
    no label."""
    mine = _groups().get(case_id)
    return [eid for eid, e in atlas_entries(tree)
            if e.get("source") != case_id and (mine is None or _groups().get(e.get("source")) != mine)]


def stage_atlas(tree: Path, folder: Path, case_id: str) -> list[str]:
    """Copy this scan's atlas images into atlas_images/ with an index of each entry's findings and slices. Returns the
    image names. Every file is made read-only."""
    import shutil

    out = Path(folder) / ATLAS_DIR
    out.mkdir(parents=True, exist_ok=True)
    by = dict(atlas_entries(tree))
    lines = ["# Atlas", "", "Clear examples of each cause from other patients' scans, the same for every scan. Each "
             "image: up to four axial slices of the example, with arrows and boxes at the signs. Open every image "
             "below with Read.", ""]
    written, cause = [], None
    for n, eid in enumerate(planned_atlas(tree, case_id), 1):
        e = by[eid]
        if e.get("variant") != cause:
            cause = e.get("variant")
            lines += [f"## {cause}", ""]
        src = Path(tree) / "atlas" / "entries" / eid / e["image"]
        name = f"{n:02d}-{cause}-{eid}{src.suffix.lower() or '.jpg'}"
        shutil.copy2(src, out / name)
        written.append(name)
        lines += [f"### {name}", f"- {e.get('title', '')}"]
        lines += [f"- slices {x.get('slices')}{', ' + x['place'] if x.get('place') else ''}: {x.get('text', '')}"
                  for x in e.get("findings", [])]
        if e.get("patterns"):
            lines.append(f"- guide patterns: {', '.join(map(str, e['patterns']))}")
        if e.get("report_sentence"):
            lines.append(f"- radiologist's words: {e['report_sentence']}")
        lines.append("")
    if not written:
        lines += ["The atlas has no examples, so there are no images here. Answer from your scan alone.", ""]
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for f in out.iterdir():
        f.chmod(0o444)
    return written


# --------------------------------------------------------------------------------------------------
# The library arm (config.LIBRARY; spec 2026-11-25-ct-pneumonitis-library.md)
# --------------------------------------------------------------------------------------------------


def library_examples(tree: Path) -> dict[str, dict]:
    out = {}
    for f in sorted((Path(tree) / "library" / "examples").glob("*/entry.json")):
        try:
            out[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def own_patient_examples(tree: Path, case_id: str) -> list[str]:
    """Library examples from this scan or another scan of its patient: hidden from this session. Reads no label."""
    mine = _groups().get(case_id)
    return sorted(eid for eid, e in library_examples(tree).items()
                  if e.get("source") == case_id or (mine is not None and _groups().get(e.get("source")) == mine))


SUGGEST_PER_CAUSE = 3


def suggest(tree: Path, case_id: str, hidden: list[str], k: int = SUGGEST_PER_CAUSE) -> dict:
    """For tools/library.py suggest: per cause, the visible library examples whose source scans are most like this
    scan (MerlinOnc cosine), most similar first, with their similarity. Reads embeddings and patient groups only: never
    a label (the example's cause is the builder's, from TRAIN)."""
    exs = {eid: e for eid, e in library_examples(tree).items() if eid not in hidden}
    sims = similarities(case_id, {e.get("source") for e in exs.values() if e.get("source")})
    out = {}
    for v, _t, _q in config.QUESTIONS:
        rows = [(eid, e) for eid, e in exs.items() if e.get("cause") == v and e.get("source") in sims]
        rows.sort(key=lambda r: (-sims[r[1]["source"]], r[0]))
        out[v] = [{"id": eid, "similarity": round(sims[e["source"]], 3)} for eid, e in rows[:k]]
    if config.LIBRARY_HIDE_SIM:  # keep the pick and its order within a cause, drop the scores the reader would see
        out = {v: [{"id": x["id"]} for x in xs] for v, xs in out.items()}
    return out


def _link_tree(src: Path, dest: Path) -> None:
    """Hard-link every file under src into dest (same disk by construction: the work folder's caches)."""
    import os

    for f in sorted(Path(src).rglob("*")):
        if f.is_file():
            out = dest / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            os.link(f, out)


def stage_library(tree: Path, folder: Path, case_id: str) -> dict:
    """library_assets/ (the literature images and their index, and exclude.json: the examples hidden from this scan)
    and library_scans/<source>/ (each visible example's full CT and organ mask) in the call folder, hard-linked from
    config.LIT_ASSETS_DIR and config.TRAIN_SCANS_DIR. Returns counts. Raises when a cache is missing."""
    folder = Path(folder)
    assets = folder / "library_assets"
    _link_tree(config.LIT_ASSETS_DIR, assets)
    hidden = own_patient_examples(tree, case_id)
    (assets / "exclude.json").write_text(json.dumps(hidden), encoding="utf-8")
    (assets / "suggest.json").write_text(json.dumps(suggest(tree, case_id, hidden)), encoding="utf-8")
    sources = sorted({e.get("source") for eid, e in library_examples(tree).items() if eid not in hidden and e.get("source")})
    for src in sources:
        base = config.TRAIN_SCANS_DIR / src
        _link_tree(base / "ct", folder / "library_scans" / src / "ct")
        for name in ("organs.json", "organ_mask.npz"):
            import os

            os.link(base / name, folder / "library_scans" / src / name)
    for f in [assets / "exclude.json", assets / "suggest.json"]:
        f.chmod(0o444)
    return {"hidden_examples": len(hidden), "example_scans": len(sources)}


DRAWN_DIR = "library_drawn"


def stage_library_drawn(tree: Path, folder: Path, case_id: str, k: int) -> list[str]:
    """config.LIBRARY_DRAWN: library_drawn/ in the call folder: for each cause, the k visible library examples most
    like this scan (library_assets/suggest.json, written by stage_library; never the scan's own patient) drawn beside
    the same region of the scan, and one literature sheet per cause (the one whose pattern group matches the drawn
    examples' patterns best), with index.md (each image's cause, title and findings). Reads no label."""
    import re
    import shutil

    from PIL import Image

    folder = Path(folder)
    tool = load_tool(tree)  # the reader's tools/examples.py: render and side_by_side
    out = folder / DRAWN_DIR
    out.mkdir(parents=True, exist_ok=True)
    sug = json.loads((folder / "library_assets" / "suggest.json").read_text(encoding="utf-8"))
    exs = library_examples(tree)
    organs = json.loads((folder / "organs.json").read_text(encoding="utf-8"))
    sheets_f = folder / "library_assets" / "lit" / "sheets.json"
    sheets = json.loads(sheets_f.read_text(encoding="utf-8")) if sheets_f.is_file() else []

    def slug(t):
        return re.sub(r"[^a-z0-9]+", "-", str(t or "").lower()).strip("-")

    lines = ["# Drawn for your scan", "", "For each cause: the Stanford examples (other patients at this hospital, with this "
             "hospital's answer key) most like your scan, each drawn beside the same region of YOUR scan (left: the "
             "example, marked; right: your scan), and one literature sheet (published figures of a pattern of that "
             "cause). Open every image below with Read.", ""]
    written, n = [], 0
    for v, _t, _q in config.QUESTIONS:
        lines += [f"## {v}", ""]
        pats = set()
        for s in sug.get(v, [])[:k]:
            e = exs.get(s["id"])
            if not e or not e.get("images"):
                continue
            im = e["images"][0]
            example = Image.open(Path(tree) / "library" / "examples" / s["id"] / im["image"]).convert("L")
            mine, _why = tool.render(folder / "ct", organs, im.get("view", {}))
            pic = example if mine is None else tool.side_by_side(example, mine, f"EXAMPLE ({v})", "YOUR SCAN, same region")
            n += 1
            name = f"{n:02d}-{v}-{s['id']}.jpg"
            pic.convert("RGB").save(out / name, format="JPEG", quality=85)
            written.append(name)
            pats.update(slug(p) for p in e.get("patterns", []))
            lines += [f"### {name}", f"- {e.get('title', '')}" + (f" (similarity to your scan {s['similarity']})" if "similarity" in s else "")]
            lines += [f"- slices {x.get('slices')}{', ' + x['place'] if x.get('place') else ''}: {x.get('text', '')}"
                      for x in e.get("findings", [])[:5]]
            lines.append("")
        cands = [m for m in sheets if m["cause"] == v]
        cands.sort(key=lambda m: (not any(p and (p in slug(m["pattern_group"]) or slug(m["pattern_group"]) in p) for p in pats),
                                  -len(m["ids"]), m["sheet"]))
        if cands:
            n += 1
            name = f"{n:02d}-{v}-literature-{slug(cands[0]['pattern_group'])}.jpg"
            shutil.copy2(folder / "library_assets" / "lit" / cands[0]["sheet"], out / name)
            written.append(name)
            lines += [f"### {name}", f"- published figures of: {cands[0]['pattern_group']} ({len(cands[0]['ids'])} figures)", ""]
    if not written:
        lines += ["The library has no examples, so there are no images here. Answer from your scan alone.", ""]
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for f in out.iterdir():
        f.chmod(0o444)
    return written
