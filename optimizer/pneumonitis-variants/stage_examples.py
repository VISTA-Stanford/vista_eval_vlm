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
