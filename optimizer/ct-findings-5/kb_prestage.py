"""Pre-drawn knowledge-base examples for each tester scan (config.KB_PRESTAGE; Phil 2026-11-24).

In the pilot the tester ran ``kb.py open`` because it had to, but opened the saved side-by-side image in only 61 of
381 opens: it read the note and answered. Here the images are drawn before the session starts, into ``examples/``
beside its CT, so looking at them is one Read each and the instructions can require it the way arm D did.

For each target question, the ``k`` entries of that finding whose source scans are most similar to this scan
(kb_similarity.json, MerlinOnc whole-scan cosine) are drawn with the library's own tool (program tools/kb.py:
the entry's first view on the left, the same region of this scan on the right) as examples/<n>-<finding>-<entry>.png,
and examples/index.md lists each with its finding, the answer key's verdict on the example's scan, the radiologist's
words and the note. An empty library gives an index that says so and no images (the empty-library arm).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TARGETS = ("lesion_lymph_nodes", "pneumonitis")


def pick(entries: list[tuple[str, dict]], sims: dict, finding: str, k: int) -> list[str]:
    """The ``k`` entries of ``finding`` with views, most similar source scan first (ties by entry name)."""
    rows = [(eid, e) for eid, e in entries if e.get("finding") == finding and not e.get("text_only") and e.get("views")]
    rows.sort(key=lambda r: (-float(sims.get(r[1].get("source"), -1.0)), r[0]))
    return [eid for eid, _e in rows[:k]]


def load_kb(tree: Path):
    """The tester's own kb.py from a program version, pointed at that version's library."""
    spec = importlib.util.spec_from_file_location(f"kb_{abs(hash(str(tree)))}", Path(tree) / "tools" / "kb.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(tree) / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(Path(tree) / "tools"))
    mod.LIBRARY = Path(tree) / "library"
    mod.ENTRIES = mod.LIBRARY / "entries"
    return mod


def stage(tree: Path, folder: Path, k: int = 2) -> list[str]:
    """Write examples/ for the call folder ``folder`` (holding ct/, organs.json, kb_similarity.json) from the library of
    program version ``tree``. Returns the image file names written. Every file is made read-only."""
    from PIL import Image

    kb = load_kb(tree)
    folder = Path(folder)
    out = folder / "examples"
    out.mkdir(parents=True, exist_ok=True)
    sims = json.loads((folder / "kb_similarity.json").read_text(encoding="utf-8"))
    organs = json.loads((folder / "organs.json").read_text(encoding="utf-8"))
    entries = kb.entries()
    by = dict(entries)
    written, lines = [], ["# Examples for your scan", "",
                          "Each image: LEFT the example from another patient's scan, RIGHT the same region of YOUR scan, "
                          "same window and slices. Open every image below with Read.", ""]
    n = 0
    for finding in TARGETS:
        for eid in pick(entries, sims, finding, k):
            e, v = by[eid], by[eid]["views"][0]
            example = Image.open(kb.ENTRIES / eid / v["image"]).convert("L")
            mine, why = kb.render(folder / "ct", organs, v)
            n += 1
            name = f"{n}-{finding}-{eid}.png"
            if mine is None:
                example.save(out / name)
                where = f"example only: {why}"
            else:
                kb.side_by_side(example, mine, f"EXAMPLE (answer {e.get('answer', '-')})", "YOUR SCAN, same region").save(out / name)
                where = "left: example; right: your scan"
            written.append(name)
            lines += [f"## {name}", f"- finding: {finding}; answer key on the example's scan: {e.get('answer', '-')}",
                      f"- {e.get('title', '')}", f"- radiologist's words: {e.get('report_sentence', '')}",
                      f"- note: {e.get('note', '')}", f"- image: {where}", ""]
    if not written:
        lines += ["The library has no examples, so there are no images here. Answer from your scan alone.", ""]
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for f in out.iterdir():
        f.chmod(0o444)
    return written
