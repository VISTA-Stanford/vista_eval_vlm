"""The atlas: 10 marked examples of each pneumonitis cause from other patients' CT scans, the same for every scan. Each
entry is one image (a crop of up to 4 slices, 768 px, with arrows and boxes at the signs), the findings written out
and tied to slice numbers, and the radiologist's words. The atlas is ../atlas/ next to this tools folder; entries are
atlas/entries/<entry>/entry.json plus image.png. The guide (../guide/README.md, ../guide/literature.md) explains the
patterns and points at atlas entries and slices.

    python3 -B tools/atlas.py draw CT_DIR --organs ORGANS_JSON --view '{...}' [--marks '[...]'] --out FILE.jpg
    python3 -B tools/atlas.py list [--variant cop]
    python3 -B tools/atlas.py check            the atlas and the guide's literature citations; PASS or the problems

A view is examples.py's (organ, plane, window, from, to, count <= 4, crop). Marks are drawn on the final image, in
fractions of its width and height (0,0 top left): {"arrow": [x_tail, y_tail, x_head, y_head], "label": "..."} or
{"box": [x0, y0, x1, y1], "label": "..."}. Draw the image once without marks, look at it, then place the marks.

An entry.json: {"variant": one of VARIANTS, "source": the scan's id, "title": "...",
"findings": [{"text": "what is seen, in the literature's terms", "slices": "38-52" (the source scan's axial slice
numbers), "place": "right paramediastinal"}, ...] (1 to 6), "report_sentence": "the radiologist's words for the change
(no names or dates)", "patterns": ["the guide's pattern names this entry shows"], "image": "image.jpg",
"view": {...}, "marks": [...]}. ``check`` needs exactly ATLAS_PER_VARIANT entries of each variant (or none at all).

guide/literature.md: one section per cause headed exactly `## radiation`, `## immunotherapy`, `## other_drug`,
`## cop`, `## infection`, citing sources as [L1], [L2], ...; and a `## Sources` section with one line per source,
`- [L1] authors or site, title, journal, year. link`. guide/README.md: a `## Patterns` section, one bullet per pattern,
each citing at least one [Ln]. ``check`` needs at least 5 distinct cited sources per cause, every cited [Ln] listed in
Sources with a link, and every pattern bullet citing one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import examples as ex  # noqa: E402  (the view format and the renderer)

PROGRAM = Path(__file__).resolve().parents[1]
ATLAS = PROGRAM / "atlas"
ENTRIES = ATLAS / "entries"
GUIDE = PROGRAM / "guide"
VARIANTS = ex.VARIANTS
PER_VARIANT = 10
MAX_TILES = 4
SIZE = 768
MAX_IMAGE_BYTES = 250_000
MIN_SOURCES = 5
ENTRY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SLICES = re.compile(r"^\d+(-\d+)?$")
TAG = re.compile(r"\[(L\d+)\]")


def draw(ct_dir: Path, organs: dict | None, view: dict, marks: list, size: int = SIZE):
    """The view on one CT with the marks, scaled to ``size`` px on its long side, as RGB; or (None, why)."""
    from PIL import ImageDraw

    if int(view.get("count", 4)) > MAX_TILES:
        return None, f"count must be at most {MAX_TILES} for an atlas image"
    img, why = ex.render(ct_dir, organs, view, size=2048)
    if img is None:
        return None, why
    scale = size / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale))).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for m in marks or []:
        p = mark_problem(m)
        if p:
            return None, p
        lab = str(m.get("label", ""))[:40]
        if "arrow" in m:
            x0, y0, x1, y1 = (v * s for v, s in zip(m["arrow"], (w, h, w, h)))
            d.line((x0, y0, x1, y1), fill=(255, 220, 0), width=3)
            ang_x, ang_y = x1 - x0, y1 - y0
            n = max((ang_x ** 2 + ang_y ** 2) ** 0.5, 1)
            ux, uy = ang_x / n, ang_y / n
            for sx in (1, -1):  # the head: two short strokes back from the tip
                d.line((x1, y1, x1 - 12 * ux + sx * 7 * uy, y1 - 12 * uy - sx * 7 * ux), fill=(255, 220, 0), width=3)
            if lab:
                d.text((x0 + 3, y0 + 3), lab, fill=(255, 220, 0))
        else:
            x0, y0, x1, y1 = (v * s for v, s in zip(m["box"], (w, h, w, h)))
            d.rectangle((x0, y0, x1, y1), outline=(0, 230, 255), width=3)
            if lab:
                d.text((x0 + 3, max(y0 - 12, 0)), lab, fill=(0, 230, 255))
    return img, None


def save(img, path: Path) -> None:
    """A JPEG at quality 85 (whatever the file name), so 50 atlas images fit a session's image memory: CT noise does
    not compress losslessly (2026-11-25: a 768 px crop was ~400 KB as RGB PNG, ~360 KB as a palette PNG)."""
    img.save(path, format="JPEG", quality=85, optimize=True)


def mark_problem(m) -> str | None:
    if not isinstance(m, dict) or ("arrow" in m) == ("box" in m):
        return "a mark is {\"arrow\": [...]} or {\"box\": [...]}"
    v = m.get("arrow", m.get("box"))
    if not (isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)):
        return "a mark's four numbers are fractions 0..1 of the image"
    if "box" in m and not (v[0] < v[2] and v[1] < v[3]):
        return "a box is x0,y0,x1,y1 with x1 > x0 and y1 > y0"
    return None


def entries() -> list[tuple[str, dict]]:
    out = []
    for f in sorted(ENTRIES.glob("*/entry.json")) if ENTRIES.is_dir() else []:
        try:
            out.append((f.parent.name, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            out.append((f.parent.name, {}))
    return out


def entry_problems(eid: str, e: dict) -> list[str]:
    bad = []
    if not ENTRY_ID.match(eid):
        bad.append("entry folder name must be lowercase letters, digits and dashes")
    if not isinstance(e, dict) or not e:
        return bad + ["entry.json is missing or not a JSON object"]
    if e.get("variant") not in VARIANTS:
        bad.append(f"variant must be one of {VARIANTS}")
    for k in ("title", "source"):
        if not str(e.get(k, "")).strip():
            bad.append(f"{k} is required")
    f = e.get("findings")
    if not isinstance(f, list) or not 1 <= len(f) <= 6:
        bad.append("findings must be a list of 1 to 6")
    else:
        for x in f:
            if not (isinstance(x, dict) and str(x.get("text", "")).strip() and SLICES.match(str(x.get("slices", "")))):
                bad.append("each finding has text and slices (\"38\" or \"38-52\": the source scan's axial slices)")
                break
    if not isinstance(e.get("patterns", []), list):
        bad.append("patterns must be a list of the guide's pattern names")
    img = ENTRIES / eid / str(e.get("image", ""))
    if not e.get("image") or "/" in str(e["image"]) or not img.is_file():
        bad.append("image names a file in the entry's folder")
    else:
        if img.stat().st_size > MAX_IMAGE_BYTES:
            bad.append(f"{e['image']} is over {MAX_IMAGE_BYTES // 1000} KB: draw it with atlas.py draw (768 px)")
        try:
            from PIL import Image

            with Image.open(img) as im:
                if max(im.size) > SIZE + 32:
                    bad.append(f"{e['image']} is {im.size}: at most {SIZE} px on its long side")
        except Exception:  # noqa: BLE001
            bad.append(f"{e['image']} is not a readable image")
    v = e.get("view")
    p = ex.view_problem(v) if v is not None else "view (how the image was drawn) is required"
    if p:
        bad.append(p)
    elif int(v.get("count", 4)) > MAX_TILES:
        bad.append(f"view count must be at most {MAX_TILES}")
    for m in e.get("marks", []) or []:
        p = mark_problem(m)
        if p:
            bad.append(p)
            break
    return bad


def atlas_problems() -> list[str]:
    rows = entries()
    if not rows:
        return []
    bad = [f"{eid}: {p}" for eid, e in rows for p in entry_problems(eid, e)]
    for v in VARIANTS:
        n = sum(1 for _eid, e in rows if e.get("variant") == v)
        if n != PER_VARIANT:
            bad.append(f"variant {v} has {n} entries; the atlas has exactly {PER_VARIANT} of each")
    return bad


def sections(text: str) -> dict[str, str]:
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^##\s+(\S.*?)\s*$", line)
        if m:
            cur = m.group(1).strip().lower()
            out[cur] = ""
        elif cur is not None:
            out[cur] += line + "\n"
    return out


def literature_problems() -> list[str]:
    """guide/literature.md and guide/README.md as the module docstring describes."""
    lit, readme = GUIDE / "literature.md", GUIDE / "README.md"
    if not lit.is_file():
        return ["guide/literature.md is missing"]
    sec = sections(lit.read_text(encoding="utf-8"))
    sources = {}
    for line in sec.get("sources", "").splitlines():
        m = re.match(r"^\s*[-*]\s*\[(L\d+)\]\s+(.*)$", line)
        if m:
            sources[m.group(1)] = m.group(2)
    bad = []
    if not sources:
        bad.append("guide/literature.md has no `## Sources` lines like `- [L1] ... link`")
    for tag, rest in sources.items():
        if not re.search(r"https?://|doi", rest, re.I):
            bad.append(f"source {tag} has no link or DOI")
    for v in VARIANTS:
        cited = set(TAG.findall(sec.get(v, "")))
        if v not in sec:
            bad.append(f"guide/literature.md has no `## {v}` section")
        elif len(cited & set(sources)) < MIN_SOURCES:
            bad.append(f"`## {v}` cites {len(cited & set(sources))} listed sources; at least {MIN_SOURCES}")
        for t in sorted(cited - set(sources)):
            bad.append(f"`## {v}` cites [{t}], which is not in Sources")
    if not readme.is_file():
        bad.append("guide/README.md is missing")
    else:
        pats = [ln for ln in sections(readme.read_text(encoding="utf-8")).get("patterns", "").splitlines()
                if re.match(r"^\s*[-*]\s+\S", ln)]
        if not pats:
            bad.append("guide/README.md has no `## Patterns` bullets")
        for ln in pats:
            tags = set(TAG.findall(ln))
            if not tags & set(sources):
                bad.append(f"pattern bullet cites no listed source: {ln.strip()[:60]}")
    return bad


def cited_sources() -> int:
    lit = GUIDE / "literature.md"
    if not lit.is_file():
        return 0
    return len({m.group(1) for line in sections(lit.read_text(encoding="utf-8")).get("sources", "").splitlines()
                for m in [re.match(r"^\s*[-*]\s*\[(L\d+)\]", line)] if m})


def cmd_draw(args) -> int:
    organs = json.loads(Path(args.organs).read_text(encoding="utf-8")) if args.organs else None
    img, why = draw(Path(args.ct_dir), organs, json.loads(args.view), json.loads(args.marks), size=args.size)
    if img is None:
        print(f"cannot draw: {why}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(img, args.out)
    print(f"wrote {args.out} ({img.width}x{img.height}, {args.out.stat().st_size // 1000} KB)")
    return 0


def cmd_list(args) -> int:
    rows = [(eid, e) for eid, e in entries() if not args.variant or e.get("variant") == args.variant]
    if not rows:
        print("the atlas has no entries")
        return 0
    for eid, e in rows:
        print(f"{eid}  variant={e.get('variant')}  {e.get('title', '')}")
    return 0


def cmd_check(_args) -> int:
    bad = atlas_problems() + (literature_problems() if (GUIDE / "literature.md").exists() or entries() else [])
    for p in bad[:40]:
        print(p)
    if bad:
        print(f"FAIL: {len(bad)} problems")
        return 1
    print(f"PASS: {len(entries())} entries; {cited_sources()} sources")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("ct_dir")
    d.add_argument("--organs")
    d.add_argument("--view", required=True)
    d.add_argument("--marks", default="[]")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--size", type=int, default=SIZE)
    ls = sub.add_parser("list")
    ls.add_argument("--variant", choices=VARIANTS)
    sub.add_parser("check")
    args = ap.parse_args(argv)
    return {"draw": cmd_draw, "list": cmd_list, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
