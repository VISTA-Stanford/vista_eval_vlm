"""Worked examples of each pneumonitis variant from other patients' CT scans: each entry is an image (or a few) of one
scan whose answer key is that variant, the radiologist's words for the change, and a note on how to recognise it and
tell it from the other four. The library is the folder ../examples/ next to this tools folder; entries are
examples/entries/<entry>/entry.json plus images. Adapted from ct-findings-5's knowledge-base tool (kb.py).

    python3 -B tools/examples.py list [--variant radiation] [--words "ground glass"]
    python3 -B tools/examples.py open ENTRY --scan CALL_DIR [--out /tmp/ex]
    python3 -B tools/examples.py render CT_DIR --organs ORGANS_JSON --view '{"organ": "lungs", ...}' --out FILE.png
    python3 -B tools/examples.py check

list    entries matching a variant and/or words (in the title, words, note or report sentence). One line each.
open    prints the entry's note and report sentence, and writes one image per view: the example on the left and the
        SAME region of your own scan on the right, drawn with the same plane, window, slices (mapped through each
        scan's own organ ranges) and crop. Open each with the Read tool and compare the two halves.
render  draw one view of a CT the way entries are drawn (used to make entries; also works on your own scan).
check   validate every entry (fields, files, sizes) and that each variant has at least 2 entries; PASS or the problems.

A view is {"organ": <an organ in organs.json: "lungs", "lung_upper_lobe_right", "lung_lower_lobe_left", ..., or
"whole" for the whole scan>, "plane": "axial" | "coronal" | "sagittal", "window": "lung" | "soft" | "bone" | "liver" |
"LO,HI", "from": 0..1, "to": 0..1, "count": 1..16, "crop": "x0,y0,x1,y1" (optional, fractions of the image)}. For
axial views, from/to are fractions of the organ's own extent along the long axis (0 = its lowest slice, 1 = its
highest), so the same view lands on the same part of the organ in any patient. For coronal and sagittal views,
from/to are fractions of that axis across the whole scan and the organ is ignored.

An entry.json: {"variant": one of VARIANTS, "source": the source scan's id, "title": "...", "note": "how to recognise
it and tell it from the others", "report_sentence": "the radiologist's words for the change (no names or dates)",
"words": ["..."], "views": [{...view..., "image": "view1.png"}]} with 1 to 4 views.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_ct  # noqa: E402

LIBRARY = Path(__file__).resolve().parents[1] / "examples"
ENTRIES = LIBRARY / "entries"
VARIANTS = ("radiation", "immunotherapy", "other_drug", "cop", "infection")
PLANES = tuple(view_ct.AXIS)
MAX_TILES = 16
MAX_VIEWS = 4
MIN_PER_VARIANT = 2
MAX_ENTRIES = 60
MAX_IMAGE_BYTES = 2_000_000
ENTRY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
TEXT_FIELDS = ("title", "note", "report_sentence")


# --------------------------------------------------------------------------------------------------
# Views: the same part of an organ in any scan
# --------------------------------------------------------------------------------------------------


def axis_range(view: dict, organs: dict | None) -> tuple[float, float] | None:
    """The view's from/to as fractions of the whole scan's long axis, or None when the organ is not in this scan's
    organ ranges. "whole", or a coronal/sagittal view, takes from/to as they are."""
    lo, hi = float(view.get("from", 0.0)), float(view.get("to", 1.0))
    organ = view.get("organ", "whole")
    if view.get("plane", "axial") != "axial" or organ == "whole":
        return lo, hi
    if not organs or not organs.get("available"):
        return None
    by = {o["organ"]: o for o in organs.get("organs", []) if o.get("present")}
    if organ not in by:
        return None
    a, b = by[organ]["from_fraction"], by[organ]["to_fraction"]
    return a + lo * (b - a), a + hi * (b - a)


def view_problem(view) -> str | None:
    if not isinstance(view, dict):
        return "a view is not an object"
    if view.get("plane", "axial") not in PLANES:
        return f"plane must be one of {PLANES}"
    try:
        lo, hi = float(view.get("from", 0.0)), float(view.get("to", 1.0))
        count = int(view.get("count", 9))
    except (TypeError, ValueError):
        return "from, to and count must be numbers"
    if not (0.0 <= lo <= hi <= 1.0):
        return "from and to must satisfy 0 <= from <= to <= 1"
    if not (1 <= count <= MAX_TILES):
        return f"count must be 1 to {MAX_TILES}"
    w = str(view.get("window", "lung"))
    if w not in view_ct.WINDOWS and not re.match(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$", w):
        return f"window must be one of {tuple(view_ct.WINDOWS)} or LO,HI"
    crop = view.get("crop", "")
    if crop:
        try:
            x0, y0, x1, y1 = (float(v) for v in str(crop).split(","))
        except ValueError:
            return "crop must be x0,y0,x1,y1"
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            return "crop fractions must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
    return None


def render(ct_dir: Path, organs: dict | None, view: dict, size: int = 1024):
    """The view drawn on one CT, or (None, why) when its organ is not in this scan."""
    problem = view_problem(view)
    if problem:
        return None, problem
    rng = axis_range(view, organs)
    if rng is None:
        return None, f"the organ {view.get('organ')!r} is not in this scan's organ ranges"
    hu, meta = view_ct.load(ct_dir)
    plane = view.get("plane", "axial")
    n = hu.shape[view_ct.AXIS[plane]]
    count = int(view.get("count", 9))
    lo, hi = int(round(rng[0] * (n - 1))), int(round(rng[1] * (n - 1)))
    idxs = sorted({int(round(lo + (hi - lo) * j / max(count - 1, 1))) for j in range(count)})
    img = view_ct.grid(hu, meta["spacing_mm"], plane, idxs, str(view.get("window", "lung")), str(view.get("crop", "")), size)
    return img, None


def side_by_side(left, right, left_label: str, right_label: str, size: int = 2048):
    """Two images in one, scaled to the same height, each with a label strip above it."""
    from PIL import Image, ImageDraw

    h = max(left.height, right.height)
    imgs = [im.resize((max(1, round(im.width * h / im.height)), h)) for im in (left, right)]
    strip, gap = 18, 12
    out = Image.new("L", (imgs[0].width + gap + imgs[1].width, h + strip), color=0)
    out.paste(imgs[0], (0, strip))
    out.paste(imgs[1], (imgs[0].width + gap, strip))
    d = ImageDraw.Draw(out)
    d.text((4, 2), left_label, fill=255)
    d.text((imgs[0].width + gap + 4, 2), right_label, fill=255)
    scale = size / max(out.size)
    if scale < 1:
        out = out.resize((round(out.width * scale), round(out.height * scale)))
    return out


# --------------------------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------------------------


def entries() -> list[tuple[str, dict]]:
    out = []
    for f in sorted(ENTRIES.glob("*/entry.json")) if ENTRIES.is_dir() else []:
        try:
            out.append((f.parent.name, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            out.append((f.parent.name, {}))
    return out


def entry_problems(eid: str, e: dict) -> list[str]:
    """What is wrong with one entry (its fields and files)."""
    bad = []
    if not ENTRY_ID.match(eid):
        bad.append("entry folder name must be lowercase letters, digits and dashes")
    if not isinstance(e, dict) or not e:
        return bad + ["entry.json is missing or not a JSON object"]
    if e.get("variant") not in VARIANTS:
        bad.append(f"variant must be one of {VARIANTS}")
    for k in TEXT_FIELDS:
        if not isinstance(e.get(k, ""), str):
            bad.append(f"{k} must be text")
    if not str(e.get("title", "")).strip() or not str(e.get("note", "")).strip():
        bad.append("title and note are required")
    if not str(e.get("source", "")).strip():
        bad.append("source names the scan the images come from")
    if not isinstance(e.get("words", []), list) or not all(isinstance(w, str) for w in e.get("words", [])):
        bad.append("words must be a list of strings")
    views = e.get("views", [])
    if not isinstance(views, list) or not 1 <= len(views) <= MAX_VIEWS:
        bad.append(f"views must be a list of 1 to {MAX_VIEWS}")
        views = []
    for v in views:
        p = view_problem(v)
        if p:
            bad.append(p)
            continue
        img = ENTRIES / eid / str(v.get("image", ""))
        if not v.get("image") or "/" in str(v["image"]) or not img.is_file():
            bad.append("each view names its image, a file in the entry's folder")
        elif img.stat().st_size > MAX_IMAGE_BYTES:
            bad.append(f"{v['image']} is over {MAX_IMAGE_BYTES // 1_000_000} MB")
    return bad


def library_problems() -> list[str]:
    """Problems with the library as a whole: too many entries, or a variant with fewer than MIN_PER_VARIANT entries.
    An empty library (no entries at all) is valid: it is the zero-shot arm."""
    rows = entries()
    if not rows:
        return []
    bad = [f"{eid}: {p}" for eid, e in rows for p in entry_problems(eid, e)]
    if len(rows) > MAX_ENTRIES:
        bad.append(f"the library has {len(rows)} entries; at most {MAX_ENTRIES}")
    for v in VARIANTS:
        n = sum(1 for _eid, e in rows if e.get("variant") == v)
        if n < MIN_PER_VARIANT:
            bad.append(f"variant {v} has {n} entries; each variant needs at least {MIN_PER_VARIANT}")
    return bad


def matches(e: dict, variant: str | None, words: str | None) -> bool:
    if variant and e.get("variant") != variant:
        return False
    if words:
        hay = " ".join([*(str(e.get(k, "")) for k in TEXT_FIELDS), *map(str, e.get("words", []))]).lower()
        return all(w in hay for w in words.lower().split())
    return True


def cmd_list(args) -> int:
    rows = [(eid, e) for eid, e in entries() if matches(e, args.variant, args.words)]
    if not rows:
        print("no entries match" if entries() else "the library has no entries")
        return 0
    print(f"{len(rows)} entries match")
    for eid, e in rows[: args.limit]:
        print(f"{eid}  variant={e.get('variant')}  {len(e.get('views', []))} view(s)  {e.get('title', '')}")
    return 0


def cmd_open(args) -> int:
    e = dict(entries()).get(args.entry)
    if e is None:
        print(f"no entry {args.entry!r}; run list")
        return 1
    print(f"# {e.get('title')}\nvariant (the answer key on the example's scan): {e.get('variant')}")
    if e.get("report_sentence"):
        print(f"radiologist's words: {e['report_sentence']}")
    print(f"note: {e.get('note')}")
    organs_path = Path(args.scan) / "organs.json"
    organs = json.loads(organs_path.read_text(encoding="utf-8")) if organs_path.is_file() else None
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    for k, v in enumerate(e.get("views", []), 1):
        example = Image.open(ENTRIES / args.entry / v["image"]).convert("L")
        mine, why = render(Path(args.scan) / "ct", organs, v)
        out = args.out / f"{args.entry}-view{k}.png"
        if mine is None:
            example.save(out)
            print(f"view {k}: wrote {out} (example only: {why})")
            continue
        side_by_side(example, mine, f"EXAMPLE ({e.get('variant')})", "YOUR SCAN, same region").save(out)
        print(f"view {k}: wrote {out} (left: example; right: your scan)")
    print("Open each image with Read. Then look again at your own scan around that region.")
    return 0


def cmd_render(args) -> int:
    organs = json.loads(Path(args.organs).read_text(encoding="utf-8")) if args.organs else None
    img, why = render(Path(args.ct_dir), organs, json.loads(args.view), size=args.size)
    if img is None:
        print(f"cannot render: {why}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out} ({img.width}x{img.height})")
    return 0


def cmd_check(_args) -> int:
    bad = library_problems()
    for p in bad[:30]:
        print(p)
    if bad:
        print(f"FAIL: {len(bad)} problems")
        return 1
    print(f"PASS: {len(entries())} entries")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--variant", choices=VARIANTS)
    ls.add_argument("--words")
    ls.add_argument("--limit", type=int, default=30)
    op = sub.add_parser("open")
    op.add_argument("entry")
    op.add_argument("--scan", type=Path, required=True)
    op.add_argument("--out", type=Path, default=Path("/tmp/ex"))
    r = sub.add_parser("render")
    r.add_argument("ct_dir")
    r.add_argument("--organs")
    r.add_argument("--view", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--size", type=int, default=1024)
    sub.add_parser("check")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "open": cmd_open, "render": cmd_render, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
