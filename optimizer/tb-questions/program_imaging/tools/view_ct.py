"""Render views of the PRIOR and MOST RECENT CT as one PNG grid you can open with the Read tool.

CT_DIR is the folder holding prior/ and current/ (each one CT volume), or one of those two folders. With both, each
row of the grid shows the PRIOR scan (left) and the MOST RECENT scan (right) at the same position, given as a
fraction of each scan's own axis, so the two tiles show roughly the same level (the scans may cover different
lengths of the body). --scan prior or --scan current shows one scan only.

Each volume is ct.npz (key "hu": Hounsfield units, int16, shape (x, y, z)) with meta.json beside it: spacing in
mm, and RAS orientation (index grows toward the patient's Right on x, Anterior on y, Superior on z). Images are
drawn the radiology way: axial and coronal with the patient's right on the image's left; axial with anterior
at the top; coronal and sagittal with superior at the top; sagittal with anterior on the left.

    python3 -B view_ct.py CT_DIR --plane axial --count 8 --window lung --out /tmp/axial_lung.png
    python3 -B view_ct.py CT_DIR --plane coronal --at 0.5 --window soft --out /tmp/cor.png
    python3 -B view_ct.py CT_DIR --scan current --plane axial --from 0.4 --to 0.7 --count 9 --window bone --crop 0.2,0.2,0.8,0.8 --out /tmp/zoom.png
    python3 -B view_ct.py CT_DIR --info

--count N evenly spaced slices between --from and --to (fractions of the axis, default the whole axis); --at
one or more positions instead, comma separated: 0.5 is a fraction of the axis, 120 a slice index. Slice indices
differ between the two scans, so --at with indices shows one scan only: MOST RECENT, or the one --scan names. --crop x0,y0,x1,y1 as fractions of the
image. Windows: lung (-1350..150), soft (soft tissue, -160..240), bone (-500..1300), or LO,HI in HU.
Each tile is labelled with its scan, slice index and position in mm. Keep a grid at about 16 tiles or fewer (with
both scans, --count 8 makes 16 tiles): the image is scaled down to --size pixels on its long side (default 1536).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WINDOWS = {"lung": (-1350, 150), "soft": (-160, 240), "bone": (-500, 1300)}
SCANS = (("prior", "PRIOR"), ("current", "MOST RECENT"))
AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def load(ct_dir: Path):
    meta = json.loads((ct_dir / "meta.json").read_text())
    return np.load(ct_dir / "ct.npz")["hu"], meta


def window(arr: np.ndarray, spec: str) -> np.ndarray:
    lo, hi = WINDOWS[spec] if spec in WINDOWS else (float(v) for v in spec.split(","))
    return (np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def view(hu: np.ndarray, plane: str, idx: int, spacing) -> tuple[np.ndarray, float]:
    """One 2-D slice in display orientation (rows top->bottom, columns left->right), and its row/col aspect."""
    sx, sy, sz = spacing
    if plane == "axial":  # rows: anterior -> posterior; columns: patient right -> left
        return hu[::-1, ::-1, idx].T, sy / sx
    if plane == "coronal":  # rows: superior -> inferior; columns: patient right -> left
        return hu[::-1, idx, ::-1].T, sz / sx
    return hu[idx, ::-1, ::-1].T, sz / sy  # sagittal: rows superior -> inferior; columns anterior -> posterior


def indices(n: int, args) -> list[int]:
    if args.at:
        # "0.5" is a fraction of the axis; "120" is a slice index.
        out = [int(round(float(t) * (n - 1))) if "." in t else int(t) for t in args.at.split(",")]
        return [min(max(i, 0), n - 1) for i in out]
    lo, hi = int(round(args.from_ * (n - 1))), int(round(args.to * (n - 1)))
    k = max(1, args.count)
    return sorted({int(round(lo + (hi - lo) * j / max(k - 1, 1))) for j in range(k)})


def fractions(args) -> list[float]:
    """Positions as fractions of an axis, so the two scans are cut at matching levels."""
    if args.at:
        return [float(t) for t in args.at.split(",")]
    k = max(1, args.count)
    return [args.from_ + (args.to - args.from_) * j / max(k - 1, 1) for j in range(k)]


def scans(ct_dir: Path, which: str) -> list[tuple[str, Path]]:
    if (ct_dir / "ct.npz").is_file():
        return [("CT", ct_dir)]
    found = [(label, ct_dir / name) for name, label in SCANS if (ct_dir / name / "ct.npz").is_file()]
    if not found:
        raise SystemExit(f"no ct.npz in {ct_dir} or in its prior/ and current/ folders")
    return found if which == "both" else [(label, d) for label, d in found if d.name == which]


def tile(hu, spacing, plane: str, i: int, args, label: str) -> Image.Image:
    img, aspect = view(hu, plane, i, spacing)
    if args.crop:
        x0, y0, x1, y1 = (float(v) for v in args.crop.split(","))
        h, w = img.shape
        img = img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
    t = Image.fromarray(window(img, args.window))
    t = t.resize((t.width, max(1, round(t.height * aspect))), Image.BILINEAR)
    ImageDraw.Draw(t).text((4, 2), f"{label} {plane} {i} ({i * spacing[AXIS[plane]]:.0f} mm)", fill=255)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ct_dir", type=Path)
    ap.add_argument("--scan", choices=("both", "prior", "current"), default="both")
    ap.add_argument("--info", action="store_true", help="print each scan's shape, spacing and extent in mm, then stop")
    ap.add_argument("--plane", choices=tuple(AXIS), default="axial")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--at", default="")
    ap.add_argument("--from", dest="from_", type=float, default=0.0)
    ap.add_argument("--to", type=float, default=1.0)
    ap.add_argument("--window", default="lung")
    ap.add_argument("--crop", default="")
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--out", type=Path, default=Path("/tmp/view.png"))
    args = ap.parse_args()

    vols = [(label, *load(d)) for label, d in scans(args.ct_dir, args.scan)]
    if args.info:
        for label, hu, meta in vols:
            spacing = meta["spacing_mm"]
            print(json.dumps({"scan": label, "shape": hu.shape, "spacing_mm": spacing,
                              "extent_mm": [round(s * n) for s, n in zip(spacing, hu.shape)],
                              "orientation": meta["orientation"]}))
        return
    ax = AXIS[args.plane]
    index_mode = bool(args.at) and not any("." in t for t in args.at.split(","))
    rows = []
    if len(vols) == 1 or index_mode:
        label, hu, meta = vols[-1]
        rows = [[tile(hu, meta["spacing_mm"], args.plane, i, args, label)] for i in indices(hu.shape[ax], args)]
    else:
        for f in fractions(args):
            row = []
            for label, hu, meta in vols:
                n = hu.shape[ax]
                i = min(max(int(round(f * (n - 1))), 0), n - 1)
                row.append(tile(hu, meta["spacing_mm"], args.plane, i, args, label))
            rows.append(row)
    if len(vols) == 1 or index_mode:  # one scan: pack the tiles into a square-ish grid
        flat = [t for r in rows for t in r]
        cols = math.ceil(math.sqrt(len(flat)))
        rows = [flat[j:j + cols] for j in range(0, len(flat), cols)]
    else:  # two scans: two pairs per row when there are many positions
        per = 2 if len(rows) > 4 else 1
        rows = [sum(rows[j:j + per], []) for j in range(0, len(rows), per)]
    tw = max(t.width for r in rows for t in r)
    th = max(t.height for r in rows for t in r)
    cols = max(len(r) for r in rows)
    grid = Image.new("L", (cols * tw, len(rows) * th))
    for y, r in enumerate(rows):
        for x, t in enumerate(r):
            grid.paste(t, (x * tw, y * th))
    scale = args.size / max(grid.size)
    if scale < 1:
        grid = grid.resize((round(grid.width * scale), round(grid.height * scale)), Image.BILINEAR)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    n_tiles = sum(len(r) for r in rows)
    print(f"wrote {args.out} ({grid.width}x{grid.height}, {n_tiles} tiles)")


if __name__ == "__main__":
    main()
