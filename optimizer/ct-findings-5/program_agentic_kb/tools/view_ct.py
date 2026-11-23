"""Render views of a CT volume as one PNG grid you can open with the Read tool.

The volume is ct.npz (key "hu": Hounsfield units, int16, shape (x, y, z)) with meta.json beside it: spacing in
mm, and RAS orientation (index grows toward the patient's Right on x, Anterior on y, Superior on z). Images are
drawn the radiology way: axial and coronal with the patient's right on the image's left; axial with anterior
at the top; coronal and sagittal with superior at the top; sagittal with anterior on the left.

    python3 -B view_ct.py CT_DIR --plane axial --count 16 --window lung --out /tmp/axial_lung.png
    python3 -B view_ct.py CT_DIR --plane coronal --at 0.5 --window soft --out /tmp/cor.png
    python3 -B view_ct.py CT_DIR --plane axial --from 0.4 --to 0.7 --count 9 --window bone --crop 0.2,0.2,0.8,0.8 --out /tmp/zoom.png
    python3 -B view_ct.py CT_DIR --info

--count N evenly spaced slices between --from and --to (fractions of the axis, default the whole axis); --at
one or more positions instead, comma separated: 0.5 is a fraction of the axis, 120 a slice index. --crop x0,y0,x1,y1 as fractions of the
image. Windows: lung (-1350..150), soft (-160..240), bone (-500..1300), liver (-40..160), or LO,HI in HU.
Each tile is labelled with its slice index and position in mm. Keep a grid at about 16 tiles or fewer: the
image is scaled down to --size pixels on its long side (default 1536).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WINDOWS = {"lung": (-1350, 150), "soft": (-160, 240), "bone": (-500, 1300), "liver": (-40, 160)}
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


def grid(hu: np.ndarray, spacing, plane: str, idxs: list[int], win: str, crop: str = "", size: int = 1536) -> Image.Image:
    """The tiles for slices ``idxs`` of one plane, labelled, in one grayscale image at most ``size`` px on its long side."""
    ax = AXIS[plane]
    tiles = []
    for i in idxs:
        img, aspect = view(hu, plane, i, spacing)
        if crop:
            x0, y0, x1, y1 = (float(v) for v in crop.split(","))
            h, w = img.shape
            img = img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
        tile = Image.fromarray(window(img, win))
        tile = tile.resize((tile.width, max(1, round(tile.height * aspect))), Image.BILINEAR)
        ImageDraw.Draw(tile).text((4, 2), f"{plane} {i} ({i * spacing[ax]:.0f} mm)", fill=255)
        tiles.append(tile)
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
    out = Image.new("L", (cols * tw, rows * th))
    for j, t in enumerate(tiles):
        out.paste(t, ((j % cols) * tw, (j // cols) * th))
    scale = size / max(out.size)
    if scale < 1:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.BILINEAR)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ct_dir", type=Path)
    ap.add_argument("--info", action="store_true", help="print shape, spacing and extent in mm, then stop")
    ap.add_argument("--plane", choices=tuple(AXIS), default="axial")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--at", default="")
    ap.add_argument("--from", dest="from_", type=float, default=0.0)
    ap.add_argument("--to", type=float, default=1.0)
    ap.add_argument("--window", default="lung")
    ap.add_argument("--crop", default="")
    ap.add_argument("--size", type=int, default=1536)
    ap.add_argument("--out", type=Path, default=Path("/tmp/view.png"))
    args = ap.parse_args()

    hu, meta = load(args.ct_dir)
    spacing = meta["spacing_mm"]
    if args.info:
        ext = [round(s * n) for s, n in zip(spacing, hu.shape)]
        print(json.dumps({"shape": hu.shape, "spacing_mm": spacing, "extent_mm": ext,
                          "orientation": meta["orientation"]}))
        return
    img = grid(hu, spacing, args.plane, indices(hu.shape[AXIS[args.plane]], args), args.window, args.crop, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out} ({img.width}x{img.height}, {len(indices(hu.shape[AXIS[args.plane]], args))} tiles)")


if __name__ == "__main__":
    main()
