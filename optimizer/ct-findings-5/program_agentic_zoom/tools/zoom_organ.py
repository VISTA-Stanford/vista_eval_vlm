"""Zoom into one organ on one axial slab of the CT, using the scan's organ mask, and write it as a PNG you can open
with the Read tool.

The organ mask (organ_mask.npz beside CT_DIR) is TotalSegmentator's automatic segmentation, one value per voxel on
the same slab grid as ct.npz; it may be wrong on unusual scans. For the organ you name, the image is that slab cut
to the organ's outline on that slab (plus a 10% margin) and enlarged to 1,024 pixels on its long side, in the
organ's window, drawn the radiology way (patient's right on the image's left, anterior at the top). A strip above
the image names the organ, the slab and the window.

    python3 -B zoom_organ.py CT_DIR --info
    python3 -B zoom_organ.py CT_DIR --organ liver --slab 40 --out /tmp/liver_40.png
    python3 -B zoom_organ.py CT_DIR --organ right_upper_lobe --slab 101 --box 0.5,0.2,0.9,0.6 --out /tmp/rul_box.png

--info lists each organ's slab range (the slabs where the mask has it). --slab is a slab index of ct.npz (the
viewer's axial slice index). --box x0,y0,x1,y1 enlarges a part of the organ crop instead, as fractions of that crop
(0,0 its top left, 1,1 its bottom right), cut from the full-resolution slice. --window overrides the organ's window:
lung (-1350..150), soft (-160..240), bone (-500..1300), liver (-40..160), or LO,HI in HU.

Organs: liver; lungs, right_lung, left_lung, right_upper_lobe, right_middle_lobe, right_lower_lobe, left_upper_lobe,
left_lower_lobe; lymph_nodes (the mask has no nodes: this is the middle half of the lungs' outline across, the
mediastinum and both hila, over the lungs' height). One slab per image; at most 40 images per scan. Every image is
noted in views.jsonl beside CT_DIR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import view_ct  # noqa: E402  (same folder: the grid viewer's orientation and windows)

MAX_IMAGES = 40
#: Vertex refuses a session request over 30 MB, and every opened image stays in the session: zoomed images also stop
#: at 18 MB in all (about 24 MB once encoded for the request), leaving room for the grid viewer's images.
MAX_BYTES = 18_000_000
ZOOM_PX = 1024
MARGIN = 0.10
STRIP_PX = 22
#: Bits of organ_mask.npz "bits" (scripts/build_masks.py; TotalSegmentator v2 labels liver 5, lobes 10-14).
BITS = {"liver": 1, "left_upper_lobe": 2, "left_lower_lobe": 4, "right_upper_lobe": 8, "right_middle_lobe": 16,
        "right_lower_lobe": 32}
LUNGS = 2 | 4 | 8 | 16 | 32
ORGANS = {"liver": 1, "lungs": LUNGS, "right_lung": 8 | 16 | 32, "left_lung": 2 | 4, "lymph_nodes": LUNGS,
          **{k: v for k, v in BITS.items() if k != "liver"}}
WINDOW_OF = {"liver": "liver", "lymph_nodes": "soft"}  # every lung entry: lung


def organ_box(mask2d: np.ndarray, organ: str, margin: float = MARGIN):
    """(x0, y0, x1, y1) in display pixels of the organ on this slab, with the margin, or None if absent. The same
    rule as small_finding/render.py organ_box (lymph nodes: the middle half of the lungs' box across)."""
    ys, xs = np.nonzero(mask2d)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if organ == "lymph_nodes":
        w = x1 - x0
        x0, x1 = x0 + w // 4, x1 - w // 4
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    h, w = mask2d.shape
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def enlarge(img: Image.Image, box) -> Image.Image:
    crop = img.crop(box)
    scale = ZOOM_PX / max(crop.size)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)


def slab_image(hu: np.ndarray, spacing, z: int, win: str) -> Image.Image:
    arr, aspect = view_ct.view(hu, "axial", z, spacing)
    img = Image.fromarray(view_ct.window(arr, win))
    return img.resize((img.width, max(1, round(img.height * aspect))), Image.BILINEAR)


def slab_mask(bits: np.ndarray, spacing, z: int, organ: str) -> np.ndarray:
    """The organ on slab z, in the same display orientation and size as slab_image."""
    on = ((bits[:, :, z] & ORGANS[organ]) > 0).astype(np.uint8)[:, :, None]
    arr, aspect = view_ct.view(on, "axial", 0, spacing)
    img = Image.fromarray(arr * 255)
    img = img.resize((img.width, max(1, round(img.height * aspect))), Image.NEAREST)
    return np.asarray(img) > 0


def organ_slabs(bits: np.ndarray, organ: str) -> list[int]:
    return [int(z) for z in np.nonzero(((bits & ORGANS[organ]) > 0).any(axis=(0, 1)))[0]]


def sub_box(box, frac: str):
    """A box inside ``box`` from fractions x0,y0,x1,y1 of it (clipped to it, at least one pixel)."""
    f = [min(max(float(v), 0.0), 1.0) for v in frac.split(",")]
    if len(f) != 4 or f[2] <= f[0] or f[3] <= f[1]:
        raise ValueError("--box needs x0,y0,x1,y1 with x1 > x0 and y1 > y0, as fractions 0..1")
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    nx0, ny0 = x0 + int(f[0] * w), y0 + int(f[1] * h)
    return nx0, ny0, max(nx0 + 1, x0 + int(round(f[2] * w))), max(ny0 + 1, y0 + int(round(f[3] * h)))


def images_drawn(log: Path) -> tuple[int, int]:
    """(zoomed images drawn so far, their bytes) for this scan."""
    if not log.is_file():
        return 0, 0
    n = size = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("tool") == "zoom_organ" and "out" in e:
            n += 1
            size += int(e.get("bytes", 0))
    return n, size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ct_dir", type=Path)
    ap.add_argument("--info", action="store_true", help="list each organ's slab range, then stop")
    ap.add_argument("--organ", choices=tuple(ORGANS))
    ap.add_argument("--slab", type=int)
    ap.add_argument("--box", default="")
    ap.add_argument("--window", default="")
    ap.add_argument("--out", type=Path, default=Path("/tmp/zoom.png"))
    args = ap.parse_args()

    folder = args.ct_dir.resolve().parent
    log = folder / "views.jsonl"
    bits = np.load(folder / "organ_mask.npz")["bits"]
    if args.info:
        out = {"slabs": int(bits.shape[2]), "images_left": MAX_IMAGES - images_drawn(log)[0], "organs": {}}
        for organ in ORGANS:
            zs = organ_slabs(bits, organ)
            out["organs"][organ] = {"from_slab": zs[0], "to_slab": zs[-1], "slabs_with_organ": len(zs)} if zs else None
        print(json.dumps(out, indent=1))
        return 0
    if not args.organ or args.slab is None:
        ap.error("name --organ and --slab (or use --info)")
    n, size = images_drawn(log)
    if n >= MAX_IMAGES or size >= MAX_BYTES:
        view_ct.log_view(args.ct_dir, {"tool": "zoom_organ", "refused": "image limit", "organ": args.organ,
                                       "slab": args.slab})
        print(f"REFUSED: this scan has already had {n} zoomed images, the limit. Answer from what you have seen, "
              f"or use view_ct.py.")
        return 3
    hu, meta = view_ct.load(args.ct_dir)
    spacing = meta["spacing_mm"]
    if bits.shape != hu.shape:
        raise SystemExit(f"organ mask {bits.shape} does not match the CT {hu.shape}")
    zs = organ_slabs(bits, args.organ)
    if not 0 <= args.slab < hu.shape[2] or args.slab not in zs:
        where = f"it is on slabs {zs[0]}-{zs[-1]}" if zs else "the mask does not have it on this scan"
        view_ct.log_view(args.ct_dir, {"tool": "zoom_organ", "refused": "organ not on slab", "organ": args.organ,
                                       "slab": args.slab})
        print(f"{args.organ} is not on slab {args.slab}: {where}. Nothing drawn.")
        return 2
    win = args.window or WINDOW_OF.get(args.organ, "lung")
    full = slab_image(hu, spacing, args.slab, win)
    m = slab_mask(bits, spacing, args.slab, args.organ)
    box = organ_box(m, args.organ)
    if args.box:
        box = sub_box(box, args.box)
    img = enlarge(full, box)
    canvas = Image.new("L", (img.width, img.height + STRIP_PX))
    canvas.paste(img, (0, STRIP_PX))
    label = (f"{args.organ}, slab {args.slab} (organ on slabs {zs[0]}-{zs[-1]}), {win} window"
             + (f", box {args.box}" if args.box else ""))
    ImageDraw.Draw(canvas).text((4, 4), label, fill=255)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    view_ct.log_view(args.ct_dir, {"tool": "zoom_organ", "organ": args.organ, "slab": args.slab, "box": args.box,
                                   "window": win, "out": str(args.out.resolve()), "bytes": args.out.stat().st_size})
    left = MAX_IMAGES - images_drawn(log)[0]
    print(f"wrote {args.out} ({canvas.width}x{canvas.height}); {left} zoomed images left for this scan")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - noted as a tool crash for the usage count, then shown to the session
        try:
            ct = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
            if ct:
                view_ct.log_view(ct, {"tool": "zoom_organ", "crash": type(e).__name__})
        finally:
            raise
