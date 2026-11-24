"""Render each item's slice two ways, with the agent's own viewer functions
(program_agentic_examples_d/tools/view_ct.py: same windows, radiology orientation, aspect correction):

    full   the whole axial slice at native resolution (about 0.7 mm a pixel), organ window
    tile   the same slice scaled to an arm D grid tile: arm D's grids are 4 columns at 1536 px, so 384 px on the
           long side (about 1 mm a pixel)

Windows: lung window for both lung tracks, liver window for liver, soft-tissue window for lymph nodes. No label is
drawn on the image (arm D's tiles carry the slice index; here that would be a hint). PNGs go to a local cache
(not the mount: many small files), CACHE/<item>_<view>.png.

    cd optimizer/ct-findings-5
    uv run -q --no-project --with numpy --with pillow python -B small_finding/render.py build --workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "program_agentic_examples_d" / "tools"))

import items as it  # noqa: E402

CACHE = Path(os.environ.get("SF_IMAGES", Path.home() / ".cache" / "small-finding-perception" / "images"))
TILE_PX = 1536 // 4
WINDOW = {"lung_solid": "lung", "lung_ground_glass": "lung", "liver": "liver", "lymph_node": "soft"}
WINDOW_NAME = {"lung": "lung window", "liver": "liver window", "soft": "soft-tissue window"}


def slice_image(hu, spacing, z: int, win: str):
    import view_ct
    from PIL import Image

    arr, aspect = view_ct.view(hu, "axial", int(z), spacing)
    img = Image.fromarray(view_ct.window(arr, win))
    return img.resize((img.width, max(1, round(img.height * aspect))), Image.BILINEAR)


def tile_of(img):
    from PIL import Image

    scale = TILE_PX / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR) if scale < 1 else img


def path_of(item_id: str, view: str) -> Path:
    return CACHE / f"{item_id}_{view}.png"


# zoom view (Phil 2026-11-24): crop the full slice to the organ's box on that slice from the TotalSegmentator mask,
# plus a 10% margin, and enlarge so the long side is ZOOM_PX. Labels are TotalSegmentator v2 numbering, matched on
# 2026-11-24 against the organ table (liver 5, lung lobes 10-14, heart 51). Lymph nodes have no label: their box is
# the middle half of the lungs' box across (mediastinum and both hila), the lungs' full height.
MASKS = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/organ_localization/masks_thoracic_june26")
ZOOM_PX = 1024
ZOOM_LABELS = {"lung_solid": (10, 11, 12, 13, 14), "lung_ground_glass": (10, 11, 12, 13, 14), "liver": (5,),
               "lymph_node": (10, 11, 12, 13, 14)}


def organ_box(mask2d, track: str, margin: float = 0.10):
    """(x0, y0, x1, y1) in display pixels of the organ on this slice, with the margin, or None if absent."""
    import numpy as np

    ys, xs = np.nonzero(mask2d)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if track == "lymph_node":
        w = x1 - x0
        x0, x1 = x0 + w // 4, x1 - w // 4
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    h, w = mask2d.shape
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def zoom_of(full, box):
    from PIL import Image

    crop = full.crop(box)
    scale = ZOOM_PX / max(crop.size)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)


def display_mask(mask3d, z: int, k: int, labels, spacing):
    """The organ mask for ct.npz slab z (the union over the k NIfTI slices averaged into it), in the same display
    orientation and aspect as slice_image."""
    import numpy as np
    import view_ct
    from PIL import Image

    m = np.isin(mask3d[:, :, z * k:(z + 1) * k], labels).any(axis=2)[:, :, None]
    arr, aspect = view_ct.view(m.astype(np.uint8), "axial", 0, spacing)
    img = Image.fromarray(arr * 255)
    img = img.resize((img.width, max(1, round(img.height * aspect))), Image.NEAREST)
    return np.asarray(img) > 0


def _zoom_scan(job) -> tuple[int, int, int]:
    """(rendered, already there, organ absent on that slice) for one scan's items."""
    import nibabel as nib
    import numpy as np

    ct_dir, nifti_name, rows = job
    todo = [r for r in rows if not path_of(r["item"], "zoom").is_file() and not path_of(r["item"], "zoom.none").is_file()]
    if not todo:
        return 0, len(rows), 0
    meta = json.loads((Path(ct_dir) / "meta.json").read_text())
    hu = np.load(Path(ct_dir) / "ct.npz")["hu"]
    mask = np.asarray(nib.as_closest_canonical(nib.load(str(MASKS / nifti_name))).dataobj)
    k = int(meta.get("slab_averaged", 1))
    made = absent = 0
    for r in todo:
        full = slice_image(hu, meta["spacing_mm"], r["z"], WINDOW[r["track"]])
        dm = display_mask(mask, r["z"], k, ZOOM_LABELS[r["track"]], meta["spacing_mm"])
        box = organ_box(dm, r["track"]) if dm.shape == (full.height, full.width) else None
        if box is None:
            path_of(r["item"], "zoom.none").touch()  # counted, and the item has no zoom view
            absent += 1
            continue
        zoom_of(full, box).save(path_of(r["item"], "zoom"))
        made += 1
    return made, len(rows) - len(todo), absent


def cmd_zoom(args) -> int:
    scans = it.load_scans()
    by_scan = defaultdict(list)
    for i in it.load_items():  # examples too: layer 2 on the zoom view shows zoomed references
        by_scan[(scans[i["scan"]]["ct_dir"], i["name"])].append({"item": i["item"], "z": i["z"], "track": i["track"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    made = kept = absent = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(d, n, rows) for (d, n), rows in by_scan.items()]
        for j, (m, k, a) in enumerate(pool.map(_zoom_scan, jobs), 1):
            made, kept, absent = made + m, kept + k, absent + a
            if j % 100 == 0:
                print(f"{j}/{len(jobs)} scans; zoom rendered {made}, already there {kept}, organ absent {absent}", flush=True)
    print(f"done: {len(by_scan)} scans; zoom rendered {made}, already there {kept}, organ absent on the slice {absent}")
    return 0


def _render_scan(job) -> tuple[int, int]:
    import numpy as np

    ct_dir, rows = job
    todo = [r for r in rows if not all(path_of(r["item"], v).is_file() for v in r["views"])]
    if not todo:
        return 0, len(rows)
    meta = json.loads((Path(ct_dir) / "meta.json").read_text())
    hu = np.load(Path(ct_dir) / "ct.npz")["hu"]
    for r in todo:
        full = slice_image(hu, meta["spacing_mm"], r["z"], WINDOW[r["track"]])
        if "full" in r["views"]:
            full.save(path_of(r["item"], "full"))
        if "tile" in r["views"]:
            tile_of(full).save(path_of(r["item"], "tile"))
    return len(todo), len(rows) - len(todo)


def cmd_build(args) -> int:
    scans = it.load_scans()
    by_scan = defaultdict(list)
    for i in it.load_items():
        views = ["full"] if i["role"] == "example" else ["full", "tile"]
        by_scan[scans[i["scan"]]["ct_dir"]].append({"item": i["item"], "z": i["z"], "track": i["track"], "views": views})
    CACHE.mkdir(parents=True, exist_ok=True)
    made = kept = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (m, k) in enumerate(pool.map(_render_scan, list(by_scan.items())), 1):
            made, kept = made + m, kept + k
            if n % 100 == 0:
                print(f"{n}/{len(by_scan)} scans; items rendered {made}, already there {kept}", flush=True)
    print(f"done: {len(by_scan)} scans; items rendered {made}, already there {kept}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=3)
    z = sub.add_parser("zoom")
    z.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    return {"build": cmd_build, "zoom": cmd_zoom}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
