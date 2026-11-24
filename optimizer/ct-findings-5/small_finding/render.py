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
    args = ap.parse_args()
    return {"build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
