"""Convert the TRAIN scans the items need that have no ct.npz yet, with ct-findings-5's own converter
(scripts/build_data.py convert: Hounsfield units, int16, RAS, ~2.5 mm slabs, no header, date or UID kept), into
OUT/scans/<scan id>/. Existing volumes (the ct-findings-5 TRAIN pool and TRAIN neighbours) are reused, not copied.

    cd optimizer/ct-findings-5
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B small_finding/convert.py run --workers 3
    uv run -q --no-project --with numpy python -B small_finding/convert.py check

check prints PASS when every item's scan folder has a readable ct.npz whose slab count matches the item's and every
item's slice index is inside it. Counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import items as it  # noqa: E402


def cmd_run(args) -> int:
    import build_data as bd

    todo = [(s["scan"], s["nifti"], str(it.SCANS_DIR)) for s in it.load_scans().values() if s["convert"]]
    if args.limit:
        todo = todo[: args.limit]
    it.SCANS_DIR.mkdir(parents=True, exist_ok=True)
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, detail) in enumerate(pool.map(bd.convert, todo), 1):
            status[st] += 1
            if detail:
                errors[detail] += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)} {dict(status)}", flush=True)
    print(f"done {len(todo)}: {dict(status)}; errors by type {dict(errors)}")
    return 1 if status["failed"] else 0


def cmd_check(_args) -> int:
    import numpy as np

    scans, items = it.load_scans(), it.load_items()
    shape, bad = {}, Counter()
    for s in scans.values():
        d = Path(s["ct_dir"])
        try:
            meta = json.loads((d / "meta.json").read_text())
            shape[s["scan"]] = int(meta["shape"][2])
            if (d / "ct.npz").stat().st_size == 0:
                bad["empty ct.npz"] += 1
        except Exception:  # noqa: BLE001
            bad["missing or unreadable"] += 1
    for i in items:
        n = shape.get(i["scan"])
        if n is None:
            continue
        if n != i["n_npz"]:
            bad["slab count differs from the item's"] += 1
        if not 0 <= i["z"] < n:
            bad["slice outside the volume"] += 1
    # one full read per 50 scans: the array opens and its depth matches meta.json
    for sid in sorted(shape)[::50]:
        hu = np.load(Path(scans[sid]["ct_dir"]) / "ct.npz")["hu"]
        if hu.shape[2] != shape[sid]:
            bad["array depth differs from meta.json"] += 1
    print(f"scans {len(scans)}, with a volume {len(shape)}, items {len(items)}; problems {dict(bad)}")
    print("PASS" if not bad and len(shape) == len(scans) else "FAIL")
    return 0 if not bad and len(shape) == len(scans) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("check")
    args = ap.parse_args()
    return {"run": cmd_run, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
