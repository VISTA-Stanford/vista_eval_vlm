"""Copy the image packs a run will read from the mount to a local eval-set root, many files at once.

The mount costs ~0.15 s per file, and ``run`` reads every slice of every case one after another before its first
call: ~4.6 scans a minute on weill (10-31), so step 3's whole TEST split would take more than a day just to read.
Read 32 at a time the same files come ~9x faster. The local root links ``samples`` and ``runs`` back to the mount,
so only the image reads move; answers still land on the mount. Pixels cannot change: ``packs.load`` checks every
slice against its manifest hash, and the manifest is copied last (a scan with a manifest is complete).

    python -m evalset.stage --recipe medgemma --steps 1 3 --which test --limit 50 --to /data/.../evalset_local
    EVALSET_ROOT=/data/.../evalset_local bash scripts/evalset_weill.sh   # then run from the local root

Prints counts only, never an identifier.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalset import packs, paths, samples
from evalset.cards import CARDS
from evalset.recipes import get


def mount_eval_root() -> Path:
    """The eval-set root on the mount, whatever EVALSET_ROOT says (it may already point at the local root)."""
    return paths.mount_root() / paths.BUCKET / paths.SUBDIR


def prepare_root(dest: Path, src: Path) -> None:
    """``dest`` with ``samples`` and ``runs`` linked to ``src``'s and an own ``packs`` folder."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("samples", "runs"):
        link = dest / name
        if link.is_symlink():
            if link.resolve() != (src / name).resolve():
                raise RuntimeError(f"{link} points at {link.resolve()}, not {src / name}")
        elif link.exists():
            raise RuntimeError(f"{link} exists and is not a link to the mount")
        else:
            link.symlink_to(src / name)
    (dest / "packs").mkdir(exist_ok=True)


def scans_needed(recipe, step: int, which: str, limit: int | None, root: Path) -> list[tuple[int, str, str]]:
    """(slices per scan, study, series) for every scan the run of this step would load, in the run's case order."""
    card = CARDS[step]
    df = samples.load(step, root, which)
    if limit:
        df = packs.check_cases(df, step, limit)
    n = recipe.slices_per_scan(len(card.scan_names))
    out = [(n, r["study"], r["series"]) for _, r in df.iterrows()]
    if len(card.scan_names) > 1:
        out += [(n, r["prior_study"], r["prior_series"]) for _, r in df.iterrows()]
    return out


def copy_scan(recipe, n: int, study, series, src: Path, dest: Path) -> str:
    """'copied', 'present' or 'missing' (no complete pack on the mount)."""
    sid = packs.scan_id(study, series)
    a, b = packs.pack_dir(recipe, n, src) / sid, packs.pack_dir(recipe, n, dest) / sid
    if (b / "manifest.json").is_file():
        return "present"
    if not (a / "manifest.json").is_file():
        return "missing"
    b.mkdir(parents=True, exist_ok=True)
    for f in sorted(a.iterdir()):
        if f.name != "manifest.json":
            shutil.copyfile(f, b / f.name)
    shutil.copyfile(a / "manifest.json", b / "manifest.json.tmp")
    (b / "manifest.json.tmp").replace(b / "manifest.json")
    return "copied"


def stage(recipe, steps, which: str, limit: int | None, dest: Path, workers: int = 32) -> dict:
    src = mount_eval_root()
    prepare_root(dest, src)
    todo = sorted({s for step in steps for s in scans_needed(recipe, step, which, limit, src)})
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: copy_scan(recipe, *t, src, dest), todo))
    counts = {k: res.count(k) for k in ("copied", "present", "missing")}
    if counts["missing"]:
        raise RuntimeError(f"{counts['missing']} scans have no complete pack on the mount; render them first")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--which", choices=["sample", "test"], default="sample")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--to", required=True, help="local eval-set root (created; samples and runs link to the mount)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    counts = stage(get(args.recipe), args.steps, args.which, args.limit, Path(args.to), args.workers)
    print(f"steps {args.steps} ({args.which}, limit {args.limit or 'all'}): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
