"""Image packs: each recipe's slices rendered once to the mount, so weill and phil-sllm-01 send identical images.

A pack is one set of render settings (``Recipe.render_key``: sampler and slice count, windows, size,
file format, orientation), named by ``Recipe.pack_id``. Layout under ``<root>/packs/<pack_id>/``:

    pack.json                      the render settings
    <scan_id>/000.jpg ...          slices in image order (head to feet)
    <scan_id>/manifest.json        scan (study, series), slices in the volume, display and stored
                                   indices, spacing, windows, size, and a sha256 per file

``scan_id`` is a hash of the study and series UIDs, so paths carry no UID. Readers check every file
against its manifest hash before it is sent (``load``).

    python -m evalset.packs render --recipe medgemma --step 2 --limit 5 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from context import nifti, slices, windows
from evalset import paths
from evalset.recipes import Recipe, get


def scan_id(study, series) -> str:
    return hashlib.sha256(f"{study}__{series}".encode()).hexdigest()[:20]


def pack_dir(recipe: Recipe, n: int, root: Path | None = None) -> Path:
    return Path(root or paths.root()) / "packs" / recipe.pack_id(n)


def encode(rgb_u8, recipe: Recipe) -> bytes:
    img = Image.fromarray(rgb_u8)
    if recipe.size is not None and img.size != (recipe.size, recipe.size):
        img = img.resize((recipe.size, recipe.size), Image.BICUBIC)
    buf = io.BytesIO()
    if recipe.image_format == "jpeg":
        if recipe.jpeg_quality is None:
            img.save(buf, format="JPEG")  # PIL's defaults, as the MedGemma notebook saves
        else:
            img.save(buf, format="JPEG", quality=int(recipe.jpeg_quality))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def render_scan(recipe: Recipe, n: int, study, series, root: Path | None = None, commit: str | None = None,
                stack=None) -> dict:
    """Render one scan into the pack (kept if its manifest is already there). Returns the manifest. ``stack``: an
    already opened ``nifti.AxialStack`` for this scan, so several packs can share one read of the volume."""
    out = pack_dir(recipe, n, root) / scan_id(study, series)
    mf = out / "manifest.json"
    if mf.is_file():
        return json.loads(mf.read_text())
    if recipe.orientation != "radiological":
        raise ValueError("packs render radiology display only")
    if stack is None:
        stack = nifti.AxialStack.open(nifti.ct_path(study, series, mount_root=paths.mount_root()))
    display = slices.CT_SELECTORS[recipe.sampler](len(stack), n)
    out.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if recipe.image_format == "jpeg" else "png"
    files, hashes, size = [], [], None
    for j, k in enumerate(display):
        data = encode(windows.rgb_u8(stack[k], recipe.windows), recipe)
        name = f"{j:03d}.{ext}"
        (out / name).write_bytes(data)
        files.append(name)
        hashes.append(hashlib.sha256(data).hexdigest())
        if size is None:
            size = list(Image.open(io.BytesIO(data)).size)
    manifest = {"study": str(study), "series": str(series), "n_slices_in_volume": len(stack),
                "code_commit": commit if commit is not None else paths.code_commit(),
                "display_indices": display, "voxel_indices": [stack.voxel_index(k) for k in display],
                "spacing_mm": list(stack.spacing_mm), "windows": list(map(list, windows.RGB_SETS[recipe.windows])),
                "size": size, "files": files, "sha256": hashes, "render": recipe.render_key(n)}
    tmp = mf.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(mf)  # written last: a scan with a manifest is complete
    return manifest


def load(recipe: Recipe, n: int, study, series, root: Path | None = None) -> list[bytes]:
    """The scan's slice files, in image order, each checked against the manifest hash."""
    d = pack_dir(recipe, n, root) / scan_id(study, series)
    m = json.loads((d / "manifest.json").read_text())
    if m["render"] != recipe.render_key(n):
        raise ValueError(f"pack {d} was rendered with other settings")
    out = []
    for name, h in zip(m["files"], m["sha256"]):
        data = (d / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != h:
            raise ValueError(f"{d / name}: hash does not match the manifest")
        out.append(data)
    return out


def scans_for(df, n_scans: int):
    """(study, series) pairs a step's cases need, prior scans included."""
    pairs = list(zip(df["study"], df["series"]))
    if n_scans == 2:
        pairs += list(zip(df["prior_study"], df["prior_series"]))
    return sorted(set(pairs))


def _render_one(args):
    targets, study, series, root, commit = args
    try:
        todo = [(r, n) for r, n in targets
                if not (pack_dir(r, n, root) / scan_id(study, series) / "manifest.json").is_file()]
        if todo:
            # One read of the volume (the slow part, over gcsfuse) for every pack still missing this scan.
            stack = nifti.AxialStack.open(nifti.ct_path(study, series, mount_root=paths.mount_root()))
            for r, n in todo:
                render_scan(r, n, study, series, root, commit, stack)
        return None
    except Exception as e:  # noqa: BLE001 - one bad volume is reported as a count, not a crash
        return f"{type(e).__name__}"


def render_many(recipe: Recipe, n: int, pairs, workers: int = 2, root: Path | None = None) -> dict:
    return render_many_multi([(recipe, n)], pairs, workers, root)


def render_many_multi(targets, pairs, workers: int = 2, root: Path | None = None) -> dict:
    """Render every scan in ``pairs`` into each (recipe, slice count) pack in ``targets``, reading each volume once."""
    for r, n in targets:
        d = pack_dir(r, n, root)
        d.mkdir(parents=True, exist_ok=True)
        (d / "pack.json").write_text(json.dumps({"pack_id": r.pack_id(n), "render": r.render_key(n)}, indent=1))
    commit = paths.code_commit()  # once per invocation, not once per scan
    jobs = [(targets, s, r, root, commit) for s, r in pairs]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        errs = list(ex.map(_render_one, jobs))
    return {"packs": [r.pack_id(n) for r, n in targets], "scans": len(jobs),
            "failed": sum(e is not None for e in errs), "errors": sorted({e for e in errs if e})}


def main() -> int:
    from evalset import samples
    from evalset.cards import CARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render"])
    ap.add_argument("--recipe", required=True, help="a recipe, or several comma-separated (one read per scan)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N cases of the step (check runs)")
    ap.add_argument("--n-slices", type=int, default=None, help="override the recipe's slice count")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", default=None, help="arms YAML (see evalset.arms); with --variant")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--which", choices=["sample", "test"], default="sample")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from evalset.run import recipe_for
    n_scans = len(CARDS[args.step].scan_names)
    targets = []
    for name in args.recipe.split(","):
        recipe, _ = recipe_for(name.strip(), args.arms, args.variant)
        targets.append((recipe, args.n_slices or recipe.slices_per_scan(n_scans)))
    df = samples.load(args.step, which=args.which)
    if args.limit:
        df = check_cases(df, args.step, args.limit)
    print(json.dumps(render_many_multi(targets, scans_for(df, n_scans), args.workers)))
    return 0


def check_cases(df, step: int, k: int):
    """The first ``k`` cases of a step in a fixed order (sorted by a hash of the case id): the check-run set."""
    order = df["case_id"].map(lambda c: hashlib.sha256(f"check:{step}:{c}".encode()).hexdigest())
    return df.assign(_o=order).sort_values("_o").head(k).drop(columns="_o")


if __name__ == "__main__":
    raise SystemExit(main())
