"""Arm D's pre-rendered image grids (spec planning/vista_eval_vlm/2026-11-22-ct-findings-example-ct-arms.md, arm D
added 2026-11-23): for each VAL target with organ ranges, three organ-targeted grids of the scan itself and of each of
its five neighbours, drawn with the agent's own viewer (program_agentic_examples_d/tools/view_ct.py) so they look
exactly like the views it renders itself.

    lungs.png       axial, lung window, 16 slices from the bottom to the top of the lungs
    liver.png       axial, liver window, 12 slices from the bottom to the top of the liver
    chest_soft.png  axial, soft-tissue window, 16 slices from the bottom of the heart to the top of the lungs
                    (mediastinum, hila, axillae: the lymph-node question)

A scan without organ ranges (some neighbours) gets the same three windows over the whole scan, and a views.json that
says so. Output: config.EXAMPLE_VIEWS/<target id>/{target,n1..n5}/{lungs,liver,chest_soft}.png + views.json, on
the same disk as the work folders so staging hard-links them. Read-only once written.

    cd optimizer/ct-findings-5
    uv run -q --project $AGENTIC_LABEL_OPT --with pillow python -B scripts/build_views.py build   # fills the CT cache
    uv run -q --project $AGENTIC_LABEL_OPT --with pillow python -B scripts/build_views.py check

check prints PASS when every target has all 18 grids, each a readable PNG, and every views.json names its source.
Counts only: never an id, date or report text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import agentic  # noqa: E402  (cached_ct: the neighbour CTs)
from adapters import patient_dir  # noqa: E402

VIEWER = HERE / "program_agentic_examples_d" / "tools" / "view_ct.py"
#: name -> (window, slice count, organ giving the lower end, organ giving the upper end)
GRIDS = {"lungs": ("lung", 16, "lungs", "lungs"), "liver": ("liver", 12, "liver", "liver"),
         "chest_soft": ("soft", 16, "heart", "lungs")}
WHO = ("target", "n1", "n2", "n3", "n4", "n5")


def organ_range(ranges: dict, lower: str, upper: str) -> tuple[float, float] | None:
    if not ranges.get("available"):
        return None
    by = {o["organ"]: o for o in ranges["organs"] if o.get("present")}
    if lower not in by or upper not in by:
        return None
    return by[lower]["from_fraction"], by[upper]["to_fraction"]


def jobs_for(target: str, smap: dict) -> list[tuple]:
    out = []
    for who in WHO:
        ranges = json.loads((config.ORGAN_DIR / target / f"{who}.json").read_text(encoding="utf-8"))
        if who == "target":
            ct_dir = patient_dir(target)
        else:
            nid = smap[target][int(who[1]) - 1]
            agentic.cached_ct(nid, "meta.json")
            ct_dir = agentic.cached_ct(nid, "ct.npz").parent
        out.append((target, who, ct_dir, ranges))
    return out


def render(job) -> dict:
    target, who, ct_dir, ranges = job
    dest = config.EXAMPLE_VIEWS / target / who
    dest.mkdir(parents=True, exist_ok=True)
    info = {"source": "organ ranges (TotalSegmentator)" if ranges.get("available") else
            "organ ranges not available: each grid spans the whole scan", "grids": {}}
    for name, (win, count, lower, upper) in GRIDS.items():
        rng = organ_range(ranges, lower, upper) or (0.0, 1.0)
        out = dest / f"{name}.png"
        if not out.is_file():
            tmp = dest / f".{name}.{os.getpid()}.png"
            subprocess.run([sys.executable, "-B", str(VIEWER), str(ct_dir), "--plane", "axial", "--count", str(count),
                            "--from", f"{rng[0]:.4f}", "--to", f"{rng[1]:.4f}", "--window", win, "--out", str(tmp)],
                           check=True, capture_output=True)
            tmp.chmod(0o444)
            tmp.replace(out)
        info["grids"][name] = {"window": win, "slices": count, "from_fraction": round(rng[0], 4),
                               "to_fraction": round(rng[1], 4)}
    vj = dest / "views.json"
    if not vj.is_file():
        vj.write_text(json.dumps(info, indent=1), encoding="utf-8")
        vj.chmod(0o444)
    return info


def targets() -> list[str]:
    return sorted(p.name for p in config.ORGAN_DIR.iterdir() if (p / "target.json").is_file())


def cmd_build() -> int:
    smap = json.loads((config.DATA_DIR / "neighbours_scan_map.json").read_text(encoding="utf-8"))
    jobs = [j for t in targets() for j in jobs_for(t, smap)]
    with ThreadPoolExecutor(int(os.environ.get("VIEW_WORKERS", "12"))) as ex:
        done = list(ex.map(render, jobs))
    whole = sum(1 for d in done if d["source"].startswith("organ ranges not"))
    print(f"rendered {len(done)} scans x {len(GRIDS)} grids for {len(targets())} targets; {whole} scans had no "
          f"organ ranges and got whole-scan grids")
    return cmd_check()


def cmd_check() -> int:
    from PIL import Image

    ts = targets()
    bad = whole = 0
    for t in ts:
        for who in WHO:
            d = config.EXAMPLE_VIEWS / t / who
            try:
                info = json.loads((d / "views.json").read_text(encoding="utf-8"))
                whole += info["source"].startswith("organ ranges not")
                for name in GRIDS:
                    with Image.open(d / f"{name}.png") as im:
                        im.verify()
            except (OSError, ValueError, KeyError):
                bad += 1
    print(f"targets {len(ts)}; scans with all grids {len(ts) * len(WHO) - bad} of {len(ts) * len(WHO)}; "
          f"whole-scan grids (no organ ranges) {whole}")
    if bad or len(ts) != 190:
        print("FAIL: grids missing or unreadable, or not 190 targets")
        return 1
    print("PASS: example views ready")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("build", "check"):
        raise SystemExit(__doc__)
    sys.exit(cmd_build() if cmd == "build" else cmd_check())
