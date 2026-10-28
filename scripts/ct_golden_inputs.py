"""Hash the CT images existing runs send, so a refactor of the CT code can be checked byte for byte.

Runs the real loader (``vqa_dataset.PromptDataset.__getitem__``: blob resolution, NIfTI read, slice
choice, windowing, padding, prior-then-current order) for each arm type a past run used, on a few
scans, and writes one JSON: arm -> scan position -> slice indices and a sha256 per image. Run it
once on the old code and once on the new, then compare the two files (``--compare``).

    uv run python scripts/ct_golden_inputs.py --scans SCANS.json --out before.json
    uv run python scripts/ct_golden_inputs.py --scans SCANS.json --out after.json --compare before.json

SCANS.json is a list of {"study", "series", "prior_study", "prior_series"} (prior optional). Keep it
on the mount or in job tmp, never in git. The output names scans by position only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_dataset import PromptDataset  # noqa: E402

# (arm name, experiment, model_type, ct_select, ct_preprocess, ct_windowing, use prior)
ARMS = [
    ("gemma_no_timeline_50", "no_timeline", "gemma3", None, None, None, False),
    ("gemma_no_report_10", "no_report", "gemma3", None, None, None, False),
    ("qwen_axial_all_30_grey", "axial_all_image", "qwen3vl", None, None, None, False),
    ("lesion_panel_up_to_50_chest", "ct_current", "aihub_openai", {"fn": "up_to_k", "k": 50},
     {"windowing": "chest_rgb", "target_size": 512}, None, False),
    ("radprog_pair_up_to_25_chest", "ct_pair", "aihub_openai", {"fn": "up_to_k", "k": 25},
     {"windowing": "chest_rgb", "target_size": 512}, None, True),
    ("board_rev5_axial_30_colour", "axial_all_image", "vertex_gemini", None, None, "multi_window_rgb", True),
]


def image_hash(im) -> str:
    h = hashlib.sha256()
    h.update(f"{im.mode}|{im.size}|".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def run(scans: list[dict]) -> dict:
    out = {}
    for name, exp, model_type, select, pre, windowing, use_prior in ARMS:
        rows = []
        for s in scans:
            r = {"dynamic_prompt": "q", "image_study_uid": s["study"], "image_series_uid": s["series"]}
            if use_prior and s.get("prior_study"):
                r.update(prior_image_study_uid=s["prior_study"], prior_image_series_uid=s["prior_series"])
            rows.append(r)
        ds = PromptDataset(pd.DataFrame(rows), experiment=exp, model_type=model_type, ct_select=select,
                           ct_preprocess=pre, ct_windowing=windowing, fail_on_ct_error=True)
        out[name] = {}
        for i in range(len(ds)):
            item = ds[i]
            imgs = item["image"] or []
            out[name][str(i)] = {"load_error": item["load_error"], "indices": item["selected_indices"],
                                 "hashes": [image_hash(im) for im in imgs]}
    return out


def compare(a: dict, b: dict) -> list[str]:
    diffs = []
    for arm in sorted(set(a) | set(b)):
        for scan in sorted(set(a.get(arm, {})) | set(b.get(arm, {}))):
            x, y = a.get(arm, {}).get(scan), b.get(arm, {}).get(scan)
            if x != y:
                diffs.append(f"{arm} scan {scan}: differs")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()
    result = run(json.loads(args.scans.read_text()))
    args.out.write_text(json.dumps(result, indent=1))
    n_img = sum(len(v["hashes"]) for arm in result.values() for v in arm.values())
    print(f"{len(result)} arms, {n_img} images hashed -> {args.out}")
    if args.compare:
        diffs = compare(json.loads(args.compare.read_text()), result)
        print("IDENTICAL to " + str(args.compare) if not diffs else "DIFFERENT:\n" + "\n".join(diffs))
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
