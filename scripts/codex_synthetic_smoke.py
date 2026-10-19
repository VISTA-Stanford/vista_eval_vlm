"""One full-size synthetic pair through the real Codex adapter (plan: vlm-radiology-progression-codex.md).

Two phantom chest volumes (512 x 512 x 400, air / body / a lesion that grows from prior to
current; no patient data), 250 + 250 chest-window slices each, the arm's prompt from the
mount, one sealed call. Pass: a readable answer on the four options, with tokens and time
printed. A fresh cache folder, so the call is real.

    .venv/bin/python scripts/codex_synthetic_smoke.py --config configs/radiology_progression.codex_pilot.yaml
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context.adapters.ct import CTAdapter  # noqa: E402
from models.codex_exec import CodexExecAdapter  # noqa: E402
from tasks.registry import load_valid_tasks  # noqa: E402
from vqa_dataset import IMAGES_LINE_TOKEN, images_line  # noqa: E402


def phantom(lesion_r, depth=400, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:512, 0:512]
    body = ((x - 256) / 200) ** 2 + ((y - 256) / 150) ** 2 < 1
    lungs = (((x - 180) / 70) ** 2 + ((y - 250) / 100) ** 2 < 1) | (((x - 332) / 70) ** 2 + ((y - 250) / 100) ** 2 < 1)
    vol = np.empty((512, 512, depth), dtype=np.float32)
    for z in range(depth):
        sl = np.full((512, 512), -1000.0)
        sl[body] = 40.0
        sl[lungs] = -850.0
        dz = abs(z - depth // 2)
        if dz < lesion_r:
            r = np.sqrt(lesion_r ** 2 - dz ** 2)
            sl[((x - 180) ** 2 + (y - 230) ** 2) < r ** 2] = 30.0
        vol[:, :, z] = sl + rng.normal(0, 8, (512, 512))
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="gpt-5.6-sol")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    task = load_valid_tasks(Path(cfg["paths"]["base_dir"]) / cfg["paths"]["valid_tasks"])[0]
    prompts = json.load(open(Path(cfg["paths"]["base_dir"]) / cfg["paths"]["prompts"]))
    ct = CTAdapter(config={"select": cfg["ct_arms"]["select"], "preprocess": cfg["ct_arms"]["preprocess"]})
    prior = ct.contextualize(phantom(12), {"model_type": "codex_exec"}).payload
    current = ct.contextualize(phantom(30, seed=1), {"model_type": "codex_exec"}).payload
    q = prompts[f"{task['task_name']}::ct_pair"].replace(
        IMAGES_LINE_TOKEN, images_line(len(prior), len(current), 120, cfg["ct_arms"]["preprocess"]["windowing"]), 1)
    a = CodexExecAdapter(args.name, cache_dir=tempfile.mkdtemp(prefix="vlm-smoke-cache-"))
    a.configure(cfg.get("codex", {}))
    a.load()
    t0 = time.time()
    out = a.infer(None, None, [{"question": q, "images": prior + current}], 1024, task["choices"])[0]
    print(f"images {len(prior)} + {len(current)} | {time.time() - t0:.0f} s | answer {out.get('text')!r} | "
          f"error {out.get('error')!r} | usage {out.get('usage')}")
    ok = out.get("text") in task["choices"] and not out.get("error")
    print("SMOKE " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
