"""Preflight for the radiology-progression VLM run: no model call, no API (plan:
planning/vista_eval_vlm/vlm-radiology-progression-codex.md).

On the config's sample (the pilot: 10 per class from test), for each image-only arm:
  * the sample is class-balanced and test-only, one row per scan;
  * the pair arm has a prior on every row and the current-only arm none;
  * every row's CTs load; the image count equals what the prompt's IMAGES line says
    (prior first, then current), each scan at most the arm's cap and at native 512 px;
  * the prompt holds the four options and none of the row's report-derived text;
  * the encoded upload per row (the adapter's own image format) stays under 120 MB, the
    largest request tested through Codex (10-15);
  * the token estimate stays under the model's window.
Prints aggregates only (no ids). Exit 1 on any failed check.

    .venv/bin/python scripts/radprog_preflight.py --config configs/radiology_progression.codex_pilot.yaml
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.codex_exec import encode_image  # noqa: E402
from vista_run.run_bq import IMAGE_ONLY_EXPERIMENTS, TaskOrchestrator  # noqa: E402
from vqa_dataset import PromptDataset  # noqa: E402

# Largest request tested through Codex: 120 MB passed (470 JPEG images, 10-15); 165 MB of PNG failed.
MAX_UPLOAD_MB = 120
LINE = re.compile(r"first (\d+) images are axial slices of the PRIOR CT.*?the next (\d+) are|the (\d+) images are axial slices of the CURRENT CT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--type", default="codex_exec")
    ap.add_argument("--name", default="gpt-5.6-sol")
    args = ap.parse_args()
    orch = TaskOrchestrator(args.config, args.type, args.name)
    cap = orch._ct_arm_cap()
    fails = []
    for task in orch.cfg["tasks"]:
        info = next(t for t in orch.valid_tasks if t["task_name"] == task)
        df, col, _ = orch._load_bq_table_task_data(info)
        per_class = int(orch.cfg["row_sample"]["per_class"])
        counts = df["label"].value_counts().sort_index().to_dict()
        print(f"\n=== {task}: {len(df)} rows, per class {counts}, splits {df['split'].value_counts().to_dict()}")
        if set(counts.values()) != {per_class} or set(df["split"]) != {"test"} or df["index"].duplicated().any():
            fails.append(f"{task}: sample is not {per_class} per class, test-only, one row per scan")
        for arm in orch.cfg["experiments"]:
            if arm not in IMAGE_ONLY_EXPERIMENTS:
                continue
            exp = orch._build_prompts_for_experiment(df, info, arm, col)
            ct_arms = orch.cfg["ct_arms"]
            ds = PromptDataset(exp, prompt_col="dynamic_prompt", experiment=arm, model_type=args.type,
                               ct_snapshot_prefix=orch.cfg["paths"].get("ct_snapshot_prefix"),
                               mount_root=orch.cfg["paths"].get("mount_root"), fail_on_ct_error=True,
                               ct_select=ct_arms["select"], ct_preprocess=ct_arms.get("preprocess"))
            has_prior = exp["prior_image_study_uid"].notna().sum() if "prior_image_study_uid" in exp.columns else 0
            n_img, mb, bad = [], [], 0
            for i in range(len(ds)):
                item = ds[i]
                q = item["question"]
                if item["load_error"] or not item["image"]:
                    bad += 1
                    continue
                m = LINE.search(q)
                n_prior, n_cur = (int(m.group(1)), int(m.group(2))) if m and m.group(1) else (0, int(m.group(3)) if m else -1)
                imgs = item["image"]
                ok = (m is not None and len(imgs) == n_prior + n_cur and max(n_prior, n_cur) <= cap
                      and all(im.size == (512, 512) and im.mode == "RGB" for im in imgs)
                      and all(f'"{c}"' in q for c in info["choices"]) and "[IMAGES_LINE]" not in q)
                row = item["raw_row"]
                for c in ("label_description", "raw_label"):
                    v = row.get(c)
                    if isinstance(v, str) and v and len(v) > 25 and v in q:
                        ok = False  # report-derived text in the prompt
                if (arm == "ct_pair") != (n_prior > 0):
                    ok = False
                if not ok:
                    bad += 1
                n_img.append(len(imgs))
                mb.append(sum(len(encode_image(im, getattr(orch.adapter, "image_format", "jpeg"), orch.adapter.jpeg_quality))
                              for im in imgs) / 1e6)
            tokens = [n * orch.adapter.image_token_estimate + 12000 for n in n_img]
            print(f"[{arm}] rows {len(exp)} | with prior {int(has_prior)} | bad rows {bad} | images per row "
                  f"median {int(np.median(n_img)) if n_img else 0} max {max(n_img) if n_img else 0} | upload MB median "
                  f"{np.median(mb):.1f} max {max(mb) if mb else 0:.1f} | est. tokens max {max(tokens) if tokens else 0}")
            if bad:
                fails.append(f"{arm}: {bad} rows failed")
            if arm == "ct_pair" and has_prior != len(exp):
                fails.append(f"{arm}: not every row has a prior")
            if arm == "ct_current" and has_prior:
                fails.append(f"{arm}: rows carry a prior")
            if mb and max(mb) > MAX_UPLOAD_MB:
                fails.append(f"{arm}: upload {max(mb):.0f} MB over {MAX_UPLOAD_MB}")
            if tokens and max(tokens) > orch._input_token_limit():
                fails.append(f"{arm}: token estimate over the window")
    print("\nPREFLIGHT " + ("PASSED" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
