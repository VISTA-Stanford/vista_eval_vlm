"""Preflight for registry tasks read straight from BigQuery (tumor-board plan V1): no weights, no API calls.

For each task x arm in the config, on a fixed sample of rows (default 20 per task):
  * the prompt holds the question and every answer option;
  * arms that attach a CT carry exactly 30 slices on every row (60 with a prior CT, whose
    prompt must say which images are the prior);
  * no EHR event shown is dated after the prediction time (embed_time, a DATE, so
    after its midnight) -- any hit is leakage of the board itself and fails the run;
  * the full task row counts (scored labels, with a CT, with a summary, with a LUMIA
    file) are printed so nothing is hidden.
Writes the per-row capture (PHI) as JSONL under <results_dir>/golden/ on the mount and
prints aggregates only. Exit 1 on any failed check.

    cd src && python ../scripts/vlm_preflight.py --config ../configs/all_tasks.tb.gemini.yaml
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context.adapters.ehr import lumia_path_for  # noqa: E402
from vista_run.context_capture import iter_captured_examples  # noqa: E402
from vista_run.run_bq import TaskOrchestrator, _CT_EXPERIMENTS  # noqa: E402

_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--type", default="vertex_gemini")
    ap.add_argument("--name", default="gemini-3.1-pro-preview")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    orch = TaskOrchestrator(args.config, args.type, args.name)
    corpus = orch.cfg["retrieval"]["corpus_dir"]
    failures = []
    out_dir = Path(orch.results_base) / "golden" / "vlm_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in orch.cfg["tasks"]:
        info = next(t for t in orch.valid_tasks if t["task_name"] == task)
        options = info.get("choices") or ["Yes", "No"]

        orch.cfg.pop("row_sample", None)
        full, _, _ = orch._load_bq_table_task_data(info)
        has_ct = full["image_series_uid"].notna() & full["image_study_uid"].notna()
        has_sum = full["compact_summary"].notna() & (full["compact_summary"].astype(str).str.strip() != "")
        has_lumia = full["person_id"].apply(lambda p: lumia_path_for(p, corpus).exists())
        print(f"\n=== {task}: {len(full)} scored rows | with CT {int(has_ct.sum())} | with summary "
              f"{int(has_sum.sum())} | with LUMIA file {int(has_lumia.sum())} | splits "
              f"{full['split'].value_counts().to_dict()}")

        orch.cfg["row_sample"] = {"n": args.n, "seed": args.seed}
        for arm in orch.cfg["experiments"]:
            recs = []
            leaks = n_bad_opts = n_bad_ct = 0
            for ex in iter_captured_examples(orch, info, arm, args.type):
                raw = ex.raw_row
                prompt = ex.dynamic_prompt or ""
                if info["question"] not in prompt or not all(f'"{o}"' in prompt for o in options):
                    n_bad_opts += 1
                # 30 slices of the current CT, plus 30 of the prior when the task gives one.
                has_prior = pd.notna(raw.get("prior_image_study_uid")) if "prior_image_study_uid" in raw.index else False
                if arm in _CT_EXPERIMENTS and ex.image_count != (60 if has_prior else 30):
                    n_bad_ct += 1
                if has_prior and "PRIOR CT" not in prompt:
                    n_bad_opts += 1
                if arm not in _CT_EXPERIMENTS and ex.image_count != 0:
                    n_bad_ct += 1
                pred_time = pd.Timestamp(raw["embed_time"])
                times = [pd.Timestamp(t) for t in _TS.findall(ex.adapter_prompt_string or "")]
                late = [t for t in times if t > pred_time]
                leaks += bool(late)
                recs.append({"person_id": ex.person_id, "arm": arm, "image_count": ex.image_count,
                             "prompt_chars": len(prompt), "n_events_shown": len(times),
                             "last_event": str(max(times)) if times else None,
                             "prediction_time": str(pred_time), "n_after_prediction_time": len(late),
                             "dynamic_prompt": prompt})
            missing = orch.__dict__.get("_missing", {}).pop((task, arm), [])
            chars = pd.Series([r["prompt_chars"] for r in recs], dtype=float)
            print(f"  {arm:16s} rows {len(recs):3d} | missing {len(missing):2d} "
                  f"({sorted({m['reason'] for m in missing})}) | images "
                  f"{sorted({r['image_count'] for r in recs})} | prompt chars median "
                  f"{int(chars.median()) if len(chars) else 0} max {int(chars.max()) if len(chars) else 0} | "
                  f"rows with an event after prediction time: {leaks} | bad options {n_bad_opts} | bad CT {n_bad_ct}")
            if leaks:
                failures.append(f"{task}/{arm}: {leaks} rows show events after the prediction time")
            if n_bad_opts:
                failures.append(f"{task}/{arm}: {n_bad_opts} prompts missing the question or an option")
            if n_bad_ct:
                failures.append(f"{task}/{arm}: {n_bad_ct} rows with the wrong image count")
            if not recs:
                failures.append(f"{task}/{arm}: no rows captured")
            with open(out_dir / f"{task}_{arm}_vlm_preflight_golden.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")

    print("\nPREFLIGHT " + ("PASS" if not failures else "FAIL"))
    for msg in failures:
        print("  FAIL: " + msg)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
