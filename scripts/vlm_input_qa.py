"""Input QA over every scored case (tumor-board plan rev 5, layer B). No weights, no API calls.

For each task in the config, on ALL scored rows:
  EHR inputs (trimmed, no notes, full): the age/sex header is present and known; the board CT's own
  report was found and removed (or was never in the record); how many days before the prediction
  time the newest shown record is; whether the trim cut anything; nothing dated after the
  prediction time.
  CT inputs: slice count (30, or 60 with a prior), near-blank slices, and for two-scan rows that
  the prior is dated before the current CT.
Writes one row per case and input to <results_dir>/input_qa/<task>_input_qa.csv on the mount (holds
person ids: mount only, never git) and prints totals only. Exit 1 on any hard failure (an event after
the prediction time, a missing header, the report still present, a wrong slice count, a prior that
isn't earlier).

    cd src && python ../scripts/vlm_input_qa.py --config ../configs/all_tasks.tb.gemini.yaml [--skip-ct]
"""

import argparse
import copy
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context.adapters.ehr import EHRAdapter, lumia_path_for, parse_lumia  # noqa: E402
from context.presets import get_preset  # noqa: E402
from data_tools.utils.meds_timeline_utils import KEEP_LATEST_MARKER  # noqa: E402
from vista_run.run_bq import TaskOrchestrator, attach_prior_ct, _CT_EXPERIMENTS  # noqa: E402

_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")
EHR_ARMS = ("no_image", "no_notes", "no_image_uncut")
_CFG = {}


def _ehr_row(args):
    """Render one patient's three EHR inputs the way the runner does, and measure them."""
    pid, embed_time, note_id = args
    cfg = _CFG
    ev = parse_lumia(lumia_path_for(pid, cfg["corpus"]))
    rep = (ev["note_id"].astype(str).eq(str(int(note_id))) & ev["code"].astype(str).str.startswith("STANFORD_NOTE")) \
        if pd.notna(note_id) else pd.Series(False, index=ev.index)
    report_in_record = bool(rep.any())
    # Inside the shown 24-month window ending at the prediction time (midnight): only then can it reach
    # the model. A report signed after the prediction day is outside the window already.
    t = pd.to_datetime(ev.loc[rep, "time"])
    pred_ts = pd.Timestamp(embed_time)
    report_in_window = bool(((t >= pred_ts - pd.DateOffset(months=24)) & (t <= pred_ts)).any())
    report_signed_after_prediction = bool((t > pred_ts).any())
    out = []
    for arm in EHR_ARMS:
        block = copy.deepcopy(next(b for b in get_preset(arm)["blocks"] if b["id"] == "ehr"))
        ctx = {"embed_time": embed_time, "timeline_truncation": cfg["trunc"],
               "demographics_header": cfg["header"],
               "drop_note_ids": [int(note_id)] if cfg["drop"] and pd.notna(note_id) else []}
        blk = EHRAdapter(config=block["config"]).contextualize(ev, ctx)
        text = blk.payload
        times = [pd.Timestamp(t) for t in _TS.findall(text)]
        pred = pd.Timestamp(embed_time)
        first = text.split("\n", 1)[0]
        out.append({
            "person_id": pid, "arm": arm, "chars": len(text),
            "header_present": first.startswith("PATIENT:"),
            "age_known": first.startswith("PATIENT:") and "age unknown" not in first,
            "sex_known": first.startswith("PATIENT:") and "sex unknown" not in first,
            "report_in_record": report_in_record, "report_in_window": report_in_window,
            "report_signed_after_prediction": report_signed_after_prediction,
            "report_removed": blk.metadata.get("dropped_report_events", 0) > 0,
            "trim_cut": KEEP_LATEST_MARKER in text or "[TRUNCATED" in text,
            "days_newest_record_before_prediction": (pred - max(times)).days if times else None,
            "n_after_prediction": sum(t > pred for t in times),
            "n_events_shown": len(times),
        })
    return out


def _ct_rows(orch, info, arm, df):
    from vista_run.context_capture import iter_captured_examples  # noqa: F401  (same capture path)
    from vqa_dataset import PromptDataset
    d = orch._build_prompts_for_experiment(df, info, arm, "patient_string")
    ds = PromptDataset(df=d, prompt_col="dynamic_prompt", experiment=arm, storage_client=orch.storage_client,
                       model_type=orch.model_type, ct_snapshot_prefix=orch.cfg["paths"].get("ct_snapshot_prefix"),
                       fail_on_ct_error=True, ct_windowing=orch.cfg.get("ct_windowing"))
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        raw, imgs = item["raw_row"], item["image"] or []
        stds = [float(np.asarray(im).std()) for im in imgs]
        has_prior = "prior_image_study_uid" in raw.index and pd.notna(raw.get("prior_image_study_uid"))
        rows.append({
            "person_id": raw["person_id"], "arm": arm, "n_images": len(imgs),
            "expected_images": 60 if has_prior else 30, "load_error": item.get("load_error"),
            "near_blank_slices": sum(s < 2.0 for s in stds),
            "has_prior": has_prior,
            "prior_gap_days": raw.get("prior_gap_days") if has_prior else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--type", default="vertex_gemini")
    ap.add_argument("--name", default="gemini-3.1-pro-preview")
    ap.add_argument("--skip-ct", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    orch = TaskOrchestrator(args.config, args.type, args.name)
    orch.cfg.pop("row_sample", None)
    _CFG.update(corpus=orch.cfg["retrieval"]["corpus_dir"], trunc=orch.cfg.get("timeline_truncation"),
                header=bool(orch.cfg.get("ehr_demographics_header")), drop=bool(orch.cfg.get("ehr_drop_current_ct_report")))
    out_dir = Path(orch.results_base) / "input_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    fails = []
    for task in orch.cfg["tasks"]:
        info = next(t for t in orch.valid_tasks if t["task_name"] == task)
        df, _, _ = orch._load_bq_table_task_data(info)
        with ProcessPoolExecutor(args.workers) as ex:  # forked workers inherit _CFG
            ehr = [r for rows in ex.map(_ehr_row, zip(df.person_id, df.embed_time, df.ct_note_id), chunksize=4) for r in rows]
        e = pd.DataFrame(ehr)
        print(f"\n=== {task}: {len(df)} scored cases")
        for arm, g in e.groupby("arm", sort=False):
            d = g["days_newest_record_before_prediction"].dropna()
            print(f"  {arm:15s} header {int(g.header_present.sum())}/{len(g)} (age known {int(g.age_known.sum())}, sex known {int(g.sex_known.sum())}) | "
                  f"report in record {int(g.report_in_record.sum())}, in window {int(g.report_in_window.sum())} (signed after prediction day {int(g.report_signed_after_prediction.sum())}), removed {int(g.report_removed.sum())} | trimmed {int(g.trim_cut.sum())} | "
                  f"newest record before prediction: median {int(d.median())} d, p90 {int(d.quantile(.9))} d, max {int(d.max())} d | "
                  f"after prediction time {int(g.n_after_prediction.sum())}")
        bad = e[(e.n_after_prediction > 0) | ~e.header_present | (e.report_in_window & ~e.report_removed & (e.arm != "no_notes"))]
        if len(bad):
            fails.append(f"{task}: {len(bad)} EHR rows fail (event after prediction time, no header, or report kept)")
        frames = [e]
        if not args.skip_ct:
            ct = pd.DataFrame([r for arm in sorted(_CT_EXPERIMENTS) for r in _ct_rows(orch, info, arm, df)])
            for arm, g in ct.groupby("arm"):
                gp = g[g.has_prior]
                print(f"  {arm:15s} cases {len(g)} | images as expected {int((g.n_images == g.expected_images).sum())} | load errors {int(g.load_error.notna().sum())} | "
                      f"cases with any near-blank slice {int((g.near_blank_slices > 0).sum())} (median per case {int(g.near_blank_slices.median())}) | "
                      f"with prior {len(gp)}" + (f", prior earlier {int((gp.prior_gap_days > 0).sum())}, gap median {int(gp.prior_gap_days.median())} d" if len(gp) else ""))
            badct = ct[(ct.n_images != ct.expected_images) & ct.load_error.isna()]
            if len(badct) or (ct.has_prior & ~(ct.prior_gap_days > 0)).any():
                fails.append(f"{task}: {len(badct)} CT rows with the wrong image count, or a prior not earlier")
            missing = sum(len(v) for k, v in getattr(orch, "_missing", {}).items() if k[0] == task)
            print(f"  CT inputs: {missing} rows listed missing (no CT or no prior)")
            orch._missing = __import__("collections").defaultdict(list)
            frames.append(ct)
        pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{task}_input_qa.csv", index=False)
    print("\nINPUT QA " + ("PASS" if not fails else "FAIL"))
    for f in fails:
        print("  FAIL: " + f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
