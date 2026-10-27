"""Gemini probes of what the model can see (tumor-board plan rev 5, input-QA layer C).

On a fixed sample of progression cases that have a prior CT, using the board config's inputs:
  images  - 60 slices (prior then current): per set, body region and contrast, and which set looks
            earlier. Pass: chest on both sets; the first set read as the earlier one.
  record  - the trimmed EHR: age, sex, latest cancer treatment and the newest record's date.
            Pass: age and sex match the header; newest date within 7 days of the true one.
  order   - the forced progression question on the trimmed EHR, asked with the two options in
            each order. Pass: answers barely change between orders.
Per-case answers go to <results_dir>/input_qa/probes_*.csv (mount only); totals print here. Uses
the Vertex adapter's transport and its hard spend cap (--max-spend, default $20).

    cd src && python ../scripts/vlm_probes.py --config ../configs/all_tasks.tb.gemini.yaml
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.vertex_gemini import _png_b64  # noqa: E402
from vista_run.run_bq import TaskOrchestrator, attach_prior_ct, CT_CHANNELS_LINE  # noqa: E402
from vqa_dataset import PromptDataset  # noqa: E402

TASK = "progression_assessment_answer"
_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}\]")

IMAGE_PROBE = ("You are shown 60 axial CT slices: the first 30 come from one CT scan and the last 30 from another "
               "scan of the same patient. For each set, name the body region covered and whether IV contrast "
               "appears to have been given. Then say which set looks like the EARLIER scan, if you can tell.")
IMAGE_SCHEMA = {"type": "OBJECT", "properties": {
    "first_set_region": {"type": "STRING", "enum": ["chest", "chest and abdomen", "abdomen/pelvis", "head/neck", "other", "unclear"]},
    "second_set_region": {"type": "STRING", "enum": ["chest", "chest and abdomen", "abdomen/pelvis", "head/neck", "other", "unclear"]},
    "first_set_contrast": {"type": "STRING", "enum": ["yes", "no", "unclear"]},
    "second_set_contrast": {"type": "STRING", "enum": ["yes", "no", "unclear"]},
    "earlier_set": {"type": "STRING", "enum": ["first", "second", "cannot tell"]}},
    "required": ["first_set_region", "second_set_region", "first_set_contrast", "second_set_contrast", "earlier_set"]}

RECORD_PROBE = ("Read the patient record below and report: the patient's age in years and sex as stated in the record; "
                "the most recent cancer treatment (drug, radiation or surgery) and its date; and the date of the newest "
                "entry in the record. Use YYYY-MM-DD dates. Say 'not stated' when the record doesn't say.\n\nPATIENT RECORD:\n\n")
RECORD_SCHEMA = {"type": "OBJECT", "properties": {
    "age_years": {"type": "STRING"}, "sex": {"type": "STRING"},
    "latest_cancer_treatment": {"type": "STRING"}, "latest_cancer_treatment_date": {"type": "STRING"},
    "newest_entry_date": {"type": "STRING"}},
    "required": ["age_years", "sex", "latest_cancer_treatment", "latest_cancer_treatment_date", "newest_entry_date"]}


def _call(adapter, parts, schema=None, choices=None):
    gen = {"temperature": 0.0, "maxOutputTokens": 8192}
    if schema:
        gen.update(responseMimeType="application/json", responseSchema=schema)
    if choices:
        gen.update(responseMimeType="text/x.enum", responseSchema={"type": "STRING", "enum": choices})
    if adapter.max_spend_usd is not None and adapter.spent_usd >= adapter.max_spend_usd:
        return None, "spend cap reached"
    body, err = adapter._post("generateContent", {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen})
    if err:
        return None, err
    um = body.get("usageMetadata", {})
    with adapter._lock:
        adapter.spent_usd += adapter.call_cost_usd({"prompt_tokens": um.get("promptTokenCount"),
                                                    "output_tokens": um.get("candidatesTokenCount"),
                                                    "thinking_tokens": um.get("thoughtsTokenCount")})
    try:
        text = "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"] if not p.get("thought")).strip()
    except (KeyError, IndexError):
        return None, "no answer"
    return text, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--n-record", type=int, default=30)
    ap.add_argument("--n-order", type=int, default=40)
    ap.add_argument("--max-spend", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the images CSV, e.g. _rgb")
    args = ap.parse_args()
    orch = TaskOrchestrator(args.config, "vertex_gemini", "gemini-3.1-pro-preview")
    orch.cfg.pop("row_sample", None)
    a = orch.adapter
    a.load()
    a.max_spend_usd = args.max_spend
    info = next(t for t in orch.valid_tasks if t["task_name"] == TASK)
    df, col, _ = orch._load_bq_table_task_data(info)
    df = df[df.image_series_uid.notna()]
    with_prior, _ = attach_prior_ct(df, orch._prior_link(info["prior_ct"]), info["prior_ct"])
    pool = with_prior.sample(n=max(args.n_images, args.n_record, args.n_order), random_state=args.seed)
    out_dir = Path(orch.results_base) / "input_qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    # images
    d = orch._build_prompts_for_experiment(pool.head(args.n_images).drop(columns=[c for c in pool.columns if c.startswith("prior_")]),
                                           info, "axial_all_image", col)
    ds = PromptDataset(df=d, prompt_col="dynamic_prompt", experiment="axial_all_image", storage_client=orch.storage_client,
                       model_type="vertex_gemini", ct_snapshot_prefix=orch.cfg["paths"].get("ct_snapshot_prefix"), fail_on_ct_error=True, ct_windowing=orch.cfg.get("ct_windowing"))
    items = [ds[i] for i in range(len(ds))]

    def img_job(it):
        if it.get("load_error") or not it["image"]:
            return {"person_id": it["raw_row"]["person_id"], "error": it.get("load_error") or "no images"}
        probe = (CT_CHANNELS_LINE + "\n\n" + IMAGE_PROBE) if orch.cfg.get("ct_windowing") == "multi_window_rgb" else IMAGE_PROBE
        parts = [{"inlineData": {"mimeType": "image/png", "data": _png_b64(im)}} for im in it["image"]] + [{"text": probe}]
        txt, err = _call(a, parts, schema=IMAGE_SCHEMA)
        row = {"person_id": it["raw_row"]["person_id"], "n_images": len(it["image"]), "error": err}
        if txt:
            row.update(json.loads(txt))
        return row
    with ThreadPoolExecutor(8) as ex:
        img = pd.DataFrame(list(ex.map(img_job, items)))
    img.to_csv(out_dir / f"probes_images{args.tag}.csv", index=False)
    if args.images_only:
        ok = img[img.error.isna()] if "error" in img else img
        print(f"IMAGES  {len(ok)}/{len(img)} answered | first set region {ok.first_set_region.value_counts().to_dict()} | "
              f"second {ok.second_set_region.value_counts().to_dict()} | earlier set {ok.earlier_set.value_counts().to_dict()} | "
              f"contrast first {ok.first_set_contrast.value_counts().to_dict()} second {ok.second_set_contrast.value_counts().to_dict()}")
        print(f"SPEND   ${a.spent_usd:.2f} (cap ${args.max_spend:.0f}), {a.api_calls} API calls")
        return

    # record (trimmed EHR, with the header)
    rec_in = orch._build_prompts_for_experiment(pool.head(args.n_record), info, "no_image", col)

    def rec_job(r):
        record = r["dynamic_prompt"].split("PATIENT RECORD:\n\n", 1)[-1]
        header = record.split("\n", 1)[0]
        dates = _TS.findall(record)
        txt, err = _call(a, [{"text": RECORD_PROBE + record}], schema=RECORD_SCHEMA)
        row = {"person_id": r["person_id"], "header": header, "true_newest": max(dates) if dates else None, "error": err}
        if txt:
            row.update(json.loads(txt))
        return row
    with ThreadPoolExecutor(8) as ex:
        rec = pd.DataFrame(list(ex.map(rec_job, [r for _, r in rec_in.iterrows()])))
    rec.to_csv(out_dir / "probes_record.csv", index=False)

    # answer order
    ord_in = orch._build_prompts_for_experiment(pool.head(args.n_order), info, "no_image", col)
    fwd, rev = ["Progression", "Response or Stable"], ["Response or Stable", "Progression"]

    def ord_job(r):
        p_fwd = r["dynamic_prompt"]
        p_rev = p_fwd.replace('OPTIONS: "Progression", "Response or Stable"', 'OPTIONS: "Response or Stable", "Progression"')
        a1, e1 = _call(a, [{"text": p_fwd}], choices=fwd)
        a2, e2 = _call(a, [{"text": p_rev}], choices=rev)
        return {"person_id": r["person_id"], "label": r["label"], "answer_fwd": a1, "answer_rev": a2, "error": e1 or e2}
    with ThreadPoolExecutor(8) as ex:
        order = pd.DataFrame(list(ex.map(ord_job, [r for _, r in ord_in.iterrows()])))
    order.to_csv(out_dir / "probes_order.csv", index=False)

    # totals
    ok = img[img.error.isna()] if "error" in img else img
    print(f"IMAGES  {len(ok)}/{len(img)} answered | first set chest-ish {int(ok.first_set_region.str.contains('chest').sum())}, "
          f"second set chest-ish {int(ok.second_set_region.str.contains('chest').sum())} | earlier set: "
          f"{ok.earlier_set.value_counts().to_dict()} | contrast first/second yes: {int((ok.first_set_contrast=='yes').sum())}/{int((ok.second_set_contrast=='yes').sum())}")
    r = rec[rec.error.isna()].copy()
    r["hdr_age"] = r.header.str.extract(r"age (\d+)")[0]
    r["hdr_sex"] = r.header.str.extract(r"sex (\w+)")[0]
    age_ok = (r.age_years.str.extract(r"(\d+)")[0] == r.hdr_age).sum()
    sex_ok = sum(str(a).lower()[:1] == str(b).lower()[:1] for a, b in zip(r.sex, r.hdr_sex))
    nd = (pd.to_datetime(r.newest_entry_date, errors="coerce") - pd.to_datetime(r.true_newest)).dt.days.abs()
    print(f"RECORD  {len(r)}/{len(rec)} answered | age matches header {int(age_ok)} | sex matches {int(sex_ok)} | "
          f"newest date within 7 d {int((nd <= 7).sum())} (median off {nd.median()} d) | treatment named {int((~r.latest_cancer_treatment.str.lower().str.contains('not stated')).sum())}")
    o = order[order.error.isna()]
    print(f"ORDER   {len(o)}/{len(order)} answered | same answer both orders {int((o.answer_fwd == o.answer_rev).sum())} | "
          f"'Progression' when listed first {int((o.answer_fwd=='Progression').sum())}, when listed second {int((o.answer_rev=='Progression').sum())}")
    print(f"SPEND   ${a.spent_usd:.2f} (cap ${args.max_spend:.0f}), {a.api_calls} API calls")


if __name__ == "__main__":
    main()
