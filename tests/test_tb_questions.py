"""Tumor-board build: Gemini request/parse/cache, forced-answer metrics, overflow accounting,
the new EHR arms. No network, no weights, no patient data (all inputs synthetic)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from models.vertex_gemini import VertexGeminiAdapter, build_request, parse_response
from results.final_metrics import answer_to_label, forced_answer_metrics, gemini_cost_usd, bootstrap_ci
from context.presets import get_preset
from context.adapters.ehr import EHRAdapter
from context.selectors.ehr_filters import window
from vista_run.run_bq import TaskOrchestrator, forced_choices_for

PNEUMO_MAP = {"0": "Pneumonitis", "1": "Infection", "2": "Both pneumonitis and infection",
              "3": "Cancer progression", "4": "Neither", "-1": "Indeterminate/Deferred"}


# --- Gemini request / response -----------------------------------------------------------
def test_build_request_forces_options_and_puts_images_first():
    imgs = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(3)]
    body = build_request("Q?", imgs, ["Yes", "No"], max_output_tokens=50)
    parts = body["contents"][0]["parts"]
    assert [("inlineData" in p) for p in parts] == [True, True, True, False]
    assert parts[-1]["text"] == "Q?"
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    gen = body["generationConfig"]
    assert gen["temperature"] == 0.0 and gen["maxOutputTokens"] == 50
    assert gen["responseMimeType"] == "text/x.enum"
    assert gen["responseSchema"] == {"type": "STRING", "enum": ["Yes", "No"]}
    assert "thinkingConfig" not in gen and "mediaResolution" not in gen


def _resp(text, thought=None, finish="STOP"):
    parts = ([{"text": thought, "thought": True}] if thought else []) + ([{"text": text}] if text is not None else [])
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 1,
                              "thoughtsTokenCount": 300, "totalTokenCount": 1301}}


def test_parse_response_reads_answer_and_usage_and_skips_thoughts():
    text, usage, err = parse_response(_resp("No", thought="thinking..."), ["Yes", "No"])
    assert (text, err) == ("No", None)
    assert usage == {"prompt_tokens": 1000, "output_tokens": 1, "thinking_tokens": 300, "total_tokens": 1301}


@pytest.mark.parametrize("body", [_resp("Maybe"), _resp(None, finish="MAX_TOKENS"), {"candidates": []}])
def test_parse_response_unreadable_is_an_error_not_a_guess(body):
    text, _, err = parse_response(body, ["Yes", "No"])
    assert text is None and err


def test_cache_makes_rerun_free_and_errors_are_not_cached(tmp_path):
    a = VertexGeminiAdapter("gemini-test", cache_dir=str(tmp_path))
    calls = []

    def fake_post(method, body):
        calls.append(body)
        q = body["contents"][0]["parts"][-1]["text"]
        if q == "bad":
            return None, "HTTP 400: too long"
        return _resp("Yes"), None

    a._post = fake_post
    msgs = [{"question": f"q{i}", "images": []} for i in range(4)] + [{"question": "bad", "images": []}]
    out1 = a.infer(a, None, msgs, 100, ["Yes", "No"])
    assert len(calls) == 5
    assert [o.get("cached") for o in out1[:4]] == [False] * 4 and out1[4]["error"].startswith("HTTP 400")
    out2 = a.infer(a, None, msgs, 100, ["Yes", "No"])
    assert len(calls) == 6  # only the failed row is asked again
    assert all(o["cached"] and o["text"] == "Yes" for o in out2[:4])
    assert out2[0]["usage"] == {"prompt_tokens": 1000, "output_tokens": 1, "thinking_tokens": 300,
                                "total_tokens": 1301, "image_count": 0}


# --- forced answers and metrics (hand-worked) ----------------------------------------------
def test_answer_to_label():
    assert answer_to_label("Infection", PNEUMO_MAP) == 1
    assert answer_to_label(" neither ", PNEUMO_MAP) == 4
    assert answer_to_label("Indeterminate/Deferred", PNEUMO_MAP) is None
    assert answer_to_label("Yes", PNEUMO_MAP) is None
    assert answer_to_label(float("nan"), PNEUMO_MAP) is None


def test_binary_metrics_hand_worked():
    # truth 1,1,1,0,0 ; predicted 1,0,unreadable,0,1
    m = forced_answer_metrics([1, 1, 1, 0, 0], [1, 0, None, 0, 1], [0, 1])
    assert m["n"] == 5 and m["n_unreadable"] == 1
    assert m["accuracy"] == pytest.approx(2 / 5)
    assert m["sensitivity"] == pytest.approx(1 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["balanced_accuracy"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_multiclass_metrics_hand_worked():
    # classes present in truth: 0, 1, 2. recalls 1/2, 1, 0. F1: 2/3, 2/3, 0.
    m = forced_answer_metrics([0, 0, 1, 2], [0, 1, 1, None], [0, 1, 2, 3, 4])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert "sensitivity" not in m


def test_constant_predictor_is_exactly_chance():
    m = forced_answer_metrics([1] * 30 + [0] * 20, [1] * 50, [0, 1])
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    lo, hi = bootstrap_ci([1] * 30 + [0] * 20, [1] * 50, [0, 1], n_boot=200)["balanced_accuracy"]
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_gemini_cost_long_context_bills_whole_request():
    prices = {"long_context_over": 200000, "input": 2.0, "output": 12.0, "input_long": 4.0, "output_long": 18.0}
    df = pd.DataFrame({"usage_prompt_tokens": [100_000, 300_000],
                       "usage_output_tokens": [1, 1], "usage_thinking_tokens": [999, 999]})
    # 0.1M*2 + 1000*12/1e6 = 0.212 ; 0.3M*4 + 1000*18/1e6 = 1.218
    assert gemini_cost_usd(df, prices) == pytest.approx(0.212 + 1.218)


def test_forced_choices():
    assert forced_choices_for({"choices": ["A", "B", "C"]}) == ["A", "B", "C"]
    assert forced_choices_for({"mapping": {"1": "Yes", "0": "No", "-1": "x"}}) == ["Yes", "No"]
    assert forced_choices_for({"mapping": PNEUMO_MAP}) is None
    assert forced_choices_for({"choices": ["A"]}, use_constrained=False) is None


# --- overflow accounting ----------------------------------------------------------------
class _StubAdapter:
    input_token_limit = 10
    image_token_estimate = 3

    def count_text_tokens(self, text):
        return len(text.split())


def _orch():
    o = TaskOrchestrator.__new__(TaskOrchestrator)  # no BQ/GCS clients, no weights
    o.adapter = _StubAdapter()
    o.cfg = {"runtime": {"max_new_tokens": 5}}
    o.model = o.processor = None
    return o


def test_overflow_rows_are_listed_with_their_count_not_dropped(tmp_path):
    df = pd.DataFrame({"index": [1, 2, 3], "person_id": [1, 2, 3], "split": ["train", "test", "valid"],
                       "label": [0, 1, 1], "dynamic_prompt": ["a b", "a " * 11, "a " * 10]})
    out_file = tmp_path / "t_results_no_image.csv"
    kept = _orch()._split_overflow(df, {"task_name": "t"}, "no_image", out_file)
    assert kept["index"].tolist() == [1, 3]
    assert kept["prompt_token_count"].tolist() == [2, 10]
    ov = pd.read_csv(tmp_path / "t_overflow_no_image.csv")
    assert ov[["index", "prompt_token_count", "limit"]].values.tolist() == [[2, 11, 10]]


def test_overflow_counts_ct_images(tmp_path):
    df = pd.DataFrame({"index": [1], "person_id": [1], "dynamic_prompt": ["a"]})
    o = _orch()
    o.adapter.input_token_limit = 10_000
    kept = o._split_overflow(df, {"task_name": "t"}, "axial_all_image", tmp_path / "t_results_axial_all_image.csv")
    assert kept["prompt_token_count"].tolist() == [1 + 30 * 3]


# --- the new EHR arms -------------------------------------------------------------------
def _events():
    return pd.DataFrame({
        "time": pd.to_datetime(["2020-01-01 09:00", "2020-02-01 10:00", "2020-03-01 00:00", "2020-03-01 08:00"]),
        "code": ["LOINC/1", "STANFORD_NOTE/progress", "RxNorm/2", "LOINC/3"],
        "description": ["lab", "note", "drug", "same-day lab"],
        "numeric_value": [1.0, None, None, 2.0], "unit": [None] * 4,
        "text_value": [None, "long clinical note", None, None],
        "type": [None] * 4, "note_id": [None] * 4, "provider_id": [None] * 4, "speciality": [None] * 4,
    })


def test_no_notes_drops_only_notes_and_keeps_the_window():
    block = next(b for b in get_preset("no_notes")["blocks"] if b["id"] == "ehr")
    assert [f["fn"] for f in block["config"]["select"]] == ["window", "code_filter"]
    ctx = {"embed_time": pd.Timestamp("2020-03-01")}
    text = EHRAdapter(config=block["config"]).contextualize(_events(), ctx).payload
    assert "STANFORD_NOTE" not in text and "long clinical note" not in text
    assert "LOINC/1" in text and "RxNorm/2" in text
    with_notes = EHRAdapter(config=get_preset("no_image")["blocks"][0]["config"]).contextualize(_events(), ctx).payload
    assert "long clinical note" in with_notes


def test_window_ends_at_the_prediction_time_midnight():
    # embed_time is a DATE; an event later that same day must not be shown.
    kept = window(_events(), {"embed_time": pd.Timestamp("2020-03-01")}, before="24mo", after="0d")
    assert kept["code"].tolist() == ["LOINC/1", "STANFORD_NOTE/progress", "RxNorm/2"]


def test_uncut_arm_ignores_the_run_truncation():
    block = next(b for b in get_preset("no_image_uncut")["blocks"] if b["id"] == "ehr")
    long_events = pd.concat([_events()] * 50, ignore_index=True)
    long_events["time"] = pd.date_range("2019-06-01", periods=len(long_events), freq="D")
    ctx = {"embed_time": pd.Timestamp("2020-12-01"), "timeline_truncation": {"mode": "last_k_events", "k": 5}}
    uncut = EHRAdapter(config=block["config"]).contextualize(long_events, ctx)
    cut = EHRAdapter(config=get_preset("no_image")["blocks"][0]["config"]).contextualize(long_events, ctx)
    assert uncut.metadata["truncation"] == "off"
    assert uncut.payload.count("\n") + 1 == 200
    assert cut.payload.count("[20") < 10


def test_spend_cap_stops_new_calls_but_not_cached_ones(tmp_path):
    a = VertexGeminiAdapter("gemini-test", cache_dir=str(tmp_path))
    a.configure({"price_per_mtok": {"long_context_over": 200000, "input": 2.0, "output": 12.0,
                                    "input_long": 4.0, "output_long": 18.0},
                 "max_spend_usd": 0.3, "concurrency": 1})
    a._post = lambda method, body: (_resp("Yes"), None)
    a.max_spend_usd = 0.009  # first call: 0 + 1 x $0.25 default > 0.009 -> refused
    first = a.infer(a, None, [{"question": "z", "images": []}], 100, ["Yes", "No"])
    assert first[0]["error"].startswith("spend cap")
    a.max_spend_usd = 0.3
    # first call reserves the $0.25 default (< 0.3) and costs 1000*2/1e6 + 301*12/1e6 = 0.005612;
    # the second then reserves 0.005612 more and fits, so set the cap so only one fits:
    out = a.infer(a, None, [{"question": "q0", "images": []}, {"question": "q1", "images": []}], 100, ["Yes", "No"])
    assert out[0]["text"] == "Yes"
    a.max_spend_usd = 0.01  # spent 0.005612 + 1 in-flight x 0.005612 mean = 0.0112 > 0.01 -> refused
    assert out[1]["text"] == "Yes"  # (made under the 0.3 cap)
    assert a.spent_usd == pytest.approx(2 * 0.005612)
    blocked = a.infer(a, None, [{"question": "q2", "images": []}], 100, ["Yes", "No"])
    assert blocked[0]["error"].startswith("spend cap")
    again = a.infer(a, None, [{"question": "q0", "images": []}], 100, ["Yes", "No"])
    assert again[0]["cached"] and again[0]["text"] == "Yes"


# --- prior CT -----------------------------------------------------------------------------
from vista_run.run_bq import attach_prior_ct  # noqa: E402
from vqa_dataset import PromptDataset  # noqa: E402

PRIOR_SPEC = {"key": "current_accession_number", "row_key": "ct_accession_number",
              "study": "prior_image_study_uid", "series": "prior_image_series_uid",
              "gap_days": "prior_gap_days", "source": "prior_source"}


def _link(rows):
    return pd.DataFrame(rows, columns=["current_accession_number", "prior_image_study_uid",
                                       "prior_image_series_uid", "prior_gap_days", "prior_source"])


def test_attach_prior_joins_by_accession_and_lists_rows_without_one():
    df = pd.DataFrame({"index": [1, 2, 3], "ct_accession_number": ["A1", "A2", "A3"]})
    link = _link([("A1", "s1", "r1", 80, "report_named"), ("A3", "s3", "r3", 400, "previous_no_usable_date"),
                  ("A9", "s9", "r9", 5, "report_named")])
    has, none = attach_prior_ct(df, link, PRIOR_SPEC)
    assert has["index"].tolist() == [1, 3] and none["index"].tolist() == [2]
    assert has["prior_image_series_uid"].tolist() == ["r1", "r3"]
    assert has["prior_gap_days"].tolist() == [80, 400]


def test_attach_prior_refuses_two_priors_for_one_scan():
    df = pd.DataFrame({"index": [1], "ct_accession_number": ["A1"]})
    with pytest.raises(ValueError):
        attach_prior_ct(df, _link([("A1", "s1", "r1", 80, "x"), ("A1", "s2", "r2", 90, "x")]), PRIOR_SPEC)


def test_prior_images_line_sits_before_the_record():
    df = pd.DataFrame({"dynamic_prompt": ["Q\n\nPATIENT RECORD:\n\nstuff"], "prior_gap_days": [78.0]})
    out = TaskOrchestrator._add_prior_images_line(df)["dynamic_prompt"].iloc[0]
    assert "the first 30 images are axial slices of the PRIOR CT (78 days before the current CT)" in out
    assert out.index("PRIOR CT") < out.index("PATIENT RECORD:") and out.count("PATIENT RECORD:") == 1


def test_overflow_counts_sixty_slices_for_a_two_scan_row(tmp_path):
    df = pd.DataFrame({"index": [1, 2], "person_id": [1, 2], "dynamic_prompt": ["a", "a"],
                       "prior_image_study_uid": ["s", None]})
    o = _orch()
    o.adapter.input_token_limit = 10_000
    kept = o._split_overflow(df, {"task_name": "t"}, "axial_all_image", tmp_path / "t_results_axial_all_image.csv")
    assert kept["prompt_token_count"].tolist() == [1 + 60 * 3, 1 + 30 * 3]


def _ds(rows, fail=True):
    ds = PromptDataset(pd.DataFrame(rows), experiment="axial_all_image", model_type="vertex_gemini",
                       fail_on_ct_error=fail)
    ds._load_ct = lambda study, series, row, idx: (
        (None, None) if study == "BAD" else ([f"{study}-{i}" for i in range(30)], list(range(30))))
    return ds


def test_two_scan_item_has_prior_slices_first():
    item = _ds([{"dynamic_prompt": "q", "image_study_uid": "cur", "image_series_uid": "x",
                 "prior_image_study_uid": "pri", "prior_image_series_uid": "y"}])[0]
    assert len(item["image"]) == 60 and item["image"][0] == "pri-0" and item["image"][30] == "cur-0"
    assert item["selected_indices"] == list(range(30)) * 2
    assert item["load_error"] is None


def test_unreadable_prior_or_current_ct_is_an_error_not_a_one_scan_answer():
    bad_prior = _ds([{"dynamic_prompt": "q", "image_study_uid": "cur", "image_series_uid": "x",
                      "prior_image_study_uid": "BAD", "prior_image_series_uid": "y"}])[0]
    assert bad_prior["load_error"] == "prior CT could not be loaded"
    bad_cur = _ds([{"dynamic_prompt": "q", "image_study_uid": "BAD", "image_series_uid": "x"}])[0]
    assert bad_cur["load_error"] == "CT could not be loaded"
    legacy = _ds([{"dynamic_prompt": "q", "image_study_uid": "BAD", "image_series_uid": "x"}], fail=False)[0]
    assert legacy["load_error"] is None and legacy["image"] is None  # Ryan's tasks: text-only, as before


def test_attach_prior_when_link_key_has_the_task_column_name():
    # S4's link keys on `ct_accession_number`, the same name as the board table's column.
    df = pd.DataFrame({"index": [1, 2], "ct_accession_number": ["A1", "A2"], "image_series_uid": ["c1", "c2"]})
    link = pd.DataFrame({"ct_accession_number": ["A1"], "image_series_uid": ["c1"],
                         "prior_image_study_uid": ["s1"], "prior_image_series_uid": ["r1"],
                         "prior_gap_days": [80], "prior_source": ["report_named"]})
    spec = {**PRIOR_SPEC, "key": "ct_accession_number"}
    has, none = attach_prior_ct(df, link, spec)
    assert has.columns.tolist().count("ct_accession_number") == 1
    assert has["ct_accession_number"].tolist() == ["A1"] and none["ct_accession_number"].tolist() == ["A2"]
    assert has["image_series_uid"].tolist() == ["c1"] and "image_series_uid_x" not in has.columns


# --- review round 1 (finding 9): behaviours that had no test --------------------------------
import vista_run.run_bq as run_bq_mod  # noqa: E402
from results.final_metrics import score_tb  # noqa: E402


def test_unreadable_answer_is_not_cached(tmp_path):
    a = VertexGeminiAdapter("gemini-test", cache_dir=str(tmp_path))
    calls = []
    a._post = lambda method, body: (calls.append(1), (_resp("Maybe"), None))[1]
    a.infer(a, None, [{"question": "q", "images": []}], 100, ["Yes", "No"])
    out = a.infer(a, None, [{"question": "q", "images": []}], 100, ["Yes", "No"])
    assert len(calls) == 2 and out[0]["error"] and not out[0]["cached"]


def _board_rows():
    return pd.DataFrame({"person_id": [30, 10, 20, 40], "label": [1, -1, 0, 1], "split": ["train"] * 4,
                         "image_study_uid": ["s", "s", None, "s"], "image_series_uid": ["x", "x", None, "x"],
                         "compact_summary": ["sum", "sum", "sum", None]})


def _board_orch(monkeypatch, rows, cfg=None):
    o = _orch()
    o.project_id, o.bq_client = "p", None
    o.cfg = {"runtime": {"max_new_tokens": 5}, **(cfg or {})}
    monkeypatch.setattr(run_bq_mod, "fetch_task_data_from_bq", lambda client, table, task: rows.copy())
    return o


TB_TASK = {"task_name": "t", "task_source_csv": "tb", "loader": "bq_table", "unscored_labels": [-1],
           "mapping": {"1": "Progression", "0": "Response or Stable", "-1": "Indeterminate/Deferred"}}


def test_board_loader_drops_unscored_labels_and_keys_on_patient(monkeypatch):
    df, col, _ = _board_orch(monkeypatch, _board_rows())._load_bq_table_task_data(TB_TASK)
    assert df["person_id"].tolist() == [20, 30, 40] and df["index"].tolist() == [20, 30, 40]
    assert col == "patient_string"


def test_board_loader_refuses_two_rows_per_patient(monkeypatch):
    rows = pd.concat([_board_rows(), _board_rows().iloc[:1]])
    with pytest.raises(ValueError):
        _board_orch(monkeypatch, rows)._load_bq_table_task_data(TB_TASK)


def test_board_loader_sample_keeps_only_eligible_rows(monkeypatch):
    cfg = {"row_sample": {"n": 5, "seed": 0, "require_ct": True, "require_summary": True}}
    df, _, _ = _board_orch(monkeypatch, _board_rows(), cfg)._load_bq_table_task_data(TB_TASK)
    assert df["person_id"].tolist() == [30]  # 20 has no CT, 40 no summary, 10 is unscored


def test_summary_arm_lists_rows_without_a_summary(monkeypatch):
    o = _board_orch(monkeypatch, _board_rows())
    o.prompts_map = {"t": "Q\\n\\nPATIENT RECORD:\\n\\n[PATIENT_TIMELINE]"}
    df, col, _ = o._load_bq_table_task_data(TB_TASK)
    out = o._build_prompts_for_experiment(df, TB_TASK, "summary", col)
    assert out["person_id"].tolist() == [20, 30]
    assert out["dynamic_prompt"].tolist()[0].endswith("sum")
    assert [(m["person_id"], m["reason"]) for m in o._missing[("t", "summary")]] == [(40, "no compact_summary")]


def test_result_row_keeps_prompt_and_notes_out():
    raw = pd.Series({"index": 1, "person_id": 1, "label": 0, "dynamic_prompt": "PROMPT", "patient_string": "EHR",
                     "compact_summary": "S", "ct_note_text": "R", "progress_note_text": "N"})
    row = _orch()._build_result_row({"raw_row": raw, "image": None}, {"text": "No", "usage": {"prompt_tokens": 3}},
                                    "no_image", "patient_string", extra_drop=("dynamic_prompt",))
    assert set(row) >= {"index", "person_id", "label", "model_response", "usage_prompt_tokens"}
    assert not {"dynamic_prompt", "patient_string", "compact_summary", "ct_note_text", "progress_note_text"} & set(row)


def test_score_tb_end_to_end(tmp_path):
    import yaml
    base, res = tmp_path / "base", tmp_path / "res"
    (base / "tasks").mkdir(parents=True)
    json.dump([{**TB_TASK, "task_source_csv": "tb"}], open(base / "tasks" / "reg.json", "w"))
    yaml.safe_dump({"paths": {"base_dir": str(base), "results_dir": str(res), "valid_tasks": "tasks/reg.json"},
                    "tasks": ["t"], "models": [{"type": "x", "name": "m"}]}, open(tmp_path / "c.yaml", "w"))
    d = res / "tb" / "t" / "m"
    d.mkdir(parents=True)
    lab = [1, 1, 1, 0, 0, 0]
    split = ["train", "train", "test", "test", "valid", "valid"]
    pd.DataFrame({"index": range(6), "label": lab, "split": split,
                  "model_response": ["Progression"] * 3 + ["Response or Stable"] * 3}).to_csv(d / "t_results_no_image.csv", index=False)
    pd.DataFrame({"index": range(4), "label": lab[:4], "split": split[:4],
                  "model_response": ["Progression", "x", "Progression", "Progression"]}).to_csv(d / "t_results_summary.csv", index=False)
    pd.DataFrame({"index": [4, 5], "reason": ["no compact_summary"] * 2}).to_csv(d / "t_missing_summary.csv", index=False)
    out = score_tb(tmp_path / "c.yaml", tmp_path / "scores.csv", n_boot=50)
    p = out[(out.arm == "no_image") & (out.split == "pooled")].iloc[0]
    assert p.n == 6 and p.balanced_accuracy == pytest.approx(1.0)
    s = out[(out.arm == "summary") & (out.split == "pooled")].iloc[0]
    # summary: truth 1,1,1,0 ; pred 1,unreadable,1,1 -> recall(1)=2/3, recall(0)=0
    assert s.n == 4 and s.n_unreadable == 1 and s.n_missing == 2
    assert s.balanced_accuracy == pytest.approx((2 / 3 + 0) / 2)
    b = out[out.model.str.startswith("baseline") & (out.split == "pooled")].iloc[0]
    assert b.n == 6 and b.balanced_accuracy == pytest.approx(0.5)  # 3 vs 3: tie -> one class, still 0.5
    cmp = pd.read_csv(tmp_path / "scores_matched_comparisons.csv")
    c = cmp[(cmp.arm == "summary") & (cmp.versus == "no_image")].iloc[0]
    # on the 4 shared rows, no_image is perfect (1.0) and summary is 1/3 -> diff -2/3
    assert c.n == 4 and c.balanced_accuracy_diff == pytest.approx(1 / 3 - 1.0)
    assert "above_chance" not in out.columns


def test_constant_guess_on_a_tiny_unbalanced_split_has_interval_exactly_chance():
    # The review's symptom: always-Yes on 15 yes / 2 no showed [0.50, 1.00] before stratifying.
    lo, hi = bootstrap_ci([1] * 15 + [0] * 2, [1] * 17, [0, 1], n_boot=300)["balanced_accuracy"]
    assert (lo, hi) == (pytest.approx(0.5), pytest.approx(0.5))


# --- review round 2 ---------------------------------------------------------------------
def test_arm_splits_limits_only_the_named_arm():
    o = _orch()
    o.cfg["arm_splits"] = {"no_image_uncut": ["test"]}
    df = pd.DataFrame({"index": [1, 2, 3], "split": ["train", "test", "valid"]})
    assert o._apply_arm_splits(df, "no_image_uncut")["index"].tolist() == [2]
    assert o._apply_arm_splits(df, "no_image")["index"].tolist() == [1, 2, 3]


class _ErrAdapter(_StubAdapter):
    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        return [{"text": "", "error": "HTTP 400: bad"} if m == "fail" else {"text": "Yes"} for m in inputs]

    def create_template(self, item):
        return item["question"]

    def prepare_inputs(self, messages, processor, model):
        return list(messages)


def test_failed_calls_go_to_the_errors_file_not_the_results(tmp_path):
    o = _orch()
    o.adapter, o.model_type = _ErrAdapter(), "vertex_gemini"
    o.cfg["runtime"]["max_new_tokens"] = 5
    items = [{"question": q, "image": None,
              "raw_row": pd.Series({"index": i, "person_id": i, "split": "train", "label": 1, "dynamic_prompt": q})}
             for i, q in enumerate(["ok", "fail", "ok"])]
    out_file = tmp_path / "t_results_no_image.csv"
    o._run_inference_loop([items], set(), ["Yes", "No"], out_file, "t", "no_image", None, ("dynamic_prompt",))
    res, err = pd.read_csv(out_file), pd.read_csv(tmp_path / "t_errors_no_image.csv")
    assert res["index"].tolist() == [0, 2] and "dynamic_prompt" not in res.columns
    assert err["index"].tolist() == [1] and err["error"].iloc[0] == "HTTP 400: bad"
