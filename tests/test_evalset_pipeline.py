"""Image packs, the runner, the scorer, and the adapter pieces the eval set adds (offline: synthetic
NIfTIs under a temporary mount, fake transports). Spec: planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md."""

import io
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from context import nifti
from evalset import packs, run, score
from evalset.cards import CARDS
from evalset.recipes import get


@pytest.fixture
def mount(tmp_path, monkeypatch):
    """A fake bucket mount with two RAS CTs: 'a__b' (6 slices; head slice bright) and 'c__d' (prior)."""
    monkeypatch.setenv("EVALSET_MOUNT_ROOT", str(tmp_path))
    monkeypatch.setenv("EVALSET_ROOT", str(tmp_path / "evalset"))
    d = tmp_path / "su-vista-uscentral1" / nifti.DEFAULT_CT_SNAPSHOT_PREFIX
    d.mkdir(parents=True)
    for study, series in (("a", "b"), ("c", "d")):
        vol = np.full((16, 16, 6), -1000, dtype=np.int16)
        vol[:, :, 5] = 1024  # most superior slice (RAS: last index) is bone-bright
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(d / f"{study}__{series}.nii.gz"))
    return tmp_path


def test_pack_renders_head_first_and_checks_hashes(mount):
    r = get("astra")
    m = packs.render_scan(r, 50, "a", "b")
    assert m["display_indices"] == list(range(6)) and m["voxel_indices"] == [5, 4, 3, 2, 1, 0]
    imgs = packs.load(r, 50, "a", "b")
    first = np.asarray(Image.open(io.BytesIO(imgs[0])))
    last = np.asarray(Image.open(io.BytesIO(imgs[-1])))
    assert first[..., 0].mean() > 250 and last[..., 0].mean() < 5  # head slice first
    d = packs.pack_dir(r, 50) / packs.scan_id("a", "b")
    (d / "001.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        packs.load(r, 50, "a", "b")


def test_medgemma_pack_uses_pil_default_jpeg(mount):
    r = get("medgemma")
    packs.render_scan(r, 85, "a", "b")
    data = packs.load(r, 85, "a", "b")[0]
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB")  # decodes
    from context import windows
    stack = nifti.AxialStack.open(nifti.ct_path("a", "b", mount_root=mount))
    Image.fromarray(windows.rgb_u8(stack[0], "medgemma")).save(buf, format="JPEG")
    assert data == buf.getvalue()


class FakeBackend:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def ask_case(self, msgs):
        self.calls += len(msgs)
        return [{"text": self.reply(m), "usage": {"prompt_tokens": 10}} for m in msgs]


def _samples(root, step, rows):
    (root / "samples").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / "samples" / f"step{step}.parquet", index=False)


def test_run_writes_every_question_and_resumes(mount):
    root = mount / "evalset"
    labels = {f"label__{q.task}": 1 for q in CARDS[2].questions}
    _samples(root, 2, [dict(case_id="x1", person_id="1", split="test", study="a", series="b", **labels),
                       dict(case_id="x2", person_id="2", split="test", study="a", series="b",
                            **{**labels, "label__per_ct_pneumonitis": float("nan")})])
    r = get("argon", n_slices=4)
    packs.render_many(r, 4, [("a", "b")], workers=1)
    fb = FakeBackend(lambda m: "yes" if "lobe" in m.question else "no")
    res = run.run(r, 2, "ct", "t", backend=fb)
    assert len(res) == 34 and fb.calls == 34
    assert set(res["answer"]) == {"yes", "no"}
    assert res["label"].isna().sum() == 1  # the unlabelled pneumonitis row is kept, unscored
    one = res[(res.case_id == "x1") & (res.question_id == "lobe_rul")].iloc[0]
    assert one["recipe"] == "argon" and one["task"] == "per_ct_lung_lobe_rul"
    fb2 = FakeBackend(lambda m: "yes")
    res2 = run.run(r, 2, "ct", "t", backend=fb2)
    assert fb2.calls == 0 and len(res2) == 34


def test_run_pair_step_sends_prior_then_current(mount):
    root = mount / "evalset"
    _samples(root, 3, [dict(case_id="p1", person_id="1", split="test", study="a", series="b", prior_study="c",
                            prior_series="d", label__radiology_progression=1)])
    r = get("astra", n_slices_pair=3)
    packs.render_many(r, 3, [("a", "b"), ("c", "d")], workers=1)
    seen = []
    fb = FakeBackend(lambda m: seen.append(m) or '{"answer": "yes"}')
    run.run(r, 3, "ct", "t", backend=fb)
    labels = [p["text"] for p in seen[0].parts if p["type"] == "text"][:-1]
    assert labels == ["PRIOR scan, image 1:", "PRIOR scan, image 2:", "PRIOR scan, image 3:",
                      "CURRENT scan, image 1:", "CURRENT scan, image 2:", "CURRENT scan, image 3:"]


def test_summary_arm_sends_no_images(mount):
    root = mount / "evalset"
    _samples(root, 1, [dict(case_id="1", person_id="1", split="test", study="a", series="b", summary="Some text.",
                            label__died_any_cause_1_yr=0, label__died_any_cause_2_yr=1)])
    seen = []
    fb = FakeBackend(lambda m: seen.append(m) or "Final Answer: no")
    res = run.run(get("medgemma"), 1, "summary", "t", backend=fb)
    assert all(not m.images for m in seen) and len(res) == 2 and set(res["answer"]) == {"no"}


# -- scoring ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,form,want", [
    ("The nodule ... Final Answer: yes", "final_answer", "yes"),
    ("Final Answer: no. On reflection, **Final Answer: Yes**", "final_answer", "yes"),
    ("I think yes.", "final_answer", None),
    ("no", "schema", "no"),
    ('{"answer": "Yes"}', "schema", "yes"),
    ('{"answer": "maybe"}', "schema", None),
])
def test_parse_answer(text, form, want):
    assert score.parse_answer(text, form) == want


def test_refusal_detection():
    assert score.is_refusal("", "empty answer (finish_reason=content_filter)")
    assert score.is_refusal("I'm sorry, but I can't interpret CT scans.", None)
    assert not score.is_refusal("Final Answer: no", None)


def test_unparseable_counts_wrong_in_balanced_accuracy():
    # yes class: 2 right of 4 (one unparseable); no class: 3 of 4 -> (0.5 + 0.75) / 2
    assert score.balanced_accuracy([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, None, 0, 0, 0, 0, 1]) == pytest.approx(0.625)


def test_validation_cut_hand_example():
    # perfect split between 0.3 and 0.6 -> midpoint 0.45 (nearer 0.5 than any other best cut)
    assert score.validation_cut([0.1, 0.3, 0.6, 0.9], [0, 0, 1, 1]) == pytest.approx(0.45)


def test_probe_predictions_cut_on_validation_only():
    pe = pd.DataFrame({"task": "t", "accession_number": list("abcdef"), "split": ["val"] * 4 + ["test"] * 2,
                       "label": [0, 0, 1, 1, 0, 1], "score": [0.1, 0.2, 0.7, 0.8, 0.5, 0.3]})
    t, pred = score.probe_predictions(pe, "t")
    assert t == pytest.approx(0.45)
    assert pred.set_index("key").loc[["e", "f"], "pred"].tolist() == [1, 0]  # test rows cut at the frozen 0.45


def test_score_rows_macro_and_patient_bootstrap():
    rows = []
    for p in range(40):
        for q in ("q1", "q2"):
            lab = p % 2
            pred = lab if q == "q1" else (lab if p < 20 else None)
            rows.append({"recipe": "r", "arm": "ct", "question_id": q, "person_id": str(p), "label": lab,
                         "pred": pred, "refused": pred is None and p == 39})
    t = score.score_rows(pd.DataFrame(rows), n_boot=200)
    q1 = t[t.question_id == "q1"].iloc[0]
    q2 = t[t.question_id == "q2"].iloc[0]
    macro = t[t.question_id == "MACRO"].iloc[0]
    assert q1["balanced_accuracy"] == 1.0 and q2["balanced_accuracy"] == 0.5
    assert macro["balanced_accuracy"] == pytest.approx(0.75)
    assert q2["unparseable_rate"] == 0.5 and q2["refusal_rate"] == pytest.approx(1 / 40)
    assert q1["ba_lo"] == 1.0 and macro["ba_lo"] < 0.75 < macro["ba_hi"]


# -- adapters ----------------------------------------------------------------------------------------

def _parts():
    return [{"type": "text", "text": "Image 1:"}, {"type": "image", "data": b"\xff\xd8x", "mime": "image/jpeg"},
            {"type": "text", "text": "Question?"}]


def test_gemini_parts_request_with_cache_sends_only_the_question():
    from models.vertex_gemini import build_parts_request
    full = build_parts_request(_parts(), ["yes", "no"], 100, "MEDIA_RESOLUTION_HIGH", "low")
    assert [list(p) for p in full["contents"][0]["parts"]] == [["text"], ["inlineData"], ["text"]]
    assert full["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"
    c = build_parts_request(_parts(), ["yes", "no"], 100, "MEDIA_RESOLUTION_HIGH", "low", "caches/1", 2)
    assert c["cachedContent"] == "caches/1" and c["contents"][0]["parts"] == [{"text": "Question?"}]
    assert "mediaResolution" not in c["generationConfig"]
    assert c["generationConfig"]["responseSchema"]["enum"] == ["yes", "no"]


def test_gemini_cost_bills_cached_tokens_at_cached_rate():
    from models.vertex_gemini import VertexGeminiAdapter
    a = VertexGeminiAdapter("m")
    a.prices = run.ARGON_PRICES
    # 1,000,000 prompt tokens of which 900,000 cached, 100,000 output+thinking:
    # 0.1M x $2 + 0.9M x $0.10 + 0.1M x $10 = 0.2 + 0.09 + 1.0
    cost = a.call_cost_usd({"prompt_tokens": 1_000_000, "cached_tokens": 900_000, "output_tokens": 60_000,
                            "thinking_tokens": 40_000})
    assert cost == pytest.approx(1.29)


def test_gemini_ask_group_caches_once_and_reuses_answers(tmp_path):
    from models.vertex_gemini import VertexGeminiAdapter
    a = VertexGeminiAdapter("gemini-4-argon", cache_dir=str(tmp_path))
    a.configure({"explicit_cache": True, "concurrency": 1})
    a.response_cache.mkdir(parents=True)
    calls = []

    def fake_post(method, body, url=None, http="POST"):
        calls.append((http, method, (url or "").rsplit("/", 1)[-1], body))
        if url and url.endswith("cachedContents"):
            return {"name": "projects/p/locations/global/cachedContents/9", "usageMetadata": {"totalTokenCount": 1000}}, None
        if http == "DELETE":
            return {}, None
        q = body["contents"][0]["parts"][-1]["text"]
        return {"candidates": [{"content": {"parts": [{"text": "yes" if "A" in q else "no"}]}}],
                "usageMetadata": {"promptTokenCount": 1005, "cachedContentTokenCount": 1000,
                                  "candidatesTokenCount": 1}}, None
    a._post = fake_post
    msgs = []
    for q in ("A?", "B?"):
        p = _parts()
        p[-1] = {"type": "text", "text": q}
        msgs.append({"parts": p, "prefix_len": 2})
    out = a.ask_group(msgs, 100, ["yes", "no"])
    assert [o["text"] for o in out] == ["yes", "no"]
    kinds = [c[0] + ":" + (c[1] or c[2]) for c in calls]
    assert kinds.count("POST:cachedContents") == 1 and kinds.count("POST:generateContent") == 2
    assert kinds[-1].startswith("DELETE")
    gen = [c[3] for c in calls if c[1] == "generateContent"]
    assert all(b["cachedContent"].endswith("/9") and len(b["contents"][0]["parts"]) == 1 for b in gen)
    assert out[0]["usage"]["cached_tokens"] == 1000
    calls.clear()
    again = a.ask_group(msgs, 100, ["yes", "no"])
    assert calls == [] and all(o["cached"] for o in again)


def test_aihub_parts_request_keeps_label_order():
    from models.aihub_openai import build_parts_request
    b = build_parts_request(_parts(), ["yes", "no"], 50, "low")
    content = b["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert b["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]
    assert b["reasoning_effort"] == "low"


def test_run_question_subset(mount):
    root = mount / "evalset"
    labels = {f"label__{q.task}": 0 for q in CARDS[2].questions}
    _samples(root, 2, [dict(case_id="x1", person_id="1", split="test", study="a", series="b", **labels)])
    r = get("argon", n_slices=4)
    packs.render_many(r, 4, [("a", "b")], workers=1)
    fb = FakeBackend(lambda m: "no")
    res = run.run(r, 2, "ct", "sub", backend=fb, questions=["lobe_rul", "pneumonitis"])
    assert sorted(res["question_id"]) == ["lobe_rul", "pneumonitis"] and fb.calls == 2


# -- MedGemma's vLLM path, without vLLM: request content and the yes/no token probabilities -----------

def test_openai_content_keeps_notebook_order():
    m = get("medgemma").build("Answer yes or no.", [(None, [b"\xff\xd8a", b"\xff\xd8b"])])
    c = run.openai_content(m)
    assert [x["type"] for x in c] == ["image_url", "text", "image_url", "text", "text"]
    assert c[1]["text"] == "SLICE 1" and c[3]["text"] == "SLICE 2"
    assert c[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert c[-1]["text"].endswith("'Final Answer: no'.")


class _LP:
    def __init__(self, tok, lp):
        self.decoded_token, self.logprob = tok, lp


class _Completion:
    def __init__(self, tokens):
        # tokens: [(token_id, text, {alt_id: (text, logprob)})]
        self.token_ids = [t for t, _, _ in tokens]
        self.logprobs = [{tid: _LP(txt, -0.1), **{a: _LP(at, alp) for a, (at, alp) in alts.items()}}
                         for tid, txt, alts in tokens]


def test_answer_logprobs_reads_the_last_yes_no_token():
    c = _Completion([(1, "Yes", {}), (2, " Final", {}), (3, " Answer", {}), (4, ":", {}),
                     (5, " no", {6: (" yes", -2.3)}), (7, "<eos>", {})])
    got = json.loads(run.answer_logprobs(c))
    assert got["position"] == 4 and got["top"] == {" no": -0.1, " yes": -2.3}
    assert run.answer_logprobs(_Completion([(1, "maybe", {})])) is None


# -- code-review fixes (2026-10-28) ------------------------------------------------------------------

def test_score_rows_one_macro_per_step():
    rows = []
    for step, qs, right in ((1, ("a", "b"), True), (2, ("c", "d", "e"), False)):
        for p in range(20):
            for q in qs:
                lab = p % 2
                rows.append({"step": step, "recipe": "r", "arm": "ct", "question_id": q, "person_id": str(p),
                             "label": lab, "pred": lab if right else 1 - lab, "refused": False})
    t = score.score_rows(pd.DataFrame(rows), n_boot=50)
    macro = t[t.question_id == "MACRO"].set_index("step")["balanced_accuracy"]
    assert macro.to_dict() == {1: 1.0, 2: 0.0}


def test_weighted_macro_with_unit_weights_equals_plain_macro():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"question_id": rng.choice(["a", "b", "c"], 300), "person_id": rng.integers(0, 60, 300).astype(str),
                       "label": rng.integers(0, 2, 300)})
    df["pred"] = [None if r < 0.1 else int(r < 0.6) for r in rng.random(300)]
    qcode, uni = pd.factorize(df["question_id"])
    correct = np.array([p is not None and p == t for p, t in zip(df["pred"], df["label"])], dtype=float)
    got = score._weighted_macro_ba(qcode, df["label"].to_numpy(), correct, np.ones(300), len(uni))
    assert got == pytest.approx(score.macro_balanced_accuracy(df))


def test_disclaimer_next_to_a_readable_answer_is_not_a_refusal():
    r = get("medgemma")
    q = CARDS[2].questions[0]
    row = {"case_id": "1", "person_id": "1", "split": "test", f"label__{q.task}": 1}
    ok = run.result_row(r, 2, "ct", row, q, {"text": "I cannot provide a diagnosis; consult a radiologist. Final Answer: yes"})
    no = run.result_row(r, 2, "ct", row, q, {"text": "I'm sorry, I can't interpret CT scans."})
    assert ok["answer"] == "yes" and not ok["refused"]
    assert no["answer"] is None and no["refused"]


def test_check_counts_only_the_cases_and_questions_asked(mount):
    from evalset import check
    root = mount / "evalset"
    labels = {f"label__{q.task}": 1 for q in CARDS[2].questions}
    _samples(root, 2, [dict(case_id=f"x{i}", person_id=str(i), split="test", study="a", series="b", **labels)
                       for i in range(3)])
    r = get("argon", n_slices=4)
    packs.render_many(r, 4, [("a", "b")], workers=1)
    run.run(r, 2, "ct", "c", backend=FakeBackend(lambda m: "yes"))  # all 3 cases x 17 questions
    assert check.check("argon", "c", [2], None, root=root)
    assert check.check("argon", "c", [2], 2, root=root)  # fewer cases than the file holds: still a pass
    assert check.check("argon", "c", [2], 2, questions=["lobe_rul"], root=root)
    assert not check.check("argon", "missing", [2], 2, root=root)


def test_run_with_nothing_to_ask_raises_and_writes_nothing(mount):
    root = mount / "evalset"
    _samples(root, 3, [dict(case_id="p", person_id="1", split="test", study="a", series="b", prior_study="c",
                            prior_series="d", label__radiology_progression=1)])
    r = get("astra", n_slices_pair=2)
    packs.render_many(r, 2, [("a", "b"), ("c", "d")], workers=1)
    with pytest.raises(RuntimeError):
        run.run(r, 3, "ct", "e", backend=FakeBackend(lambda m: "yes"), questions=["not_a_question"])
    assert not run.out_path("e", r, 3, "ct", root).exists()


def test_gemini_cache_respects_spend_cap(tmp_path):
    from models.vertex_gemini import VertexGeminiAdapter
    a = VertexGeminiAdapter("gemini-4-argon", cache_dir=str(tmp_path))
    a.configure({"explicit_cache": True, "max_spend_usd": 0.0})
    a.response_cache.mkdir(parents=True)
    calls = []
    a._post = lambda *x, **k: calls.append(x) or ({}, None)
    msgs = [{"parts": _parts(), "prefix_len": 2}, {"parts": _parts()[:2] + [{"type": "text", "text": "B?"}], "prefix_len": 2}]
    out = a.ask_group(msgs, 100, ["yes", "no"])
    assert calls == [] and all("spend cap" in o["error"] for o in out)


# -- raw results kept for rescoring (2026-10-28) -------------------------------------------------------

def test_every_attempt_is_logged_and_the_run_recorded(mount):
    root = mount / "evalset"
    _samples(root, 3, [dict(case_id="p", person_id="1", split="test", study="a", series="b", prior_study="c",
                            prior_series="d", label__radiology_progression=1)])
    r = get("astra", n_slices_pair=2)
    packs.render_many(r, 2, [("a", "b"), ("c", "d")], workers=1)

    class Raw(FakeBackend):
        def ask_case(self, msgs):
            return [{**o, "raw": {"body": o["text"]}, "request_key": "k"} for o in super().ask_case(msgs)]
    run.run(r, 3, "ct", "raw", backend=Raw(lambda m: "maybe"))   # unreadable -> asked again next time
    res = run.run(r, 3, "ct", "raw", backend=Raw(lambda m: "yes"))
    p = run.out_path("raw", r, 3, "ct", root)
    attempts = [json.loads(x) for x in p.with_suffix(".attempts.jsonl").read_text().splitlines()]
    assert [a["response_text"] for a in attempts] == ["maybe", "yes"]
    assert attempts[0]["raw"] == {"body": "maybe"} and attempts[0]["answer"] is None
    assert attempts[0]["message_key"] == attempts[1]["message_key"] == res.iloc[0]["message_key"]
    assert res.iloc[0]["answer"] == "yes" and res.iloc[0]["request_key"] == "k" and len(res) == 1
    runs = [json.loads(x) for x in p.with_suffix(".runs.jsonl").read_text().splitlines()]
    assert len(runs) == 2 and runs[1]["n_parsed"] == 1 and runs[0]["n_parsed"] == 0
    assert runs[0]["pack"] == r.pack_id(2) and runs[0]["recipe"]["name"] == "astra" and runs[0]["commit"]


def test_p_yes_sums_spellings_and_normalises():
    import math
    lp = json.dumps({"position": 3, "top": {" no": math.log(0.6), " yes": math.log(0.15), "Yes": math.log(0.05),
                                            " maybe": math.log(0.2)}})
    assert run.p_yes(lp) == pytest.approx(0.2 / 0.8)
    assert run.p_yes(None) is None and run.p_yes(json.dumps({"top": {" maybe": -0.1}})) is None


def test_model_in_cache_needs_weights(tmp_path):
    snap = tmp_path / "models--google--medgemma-1.5-4b-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert not run.model_in_cache(str(tmp_path), "google/medgemma-1.5-4b-it")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert run.model_in_cache(str(tmp_path), "google/medgemma-1.5-4b-it")
    assert run.model_in_cache("/mnt/su-vista-uscentral1/vistabench/vlm/model_cache", "google/medgemma-1.5-4b-it") \
        or not __import__("os").path.isdir("/mnt/su-vista-uscentral1")
