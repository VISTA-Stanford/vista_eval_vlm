"""The probability answer form (recipe ``answer_form: "probability"``): card line, forced schema in the three API
adapters, parsing, the runner's resume check, and AUROC scoring. Offline. Hand-off:
planning/vista_eval_vlm/session/2026-11-17-vlm-probability-answer-build-handoff.md."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalset import arms, run, score
from evalset.cards import CARDS
from evalset.cards.base import ANSWER, ANSWER_PROB
from evalset.recipes import get
from models import aihub_openai, vertex_claude, vertex_gemini
from models.base import PROBABILITY, read_probability

ARMS = Path(__file__).resolve().parents[1] / "configs" / "evalset" / "probability-2026-11-17.yaml"


# -- parsing --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ('{"probability": 42}', 0.42),
    ('{"probability": 0}', 0.0),
    ('{"probability": 100}', 1.0),
    ("42", 0.42),
    ("42%", 0.42),
    ("  7 % ", 0.07),
    ("Probability: 30", 0.30),
    ("I estimate 65% here.", 0.65),
    ("101", None),
    ("-5", None),
    ('{"probability": 150}', None),
    ('{"probability": true}', None),
    ('{"probability": "42"}', None),
    ('{"answer": "yes"}', None),
    ("30 or 40", None),
    ("yes", None),
    ("", None),
    ("I'm sorry, I can't provide a probability for death within 1 year.", None),
    ("I cannot determine this from a summary.", None),
])
def test_parse_probability(text, want):
    got = score.parse_answer(text, "probability")
    assert (got is None and want is None) or got == pytest.approx(want)


def test_parse_probability_missing_text():
    assert score.parse_answer(None, "probability") is None
    assert score.parse_answer(float("nan"), "probability") is None


def test_yes_no_parsing_unchanged():
    assert score.parse_answer('{"answer": "yes"}', "schema") == "yes"
    assert score.parse_answer("42", "schema") is None


@pytest.mark.parametrize("text,ok", [
    ('{"probability": 42}', True), ('{"probability": 42.0}', True), ('{"probability": 0}', True),
    ('{"probability": 42.5}', False), ('{"probability": 101}', False), ('{"probability": -1}', False),
    ('{"probability": "42"}', False), ('{"probability": true}', False), ('{"probability": Infinity}', False),
    ("42", False), ("not json", False),
])
def test_adapter_reader_is_strict(text, ok):
    out, err = read_probability(text)
    assert (out is not None) == ok and (err is None) == ok
    if ok:
        assert json.loads(out) == {"probability": int(json.loads(text)["probability"])}


# -- schema switch in the adapters ----------------------------------------------------------------

PARTS = [{"type": "text", "text": "Image 1:"}, {"type": "image", "data": b"\x89PNG", "mime": "image/png"},
         {"type": "text", "text": "Question?"}]


def test_claude_schema_switch():
    yn = vertex_claude.build_parts_request(PARTS, ["yes", "no"], effort="low", prefix_len=2)
    pr = vertex_claude.build_parts_request(PARTS, PROBABILITY, effort="low", prefix_len=2)
    assert yn["output_config"]["format"]["schema"]["properties"] == {"answer": {"type": "string", "enum": ["yes", "no"]}}
    s = pr["output_config"]["format"]["schema"]
    assert s["properties"]["probability"]["type"] == "integer" and s["required"] == ["probability"]
    assert "minimum" not in s["properties"]["probability"]
    assert pr["output_config"]["effort"] == "low" and pr["messages"] == yn["messages"]
    body = {"content": [{"type": "text", "text": '{"probability": 37}'}], "stop_reason": "end_turn", "usage": {}}
    assert vertex_claude.parse_response(body, PROBABILITY)[0] == '{"probability": 37}'
    bad = {"content": [{"type": "text", "text": '{"probability": 370}'}], "stop_reason": "end_turn", "usage": {}}
    text, _, err = vertex_claude.parse_response(bad, PROBABILITY)
    assert text is None and "outside 0-100" in err
    with pytest.raises(ValueError):
        vertex_claude.answer_schema(PROBABILITY, reason_field=True)


def test_gemini_schema_switch():
    yn = vertex_gemini.build_parts_request(PARTS, ["yes", "no"])
    pr = vertex_gemini.build_parts_request(PARTS, PROBABILITY)
    assert yn["generationConfig"]["responseMimeType"] == "text/x.enum"
    g = pr["generationConfig"]
    assert g["responseMimeType"] == "application/json"
    assert g["responseSchema"]["properties"]["probability"]["type"] == "INTEGER"
    assert pr["contents"] == yn["contents"]
    body = {"candidates": [{"content": {"parts": [{"text": '{"probability": 5}'}]}, "finishReason": "STOP"}]}
    assert vertex_gemini.parse_response(body, PROBABILITY)[0] == '{"probability": 5}'
    body["candidates"][0]["content"]["parts"][0]["text"] = "yes"
    text, _, err = vertex_gemini.parse_response(body, PROBABILITY)
    assert text is None and err


def test_aihub_schema_switch():
    yn = aihub_openai.build_parts_request(PARTS, ["yes", "no"], reasoning_effort="low")
    pr = aihub_openai.build_parts_request(PARTS, PROBABILITY, reasoning_effort="low")
    assert yn["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]
    s = pr["response_format"]["json_schema"]
    assert s["strict"] is True and s["schema"]["properties"]["probability"]["type"] == "integer"
    assert s["schema"]["required"] == ["probability"] and pr["messages"] == yn["messages"]
    reply = json.dumps({"choices": [{"message": {"content": '{"probability": 88}'}, "finish_reason": "stop"}], "usage": {}})
    assert aihub_openai.parse_response(200, "application/json", reply, PROBABILITY)[0] == '{"probability": 88}'
    reply = json.dumps({"choices": [{"message": {"content": '{"probability": 8.5}'}}], "usage": {}})
    ans, _, err, retry = aihub_openai.parse_response(200, "application/json", reply, PROBABILITY)
    assert ans is None and "whole number" in err and not retry


# -- card text and recipes ------------------------------------------------------------------------

def test_probability_line_replaces_only_the_answer_line():
    card = CARDS[1]
    q = card.question("death_2yr")
    text = card.text(q, "x", "summary", summary="Some text.")
    yn = get("opus").build(text, None).question
    pr = get("opus", answer_form="probability").build(text, None).question
    assert yn.endswith(ANSWER) and pr.endswith(ANSWER_PROB)
    assert pr[: -len(ANSWER_PROB)] == yn[: -len(ANSWER)]
    assert ANSWER_PROB == "Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number."
    with pytest.raises(ValueError):
        get("opus", answer_form="probability", answer_line="Say it.").build(text, None)


def test_arms_file_variants():
    want = {"argon_prob": ("argon", "gemini-4-argon"), "opus_prob": ("opus", "claude-opus-5-5"),
            "sonnet_prob": ("opus", "claude-sonnet-5-5"), "haiku_prob": ("opus", "claude-haiku-5-5"),
            "astra_prob": ("astra", "gpt-6-astra")}
    for variant, (recipe, model) in want.items():
        over, _ = arms.load(ARMS, variant)
        r = arms.apply(get(recipe), variant, over)
        assert r.answer_form == "probability" and r.model_name == model and r.name == f"{recipe}__{variant}"
        assert run.answer_spec(r) == PROBABILITY
        # Effort / thinking stays as the base recipe sets it.
        for k in ("effort", "thinking_level", "reasoning_effort"):
            assert r.setting(k) == get(recipe).setting(k)
    over, _ = arms.load(ARMS, "haiku_prob")
    assert dict(arms.apply(get("opus"), "haiku_prob", over).settings)["price_per_mtok"]["output"] == 0.5
    assert run.answer_spec(get("opus")) == run.CHOICES


# -- runner ---------------------------------------------------------------------------------------

class FakeBackend:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def ask_many(self, msgs):
        self.calls += len(msgs)
        return [{"text": self.reply(m), "usage": {"prompt_tokens": 10}} for m in msgs]


def test_run_probability_summary_arm_and_resume(tmp_path, monkeypatch):
    root = tmp_path / "evalset"
    monkeypatch.setenv("EVALSET_ROOT", str(root))
    (root / "samples").mkdir(parents=True)
    pd.DataFrame([dict(case_id=str(i), person_id=str(i), split="test", study="a", series="b", summary="Text.",
                       label__died_any_cause_1_yr=i % 2, label__died_any_cause_2_yr=(i + 1) % 2) for i in range(3)]
                 ).to_parquet(root / "samples" / "step1.parquet", index=False)
    r = get("opus", answer_form="probability").replace(name="opus__p")
    replies = iter(['{"probability": 70}', '{"probability": 0}', "garbled", '{"probability": 100}',
                    '{"probability": 12}', '{"probability": 55}'])
    fb = FakeBackend(lambda m: next(replies))
    res = run.run(r, 1, "summary", "t", backend=fb, root=root)
    assert fb.calls == 6 and len(res) == 6
    assert int(run.answered_mask(res["answer"]).sum()) == 5
    assert res["p_yes"].notna().sum() == 5 and set(res["p_yes"].dropna()) <= {0.7, 0.0, 1.0, 0.12, 0.55}
    assert (res["answer"].dropna().astype(float) == res["p_yes"].dropna()).all()
    # Rerun asks only the unreadable row; a 0 probability counts as answered.
    fb2 = FakeBackend(lambda m: '{"probability": 33}')
    res2 = run.run(r, 1, "summary", "t", backend=fb2, root=root)
    assert fb2.calls == 1 and int(run.answered_mask(res2["answer"]).sum()) == 6
    # The file on disk scores as AUROC.
    df = pd.read_csv(run.out_path("t", r, 1, "summary", root), dtype={"case_id": str, "person_id": str})
    t = score.score_rows(score.prepare(df), n_boot=50)
    assert t["auroc"].notna().all() and t["balanced_accuracy"].isna().all()


@pytest.mark.parametrize("v,ok", [("yes", True), ("no", True), ("maybe", False), (0.0, True), (1.0, True),
                                  (0.42, True), (1.5, False), (float("nan"), False), (None, False), (True, False)])
def test_is_answered(v, ok):
    assert run.is_answered(v) == ok


# -- AUROC ----------------------------------------------------------------------------------------

def test_auroc_hand_examples():
    # Positives at 0.35 and 0.8, negatives at 0.1 and 0.4: 3 of 4 pairs ranked right.
    assert score.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert score.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)  # a tie counts half
    assert score.auroc([0.9, 0.1], [0, 1]) == pytest.approx(0.0)
    assert np.isnan(score.auroc([0.2, 0.3], [1, 1]))


def test_auroc_matches_pairwise_count():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 101, 200) / 100
    y = rng.integers(0, 2, 200)
    pos, neg = s[y == 1], s[y == 0]
    want = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert score.auroc(s, y) == pytest.approx(want)


def _rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        for qid in ("death_1yr", "death_2yr"):
            y = int(rng.integers(0, 2))
            out.append({"step": 1, "recipe": "opus__p", "arm": "summary", "question_id": qid, "person_id": str(i),
                        "label": y, "answer": float(np.clip(0.3 * y + rng.random() * 0.7, 0, 1)), "refused": False})
    df = pd.DataFrame(out)
    df["p_yes"] = df["answer"]
    return df


def test_score_rows_auroc_macro_and_interval():
    df = _rows()
    df.loc[0, ["answer", "p_yes"]] = np.nan  # one unreadable row: left out of the AUROC and counted
    t = score.score_rows(score.prepare(df), n_boot=200).set_index("question_id")
    per_q = {q: score.auroc(g["p_yes"].dropna(), g.loc[g["p_yes"].notna(), "label"])
             for q, g in df.groupby("question_id")}
    assert t.loc["death_1yr", "auroc"] == pytest.approx(per_q["death_1yr"])
    assert t.loc["MACRO", "auroc"] == pytest.approx(np.mean(list(per_q.values())))
    assert t.loc["MACRO", "auroc_lo"] < t.loc["MACRO", "auroc"] < t.loc["MACRO", "auroc_hi"]
    assert t.loc["death_1yr", "n_prob"] == 59 and t.loc["death_1yr", "prob_missing_rate"] == pytest.approx(1 / 60)
    assert t["balanced_accuracy"].isna().all()


def test_yes_no_rows_score_as_before_with_no_auroc():
    df = _rows()
    df["answer"] = np.where(df["p_yes"] >= 0.5, "yes", "no")
    df["p_yes"] = np.nan
    t = score.score_rows(score.prepare(df), n_boot=50).set_index("question_id")
    g = df[df.question_id == "death_1yr"]
    want = score.balanced_accuracy(g["label"], [1 if a == "yes" else 0 for a in g["answer"]])
    assert t.loc["death_1yr", "balanced_accuracy"] == pytest.approx(want)
    assert "auroc" not in t.columns or t["auroc"].isna().all()


def test_mixed_files_score_each_run_its_own_way():
    p = _rows()
    yn = _rows().assign(recipe="opus")
    yn["answer"] = np.where(yn["p_yes"] >= 0.5, "yes", "no")
    yn["p_yes"] = np.nan
    t = score.score_rows(score.prepare(pd.concat([p, yn], ignore_index=True)), n_boot=20)
    assert t[t.recipe == "opus"]["balanced_accuracy"].notna().all() and t[t.recipe == "opus"]["auroc"].isna().all()
    assert t[t.recipe == "opus__p"]["auroc"].notna().all() and t[t.recipe == "opus__p"]["balanced_accuracy"].isna().all()
