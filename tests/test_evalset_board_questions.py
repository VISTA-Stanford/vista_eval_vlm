"""Board question cards (steps 6-8) and the several-option answer: one probability per option (Phil 11-20)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evalset import run, score
from evalset.cards import CARDS, board, board_questions
from evalset.cards.base import ANSWER_OPTIONS
from evalset.recipes import get
from models import vertex_claude, vertex_gemini
from models.base import Distribution, read_distribution

KEYS = ("A", "B", "C", "D", "E")
TASK_METADATA = Path.home() / "code/vista_bench/src/vistabench/task_metadata.py"


# -- the adapters' reader -------------------------------------------------------------------------

@pytest.mark.parametrize("reply,ok", [
    ('{"A": 40, "B": 30, "C": 20, "D": 10, "E": 0}', True),
    ('{"A": 40, "B": 30, "C": 20, "D": 5, "E": 0}', True),     # 95: within 10 of 100
    ('{"A": 40, "B": 30, "C": 20, "D": 0, "E": 0}', True),     # 90: the edge
    ('{"A": 40, "B": 30, "C": 15, "D": 0, "E": 0}', False),    # 85
    ('{"A": 60, "B": 30, "C": 20, "D": 1, "E": 0}', False),    # 111
    ('{"A": 40, "B": 30, "C": 20, "D": 10}', False),           # an option missing
    ('{"A": 40, "B": 30, "C": 20, "D": 10, "E": 0, "F": 0}', False),
    ('{"A": 40.5, "B": 29.5, "C": 20, "D": 10, "E": 0}', False),
    ('{"A": 140, "B": -40, "C": 0, "D": 0, "E": 0}', False),
    ('{"A": true, "B": 30, "C": 20, "D": 10, "E": 40}', False),
    ("A", False),
])
def test_read_distribution(reply, ok):
    out, err = read_distribution(reply, KEYS)
    assert (out is not None) == ok and (err is None) == ok
    if ok:
        assert list(json.loads(out)) == list(KEYS)


def test_gemini_forces_one_whole_number_per_option():
    body = vertex_gemini.build_request("q", [], Distribution(KEYS))
    sch = body["generationConfig"]["responseSchema"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert list(sch["properties"]) == list(KEYS) and sch["required"] == list(KEYS) == sch["propertyOrdering"]
    assert all(p["type"] == "INTEGER" and p["minimum"] == 0 and p["maximum"] == 100 for p in sch["properties"].values())
    good = {"candidates": [{"content": {"parts": [{"text": '{"A": 10, "B": 20, "C": 30, "D": 40, "E": 0}'}]},
                            "finishReason": "STOP"}]}
    text, _, err = vertex_gemini.parse_response(good, Distribution(KEYS))
    assert err is None and json.loads(text)["D"] == 40
    good["candidates"][0]["content"]["parts"][0]["text"] = '{"A": 10, "B": 20, "C": 30, "D": 4, "E": 0}'
    text, _, err = vertex_gemini.parse_response(good, Distribution(KEYS))
    assert text is None and "add up to 64" in err


def test_claude_forces_one_whole_number_per_option():
    sch = vertex_claude.answer_schema(Distribution(KEYS))
    assert list(sch["properties"]) == list(KEYS) and sch["required"] == list(KEYS)
    assert sch["additionalProperties"] is False
    body = {"content": [{"type": "text", "text": '{"A": 0, "B": 0, "C": 100, "D": 0, "E": 0}'}],
            "stop_reason": "end_turn", "usage": {}}
    text, _, err = vertex_claude.parse_response(body, Distribution(KEYS))
    assert err is None and json.loads(text)["C"] == 100
    # The yes/no and 0-100 schemas are unchanged.
    assert vertex_claude.answer_schema(["yes", "no"])["properties"]["answer"]["enum"] == ["yes", "no"]


# -- the cards ------------------------------------------------------------------------------------

def test_board_cards_registered_with_their_arms():
    assert CARDS[6].arms == ("summary",) and CARDS[7].arms == ("summary",) and CARDS[8].arms == ("ct",)
    assert [q.task for q in CARDS[6].questions] == ["early_stage_management_answer", "optimal_systemic_therapy_answer"]
    assert all(c.labelled_only for c in board_questions.CARDS)


def test_progression_from_the_summary_asks_step_4s_question_word_for_word():
    b4 = board.CARD.questions[0].body
    b7 = CARDS[7].questions[0].body
    assert b4[b4.index("Question:"):] == b7


def _label_mapping(task: str) -> dict:
    """{code: text} for ``task`` from vista_bench's task registry source (read as text, not imported)."""
    src = TASK_METADATA.read_text()
    block = src[src.index(f'"{task}": {{'):]
    block = block[block.index('"label_mapping": {') + len('"label_mapping": {'):]
    block = block[: block.index("}")]
    return {int(k): v for k, v in re.findall(r"(-?\d+):\s*\"([^\"]+)\"", block)}


@pytest.mark.skipif(not TASK_METADATA.exists(), reason="vista_bench checkout not on this machine")
@pytest.mark.parametrize("q", [q for c in board_questions.CARDS for q in c.questions if q.options], ids=lambda q: q.id)
def test_option_codes_match_vista_bench(q):
    """Each option stands for the vista_bench answer code whose text it carries (wording lightly edited)."""
    want = {c: t for c, t in _label_mapping(q.task).items() if c >= 0}
    assert sorted(want) == [c for _, c, _ in q.options]
    norm = lambda t: re.sub(r"[^a-z]", "", t.lower().replace("+/-", "withorwithout").replace("both", ""))
    for _, c, text in q.options:
        a, b = norm(text), norm(want[c])
        assert a == b or a in b or b in a, (c, text, want[c])


def test_summary_card_text_keeps_its_options_line_under_every_recipe():
    q = CARDS[6].questions[0]
    t = CARDS[6].text(q, "", "summary", summary="Some summary.")
    assert t.startswith(board_questions.SUMMARY_OPENING) and "PATIENT SUMMARY:\nSome summary." in t
    assert "\nA. Systemic therapy followed by surgery" in t and t.endswith(ANSWER_OPTIONS)
    for name, form in (("argon", "probability"), ("opus", "schema"), ("opus", "probability")):
        assert get(name, answer_form=form).wrap_text(t, False) == t


def test_answer_shape_per_card():
    prob = get("argon", answer_form="probability")
    assert run.answer_spec(prob, CARDS[6]) == Distribution(KEYS)
    assert run.answer_spec(prob, CARDS[8]) == Distribution(KEYS)
    assert run.answer_spec(prob, CARDS[7]) == run.PROBABILITY
    assert run.answer_spec(get("argon"), CARDS[7]) == run.CHOICES


def test_only_the_answered_question_is_asked():
    r = get("opus", answer_form="probability")
    row = {"case_id": "1", "person_id": "1", "summary": "S.", "label__early_stage_management_answer": 2.0,
           "label__optimal_systemic_therapy_answer": float("nan")}
    qm = run.messages_for_case(r, CARDS[6], row, "summary")
    assert [q.id for q, _ in qm] == ["board_early_stage"]


# -- parsing and scoring --------------------------------------------------------------------------

def test_parse_options_rescales_and_rejects():
    opts = CARDS[8].questions[0].options
    got = json.loads(score.parse_options('{"A": 50, "B": 25, "C": 20, "D": 0, "E": 0}', opts))
    assert list(got) == ["0", "1", "2", "3", "4"] and abs(sum(got.values()) - 1) < 1e-9
    assert abs(got["0"] - 50 / 95) < 1e-9
    assert score.parse_options('{"A": 50, "B": 25, "C": 0, "D": 0, "E": 0}', opts) is None
    assert score.parse_options('{"A": 100}', opts) is None
    assert score.parse_options(None, opts) is None
    assert run.is_answered(score.parse_options('{"A": 100, "B": 0, "C": 0, "D": 0, "E": 0}', opts))
    assert not run.is_answered('{"0": 0.5}')


def _rows(answers, labels, qid="board_pneumonitis", step=8):
    return pd.DataFrame([dict(step=step, recipe="m", arm="ct", question_id=qid, person_id=str(i), label=lab,
                              answer=a, refused=False) for i, (a, lab) in enumerate(zip(answers, labels))])


def test_option_table_hand_worked():
    """Worked by hand. Readable p0 (answer 0), p1 (1), p2 (0); p3 (1) unreadable.
    Answer 0: yes 0.6 and 0.2 vs no 0.5 -> 1 of 2 pairs -> 0.5. Answer 1: yes 0.3 vs no 0.4, 0.8 -> 0.
    Answers 2-4 have no yes rows -> skipped. Macro AUROC 0.25.
    Top pick: p0 -> 0 right, p1 -> 0 wrong, p2 -> 1 wrong, p3 wrong: answer 0 1/2, answer 1 0/2 -> BA 0.25.
    Top two: p0 {0,1} yes, p1 {0,1} yes, p2 {1,0} yes, p3 no -> 0.75."""
    j = lambda d: json.dumps({str(k): v for k, v in d.items()})
    answers = [j({0: .6, 1: .4, 2: 0, 3: 0, 4: 0}), j({0: .5, 1: .3, 2: .2, 3: 0, 4: 0}),
               j({0: .2, 1: .8, 2: 0, 3: 0, 4: 0}), None]
    t = score.option_table(_rows(answers, [0, 1, 0, 1]), n_boot=50).set_index("question_id")
    r = t.loc["board_pneumonitis"]
    assert r["auroc"] == pytest.approx(0.25) and r["balanced_accuracy"] == pytest.approx(0.25)
    assert r["top2"] == pytest.approx(0.75) and r["unparseable_rate"] == pytest.approx(0.25)
    assert r["chance"] == pytest.approx(0.2) and r["top2_chance"] == pytest.approx(0.4)
    assert json.loads(r["option_auroc"]) == {"0": 0.5, "1": 0.0, "2": None, "3": None, "4": None}
    assert r["auroc_lo"] <= r["auroc_hi"]


def test_option_table_perfect_ranking_and_macro_row():
    j = lambda c: json.dumps({str(k): (0.9 if k == c else 0.025) for k in range(5)})
    labels6 = [0, 2, 4, 0, 2, 4]
    a = _rows([j(c) for c in labels6], labels6, "board_early_stage", 6).assign(arm="summary")
    labels6b = [1, 3, 1, 3]
    b = _rows([j(c) for c in labels6b], labels6b, "board_systemic", 6).assign(arm="summary")
    b["person_id"] = ["10", "11", "12", "13"]
    t = score.option_table(pd.concat([a, b], ignore_index=True), n_boot=50).set_index("question_id")
    assert set(t.index) == {"board_early_stage", "board_systemic", "MACRO"}
    assert (t[["auroc", "balanced_accuracy", "top2"]] == 1.0).all().all()


class FakeBackend:
    def __init__(self, reply):
        self.reply, self.calls, self.choices = reply, 0, None

    def use_card(self, card):
        self.choices = run.answer_spec(get("opus", answer_form="probability"), card)

    def ask_many(self, msgs):
        self.calls += len(msgs)
        return [{"text": self.reply(m), "usage": {"prompt_tokens": 10}} for m in msgs]


def test_run_step6_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "evalset"
    monkeypatch.setenv("EVALSET_ROOT", str(root))
    (root / "samples").mkdir(parents=True)
    pd.DataFrame([
        dict(case_id="1", person_id="1", split="test", summary="S1.", label__early_stage_management_answer=2.0,
             label__optimal_systemic_therapy_answer=np.nan),
        dict(case_id="2", person_id="2", split="test", summary="S2.", label__early_stage_management_answer=np.nan,
             label__optimal_systemic_therapy_answer=3.0),
        dict(case_id="3", person_id="3", split="test", summary="S3.", label__early_stage_management_answer=4.0,
             label__optimal_systemic_therapy_answer=np.nan),
    ]).to_parquet(root / "samples" / "step6_test.parquet", index=False)
    r = get("opus", answer_form="probability").replace(name="opus__p")
    replies = iter(['{"A": 0, "B": 0, "C": 90, "D": 0, "E": 10}', '{"A": 10, "B": 0, "C": 0, "D": 90, "E": 0}',
                    "garbled"])
    fb = FakeBackend(lambda m: next(replies))
    res = run.run(r, 6, "summary", "t", backend=fb, root=root, which="test")
    assert fb.calls == 3 and fb.choices == Distribution(KEYS)
    assert int(run.answered_mask(res["answer"]).sum()) == 2
    fb2 = FakeBackend(lambda m: '{"A": 0, "B": 0, "C": 0, "D": 0, "E": 100}')
    res2 = run.run(r, 6, "summary", "t", backend=fb2, root=root, which="test")
    assert fb2.calls == 1 and int(run.answered_mask(res2["answer"]).sum()) == 3
    df = pd.read_csv(run.out_path("t", r, 6, "summary", root), dtype={"case_id": str, "person_id": str})
    assert score.is_option_row(df).all()
    o = df.assign(label=df["label"].astype(int))
    t = score.option_table(o, n_boot=20).set_index("question_id")
    assert t.loc["board_early_stage", "top2"] == 1.0 and t.loc["board_systemic", "balanced_accuracy"] == 1.0
