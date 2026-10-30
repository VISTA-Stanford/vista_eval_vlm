"""Step-5 positive control and the MedGemma system message (spec planning/vista_eval_vlm/2026-10-30-medgemma-fair-eval.md)."""

import math

import pandas as pd
import pytest

from evalset import recipes, run, samples
from evalset.cards import CARDS, all_questions


def nan(x):
    return isinstance(x, float) and math.isnan(x)


def test_control_card_is_step_5_and_not_an_eval_set_question():
    c = CARDS[5]
    assert [q.id for q in c.questions] == ["meta_iv_contrast", "meta_female"]
    assert all(s != 5 for s, _, _ in all_questions())
    t = c.text(c.questions[0], recipes.get("medgemma").windows_phrase())
    assert t.startswith("These are axial slices of a chest CT") and t.endswith("Answer yes or no.")


@pytest.mark.parametrize("desc,want", [
    ("CT CHEST W IV CONTRAST", 1.0), ("CT THORAX WITH CONTRAST", 1.0), ("CT CHEST ABD PEL W IMGCT0148", 1.0),
    ("CT CHEST ANGIO W PE", 1.0), ("CT CHEST WO IV CONTRAST", 0.0), ("CT CHEST WITHOUT CONTRAST", 0.0),
    ("CT CHEST WO IV CONTRAST VOLUMETRIC ADD HRCT WITH 2 RESP PHASE", 0.0),
    ("CT CHEST ANGIOGRAPHY W AND WO IV CONTRAST", None), ("CT THORAX", None), ("THORAX^ROUTINE_CHEST (ADULT)", None),
    ("", None)])
def test_contrast_from_description(desc, want):
    got = samples.contrast_from_description(desc)
    assert nan(got) if want is None else got == want


@pytest.mark.parametrize("agent,want", [
    ("ISOVUE 370", 1.0), ("H20 & ISOVUE 370", 1.0), ("IOSVUE 370", 1.0), ("OMNI350/46ML", 1.0), ("APPLIED", 1.0),
    ("CE", 1.0), ("", 0.0), ("NONE", 0.0), ("H20", 0.0), ("H2O", 0.0), ("UNDEFINED", 0.0), ("BARIUM", None)])
def test_contrast_from_agent(agent, want):
    got = samples.contrast_from_agent(agent)
    assert nan(got) if want is None else got == want


def test_step5_keeps_agreeing_scans_and_balances_on_contrast(monkeypatch):
    hdr = {"s1": ("CT CHEST W IV CONTRAST", "ISOVUE 370", "F"), "s2": ("CT CHEST W IV CONTRAST", "ISOVUE 300", "M"),
           "s3": ("CT CHEST W IV CONTRAST", "NONE", "F"),       # disagree -> no contrast label
           "s4": ("CT CHEST WO IV CONTRAST", "", "M"), "s5": ("CT THORAX", "", "F"),
           "s6": ("CT CHEST W IV CONTRAST", "OMNIPAQUE 350", "F")}
    base = pd.DataFrame({"case_id": list(hdr), "person_id": ["p"] * 6, "split": ["test"] * 6,
                         "ct_accession_number": list(hdr), "study": list(hdr), "series": ["x"] * 6})
    monkeypatch.setattr(samples, "load", lambda step, *a, **k: base.assign(extra=1))
    read = lambda study, series: dict(zip(("desc", "agent", "sex"), hdr[study]))
    df = samples.step5(None, read=read)
    assert df["label__meta_iv_contrast"].tolist().count(0) == 1 == df["label__meta_iv_contrast"].tolist().count(1)
    assert "s3" not in set(df["case_id"]) and "s5" not in set(df["case_id"]) and "extra" not in df.columns
    assert set(df["label__meta_female"]) <= {0.0, 1.0}
    assert samples.step5(None, read=read).equals(df)  # seeded


# -- system message -------------------------------------------------------------------------------

def test_system_message_rides_on_the_message_and_changes_its_key():
    plain = recipes.get("medgemma")
    think = plain.replace(system="SYSTEM INSTRUCTION: think silently if needed.")
    a, b = plain.build("Q?\nAnswer yes or no.", [(None, [b"x"])]), think.build("Q?\nAnswer yes or no.", [(None, [b"x"])])
    assert a.system == "" and b.system == "SYSTEM INSTRUCTION: think silently if needed."
    assert a.parts == b.parts and run.message_key(a) != run.message_key(b)


def test_message_key_without_system_is_unchanged():
    """Old answer files keep matching: the key of a call with no system message is the parts-only hash."""
    import hashlib
    m = recipes.get("medgemma").build("Q?\nAnswer yes or no.", [(None, [b"x"])])
    h = hashlib.sha256()
    for p in m.parts:
        h.update(b"T" + p["text"].encode() if p["type"] == "text" else b"I" + run._image_digest(p["data"]))
    assert run.message_key(m) == h.hexdigest()


def test_vllm_sends_system_first_and_api_refuses_it():
    class LLM:
        def chat(self, convs, params, use_tqdm):
            self.convs = convs
            return []
    be = object.__new__(run.VLLMBackend)
    be.llm, be.params = LLM(), None
    r = recipes.get("medgemma")
    be.ask_many([r.replace(system="S").build("Answer yes or no.", [(None, [b"x"])]), r.build("Answer yes or no.", [(None, [b"x"])])])
    assert [m["role"] for m in be.llm.convs[0]] == ["system", "user"] and be.llm.convs[0][0]["content"] == "S"
    assert [m["role"] for m in be.llm.convs[1]] == ["user"]
    with pytest.raises(ValueError):
        run.APIBackend._wire(r.replace(system="S").build("Answer yes or no.", [(None, [b"x"])]))
