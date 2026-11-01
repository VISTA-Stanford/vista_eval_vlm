"""Qwen3-VL-32B and OctoMed-7B recipes (spec planning/vista_eval_vlm/2026-11-01-open-vlm-fair-eval.md): their
messages, OctoMed's options block and boxed answers, A/B token probabilities, and pack reuse with Astra."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalset import arms, run, score  # noqa: E402
from evalset.cards import CARDS  # noqa: E402
from evalset.recipes import get  # noqa: E402

IMG = b"\x89PNG fake"


def _msg(name, step=2, arm="ct"):
    r = get(name)
    card = CARDS[step]
    text = card.text(card.questions[0], r.windows_phrase(), arm, summary="S")
    n = r.slices_per_scan(len(card.scan_names))
    scans = None if arm != "ct" else [(s, [IMG] * n) for s in card.scan_names]
    return r, r.build(text, scans)


def test_qwen_uses_astras_packs_and_plain_prompt():
    q, a = get("qwen3vl"), get("astra")
    assert (q.n_slices, q.n_slices_pair) == (50, 25)
    assert q.pack_id(50) == a.pack_id(50) and q.pack_id(25) == a.pack_id(25)
    r, m = _msg("qwen3vl")
    assert len(m.images) == 50 and m.system == "" and "expert radiologist" not in m.question
    assert m.parts[1]["text"] == "SLICE 1\n"
    assert m.question.endswith("Answer with 'Final Answer: yes' or 'Final Answer: no'.")
    assert r.setting("temperature") == 0.0 and r.setting("max_new_tokens") == 2000


def test_octomed_options_system_and_budget():
    r, m = _msg("octomed")
    assert len(m.images) == 10 and m.system == "You are a helpful assistant."
    assert m.question.endswith("A: Yes\nB: No\nPlease reason step-by-step, and put your final answer within \\boxed{}.")
    assert "Answer yes or no." not in m.question
    assert dict(r.setting("mm_processor_kwargs")) == {"min_pixels": 262144, "max_pixels": 262144}
    assert (r.setting("temperature"), r.setting("top_p"), r.setting("max_new_tokens")) == (0.6, 0.95, 8192)
    assert r.setting("seed") is not None


def test_octomed_two_scan_card_gets_5_plus_5():
    _, m = _msg("octomed", step=3)
    assert len(m.images) == 10 and m.parts[1]["text"] == "PRIOR SLICE 1\n"


def test_summary_arm_has_no_images():
    _, m = _msg("octomed", step=1, arm="summary")
    assert m.images == [] and m.question.endswith("\\boxed{}.")


def test_medgemma_labels_unchanged():
    _, m = _msg("medgemma")
    assert m.parts[1]["text"] == "SLICE 1"


@pytest.mark.parametrize("text,want", [
    ("<think>maybe A</think>\nThe answer is \\boxed{B}", "no"),
    ("\\boxed{A}", "yes"),
    ("\\boxed{\\text{A: Yes}}", "yes"),
    ("$\\boxed{(B)}$", "no"),
    ("\\boxed{no}", "no"),
    ("<think>it could be \\boxed{A}</think>\nB", "no"),       # an option weighed in the think block never counts
    ("Answer: (A) Yes", "yes"),
    ("<think>runs out at the token limit \\boxed{A}", None),  # think never closed
    ("There is a lesion in the lung.", None),
    ("", None),
])
def test_boxed_choice(text, want):
    assert score.parse_answer(text, "boxed_choice") == want


def test_p_yes_reads_option_letters_only_for_boxed_choice():
    lp = json.dumps({"top": {"A": -0.5, "B": -1.5, "C": -3.0}})
    assert run.p_yes(lp, "boxed_choice") == pytest.approx(1 / (1 + 2.718281828 ** -1.0), rel=1e-6)
    assert run.p_yes(lp, "final_answer") is None


def _completion(tokens):
    ids = list(range(len(tokens)))
    lps = [{i: SimpleNamespace(decoded_token=t, logprob=-0.1)} for i, t in zip(ids, tokens)]
    return SimpleNamespace(token_ids=ids, logprobs=lps)


def test_answer_logprobs_finds_the_boxed_letter():
    c = _completion(["</think>", "\\boxed", "{", "B", "}"])
    assert json.loads(run.answer_logprobs(c, run.ANSWER_TOKENS["boxed_choice"]))["position"] == 3
    assert run.answer_logprobs(c, run.ANSWER_TOKENS["final_answer"]) is None


def test_answer_logprobs_prefers_the_letter_over_the_word_after_it():
    # OctoMed's real form (control 11-01): "\boxed{B: No}". The choice is the letter; the word after it is decided.
    c = _completion(["</think>", "\\boxed", "{", "B", ":", " No", "}"])
    assert json.loads(run.answer_logprobs(c, run.ANSWER_TOKENS["boxed_choice"]))["position"] == 3
    # No letter written: fall back to the word.
    c = _completion(["\\boxed", "{", "Yes", "}"])
    assert json.loads(run.answer_logprobs(c, run.ANSWER_TOKENS["boxed_choice"]))["position"] == 2


def test_ablation_variants_apply():
    f = Path("/data/fries/users/padamson/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/arms/open_vlm_fair_eval.yaml")
    if not f.exists():
        pytest.skip("arms file lives on the mount")
    for name, v, n, ctx in [("qwen3vl", "qwen3vl_85", 85, 32768), ("octomed", "octomed_50", 50, 32768),
                            ("octomed", "octomed_85", 85, 49152)]:
        r = arms.apply(get(name), v, arms.load(f, v)[0])
        assert r.n_slices == n and int(r.setting("max_model_len")) == ctx and r.name == f"{name}__{v}"
        assert r.answer_form == get(name).answer_form
