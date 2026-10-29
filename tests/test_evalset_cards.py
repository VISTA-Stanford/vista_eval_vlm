"""Task cards and model recipes (src/evalset/cards, src/evalset/recipes).
Spec: planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md."""

import html
import re
from pathlib import Path

import pytest

from evalset import recipes
from evalset.cards import CARDS, all_questions, board

PAGE = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/plan-explainers/vista_eval_vlm/2026-10-27-task-card-questions.html")


def test_twenty_one_questions_with_unique_ids():
    qs = all_questions()
    assert len(qs) == 21
    assert [len(CARDS[s].questions) for s in (1, 2, 3, 4)] == [2, 17, 1, 1]
    assert len({(s, q.id) for s, _, q in qs}) == 21


@pytest.mark.skipif(not PAGE.exists(), reason="questions page is on the shared mount")
def test_cards_match_the_questions_page_word_for_word():
    """The page Phil reviewed is the source; rendered with its windows (lung / soft tissue / bone) and
    its board placeholder, every card question must equal the page's text, in page order."""
    page = PAGE.read_text()
    blocks = [html.unescape(b) for b in re.findall(r"<pre>(.*?)</pre>", page, flags=re.S)]
    labels = [html.unescape(x) for x in re.findall(r'<span class="lab">(.*?)</span>', page)]
    ours = []
    for step, card, q in all_questions():
        t = card.text(q, "lung, soft-tissue and bone windows")
        if step == 4:
            t = t.replace(board.AND_PRIOR, "[and the prior CT]")
        ours.append(t)
    assert len(blocks) == 21
    for i, (a, b) in enumerate(zip(blocks, ours)):
        assert a == b, f"question {i + 1} ({labels[i]}) differs from the page"
    # Answer keys: the page's label column names the same task (step 3 adds "(two-way)").
    assert [lab.split(" ")[0] for lab in labels] == [q.task for _, _, q in all_questions()]


def test_card_text_uses_the_recipes_windows():
    t = CARDS[2].text(CARDS[2].questions[0], recipes.get("medgemma").windows_phrase())
    assert t.startswith("These are axial slices of a chest CT, ordered from top (head) to bottom (feet), "
                        "each shown in wide, soft-tissue and brain windows.")


def test_summary_arm_only_on_step_1():
    c = CARDS[1]
    t = c.text(c.questions[1], "x", arm="summary", summary="  Stage IIIA adenocarcinoma ...  ")
    assert t.startswith("You are given a summary of this patient's record")
    assert "PATIENT SUMMARY:\nStage IIIA adenocarcinoma ...\n\nQuestion: Will this patient die" in t
    assert t.endswith("within 2 years of their cancer diagnosis?\nAnswer yes or no.")
    assert "axial slices" not in t
    with pytest.raises(ValueError):
        CARDS[2].text(CARDS[2].questions[0], "x", arm="summary", summary="s")
    with pytest.raises(ValueError):
        c.text(c.questions[0], "x", arm="summary", summary=" ")


# -- recipes ------------------------------------------------------------------------------------

def _scans(n_scans, k):
    return [(name, [bytes([i, j]) for j in range(k)]) for i, name in enumerate(["PRIOR", "CURRENT"][:n_scans])]


def test_medgemma_wraps_like_the_notebook():
    r = recipes.get("medgemma")
    card = CARDS[2]
    m = r.build(card.text(card.questions[0], r.windows_phrase()), [(None, [b"a", b"b", b"c"])])
    kinds = [(p["type"], p.get("text")) for p in m.parts]
    assert kinds[:6] == [("image", None), ("text", "SLICE 1"), ("image", None), ("text", "SLICE 2"),
                         ("image", None), ("text", "SLICE 3")]
    q = m.question
    assert q.startswith("You are an expert radiologist for chest CT.\nThese are axial slices")
    assert q.endswith("Answer with 'Final Answer: yes' or 'Final Answer: no'.")
    assert "Answer yes or no." not in q
    assert m.prefix_len == 6 and m.images == [b"a", b"b", b"c"]


def test_frontier_labels_before_images_and_pairs_named():
    r = recipes.get("argon")
    card = CARDS[3]
    m = r.build(card.text(card.questions[0], r.windows_phrase()), _scans(2, 2))
    texts = [p["text"] for p in m.parts[:-1] if p["type"] == "text"]
    assert texts == ["PRIOR scan, image 1:", "PRIOR scan, image 2:", "CURRENT scan, image 1:", "CURRENT scan, image 2:"]
    assert m.parts[0]["type"] == "text" and m.parts[1]["type"] == "image"
    assert m.question.endswith("Answer yes or no.") and not m.question.startswith("You are")


def test_astra_cap_and_pair_counts():
    r = recipes.get("astra")
    assert r.slices_per_scan(1) == 50 and 2 * r.slices_per_scan(2) == 50
    with pytest.raises(ValueError):
        r.build("Answer yes or no.", [(None, [b"x"] * 51)])
    # overrides make a copy, the registry recipe is untouched
    assert recipes.get("astra", n_slices=25).n_slices == 25 and recipes.get("astra").n_slices == 50


def test_pack_ids_shared_only_when_pixels_match():
    a, g, m = recipes.get("astra"), recipes.get("argon"), recipes.get("medgemma")
    assert a.pack_id(50) == g.pack_id(50)  # same render settings -> one pack for both
    assert a.pack_id(50) != m.pack_id(50)  # MedGemma: PIL-default JPEG quality
    assert a.pack_id(50) != a.pack_id(25)


def test_frontier_recipes_send_lossless_png():
    for name in ("astra", "argon"):
        r = recipes.get(name)
        assert r.image_format == "png" and r.jpeg_quality is None
        assert r.build("Answer yes or no.", [(None, [b"x"])]).parts[1]["mime"] == "image/png"
    assert recipes.get("medgemma").image_format == "jpeg"  # Google's published recipe
