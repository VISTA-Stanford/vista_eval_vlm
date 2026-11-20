"""One question card per step. ``CARDS[step]``; ``all_questions()`` lists every (step, card, question): the 21 of
steps 1-4 plus the board questions of steps 6-8 (step 5 lives on other branches)."""

from .base import Card, Question  # noqa: F401
from . import board, board_questions, death, lesions, progression

CARDS = {c.step: c for c in (death.CARD, lesions.CARD, progression.CARD, board.CARD, *board_questions.CARDS)}

#: Questions asked by default per step when a run names none (v1, Phil 10-29): step 2 = five findings, "2 lungs, 1
#: pneumonitis, 1 liver, 1 lymph node". "Two lungs" is read as the any-lung finding plus the right upper lobe (the
#: program board's reading, flagged to Phil to confirm). Other steps ask all their questions.
DEFAULT_QUESTIONS = {2: ("lesion_lungs", "lobe_rul", "pneumonitis", "lesion_liver", "lesion_lymph_nodes")}


def default_questions(step: int, questions=None):
    """The question ids a run asks: those given, else the step's v1 default (None = every question)."""
    if questions:
        return list(questions)
    d = DEFAULT_QUESTIONS.get(step)
    return list(d) if d else None


def all_questions():
    return [(s, CARDS[s], q) for s in sorted(CARDS) for q in CARDS[s].questions]
