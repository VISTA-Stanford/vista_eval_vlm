"""One question card per step. ``CARDS[step]``; ``all_questions()`` lists the eval set's 21 (step, card, question).

Step 5 is the positive control (``metadata``): it has a card so runs and scores treat it like any step, but it is not
one of the eval set's questions, so ``all_questions`` leaves it out."""

from .base import Card, Question  # noqa: F401
from . import board, death, lesions, metadata, progression

CARDS = {c.step: c for c in (death.CARD, lesions.CARD, progression.CARD, board.CARD, metadata.CARD)}
#: The eval set's steps; step 5 is the control.
EVAL_STEPS = (1, 2, 3, 4)

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
    return [(s, CARDS[s], q) for s in EVAL_STEPS for q in CARDS[s].questions]
