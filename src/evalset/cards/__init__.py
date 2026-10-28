"""One question card per step. ``CARDS[step]``; ``all_questions()`` lists the 21 (step, card, question)."""

from .base import Card, Question  # noqa: F401
from . import board, death, lesions, progression

CARDS = {c.step: c for c in (death.CARD, lesions.CARD, progression.CARD, board.CARD)}


def all_questions():
    return [(s, CARDS[s], q) for s in sorted(CARDS) for q in CARDS[s].questions]
