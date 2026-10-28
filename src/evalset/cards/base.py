"""What a task card is: the questions for one eval-set step, each matched to a vista_bench label.

Question text is copied from the questions page Phil reviewed
(plan-explainers/vista_eval_vlm/2026-10-27-task-card-questions.html, built by
session/2026-10-27-task-card-questions-build.py). The page's opening image line names lung,
soft-tissue and bone windows; here that line is filled from the recipe's real windows
(``windows_phrase``). Rendered with the chest set, every question equals the page's text
(``tests/test_evalset_cards.py`` checks against the page when the mount is there).
"""

from __future__ import annotations

import dataclasses

ANSWER = "Answer yes or no."


@dataclasses.dataclass(frozen=True)
class Question:
    id: str          # short name, unique within the card
    task: str        # vista_bench task whose label is the answer key
    body: str        # everything after the opening image line
    yes_label: int = 1


@dataclasses.dataclass(frozen=True)
class Card:
    step: int
    title: str
    opening: str                 # image line, with {windows}
    joiner: str                  # between the image line and the body: "\n" or "\n\n"
    questions: tuple
    scan_names: tuple = (None,)  # image-order scan labels; (None,) = one scan, unlabelled
    arms: tuple = ("ct",)
    summary_opening: str = ""    # text-only summary arm (card 1)
    question_start: str = "Question:"

    def question(self, qid: str) -> Question:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)

    def text(self, q: Question, windows_phrase: str, arm: str = "ct", summary: str | None = None) -> str:
        if arm not in self.arms:
            raise ValueError(f"step {self.step} has no {arm!r} arm (arms: {self.arms})")
        if arm == "ct":
            return self.opening.format(windows=windows_phrase) + self.joiner + q.body
        if arm == "summary":
            if not summary or not str(summary).strip():
                raise ValueError("summary arm needs the patient's summary text")
            ask = q.body[q.body.index(self.question_start):]
            return f"{self.summary_opening}\n\nPATIENT SUMMARY:\n{str(summary).strip()}\n\n{ask}"
        raise ValueError(arm)
