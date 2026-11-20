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
#: Replaces ANSWER for recipes with ``answer_form: "probability"`` (Phil 11-17: AUROC for the LLM rows); the rest of
#: the card text is unchanged, so the question matches the yes/no runs word for word.
ANSWER_PROB = "Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number."
#: Answer line of a several-option question (tumor board treatment and pneumonitis cards; Phil's wording 11-20): one
#: call, one probability per option letter. Kept for every recipe (it is not a yes/no line to replace).
ANSWER_OPTIONS = "Give each letter a probability from 0 to 100. These probabilities must add up to 100."


@dataclasses.dataclass(frozen=True)
class Question:
    id: str          # short name, unique within the card
    task: str        # vista_bench task whose label is the answer key
    body: str        # everything after the opening image line
    yes_label: int = 1
    #: Several-option question: ((key, label code, option text), ...), keys "A", "B", ... in the order shown. The
    #: label code is the vista_bench answer the option stands for. Empty for a yes/no question.
    options: tuple = ()

    def option_code(self, key: str) -> int:
        for k, code, _ in self.options:
            if k == key:
                return code
        raise KeyError(key)


def options_block(options) -> str:
    """The option lines shown under a several-option question: "A. <text>" one per line."""
    return "\n".join(f"{k}. {text}" for k, _, text in options)


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
    #: Ask a question only of cases with a scored answer for it (board cards: an early-stage patient is not asked the
    #: metastatic-regimen question). Other cards ask every question of every case, as before.
    labelled_only: bool = False

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
