"""Step 1: death within 1 year / within 2 years, from the CT taken shortly before first treatment.

Answer key: vista_bench ``died_any_cause_1_yr`` / ``died_any_cause_2_yr``; the horizon counts from
diagnosis (vista_bench progression_recurrence_survival.md). One call per horizon. Arms: CT, and
Tuomo's treatment-start summary alone (Phil 10-27). No diagnosis-to-scan interval (Phil 10-27).
"""

from .base import ANSWER, Card, Question

_BODY = ("The patient was recently diagnosed with thoracic cancer, and this scan was taken shortly before "
         "their first cancer treatment.\n\nQuestion: Will this patient die, from any cause, within {h} of their "
         "cancer diagnosis?\n" + ANSWER)

CARD = Card(
    step=1,
    title="Death, pre-treatment CT (one call per horizon)",
    opening="These are axial slices of a chest CT, ordered from top (head) to bottom (feet), each shown in {windows}.",
    joiner="\n",
    questions=(
        Question("death_1yr", "died_any_cause_1_yr", _BODY.format(h="1 year")),
        Question("death_2yr", "died_any_cause_2_yr", _BODY.format(h="2 years")),
    ),
    arms=("ct", "summary"),
    summary_opening=("You are given a summary of this patient's record, written from the record up to one week "
                     "before first treatment."),
)
