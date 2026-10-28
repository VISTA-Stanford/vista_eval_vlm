"""Step 4: tumor-board progression, from the most recent chest CT before the board and its prior.

Answer key: the board's own answer (``progression_assessment_answer``: 1 Progression, 0 Response or
Stable). CT only, no summary (Phil 10-27). Every case in the step has a prior CT (the 334 of 491 board
cases the shared prior-CT link covers), so the page's optional "[and the prior CT]" is always filled.
"""

from .base import ANSWER, Card, Question

#: The page's placeholder; ``AND_PRIOR`` fills it.
AND_PRIOR = "and the prior CT"

CARD = Card(
    step=4,
    title="Tumor-board progression",
    opening=("These are axial slices of this patient's most recent chest CT before their thoracic tumor board "
             + AND_PRIOR + ", ordered from top to bottom and shown in {windows}."),
    joiner="\n\n",
    questions=(Question("board_progression", "progression_assessment_answer",
                        "Question: As of one week before the tumor board, does the most recent imaging show cancer "
                        "progression?\n- Yes: progression.\n- No: response or stable disease.\n" + ANSWER),),
    scan_names=("PRIOR", "MOST RECENT"),
)
