"""Steps 6-8: more tumor-board questions (spec planning/vista_eval_vlm/2026-11-20-tb-agent-questions.md).

Answer key: the board's own answer in vista_bench v1.7 ``tb_classification_tasks``, read from the board note by a
language model; "Indeterminate/Deferred" (-1) is not scored. Prediction time is one week before the patient's first
board. The board-time summary (``compact_summary``) is written from the record up to that time.

  step 6  treatment questions, summary only: early-stage curative plan; first-line regimen for metastatic disease.
          Five options each, one probability per option (Phil 11-20). A patient is asked only the question(s) the
          board answered for them.
  step 7  board progression, summary only (the CT-pair version is step 4; same question text).
  step 8  pneumonitis vs infection, the most recent CT before the board, CT only (Phil 11-20: current CT to start).

The same question text is quoted by the tumor-board agents' starting instructions (Phil 11-20), so the agent and the
models asked directly differ in how they work, not in the wording. DRAFT wording until Phil approves the card page.
"""

from .base import ANSWER, ANSWER_OPTIONS, Card, Question, options_block

#: Shared by the summary arms of steps 6 and 7.
SUMMARY_OPENING = ("You are given a summary of this patient's record, written from the record up to one week before "
                   "their first thoracic tumor board.")

#: Codes and texts from vista_bench ``task_metadata.TASK_REGISTRY[<task>]["label_mapping"]`` (v1.7), in code order.
EARLY_STAGE_OPTIONS = (
    ("A", 0, "Systemic therapy followed by surgery, with or without adjuvant therapy"),
    ("B", 1, "Surgery followed by systemic therapy"),
    ("C", 2, "Surgery alone"),
    ("D", 3, "Radiation alone"),
    ("E", 4, "Definitive chemoradiation, with or without adjuvant systemic therapy"),
)
SYSTEMIC_OPTIONS = (
    ("A", 0, "Chemotherapy alone"),
    ("B", 1, "Chemotherapy with immunotherapy"),
    ("C", 2, "Chemotherapy with targeted therapy"),
    ("D", 3, "Targeted therapy alone"),
    ("E", 4, "Immunotherapy alone"),
)
PNEUMONITIS_OPTIONS = (
    ("A", 0, "Pneumonitis"),
    ("B", 1, "Infection"),
    ("C", 2, "Both pneumonitis and infection"),
    ("D", 3, "Cancer progression"),
    ("E", 4, "Neither"),
)

_EARLY = ("Question: The tumor board discussed curative-intent treatment for this patient's stage I-III non-small "
          "cell lung cancer. Which treatment will the board recommend?\n" + options_block(EARLY_STAGE_OPTIONS)
          + "\n" + ANSWER_OPTIONS)
_SYSTEMIC = ("Question: The tumor board discussed first-line systemic treatment for this patient's metastatic "
             "non-small cell lung cancer. Which regimen will the board recommend?\n" + options_block(SYSTEMIC_OPTIONS)
             + "\n" + ANSWER_OPTIONS)
#: Step 4's question, word for word (``board.CARD``).
_PROGRESSION = ("Question: As of one week before the tumor board, does the most recent imaging show cancer "
                "progression?\n- Yes: progression.\n- No: response or stable disease.\n" + ANSWER)
_PNEUMONITIS = ("Question: The tumor board discussed whether lung changes on this patient's imaging were "
                "pneumonitis, infection, both, cancer progression, or none of these. Which will the board conclude?\n"
                + options_block(PNEUMONITIS_OPTIONS) + "\n" + ANSWER_OPTIONS)

TREATMENT = Card(
    step=6,
    title="Tumor-board treatment recommendation (summary)",
    opening="",
    joiner="\n",
    questions=(
        Question("board_early_stage", "early_stage_management_answer", _EARLY, options=EARLY_STAGE_OPTIONS),
        Question("board_systemic", "optimal_systemic_therapy_answer", _SYSTEMIC, options=SYSTEMIC_OPTIONS),
    ),
    arms=("summary",),
    summary_opening=SUMMARY_OPENING,
    labelled_only=True,
)

PROGRESSION_SUMMARY = Card(
    step=7,
    title="Tumor-board progression (summary)",
    opening="",
    joiner="\n",
    questions=(Question("board_progression", "progression_assessment_answer", _PROGRESSION),),
    arms=("summary",),
    summary_opening=SUMMARY_OPENING,
    labelled_only=True,
)

PNEUMONITIS = Card(
    step=8,
    title="Tumor-board pneumonitis vs infection (CT)",
    opening=("These are axial slices of this patient's most recent chest CT before their thoracic tumor board, "
             "ordered from top to bottom and shown in {windows}."),
    joiner="\n\n",
    questions=(Question("board_pneumonitis", "pneumonitis_infection_answer", _PNEUMONITIS,
                        options=PNEUMONITIS_OPTIONS),),
    labelled_only=True,
)

CARDS = (TREATMENT, PROGRESSION_SUMMARY, PNEUMONITIS)
