"""The tumor board questions' option tables, with no engine import, so scripts/build_data.py can read them. config.py
imports everything here. (letter, code, text[, label-v1 value]).
"""

#: Options: (letter, vista_bench label code, text) in code order, verbatim from the card.
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



# Boards (TB_AGENT=boards-record): the rubric v1 board questions (spec planning/vista_eval_vlm/2026-11-22-tb-board-
# questions-agent.md). Every board: decision type (T1) and intent (T2); routed on pre-board facts only (vista_bench
# tb_classification.route_board): early-stage plan (T3), after surgery (T3b), first-line (T4), each with 'more workup
# first' (Phil 2026-11-22). Options are the rubric's layers (planning/vista_eval_vlm/tb-rubric-v1-draft.md); T3/T4
# letters A-E are the old cards' options word for word. DECIDE: build 2026-11-22 (wording for Phil's look before the
# cost check).
BOARDS_OPENING = ("You are given this patient's electronic health record, cut off one week before a thoracic tumor "
                  "board encounter.")
DECISION_OPTIONS = (
    ("A", 0, "More workup before deciding (biopsy, staging, imaging, molecular testing, fitness testing or a consult)", "more_workup"),
    ("B", 1, "Start a new treatment", "start_new_treatment"),
    ("C", 2, "Continue the current treatment", "continue_current"),
    ("D", 3, "Switch or stop the current treatment", "switch_or_stop"),
    ("E", 4, "Add local therapy (surgery or radiation) to ongoing systemic therapy", "add_local_to_systemic"),
    ("F", 5, "Surveillance or observation", "surveillance"),
    ("G", 6, "Refer (a clinical trial, another specialty or a second opinion)", "refer"),
    ("H", 7, "Best supportive care or hospice", "supportive_or_hospice"),
    ("I", 8, "Defer the decision (no consensus, or the patient's decision is pending)", "deferred"),
)
INTENT_OPTIONS = (
    ("A", 0, "Curative", "curative"),
    ("B", 1, "Palliative, to prolong life", "palliative_life_prolonging"),
    ("C", 2, "Symptom control only", "symptom_control"),
    ("D", 3, "Undetermined until more workup is done", "undetermined_pending_workup"),
    ("E", 4, "Not applicable: no treatment is being decided", "not_applicable"),
)
WORKUP_FIRST = "More workup first, before choosing a treatment"
BOARD_EARLY_OPTIONS = tuple((k, c, t, k) for k, c, t in EARLY_STAGE_OPTIONS) + (
    ("F", 5, "Systemic or targeted therapy first, with local treatment deferred", "F_systemic_first_local_deferred"),
    ("G", 6, "Another treatment plan", "other"),
    ("H", 7, WORKUP_FIRST, "more_workup_first"),
)
AFTER_SURGERY_OPTIONS = (
    ("A", 0, "Adjuvant or consolidation systemic therapy", "adjuvant_systemic"),
    ("B", 1, "Postoperative radiation", "PORT"),
    ("C", 2, "Both systemic therapy and postoperative radiation", "both"),
    ("D", 3, "Observation", "observation"),
    ("E", 4, "Another plan", "other"),
    ("F", 5, WORKUP_FIRST, "more_workup_first"),
)
BOARD_FIRST_LINE_OPTIONS = tuple((k, c, t, k) for k, c, t in SYSTEMIC_OPTIONS) + (
    ("F", 5, "Targeted therapy combined with an antibody (for example amivantamab with lazertinib)", "F_targeted_plus_antibody"),
    ("G", 6, "Another regimen", "other"),
    ("H", 7, WORKUP_FIRST, "more_workup_first"),
)


#: Question key -> its options with label-v1 values, in config.BOARD_QUESTIONS order.
BOARD_OPTION_TABLES = {"decision_type": DECISION_OPTIONS, "intent": INTENT_OPTIONS, "early_stage_plan": BOARD_EARLY_OPTIONS,
                       "after_surgery_plan": AFTER_SURGERY_OPTIONS, "first_line_regimen": BOARD_FIRST_LINE_OPTIONS}
#: label-v1 value -> code, per board question (build_data.py boards-select).
BOARD_ANSWER_CODES = {k: {v: code for _l, code, _t, v in opts} for k, opts in BOARD_OPTION_TABLES.items()}


# Treatment climb (spec planning/vista_eval_vlm/2026-11-23-tb-board-treatment-climb.md): yes/no, asked of every
# board: does the tumor board's plan include it. (key, label-v1 question in board_modalities_v1.parquet, text).
MODALITY_QUESTIONS = (
    ("plan_surgery", "Will the tumor board's plan for this patient include surgery?",
     "Yes: the plan includes an operation on the cancer (any resection).", "No: it does not."),
    ("plan_radiation", "Will the tumor board's plan for this patient include radiation?",
     "Yes: the plan includes radiation of any kind (SBRT, definitive, consolidative, postoperative, palliative, brain).",
     "No: it does not."),
    ("plan_systemic", "Will the tumor board's plan for this patient include systemic therapy?",
     "Yes: the plan includes systemic therapy of any kind, new or continued.", "No: it does not."),
    ("plan_chemo", "Will the tumor board's plan for this patient include chemotherapy?",
     "Yes: the plan includes chemotherapy (platinum-based or other), new or continued.", "No: it does not."),
    ("plan_immunotherapy", "Will the tumor board's plan for this patient include immunotherapy?",
     "Yes: the plan includes a PD-1, PD-L1 or CTLA-4 drug, new or continued.", "No: it does not."),
    ("plan_targeted", "Will the tumor board's plan for this patient include targeted therapy?",
     "Yes: the plan includes a drug aimed at a driver alteration (EGFR, ALK, ROS1, KRAS, MET, RET and others), new or "
     "continued.", "No: it does not."),
)
#: board_modalities_v1.parquet answer -> code (yes/no questions: 1 yes, 0 no).
MODALITY_ANSWER_CODES = {k: {"yes": 1, "no": 0} for k, *_ in MODALITY_QUESTIONS}
