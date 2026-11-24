You are given this patient's electronic health record, cut off one week before a thoracic tumor board encounter.

Read the record (the folder the prompt names) and its guide. The notes reader `tools/digest.py` prints the clinicians'
recent assessment-and-plan text, the latest CT/PET/MRI impressions and pathology diagnoses: run
`python3 -B <program>/tools/digest.py <record folder>` and read its output. Then answer the questions you are asked.
Every tumor board encounter is asked decision_type, intent and six yes/no questions on which treatments the tumor
board's plan includes; some are also asked one of early_stage_plan, after_surgery_plan or first_line_regimen. The
questions, word for word:

[decision_type]
Question: What will the tumor board decide for this patient at this encounter?
A. More workup before deciding (biopsy, staging, imaging, molecular testing, fitness testing or a consult)
B. Start a new treatment
C. Continue the current treatment
D. Switch or stop the current treatment
E. Add local therapy (surgery or radiation) to ongoing systemic therapy
F. Surveillance or observation
G. Refer (a clinical trial, another specialty or a second opinion)
H. Best supportive care or hospice
I. Defer the decision (no consensus, or the patient's decision is pending)
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[intent]
Question: What will the intent of the tumor board's plan be?
A. Curative
B. Palliative, to prolong life
C. Symptom control only
D. Undetermined until more workup is done
E. Not applicable: no treatment is being decided
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[early_stage_plan] (asked only for the patients it names)
Question: This patient has untreated stage I-III non-small cell lung cancer. What will the tumor board recommend?
A. Systemic therapy followed by surgery, with or without adjuvant therapy
B. Surgery followed by systemic therapy
C. Surgery alone
D. Radiation alone
E. Definitive chemoradiation, with or without adjuvant systemic therapy
F. Systemic or targeted therapy first, with local treatment deferred
G. Another treatment plan
H. More workup first, before choosing a treatment
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[after_surgery_plan] (asked only for the patients it names)
Question: This patient has stage I-III non-small cell lung cancer and has had surgery or chemoradiation. What will the tumor board recommend next?
A. Adjuvant or consolidation systemic therapy
B. Postoperative radiation
C. Both systemic therapy and postoperative radiation
D. Observation
E. Another plan
F. More workup first, before choosing a treatment
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[first_line_regimen] (asked only for the patients it names)
Question: This patient has untreated stage IV or recurrent non-small cell lung cancer. Which first-line plan will the tumor board recommend?
A. Chemotherapy alone
B. Chemotherapy with immunotherapy
C. Chemotherapy with targeted therapy
D. Targeted therapy alone
E. Immunotherapy alone
F. Targeted therapy combined with an antibody (for example amivantamab with lazertinib)
G. Another regimen
H. More workup first, before choosing a treatment
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[plan_surgery]
Question: Will the tumor board's plan for this patient include surgery?
- Yes: the plan includes an operation on the cancer (any resection).
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

[plan_radiation]
Question: Will the tumor board's plan for this patient include radiation?
- Yes: the plan includes radiation of any kind (SBRT, definitive, consolidative, postoperative, palliative, brain).
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

[plan_systemic]
Question: Will the tumor board's plan for this patient include systemic therapy?
- Yes: the plan includes systemic therapy of any kind, new or continued.
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

[plan_chemo]
Question: Will the tumor board's plan for this patient include chemotherapy?
- Yes: the plan includes chemotherapy (platinum-based or other), new or continued.
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

[plan_immunotherapy]
Question: Will the tumor board's plan for this patient include immunotherapy?
- Yes: the plan includes a PD-1, PD-L1 or CTLA-4 drug, new or continued.
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

[plan_targeted]
Question: Will the tumor board's plan for this patient include targeted therapy?
- Yes: the plan includes a drug aimed at a driver alteration (EGFR, ALK, ROS1, KRAS, MET, RET and others), new or continued.
- No: it does not.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

For each question asked, give every letter a probability from 0 to 100 (each yes/no question one probability that
the answer is yes), with the few facts from the record that moved them most.
