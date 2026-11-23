You are given this patient's electronic health record, cut off one week before a thoracic tumor board encounter.
This is Stanford's thoracic tumor board. Your job is to predict what this tumor board decided, as probabilities that rank
well across many patients: for each letter, patients whose tumor board chose it should get more on it than patients whose
tumor board chose something else.

## How to read the record (aim for 6 to 12 tool calls, well inside the time and budget)

1. Run the notes reader and read ALL of its output (do not cut it with `head`; it is about 15-18 KB):
   `python3 -B <program>/tools/digest.py <record folder>`. It prints, in order:
   - a FACT SHEET: the tumor board year; the registry's cancer **at first diagnosis** (site, histology, clinical stage,
     EGFR/ALK/PD-L1 codes; this is not the current stage if the cancer has since recurred or progressed); the
     registry's treatment events; the cancer drugs in `drug_exposure` with first and last dates (a drug whose last
     date is within a few weeks of the cutoff is probably still being given);
   - every passage of the last year that mentions a tumor board or presenting the case: earlier tumor boards'
     recommendations and, most useful of all, "will present at tumor board to decide X" (the question this tumor board was
     asked);
   - a one-line list of every note of the last 30 days: biopsies, orders, simulations, refills, admissions, hospice;
   - the latest clinicians' assessment-and-plan text, imaging impressions and pathology.
2. Then answer three things for yourself, in this order, and check in the record (grep `notes.txt`, query the tables)
   whatever the digest leaves unclear:
   - **Setting.** (a) no tissue diagnosis yet / suspected cancer; (b) newly diagnosed and untreated; (c) on active
     treatment; (d) treatment finished, under observation; (e) new recurrence or progression. Which cancer (lung
     NSCLC, small cell, mesothelioma, thymic, esophageal, other or unclear)?
   - **The question this tumor board was asked.** Read the newest oncology / surgery / radiation oncology plan and the
     tumor board passages. Is something pending (biopsy, PET, brain MRI, molecular result, PFTs)? Is a new or growing
     lesion being questioned? Has a plan already been agreed with the patient (a referral placed, a start date, a
     simulation, an operation booked)? A plan already written down before the cutoff is usually what the tumor board
     endorses.
   - **What standard was then.** The year is in the fact sheet (see "The year" below).
3. Write the findings note and the answer. Every letter gets a number; set them from what you read, using the
   tendencies below. Do not give every patient the same spread.

If the record has no notes (some older records hold only registry rows or a few tables), use the registry (stage,
histology, events, date of diagnosis vs cutoff) and the base rates below; a diagnosis made in the weeks before the
cutoff with no treatment event usually means a first tumor board deciding the first treatment.

## How this tumor board decided: decision_type

Shares in a TRAIN batch of 300 encounters: start a new treatment (B) 95, surveillance (F) 52, continue (C) 48, more
workup (A) 44, switch/stop (D) 22, add local therapy (E) 20, refer (G) 17, supportive care/hospice (H) 1, defer (I) 1.
In the full TRAIN set: A 21%, B 31%, C 15%, D 7%, E 6%, F 14%, G 5%, H 1%, I under 1%.

**Defer (I) is almost never the answer** (1 in 300). Earlier sessions put about 10% on I for everyone, which wastes
probability. Give I only a token amount unless the record says the patient is still deciding between offered options
or the team explicitly could not agree. Likewise H (hospice / best supportive care) is rare: raise it only when the
record shows hospice discussions, very poor performance status (ECOG 3-4, bed-bound) or the patient declining further
cancer treatment.

**Newly diagnosed, untreated (setting b): start treatment (B) beats more workup (A).** Earlier sessions put "more
workup" on top for 64 patients, but the tumor board's answer was B for 25 of them and workup for only 17. When there is a
tissue diagnosis and basic staging (CT and usually PET), this tumor board usually decides the first treatment even when a
molecular result, PFTs, a brain MRI or a dental/cardiac clearance is still pending: it says "surgery", "chemoradiation"
or "systemic therapy" and lets the pending test follow. This held across lung, esophageal, thymic and mesothelioma
cases. Keep A as the leading letter when: there is **no tissue diagnosis** yet or the biopsy was non-diagnostic or
atypical; PET or imaging shows a **new lesion that would change the stage** (a contralateral nodule, an indeterminate
bone or adrenal lesion, a suspicious node in a patient otherwise early stage) and the plan says it must be biopsied;
two primaries or a primary-vs-metastasis question is open; or staging is plainly incomplete for a stage I-III patient
heading to curative treatment (no PET, nodes not sampled when the PET shows avid nodes). Even then give B a real share.

**On active treatment (setting c): "continue" is right only about a third of the time.** A patient on stable therapy is
brought to tumor board because someone has a question. Earlier sessions put C on top for 81 patients; the tumor board's
answer was C for 26, E 13, A 11, B 10, F 9, D 9. So when the plan is "continue", still spread probability by the
question you find:
- a single site growing slowly, oligoprogression, a residual primary after a good response on targeted therapy or
  immunotherapy, brain metastases, a painful bone lesion: **add local therapy (E)** (radiation/SBRT/SRS or surgery
  while the drug continues) competes with C and A;
- an equivocal finding (growth vs atelectasis, infection, radiation change or pseudo-progression; a new nodule; an
  earlier tumor board asked for a repeat PET): **more workup (A)** (biopsy, PET, short-interval imaging) competes with C;
- clear progression, a new resistance mutation, or toxicity (pneumonitis, hepatitis, rash/DRESS): **switch or stop
  (D)**, or B when a new line starts; a trial is also possible (G);
- long-running oral targeted therapy (osimertinib, alectinib, lorlatinib and similar) or adjuvant therapy with stable
  scans and a "continue and rescan in 3 months" plan: the answer key often reads this as **surveillance (F, intent
  E)** rather than C (at least 5 such patients in one batch). Split C and F here rather than piling everything on C.
- if radiation to a site was already planned or simulated before the cutoff while systemic therapy goes on, the
  tumor board often just "continued" (C) the agreed plan; if the tumor board was asked whether to add it, E. Split C and E.

**Finished treatment / under observation (setting d), or a questioned finding after treatment:** surveillance (F) is
common; with a new suspicious lesion after surgery, SBRT or chemoradiation, the tumor board chose between F (repeat imaging
in 2-3 months), A (biopsy / PET) and B (treat a confirmed recurrence). A finding the radiologist calls indeterminate,
small (under about 1 cm) or likely post-radiation change leaned to F more often than sessions expected; a growing
lesion with a plan to biopsy leaned to A; a biopsy-proven recurrence leaned to B. Spread across F, A and B.

**Refer (G)** was the answer for 17 of 300 and the sessions never put more than a few points on it. It covers
referral for an opinion or a treatment the tumor board itself does not decide: radiation oncology or thoracic surgery
consultation for a borderline-operable or poor-lung-function patient (ILD, oxygen), a clinical trial or trial
screening, a second opinion or transfer of care, palliative care, lung transplant or another specialty (GI, ENT, IR).
Give G a real share when the newest plan says "refer to ...", "consult ... for evaluation", "trial", "second
opinion", or the patient is changing institutions; leave it small otherwise.

## Intent follows the decision

Intent in the same batch, by the tumor board's decision: surveillance (F) went with intent E (not applicable) in 43 of 52;
workup (A) with D (undetermined) in 28 of 44 (curative A in 8, mostly early-stage patients being staged for curative
treatment); start (B) with curative A in 57 and palliative B in 37 of 95 (by stage and setting); continue (C) with B in
38 and A in 10 of 48 (A when the ongoing therapy is induction, neoadjuvant, adjuvant or part of chemoradiation);
switch (D) and add-local (E) with B in nearly all; refer (G) was mixed (A 7, D 4, C 3, B 3).
So build intent from your decision probabilities: the share you put on F should mostly reappear on intent E, the share
on A mostly on D (or on curative A for an early-stage curative workup), and the treatment shares on A or B by stage.
Intent C (symptom control only, 3%) is the tumor board choosing treatment only for symptoms: palliative radiation for pain,
bleeding, airway obstruction or cord compression, a pleural catheter, or comfort-focused care in a frail patient.
When such a symptom is the reason for the visit, put real weight on C, not only on B.

Curative (A) vs palliative (B): stage I-III treated with surgery, SBRT, chemoradiation or perioperative therapy is
curative; stage IV is palliative except oligometastatic disease treated with definitive local therapy to every site,
which this tumor board sometimes calls curative; a recurrence confined to the chest or a node after earlier curative
treatment is often treated with curative intent again.

## early_stage_plan (untreated stage I-III NSCLC)

TRAIN shares: definitive chemoradiation (E) 23%, more workup first (H) 22%, radiation alone (D) 18%, surgery alone (C)
17%, systemic then surgery (A) 11%, another plan (G) 4%, systemic first with local deferred (F) 3%, surgery then
systemic (B) 1%.
- **B is almost never the answer**: when the tumor board recommends an operation it says "surgery" and decides adjuvant
  therapy after the pathology, so a surgery-first plan is C, even for stage II-IIIA.
- **H (more workup first) is common**: staging incomplete (no PET or no brain MRI for stage II-III, PET-avid nodes not
  yet sampled by EBUS or mediastinoscopy), a second nodule or an indeterminate distant lesion on PET, a non-diagnostic
  biopsy, or PFTs/fitness unknown before an operation. In one batch 8 of 28 were H and the sessions picked H for only 3:
  they jumped to chemoradiation or surgery. If the stage itself is still in doubt, H should lead.
- **E (chemoradiation)**: N2 bulky or multistation, N3, T4 unresectable, or stage III in a patient not fit for surgery.
- **D (radiation alone, SBRT)**: stage I-II node-negative in an older or frail patient, poor lung function (ILD,
  oxygen, low FEV1/DLCO), a patient who declines surgery, or no biopsy possible.
- **C (surgery alone)**: stage I-II, node-negative after mediastinal staging, fit (PFTs adequate).
- **A (neoadjuvant systemic, then surgery)**: grew after 2021-2022 (neoadjuvant chemo-immunotherapy, CheckMate 816;
  perioperative regimens from 2023). For a tumor board in 2022 or later, a resectable tumor of 4 cm or more, T3-T4 or N1-N2
  in a fit patient without EGFR/ALK gives A a real share against C and E; before 2021 it was mostly reserved for
  superior sulcus tumors and selected N2.
- **F (systemic or targeted first, local deferred)**: a driver mutation with locally advanced disease, or doubt
  whether the disease is truly stage III.

## The year

The tumor board year is in the fact sheet. Standards changed inside this data: consolidation durvalumab after
chemoradiation from 2018; adjuvant osimertinib for EGFR-mutant resected stage IB-IIIA from 2020-2021; adjuvant
atezolizumab from 2021; neoadjuvant chemo-immunotherapy from 2022; perioperative immunotherapy from 2023; first-line
chemo-immunotherapy for driver-negative stage IV from 2017-2018 (chemotherapy alone before that); osimertinib first line
for EGFR from 2018. Records from 2015 and earlier are often sparse (registry only), and those encounters are mostly
first decisions (start treatment or workup). From 2023 on, surveillance and continue are more common than in earlier
years.

## after_surgery_plan and first_line_regimen (when asked)

after_surgery_plan TRAIN shares: adjuvant/consolidation systemic (A) 33%, observation (D) 33%, both (C) 13%, another
plan (E) 9%, more workup (F) 7%, postoperative radiation (B) 6%. Node-negative stage I after resection: observation;
stage II-IIIA, or IB with a driver after 2020: adjuvant (osimertinib for EGFR; chemotherapy, then immunotherapy after
2021); after chemoradiation from 2018: durvalumab consolidation (A). An elderly or frail patient, or a slow recovery
after surgery, moves weight to D.

first_line_regimen TRAIN shares: more workup (H) 24%, chemotherapy alone (A) 21%, targeted alone (D) 20%, chemo +
immunotherapy (B) 19%, another regimen (G) 13%. EGFR/ALK/ROS1/RET/MET/BRAF-positive: D. Driver status still pending:
H. Driver-negative after 2017: B (or immunotherapy alone if PD-L1 high, rare here). Before 2017, or a frail patient,
or small cell / non-NSCLC histology: A or G.

## Other cancers

Esophageal, thymic, mesothelioma and neuroendocrine cases are discussed here too. A newly diagnosed locally advanced
esophageal cancer usually goes straight to treatment (chemoradiation, with or without surgery afterwards), even with
the PET or EUS still pending: weigh B and curative A. Thymoma is often watched or resected; slow pleural recurrence of
thymoma is often watched (F). Mesothelioma usually goes to systemic therapy (B, palliative) unless clearly early.

## Probabilities

Your numbers are compared across patients letter by letter, so let them follow the record: a clear, written plan
deserves a clear lead on its letter; an open question deserves two or three letters with real shares. Never put a letter at
zero if the record leaves it plausible, and keep some weight on the base-rate letters (B, F, C, A) when the record
says little.

## Which treatments the plan includes (asked of every encounter)

Every encounter is also asked six yes/no questions: does the tumor board's plan include surgery, radiation, systemic
therapy, chemotherapy, immunotherapy, targeted therapy. Answer each with one probability that the plan includes it.
These six and early_stage_plan are what is scored; decision_type and intent are still asked. Build them from the same
reading: the plan you predict for decision_type and the treatment questions should agree. Continuing a drug counts
as including it.

## The questions, word for word

Every encounter is asked decision_type, intent and the six plan_ questions; some are also asked one of
early_stage_plan, after_surgery_plan or first_line_regimen.

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

[early_stage_plan]
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

[after_surgery_plan]
Question: This patient has stage I-III non-small cell lung cancer and has had surgery or chemoradiation. What will the tumor board recommend next?
A. Adjuvant or consolidation systemic therapy
B. Postoperative radiation
C. Both systemic therapy and postoperative radiation
D. Observation
E. Another plan
F. More workup first, before choosing a treatment
Give each letter a probability from 0 to 100. These probabilities must add up to 100.

[first_line_regimen]
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

