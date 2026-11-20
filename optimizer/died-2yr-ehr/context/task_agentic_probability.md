# The task: how likely is it that this patient died within 1 year, and within 2 years, judged from their medical record

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **the patient's medical record up to 7 days before their first cancer treatment**: every
structured EHR table, every clinical note, and the cancer registry's diagnosis-time fields for this cancer.
From that record it gives two probabilities, each 0 to 100: the chance the patient died, of any cause, within 1 year
of the cancer diagnosis, and within 2 years. Nothing dated after the cutoff is in the record.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the record, writes a short findings note, and
answers one JSON object with `died_within_1_year` and `died_within_2_years`, each a number from 0 to 100 (its
probability in percent), and `reasons`: the 3 to 5 record facts that moved the probabilities most, each with the
record file it was read in and its date. The reasons are not scored; they are kept, checked against the record, and
shown to you beside each TRAIN patient. The wrapper that tells it where things are and asks for that output is fixed; you write only what
it follows. The wrapper already states both questions, that the record stops 7 days before first treatment, the
folder layout, and the death rates below.

What the session has, as a read-only folder `record/`:

- `notes.txt`: every clinical note on or before the cutoff, in date order, each starting with a header line
  `===== NOTE <note id> | <date or datetime> | <note type> =====`. A patient has a median of about 55 notes,
  some many more: grep it, don't read it whole.
- One parquet file per OMOP common-data-model table that has rows for the patient: `person`,
  `observation_period`, `visit_occurrence`, `visit_detail`, `condition_occurrence`, `procedure_occurrence`,
  `drug_exposure`, `drug_era`, `device_exposure`, `episode`, `image_occurrence`, `measurement`, `observation`,
  `note` (the notes again, as a table). Concept ids come with their concept names beside them. Rows dated
  after the cutoff are gone and end dates are clipped to it; the `death` table is never included.
- `registry_diagnoses`, `registry_treatments`, `registry_outcomes`, `registry_events`, `registry_misc_details`
  (`.parquet`): the tumour registry's abstract of this cancer only, limited to fields allowed at the cutoff.
  Kept: diagnosis-time tumour fields (site, histology, clinical TNM, metastases at diagnosis, biomarkers,
  pre-treatment labs), demographics, smoking and body size, and the first-course treatment codes (which
  treatment was given first, without any dates). Removed: pathologic and summary stage, all free text about
  the workup or treatment, every outcome, follow-up and later-line field. A dated field whose date is after
  the cutoff is blank.
- `meta.json`: the patient's opaque id and the cutoff date.
- Beside the record, `record_guide.md`: a fixed map of where each kind of fact lives in the folder (which
  registry field holds clinical stage, metastases at diagnosis, biomarkers and the first-course codes, how
  those codes read, how notes link to imaging, which tables are mostly flowsheet rows, which sources mislead).
  The wrapper tells every session to read it first; you can't edit it, and your instructions can refer to it.
  Its full text is the appendix at the end of this brief.
- Python 3.11 with pandas, pyarrow and duckdb; grep and ripgrep; Read, Write, Edit, Glob, Grep and Bash. No
  network, no other patient, no internet lookups, no subagents.

Every line a session reads costs tokens, and each session has a cap of $0.50 and a 15-minute limit. A session
that runs out writes no answer and is scored at 50 at both horizons (the middle of the ranking), as is any answer
that is not a number from 0 to 100. Ask for what is worth reading, not everything: a tool that
pulls the structured facts into one short summary is cheaper than the session reading tables by hand.

## The answer

**Died within 2 years** (`gold_died_within_2_years` = 1): the cancer registry records the patient as dead, of any
cause, within 24 months of the cancer **diagnosis date**. **Died within 1 year** (`gold_died_within_1_year` = 1):
the same, within 12 months. The cutoff is near the start of treatment, usually weeks to a few months after
diagnosis, so the window left after the cutoff is usually a little shorter than the horizon. Deaths from causes
other than cancer count.

**Lived** (gold 0): the registry has the patient alive at that horizon. Patients without 2 years of follow-up and no
recorded death were removed before you see anything, so every patient has both answers, and a patient who died
within 1 year also died within 2. Of the training patients, 18.5% died within 1 year and 30.6% within 2 years.

What is known to predict death in this population, as hypotheses for where to look (gold is the objective,
not this list):

- **How far the cancer has spread at diagnosis.** Clinical stage, nodal and distant metastases (brain, bone,
  liver, adrenal, pleural or pericardial effusion), histology (small-cell versus non-small-cell).
- **The patient.** Age, performance status (ECOG or Karnofsky in the notes), weight loss, comorbidities
  (heart failure, COPD, interstitial lung disease, kidney disease, a second cancer), smoking.
- **Biology and labs.** Targetable mutations (EGFR, ALK) and PD-L1; albumin, haemoglobin, LDH, sodium,
  white count and neutrophil-lymphocyte ratio near the cutoff.
- **The plan.** The first-course treatment codes say what the first treatment is: curative surgery versus
  systemic therapy versus palliative radiation or palliative care are strongly prognostic. Hospice or
  goals-of-care discussions in the notes before the cutoff.

## How it is scored

**AUROC** at each horizon, averaged over the two. At one horizon it is the chance that a patient who died was given a
higher probability than a patient who lived, over every such pair; equal probabilities count as half. 0.5 is chance
(the same number for everyone, or guessing), 1.0 every patient who died ranked above every one who lived. Only the
order of the probabilities counts, not their level, so the score rewards telling patients apart: a program that
gives most patients the same few round numbers throws away ranking it could have had. A failed session or an
unreadable answer counts as 50 at both horizons. The score that counts is on the validation patients (VAL, 200
patients, of whom 35 died within 1 year and 63 within 2), which you never see.

The TRAIN breakdown gives, per horizon, the AUROC; the Brier score (the mean squared gap between the probability
and the outcome, 0 to 1, lower is better: it rewards honest levels as well as ranking); the mean probability given
to those who died and those who lived; a calibration table by tenth of probability (how many patients fell in each
tenth and how many of them died); the number of distinct probabilities given and the share on the most common one;
and how often a session gave a higher chance of death by 1 year than by 2, which is impossible. Calibration does not
change the score, but a program whose probabilities track the death rate in each tenth is one whose numbers mean
something. The mistakes file marks a patient as a mistake at a horizon when its probability is on the wrong side
of that horizon's death rate (at or above 18.5% at 1 year or 30.6% at 2 years for a patient who lived, below it for
one who died).

## Reference points on this task

Models trained on about 3,900 labelled patients rank these same VAL patients well for 2-year death (vista_bench
v1.7, AUROC about 0.80 from the structured clinical facts at the cutoff, the coded EHR, or radiology-report text).
Your program learns from far fewer patients than they did. A VAL score above about 0.90 would more likely mean something leaked than a
breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a record. Ask what predicts 1- and 2-year death
in thoracic cancer, how strong each factor is, and the death rates by stage, histology and performance status.
Questions must be general: no ids, no values copied from a record, no note text; such a question is refused. Read
the findings file before asking, and record in your notebook what you learned. Use it: there is no practical
limit on questions, ask as many as help. An answer takes a few minutes, so send your questions early in the round
and work on the mistakes while they are answered.

## What is noise

Some deaths are unrelated to the cancer and not foreshadowed in the record. Some patients with advanced
disease live past 2 years on modern treatment. Records vary: a patient referred in shortly before treatment
may have only a few notes here. Registry death dates can lag. Expect a ceiling well below 1.0.
