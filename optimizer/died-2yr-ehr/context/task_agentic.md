# The task: did this patient die within 1 year, and within 2 years, judged from their medical record

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **the patient's medical record up to 7 days before their first cancer treatment**: every
structured EHR table, every clinical note, and the cancer registry's diagnosis-time fields for this cancer.
From that record it answers two questions, each yes or no: did the patient die, of any cause, within 1 year of the
cancer diagnosis, and within 2 years? Nothing dated after the cutoff is in the record.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the record, writes a short findings note, and
answers one JSON object with `died_within_1_year` and `died_within_2_years`, each exactly `"yes"` or `"no"`, and a
one-line `reason`. The wrapper that tells it where things are and asks for that output is fixed; you write only what
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
- Python 3.11 with pandas, pyarrow and duckdb; grep and ripgrep; Read, Write, Edit, Glob, Grep and Bash. No
  network, no other patient, no internet lookups, no subagents.

Every line a session reads costs tokens, and each session has a cap of $0.50 and a 15-minute limit. A session
that runs out writes no answer and is scored wrong at both horizons, as is any answer that is not exactly yes or no. Ask for what is worth reading, not everything: a tool that
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

**Balanced accuracy** at each horizon, averaged over the two. At one horizon it is the mean of two rates: the share
of patients who died that the program answered yes, and the share of patients who lived that it answered no. 0.5 is
chance (all yes, all no, or guessing), 1.0 every answer right. The two classes count equally however rare deaths
are, so answering no to everyone scores 0.5, not the 70-80% plain accuracy it would get. A yes is worth giving when
the record makes death more likely than the death rate above, not only when death is more likely than not. A failed
session or an answer that is not exactly yes or no is wrong at both horizons. The score that counts is on the
validation patients (VAL, 200 patients, of whom 35 died within 1 year and 63 within 2), which you never see.

The TRAIN breakdown gives, per horizon, the balanced accuracy, how many who died were answered yes and how many who
lived were answered no, and how often a session answered yes at 1 year but no at 2 years, which is impossible. The
mistakes file marks a patient as a mistake when it is wrong at either horizon.

## Reference points on this task

Linear models trained on about 3,900 labelled patients rank these same VAL patients well for 2-year death (vista_bench
v1.7, AUROC about 0.80 from the structured clinical facts at the cutoff, the coded EHR, or radiology-report text).
Their balanced accuracy at a fixed cut-off is computed separately and is not given here. Your program learns from far
fewer patients than they did. A VAL score above about 0.90 would more likely mean something leaked than a
breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a record. Ask what predicts 1- and 2-year death
in thoracic cancer, how strong each factor is, and the death rates by stage, histology and performance status.
Questions must be general: no ids, no values copied from a record, no note text; such a question is refused. Read
the findings file before asking, and record in your notebook what you learned.

## What is noise

Some deaths are unrelated to the cancer and not foreshadowed in the record. Some patients with advanced
disease live past 2 years on modern treatment. Records vary: a patient referred in shortly before treatment
may have only a few notes here. Registry death dates can lag. Expect a ceiling well below 1.0.
