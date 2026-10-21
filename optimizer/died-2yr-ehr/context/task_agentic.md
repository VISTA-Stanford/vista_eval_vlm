# The task: will this patient die within 2 years, judged from their medical record

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **the patient's medical record up to 7 days before their first cancer treatment**: every
structured EHR table, every clinical note, and the cancer registry's diagnosis-time fields for this cancer.
From that record it gives the probability that the patient dies, of any cause, within 2 years of the cancer
diagnosis. Nothing dated after the cutoff is in the record.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the record, writes a short findings note, and
answers one JSON object with `probability` (a number from 0 to 1) and a one-line `reason`. The wrapper that tells
it where things are and asks for that output is fixed; you write only what it follows. The wrapper already
states the question, that the record stops 7 days before first treatment, the folder layout, and the death rate
below.

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

Every line a session reads costs tokens, and each session has a cap of $0.40 and a 15-minute limit. A session
that runs out writes no answer and is scored as 0.5. Ask for what is worth reading, not everything: a tool that
pulls the structured facts into one short summary is cheaper than the session reading tables by hand.

## The answer

**Died within 2 years** (`gold_died` = 1): the cancer registry records the patient as dead, of any cause,
within 24 months of the cancer **diagnosis date**. The cutoff is near the start of treatment, usually weeks to
a few months after diagnosis, so the window left after the cutoff is usually a little under 2 years. Deaths
from causes other than cancer count.

**Lived** (`gold_died` = 0): the registry has the patient alive with follow-up past 2 years. Patients without
2 years of follow-up and no recorded death were removed before you see anything. Of the training patients,
30.6% died.

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

**AUROC** over the patients of a pass: the chance that a patient who died gets a higher probability than a
patient who lived (ties count half). Only the **order** matters: 0.5 is chance, 1.0 a perfect ranking. So the
program gains by spreading patients out, using the whole range, and avoiding many patients at the same value.
The score that counts is AUROC on the validation patients (VAL, 200 patients), which you never see.

The probabilities should also mean what they say: of patients given about 0.4, about 40% should die. That is
not the score, but it is reported every iteration (Brier score, and deaths per probability decile in the
TRAIN breakdown), and a calibrated program is the one worth keeping.

The mistakes file marks a patient as a mistake when its probability is on the wrong side of 0.306 (the death
rate): that is only a way of choosing whom to study, not the score.

## Reference scores on this task

Linear models trained on about 3,900 labelled patients, scored on the same 200 VAL patients you are scored on
(vista_bench v1.7):

- structured clinical facts at the cutoff (stage, age, labs and so on): AUROC **0.80**;
- foundation-model embeddings of the whole coded EHR (MOTOR): **0.78-0.80**;
- radiology-report text embeddings, on the half of patients who have a report: **0.81-0.86**.

Your program learns from far fewer patients than these models did, so matching 0.80 would be a strong
result; above it would be remarkable. A VAL score above 0.95 would more likely mean something leaked than a
breakthrough.

## Reading the literature

You have WebSearch: use it to find what predicts 2-year death in thoracic cancer, how strong each factor is,
and the death rates to calibrate against (by stage, histology, performance status). It returns search results
and their summaries, not whole papers (page fetching is off). Search for findings and topics, never for
anything about a patient: no ids, no values copied from a record, no note text. Record in your notebook what
you found and where, so later iterations don't search for it again.

## What is noise

Some deaths are unrelated to the cancer and not foreshadowed in the record. Some patients with advanced
disease live past 2 years on modern treatment. Records vary: a patient referred in shortly before treatment
may have only a few notes here. Registry death dates can lag. Expect a ceiling well below 1.0.
