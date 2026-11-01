# The task: this patient's survival curve, judged from their medical record

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **the patient's medical record up to 7 days before their first cancer treatment**: every
structured EHR table, every clinical note, and the cancer registry's diagnosis-time fields for this cancer.
From that record it gives a survival curve: the chance the patient is still alive, of any cause, at each point in the
years after the record's last day. Nothing dated after the cutoff is in the record.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the record, writes a short findings note, and
answers one JSON object with `survival_curve`: the patient's chance of being alive at each point after the record's
last day, as points `[months, chance 0 to 1]`, months from 0 and strictly increasing, the chance never rising, to at
least month 120, as many points as it likes up to 2,000 (a session may compute them with code); and `reasons`: the 3
to 5 record facts that shaped the curve most, each with the record file it was read in and its date. The reasons are not scored; they are kept, checked against the record, and
shown to you beside each TRAIN patient. The wrapper that tells it where things are and asks for that output is fixed; you write only what
it follows. The wrapper already states the question, that the record stops 7 days before first treatment, the
folder layout, the curve's format, and the cohort's average survival below.

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
that runs out writes no answer, and a curve that rises, leaves 0 to 1, or stops before about month 75 is unreadable:
either one **stops the whole run** so a person can look (there is no fallback score), so a program that ever fails to
answer costs far more than one that answers plainly. Ask for what is worth reading, not everything: a tool that
pulls the structured facts into one short summary is cheaper than the session reading tables by hand.

## The answer

The gold is each patient's overall survival from the benchmark (vista_bench v1.7), counted from **the record's last
day** (first treatment minus 7 days): either the day the patient died, of any cause, or the last day the patient
was known alive (censored: alive then, unknown after). In the mistakes file, `gold_died` is 1 for a death and 0 for a
censored patient, and `gold_months_followed` is the months from the record's last day to that death or last
contact. Deaths from causes other than cancer count.

Of the training patients, counted from the record's last day, 80.2% were alive at 1 year, 68.5% at 2 years, 61.2%
at 3, 51.2% at 5 and 32.8% at 10 (the Kaplan-Meier curve of 3,836 TRAIN patients). That curve given to everyone is
the **average curve**: the answer with no information about the patient. Follow-up is long: a median of about 37
months, some patients over 15 years.

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

The shape matters as well as the level: some patients carry most of their risk in the first months (advanced disease,
frailty), others a slow steady risk (early-stage disease cured or not), and a curve can say which.

## How it is scored

Exactly as the benchmark scores its survival models, against those deaths and censorings.

- **Integrated Brier score: the score that counts, lower is better.** At each month on a monthly grid, the squared
  gap between the curve's chance of being alive and what happened (1 alive, 0 dead), averaged over patients, with
  censored patients counted only while they were followed and the rest re-weighted for that (inverse-probability-of-
  censoring weights, as scikit-survival computes them). Averaged over the months from about month 1 to about month
  74 (the 90th percentile of TRAIN death times: past that too few patients are followed to trust). It rewards both
  ordering patients and honest levels: a curve that is confidently wrong costs more than the average curve. The
  average curve scores about 0.21 on these patients; the benchmark's survival models about 0.15 to 0.18.
- **C-index**, reported: the share of comparable pairs (the earlier time a death) where the patient who died first
  had the lower curve (ranked by the area under the curve over the scoring months), ties half. 0.5 is chance.
- **Brier and time-dependent AUC at 12, 24, 36 and 60 months**, reported.

A session that fails or a curve that is unreadable stops the run (see the contract). The score that counts is on the
validation patients (VAL, 198 patients), which you never see.

The TRAIN breakdown gives the integrated Brier and the same numbers for the average curve on the same patients (beat
that first), the C-index, Brier and AUC at 12, 24, 36 and 60 months, the number of deaths and censored patients, the
number of curves within 0.05 of the average curve at every scoring month (a program whose curves all hug the average
has learned nothing about the patient), and the number of distinct curves. The mistakes file marks a patient as a
mistake when its curve did worse on that patient than the average curve would have (its own censoring-weighted
Brier over the scoring months), and shows each patient's curve at 12, 24, 36 and 60 months.

## Reference points on this task

Models trained on about 3,800 labelled patients score these patients at an integrated Brier of about 0.15 to 0.18
and a C-index of about 0.71 to 0.79 (vista_bench v1.7 survival models on the coded EHR, radiology-report text and the
structured clinical facts). Your program learns from far fewer patients than they did. A VAL integrated Brier far
below about 0.13 would more likely mean something leaked than a breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a record. Ask what predicts survival in thoracic
cancer, how strong each factor is, and the survival curves by stage, histology, treatment and performance status
(median survival, 1-, 2- and 5-year survival). Questions must be general: no ids, no values copied from a record, no
note text; such a question is refused. Read the findings file before asking, and record in your notebook what you
learned. Use it: there is no practical limit on questions, ask as many as help. An answer takes a few minutes, so
send your questions early in the round and work on the mistakes while they are answered.

## What is noise

Some deaths are unrelated to the cancer and not foreshadowed in the record. Some patients with advanced
disease live for years on modern treatment. Records vary: a patient referred in shortly before treatment
may have only a few notes here. Registry death dates can lag, and a censored patient may have died elsewhere unseen.
Expect a floor well above 0.
