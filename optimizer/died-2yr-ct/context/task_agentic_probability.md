# The task: how likely is it that this patient died within 1 year, and within 2 years, judged from one chest CT

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **one CT scan**: the latest chest CT taken before the patient's first cancer treatment
(typically about three weeks before it; occasionally months). From that scan alone it gives two probabilities,
each 0 to 100: the chance the patient died, of any cause, within 1 year of the cancer diagnosis, and within 2 years.
Nothing else about the patient is available: no age,
sex, stage, notes, reports or labs. Whatever the answer depends on has to be seen in the images.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code
session (Claude Sonnet, which can look at images) reads `instructions.md`, examines the CT, writes a short
findings note, and answers one JSON object with `died_within_1_year` and `died_within_2_years`, each a number from 0 to
100 (its probability in percent), and `reasons`: the 3 to 5 things in the scan that moved the probabilities most,
each with where in the scan it was seen. The reasons are not scored; they are shown to you beside each TRAIN
patient. The wrapper that tells it where things are, states the two questions and asks for that output is fixed;
you write only what it follows. The wrapper does not state the death rates below: if the sessions should know
them, your instructions must say so.

What the session has:

- the CT as a folder: `ct.npz` (one array, key `hu`: Hounsfield units, int16, shape (x, y, z)) and
  `meta.json` (shape, voxel spacing in mm, orientation). Orientation is RAS: the index grows toward the
  patient's Right on axis 0, Anterior on axis 1, Superior on axis 2.
- `tools/view_ct.py`, which renders axial, coronal or sagittal slices in lung, soft-tissue, bone or liver
  windows (or any HU range) as one PNG grid, with optional crops; the session opens the PNG with its Read
  tool. You may change it, replace it, or add tools (for example, measurements).
- Python 3.11 with numpy, scipy, Pillow and nibabel; Read, Write, Edit, Glob, Grep and Bash. No network,
  no other patient, no internet lookups, no subagents.

Every image a session opens costs tokens, and each session has a cap of $0.50 and a 15-minute limit (a
session that reads a dozen image grids and runs a few tools costs about $0.10-0.15 and takes 2-3 minutes). A
session that runs out writes no answer and is scored at 50 at both horizons (the middle of the ranking), as is any
answer that is not a number from 0 to 100. Ask for what is worth looking at, not everything.

## The answer

**Died within 2 years** (`gold_died_within_2_years` = 1): the cancer registry records the patient as dead, of
any cause, within 24 months of the cancer **diagnosis date**. **Died within 1 year** (`gold_died_within_1_year`
= 1): the same, within 12 months. The scan is near the start of treatment, which is usually weeks to a few months
after diagnosis, so the window left after the scan is usually a little shorter than the horizon. Deaths from
causes other than cancer count.

**Lived** (gold 0): the registry has the patient alive at that horizon. Patients without 2 years of follow-up and
no recorded death were removed before you see anything, so every patient here has both answers, and a patient
who died within 1 year also died within 2. Of the training patients, 18.5% died within 1 year and 30.6% within
2 years.

What is known to predict death in this population, as hypotheses for where to look (gold is the
objective, not this list):

- **How far the cancer has spread.** Primary tumour size and invasion (chest wall, mediastinum, vessels),
  enlarged mediastinal or supraclavicular nodes, nodules in the other lung, pleural or pericardial
  effusion or pleural nodularity, and distant metastases visible in the field of view (liver, adrenals,
  bone lesions, sometimes brain on a scan that includes it). Distant spread is the strongest single signal.
- **The patient's reserve.** Muscle wasting (sarcopenia) and loss of fat, emphysema, fibrotic or
  interstitial lung disease, cardiomegaly, heavy coronary or aortic calcification, signs of heart failure.
- **Other clues.** Collapse or obstruction of a lung or lobe, cavitation, a central airway or
  superior vena cava compressed by tumour, a scan that looks like a staging CT for widespread disease.

Scans vary: chest-only versus chest, abdomen and pelvis; with or without contrast; slices from about 0.6 mm
to 5 mm (thin scans arrive averaged into slabs of about 2.5 mm). The field of view decides what can be
seen: a chest-only scan shows only the top of the liver and adrenals.

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

The run stops by itself after two rounds in a row that set no new best VAL score, after 10 rounds, or when spending
passes $50 in a round or $200 in all. TRAIN grows from 100 patients in round 1 to 400 by round 10.

## Reference points on this task

- CT image models trained on about 3,500 labelled patients' scans (a linear model on frozen embeddings) rank
  2-year death best with MerlinOnc, **AUROC 0.76** on held-out patients. A model given every radiology report
  reaches about 0.85; structured clinical facts (stage, age and so on) about 0.78.
- Small open vision-language models asked the yes/no question from CT images scored **0.52 to 0.61**.
- On this task's 200 VAL patients the same CT probes score higher, about **0.81 to 0.82** at both horizons (clinical facts 0.75 at 1 year, 0.82 at 2 years); the 0.76 above is on held-out test patients.
- Your program learns from far fewer patients than the image models did, so matching the probes on VAL would be a strong result.
  A VAL score above about 0.85 would more likely mean something leaked than a breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a scan. Ask what predicts 1- and 2-year death from
a staging chest CT, how strong each finding is, and the death rates by finding. Questions must be general: no ids,
no measurements copied from a scan, no findings notes; such a question is refused. Read the findings file before
asking, and record in your notebook what you learned.

## What is noise

Some deaths are unrelated to the cancer and invisible on any CT. Some patients with advanced disease live
past 2 years on modern treatment. Registry death dates can lag. Expect a ceiling well below 1.0.
