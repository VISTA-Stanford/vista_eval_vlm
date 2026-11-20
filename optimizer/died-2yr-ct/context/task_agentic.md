# The task: did this patient die within 1 year, and within 2 years, judged from one chest CT

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **one CT scan**: the latest chest CT taken before the patient's first cancer treatment
(typically about three weeks before it; occasionally months). From that scan alone it answers two questions,
each yes or no: did the patient die, of any cause, within 1 year of the cancer diagnosis, and within 2 years? Nothing else about the patient is available: no age,
sex, stage, notes, reports or labs. Whatever the answer depends on has to be seen in the images.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code
session (Claude Sonnet, which can look at images) reads `instructions.md`, examines the CT, writes a short
findings note, and answers one JSON object with `died_within_1_year` and `died_within_2_years`, each exactly
`"yes"` or `"no"`, and a one-line `reason`. The wrapper that tells it where things are, states the two questions
and asks for that output is fixed; you write only what it follows.

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
session that runs out writes no answer and is scored wrong at both horizons, as is any answer that is not
exactly yes or no. Ask for what is worth looking at, not
everything.

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

**Balanced accuracy** at each horizon, averaged over the two. At one horizon it is the mean of two rates: the share
of patients who died that the program answered yes, and the share of patients who lived that it answered no. 0.5 is
chance (all yes, all no, or guessing), 1.0 every answer right. The two classes count equally however rare deaths
are, so answering no to everyone scores 0.5, not the 70-80% plain accuracy it would get. A yes is worth giving when
the scan makes death more likely than the death rate above, not only when death is more likely than not. A failed
session or an answer that is not exactly yes or no is wrong at both horizons. The score that counts is on the
validation patients (VAL, 200 patients, of whom 35 died within 1 year and 63 within 2), which you never see.

The TRAIN breakdown gives, per horizon, the balanced accuracy, how many who died were answered yes and how many who
lived were answered no, and how often a session answered yes at 1 year but no at 2 years, which is impossible. The
mistakes file marks a patient as a mistake when it is wrong at either horizon.

## Reference points on this task

- CT image models trained on about 3,500 labelled patients' scans (a linear model on frozen embeddings) rank
  2-year death best with MerlinOnc, **AUROC 0.76** on held-out patients; their balanced accuracy at a fixed cut-off
  is computed separately and is not given here.
- Your program learns from far fewer patients than the image models did. A VAL score above about 0.85 would more
  likely mean something leaked than a breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a scan. Ask what predicts 1- and 2-year death from
a staging chest CT, how strong each finding is, and the death rates by finding. Questions must be general: no ids,
no measurements copied from a scan, no findings notes; such a question is refused. Read the findings file before
asking, and record in your notebook what you learned.

## What is noise

Some deaths are unrelated to the cancer and invisible on any CT. Some patients with advanced disease live
past 2 years on modern treatment. Registry death dates can lag. Expect a ceiling well below 1.0.
