# The task: will this patient die within 2 years, judged from one chest CT

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center. The
program is shown **one CT scan**: the latest chest CT taken before the patient's first cancer treatment
(typically about three weeks before it; occasionally months). From that scan alone it gives the probability
that the patient dies, of any cause, within 2 years. Nothing else about the patient is available: no age,
sex, stage, notes, reports or labs. Whatever the answer depends on has to be seen in the images.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code
session (Claude Sonnet, which can look at images) reads `instructions.md`, examines the CT, writes a short
findings note, and answers one JSON object with `probability` (a number from 0 to 1) and a one-line
`reason`. The wrapper that tells it where things are and asks for that output is fixed; you write only what
it follows.

What the session has:

- the CT as a folder: `ct.npz` (one array, key `hu`: Hounsfield units, int16, shape (x, y, z)) and
  `meta.json` (shape, voxel spacing in mm, orientation). Orientation is RAS: the index grows toward the
  patient's Right on axis 0, Anterior on axis 1, Superior on axis 2.
- `tools/view_ct.py`, which renders axial, coronal or sagittal slices in lung, soft-tissue, bone or liver
  windows (or any HU range) as one PNG grid, with optional crops; the session opens the PNG with its Read
  tool. You may change it, replace it, or add tools (for example, measurements).
- Python 3.11 with numpy, scipy, Pillow and nibabel; Read, Write, Edit, Glob, Grep and Bash. No network,
  no other patient, no internet lookups, no subagents.

Every image a session opens costs tokens, and each session has a cap of $0.40 and a 15-minute limit (a
session that reads a dozen image grids and runs a few tools costs about $0.10-0.15 and takes 2-3 minutes). A
session that runs out writes no answer and is scored as 0.5. Ask for what is worth looking at, not
everything.

## The answer

**Died within 2 years** (`gold_died` = 1): the cancer registry records the patient as dead, of any cause,
within 24 months of the cancer **diagnosis date**. The scan is near the start of treatment, which is usually
weeks to a few months after diagnosis, so the window left after the scan is usually a little under 2 years.
Deaths from causes other than cancer count.

**Lived** (`gold_died` = 0): the registry has the patient alive with follow-up past 2 years. Patients
without 2 years of follow-up and no recorded death were removed before you see anything, so every patient
here has a known answer. About 31% died.

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

**AUROC** over the patients of a pass: the chance that a patient who died gets a higher probability than a
patient who lived (ties count half). Only the **order** of the probabilities matters, not their calibration:
0.5 is chance, 1.0 is a perfect ranking. So the program gains by spreading patients out, using the whole
range, and avoiding many patients at the same value (ties waste information). The score that counts is
AUROC on the validation patients (VAL, 200 patients), which you never see.

The probabilities should also mean what they say: of patients given about 0.4, about 40% should die.
That is not the score, but it is reported every iteration (Brier score, and deaths per probability decile in
the TRAIN breakdown), and a calibrated program is the one worth keeping. Published death rates (by stage,
by finding) are the natural anchor for calibration.

The mistakes file marks a patient as a mistake when its probability is on the wrong side of 0.31 (the death
rate): that is only a way of choosing whom to study, not the score.

## Reference scores on this task

- CT image models trained on about 3,500 labelled patients' scans (a linear model on frozen embeddings):
  the best, MerlinOnc, reaches **AUROC 0.76** on held-out patients. A model given every radiology report
  reaches 0.85; structured clinical facts (stage, age and so on) about 0.78.
- Small open vision-language models asked the yes/no question from CT images scored **0.52 to 0.61**.
- Your program learns from far fewer patients than the image models did, so matching 0.76 would be a strong
  result; above it would be remarkable.

## Reading the literature

You have WebSearch: use it to find what predicts death from a staging chest CT, how strong each finding is,
and the death rates to calibrate against. It returns search results and their summaries, not whole papers
(page fetching is off). Search for findings and topics, never for anything about a patient: no ids, no
measurements copied from a scan, no findings notes. Record in your notebook what you found and where, so
later iterations don't search for it again.

## What is noise

Some deaths are unrelated to the cancer and invisible on any CT. Some patients with advanced disease live
past 2 years on modern treatment. Registry death dates can lag. Expect a ceiling well below 1.0.
