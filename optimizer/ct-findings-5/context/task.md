# The task: five findings questions about one CT scan, each answered as a probability

Each scan is a CT of an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center, taken at
any point in their care (staging, follow-up, restaging). The program is shown **one CT scan** and nothing else: no
report, no age, no history. For that scan it gives five probabilities, each 0 to 100, that the answer is yes:

| key | question |
|---|---|
| `lesion_liver` | Is there a non-benign finding in the liver? |
| `lesion_lungs` | Is there a non-benign finding in the lungs? |
| `lesion_lymph_nodes` | Is there a non-benign finding in the lymph nodes? |
| `lung_lobe_rul` | Is there a malignant or possibly malignant lung lesion in the right upper lobe? (The right apex counts as the right upper lobe.) |
| `pneumonitis` | Does this scan show non-infectious pneumonitis? |

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each scan a separate Claude Code session
(Claude Sonnet, which can look at images) reads `instructions.md`, examines the CT, writes a short findings note,
and answers one JSON object with the five keys above, each a number from 0 to 100, and `reasons`: the 3 to 5 things
in the scan that moved the probabilities most, each with where in the scan it was seen. The reasons are not scored;
they are shown to you beside each TRAIN scan. The wrapper that tells it where things are, lists the five questions
and asks for that output is fixed; you write only what it follows. The wrapper does not state how the answers were
made or how common each is: if the sessions should know, your instructions must say so.

What the session has:

- the CT as a folder: `ct.npz` (one array, key `hu`: Hounsfield units, int16, shape (x, y, z)) and
  `meta.json` (shape, voxel spacing in mm, orientation). Orientation is RAS: the index grows toward the
  patient's Right on axis 0, Anterior on axis 1, Superior on axis 2.
- `tools/view_ct.py`, which renders axial, coronal or sagittal slices in lung, soft-tissue, bone or liver
  windows (or any HU range) as one PNG grid, with optional crops; the session opens the PNG with its Read
  tool. You may change its views (windows, crops, which slices). Adding other tools is discouraged: see
  "Interpretation, not tools" in your standing instructions.
- Python 3.11 with numpy, scipy, Pillow and nibabel; Read, Write, Edit, Glob, Grep and Bash. No network,
  no other scan, no internet lookups, no subagents.

Every image a session opens costs tokens, and each session has a cap of $0.50 and a 15-minute limit for all five
questions together. A session that runs out writes no answer and is scored at 50 on every question (the middle of
the ranking), as is any answer that is not a number from 0 to 100 (one bad number fails the whole answer). Ask for
what is worth looking at, not everything.

## The answers

The answer key was read from the radiologist's report for this same scan, not from the images: a language model
read the report and classified each finding as malignant; possibly malignant (the report raises cancer as a real
possibility, for example "cannot exclude malignancy" or a differential that includes cancer); nonspecific (called
nonspecific or indeterminate, with no lean either way); benign (reactive, inflammatory, physiologic, or a benign
tumour such as a cyst or hemangioma); or not characterized.

- **Liver, lungs, lymph nodes**: yes when the report has any finding there not classified benign (so a nonspecific
  or uncharacterized finding counts as yes).
- **Right upper lobe**: yes when the report has a malignant or possibly malignant lung lesion there. Scans whose
  report did not let the lobe be decided have no answer for this question (about a third).
- **Pneumonitis**: yes when the report describes non-infectious pneumonitis (radiation-, drug- or
  immunotherapy-related, or organizing pneumonia) seen on this scan. Infection-type causes and past episodes
  mentioned only in the history do not count.

For TRAIN mistakes you also get the report itself (`report_file` in the mistakes file), so you can see the
wording behind each answer; the session never sees reports, and VAL reports are not available to anyone. A scan with no answer for a question (`null` in the mistakes file) is simply not scored on that question. The
scans were drawn one per patient and **enriched**: each question has at least 30 yes and 30 no scans among the 200
VAL scans, so the share of yes here is higher than in the clinic for the rare questions. Of the TRAIN scans with an
answer: liver 28% yes, lungs 75%, lymph nodes 42%, right upper lobe 34%, pneumonitis 15%.

## How it is scored

**AUROC** per question, over the scans with an answer for it, then averaged over the five. For one question it is the
chance that a yes scan was given a higher probability than a no scan, over every such pair; equal probabilities count
as half. 0.5 is chance (the same number for every scan, or guessing), 1.0 every yes scan ranked above every no scan.
Only the order of the probabilities counts, so the score rewards telling scans apart: a program that gives most scans
the same few round numbers throws away ranking it could have had. A failed session or an unreadable answer counts as
50 on every question. The score that counts is on the validation scans (VAL, 200 scans from other patients), which
you never see. VAL has 187 scored scans for liver (50 yes), 193 for lungs (146 yes), 193 for lymph nodes (99 yes),
130 for the right upper lobe (43 yes) and 197 for pneumonitis (30 yes).

The TRAIN breakdown gives, per question: the AUROC; the Brier score (the mean squared gap between the probability and
the answer, lower is better); the mean probability given to yes and to no scans; a calibration table by tenth of
probability (how many scans fell in each tenth and how many were yes); the number of distinct probabilities and the
share on the most common one; and the yes, no and unscored counts. Plus the number of failed sessions. The mistakes
file marks a scan as a mistake on a question when its probability is on the wrong side of that question's TRAIN
rate above (at or above it for a no scan, below it for a yes scan).

The run stops by itself after two rounds in a row that set no new best VAL score, after 10 rounds, or when spending
passes $50 in a round or $200 in all. TRAIN grows from 100 scans in round 1 to 400 by round 10.

## Reference points on this task

- CT image models trained on tens of thousands of labelled scans (a linear model on frozen embeddings), on these
  same 200 VAL scans: **Pillar0 0.818** mean AUROC (liver 0.71, lungs 0.82, lymph nodes 0.85, right upper lobe
  0.89, pneumonitis 0.82); MerlinOnc 0.784; COLIPRI 0.775.
- Your program learns from far fewer scans than the image models did, so matching them would be a strong result.
  A mean above about 0.92 would more likely mean something leaked than a breakthrough.

## Reading the literature

You cannot search the internet yourself: you can ask the search assistant described in your standing instructions,
which sees only your question and the shared findings file, never a scan. Useful questions: how radiologists tell
benign from non-benign liver, lung and nodal findings on CT; size and attenuation rules for lymph nodes; how to
locate the right upper lobe and its fissures on axial slices; the CT patterns of radiation, drug and immunotherapy
pneumonitis versus infection. Questions must be general: no ids, no measurements copied from a scan, no findings
notes; such a question is refused. Read the findings file before asking, and record in your notebook what you
learned.

## What is noise

The answer is what the radiologist wrote, read by a language model, not the truth in the images. Reports differ in
what they mention: a tiny liver lesion "too small to characterize" is a yes; a stable treated nodule may be called
benign or not; a radiologist may not name the lobe. Low-dose and non-contrast scans hide liver lesions. Expect a
ceiling well below 1.0.
