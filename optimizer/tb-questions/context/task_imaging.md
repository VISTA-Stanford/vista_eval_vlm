# The task: did the tumor board call progression, from the two most recent CTs?

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center who was discussed
at a thoracic tumor board. The program is shown **two chest CT volumes** and nothing else: the most recent CT before
the tumor board (MOST RECENT) and the CT before it (PRIOR). No report, no age, no history. It answers one question as a
probability from 0 to 100:

> Question: As of one week before the tumor board, does the most recent imaging show cancer progression?
> - Yes: progression.
> - No: response or stable disease.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code session (Claude
Sonnet, which can look at images) reads `instructions.md`, examines the two CTs, writes a short findings note, and
answers one JSON object: `progression`, a number from 0 to 100, and `reasons`: the 3 to 5 things in the scans
that moved the probability most, each with where they were seen. The reasons are not scored; they are shown to you
beside each TRAIN pair. The wrapper that says where the scans are, quotes the question word for word and asks for that
output is fixed; you write only what it follows. The wrapper does not say how the answer was made or how common it is:
if the sessions should know, your instructions must say so.

What the session has:

- the two CTs as folders `ct/prior/` and `ct/current/`, each `ct.npz` (one array, key `hu`: Hounsfield units, int16,
  shape (x, y, z)) and `meta.json` (shape, voxel spacing in mm, orientation). Orientation is RAS: the index grows
  toward the patient's Right on axis 0, Anterior on axis 1, Superior on axis 2. Thin scans were averaged into ~2.5 mm
  slabs.
- `tools/view_ct.py`, which renders axial, coronal or sagittal slices in lung, soft-tissue or bone windows (or any HU
  range) as one PNG grid, PRIOR and MOST RECENT side by side at the same fraction of each scan's axis, or one scan
  alone, with optional crops; the session opens the PNG with its Read tool. The two scans often cover different
  lengths of the body and were taken on different machines, so "the same fraction" is only roughly the same level:
  you may change how it lines them up, and its views. Adding other analysis tools is discouraged: see
  "Interpretation, not tools" in your standing instructions.
- Python 3.11 with numpy, scipy, Pillow and nibabel; Read, Write, Edit, Glob, Grep and Bash. No network, no other
  scan, no internet lookups, no subagents.

Every image a session opens costs tokens, and each session has a cap of $0.75 and a 20-minute limit. A session that
runs out writes no answer and is scored at 50 (the middle of the ranking), as is any answer that is not a number from
0 to 100. Ask for what is worth looking at, not everything.

## The answers

The answer key is the tumor board's own call, read from the tumor board note by a language model (not by a
clinician): 1 = progression, 0 = response or stable disease. It is what the tumor board concluded, from everything it
had (reports, the scans, the clinical picture, sometimes PET or MRI), not a measurement on these two CTs. In an earlier
check the tumor board's call and the radiology report's call agreed on about three in four patients. The PRIOR CT is
the scan before the MOST RECENT one, which may be weeks or many months earlier, before or after a treatment.

The patients were drawn from the benchmark's train and validation patients (the 288 pairs where both CTs exist) and
split at random, stratified by the tumor board's answer, into TRAIN (188 pairs) and VAL (100). 60% of the TRAIN pairs are progression.

## How it is scored

**AUROC**: the chance that a progression pair was given a higher probability than a response/stable pair, over every
such pair of patients; equal probabilities count half. 0.5 is chance (the same number for every pair), 1.0 every
progression pair ranked above every other. Only the order of the probabilities counts, so a program that gives most
pairs the same few round numbers throws away ranking it could have had. A failed session counts as 50. The score that
counts is on the 100 VAL pairs, which you never see; it moves by about 0.05 by chance.

The TRAIN breakdown gives the AUROC; the Brier score (the mean squared gap between the probability and the answer,
lower is better); the mean probability given to progression and to response/stable pairs; a calibration table by
tenth of probability; the number of distinct probabilities and the share on the most common one; the counts; and the
number of failed sessions. The mistakes file marks a pair as a mistake when its probability is on the wrong side of
the TRAIN progression rate (60%).

The run stops by itself after two rounds in a row that set no new best VAL score, after 10 rounds, or when spending
passes $60 in a round or $250 in all. The TRAIN batch grows from 90 pairs in round 1 toward all 188 by round 5; pairs
the program has answered right several rounds in a row are retired from it, so later batches are smaller.

## Reference points

- Asked directly with 25 slices of each scan in chest windows, on other patients (the 45 TEST pairs): Gemini 4 Argon
  0.59, Claude Opus 5.5 0.70 AUROC.
- A linear model on CT embeddings (MerlinOnc), on the same 45 TEST pairs: 0.56 AUROC from the most recent CT alone,
  0.51 from the two scans together. Image models have not found the tumor board's call in these scans.
- Models reading a text summary of the record have done better than models reading the scans: the tumor board's call
  depends on more than the images. Expect a ceiling well below 1.0; a VAL AUROC above about 0.85 would more likely
  mean something leaked than a breakthrough.

## What is noise

The tumor board's call is not a RECIST measurement: it can call progression for a new lesion on PET, a rising marker
or symptoms, and call stability despite a growing lesion it attributes to treatment effect. Post-radiation and
post-surgical change mimic growth; atelectasis and effusion come and go. The PRIOR scan may be from a different phase
of care.
