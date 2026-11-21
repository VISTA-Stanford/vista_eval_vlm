# The task: did the radiologist call progression, comparing this CT with the prior one?

Each pair is two chest CTs of one adult with a thoracic cancer (mostly lung cancer) from one academic cancer center:
a CURRENT scan and the PRIOR scan its radiology report compared it with. The program is shown **the two CT volumes
and the number of days between them**, nothing else: no report, no age, no history. It answers one question as a
probability from 0 to 100:

> Question: Comparing the CURRENT scan with the PRIOR scan, has the cancer progressed?
> - Yes: progression or mixed response: clear growth of known malignant lesions, or new lesions suspicious for cancer,
>   even if other sites improved.
> - No: stable disease or response.
> Do not count non-cancer changes (infection, aspiration, organizing pneumonia, drug reaction, evolving post-radiation
> change).

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code session (Claude
Sonnet, which can look at images) reads `instructions.md`, examines the two CTs, writes a short findings note, and
answers one JSON object: `progression`, a number from 0 to 100, and `reasons`: the 3 to 5 things in the scans
that moved the probability most, each with where they were seen. The reasons are not scored; they are shown to you
beside each TRAIN pair. The wrapper that says where the scans are, states the days between them, quotes the question
word for word and asks for that output is fixed; you write only what it follows. The wrapper does not say how the answer was made or how common it is:
if the sessions should know, your instructions must say so.

What the session has:

- the two CTs as folders `ct/prior/` and `ct/current/`, each `ct.npz` (one array, key `hu`: Hounsfield units, int16,
  shape (x, y, z)) and `meta.json` (shape, voxel spacing in mm, orientation). Orientation is RAS: the index grows
  toward the patient's Right on axis 0, Anterior on axis 1, Superior on axis 2. Thin scans were averaged into ~2.5 mm
  slabs.
- `tools/view_ct.py`, which renders axial, coronal or sagittal slices in lung, soft-tissue or bone windows (or any HU
  range) as one PNG grid, PRIOR and CURRENT side by side at the same fraction of each scan's axis, or one scan
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

The answer key is the CURRENT scan's own radiology report, read by a language model (not by a clinician) into one of
four calls: progression, mixed response, stable, response. Progression or mixed = 1; stable or response = 0. It is what
the radiologist concluded comparing these two scans, sometimes with other studies or the clinical question in mind,
not a measurement. The PRIOR scan may be weeks or many months earlier, before or after a treatment; the two scans often
cover different lengths of the body.

The pairs were drawn at random from the benchmark's train patients (TRAIN, 400 pairs) and validation patients (VAL,
200 pairs), at the natural rate. 31% of the TRAIN pairs are 1 (progression or mixed). A patient may have more than one
pair in a pool.

## How it is scored

**AUROC**: the chance that a progression pair was given a higher probability than a response/stable pair, over every
such pair of patients; equal probabilities count half. 0.5 is chance (the same number for every pair), 1.0 every
progression pair ranked above every other. Only the order of the probabilities counts, so a program that gives most
pairs the same few round numbers throws away ranking it could have had. A failed session counts as 50. The score that
counts is on the 200 VAL pairs, which you never see; it moves by about 0.04 by chance.

The TRAIN breakdown gives the AUROC; the Brier score (the mean squared gap between the probability and the answer,
lower is better); the mean probability given to progression and to response/stable pairs; a calibration table by
tenth of probability; the number of distinct probabilities and the share on the most common one; the counts; and the
number of failed sessions. The mistakes file marks a pair as a mistake when its probability is on the wrong side of
the TRAIN progression rate (31%).

The run stops by itself after two rounds in a row that set no new best VAL score, after 10 rounds, or when spending
passes $60 in a round or $250 in all. The TRAIN batch grows from 40 pairs in round 1 to 100 by round 4; pairs
the program has answered right several rounds in a row are retired from it, so later batches are smaller.

## Reference points

- Linear models on CT embeddings, given both scans' embeddings, on the benchmark's whole validation split (about 4,300
  pairs, the same answer key): Pillar0 0.79, COLIPRI 0.78, MerlinOnc 0.75 AUROC. From the CURRENT scan alone they score
  only 0.005-0.007 lower: the embeddings find the current disease burden, not the change.
- Frontier models asked directly about a different question (whether a tumor board called progression, 45 pairs):
  Gemini 4 Argon 0.59, Claude Opus 5.5 0.70 AUROC.
- The radiologist saw the same two scans, so a careful comparison can in principle get close to the report's call.
  A VAL AUROC far above 0.9 would more likely mean something leaked than a breakthrough.

## What is noise

The report's call is not a RECIST measurement: a radiologist may call progression for a single new nodule, or stable
despite slight growth, and may describe "mixed" findings in words a language model then read one way. Post-radiation
and post-surgical change mimic growth; atelectasis and effusion come and go; breath-hold and slice levels differ.
