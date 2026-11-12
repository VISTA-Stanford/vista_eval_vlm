# The task: how likely is it that this patient dies within 1 year, and within 2 years, of a decision point in their care

Each patient is an adult with a thoracic cancer (mostly lung cancer) from one academic cancer center whose first
cancer treatment was IV chemotherapy or an oral targeted drug. Each patient is assessed at up to five **decision
points**, each tied to a chest CT: treatment start (a CT from 60 days before to 30 days after the first treatment),
the first restaging CT (28 to 120 days after it), and 6, 12 and 24 months (the latest chest CT in the 3 months before
each mark). A patient has only the points they had a CT for.

At each point the program is shown **the patient's medical record up to the end of that CT's day**: every
structured EHR table, every clinical note, and the cancer registry's fields for this cancer allowed at that day.
From it, it gives two probabilities, each 0 to 100: the chance the patient dies, of any cause, within 1 year of that
day, and within 2 years. Nothing dated after that day is in the record.

## Two arms, one program

Every point is answered twice, by two separate sessions:

- **fresh**: the session sees the record at that point and nothing else.
- **memory**: the session also sees, in its prompt, its own earlier answers for this patient: for each earlier
  decision point, its date, the two probabilities it gave then and the reasons it gave (a reason dated after the
  current day is left out). At treatment start there is no earlier answer, so the memory arm takes the fresh arm's
  answer there.

You edit **one** program, used by both arms at every point. The memory arm's copy of `instructions.md` gets one
fixed paragraph added at its end, which you cannot edit:

> You have assessed this patient before, at earlier points in their care; the prompt lists your earlier
> probabilities and reasons, if any. Read what has happened in the record since your last assessment, then give your
> updated probabilities.

So write `instructions.md` to work at any point in care, with or without that block. Points run in order within a
pass, so the memory arm's earlier answers are the ones this same version gave.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient, point and arm
a separate Claude Code session (Claude Haiku) reads `instructions.md`, reads the record, writes a short findings
note, and answers one JSON object with `died_within_1_year` and `died_within_2_years`, each a number from 0 to 100,
and `reasons`: the 3 to 5 record facts that moved the probabilities most, each with the record file and its date.
The wrapper that tells it where things are and asks for that output is fixed: it states both questions, that the
record runs through the day of a chest CT at a decision point, the folder layout, and (memory arm) the earlier
answers. It states no death rate.

What the session has, as a read-only folder `record/`: `notes.txt` (every note on or before that day, in date order,
each starting `===== NOTE <note id> | <date or datetime> | <note type> =====`; grep it, don't read it whole), one
parquet file per OMOP table with rows (`person`, `observation_period`, `visit_occurrence`, `visit_detail`,
`condition_occurrence`, `procedure_occurrence`, `drug_exposure`, `drug_era`, `device_exposure`, `episode`,
`image_occurrence`, `measurement`, `observation`, `note`), the registry files (`registry_diagnoses`,
`registry_treatments`, `registry_outcomes`, `registry_events`, `registry_misc_details`), `meta.json` (opaque id and
the record's last day), and beside it `record_guide.md` (the fixed data map in the appendix). Python 3.11 with
pandas, pyarrow and duckdb; grep and ripgrep; Read, Write, Edit, Glob, Grep and Bash. No network, no subagents.

A later decision point's record holds everything up to that day: the treatment given so far (`drug_exposure`,
`drug_era`, notes), restaging scans and their reports, labs, visits and admissions after treatment began.

Each session has a cap of $0.50 and a 15-minute limit; a session that runs out, or an answer that is not a number
from 0 to 100, is scored at 50 at both horizons. Later records are longer: ask for what is worth reading.

## The answer

**Died within 1 year / 2 years** (`gold_died_within_1_year`, `gold_died_within_2_years` = 1): the patient died, of
any cause, within 12 / 24 months of the decision point's CT. A row followed less than a horizon with no death has no
gold at that horizon (`null`) and is left out of that horizon's score only.

## How it is scored

**AUROC**, computed separately for each decision point, horizon and arm (5 x 2 x 2 = 20 cells, on every validation
row), then averaged. At one cell it is the chance that a row whose patient died was given a higher probability than
one whose patient lived; ties count half; 0.5 is chance. Only the order counts within a cell, so the program must rank
patients well at every point, not just tell early points from late ones. The validation patients (VAL) are other
patients of the same group, which you never see; only the score reaches you.

The TRAIN breakdown gives the same mean, and per point x arm x horizon cell the AUROC, the Brier score (mean squared
gap between probability and outcome, lower is better), observed / expected deaths (below 1 means the probabilities
are too high) and the calibration slope (below 1 means too spread out, above 1 too bunched). These do not change the
score. The mistakes file has one entry per TRAIN row and arm (`id` = `<row>.<arm>`), with its point, gold and
probability at each horizon, its reasons, and for the memory arm the earlier-answers block it saw. A mistake is a
probability on the wrong side of that horizon's TRAIN death rate (all TRAIN rows pooled), or a failed answer.
Mistakes come with their record, copied once per row as `<row>-<file>` and listed in both arms' entries, and with each
session's findings note and transcript as `<row>.<arm>-trace-<file>`.

## Reference points

None is given from the validation patients. A large jump in one round more likely means something leaked, or a
version learned something about the TRAIN batch, than a breakthrough.

## What is noise

Deaths unrelated to the cancer; long survivors on modern treatment; sparse records for patients treated partly
elsewhere; registry death dates that lag. Each cell has a few dozen to a few hundred rows, so a cell's AUROC moves by
several hundredths by chance; the mean over 20 cells moves less.
