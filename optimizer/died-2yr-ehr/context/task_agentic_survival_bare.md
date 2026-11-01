# The task: this patient's survival curve, judged from a summary of their record

Each patient is an adult with a thoracic cancer. The program is shown **only a short summary of the patient's
medical record**, written earlier by a language model from the record up to 7 days before the patient's first cancer
treatment. From that summary alone it gives a survival curve: the chance the patient is still alive, of any cause, at
each point in the years after the last day of the record. The summary can be wrong or leave things out, and there is
no other record to check it against.

This is a deliberately bare setup: you are told the task, the data and how you are scored, and nothing else. What
makes a good curve here — how common death is and when, how far apart patients' curves should be, which facts in a
summary matter — is for you to work out from the training patients, not given to you.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the summary (the file `summary.txt` in its
folder), writes a short findings note, and answers one JSON object: `survival_curve`, the chance of being alive at
each point after the record's last day as points `[months, chance 0 to 1]` (months from 0 and strictly increasing,
the chance never rising, to at least month 120, up to 2,000 points, read as a step curve), and `reasons`: the 3 to 5
facts from the summary that shaped the curve most. The reasons are not scored; they are kept and shown to you beside
each TRAIN patient. The wrapper that gives the session the summary and asks for that output is fixed; you write only
`instructions.md` and `tools/`.

A session that fails, or a curve that rises, leaves 0 to 1, or stops before about month 75, **stops the whole run**
so a person can look; there is no fallback score.

The session has Python 3.11 with pandas, pyarrow and duckdb; grep and ripgrep; Read, Write, Edit, Glob, Grep and
Bash. No network, no other patient, no internet lookups, no subagents. Every line it reads costs tokens, and each
session has a cap of $0.50 and 15 minutes.

## The answer

The truth for each patient is their overall survival in the benchmark (vista_bench v1.7), counted from the record's
last day: either the day the patient died, of any cause, or the last day they were known alive (censored: alive
then, unknown after).

## How it is scored

**Integrated Brier score, lower is better.** At each month on a monthly grid, the squared gap between the curve's
chance of being alive and what happened (1 alive, 0 dead), averaged over patients; a censored patient counts only
while they were followed, and the others are re-weighted for that (inverse-probability-of-censoring weights, as
scikit-survival computes them). Averaged over months 1 to about 74, past which too few patients are followed to
trust. The score that decides the run is on validation patients (VAL) you never see; only its integrated Brier
reaches you, once per round, beside the integrated Brier on the training patients you did see.

## What you can see and do, each round

- **The two integrated Brier numbers** (TRAIN and VAL) and simple counts (how many patients, deaths and censored,
  how many failed). That is the whole scoreboard; anything more — how your curves compare with simply giving everyone
  the same curve, how well they separate the patients who died early from those who lived long, how calibrated they
  are at a given month — you compute yourself from the patients below if you want it.
- **Every TRAIN patient of the pass**, with its true outcome (`gold_died`, `gold_months_followed`), the curve your
  program gave it, the reasons the session stated, and the session's own findings note and transcript — the ones it
  got right as well as the ones it got wrong. A patient is marked `mistake` when its curve did worse on that patient
  than the cohort's average curve would have, or its session failed. You can read all of them, and the summary each
  session read (`<id>-summary.txt`). This is how you learn what the summaries hold and what separates the patients
  who died early from those who lived long.
- **The search assistant.** You cannot browse yourself, but you can ask it general questions (no patient ids, no
  values copied from a summary, no note text; such a question is refused). Read the shared findings file before
  asking; there is no practical limit on questions.
- **Your own tools.** You may write helper scripts under `tools/` for the sessions to run, and run them yourself on
  the TRAIN patients first to check what they cost and whether they help. Build whatever you find useful.

Write for patterns that recur across patients, not for one patient's summary: an instruction fitted to a single
record scores nothing on VAL.
