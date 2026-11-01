# The task: how likely is it that this patient died within 1 year, and within 2 years

Each patient is an adult with a thoracic cancer. The program is shown **only a short summary of the patient's
medical record**, written earlier by a language model from the record up to 7 days before the patient's first cancer
treatment. From that summary alone it gives two probabilities, each 0 to 100: the chance the patient died, of any
cause, within 1 year of the cancer diagnosis, and within 2 years. The summary can be wrong or leave things out, and
there is no other record to check it against.

This is a deliberately bare setup: you are told the task, the data and how you are scored, and nothing else. What
makes a good probability here — how common death is, how to spread the numbers, which facts in a summary matter — is
for you to work out from the training patients, not given to you.

## The program's contract

The program is `instructions.md` plus anything under `tools/` (empty in the seed). For each patient a separate
Claude Code session (Claude Sonnet) reads `instructions.md`, reads the summary (the file `summary.txt` in its
folder), writes a short findings note, and answers one JSON object: `died_within_1_year` and `died_within_2_years`,
each a whole number 0 to 100 (its probability in percent), and `reasons`: the 3 to 5 facts from the summary that
moved the probabilities most. The reasons are not scored; they are kept and shown to you beside each TRAIN patient.
The wrapper that gives the session the summary and asks for that output is fixed; you write only `instructions.md`
and `tools/`. A session that fails, or an answer that is not a number from 0 to 100, is scored at 50 (the middle of
the ranking).

The session has Python 3.11 with pandas, pyarrow and duckdb; grep and ripgrep; Read, Write, Edit, Glob, Grep and
Bash. No network, no other patient, no internet lookups, no subagents. Every line it reads costs tokens, and each
session has a cap of $0.50 and 15 minutes.

## How it is scored

**AUROC** at each horizon, averaged over the two. At one horizon it is the chance that a patient who died was given
a higher probability than one who lived, over every such pair; equal probabilities count as half. 0.5 is chance (the
same number for everyone, or guessing); 1.0 is every patient who died ranked above every one who lived. Only the
order of the probabilities counts, not their level. The score that decides the run is on validation patients (VAL)
you never see; only its AUROC reaches you, once per round, beside the AUROC on the training patients you did see.

## What you can see and do, each round

- **The two AUROC numbers** (TRAIN and VAL) and simple counts (how many patients, how many failed, how many gave a
  higher chance of death by 1 year than by 2, which is impossible). That is the whole scoreboard; anything more about
  the numbers — how well they are calibrated, how spread out they are, how the deaths and survivors separate — you
  compute yourself from the patients below if you want it.
- **Every TRAIN patient of the pass**, with its true outcome at each horizon, the probability your program gave it,
  the reasons the session stated, and the session's own findings note and transcript — the ones it got right as well
  as the ones it got wrong. You can read all of them, and the summary each session read (`<id>-summary.txt`). This is
  how you learn what the summaries hold and what separates the patients who died from those who lived.
- **The search assistant.** You cannot browse yourself, but you can ask it general questions (no patient ids, no
  values copied from a summary, no note text; such a question is refused). Read the shared findings file before
  asking; there is no practical limit on questions.
- **Your own tools.** You may write helper scripts under `tools/` for the sessions to run, and run them yourself on
  the TRAIN patients first to check what they cost and whether they help. Build whatever you find useful.

Write for patterns that recur across patients, not for one patient's summary: an instruction fitted to a single
record scores nothing on VAL.
