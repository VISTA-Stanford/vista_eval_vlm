# The task: predict the thoracic tumor board's treatment recommendation from a summary and the full record

Each patient is an adult with non-small cell lung cancer from one academic cancer center who was discussed at a
thoracic tumor board. The program is shown **one text summary** of the patient's electronic health record, written by
a language model from the record up to **one week before their first thoracic tumor board**, and **the full record
itself, cut at the same day**: every note, the OMOP tables (visits, conditions, procedures, drugs, measurements,
imaging, observations) and allow-listed cancer registry fields. Nothing dated after that day is in it. It
answers the question(s) the tumor board discussed for that patient, each as a probability for every option:

| key | question (the session sees it word for word) | asked of |
|---|---|---|
| `early_stage_plan` | The tumor board discussed curative-intent treatment for this patient's stage I-III non-small cell lung cancer. Which treatment do you predict the tumor board will recommend? A. Systemic therapy followed by surgery, with or without adjuvant therapy; B. Surgery followed by systemic therapy; C. Surgery alone; D. Radiation alone; E. Definitive chemoradiation, with or without adjuvant systemic therapy | patients whose tumor board discussed the early-stage plan |
| `first_line_regimen` | The tumor board discussed first-line systemic treatment for this patient's metastatic non-small cell lung cancer. Which regimen do you predict the tumor board will recommend? A. Chemotherapy alone; B. Chemotherapy with immunotherapy; C. Chemotherapy with targeted therapy; D. Targeted therapy alone; E. Immunotherapy alone | patients whose tumor board discussed first-line metastatic treatment |

Most patients are asked one question; a few (about 1 in 30) are asked both.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each patient a separate Claude Code session (Claude
Sonnet) reads `instructions.md`, reads the summary (in its prompt and as the file `summary.txt`), may read the record
(the folder `record/`, with a guide to it, `record_guide.md`, that says where each kind of fact lives), writes a short
findings note, and answers one JSON object: for each question asked, an object with a number 0 to 100 for every letter
A to E, and `reasons`: the 3 to 5 facts from the summary or the record that moved the probabilities most. The reasons
are not scored; they are shown to you beside each TRAIN patient. The wrapper that gives the summary, describes the
record folder, quotes the question(s) word for word and asks for that output is fixed; you write only what it follows.
The wrapper does not say how the answers were made or how common each is: if the sessions should know, your
instructions must say so.

The program starts with one tool, `tools/digest.py`, the notes reader: it prints the clinicians' recent
assessment-and-plan text (last 180 days before the cutoff), the latest CT/PET/MRI impressions (last 120 days) and the
latest pathology diagnoses, about 8-10 KB. The record is large (a median patient has about 50 notes, about 100 KB of
note text, and about 1,000 structured rows; some have far more), so a session that reads it whole runs out of time or
budget.

What the session has: the summary, the record and its guide; Python 3.11 with numpy, scipy, Pillow, nibabel, pandas,
pyarrow and duckdb; Read, Write, Edit, Glob, Grep and Bash. No network, no internet lookups, no subagents. Each session
has a cap of $0.50 and a 15-minute limit. A session that runs out writes no answer and is scored as no information (an
equal share on every letter), as is an answer with a missing letter, a number outside 0 to 100, a word, or all zeros.
Numbers that do not add up to 100 are rescaled to 100, not refused.

## The answers

The answer key is the tumor board's own recommendation, read from the tumor board note by a language model (not by a
clinician); "indeterminate or deferred" is not scored. The summary is cut a week before the tumor board, but for many
patients the tumor board came months after diagnosis, so the summary and the record may already describe a plan, a
referral, an order or treatment already started. Reading that out is fair: it was all written before the cutoff.

The tumor board picks one plan where several can be acceptable, so a sensible second choice is common; the probability
on each letter is what lets that show.

Shares among the TRAIN patients asked each question (492 TRAIN patients; 355 asked the early-stage question, 155 the
metastatic one):
- Early-stage: A 22%, B 6%, C 24%, D 17%, E 31%.
- Metastatic: A 43%, B 19%, C 7%, D 24%, E 8%.

## How it is scored

For each question, over the patients asked it: for each letter, the **AUROC** of the probability on that letter for
telling patients whose answer was that letter from those whose answer was another; then the mean over the letters
(only letters that some patient in the set had). This is the same multi-class AUROC our embedding models are scored
with. The task score is the mean of the two questions. 0.5 is chance (the same numbers for every patient), 1.0 every
patient ranked right on every letter. Only the order of each letter's probabilities across patients counts: a program
that gives everyone the same spread, or puts most patients on the same few round numbers, throws away ranking.

The score that counts is on the validation patients (VAL: 111 patients of other people), which you never see. Rare
answers have few patients, so their letters' AUROCs, and the metastatic question as a whole, move a lot by chance.

The TRAIN breakdown gives, per question: the mean AUROC and each letter's AUROC; how many patients had each answer and
how many times each letter was the top pick; top-pick balanced accuracy (the mean over answers of the share whose top
letter was right); the share with the tumor board's answer in the top two letters; the mean probability on the tumor
board's answer; and how many were asked and how many sessions failed. The mistakes file marks a question as a mistake
when its top letter was not the tumor board's answer.

The run stops by itself after three rounds in a row that set no new best VAL score, after 10 rounds, or when spending
passes $60 in a round or $250 in all. The TRAIN batch grows from 200 patients in round 1 toward all 492 by round 5;
patients the program has answered right several rounds in a row are retired from it, so later batches are smaller.

## Reference points

- The same agent with the summary only (no record), on the same 111 VAL patients: the plain question 0.814 (early-stage
  0.810, metastatic 0.818); its best climbed version 0.845 (early-stage 0.815, metastatic 0.874), reached in round 1 and
  not beaten in the two rounds after. Whether the record adds anything to the summary is what this run tests.

- Gemini 4 Argon and Claude Opus 5.5 asked directly, the same summary and the same question words, one probability
  per letter, on the 119 TEST patients (other patients than VAL): mean AUROC 0.81 and 0.82 (early-stage 0.81 / 0.83,
  metastatic 0.81 / 0.82); top-pick balanced accuracy about 0.50; the tumor board's answer in the top two about 0.78.
- A linear model on embeddings of the same summaries (trained on the benchmark's train patients), on the same 119 TEST
  patients: mean AUROC 0.73 (early-stage 0.74, metastatic 0.72); top-pick balanced accuracy about 0.33. The models
  asked directly beat it by about 0.08, so reading the summary adds more than a frozen embedding of it carries.
- VAL is small: a gain of a few hundredths is within chance.

## What is noise

The answer is one tumor board's recommendation as a language model read it from the note, not the only right plan.
Plans hinge on things neither the summary nor the record may hold (performance status, lung function, patient preference, a molecular
result still pending). Expect a ceiling well below 1.0.
