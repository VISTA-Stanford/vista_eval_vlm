# The task: predict what a thoracic tumor board decides, from the record cut a week before it

Each example is one **thoracic tumor board encounter** at one academic cancer center: a patient with a thoracic
cancer (mostly non-small cell lung cancer) discussed at the tumor board, at a tumor board whose recommendation was then carried
out. A patient may have several encounters; each is its own example with its own record. The program is shown **the
patient's full record cut one week before that tumor board**: every note, the OMOP tables (visits, conditions, procedures,
drugs, measurements, imaging, observations) and allow-listed cancer registry fields (the registry's first-course
treatment table is removed: it is the answer). Nothing dated after the cutoff is in it. There is no summary.

Every encounter is asked two questions, and some one more, each answered as a probability for every option:

| key | question (the session sees it word for word, with lettered options) | asked of |
|---|---|---|
| `decision_type` | What will the tumor board decide for this patient at this encounter? A more workup; B start a new treatment; C continue; D switch or stop; E add local therapy to ongoing systemic therapy; F surveillance; G refer; H supportive care or hospice; I defer | every encounter |
| `intent` | What will the intent of the tumor board's plan be? A curative; B palliative, to prolong life; C symptom control only; D undetermined until more workup; E not applicable (no treatment decided) | every encounter |
| `early_stage_plan` | Untreated stage I-III NSCLC: what will the tumor board recommend? A-E the old five options (systemic then surgery; surgery then systemic; surgery alone; radiation alone; definitive chemoradiation), F systemic or targeted therapy first, G another plan, H more workup first | untreated stage I-III NSCLC |
| `after_surgery_plan` | Stage I-III NSCLC after surgery or chemoradiation: what next? A adjuvant or consolidation systemic; B postoperative radiation; C both; D observation; E another plan; F more workup first | stage I-III NSCLC after surgery or chemoradiation |
| `first_line_regimen` | Untreated stage IV or recurrent NSCLC: which first-line plan? A-E the old five regimens (chemo; chemo + immunotherapy; chemo + targeted; targeted alone; immunotherapy alone), F targeted + antibody, G another regimen, H more workup first | untreated stage IV or recurrent NSCLC |

Which treatment question an encounter gets depends only on facts known before the tumor board (cancer, stage, setting),
never on what the tumor board decided; "more workup first" is an answer like any other.

## The program's contract

The program is `instructions.md` plus anything under `tools/`. For each encounter a separate Claude Code session reads
`instructions.md`, may read the record (the folder `record/`, with a guide, `record_guide.md`, that says where each
kind of fact lives), writes a short findings note, and answers one JSON object: for each question asked, an object
with a number 0 to 100 for every letter, and `reasons`: the 3 to 5 facts from the record that moved the probabilities
most (not scored; shown to you beside each TRAIN encounter). The wrapper that describes the record folder, quotes the
questions word for word and asks for that output is fixed; you write only what it follows. The wrapper does not say
how common each answer is: if the sessions should know, your instructions must say so.

The program starts with one tool, `tools/digest.py`, the notes reader: the clinicians' recent assessment-and-plan text
(last 180 days before the cutoff), the latest CT/PET/MRI impressions (last 120 days) and the latest pathology
diagnoses, about 8-10 KB. The record is large (a median patient has about 50 notes, about 100 KB of note text; some
have far more), so a session that reads it whole runs out of time or budget.

What the session has: the record and its guide; Python 3.11 with numpy, scipy, Pillow, nibabel, pandas, pyarrow and
duckdb; Read, Write, Edit, Glob, Grep and Bash. No network, no subagents. Each session has a cap of $0.50 and a
15-minute limit. A session that runs out writes no answer and is scored as no information (an equal share on every
letter), as is an answer with a missing letter, a number outside 0 to 100, a word, or all zeros. Numbers that do not
add up to 100 are rescaled, not refused.

## The answers

The answer key is the tumor board's own decision, read from the tumor board note and the record by a language model, kept
only for tumor boards whose recommendation was carried out afterwards. The record may already describe a plan, a referral
or a treatment under way, or an earlier tumor board's discussion: reading that out is fair, it was all written before the
cutoff. Often several plans are acceptable and the tumor board picks one; the probability on each letter lets that show.

Shares among the 2,916 TRAIN encounters (1,591 patients):
- decision_type: A 21%, B 31%, C 15%, D 7%, E 6%, F 14%, G 5%, H 1%, I under 1%.
- intent: A 28%, B 40%, C 3%, D 15%, E 14%.
- early_stage_plan (339 asked): A 11%, B 1%, C 17%, D 18%, E 23%, F 3%, G 4%, H 22%.
- after_surgery_plan (88 asked): A 33%, B 6%, C 13%, D 33%, E 9%, F 7%.
- first_line_regimen (135 asked): A 21%, B 19%, C 1%, D 20%, E 1%, F 0%, G 13%, H 24%.

## How it is scored

For each question, over the encounters asked it: for each letter, the **AUROC** of the probability on that letter for
telling encounters whose answer was that letter from those whose answer was another; then the mean over the letters
present. The task score is the mean of three questions: `decision_type`, `intent` and `early_stage_plan`.
`after_surgery_plan` and `first_line_regimen` are scored and shown but too few to count. 0.5 is chance (the same
numbers for every encounter). Only the order of each letter's probabilities across encounters counts: giving everyone
the same spread, or the same few round numbers, throws away ranking.

The score that counts is on the validation encounters (VAL, other patients), which you never see. Rare answers have
few encounters, so their letters' AUROCs move a lot by chance.

The run stops after two rounds in a row with no new best VAL score, after 8 rounds, or when spending passes $100 in a
round or $600 in all. The TRAIN batch grows from 300 encounters in round 1 to 600 by round 4; encounters answered right
several rounds in a row are retired from it.

## Reference points

- An earlier agent on a different answer key (first tumor boards only, the old five-option early-stage and first-line
  questions) scored mean AUROC 0.83-0.87 from a summary plus this kind of record; the gain from tuning its
  instructions was mostly within chance. Decision type and intent are new questions with no earlier score.

## What is noise

The answer is one tumor board's decision as a language model read it, not the only right plan. Decisions hinge on things
the record may not hold (performance status, patient preference, a result still pending). Expect a ceiling well below
1.0.
