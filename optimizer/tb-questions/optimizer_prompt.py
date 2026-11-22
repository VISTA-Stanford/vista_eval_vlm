"""The optimizer's prompts. Three pieces, each with one job:

    prompts/optimizer_instructions.md   how to work, every iteration (the session's system prompt)
    build_prompt(iter_n, paths)         this iteration: which one, where everything is, what to write back
    context/task.md                     the task itself, read by the optimizer as a file

Name container paths only, from ``paths`` (an ``isolation.contained_agent.OptimizerPaths``). A host
path in a prompt points at nothing inside the container, and leaks your folder layout.

    python run.py print-prompt --iter 2     shows exactly what the optimizer receives
"""

from __future__ import annotations

import os
from pathlib import Path

from config import AGENTIC, EDITABLE_FILES, ITERATIONS, OPTIMIZER_INSTRUCTIONS, SEARCH_PROXY, SEARCH_REQUESTS_PER_ROUND, TASK_BRIEF, TREATMENT, BOARDS  # run.py sets ITERATIONS

HERE = Path(__file__).resolve().parent
INSTRUCTIONS = OPTIMIZER_INSTRUCTIONS

#: Read-only files mounted into the optimizer's container, by name (see optimizer.build_agent).
TASK_BRIEF_NAME = "task.md"
NOTEBOOK_NAME = "notebook.md"
CHANGE_NOTE_NAME = "change_note.json"


def system_prompt() -> str:
    """The standing instructions, plus how to ask the search proxy when it is on (the engine's own wording)."""
    text = INSTRUCTIONS.read_text(encoding="utf-8")
    if SEARCH_PROXY:
        from engine.search_proxy import optimizer_instructions

        text = f"{text.rstrip()}\n\n{optimizer_instructions(SEARCH_REQUESTS_PER_ROUND)}"
    return text


#: How the optimizer studies a TRAIN mistake, per agent. ``<files>`` and ``{program}`` are filled in below.
STUDY_TREATMENT = (
    "You cannot run the patients' sessions in here: each patient is read by its own session, outside this container.\n"
    "You can read what each session did and the summary it read. Every file a mistake lists is a flat file in\n"
    "`<files>/`, named `<id>-<file>`.\n"
    "- The summary the session read: `<files>/<id>-summary.txt`.\n"
    "- The session's findings note first: `<files>/<id>-trace-findings.md`. Then its session turn by turn:\n"
    "  `python3 -c \"import json,sys; [print(b['type'], str(b.get('input') or b.get('content') or b.get('text'))[:400].replace(chr(10), ' ')) for l in open(sys.argv[1]) for e in [json.loads(l)] if e.get('type') in ('assistant','user') and isinstance(e['message']['content'], list) for b in e['message']['content'] if b.get('type') != 'thinking']\" <files>/<id>-trace-events.jsonl`\n"
    "- Every TRAIN patient of the pass is in the mistakes file: the question(s) asked (`asked`), the tumor board's\n"
    "  answer as a letter (`tumor_board_answer`), the probabilities the session gave each letter (`answer`) and its\n"
    "  stated reasons. A mistake is a question whose top letter was not the tumor board's (`wrong_at`), or a failed\n"
    "  session. The score is per-letter AUROC: for each letter, patients whose answer was that letter should get a\n"
    "  higher probability on it than patients whose answer was another. So a confident wrong letter costs most, and so\n"
    "  does giving every patient the same spread. Only mistakes come with their summary and transcript.\n"
    "- Never copy summary text, or anything particular to one patient, into the program or your notes.\n"
)
STUDY_BOARDS = (
    "You cannot run the sessions in here: each tumor board encounter is read by its own session, outside this\n"
    "container, from the patient's record cut one week before that tumor board. You can read what each session did and the\n"
    "record it read. Every file a mistake lists is a flat file in `<files>/`, named `<id>-<file>`.\n"
    "- The record the session read: `<files>/<id>-notes.txt`, `<files>/<id>-<table>.parquet`, `<files>/<id>-meta.json`\n"
    "  (notes are left out when they are over 2 MB). Run `python3 -B {program}/tools/digest.py` on a folder holding\n"
    "  them under their own names to see what the notes reader shows.\n"
    "- The session's findings note first: `<files>/<id>-trace-findings.md`. Then its session turn by turn:\n"
    "  `python3 -c \"import json,sys; [print(b['type'], str(b.get('input') or b.get('content') or b.get('text'))[:400].replace(chr(10), ' ')) for l in open(sys.argv[1]) for e in [json.loads(l)] if e.get('type') in ('assistant','user') and isinstance(e['message']['content'], list) for b in e['message']['content'] if b.get('type') != 'thinking']\" <files>/<id>-trace-events.jsonl`\n"
    "- Every TRAIN encounter of the pass is in the mistakes file: the questions asked (`asked`; every encounter gets\n"
    "  decision_type and intent, some also one treatment question), the tumor board's answer as a letter\n"
    "  (`tumor_board_answer`), the probabilities the session gave each letter (`answer`) and its stated reasons. A\n"
    "  mistake is a question whose top letter was not the tumor board's (`wrong_at`), or a failed session. The score is\n"
    "  per-letter AUROC averaged over decision_type, intent and early_stage_plan: for each letter, encounters whose answer\n"
    "  was that letter should get a higher probability on it than encounters whose answer was another. So a confident wrong\n"
    "  letter costs most, and so does giving every encounter the same spread. Only mistakes come with their record and\n"
    "  transcript.\n"
    "- Never copy note text, or anything particular to one patient, into the program or your notes.\n"
)
STUDY_IMAGING = (
    "You cannot run the patients' sessions in here: each pair of scans is read by its own session, outside this\n"
    "container. You can read what each session did, look at the same two CTs, and run your tools on them. Every file\n"
    "a mistake lists is a flat file in `<files>/`, named `<id>-<file>`.\n"
    "- Read a session's findings note first: `<files>/<id>-trace-findings.md`. Then its session turn by turn:\n"
    "  `python3 -c \"import json,sys; [print(b['type'], str(b.get('input') or b.get('content') or b.get('text'))[:400].replace(chr(10), ' ')) for l in open(sys.argv[1]) for e in [json.loads(l)] if e.get('type') in ('assistant','user') and isinstance(e['message']['content'], list) for b in e['message']['content'] if b.get('type') != 'thinking']\" <files>/<id>-trace-events.jsonl`\n"
    "- Give a tool the folder a session sees: `mkdir -p /tmp/ct/<id>/prior /tmp/ct/<id>/current && for s in prior\n"
    "  current; do cp <files>/<id>-$s-ct.npz /tmp/ct/<id>/$s/ct.npz; cp <files>/<id>-$s-meta.json /tmp/ct/<id>/$s/meta.json;\n"
    "  done`, then `python3 -B {program}/tools/view_ct.py /tmp/ct/<id> --plane axial --count 8 --window lung --out\n"
    "  /tmp/v.png` and open `/tmp/v.png` with Read. Each image costs tokens: look at a handful of pairs, not all.\n"
    "- Every TRAIN pair of the pass is in the mistakes file with the tumor board's answer (`tumor_board_answer`: 1\n"
    "  progression, 0 response or stable), the probability the session gave (`answer`, 0-100) and its stated reasons.\n"
    "  A mistake is a probability on the wrong side of the TRAIN progression rate, or a failed session. AUROC rewards\n"
    "  ranking every progression pair above every response/stable pair, so a confident wrong probability costs most:\n"
    "  look at the most confident wrong ones first, and at the spread of probabilities overall. Only mistakes come\n"
    "  with their CTs and transcript.\n"
)


def build_prompt(iter_n: int, paths) -> str:
    editable = ", ".join(f"`{paths.program}/{f}`" for f in EDITABLE_FILES)
    start = os.environ.get("START", "")  # the launcher's START: a fresh run from an earlier climb's version
    what = "tumor board encounters" if BOARDS else "patients" if TREATMENT else "pairs of scans"
    if iter_n > 1:
        first = "Start with step 1 of your standing instructions: check last iteration's prediction."
    elif start and start != "seed":
        first = (f"This is the first iteration of a new run that starts from {start}, the best version of an earlier "
                 f"climb on these same VAL {what}, not from the seed. Read your notebook and the earlier change notes "
                 "(notes folder) first: they say what was tried and what each version scored. Then improve on this "
                 "version.")
    elif BOARDS:
        first = ("This is the first iteration. The program is the seed: how to find the record, the notes reader and the "
                 "tumor board questions word for word, so its score is what a plain zero-shot prompt gets. Read the task "
                 "brief, read the mistakes and some of their records, and write a real first version of "
                 "instructions.md: how to read a record for these decisions and how these tumor boards decided (see "
                 "step 4 of your standing instructions).")
    elif TREATMENT:
        first = ("This is the first iteration. The program is the seed: the tumor board question(s) word for word and "
                 "nothing else, so its score is what a plain zero-shot prompt gets. Read the task brief, read the "
                 "mistakes and some of their summaries, and write a real first version of instructions.md: how to read "
                 "a summary for this decision and how these tumor boards decided (see step 4 of your standing "
                 "instructions).")
    else:
        first = ("This is the first iteration. The program is the seed: the tumor board question word for word and a "
                 "viewer for the two scans, so its score is what a plain zero-shot prompt gets. Read the task brief, "
                 "read the mistakes, look at some of those pairs yourself, and write a real first version of "
                 "instructions.md: how to compare the two scans and what these tumor boards called progression (see "
                 "step 4 of your standing instructions).")
    test = ((STUDY_BOARDS if BOARDS else STUDY_TREATMENT if TREATMENT else STUDY_IMAGING)
            .replace("{program}", paths.program).replace("<files>", f"{paths.feedback}/iter/{iter_n}/files"))
    return f"""Iteration {iter_n} of {ITERATIONS}. {first}

## Where things are

You are in a sealed container. Only these exist for you:

- {editable}: the program, a copy of this iteration's version. Edit these and nothing else in
  `{paths.program}` (your working folder). Any other new or changed file there stops the run.
- `{paths.inputs[TASK_BRIEF_NAME]}`: the task brief. Read it before your first edit.
- `{paths.inputs[NOTEBOOK_NAME]}`: your notebook as it stands. Copy it to
  `{paths.outputs[NOTEBOOK_NAME]}` first, then edit it there.
- `{paths.outputs[CHANGE_NOTE_NAME]}`: write this iteration's change note here.
- `{paths.feedback}/iter/{iter_n}/release_train.json`: this iteration's TRAIN score, breakdown,
  best-so-far and spend. Its `corpus.mistakes_file` names the mistakes file, under
  `{paths.feedback}/iter/{iter_n}/files/`.
- `{paths.feedback}/iter/<k>/release_val.json`: the VAL score of every iteration so far.
  `{paths.feedback}/iter/<k>/previous_attempt.json` appears when the version before iteration k failed.
- `{paths.feedback}/run_summary.json`: every version so far, its tag and its VAL score.
- `{paths.notes}/`: your earlier notes and change notes, one folder per run and iteration.
- `{paths.versions}/`: earlier program versions, as plain folders. There is no git here.

Not mounted at all: the VAL and TEST patients and their answers, the scorer, the run's records and
any credentials. A path outside the folders above does not exist here.

{test}"""
