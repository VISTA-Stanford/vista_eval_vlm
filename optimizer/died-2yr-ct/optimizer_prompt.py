"""The optimizer's prompts. Three pieces, each with one job:

    prompts/optimizer_instructions.md   how to work, every iteration (the session's system prompt)
    build_prompt(iter_n, paths)         this iteration: which one, where everything is, what to write back
    context/task.md                     the task itself, read by the optimizer as a file

Name container paths only, from ``paths`` (an ``isolation.contained_agent.OptimizerPaths``). A host
path in a prompt points at nothing inside the container, and leaks your folder layout.

    python run.py print-prompt --iter 2     shows exactly what the optimizer receives
"""

from __future__ import annotations

from pathlib import Path

from config import AGENTIC, EDITABLE_FILES, ITERATIONS, TASK_BRIEF  # run.py sets ITERATIONS to the run's own count

HERE = Path(__file__).resolve().parent
INSTRUCTIONS = HERE / "prompts" / "optimizer_instructions.md"

#: Read-only files mounted into the optimizer's container, by name (see optimizer.build_agent).
TASK_BRIEF_NAME = "task.md"
NOTEBOOK_NAME = "notebook.md"
CHANGE_NOTE_NAME = "change_note.json"


#: TASK-SPECIFIC: names the toy's entry point (label.label) and mistake fields (gold, text).
#: It counts rather than prints, so it stays usable when the mistakes file holds thousands of examples.
TEST_RULES_BASED = (
    'To check your edit, run it over the mistakes file with `python3 -B` (a `__pycache__` folder left in\n'
    '`{program}` refuses your whole edit). It prints how many of the current mistakes the edited program\n'
    'now gets right, per gold label, and its speed:\n'
    '`cd {program} && python3 -B -c "import json, glob, time, collections, label; '
    'ms = [m for f in glob.glob(\'{feedback}/iter/{iter_n}/files/*.json\') for m in json.load(open(f))]; '
    't = time.time(); c = collections.Counter((m[\'gold\'], label.label(m[\'text\']) == m[\'gold\']) for m in ms); '
    'print(sorted(c.items()), round((time.time() - t) / max(len(ms), 1) * 1000, 1), \'ms an example\')"`\n'
    'It only covers examples the current version gets wrong: it cannot show examples your edit newly breaks.\n'
)


def system_prompt() -> str:
    return INSTRUCTIONS.read_text(encoding="utf-8")


def build_prompt(iter_n: int, paths) -> str:
    editable = ", ".join(f"`{paths.program}/{f}`" for f in EDITABLE_FILES)
    first = (
        ("This is the first iteration. The program is the seed: the plain question and a slice viewer, so its "
         "score is what a clinician-style prompt gets with no method. Read the task brief, read the mistakes, look "
         "at some of those CTs yourself, and write a real first version: instructions.md and the tools it needs."
         if AGENTIC else
         "This is the first iteration: the program is the seed, so expect a low score. Read the task brief, "
         "read the mistakes, and build a real first version rather than a one-line patch.")
        if iter_n == 1 else
        "Start with step 1 of your standing instructions: check last iteration's prediction."
    )
    test = (
        ("You cannot run the patients' sessions in here: each patient is read by its own session, outside\n"
         "this container. You can read what each session did, look at the same CT, and run your tools on it.\n"
         "Every file a mistake lists is a flat file in `<files>/`, named `<id>-<file>`.\n"
         "- Read a session's findings note first: `<files>/<id>-trace-findings.md`. Then its session turn by turn\n"
         "  (each command and the start of its output):\n"
         "  `python3 -c \"import json,sys; [print(b['type'], str(b.get('input') or b.get('content') or b.get('text'))[:400].replace(chr(10), ' ')) for l in open(sys.argv[1]) for e in [json.loads(l)] if e.get('type') in ('assistant','user') and isinstance(e['message']['content'], list) for b in e['message']['content'] if b.get('type') != 'thinking']\" <files>/<id>-trace-events.jsonl`\n"
         "- Give a tool the folder a session sees: `mkdir -p /tmp/ct/<id> && cp <files>/<id>-ct.npz /tmp/ct/<id>/ct.npz\n"
         "  && cp <files>/<id>-meta.json /tmp/ct/<id>/meta.json`, then\n"
         "  `python3 -B {program}/tools/view_ct.py /tmp/ct/<id> --plane axial --count 16 --window lung --out /tmp/v.png`\n"
         "  and open `/tmp/v.png` with Read. Each image costs tokens: look at a handful of patients, not all.\n"
         "- Every TRAIN patient of the pass is in the mistakes file, right or wrong (`mistake` says which), with\n"
         "  its probability: read the right ones' reasons too, since the score is how well the ranking separates\n"
         "  the two groups. Only mistakes come with their CT and transcript.\n"
         .replace("{program}", paths.program).replace("<files>", f"{paths.feedback}/iter/{iter_n}/files"))
        if AGENTIC else
        TEST_RULES_BASED.format(program=paths.program, feedback=paths.feedback, iter_n=iter_n)
    )
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

Not mounted at all: the VAL and TEST patients and their outcomes, the scorer, the run's records and
any credentials. A path outside the folders above does not exist here.

{test}"""
