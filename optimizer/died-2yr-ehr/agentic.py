"""The agentic program mode: the program is an instructions file (plus tools), and each patient is read by its
own sealed Claude Code session that follows it. The engine's ``ProgramRunner`` runs the sessions; this file
says what one call is.

Each call: the patient's record folder (the shared exporter's layout: one parquet per OMOP table, ``notes.txt``,
``registry_<table>.parquet``, ``meta.json``; everything cut at first treatment - 7 days) is copied, files read-only,
to ``record/`` in the call's own folder (its only writable mount); the session reads ``instructions.md`` from the
read-only program, writes a short findings note, then ``output.json`` with two forced answers, yes or no: did the
patient die within 1 year, and within 2 years, of diagnosis; and a reason.
``summarize`` writes the same ``{"answers": {...}}`` file the rules-based runner writes (one yes/no per horizon,
True/False, or None when unreadable), so the scorer, the mistake builder and ``run.py test`` read one format.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config
from adapters import patient_dir, rows_for

ENTRY = "instructions.md"
OUTPUT = "output.json"
#: Part of the key that decides whether a finished pass can be reused. Bump it whenever you change
#: ``prompt`` below, or a pass run under the old wording is served again.
PROMPT_TAG = "died-2yr-ehr-v2-yesno"
#: The session's reasoning is not kept in its transcript (thinking blocks arrive empty), so it writes what it
#: saw; the optimizer reads it beside the transcript for TRAIN mistakes.
FINDINGS = "findings.md"
#: No Agent tool: a subagent would read the record again at full price. No network but Claude; nothing mounted
#: but the program and this call's folder.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
#: The session's copy of the record, inside its call folder.
RECORD = "record"
#: The share of TRAIN patients who died within each horizon (718 and 1,188 of 3,886), stated in the prompt (Phil
#: 2026-10-21: the cohort's death rate). From splits/train.jsonl; scripts/build_data.py prints it.
TRAIN_DEATH_RATE = {"1 year": "18.5%", "2 years": "30.6%"}
#: The OMOP tables the exporter writes (no death table), in the order the prompt lists them.
OMOP_TABLES = ("person", "observation_period", "visit_occurrence", "visit_detail", "condition_occurrence",
               "procedure_occurrence", "drug_exposure", "drug_era", "device_exposure", "episode", "image_occurrence",
               "measurement", "observation", "note")
REGISTRY_TABLES = ("diagnoses", "treatments", "outcomes", "events", "misc_details")
#: Every file a session may be given. The exporter also writes OUT/registry_dates/ (the raw governing dates its
#: leak check needs, i.e. the very dates the allow-list hides); that folder sits beside patients/, and this list is
#: what keeps it, or anything else stray, out of a session even if it ever lands in a patient folder.
RECORD_FILES = frozenset({f"{t}.parquet" for t in OMOP_TABLES} | {f"registry_{t}.parquet" for t in REGISTRY_TABLES}
                         | {"notes.txt", "meta.json"})


def record_files(src: Path) -> list[Path]:
    """The files of one patient folder that may be staged, after checking the folder holds nothing else.
    Raises on a folder outside DATA_DIR/patients/, a subfolder, or any file not in RECORD_FILES."""
    src = Path(src)
    if src.resolve().parent != (config.DATA_DIR / "patients").resolve():
        raise RuntimeError("a record folder must be DATA_DIR/patients/<id>")
    files = sorted(src.iterdir())
    stray = [f for f in files if f.is_dir() or f.name not in RECORD_FILES]
    if stray:
        raise RuntimeError(f"refusing to stage a record folder with {len(stray)} unexpected entries "
                           f"(first: {stray[0].name}); only the exporter's patient files may reach a session")
    return files


def yes_no(value) -> bool | None:
    """True for "yes", False for "no" (any case, surrounding spaces ignored), None for anything else: a forced
    answer is one of the two words, and anything else is scored wrong (eval-set spec, Decided: Score)."""
    word = value.strip().lower() if isinstance(value, str) else None
    return {"yes": True, "no": False}.get(word)


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    def calls_for(inputs):
        return [Call(row["id"], {"id": row["id"]}) for row in rows_for(inputs)]  # never the label

    def stage(call, folder, pass_id):
        # Files read-only so the session cannot edit its evidence; the folder stays writable so a resume can
        # clear it.
        dest = Path(folder) / RECORD
        dest.mkdir(parents=True)
        # Readable by others: on Vertex the container runs as the credential's owner, not the host user, and a
        # 077 umask left the record unreadable there (live check 2026-10-21).
        dest.chmod(0o755)
        for f in record_files(patient_dir(call.payload["id"])):
            shutil.copy2(f, dest / f.name)
            (dest / f.name).chmod(0o444)

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not optimized. Container paths only. The facts every version needs (the question,
        # the cutoff, the folder layout, the death rate) live here; the method lives in instructions.md.
        rec = f"{paths.call}/{RECORD}"
        keys = ", ".join(f'"{key}": "yes" or "no"' for key, _gold, _words in config.HORIZONS)
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; any tools are in {paths.program}/tools/ (run "
            f"Python with python3 -B).\n\n"
            f"The questions: did this patient die, of any cause, within 1 year of their cancer diagnosis? And within "
            f"2 years? Answer each yes or no. The patient has a thoracic cancer. Their medical record is the folder "
            f"{rec}/ and it stops 7 days before their first cancer treatment: nothing dated after that day is in it. "
            f"Of the patients in this group, {TRAIN_DEATH_RATE['1 year']} died within 1 year and "
            f"{TRAIN_DEATH_RATE['2 years']} within 2 years.\n\n"
            f"The folder (read-only):\n"
            f"- notes.txt: every clinical note in date order, each starting with a header line that begins "
            f"'===== NOTE'. Search it with grep.\n"
            f"- one parquet file per table of the OMOP common data model: {', '.join(OMOP_TABLES)}. A table with "
            f"no rows for this patient may be absent.\n"
            f"- {', '.join(f'registry_{t}.parquet' for t in REGISTRY_TABLES)}: the cancer registry's record of "
            f"this cancer, limited to the fields allowed at that cutoff.\n"
            f"- meta.json: the patient's opaque id and the cutoff date.\n"
            f"python3 has pandas, pyarrow and duckdb for the parquet files.\n\n"
            f"Scratch files go in /tmp. Before answering, write {paths.call}/{FINDINGS}: "
            f"a short note of what you looked at and what you found that bears on the answer. Use no names, dates "
            f"or other identifiers in it. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on "
            f'one line: {{{keys}, "reason": "<one line: the findings you relied on>", "pass_id": "{pass_id}"}}. '
            f"Each answer must be exactly yes or no: no probabilities, no hedging; an answer that is not one of the "
            f"two words is scored as wrong."
        )

    def read_stamp(call, folder):
        try:
            return json.loads((Path(folder) / OUTPUT).read_text(encoding="utf-8")).get("pass_id")
        except (OSError, ValueError):
            return None

    def summarize(archive, receipts):
        answers = {}
        for r in receipts:
            said, reason, error = {key: None for key, _g, _w in config.HORIZONS}, None, r.get("failed")
            out = next((o["path"] for o in r.get("outputs", []) if o["rel"] == OUTPUT), None)
            log = next((o["path"] for o in r.get("outputs", []) if o["rel"] == FINDINGS), None)
            if out and not error:
                try:
                    body = json.loads(Path(out).read_text(encoding="utf-8"))
                    said = {key: yes_no(body.get(key)) for key, _g, _w in config.HORIZONS}
                    reason = str(body.get("reason") or "")[:500]
                    if None in said.values():
                        error = "not_yes_or_no"
                except (OSError, ValueError, AttributeError):
                    error = "unreadable_output"
            elif not error:
                error = "no_output"
            answers[r["call_id"]] = {**said, "reason": reason, "error": error, "findings_log": log}
        manifest = Path(archive) / "answers.json"
        manifest.write_text(json.dumps({"answers": answers}), encoding="utf-8")
        return manifest

    return ProgramSpec(
        calls_for=calls_for, stage=stage, prompt=prompt, read_stamp=read_stamp, summarize=summarize,
        stages=("label",), output_patterns=(OUTPUT, FINDINGS), prompt_tag=PROMPT_TAG,
        max_concurrent=config.PROGRAM_MAX_CONCURRENT, timeout_seconds=config.PROGRAM_CALL_TIMEOUT_SECONDS,
        max_budget_usd=config.PROGRAM_CALL_BUDGET_USD, forbidden=forbidden,
        # On Vertex the sessions use the mounted key; the seat token is read only on the seat route.
        token_env_name=None if config.ROUTE.is_vertex else config.TOKEN_ENV_NAME,
    )


def program_runner(layout, *, image: str, pin: str, forbidden):
    """The engine's ProgramRunner over this spec. Pass it to the driver as is; the driver enters it."""
    from isolation import sealed_session as ss
    from isolation.program_runner import ProgramRunner
    from isolation.session_scope import SessionScope

    profile, credential = config.ROUTE.session_auth()
    template = ss.SessionDescription(
        # A placeholder grant: the runner builds each call's real one. denied may not be empty.
        scope=SessionScope(program=layout.versions, denied=forbidden),
        posture="deny-by-default", tools=TOOLS, allowed_tools=TOOLS,
        model=config.PROGRAM_MODEL, image=image,
        profile=profile, credential=credential,
        transcript_dir=layout.transcripts / "program", expected_fingerprint=pin,
    )
    return ProgramRunner(spec=program_spec(forbidden), template=template, output_root=layout.program_outputs,
                         materialize_root=layout.versions)
