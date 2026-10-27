"""The agentic program mode: the program is an instructions file (plus tools), and each patient is read by its
own sealed Claude Code session that follows it. The engine's ``ProgramRunner`` runs the sessions; this file
says what one call is.

Each call: the patient's CT (``ct.npz`` + ``meta.json``, from scripts/build_data.py) is copied to ``ct/`` in the
call's own folder (its only writable mount); the session reads ``instructions.md`` from the read-only program,
writes a short findings note, then ``output.json`` with two forced answers, yes or no: did the patient die within 1
year, and within 2 years, of diagnosis; and a reason. ``summarize`` writes the same ``{"answers": {...}}`` file the
rules-based runner writes (one yes/no per horizon, True/False, or None when unreadable), so the scorer, the mistake
builder and ``run.py test`` read one format.
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
PROMPT_TAG = "died-2yr-ct-v2-yesno"
#: The session's reasoning is not kept in its transcript (thinking blocks arrive empty), so it writes what it
#: saw; the optimizer reads it beside the transcript for TRAIN mistakes.
FINDINGS = "findings.md"
#: No Agent tool: a subagent would view the images again at full price. No network but Claude; nothing mounted
#: but the program and this call's folder.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")


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
        dest = Path(folder) / "ct"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("ct.npz", "meta.json"):
            shutil.copy2(patient_dir(call.payload["id"]) / name, dest / name)

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not optimized. Container paths only. It names the two questions because the answer
        # keys fix them (2026-10-27); the method, and anything about death rates, stays in instructions.md.
        keys = ", ".join(f'"{key}": "yes" or "no"' for key, _gold, _words in config.HORIZONS)
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; its tools are in {paths.program}/tools/ (run "
            f"Python with python3 -B). The patient's CT is the folder {paths.call}/ct/ (ct.npz and meta.json). "
            f"The questions: did this patient die, of any cause, within 1 year of their cancer diagnosis? And within "
            f"2 years? Answer each yes or no. "
            f"Scratch files, including images you render, go in /tmp. Before answering, write {paths.call}/{FINDINGS}: "
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
