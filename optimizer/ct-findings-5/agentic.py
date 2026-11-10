"""The agentic program: the program is an instructions file (plus tools), and each scan is read by its own sealed
Claude Code session that follows it. The engine's ``ProgramRunner`` runs the sessions; this file says what one call
is. Copied from died-2yr-ct's probability mode (2026-11-10), with the five findings questions in place of the two
death horizons.

Each call: the scan (``ct.npz`` + ``meta.json``, from scripts/build_data.py) is copied to ``ct/`` in the call's own
folder (its only writable mount); the session reads ``instructions.md`` from the read-only program, writes a short
findings note, then ``output.json`` with a 0-100 probability for each of the five questions (stored as 0..1), plus up
to ``MAX_REASONS`` reasons: what in the scan moved the probabilities most, each naming where it was seen.
``summarize`` writes ``{"answers": {...}}`` (one probability per question, or None when unreadable), which the scorer,
the mistake builder and ``run.py test`` read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config
from adapters import patient_dir, rows_for

ENTRY = "instructions.md"
OUTPUT = "output.json"
#: Part of the key that decides whether a finished pass can be reused. Bump it whenever you change ``prompt`` below.
PROMPT_TAG = "ct-findings-5-v1-prob"
MAX_REASONS = 5
#: The session's reasoning is not kept in its transcript, so it writes what it saw; the optimizer reads it.
FINDINGS = "findings.md"
#: No Agent tool: a subagent would view the images again at full price.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")


def percent(value) -> float | None:
    """A stated chance 0-100 (a number, not a word or a bool) as 0..1, or None when unreadable or out of range. A
    failed answer is scored at the middle of the ranking (adapters.FAILED_PROBABILITY)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:  # NaN is not a number here
        return None
    return value / 100 if 0 <= value <= 100 else None


def reasons_of(value) -> list[dict]:
    """The stated reasons, kept as {"finding", "where"} strings (other keys dropped, non-objects skipped); never
    scored."""
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            out.append({k: str(item.get(k) or "")[:300] for k in ("finding", "where")})
    return out[:MAX_REASONS]


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    def calls_for(inputs):
        return [Call(row["id"], {"id": row["id"]}) for row in rows_for(inputs)]  # never the labels

    def stage(call, folder, pass_id):
        dest = Path(folder) / "ct"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("ct.npz", "meta.json"):
            shutil.copy2(patient_dir(call.payload["id"]) / name, dest / name)

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not optimized. Container paths only. It lists the five questions because the answer
        # keys fix them; what each means and how to look stays in instructions.md.
        questions = " ".join(f'({n}) "{key}": {text}' for n, (key, _t, text) in enumerate(config.QUESTIONS, 1))
        keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key, _t, _q in config.QUESTIONS)
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; its tools are in {paths.program}/tools/ (run "
            f"Python with python3 -B). The CT scan is the folder {paths.call}/ct/ (ct.npz and meta.json). "
            f"Answer five questions about this scan: {questions} "
            f"Give each as the chance, from 0 to 100, that the answer is yes. "
            f"Scratch files, including images you render, go in /tmp. Before answering, write {paths.call}/{FINDINGS}: "
            f"a short note of what you looked at and what you found that bears on the answers. Use no names, dates "
            f"or other identifiers in it. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on "
            f'one line: {{{keys}, "reasons": [{{"finding": "<one thing you saw in the scan>", "where": "<where in '
            f'the scan, e.g. right upper lobe, liver, mediastinal nodes>"}}, ...], "pass_id": "{pass_id}"}}. '
            f"Each number is your probability in percent, 0 to 100; a number outside that range, or a word, "
            f'counts as no answer. In "reasons" list the 3 to 5 things in the scan that moved your probabilities '
            f"most, most important first."
        )

    def read_stamp(call, folder):
        try:
            return json.loads((Path(folder) / OUTPUT).read_text(encoding="utf-8")).get("pass_id")
        except (OSError, ValueError):
            return None

    def summarize(archive, receipts):
        answers = {}
        for r in receipts:
            said, reason, error = {key: None for key, _t, _q in config.QUESTIONS}, None, r.get("failed")
            reasons: list[dict] = []
            out = next((o["path"] for o in r.get("outputs", []) if o["rel"] == OUTPUT), None)
            log = next((o["path"] for o in r.get("outputs", []) if o["rel"] == FINDINGS), None)
            if out and not error:
                try:
                    body = json.loads(Path(out).read_text(encoding="utf-8"))
                    said = {key: percent(body.get(key)) for key, _t, _q in config.QUESTIONS}
                    reasons = reasons_of(body.get("reasons"))
                    reason = "; ".join(f"{x['finding']} ({x['where']})" for x in reasons)[:500]
                    if None in said.values():
                        error = "not_a_probability"
                except (OSError, ValueError, AttributeError):
                    error = "unreadable_output"
            elif not error:
                error = "no_output"
            answers[r["call_id"]] = {**said, "reason": reason, "error": error, "findings_log": log, "reasons": reasons}
        manifest = Path(archive) / "answers.json"
        manifest.write_text(json.dumps({"answers": answers}), encoding="utf-8")
        return manifest

    return ProgramSpec(
        calls_for=calls_for, stage=stage, prompt=prompt, read_stamp=read_stamp, summarize=summarize,
        stages=("label",), output_patterns=(OUTPUT, FINDINGS), prompt_tag=PROMPT_TAG,
        max_concurrent=config.PROGRAM_MAX_CONCURRENT, timeout_seconds=config.PROGRAM_CALL_TIMEOUT_SECONDS,
        max_budget_usd=config.PROGRAM_CALL_BUDGET_USD, forbidden=forbidden,
        # On Vertex the sessions use the mounted optimizer login; the seat token is read only on the seat route.
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
