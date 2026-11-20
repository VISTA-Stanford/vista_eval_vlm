"""The agentic program: the program is an instructions file (plus tools), and each patient is read by its own sealed
Claude Code session that follows it. The engine's ``ProgramRunner`` runs the sessions; this file says what one call
is. Copied from ct-findings-5 (2026-11-20), with died-2yr-ehr's summary staging for the treatment agent.

Each call gets, in its own folder (its only writable mount):
    treatment  summary.txt: the tumor board-time summary, read-only (also quoted in the prompt, as the eval-set card does)
    imaging    ct/prior/ and ct/current/, each ct.npz + meta.json (scripts/build_data.py volumes)
The session reads ``instructions.md`` from the read-only program, writes a short findings note, then ``output.json``
with an answer for each question the tumor board answered for this patient (the wrapper names them): one 0-100
probability of yes (yes/no question), or one 0-100 probability per letter A-E (several-option question), plus up to
``MAX_REASONS`` reasons. ``summarize`` writes ``{"answers": {...}}``, which the scorer, the mistake builder and
``run.py test`` read: a yes/no answer stored 0..1, a several-option answer as {label code: 0..1} rescaled to sum to 1.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import config
from adapters import QUESTION, asked, asked_of, patient_dir, rows_for

ENTRY = "instructions.md"
OUTPUT = "output.json"
#: Part of the key that decides whether a finished pass can be reused. Bump it whenever you change ``prompt`` below.
PROMPT_TAG = f"tb-questions-{config.AGENT}-v1"
MAX_REASONS = 5
#: The session's reasoning is not kept in its transcript, so it writes what it used; the optimizer reads it.
FINDINGS = "findings.md"
SUMMARY = "summary.txt"
SCANS = ("prior", "current")
#: No Agent tool: a subagent would read the inputs again at full price.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")


def percent(value) -> float | None:
    """A stated chance 0-100 (a number, not a word or a bool) as 0..1, or None when unreadable or out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value / 100 if 0 <= value <= 100 else None


def options_of(value, options) -> dict[str, float] | None:
    """A several-option answer {"A": n, ..., "E": n} as {label code (str): 0..1} rescaled to sum to 1, whatever the
    stated total (Phil 2026-11-20); None when it is not an object, a letter is missing, a value is not a number from 0
    to 100, or every value is 0 (the only unreadable total). Letters are matched ignoring case and spaces; other keys
    are ignored."""
    if not isinstance(value, dict):
        return None
    norm = {str(k).strip().upper(): v for k, v in value.items()}
    raw = {}
    for letter, code, _text in options:
        p = percent(norm.get(letter))
        if p is None:
            return None
        raw[str(code)] = p
    total = sum(raw.values())
    if total <= 0:
        return None
    return {c: v / total for c, v in raw.items()}


def reasons_of(value) -> list[dict]:
    """The stated reasons as {"fact", "where"} strings (other keys dropped, non-objects skipped); never scored."""
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            out.append({k: str(item.get(k) or "")[:300] for k in ("fact", "where")})
    return out[:MAX_REASONS]


def answer_template(keys: list[str]) -> str:
    """The JSON shape the wrapper asks for, one entry per question asked."""
    parts = []
    for key in keys:
        q = QUESTION[key]
        if q.options:
            parts.append(f'"{key}": {{' + ", ".join(f'"{letter}": <0-100>' for letter, _c, _t in q.options) + "}")
        else:
            parts.append(f'"{key}": <a whole number from 0 to 100>')
    return ", ".join(parts)


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    def calls_for(inputs):
        # Which questions to ask comes from which ones the tumor board answered (as the eval-set card asks them); the
        # answers themselves never leave the host.
        return [Call(row["id"], {"id": row["id"], "asked": asked(row)}) for row in rows_for(inputs)]

    def stage(call, folder, pass_id):
        # Leave the call folder's own permissions as the runner set them (on Vertex it is 0o777 so the container user
        # can write; died-2yr-ehr 2026-11-01). Staged files are made read-only and readable by others.
        src = patient_dir(call.payload["id"])
        if config.TREATMENT:
            shutil.copy2(src / SUMMARY, Path(folder) / SUMMARY)
            (Path(folder) / SUMMARY).chmod(0o444)
            return
        for scan in SCANS:
            dest = Path(folder) / "ct" / scan
            dest.mkdir(parents=True, exist_ok=True)
            dest.parent.chmod(0o755)
            dest.chmod(0o755)
            for name in ("ct.npz", "meta.json"):
                shutil.copy2(src / scan / name, dest / name)
                (dest / name).chmod(0o444)

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not optimized. Container paths only. It quotes the card's question(s) verbatim, so
        # every version asks the same thing; how to answer them lives in instructions.md.
        keys = call.payload["asked"]
        questions = "\n\n".join(f'[{key}]\n{QUESTION[key].text}' for key in keys)
        if config.TREATMENT:
            summary = (patient_dir(call.payload["id"]) / SUMMARY).read_text(encoding="utf-8").strip()
            given = (f"{config.SUMMARY_OPENING}\n\nPATIENT SUMMARY:\n{summary}\n\n"
                     f"The summary is also the file {paths.call}/{SUMMARY}.")
            reason = '{"fact": "<one fact from the summary>", "where": "<the part of the summary it is in>"}'
        else:
            given = (f"{config.IMAGING_OPENING} The PRIOR CT is the folder {paths.call}/ct/prior/ and the MOST RECENT "
                     f"CT is {paths.call}/ct/current/ (each ct.npz and meta.json).")
            reason = ('{"fact": "<one thing you saw, comparing the two scans>", "where": "<where in the scan, e.g. '
                      'right upper lobe, liver, mediastinal nodes>"}')
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; any tools are in {paths.program}/tools/ (run Python "
            f"with python3 -B).\n\n{given}\n\n"
            f"Answer {'this question' if len(keys) == 1 else 'these questions'} (the key in brackets is the answer's "
            f"name):\n\n{questions}\n\n"
            f"Scratch files, including any images you render, go in /tmp. Before answering, write "
            f"{paths.call}/{FINDINGS}: a short note of what you used and what bears on the answer. Use no names, "
            f"dates or other identifiers in it. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on "
            f'one line: {{{answer_template(keys)}, "reasons": [{reason}, ...], "pass_id": "{pass_id}"}}. '
            f"Every number is a probability in percent, 0 to 100; a number outside that range, a word, or a missing "
            f"letter counts as no answer. In \"reasons\" list the 3 to 5 things that moved your probabilities most, "
            f"most important first."
        )

    def read_stamp(call, folder):
        try:
            return json.loads((Path(folder) / OUTPUT).read_text(encoding="utf-8")).get("pass_id")
        except (OSError, ValueError):
            return None

    def summarize(archive, receipts):
        answers = {}
        for r in receipts:
            said, reason, error = {q.key: None for q in config.QUESTIONS}, None, r.get("failed")
            reasons: list[dict] = []
            out = next((o["path"] for o in r.get("outputs", []) if o["rel"] == OUTPUT), None)
            log = next((o["path"] for o in r.get("outputs", []) if o["rel"] == FINDINGS), None)
            keys = asked_of(r["call_id"])  # receipts do not carry the payload; the split says what was asked
            if out and not error:
                try:
                    body = json.loads(Path(out).read_text(encoding="utf-8"))
                    for key in keys:
                        q = QUESTION[key]
                        said[key] = options_of(body.get(key), q.options) if q.options else percent(body.get(key))
                    reasons = reasons_of(body.get("reasons"))
                    reason = "; ".join(f"{x['fact']} ({x['where']})" for x in reasons)[:500]
                    if not keys or any(said[k] is None for k in keys):
                        error = "not_a_probability"
                except (OSError, ValueError, AttributeError, TypeError):
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
