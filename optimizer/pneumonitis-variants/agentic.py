"""The reader: one sealed Claude Code session per scan follows the program's instructions.md. Copied from ct-findings-5
(feat/ct-organ-zoom @6c1d699, the organ-zoom tool arm and the knowledge-base pre-drawn examples), arm switches removed.

Each call folder (the session's only writable mount) gets: ``ct/`` (ct.npz + meta.json), ``organs.json`` and
``organ_mask.npz`` (scripts/build_data.py masks), and ``drawn_examples/`` (stage_examples.py) drawn from the library of
the version being run. The session writes a findings note, then ``output.json`` with a 0-100 probability for each of
the five variants (stored 0..1) and up to ``MAX_REASONS`` reasons. Both viewers log each image to ``views.jsonl``.

The version whose library is drawn is the tree the pass runs (``set_tree`` before each pass: run.py val-pass and
scripts/run_loop.py do); a pass without it raises rather than staging a session with no examples.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config
from adapters import patient_dir, rows_for

ENTRY = "instructions.md"
OUTPUT = "output.json"
#: Part of the key that decides whether a finished pass can be reused. Bump it whenever ``prompt`` below changes.
PROMPT_TAG = "pneumonitis-variants-v1"
MAX_REASONS = 5
FINDINGS = "findings.md"
VIEW_LOG = "views.jsonl"
#: No Agent tool: a subagent would view the images again at full price.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
_TREE: Path | None = None


def set_tree(tree: Path) -> None:
    """The materialized program version this process's next passes run: its library is drawn beside each scan."""
    global _TREE
    _TREE = Path(tree)


def percent(value) -> float | None:
    """A stated chance 0-100 (a number, not a word or a bool) as 0..1, or None when unreadable or out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return value / 100 if 0 <= value <= 100 else None


def reasons_of(value) -> list[dict]:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            out.append({k: str(item.get(k) or "")[:300] for k in ("finding", "where")})
    return out[:MAX_REASONS]


def stage_scan(scan_id: str, folder: Path) -> None:
    """ct/, organs.json and organ_mask.npz for one scan, the last two read-only. Raises when any is missing."""
    dest = Path(folder) / "ct"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ct.npz", "meta.json"):
        shutil.copy2(patient_dir(scan_id) / name, dest / name)
    for src, name in ((config.DATA_DIR / "organs" / f"{scan_id}.json", "organs.json"),
                      (config.DATA_DIR / "masks" / f"{scan_id}.npz", "organ_mask.npz")):
        shutil.copy2(src, Path(folder) / name)
        (Path(folder) / name).chmod(0o444)


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    import stage_examples

    def calls_for(inputs):
        return [Call(row["id"], {"id": row["id"]}) for row in rows_for(inputs)]  # never the labels

    def stage(call, folder, pass_id):
        if _TREE is None:
            raise RuntimeError("agentic.set_tree was not called: no program version to draw examples from")
        stage_scan(call.payload["id"], Path(folder))
        stage_examples.stage(_TREE, Path(folder), call.payload["id"])

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not edited by the builder. Container paths only.
        questions = " ".join(f'({n}) "{key}": {text}' for n, (key, _t, text) in enumerate(config.QUESTIONS, 1))
        keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key, _t, _q in config.QUESTIONS)
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; its tools are in {paths.program}/tools/ (run "
            f"Python with python3 -B), its guide in {paths.program}/guide/ and its example library in "
            f"{paths.program}/examples/. The CT scan is the folder {paths.call}/ct/ (ct.npz and meta.json); beside it "
            f"are {paths.call}/organs.json, {paths.call}/organ_mask.npz and {paths.call}/drawn_examples/. "
            f"Answer five questions about the lung change on this scan: {questions} "
            f"Give each as the chance, from 0 to 100, that the answer is yes. "
            f"Scratch files, including images you render, go in /tmp. Before answering, write {paths.call}/{FINDINGS}: "
            f"a short note of what you looked at and what you found that bears on the answers. Use no names, dates "
            f"or other identifiers in it. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on "
            f'one line: {{{keys}, "reasons": [{{"finding": "<one thing you saw in the scan>", "where": "<where in '
            f'the scan, e.g. right upper lobe, both lower lobes, around the treated tumour>"}}, ...], '
            f'"pass_id": "{pass_id}"}}. '
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
            views = next((o["path"] for o in r.get("outputs", []) if o["rel"] == VIEW_LOG), None)
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
            answers[r["call_id"]] = {**said, "reason": reason, "error": error, "findings_log": log,
                                     "views_log": views, "reasons": reasons}
        manifest = Path(archive) / "answers.json"
        manifest.write_text(json.dumps({"answers": answers}), encoding="utf-8")
        return manifest

    return ProgramSpec(
        calls_for=calls_for, stage=stage, prompt=prompt, read_stamp=read_stamp, summarize=summarize,
        stages=("label",), output_patterns=(OUTPUT, FINDINGS, VIEW_LOG), prompt_tag=PROMPT_TAG,
        max_concurrent=config.PROGRAM_MAX_CONCURRENT, timeout_seconds=config.PROGRAM_CALL_TIMEOUT_SECONDS,
        max_budget_usd=config.PROGRAM_CALL_BUDGET_USD, forbidden=forbidden,
        token_env_name=None if config.ROUTE.is_vertex else config.TOKEN_ENV_NAME,
    )


def program_runner(layout, *, image: str, pin: str, forbidden):
    """The engine's ProgramRunner over this spec (as ct-findings-5)."""
    from isolation import sealed_session as ss
    from isolation.program_runner import ProgramRunner
    from isolation.session_scope import SessionScope

    profile, credential = config.ROUTE.session_auth()
    template = ss.SessionDescription(
        scope=SessionScope(program=layout.versions, denied=forbidden),
        posture="deny-by-default", tools=TOOLS, allowed_tools=TOOLS,
        model=config.PROGRAM_MODEL, image=image,
        profile=profile, credential=credential,
        transcript_dir=layout.transcripts / "program", expected_fingerprint=pin,
        )
    return ProgramRunner(spec=program_spec(forbidden), template=template, output_root=layout.program_outputs,
                         materialize_root=layout.versions)
