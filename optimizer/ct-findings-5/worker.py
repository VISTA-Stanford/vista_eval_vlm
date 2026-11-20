"""Runs the labeling program inside the sealed worker container.

The engine starts this with the program mounted read-only at /workspace/program, no network, as
user ``nobody``. It talks to the host over stdin/stdout, one JSON object per line:

    host -> worker   {"id": "...", "text": "..."}
    worker -> host   {"status": "ok", "id": "...", "label": "..."}
                     {"status": "error", "id": "...", "error": "<exception class>"}

The first line it writes is the handshake: {"status": "ready"}, or {"status": "load_error", ...} when
the program does not import. Write nothing else to stdout: a stray print breaks the protocol.
The dry run imports this same module on the host (``serve`` with other streams), so both paths run
the same code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROGRAM = Path("/workspace/program")
ENTRY = "label.py"
ENTRY_FUNCTION = "label"


def load(program_dir: Path):
    spec = importlib.util.spec_from_file_location("program_under_test", program_dir / ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, ENTRY_FUNCTION)


def answer(fn, request: dict) -> dict:
    try:
        out = fn(request["text"])
    except Exception as exc:  # noqa: BLE001 - a crash on one example is that example's answer
        return {"status": "error", "id": request.get("id"), "error": type(exc).__name__}
    if not isinstance(out, str):
        return {"status": "error", "id": request.get("id"), "error": "NotAString"}
    return {"status": "ok", "id": request.get("id"), "label": out}


def serve(program_dir: Path, stdin=sys.stdin, stdout=sys.stdout) -> None:
    def send(obj: dict) -> None:
        stdout.write(json.dumps(obj) + "\n")
        stdout.flush()

    try:
        fn = load(program_dir)
    except Exception as exc:  # noqa: BLE001 - the program not importing is a failed version, not a crash
        send({"status": "load_error", "error": type(exc).__name__})
        return
    send({"status": "ready"})
    for line in stdin:
        if line.strip():
            send(answer(fn, json.loads(line)))


if __name__ == "__main__":
    serve(PROGRAM)
