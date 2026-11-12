"""Stand-in sessions for the $0 dry run: the engine's ProgramRunner runs its real pass machinery (call folders,
staging, receipts, archive, answer parsing) but each "session" is this code, which writes an answer at once. Only
for synthetic data (scripts/check_task.py synth): it reads the labels to make answers whose scores are known.

Answer per row (both horizons), by kind(row) = sha256(row id) mod 10: 0-4 rank right (80 if the patient died by that
horizon, else 20); 5-8 say 30; 9 writes an unreadable answer (scored 0.5 and counted failed). The memory arm says its own
earlier 1-year risk plus 1 (so its change from the last point is exactly 1 point), which tests that the memory block
reached the prompt with the right numbers. Cost: FAKE_COST_USD a call (default $0.01).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import program

COST = float(os.environ.get("FAKE_COST_USD", "0.01"))


@dataclass
class Result:
    exit_code: int = 0
    cost_usd: float = COST
    stderr: str = ""
    raw_events: list = field(default_factory=list)
    model: str = "fake"
    session_id: str = "s"
    duration_seconds: float = 0.0
    log_dir: str | None = None
    container_name: str | None = None
    run_id: str | None = None


class Stack:
    def __init__(self, n: int) -> None:
        self.run_id = f"dry-net{n}"
        self.sidecar_name = f"alo-allowlist-squid-{self.run_id}"

    def __enter__(self):
        return self

    def close(self) -> None:
        pass

    def proxy_problem(self, hosts, *, client_leg=False):
        return None

    def squid_log_window(self, start, end):
        return []


def kind(row: str) -> int:
    return int(hashlib.sha256(row.encode()).hexdigest(), 16) % 10


MEMORY_LINE = re.compile(r"- On (\d{4}-\d{2}-\d{2}): chance of death within 1 year (\d+), within 2 years (\d+)\.")


class Session:
    def __init__(self, world: "World", description) -> None:
        self.world, self.description = world, description
        self.run_id = f"dry-{uuid.uuid4().hex[:6]}"

    def next_container_name(self, call_name):
        return f"alo-dry-{self.run_id}-{call_name}"

    @property
    def probe_container_name(self):
        return f"alo-dry-{self.run_id}-probe"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, prompt, *, call_name, host_env):
        row = call_name.rsplit("-", 1)[0]
        folder = Path(self.description.scope.writable[0][0])
        pass_id = re.search(r'"pass_id": "([^"]+)"', prompt).group(1)
        assert (folder / "record" / "meta.json").is_file(), "the record was not staged"
        with self.world.mu:
            self.world.prompts.setdefault(row, []).append(prompt)
        lab = program.labels()[row]
        k = kind(row)
        earlier = MEMORY_LINE.findall(prompt)
        if k == 9:
            body = {"died_within_1_year": "high", "died_within_2_years": 50, "reasons": [], "pass_id": pass_id}
        else:
            def say(field):
                g = lab[field]
                return (80 if g == 1 else 20) if k <= 4 else 30
            r1, r2 = say("label_1yr"), say("label_2yr")
            if earlier:  # memory arm: its own last 1-year risk + 1, at both horizons' positions
                r1 = min(100, int(earlier[-1][1]) + 1)
                r2 = max(r1, r2)
            body = {"died_within_1_year": r1, "died_within_2_years": r2,
                    "reasons": [{"fact": "made-up fact", "file": "notes.txt",
                                 "date": program.rows()[row]["cut"]}], "pass_id": pass_id}
        (folder / "findings.md").write_text("dry run\n")
        (folder / "output.json").write_text(json.dumps(body))
        return Result(container_name=self.next_container_name(call_name), run_id=self.run_id)


class World:
    """The fake Docker and session factory the runner's test hooks take."""

    def __init__(self) -> None:
        self.mu = threading.Lock()
        self.prompts: dict[str, list[str]] = {}
        self.n = 0

    def session(self, description, network_stack):
        return Session(self, description)

    def stack(self, description):
        self.n += 1
        return Stack(self.n)

    def kill(self, names, run_ids):
        pass

    def listed(self, names):
        return []
