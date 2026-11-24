"""How one knowledge-base tester session used the library, from its transcript (events.jsonl): the entries it opened
(``kb.py open``), the example images it read, ``kb.py list`` calls, and how often it drew its own scan after its first
open (going back and forth). Standard library only; used by scripts/run_kb_loop.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ENTRY_OPEN = re.compile(r"kb\.py\s+open\s+([a-z0-9][a-z0-9-]*)")
OWN_VIEW = re.compile(r"view_ct\.py\s+(\S*/)?ct/?(\s|$)")


def usage(events: Path | None) -> dict:
    """From one tester session's events: which entries it opened (kb.py open), how many example images it read, and how
    many times it drew its own scan after opening an entry (back and forth)."""
    opened, images_read, own_after, lists, staged_read, index_read = [], 0, 0, 0, set(), False
    if events is None or not events.is_file():
        return {"entries_opened": [], "example_images_read": 0, "own_views_after_an_open": 0, "list_calls": 0,
                "staged_examples_read": 0, "read_examples_index": False, "transcript": False}
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "assistant":
            continue
        for b in e.get("message", {}).get("content", []) or []:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            inp = b.get("input") or {}
            if b.get("name") == "Bash":
                cmd = str(inp.get("command", ""))
                hits = ENTRY_OPEN.findall(cmd)
                opened += hits
                lists += len(re.findall(r"kb\.py\s+list", cmd))
                if opened and not hits and OWN_VIEW.search(cmd):
                    own_after += 1
            elif b.get("name") == "Read":
                path = str(inp.get("file_path", ""))
                if re.search(r"-view\d+\.png$", path):
                    images_read += 1
                if re.search(r"/examples/[^/]+\.png$", path):  # the pre-drawn examples (KB_PRESTAGE), distinct files
                    staged_read.add(path.rsplit("/", 1)[-1])
                if path.endswith("/examples/index.md"):
                    index_read = True
    return {"entries_opened": sorted(set(opened)), "example_images_read": images_read,
            "own_views_after_an_open": own_after, "list_calls": lists, "staged_examples_read": len(staged_read),
            "read_examples_index": index_read, "transcript": True}
