"""How reader sessions used the guide, the drawn examples and the viewers, from their transcripts (events.jsonl).
Standard library only for ``usage``; the CLI also needs the task config (run from the engine's environment).

    python -B scripts/usage.py ANSWERS_JSON --tree VERSION_TREE [--out F]

Per pass: sessions, how many drawn example images each opened (distinct files under drawn_examples/ read with Read)
against how many were staged (stage_examples.planned on the version tree), the share that opened every one (the
spec's stop rule: under half on the guide arm's cost check stops the study), the share that read the guide, and the
mean number of zoomed and grid images drawn (views.jsonl). Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DRAWN = re.compile(r"/drawn_examples/([^/]+\.png)$")


def usage(events: Path | None) -> dict:
    opened, guide, ok = set(), False, bool(events and Path(events).is_file())
    for line in Path(events).read_text(encoding="utf-8").splitlines() if ok else []:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "assistant":
            continue
        for b in e.get("message", {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Read":
                path = str((b.get("input") or {}).get("file_path", ""))
                m = DRAWN.search(path)
                if m:
                    opened.add(m.group(1))
                if re.search(r"/guide/[^/]+\.md$", path):
                    guide = True
    return {"drawn_opened": len(opened), "read_guide": guide, "transcript": ok}


def views(log: str | None) -> dict:
    rows = []
    if log and Path(log).is_file():
        rows = [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"zoom": sum(r.get("tool") == "zoom_organ" and not r.get("refused") for r in rows),
            "grid": sum(r.get("tool") == "view_ct" for r in rows)}


def trace_events(answers: Path, call_id: str) -> Path | None:
    root = Path(answers).parent / "transcripts" / call_id
    attempts = sorted(root.glob("attempt*"), key=lambda p: int(p.name.removeprefix("attempt") or 0))
    runs = sorted(attempts[-1].glob("*/*")) if attempts else []
    f = runs[-1] / "events.jsonl" if runs else None
    return f if f is not None and f.is_file() else None


def summarize(answers_path: Path, tree: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import stage_examples

    answers = json.loads(Path(answers_path).read_text())["answers"]
    rows = []
    for i, a in sorted(answers.items()):
        u = usage(trace_events(answers_path, i))
        rows.append({**u, **views(a.get("views_log")), "staged": len(stage_examples.planned(tree, i)),
                     "error": bool(a.get("error"))})
    n = max(len(rows), 1)
    staged = [r for r in rows if r["staged"]]
    return {"sessions": len(rows), "with_transcript": sum(r["transcript"] for r in rows),
            "program_errors": sum(r["error"] for r in rows),
            "mean_drawn_staged": round(sum(r["staged"] for r in rows) / n, 2),
            "mean_drawn_opened": round(sum(r["drawn_opened"] for r in rows) / n, 2),
            "share_opened_all_drawn": (round(sum(r["drawn_opened"] >= r["staged"] for r in staged) / len(staged), 3)
                                       if staged else None),
            "share_read_guide": round(sum(r["read_guide"] for r in rows) / n, 3),
            "mean_zoom_images": round(sum(r["zoom"] for r in rows) / n, 2),
            "mean_grid_images": round(sum(r["grid"] for r in rows) / n, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    s = summarize(args.answers, args.tree)
    print(json.dumps(s))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
