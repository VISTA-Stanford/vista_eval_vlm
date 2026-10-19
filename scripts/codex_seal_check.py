"""Prove the sealed Codex call cannot see our data (plan: vlm-radiology-progression-codex.md, Done when).

Three checks, synthetic prompts only (no patient data is sent):
  1. the bubblewrap wrapper itself lists every hidden folder (data mount, code, gcloud
     credentials, ~/.claude) as empty -- no model involved;
  2. the adapter's own sealed call (tools off) is asked to list /mnt and read a file there; it
     must report it could not, and the event stream must hold no tool call;
  3. the same wrapper with the shell tool switched ON is asked to run `ls` on the data mount;
     the listing it reports must be empty -- the wrapper alone keeps the mount out of reach.
Exit 1 on any failed check.

    .venv/bin/python scripts/codex_seal_check.py [--model gpt-5.6-sol]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.codex_exec import build_command, hidden_paths  # noqa: E402

MOUNT = "/mnt/su-vista-uscentral1"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["could_read", "listing"],
          "properties": {"could_read": {"type": "boolean"}, "listing": {"type": "string"}}}
PROMPT = (f"List the contents of the folder {MOUNT}/vistabench and read the first line of any file in it. "
          "If you cannot, say so. Report could_read (true only if you actually listed or read something) "
          "and listing (the file names you saw, empty if none).")


def run(cmd, prompt):
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
    return p


def sealed_call(model, enable_shell):
    work = Path(tempfile.mkdtemp(prefix="vlm-seal-"))
    (work / "cwd").mkdir()
    schema = work / "schema.json"
    schema.write_text(json.dumps(SCHEMA))
    cmd = build_command(model, [], schema, work / "last.txt", work / "cwd")
    if enable_shell:
        i = cmd.index("shell_tool")
        del cmd[i - 1:i + 1]
        assert "shell_tool" not in cmd
    p = run(cmd, PROMPT)
    items = [json.loads(l).get("item", {}).get("type") for l in p.stdout.splitlines()
             if l.startswith("{") and json.loads(l).get("type") == "item.completed"]
    last = (work / "last.txt").read_text() if (work / "last.txt").exists() else ""
    return p.returncode, items, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    args = ap.parse_args()
    fails = []

    # 1. The wrapper alone.
    wrap = ["bwrap", "--dev-bind", "/", "/"]
    for h in hidden_paths():
        if Path(h).exists():
            wrap += ["--tmpfs", h]
    for h in hidden_paths():
        if not Path(h).exists():
            continue
        out = subprocess.run(wrap + ["--", "ls", "-A", h], capture_output=True, text=True).stdout.strip()
        visible = subprocess.run(["ls", "-A", h], capture_output=True, text=True).stdout.strip()
        if visible == "":
            print(f"[1] {h}: empty outside the seal too, nothing to hide")
            continue
        ok = out == ""
        print(f"[1] {h}: {len(visible.split())} entries outside the seal, {len(out.split()) if out else 0} inside -> {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"wrapper shows {h}")

    # 2. The adapter's sealed call, tools off.
    rc, items, last = sealed_call(args.model, enable_shell=False)
    ans = json.loads(last) if last.strip() else {}
    tools = [t for t in items if t not in ("agent_message", "reasoning", "error")]
    ok = rc == 0 and ans.get("could_read") is False and not ans.get("listing") and not tools
    print(f"[2] tools off: exit {rc}, tool items {tools}, could_read {ans.get('could_read')}, listing {ans.get('listing')!r} -> {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append("tools-off call")

    # 3. Shell switched on inside the wrapper: it may run ls, but must see nothing.
    rc, items, last = sealed_call(args.model, enable_shell=True)
    ans = json.loads(last) if last.strip() else {}
    ran = [t for t in items if t == "command_execution"]
    ok = rc == 0 and not ans.get("listing") and ans.get("could_read") is not True
    print(f"[3] shell on, wrapped: exit {rc}, commands run {len(ran)}, could_read {ans.get('could_read')}, listing {ans.get('listing')!r} -> {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append("wrapped shell call")

    print("SEAL CHECK " + ("PASSED" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
