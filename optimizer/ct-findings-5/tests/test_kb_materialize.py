"""kb_materialize: the knowledge-base pilot's binary-safe version writer (the pinned engine's decodes files as text and
fails on the library's PNGs). A throwaway git repo with a binary file and a text file; the written tree must hold the
same bytes git holds, and be read-only, as the engine's is.

    cd optimizer/ct-findings-5 && uv run -q --project $AGENTIC_LABEL_OPT --with pytest python -m pytest -q tests/test_kb_materialize.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
engine_root = os.environ.get("AGENTIC_LABEL_OPT")
if not engine_root:
    pytest.skip("needs AGENTIC_LABEL_OPT (the engine checkout)", allow_module_level=True)
sys.path.insert(0, engine_root)
import kb_materialize  # noqa: E402
from engine.schemas import ManifestEntry, ProgramManifest  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_writes_binary_files_byte_for_byte(tmp_path):
    repo = tmp_path / "repo"
    (repo / "library" / "entries" / "e1").mkdir(parents=True)
    png = bytes([0x89, 0x50, 0x4E, 0x47]) + bytes(range(256))  # a PNG signature then every byte value
    (repo / "library" / "entries" / "e1" / "view1.png").write_bytes(png)
    (repo / "instructions.md").write_text("read the scan\n")
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "v")
    sha = git(repo, "rev-parse", "HEAD")
    manifest = ProgramManifest(entries=(ManifestEntry(path="library", freeze_policy="committed"),
                                        ManifestEntry(path="instructions.md", freeze_policy="committed", contract_file=True)),
                               combined_hash="t")
    dest = tmp_path / "out"
    kb_materialize.materialize(manifest, sha, repo_root=repo, dest=dest)
    assert (dest / "library" / "entries" / "e1" / "view1.png").read_bytes() == png
    assert (dest / "instructions.md").read_text() == "read the scan\n"
    assert not os.access(dest / "instructions.md", os.W_OK)
    kb_materialize.materialize(manifest, sha, repo_root=repo, dest=dest)  # a second write over its own output works
    assert (dest / "library" / "entries" / "e1" / "view1.png").read_bytes() == png
