"""A binary-safe ``engine.materialize.materialize`` for the knowledge-base pilot (config.KB_ARM).

The pinned engine writes each frozen program file with ``git show`` decoded as UTF-8 text, which fails on the
library's PNG images (found 2026-11-23: the cost check's builder copied back 49 files, the commit worked, and writing
the version out raised UnicodeDecodeError). Copy-back, commit, staging and the tree fingerprint already work on bytes.
This is the engine's function with the one change: each file is written as the bytes git holds. Same containment check,
same clearing of an earlier output, same read-only result. config.py swaps it in for KB_ARM runs only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from engine import materialize as em


def materialize(manifest, version_sha: str, *, repo_root: Path, dest: Path, hash_only_resolver=None):
    if any(e.freeze_policy != "committed" for e in manifest.entries):
        return _ENGINE_MATERIALIZE(manifest, version_sha, repo_root=repo_root, dest=dest,
                                   hash_only_resolver=hash_only_resolver)
    repo_root, dest = Path(repo_root), Path(dest)
    if em._is_earlier_output(dest):
        em._clear(dest)
    dest.mkdir(parents=True, exist_ok=True)
    all_files = em._version_files(repo_root, version_sha)
    for entry in manifest.entries:
        matches = em._entry_matches(entry, all_files)
        if not matches:
            if entry.optional:
                continue
            raise em.MaterializeError(
                f"materialize {version_sha[:10]}: manifest entry {entry.path!r} (contract_file={entry.contract_file}) "
                "matched no file in the frozen tree", entry_path=entry.path)
        for rel in matches:
            data = subprocess.run(["git", "-C", str(repo_root), "show", f"{version_sha}:{rel}"], capture_output=True,
                                  check=True).stdout
            out = em._contained_write_path(dest, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    em._make_read_only(dest)
    return manifest


_ENGINE_MATERIALIZE = em.materialize


def install() -> None:
    """Swap it in, before any engine module binds the original (config.py runs first everywhere)."""
    em.materialize = materialize
