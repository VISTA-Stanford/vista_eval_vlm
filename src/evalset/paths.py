"""Where the eval set lives on the shared mount, on phil-sllm-01 (/mnt) and on weill (its own mount root).

Everything the eval set writes (samples, image packs, answers, scores) goes under ``root()``; it holds
patient-level data, so it stays on the mount and never in git.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = "som-nero-plevriti-deidbdf"
DATASET = "vista_bench_v1_7"
#: Mount roots tried in order when ``EVALSET_MOUNT_ROOT`` is unset: phil-sllm-01, then weill-p01.
MOUNT_ROOTS = ("/mnt", "/data/fries/users/padamson/mnt")
BUCKET = "su-vista-uscentral1"
SUBDIR = "vistabench/vlm/evalset_v1"


def mount_root() -> Path:
    env = os.environ.get("EVALSET_MOUNT_ROOT")
    if env:
        return Path(env)
    for r in MOUNT_ROOTS:
        if (Path(r) / BUCKET).is_dir():
            return Path(r)
    raise RuntimeError(f"no {BUCKET} mount under {MOUNT_ROOTS}; set EVALSET_MOUNT_ROOT")


def root() -> Path:
    env = os.environ.get("EVALSET_ROOT")
    return Path(env) if env else mount_root() / BUCKET / SUBDIR


def table(name: str) -> str:
    return f"{PROJECT}.{DATASET}.{name}"
