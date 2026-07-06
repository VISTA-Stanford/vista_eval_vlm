"""Pathology leaf selectors: (candidate tile paths) -> selected tile paths.

Operates on an **already-materialized** list of tile paths — the folder
globbing / resolution lives in the pathology adapter (lifted from
``run_bq._load_path_task_data``), NOT here, so this stays a pure sampler and
never duplicates ``tile_wsi.py``.

Byte-identity note (golden gate 1): legacy seeds the module RNG **once** with 42
and calls ``random.sample`` per slide while iterating rows in order, so the RNG
state carries across slides. ``random_n`` therefore takes a caller-owned
``random.Random`` instance; the adapter seeds one ``Random(seed)`` per cohort and
drives rows in the original order. ``Random(42).sample`` yields the same sequence
as the legacy module-level ``random.seed(42)`` + ``random.sample`` (same Mersenne
Twister), so bytes match.

Specimen/block/slide selection is upstream in VISTABench (OQ-A). ``tissue_top_n``
is a future addition (needs tissue scores, not available on bare tile paths).
"""

from __future__ import annotations

import random
from typing import Sequence


def random_n(candidates: Sequence, n: int, rng: random.Random) -> list:
    """Up to ``n`` tiles sampled with ``rng``; all of them if ``len <= n``.

    Mirrors legacy ``run_bq._load_path_task_data``: when there are at most ``n``
    candidates, keep all (no draw from ``rng``); otherwise ``rng.sample``.
    """
    candidates = list(candidates)
    if len(candidates) <= n:
        return candidates
    return rng.sample(candidates, n)


def all(candidates: Sequence, rng: random.Random | None = None) -> list:  # noqa: A001
    """Keep every candidate tile (``rng`` accepted for a uniform signature)."""
    return list(candidates)


PATH_SELECTORS = {
    "random_n": random_n,
    "all": all,
}


def resolve_path_selector(spec: dict):
    """Bind a config ``select`` dict to a ``(candidates, rng) -> list`` callable.

    ``spec`` is e.g. ``{"fn": "random_n", "n": 100, "seed": 42}``. ``seed`` (if
    present) is surfaced on the returned callable so the adapter can seed one
    cohort-wide RNG; the per-call ``rng`` argument carries the running state.
    """
    if not isinstance(spec, dict) or "fn" not in spec:
        raise ValueError(f"Path select spec must be a dict with an 'fn' key, got {spec!r}")
    fn_name = spec["fn"]
    if fn_name not in PATH_SELECTORS:
        raise ValueError(
            f"Unknown pathology selector {fn_name!r}; valid: {sorted(PATH_SELECTORS)}"
        )
    fn = PATH_SELECTORS[fn_name]
    params = {k: v for k, v in spec.items() if k not in ("fn", "seed")}
    seed = spec.get("seed", 42)

    def _select(candidates: Sequence, rng: random.Random) -> list:
        return fn(candidates, rng=rng, **params)

    _select.fn_name = fn_name
    _select.params = params
    _select.seed = seed
    return _select
