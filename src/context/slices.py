"""CT slice samplers: (volume depth) -> list of axial slice indices. The one home for them.

Replaces three samplers: main's legacy evenly-spaced sampler (below), MedGemma's notebook sampler
(``medgemma_uniform``) and the agent viewer's ``--from/--to/--count`` rule (``fraction_range``).
``context.selectors.ct_selectors`` re-exports from here.

Pure integer index math (no pixels touched here). ``evenly_spaced_k`` is the
byte-identity anchor: it reproduces the legacy per-branch sampler
(``vqa_dataset.__getitem__``) exactly, including the never-firing ``>= depth``
clamp, so the legacy presets (50 / 30 / 10 slices) map onto it with zero image
drift (golden gate 1).

Leaf-only by design: study/series selection is upstream in VISTABench (OQ-A).
The ``(depth, **params)`` signature could later host a learned selector (OQ-C).
"""

from __future__ import annotations


def evenly_spaced_k(depth: int, k: int) -> list[int]:
    """``k`` indices evenly spaced across ``[0, depth)``.

    Reproduces the legacy sampler verbatim::

        position = i / (k - 1)          # or 0.0 when k == 1
        index    = int(position * (depth - 1))

    The ``index >= depth`` guard is preserved for parity with the legacy
    branches even though it can never fire for this formula.
    """
    if depth <= 0 or k <= 0:
        return []
    indices = []
    for i in range(k):
        position = i / (k - 1) if k > 1 else 0.0
        index = int(position * (depth - 1))
        if index >= depth:
            index = depth - 1
        indices.append(index)
    return indices


def up_to_k(depth: int, k: int) -> list[int]:
    """Every slice when the volume has ``k`` or fewer, else ``evenly_spaced_k(k)``.

    The whole-volume CT arms (radiology progression, 250 per scan): a 2,000-slice CT can't go
    whole into one call, and ``evenly_spaced_k`` on a short volume would repeat slices.
    """
    if depth <= 0 or k <= 0:
        return []
    return list(range(depth)) if depth <= k else evenly_spaced_k(depth, k)


def medgemma_uniform(depth: int, k: int = 85) -> list[int]:
    """Every slice when the volume has ``k`` or fewer, else MedGemma 1.5's own uniform sample.

    Verbatim index rule from Google's CT notebook (``high_dimensional_ct_hugging_face.ipynb``)::

        [round(i / k * (depth - 1)) for i in range(1, k + 1)]

    so the first slice is skipped and the last is always kept. Python's ``round`` (half to even),
    as in the notebook.
    """
    if depth <= 0 or k <= 0:
        return []
    if depth <= k:
        return list(range(depth))
    return [int(round(i / k * (depth - 1))) for i in range(1, k + 1)]


def fraction_range(depth: int, count: int, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    """The agent viewer's ``--count`` slices between fractions ``lo`` and ``hi`` of the axis (sorted, unique)."""
    if depth <= 0:
        return []
    a, b = int(round(lo * (depth - 1))), int(round(hi * (depth - 1)))
    k = max(1, count)
    return sorted({int(round(a + (b - a) * j / max(k - 1, 1))) for j in range(k)})


def every_n(depth: int, n: int) -> list[int]:
    """Every ``n``-th slice: ``0, n, 2n, ...`` while ``< depth``."""
    if depth <= 0 or n <= 0:
        return []
    return list(range(0, depth, n))


def center(depth: int) -> list[int]:
    """The single middle slice."""
    if depth <= 0:
        return []
    return [depth // 2]


def center_k(depth: int, k: int) -> list[int]:
    """``k`` contiguous slices centered on the middle of the volume."""
    if depth <= 0 or k <= 0:
        return []
    if k >= depth:
        return list(range(depth))
    start = (depth - k) // 2
    return list(range(start, start + k))


def slice_range(depth: int, lo: int, hi: int) -> list[int]:
    """Slices in the half-open range ``[lo, hi)``, clamped to the volume."""
    if depth <= 0:
        return []
    lo = max(0, lo)
    hi = min(depth, hi)
    return list(range(lo, hi)) if hi > lo else []


def all(depth: int) -> list[int]:  # noqa: A001 - deliberate selector name
    """Every slice in the volume."""
    if depth <= 0:
        return []
    return list(range(depth))


# name -> selector function. Params come from the config `select` dict.
CT_SELECTORS = {
    "evenly_spaced_k": evenly_spaced_k,
    "up_to_k": up_to_k,
    "medgemma_uniform": medgemma_uniform,
    "fraction_range": fraction_range,
    "every_n": every_n,
    "center": center,
    "center_k": center_k,
    "slice_range": slice_range,
    "all": all,
}


def resolve_ct_selector(spec: dict):
    """Bind a config ``select`` dict to a ``(depth) -> list[int]`` callable.

    ``spec`` is e.g. ``{"fn": "every_n", "n": 10}`` or ``{"fn": "evenly_spaced_k",
    "k": 50}``. The ``fn`` key names the selector; all other keys are passed as
    params.
    """
    if not isinstance(spec, dict) or "fn" not in spec:
        raise ValueError(f"CT select spec must be a dict with an 'fn' key, got {spec!r}")
    fn_name = spec["fn"]
    if fn_name not in CT_SELECTORS:
        raise ValueError(
            f"Unknown CT selector {fn_name!r}; valid: {sorted(CT_SELECTORS)}"
        )
    fn = CT_SELECTORS[fn_name]
    params = {k: v for k, v in spec.items() if k != "fn"}

    def _select(depth: int) -> list[int]:
        return fn(depth, **params)

    _select.fn_name = fn_name
    _select.params = params
    return _select
