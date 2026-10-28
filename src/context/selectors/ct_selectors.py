"""Legacy import path for the CT slice samplers; they live in ``context.slices`` now."""

from ..slices import (  # noqa: F401
    CT_SELECTORS,
    all,
    center,
    center_k,
    every_n,
    evenly_spaced_k,
    fraction_range,
    medgemma_uniform,
    resolve_ct_selector,
    slice_range,
    up_to_k,
)
