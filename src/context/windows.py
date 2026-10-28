"""CT windows: the one home for every Hounsfield window the VLM runs and the agent viewer use.

One implementation for three copies: main's MedGemma 3-window colour (``windowing.multi_window_rgb``), radiology
progression's lung / soft tissue / bone colour (``windowing.chest_rgb``) and the agent viewer's own
named windows (``optimizer/died-2yr-ct/program_agentic/tools/view_ct.py``, on work package 2's branch;
``window_u8`` reproduces it so the viewer can switch later). ``context.windowing``
re-exports from here, so existing runs keep byte-identical pixels (``scripts/ct_golden_inputs.py``).

Two conversions to 0-255, kept apart because they round differently:
  * ``norm`` (MedGemma notebook, verbatim): float 0-255; callers round with ``np.round`` then uint8.
  * ``window_u8`` (agent viewer, verbatim): scale then truncate to uint8.
"""

from __future__ import annotations

import numpy as np

#: Named 3-window colour sets, (low, high) HU for red, green, blue.
#: ``medgemma``: wide / soft tissue / brain, exactly MedGemma 1.5's training windows (tech report and
#: the official CT notebook's code; the notebook prose drops the minus on -135).
#: ``chest``: lung (W1500/L-600) / soft tissue (W400/L40) / bone (W1800/L400), radiology progression's set.
RGB_SETS = {
    "medgemma": ((-1024, 1024), (-135, 215), (0, 80)),
    "chest": ((-1350, 150), (-160, 240), (-500, 1300)),
}

#: Plain-English names per set, for the image description a card opens with.
RGB_SET_NAMES = {
    "medgemma": ("wide", "soft-tissue", "brain"),
    "chest": ("lung", "soft-tissue", "bone"),
}

#: Single named windows (the agent viewer's).
WINDOWS = {"lung": (-1350, 150), "soft": (-160, 240), "bone": (-500, 1300), "liver": (-40, 160)}

MEDGEMMA_WINDOWS = RGB_SETS["medgemma"]
CHEST_WINDOWS = RGB_SETS["chest"]


def norm(ct_vol: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Window and normalize CT Hounsfield values to 0 - 255 (verbatim from the MedGemma notebook)."""
    ct_vol = np.clip(ct_vol, min_val, max_val)  # Clip the imaging value range
    ct_vol = ct_vol.astype(np.float32)
    ct_vol -= min_val
    ct_vol /= (max_val - min_val)  # Norm to values between 0 - 1.0
    ct_vol *= 255.0  # Norm to values been 0 - 255.0
    return ct_vol


def rgb(ct_slice: np.ndarray, windows) -> np.ndarray:
    """Float 0-255 array (..., 3): one window per channel. ``windows`` is a set name or three (lo, hi)."""
    clips = RGB_SETS[windows] if isinstance(windows, str) else windows
    return np.stack([norm(ct_slice, lo, hi) for lo, hi in clips], axis=-1)


def rgb_u8(ct_slice: np.ndarray, windows) -> np.ndarray:
    """``rgb`` rounded to uint8, as the MedGemma notebook does before ``PIL.Image.fromarray``."""
    return np.round(rgb(ct_slice, windows), 0).astype(np.uint8)


def multi_window_rgb(ct_vol: np.ndarray) -> np.ndarray:
    """MedGemma's wide / mediastinum / brain colour (float 0-255); the legacy name."""
    return rgb(ct_vol, "medgemma")


def chest_rgb(ct_vol: np.ndarray) -> np.ndarray:
    """Red = lung, green = soft tissue, blue = bone (float 0-255); the legacy name."""
    return rgb(ct_vol, "chest")


def window_u8(arr: np.ndarray, spec) -> np.ndarray:
    """One window to uint8 the agent viewer's way. ``spec``: a WINDOWS name, "LO,HI", or (lo, hi)."""
    if isinstance(spec, str):
        lo, hi = WINDOWS[spec] if spec in WINDOWS else (float(v) for v in spec.split(","))
    else:
        lo, hi = spec
    return (np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def grayscale(slice_data: np.ndarray) -> np.ndarray:
    """Per-slice min-max to uint8 (verbatim from ``utils_inference.normalize_slice``); no HU window."""
    # Handle potential NaN or inf values
    if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
        slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0-255
    if slice_data.max() > slice_data.min():
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    else:
        slice_data = np.zeros_like(slice_data, dtype=np.uint8)
    return slice_data


def describe(windows) -> str:
    """'wide, soft-tissue and brain windows' for a named set (the card's image description)."""
    a, b, c = RGB_SET_NAMES[windows]
    return f"{a}, {b} and {c} windows"
