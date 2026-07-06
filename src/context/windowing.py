"""CT slice preprocessing (windowing / normalization) — the single home.

These functions are lifted **verbatim** from ``vqa_dataset.py`` (``norm``,
``window``) and ``vista_run/utils/utils_inference.py`` (``normalize_slice``) so
the CT adapter and the legacy dataset share one implementation. ``vqa_dataset``
re-imports from here as a shim, keeping image bytes byte-identical across the
refactor (golden gate 1 / gate 2).

Two representations, selected per-model in the CT adapter:
  * ``multi_window_rgb`` (== legacy ``window``): 3-window RGB, used for Gemma.
  * ``grayscale`` (== legacy ``normalize_slice`` path): single-channel L.
"""

from __future__ import annotations

import numpy as np


def norm(ct_vol: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Window and normalize CT Hounsfield values to 0 - 255 (verbatim lift)."""
    ct_vol = np.clip(ct_vol, min_val, max_val)  # Clip the imaging value range
    ct_vol = ct_vol.astype(np.float32)
    ct_vol -= min_val
    ct_vol /= (max_val - min_val)  # Norm to values between 0 - 1.0
    ct_vol *= 255.0  # Norm to values been 0 - 255.0
    return ct_vol


def multi_window_rgb(ct_vol: np.ndarray) -> np.ndarray:
    """Window a CT slice with three windows (wide, mediastinum, brain).

    Verbatim lift of legacy ``vqa_dataset.window``. RGB channels contain
    different representations of the data (image appears color when visualized).
    """
    window_clips = [(-1024, 1024), (-135, 215), (0, 80)]
    return np.stack([norm(ct_vol, clip[0], clip[1]) for clip in window_clips], axis=-1)


def grayscale(slice_data: np.ndarray) -> np.ndarray:
    """Normalize a slice to a 0-255 uint8 array (verbatim lift of
    ``utils_inference.normalize_slice``)."""
    # Handle potential NaN or inf values
    if np.any(np.isnan(slice_data)) or np.any(np.isinf(slice_data)):
        slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0-255
    if slice_data.max() > slice_data.min():
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    else:
        slice_data = np.zeros_like(slice_data, dtype=np.uint8)
    return slice_data


# Legacy alias kept so `vqa_dataset` can shim `from context.windowing import window`.
window = multi_window_rgb
