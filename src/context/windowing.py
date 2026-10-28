"""Legacy import path for CT windowing. Everything lives in ``context.windows`` now; this module only
re-exports it, so callers and existing runs keep working with byte-identical pixels."""

from __future__ import annotations

from .windows import CHEST_WINDOWS, chest_rgb, grayscale, multi_window_rgb, norm  # noqa: F401

# Legacy alias kept so `vqa_dataset` can shim `from context.windowing import window`.
window = multi_window_rgb
