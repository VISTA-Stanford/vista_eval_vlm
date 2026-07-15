"""CT modality adapter: 3D volume -> selected + windowed axial slices.

``contextualize`` reproduces legacy ``PromptDataset._process_ct_slice`` +
per-branch slice sampling byte-for-byte:
  * gemma  -> ``multi_window_rgb`` -> round -> uint8 RGB
  * others -> ``grayscale`` -> L
  * then ``pad_to_size(target)``.
Slice indices come from a configured CT selector (default ``evenly_spaced_k(50)``,
matching the legacy no-timeline default). NIfTI *loading* (GCP/local resolution)
stays in the caller (``vqa_dataset``); this adapter operates on a loaded volume —
either a numpy array (offline viewer) or a lazy nibabel ArrayProxy
(``ct_img.dataobj``), so per-slice reads never materialize the full volume (the CT
OOM fix is preserved) — and is pure and re-drivable by the offline viewer.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vista_run.utils.utils_inference import pad_to_size

from ..block import ContextBlock, IMAGES
from ..windowing import multi_window_rgb, grayscale
from ..selectors.ct_selectors import resolve_ct_selector
from .base import ModalityAdapter, ModelCaps, resolve_model_caps

# Legacy default sampler (the no_timeline / all_vb_image_only branch used 50).
DEFAULT_CT_SELECT = {"fn": "evenly_spaced_k", "k": 50}


class CTAdapter(ModalityAdapter):
    modality = "volume3d"

    def __init__(self, block_id: str = "ct", config: dict | None = None):
        super().__init__(block_id, config)
        select_spec = self.config.get("select") or DEFAULT_CT_SELECT
        self.selector = resolve_ct_selector(select_spec)
        self.preprocess_cfg = self.config.get("preprocess", {}) or {}

    def ingest(self, raw, ctx: dict):
        """``raw`` is a loaded 3D numpy volume (or None). Pass-through."""
        return raw

    def select_indices(self, depth: int) -> list[int]:
        return self.selector(depth)

    def preprocess_slice(self, ct_slice: np.ndarray, caps: ModelCaps) -> Image.Image:
        if caps.ct_windowing == "multi_window_rgb":
            windowed = multi_window_rgb(ct_slice)
            windowed = np.round(windowed, 0).astype(np.uint8)
            pil_img = Image.fromarray(windowed, mode="RGB")
        else:
            normalized = grayscale(ct_slice)
            pil_img = Image.fromarray(normalized, mode="L")
        return pad_to_size(pil_img, caps.target_size)

    def _caps(self, ctx: dict) -> ModelCaps:
        caps = ctx.get("model_caps")
        if isinstance(caps, ModelCaps):
            return caps
        return resolve_model_caps(ctx.get("model_type"), override=self.preprocess_cfg)

    def contextualize(self, normalized, ctx: dict) -> ContextBlock:
        volume = normalized
        caps = self._caps(ctx)
        images: list[Image.Image] = []
        indices: list[int] = []
        if volume is not None and getattr(volume, "ndim", 0) > 2:
            depth = volume.shape[2]
            indices = self.select_indices(depth)
            for idx in indices:
                # `volume` is array-like: a numpy volume (offline viewer) OR a lazy
                # nibabel ArrayProxy (`ct_img.dataobj`) from the caller's load path, so
                # slicing here reads one plane and never materializes the whole CT (the
                # OOM fix stays intact). Cast to float64 to match the legacy
                # `np.asarray(dataobj[:, :, idx], dtype=np.float64)` slice exactly, so the
                # windowed bytes are byte-identical (golden gate 1/2).
                ct_slice = np.asarray(volume[:, :, idx], dtype=np.float64)
                images.append(self.preprocess_slice(ct_slice, caps))
        return ContextBlock(
            id=self.block_id,
            modality=self.modality,
            representation=IMAGES,
            payload=images or None,
            metadata={
                "selector": {"fn": self.selector.fn_name, **self.selector.params},
                "selected_indices": indices,
                "count": len(images),
                "target_size": caps.target_size,
                "windowing": caps.ct_windowing,
            },
        )
