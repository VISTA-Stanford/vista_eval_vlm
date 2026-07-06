"""Pathology modality adapter: WSI tile folder -> sampled patch images.

Two-stage, mirroring legacy ``run_bq._load_path_task_data`` +
``vqa_dataset`` path branch:

  * ``materialize(df, ...)`` — COHORT-level, single-threaded: resolve each row's
    tile folder, glob candidates, and sample with ONE seeded RNG advanced in row
    order. Writes the ``path_tile_paths`` column. This preserves byte-identity
    (the legacy sampler seeds ``random`` once and draws per row); it must NOT move
    into the multi-worker DataLoader.
  * ``contextualize(tile_paths, ...)`` — per-example: load the already-selected
    tiles, RGB-convert, ``pad_to_size``.

Folder resolution + sampling are lifted here so nothing duplicates
``tile_wsi.py`` (which produces the tiles upstream). Specimen/block/slide
selection is upstream in VISTABench (OQ-A).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from vista_run.utils.utils_inference import pad_to_size

from ..block import ContextBlock, IMAGES
from ..selectors.path_selectors import resolve_path_selector
from .base import ModalityAdapter, ModelCaps, resolve_model_caps

DEFAULT_PATH_SELECT = {"fn": "random_n", "n": 100, "seed": 42}


class PathologyAdapter(ModalityAdapter):
    modality = "patches2d"

    def __init__(self, block_id: str = "path", config: dict | None = None):
        super().__init__(block_id, config)
        select_spec = self.config.get("select") or DEFAULT_PATH_SELECT
        self.selector = resolve_path_selector(select_spec)
        self.seed = self.selector.seed

    def new_rng(self) -> random.Random:
        """A fresh cohort-scoped RNG. ``Random(seed)`` matches the legacy
        module-level ``random.seed(seed)`` sequence exactly."""
        return random.Random(self.seed)

    @staticmethod
    def folder_name_from_path_image_path(path_image_path):
        """Verbatim lift of the legacy folder-name derivation."""
        if pd.isna(path_image_path) or path_image_path is None:
            return None
        s = str(path_image_path).strip()
        if not s:
            return None
        last_after_backslash = s.split("\\")[-1]
        filename = last_after_backslash.split("/")[-1]
        folder_name = Path(filename).stem
        return folder_name or None

    def resolve_candidates(self, path_image_path, test_patch_dir: Path) -> list[Path]:
        """Glob tile candidates for one slide's folder (glob-sorted, .jpg then .jpeg)."""
        folder_name = self.folder_name_from_path_image_path(path_image_path)
        if not folder_name:
            return []
        folder = Path(test_patch_dir) / folder_name
        if not folder.is_dir():
            return []
        return sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))

    def materialize(self, df: pd.DataFrame, test_patch_dir, path_image_col: str,
                    drop_empty: bool = True) -> pd.DataFrame:
        """Cohort-level: add ``path_tile_paths`` (sampled str paths) per row.

        Reproduces legacy ``_load_path_task_data`` byte-for-byte: row-order,
        single-RNG sampling, then (``drop_empty=True``, the default) dropping rows
        that resolved no tiles — the same positional boolean mask legacy applies
        (``run_bq.py:527``). Set ``drop_empty=False`` only if a caller needs the
        full frame (e.g. the viewer showing empty rows).
        """
        test_patch_dir = Path(test_patch_dir)
        rng = self.new_rng()
        path_tile_paths_list = []
        for _, row in df.iterrows():
            candidates = self.resolve_candidates(row.get(path_image_col), test_patch_dir)
            if not candidates:
                path_tile_paths_list.append([])
                continue
            chosen = self.selector(candidates, rng)
            path_tile_paths_list.append([str(p.resolve()) for p in chosen])
        df = df.copy()
        df["path_tile_paths"] = path_tile_paths_list
        if drop_empty:
            df = df[[len(p) > 0 for p in path_tile_paths_list]].copy()
        return df

    def _caps(self, ctx: dict) -> ModelCaps:
        caps = ctx.get("model_caps")
        if isinstance(caps, ModelCaps):
            return caps
        return resolve_model_caps(ctx.get("model_type"), override=self.config.get("preprocess", {}))

    def ingest(self, raw, ctx: dict):
        """``raw`` is the already-selected list of tile path strings."""
        return raw

    def contextualize(self, tile_paths, ctx: dict) -> ContextBlock:
        caps = self._caps(ctx)
        images: list[Image.Image] = []
        if tile_paths:
            for p in tile_paths:
                p_str = str(p).strip()
                if p_str and os.path.exists(p_str):
                    try:
                        with Image.open(p_str) as pil_img:
                            pil_img.load()
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            images.append(pad_to_size(pil_img.copy(), caps.target_size))
                    except Exception:
                        pass
        return ContextBlock(
            id=self.block_id,
            modality=self.modality,
            representation=IMAGES,
            payload=images or None,
            metadata={
                "selector": {"fn": self.selector.fn_name, **self.selector.params, "seed": self.seed},
                "count": len(images),
                "target_size": caps.target_size,
            },
        )
