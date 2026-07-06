"""ModalityAdapter ABC + model-capability resolution.

Each modality (``ehr``, ``ct``, ``pathology``) implements this contract:

    ingest(raw, ctx)        -> normalized      # load / parse the raw source
    contextualize(norm, ctx) -> ContextBlock   # select + preprocess/serialize

``ctx`` is a small per-example dict carrying whatever the adapter needs
(``embed_time`` for the EHR window, ``model_caps`` for CT preprocessing, the
raw row, etc.).

Where each adapter is *driven* differs (and matters for byte-identity):
  * EHR runs at cohort-prompt-build time (single-threaded, live over LUMIA).
  * Pathology tile **selection** runs at cohort-load time (single seeded RNG,
    row order) — folder resolution + sampling; per-example only loads the
    already-selected tiles.
  * CT selection is deterministic per-volume, so it runs per-example in the
    DataLoader workers.
The adapters themselves are pure transforms; the orchestrator wiring places the
calls. The config-context viewer re-drives the very same calls offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..block import ContextBlock


@dataclass
class ModelCaps:
    """Per-model preprocessing capabilities, resolved off the model layer (OQ-B).

    Phase 1 preserves the legacy ``is_gemma`` decision (byte-identity gate 2)
    but routes it through this single resolver so the registry / config can
    override it. ``supports_inline`` mirrors the ``BaseVLMAdapter`` class attr
    read at assembler preflight (fail-closed).
    """

    model_type: str | None
    is_gemma: bool
    ct_windowing: str          # "multi_window_rgb" (gemma) | "grayscale"
    target_size: int           # 448 (gemma) | 512
    supports_inline: bool = False


def resolve_model_caps(model_type, model_adapter=None, override=None) -> ModelCaps:
    """Resolve preprocessing caps for a model.

    Order of precedence (OQ-B — "by_model off MODEL_REGISTRY, override allowed"):
      1. explicit ``override`` dict (from the block's ``preprocess`` config),
      2. class attrs on the resolved ``model_adapter`` (e.g. ``ct_windowing``,
         ``ct_target_size``, ``supports_inline``) when present,
      3. the legacy ``is_gemma`` default (byte-identity anchor).
    """
    override = override or {}
    is_gemma = bool(model_type) and "gemma" in str(model_type).lower()

    # Legacy defaults (must reproduce today's behavior exactly).
    ct_windowing = "multi_window_rgb" if is_gemma else "grayscale"
    target_size = 448 if is_gemma else 512
    supports_inline = False

    # (2) model-adapter class attrs, if the adapter exposes them.
    if model_adapter is not None:
        ct_windowing = getattr(model_adapter, "ct_windowing", None) or ct_windowing
        target_size = getattr(model_adapter, "ct_target_size", None) or target_size
        supports_inline = bool(getattr(model_adapter, "supports_inline", supports_inline))

    # (1) explicit per-block override wins.
    if isinstance(override.get("windowing"), str) and override["windowing"] != "by_model":
        ct_windowing = override["windowing"]
    if isinstance(override.get("target_size"), int):
        target_size = override["target_size"]

    return ModelCaps(
        model_type=model_type,
        is_gemma=is_gemma,
        ct_windowing=ct_windowing,
        target_size=target_size,
        supports_inline=supports_inline,
    )


class ModalityAdapter(ABC):
    """Base class for every modality adapter.

    Concrete adapters are constructed from a resolved block spec (``id``,
    ``modality``, ``config``) and hold their resolved selector(s).
    """

    #: modality label this adapter emits (``"text"``/``"volume3d"``/``"patches2d"``)
    modality: str = ""

    def __init__(self, block_id: str, config: dict | None = None):
        self.block_id = block_id
        self.config = config or {}

    @abstractmethod
    def ingest(self, raw, ctx: dict):
        """Load / parse the raw source into a normalized intermediate."""

    @abstractmethod
    def contextualize(self, normalized, ctx: dict) -> ContextBlock:
        """Select + preprocess/serialize the normalized input into a ContextBlock."""

    def build(self, raw, ctx: dict) -> ContextBlock:
        """Convenience: ``contextualize(ingest(raw))``."""
        return self.contextualize(self.ingest(raw, ctx), ctx)
