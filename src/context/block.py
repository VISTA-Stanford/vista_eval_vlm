"""ContextBlock: the uniform unit every modality adapter emits.

A ContextBlock is the single currency the assembler consumes. Each modality
adapter (`ehr`, `ct`, `pathology`) runs ingest -> contextualize and hands back
one (or more) of these. The block carries *both* the model-ready payload and the
provenance/metadata the config-context viewer renders and that Phase-1.5 inline
placement will key off (`metadata["timestamp"]`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- representation: how `payload` should be fed to the model -----------------
# The assembler turns each representation into typed content-sequence entries.
TEXT = "text"        # payload is a str (the flat-rendered timeline / prompt text)
IMAGES = "images"    # payload is a list[PIL.Image.Image] (CT slices / path tiles)

VALID_REPRESENTATIONS = (TEXT, IMAGES)


@dataclass
class ContextBlock:
    """One assembled piece of model context.

    Attributes:
        id: Stable block id within a composition (e.g. ``"ehr"``, ``"ct"``,
            ``"path"``). Used by the viewer and for provenance.
        modality: The config-level modality label (``"text"``, ``"volume3d"``,
            ``"patches2d"``). Describes the *source* modality, distinct from how
            it is represented to the model.
        representation: One of :data:`TEXT` / :data:`IMAGES` — how ``payload`` is
            consumed by the assembler.
        payload: The model-ready content. ``str`` for :data:`TEXT`; a
            ``list`` of PIL images for :data:`IMAGES`.
        metadata: Provenance + rendering hints. Conventionally carries:
            ``timestamp`` (acquisition time, for Phase-1.5 inline placement),
            ``selector`` (the selector params applied), ``count`` (n images /
            n events), and any adapter-specific notes (e.g. a declared-delta
            marker when a field was unavailable).
    """

    id: str
    modality: str
    representation: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"ContextBlock.representation must be one of {VALID_REPRESENTATIONS}, "
                f"got {self.representation!r}"
            )

    @property
    def is_text(self) -> bool:
        return self.representation == TEXT

    @property
    def is_images(self) -> bool:
        return self.representation == IMAGES

    @property
    def image_count(self) -> int:
        """Number of images this block contributes (0 for text blocks)."""
        if not self.is_images or self.payload is None:
            return 0
        return len(self.payload) if isinstance(self.payload, list) else 1
