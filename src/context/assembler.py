"""Assembler: ordered ContextBlocks -> typed content sequence / flat item.

Phase 1 ships ``ordered`` (all image blocks then the text, matching every model
adapter's ``create_template`` today) plus the **``supports_inline`` seam**:
requesting ``inline_by_timestamp`` against a model that does not declare
``supports_inline`` raises at *preflight* — before any weights load — so the
viewer benefits too (OQ-D: fail closed, never silent downgrade). Real
mid-prompt interleaving is wired in Phase 1.5.

The assembler is pure over blocks. Task-prompt template stitching (e.g.
``[PATIENT_TIMELINE]`` substitution) stays in the orchestrator/presets layer;
here we only order and shape.
"""

from __future__ import annotations

from typing import Any

from .block import ContextBlock

ORDERED = "ordered"
INLINE_BY_TIMESTAMP = "inline_by_timestamp"
VALID_STRATEGIES = (ORDERED, INLINE_BY_TIMESTAMP)


class AssemblyError(RuntimeError):
    """Raised at preflight when a config requests an unsupported assembly."""


def _supports_inline(model_adapter=None, model_caps=None) -> bool:
    if model_caps is not None:
        return bool(getattr(model_caps, "supports_inline", False))
    if model_adapter is not None:
        return bool(getattr(model_adapter, "supports_inline", False))
    return False


def preflight_assembly(strategy: str, model_adapter=None, model_caps=None) -> None:
    """Validate the assembly strategy against the resolved model's capability.

    Receives the resolved adapter (or a ``ModelCaps``), NOT a bare ``model_type``,
    to avoid a ``context -> models`` import cycle. Raises :class:`AssemblyError`
    for ``inline_by_timestamp`` on a model that has not opted in.
    """
    if strategy not in VALID_STRATEGIES:
        raise AssemblyError(
            f"Unknown assembly strategy {strategy!r}; valid: {VALID_STRATEGIES}"
        )
    if strategy == INLINE_BY_TIMESTAMP and not _supports_inline(model_adapter, model_caps):
        raise AssemblyError(
            f"assembly '{INLINE_BY_TIMESTAMP}' requires the model to declare "
            f"supports_inline=True on its BaseVLMAdapter subclass. The resolved model "
            f"does not, so mid-prompt image interleaving is unavailable (fail-closed). "
            f"Use assembly 'ordered', or enable inline support for this model in Phase 1.5."
        )


class Assembler:
    def __init__(self, strategy: str = ORDERED):
        if strategy not in VALID_STRATEGIES:
            raise AssemblyError(
                f"Unknown assembly strategy {strategy!r}; valid: {VALID_STRATEGIES}"
            )
        self.strategy = strategy

    def preflight(self, model_adapter=None, model_caps=None) -> None:
        preflight_assembly(self.strategy, model_adapter, model_caps)

    def content_sequence(self, blocks: list[ContextBlock]) -> list[dict[str, Any]]:
        """Typed ordered content sequence: ``[{type: text|image, ...}]``.

        ``ordered`` emits image entries (one per image) then text entries, in
        block-list order — the ContextBlock list *is* the content sequence.
        """
        image_entries: list[dict[str, Any]] = []
        text_entries: list[dict[str, Any]] = []
        for b in blocks:
            if b.is_text:
                if b.payload:
                    text_entries.append({"type": "text", "text": b.payload, "block_id": b.id})
            elif b.is_images:
                for img in (b.payload or []):
                    image_entries.append({"type": "image", "image": img, "block_id": b.id})
        if self.strategy == ORDERED:
            return image_entries + text_entries
        # INLINE_BY_TIMESTAMP is gated by preflight; real interleaving = Phase 1.5.
        raise AssemblyError(
            f"content_sequence for '{self.strategy}' is not implemented until Phase 1.5"
        )

    def to_flat_item(self, blocks: list[ContextBlock]) -> dict[str, Any]:
        """Collapse blocks to the legacy flat item **container shape**
        ``{question, image}`` that ``model.create_template`` consumes.

        ``question`` = text-block payloads joined in order; ``image`` = all
        image-block payloads gathered into a list (or ``None``). This produces the
        container shape, not the final prompt string: today's ``create_template``
        receives ``question = dynamic_prompt`` (the task template with
        ``[PATIENT_TIMELINE]`` already substituted). The hot-path wiring is
        responsible for building that templated text into the text block(s) before
        assembly — template stitching stays in the orchestrator (see the module
        docstring), not here.
        """
        texts = [b.payload for b in blocks if b.is_text and b.payload]
        images: list = []
        for b in blocks:
            if b.is_images and b.payload:
                images.extend(b.payload)
        return {
            "question": "\n".join(texts) if texts else "",
            "image": images or None,
        }
