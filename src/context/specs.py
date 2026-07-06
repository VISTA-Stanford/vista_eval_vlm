"""Resolve a normalized experiment spec into live adapter instances.

``ContextComposition`` bundles the per-block adapters + the assembler for one
experiment. The orchestrator wiring and the config-context viewer both drive the
*same* composition, so what the viewer shows cannot drift from what inference
feeds the model.
"""

from __future__ import annotations

from typing import Any

from .adapters import build_adapter, ModalityAdapter
from .assembler import Assembler
from .block import ContextBlock


class ContextComposition:
    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec.get("name")
        self.cohort = spec.get("cohort", "normal")
        self.assembly = spec.get("assembly", "ordered")
        self.blocks_cfg: list[dict] = spec.get("blocks", []) or []
        # ordered list of (block_cfg, adapter)
        self.adapters: list[tuple[dict, ModalityAdapter]] = [
            (blk, build_adapter(blk["adapter"], blk.get("id", blk["adapter"]), blk.get("config")))
            for blk in self.blocks_cfg
        ]
        self.assembler = Assembler(self.assembly)

    # --- typed accessors ------------------------------------------------------
    def adapter_by_name(self, adapter_name: str) -> ModalityAdapter | None:
        for blk, adapter in self.adapters:
            if blk["adapter"] == adapter_name:
                return adapter
        return None

    @property
    def ehr(self):
        return self.adapter_by_name("ehr")

    @property
    def ct(self):
        return self.adapter_by_name("ct")

    @property
    def pathology(self):
        return self.adapter_by_name("pathology")

    @property
    def has_images(self) -> bool:
        return self.ct is not None or self.pathology is not None

    # --- driving --------------------------------------------------------------
    def preflight(self, model_adapter=None, model_caps=None) -> None:
        """Validate the assembly against the resolved model (fail-closed inline)."""
        self.assembler.preflight(model_adapter=model_adapter, model_caps=model_caps)

    def assemble(self, blocks: list[ContextBlock]) -> dict[str, Any]:
        """Collapse assembled blocks to the legacy flat ``{question, image}`` item."""
        return self.assembler.to_flat_item(blocks)

    def content_sequence(self, blocks: list[ContextBlock]) -> list[dict]:
        return self.assembler.content_sequence(blocks)


def resolve_spec(spec: dict) -> ContextComposition:
    return ContextComposition(spec)
