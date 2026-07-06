"""Modality adapters + registry."""

from .base import ModalityAdapter, ModelCaps, resolve_model_caps
from .ct import CTAdapter
from .pathology import PathologyAdapter
from .ehr import EHRAdapter

# adapter name (config `adapter:` key) -> adapter class
ADAPTER_REGISTRY = {
    "ehr": EHRAdapter,
    "ct": CTAdapter,
    "pathology": PathologyAdapter,
}


def build_adapter(name: str, block_id: str, config: dict | None = None) -> ModalityAdapter:
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter {name!r}; valid: {sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name](block_id, config)


__all__ = [
    "ModalityAdapter",
    "ModelCaps",
    "resolve_model_caps",
    "CTAdapter",
    "PathologyAdapter",
    "EHRAdapter",
    "ADAPTER_REGISTRY",
    "build_adapter",
]
