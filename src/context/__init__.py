"""Modality-adapter context framework (Phase 1).

Dissolves the ``experiment`` god-enum into composable, config-selected modality
adapters that each emit a uniform :class:`~context.block.ContextBlock`, which an
:class:`~context.assembler.Assembler` orders into VLM context. See
``docs/plans/vlm-modular-preprocessing-and-context-viewer-roadmap.md``.

Import-footprint discipline: the light, pure-stdlib surface (``ContextBlock``,
``normalize_experiments``/``experiment_names``/``display_names``, presets, the
assembler) is imported eagerly. The **modality adapters** pull numpy / PIL /
pandas + project modules, so they are exposed **lazily** via ``__getattr__`` —
this keeps dependency-light consumers (the result-discovery readers, which only
need ``experiment_names``/``display_names``) from dragging the imaging/EHR stack
just to normalize experiment names.
"""

from .block import ContextBlock, TEXT, IMAGES
from .assembler import (
    Assembler,
    AssemblyError,
    preflight_assembly,
    ORDERED,
    INLINE_BY_TIMESTAMP,
)
from .normalize import (
    normalize_experiments,
    experiment_names,
    display_names,
    NormalizedExperiment,
)
from .presets import PRESETS, get_preset, is_legacy_preset

# name -> (submodule, attr) for heavy symbols, imported on first access only.
_LAZY = {
    "ModalityAdapter": ("adapters", "ModalityAdapter"),
    "ModelCaps": ("adapters", "ModelCaps"),
    "resolve_model_caps": ("adapters", "resolve_model_caps"),
    "ADAPTER_REGISTRY": ("adapters", "ADAPTER_REGISTRY"),
    "build_adapter": ("adapters", "build_adapter"),
    "CTAdapter": ("adapters", "CTAdapter"),
    "PathologyAdapter": ("adapters", "PathologyAdapter"),
    "EHRAdapter": ("adapters", "EHRAdapter"),
    "ContextComposition": ("specs", "ContextComposition"),
    "resolve_spec": ("specs", "resolve_spec"),
}


def __getattr__(name):
    """PEP 562 lazy loader for the adapter/specs symbols (numpy/PIL-heavy)."""
    if name in _LAZY:
        import importlib

        submodule, attr = _LAZY[name]
        module = importlib.import_module(f"{__name__}.{submodule}")
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    # light (eager)
    "ContextBlock",
    "TEXT",
    "IMAGES",
    "Assembler",
    "AssemblyError",
    "preflight_assembly",
    "ORDERED",
    "INLINE_BY_TIMESTAMP",
    "normalize_experiments",
    "experiment_names",
    "display_names",
    "NormalizedExperiment",
    "PRESETS",
    "get_preset",
    "is_legacy_preset",
    # heavy (lazy via __getattr__)
    "ModalityAdapter",
    "ModelCaps",
    "resolve_model_caps",
    "ADAPTER_REGISTRY",
    "build_adapter",
    "CTAdapter",
    "PathologyAdapter",
    "EHRAdapter",
    "ContextComposition",
    "resolve_spec",
]
