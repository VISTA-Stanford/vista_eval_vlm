"""normalize_experiments: the single contract for reading `experiments`.

Three result-discovery scripts used to read ``experiments`` straight from the
YAML and break on non-string entries (e.g. ``set(config["experiments"])`` ->
``TypeError: unhashable type: 'dict'``). They now all route through here.

``normalize_experiments(cfg) -> list[NormalizedExperiment]`` resolves BOTH a bare
legacy string (expanded via :mod:`context.presets`) AND a block-list dict to a
uniform spec ``{name, display, cohort, assembly, blocks}``:

  * ``name``    -> filename / resume / metrics token (must be a stable string)
  * ``display`` -> plot label (config ``display:`` or falls back to ``name``)
  * ``cohort``  -> loader key (name-dispatched in Phase 1; cohort axis is Phase 3)
  * ``assembly``-> ``ordered`` (``inline_by_timestamp`` gated in Phase 1.5)
  * ``blocks``  -> typed modality blocks (``{id, modality, adapter, config}``)

Every reader consumes normalized names/labels, never raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .presets import get_preset, is_legacy_preset


@dataclass
class NormalizedExperiment:
    name: str
    display: str
    cohort: str
    assembly: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    # `prompt_style` + `legacy_retrieval` are Phase-1 name-dispatch details the
    # orchestrator wiring needs (how the task template stitches with the EHR
    # timeline / report / path note; which cohort loader). They ride alongside
    # the spec but are NOT part of the plan's `{name, display, cohort, assembly,
    # blocks}` contract, so `.spec` deliberately omits them.
    prompt_style: str = "timeline"
    legacy_retrieval: bool = False
    unknown: bool = False  # bare string with no known preset (tolerant for readers)
    raw: Any = None  # the original config entry, for provenance

    def as_tuple(self) -> tuple[str, str, dict]:
        """``(name, display, spec)`` — the plan's stated return shape."""
        return (self.name, self.display, self.spec)

    @property
    def spec(self) -> dict:
        return {
            "name": self.name,
            "display": self.display,
            "cohort": self.cohort,
            "assembly": self.assembly,
            "blocks": self.blocks,
        }


def _normalize_entry(entry: Any) -> NormalizedExperiment:
    # bare legacy string
    if isinstance(entry, str):
        if is_legacy_preset(entry):
            preset = get_preset(entry)
            return NormalizedExperiment(
                name=entry,
                display=entry,
                cohort=preset.get("cohort", "normal"),
                assembly=preset.get("assembly", "ordered"),
                blocks=preset.get("blocks", []),
                prompt_style=preset.get("prompt_style", "timeline"),
                legacy_retrieval=preset.get("legacy_retrieval", False),
                raw=entry,
            )
        # Unknown bare string: tolerant passthrough. Result-discovery readers match
        # existing result files by name and must not crash on a custom experiment
        # name; the orchestrator validates runnability separately (`unknown=True`).
        return NormalizedExperiment(
            name=entry, display=entry, cohort="normal", assembly="ordered",
            blocks=[], unknown=True, raw=entry,
        )
    # block-list dict
    if isinstance(entry, dict):
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(
                f"experiment dict entry must carry a string 'name'; got {entry!r}"
            )
        # A dict may still reference a legacy preset by name and only override
        # some fields (e.g. cohort). Start from the preset if one exists.
        base = get_preset(name) if is_legacy_preset(name) else {}
        cohort = entry.get("cohort", base.get("cohort", "normal"))
        assembly = entry.get("assembly", base.get("assembly", "ordered"))
        blocks = entry.get("blocks", base.get("blocks", []))
        return NormalizedExperiment(
            name=name,
            display=entry.get("display", name),
            cohort=cohort,
            assembly=assembly,
            blocks=blocks,
            prompt_style=entry.get("prompt_style", base.get("prompt_style", "timeline")),
            legacy_retrieval=base.get("legacy_retrieval", False),
            raw=entry,
        )
    raise ValueError(
        f"experiment entry must be a string or a dict, got {type(entry).__name__}: {entry!r}"
    )


def normalize_experiments(cfg: dict) -> list[NormalizedExperiment]:
    """Resolve ``cfg['experiments']`` to a list of :class:`NormalizedExperiment`.

    Accepts a parsed config dict (or anything with ``.get('experiments')``).
    Defaults to ``['no_image']`` when unset (matching ``run_bq``).
    """
    experiments = cfg.get("experiments", ["no_image"]) if hasattr(cfg, "get") else cfg
    if experiments is None:
        experiments = ["no_image"]
    return [_normalize_entry(e) for e in experiments]


def experiment_names(cfg: dict) -> list[str]:
    """The ``name`` tokens (filename/resume/metrics keys), in config order."""
    return [e.name for e in normalize_experiments(cfg)]


def display_names(cfg: dict) -> dict[str, str]:
    """``name -> display`` map for plot labels."""
    return {e.name: e.display for e in normalize_experiments(cfg)}
