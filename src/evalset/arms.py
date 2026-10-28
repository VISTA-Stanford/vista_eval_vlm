"""Named variants of a model recipe, kept in a small YAML file so a run's settings live in one reviewable place.

    variants:
      argon_25:            # name used in the output folder: runs/<run>/argon__argon_25/
        recipe: argon      # optional; must match --recipe when given
        n_slices: 25
        n_slices_pair: 12
      astra_medium:
        recipe: astra
        settings: {reasoning_effort: medium}

Any ``Recipe`` field can be overridden; ``settings`` merges into the recipe's own. The file's path, its sha256, the
variant name and the overrides go into every run record, so a result always names the exact settings it came from.
Used by ``python -m evalset.run`` and ``python -m evalset.packs`` (``--arms FILE --variant NAME``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import yaml

from evalset.recipes import Recipe

FIELDS = {f.name for f in dataclasses.fields(Recipe)} - {"name"}


def load(path, variant: str) -> tuple[dict, dict]:
    """(overrides, provenance) for one variant of the arms file at ``path``."""
    raw = Path(path).read_bytes()
    doc = yaml.safe_load(raw) or {}
    variants = doc.get("variants") or {}
    if variant not in variants:
        raise KeyError(f"{path}: no variant {variant!r}; has {sorted(variants)}")
    over = dict(variants[variant] or {})
    unknown = set(over) - FIELDS - {"recipe"}
    if unknown:
        raise ValueError(f"{path}: variant {variant!r} sets unknown recipe fields {sorted(unknown)}")
    prov = {"arms_file": str(path), "arms_sha256": hashlib.sha256(raw).hexdigest(), "variant": variant,
            "overrides": over}
    return over, prov


def apply(recipe: Recipe, variant: str, over: dict) -> Recipe:
    """The recipe with the variant's overrides; its name becomes ``<recipe>__<variant>`` (the output folder)."""
    want = over.get("recipe")
    if want and want != recipe.name:
        raise ValueError(f"variant {variant!r} is for recipe {want!r}, not {recipe.name!r}")
    kw = {k: v for k, v in over.items() if k not in ("recipe", "settings")}
    if "settings" in over:
        merged = dict(recipe.settings)
        merged.update(over["settings"] or {})
        kw["settings"] = tuple(merged.items())
    return recipe.replace(name=f"{recipe.name}__{variant}", **kw)
