"""One recipe per model. ``get("argon", n_slices=25)`` returns a copy with the pilot's overrides."""

from .base import Message, Recipe  # noqa: F401
from . import argon, astra, medgemma

RECIPES = {r.name: r for r in (medgemma.RECIPE, astra.RECIPE, argon.RECIPE)}


def get(name: str, **overrides) -> Recipe:
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}")
    return RECIPES[name].replace(**overrides) if overrides else RECIPES[name]
