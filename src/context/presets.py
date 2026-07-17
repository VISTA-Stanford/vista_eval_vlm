"""Legacy experiment name -> block composition (back-compat).

Each bare legacy ``experiment`` string maps to a composition of adapter blocks +
an ``assembly`` + a ``prompt_style`` + a ``cohort`` loader key. This is the exact
back-compat spec: ``normalize_experiments`` expands a bare string through here so
old configs keep working unchanged.

Two axes stay name-dispatched in Phase 1 (dissolving them is out of scope):
  * ``cohort`` — which ``run_bq._load_*`` runs (the cohort axis is Phase 3).
  * ``prompt_style`` — how the orchestrator stitches the task template with the
    EHR timeline / report / path note (text composition).
What the presets DO dissolve is the image + EHR *modality* selection: the CT
slice counts, the pathology tile sampling, and the EHR timeline source now live
in typed ``blocks`` instead of branching inside ``__getitem__`` /
``_build_prompts_for_experiment``.

CT slice counts preserved from legacy (post Phase-0.5): axial_all_image=30,
no_report=10, no_timeline / all_vb_image_only=50, retrieval+image variants=30.
"""

from __future__ import annotations

import copy

# Reusable block fragments -----------------------------------------------------
def _ehr_block(variant: str = "timeline"):
    """EHR text block. ``variant`` records which timeline source the cohort feeds
    (``timeline`` = full, ``no_img_report`` = imaging-report-skipped,
    ``passthrough`` = pre-rendered from parquet/retrieval).

    Concrete LUMIA-live ``select`` chains (wired in the Step 5 hot-path pass,
    `docs/plans/vlm-step5-lumia-live-ehr-adapter.md`):
      * ``no_img_report`` -> ``[{fn: window, before: "6mo", after: "0d"},
        {fn: code_filter, exclude_stanford: true}]`` (reproduces the legacy
        ``get_described_events_window`` window + ``exclude_report=True`` STANFORD skip).
      * ``timeline`` -> ``select=[]`` (full timeline, unfiltered) pending that plan's
        Verification Phase 1 VM decision gate: whether the full timeline should also
        drop STANFORD-tagged codes like ``no_img_report`` does. Flip this one line to
        add ``code_filter(exclude_stanford=True)`` if the gate resolves that way —
        single edit point, not a second runtime toggle.
      * ``passthrough`` -> ``select`` key omitted entirely (the adapter auto-passes a
        pre-rendered timeline string through; filters are inapplicable to already-
        rendered text).
    """
    if variant == "no_img_report":
        select = [
            {"fn": "window", "before": "6mo", "after": "0d"},
            {"fn": "code_filter", "exclude_stanford": True},
        ]
    elif variant == "timeline":
        select = []  # Verification Phase 1 (see docstring) may flip this to add code_filter
    else:  # passthrough
        select = None
    config = {"variant": variant, "serialize": {"style": "flat_timeline"}}
    if select is not None:
        config["select"] = select
    return {"id": "ehr", "modality": "text", "adapter": "ehr", "config": config}


def _ct_block(k: int):
    return {
        "id": "ct",
        "modality": "volume3d",
        "adapter": "ct",
        "config": {
            "select": {"fn": "evenly_spaced_k", "k": k},
            "preprocess": {"windowing": "by_model", "target_size": "by_model"},
        },
    }


def _path_block(n: int = 100, seed: int = 42):
    return {
        "id": "path",
        "modality": "patches2d",
        "adapter": "pathology",
        "config": {"select": {"fn": "random_n", "n": n, "seed": seed}},
    }


# name -> preset. `cohort` + `prompt_style` stay name-dispatched (Phase 1);
# `blocks` carry the dissolved modality selection.
PRESETS: dict[str, dict] = {
    "no_image": {
        "cohort": "normal", "assembly": "ordered", "prompt_style": "timeline",
        "blocks": [_ehr_block("timeline")],
    },
    "axial_all_image": {
        "cohort": "normal", "assembly": "ordered", "prompt_style": "timeline",
        "blocks": [_ehr_block("timeline"), _ct_block(30)],
    },
    "no_timeline": {
        "cohort": "no_timeline", "assembly": "ordered", "prompt_style": "image",
        "blocks": [_ct_block(50)],
    },
    "no_report": {
        "cohort": "no_report", "assembly": "ordered", "prompt_style": "timeline",
        "blocks": [_ehr_block("no_img_report"), _ct_block(10)],
    },
    "timeline_only": {
        "cohort": "no_report", "assembly": "ordered", "prompt_style": "timeline",
        "blocks": [_ehr_block("no_img_report")],
    },
    "report": {
        "cohort": "no_report", "assembly": "ordered", "prompt_style": "report",
        "blocks": [_ehr_block("no_img_report")],
    },
    "all_vb_timeline_only": {
        "cohort": "all_vb_timeline", "assembly": "ordered", "prompt_style": "timeline",
        "blocks": [_ehr_block("passthrough")],
    },
    "all_vb_image_only": {
        "cohort": "all_vb_image", "assembly": "ordered", "prompt_style": "image",
        "blocks": [_ct_block(50)],
    },
    "path": {
        "cohort": "path", "assembly": "ordered", "prompt_style": "image",
        "blocks": [_path_block()],
    },
    "path_image_and_report": {
        "cohort": "path", "assembly": "ordered", "prompt_style": "path_image_and_report",
        "blocks": [_path_block()],
    },
    "path_full": {
        "cohort": "path_full", "assembly": "ordered", "prompt_style": "path_full",
        "blocks": [_ehr_block("passthrough"), _path_block()],
    },
}

# Retrieval experiments stay on the legacy retrieval prompt-building path (the
# cohort/prompt is produced by retrieval.prompt_building, not decomposed into
# blocks here). ``_with_image`` variants additionally attach 30 CT slices.
_RETRIEVAL_NAMES = {
    "retrieved_timeline": [],
    "retrieved_timeline_per_iteration": [],
    "retrieved_timeline_per_iteration_summarization": [],
    "retrieved_timeline_with_image": [_ct_block(30)],
    "retrieved_timeline_per_iteration_summarization_with_image": [_ct_block(30)],
}
for _name, _img_blocks in _RETRIEVAL_NAMES.items():
    PRESETS[_name] = {
        "cohort": "retrieval", "assembly": "ordered", "prompt_style": "retrieval",
        "blocks": list(_img_blocks), "legacy_retrieval": True,
    }


def is_legacy_preset(name: str) -> bool:
    return name in PRESETS


def get_preset(name: str) -> dict:
    """Return a deep copy of the preset composition for ``name``."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown legacy experiment {name!r}; known presets: {sorted(PRESETS)}"
        )
    return copy.deepcopy(PRESETS[name])
