"""Shared weight-free context-capture core.

Extracted from ``golden_harness.capture_experiment`` so BOTH consumers run the
*same* weight-free loop and can never drift:

  * the **golden-output harness** (``golden_harness.py``) hashes each item's
    images into a byte-identity gate;
  * the **Phase-2 config-context viewer** (``src/results/context_viewer.py``)
    base64-thumbnails the same images for manual pre-run input QA.

The loop builds the orchestrator WITHOUT weights, loads the experiment data,
builds prompts, and iterates ``PromptDataset.__getitem__`` directly. At the
moment each item is yielded, ``item["image"]`` holds the fully-preprocessed PIL
image(s) the model actually receives (post-window, post-pad) — the golden hashes
them and discards; the viewer keeps them. One capture path => zero drift between
what the golden records, what the viewer shows, and what inference feeds.

BYTE-IDENTITY CONTRACT
----------------------
This module owns the loop, NOT the golden record shape. Every scalar on
``CapturedExample`` is derived with the SAME coercions the golden used
(``_to_native`` / ``_clean_text`` / ``_path_tile_count``), so ``golden_harness``
rebuilding its record from a ``CapturedExample`` changes no output bytes. The
extraction is proven a no-op by ``diff_golden --mode strict`` against a
pre-refactor bank.

WEIGHT-FREE / VM-ONLY / PHI
---------------------------
Identical posture to ``golden_harness``: it reproduces the data + prompt +
selection path without loading model weights, but it DOES construct the BigQuery
+ GCS clients (in ``TaskOrchestrator.__init__``) and reads real de-identified
data, so it runs **on the VM only**. The captured examples carry real prompts /
timelines / imagery (PHI) — never write a ``CapturedExample`` (or anything
derived from ``.images`` / ``.dynamic_prompt`` / ``.raw_row``) into the repo tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from vqa_dataset import PromptDataset
from vista_run.run_bq import RETRIEVAL_EXPERIMENTS

# Legacy assembly is implicit "images then text" for every registered adapter
# (see context/assembler.py). The golden records this constant so the post-3b
# assembler's declared mode can be diffed against it; the viewer shows it as a
# provenance chip. Shared here because both consumers read it.
LEGACY_ASSEMBLY_MODE = "ordered"


@dataclass
class CapturedExample:
    """One weight-free captured example.

    Carries every scalar the golden records PLUS the fully-preprocessed PIL
    image(s) the model receives (which the golden discards after hashing but the
    viewer thumbnails). ``image_count`` / ``path_tile_count`` / ``selected_indices``
    are surfaced for the viewer's metadata chips; ``raw_row`` is kept so the viewer
    can read task-label columns off the row without re-loading data.
    """

    index: Any
    person_id: Any
    task: str
    experiment: str
    model_type: str | None
    dynamic_prompt: str | None
    adapter_prompt_string: str | None
    assembly_mode: str
    selected_indices: Any
    image_count: int
    path_tile_count: int | None
    images: Any  # item["image"]: None | PIL.Image | list[PIL.Image]
    raw_row: Any  # the pandas Series (viewer label metadata)
    is_gemma: bool
    windowing: str


def _to_native(value):
    """Coerce a pandas/numpy scalar to a JSON-serialisable Python native (or None)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _clean_text(value):
    """Return a string for a timeline/prompt cell, or None for NaN/'nan'/empty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text == "nan":
        return None
    return text


def _path_tile_count(raw_row):
    """Count materialised pathology tiles on the row (None when not a path experiment).

    List-only by design: PathologyAdapter.materialize hands __getitem__ a real list (and the whole
    path pipeline iterates it), so a stringified path_tile_paths never reaches here. We return None
    for anything non-list rather than parse a speculative CSV-reloaded string.
    """
    tiles = raw_row.get("path_tile_paths", None)
    if isinstance(tiles, (list, tuple, np.ndarray)):
        return int(len(tiles))
    return None


def _sort_key_columns(df):
    """Prefer sorting by (person_id, index); fall back gracefully if a column is absent."""
    cols = [c for c in ("person_id", "index") if c in df.columns]
    return cols or None


def _count_images(image):
    """image_count for item['image'] (None | PIL | list-of-PIL).

    The count half of golden's ``_image_summary`` without the hashing (the viewer
    does not hash). Golden still recomputes count+hashes via ``_image_summary`` on
    the same ``.images`` object, so this is a viewer convenience, not a golden input.
    """
    if image is None:
        return 0
    if isinstance(image, list):
        return len(image)
    return 1


def _load_experiment_data(orch, task_info, experiment):
    """Resolve (df, timeline_col, source_csv) for a single experiment.

    Mirrors the per-experiment loader dispatch inside ``run_bq.run_inference``
    (the ``needs_*`` / ``loaded_*`` block) for exactly one experiment. Retrieval
    experiments are rejected — their prompt build needs model weights, so they
    are out of scope for the weight-free capture core.
    """
    if experiment in RETRIEVAL_EXPERIMENTS:
        raise ValueError(
            f"experiment '{experiment}' is a retrieval experiment; its prompt build "
            "requires model weights and is out of scope for the weight-free capture core."
        )
    if experiment == "no_timeline":
        return orch._load_task_data(task_info, use_no_report_csv=True, require_timeline=False)
    if experiment == "all_vb_image_only":
        return orch._load_all_vb_image_task_data(task_info)
    if experiment == "all_vb_timeline_only":
        return orch._load_all_vb_timeline_task_data(task_info)
    if experiment in ("path", "path_image_and_report"):
        return orch._load_path_task_data(task_info)
    if experiment == "path_full":
        return orch._load_path_full_task_data(task_info)
    if experiment in ("no_report", "timeline_only", "report"):
        return orch._load_task_data(task_info, use_no_report_csv=True, require_timeline=True)
    # default: normal timeline experiments (no_image, axial_all_image, timeline_only variants, ...)
    return orch._load_task_data(task_info, use_no_report_csv=False, require_timeline=True)


def iter_captured_examples(orch, task_info, experiment, model_type, limit=None):
    # type: (...) -> Iterator[CapturedExample]
    """Yield a ``CapturedExample`` per row for one experiment, sorted by (person_id, index).

    The weight-free loop formerly inside ``golden_harness.capture_experiment``:
    load df -> build prompts -> stable-sort by (person_id, index) -> head(limit) ->
    construct ``PromptDataset`` -> iterate ``__getitem__``. Rejects retrieval
    experiments (their prompt build needs model weights). ``limit`` caps rows AFTER
    the deterministic sort (the "verify one example first" / small-subset smoke).
    """
    loaded = _load_experiment_data(orch, task_info, experiment)
    if loaded is None:
        print(f"!!! No data loaded for experiment '{experiment}' — skipping.")
        return
    df, timeline_col, _source_csv = loaded
    df_exp = orch._build_prompts_for_experiment(df, task_info, experiment, timeline_col)

    # Impose the ordering both consumers rely on (run_bq writes as-processed +
    # appends on resume, so the ordering MUST be imposed here). index is the unique
    # per-file join key.
    sort_cols = _sort_key_columns(df_exp)
    if sort_cols:
        df_exp = df_exp.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if limit is not None:
        df_exp = df_exp.head(limit).reset_index(drop=True)

    ct_dir = orch.cfg.get("paths", {}).get("ct_dir")
    ct_snapshot_prefix = orch.cfg.get("paths", {}).get("ct_snapshot_prefix")
    dataset = PromptDataset(
        df=df_exp, prompt_col="dynamic_prompt", experiment=experiment,
        storage_client=orch.storage_client, model_type=model_type, ct_dir=ct_dir,
        ct_snapshot_prefix=ct_snapshot_prefix,
    )
    is_gemma = model_type is not None and "gemma" in model_type.lower()
    # windowing branch that produced the pixels (golden gate-2 reasoning aid): legacy
    # dispatch is is_gemma-gated; 3b moves it to by_model but the pixels stay identical
    # per model.
    windowing = "multi_window_rgb" if is_gemma else "grayscale"

    for i in range(len(dataset)):
        item = dataset[i]
        raw = item["raw_row"]
        image = item.get("image")
        yield CapturedExample(
            index=_to_native(raw.get("index", item.get("index"))),
            person_id=_to_native(raw.get("person_id")),
            task=task_info["task_name"],
            experiment=experiment,
            model_type=model_type,
            dynamic_prompt=_clean_text(raw.get("dynamic_prompt")),
            adapter_prompt_string=_clean_text(raw.get(timeline_col)) if timeline_col else None,
            assembly_mode=LEGACY_ASSEMBLY_MODE,
            selected_indices=_to_native(item.get("selected_indices")),
            image_count=_count_images(image),
            path_tile_count=_path_tile_count(raw),
            images=image,
            raw_row=raw,
            is_gemma=is_gemma,
            windowing=windowing,
        )
