"""Staged golden-output harness for the Task-3b modality-adapter dissolution.

WHY THIS EXISTS
---------------
Task 3b dissolves ``vqa_dataset.__getitem__``'s duplicated CT/pathology image
branches and reroutes ``run_bq._build_prompts_for_experiment`` through the new
``src/context`` assembler + EHR adapter. On the imaging / structure surface this
is supposed to be a *behaviour-preserving no-op*, but "trust me, it's a no-op" is
not credible (windowing dispatch shifts ``is_gemma -> by_model`` and the EHR
source switches to LUMIA). So instead of trusting, we instrument
(``claude_ops.md``: "You don't trust; you instrument").

This script captures a per-example **golden tuple** from whatever code path is
live *today*, sorted by ``(person_id, index)``, and writes it as JSONL. Run it
once BEFORE 3b to bank the legacy baseline, then again AFTER 3b; ``diff_golden.py``
asserts the gates:

- **Gate 1 (byte-identical, imaging + structure):** ``dynamic_prompt`` +
  ``selected_indices`` + ``image_hashes`` + counts + ``assembly_mode`` identical.
  The true no-op proof for the CT / pathology / assembly / truncation surface.
- **Gate 2 (`by_model` delta):** run per model; image hashes identical *per model*.
- **Gate 3 (EHR LUMIA-render delta):** ``adapter_prompt_string`` (the timeline)
  before vs after, within a declared allowlist (OQ-K). Conditioned on the VM
  LUMIA field-coverage check.

WEIGHT-FREE
-----------
Reproduces the data + prompt + selection path WITHOUT loading model weights: it
calls the same loaders and ``_build_prompts_for_experiment`` the orchestrator
uses, then iterates ``PromptDataset`` directly (single-process, deterministic).
It never calls ``run_inference`` / ``_ensure_model_loaded`` and skips retrieval
experiments (which pass ``self.model``). It DOES create the BigQuery + GCS
clients (in ``TaskOrchestrator.__init__``) and reads real de-identified data, so
it runs **on the VM only** — the planner Mac has neither credentials nor data.

PHI
---
The output embeds real de-identified prompts, timelines, and image hashes ->
**PHI**. It writes only under the config's ``results_dir`` (the su-vista mount,
outside the repo tree) and is git-ignored (``*_golden.jsonl``). Never commit it;
``/phi-vet`` gates every commit in this repo.

USAGE
-----
    cd src && python -m vista_run.golden_harness \
        --config configs/all_tasks.yaml \
        --type gemma3 --name google/medgemma-1.5-4b-it \
        --task progression_recurrence_free_survival_1_yr \
        --experiments path_full \
        --tag legacy [--limit 1]

``--limit`` caps rows AFTER the ``(person_id, index)`` sort (deterministic head)
for the "verify on one example first" smoke (OQ-K); omit it for a full run.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vista_run.run_bq import TaskOrchestrator, RETRIEVAL_EXPERIMENTS
# The weight-free capture loop + its scalar-coercion helpers now live in
# context_capture (shared with the Phase-2 config-context viewer). This module
# keeps only the image *hashing* (golden-specific) and rebuilds the identical
# golden record from each CapturedExample — the extraction moved the loop, not
# the record shape or serialization (byte-identity: diff_golden --mode strict).
from vista_run.context_capture import iter_captured_examples

# Gate classification — MUST stay in lockstep with diff_golden.py.
#   STRUCTURE_FIELDS: hard byte-identity ALWAYS (Gate 1 imaging/selection/assembly + Gate 2
#     image bytes identical per model).
#   TEXT_FIELDS: gated by diff_golden --mode. strict = byte-identical (Gate 1 for the passthrough
#     post-3b run, dynamic_prompt INCLUDED); allowlist = normalised (Gate 3 for the LUMIA-live run,
#     where dynamic_prompt's embedded timeline + adapter_prompt_string legitimately change). So
#     dynamic_prompt is mode-dependent, NOT a hard structure field.
STRUCTURE_FIELDS = (
    "selected_indices",
    "image_hashes",
    "image_count",
    "path_tile_count",
    "assembly_mode",
)
TEXT_FIELDS = ("dynamic_prompt", "adapter_prompt_string")


def _hash_image(img):
    """Deterministic content hash of a single PIL image (mode+size+dtype+shape+bytes)."""
    arr = np.asarray(img)
    h = hashlib.sha256()
    h.update(f"{img.mode}|{tuple(img.size)}|{arr.dtype}|{arr.shape}|".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _image_summary(image):
    """Return (image_count, image_hashes) for item['image'] (None | PIL | list-of-PIL)."""
    if image is None:
        return 0, []
    if isinstance(image, list):
        return len(image), [_hash_image(im) for im in image]
    return 1, [_hash_image(image)]


def capture_experiment(orch, task_info, experiment, model_type, limit=None):
    """Yield golden-tuple dicts for one experiment, sorted by (person_id, index).

    Consumes the shared weight-free capture core (``iter_captured_examples``) and
    rebuilds the identical golden record from each ``CapturedExample``. Byte-identity
    is preserved because every scalar rides through unchanged from the shared
    coercions and the image count + hashes are still computed here via
    ``_image_summary(ex.images)`` (``ex.images is item["image"]``, the same object) —
    the extraction moved the loop, not the record shape or serialization.
    """
    for ex in iter_captured_examples(orch, task_info, experiment, model_type, limit=limit):
        image_count, image_hashes = _image_summary(ex.images)
        yield {
            "index": ex.index,
            "person_id": ex.person_id,
            "task": ex.task,
            "experiment": ex.experiment,
            "model_type": ex.model_type,
            # windowing branch that produced the pixels (gate-2 reasoning aid): legacy
            # dispatch is is_gemma-gated; 3b moves it to by_model but the pixels must
            # stay identical per model.
            "windowing": ex.windowing,
            "dynamic_prompt": ex.dynamic_prompt,
            "adapter_prompt_string": ex.adapter_prompt_string,
            "assembly_mode": ex.assembly_mode,
            "image_count": image_count,
            "path_tile_count": ex.path_tile_count,
            "selected_indices": ex.selected_indices,
            "image_hashes": image_hashes,
        }


def default_out_path(orch, task_name, experiment, tag):
    """Golden dump path under the config results_dir (PHI mount, git-ignored)."""
    src = None
    task_info = next((t for t in orch.valid_tasks if t["task_name"] == task_name), None)
    if task_info is not None:
        src = task_info.get("task_source_csv")
    parts = [orch.results_base, "golden"]
    if src:
        parts.append(src)
    parts.extend([task_name, orch.file_model_name])
    out_dir = Path(*[str(p) for p in parts])
    return out_dir / f"{task_name}_{experiment}_{tag}_golden.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--type", required=True, help="model_type, e.g. gemma3")
    parser.add_argument("--name", required=True, help="model HF path, e.g. google/medgemma-1.5-4b-it")
    parser.add_argument("--task", required=True, help="single task_name to capture")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="experiments to capture (default: config 'experiments'); retrieval experiments are rejected")
    parser.add_argument("--tag", default="golden", help="label for this run, e.g. legacy | post3b")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows AFTER the (person_id, index) sort (smoke: verify one example first)")
    parser.add_argument("--out", default=None, help="override output JSONL path (default: results_dir/golden/...)")
    args = parser.parse_args()

    orch = TaskOrchestrator(args.config, args.type, args.name)
    experiments = args.experiments if args.experiments else orch.cfg.get("experiments", ["no_image"])

    task_info = next((t for t in orch.valid_tasks if t["task_name"] == args.task), None)
    if task_info is None:
        raise SystemExit(f"task '{args.task}' not found in valid_tasks ({args.config})")

    if args.out is not None:
        # --out is a single-experiment escape hatch. Guard the PHI contract: it must carry the
        # _golden.jsonl suffix so the .gitignore backstop always applies wherever it lands, and it
        # cannot be shared across experiments (each would clobber the same file).
        if len(experiments) > 1:
            raise SystemExit(
                "--out cannot be combined with multiple experiments (each would clobber the same "
                "file); run one experiment at a time, or omit --out for the results_dir default."
            )
        if not str(args.out).endswith("_golden.jsonl"):
            raise SystemExit(
                "--out must end with '_golden.jsonl' so the PHI .gitignore backstop applies "
                "(golden output embeds de-identified timelines/prompts)."
            )

    for experiment in experiments:
        if experiment in RETRIEVAL_EXPERIMENTS:
            print(f"### Skipping retrieval experiment '{experiment}' (needs model weights; out of scope).")
            continue

        out_path = Path(args.out) if args.out else default_out_path(orch, args.task, experiment, args.tag)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = 0
        with open(out_path, "w") as f:
            for record in capture_experiment(orch, task_info, experiment, args.type, limit=args.limit):
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                n += 1

        meta = {
            "config": args.config,
            "task": args.task,
            "experiment": experiment,
            "model_type": args.type,
            "model_name": args.name,
            "tag": args.tag,
            "limit": args.limit,
            "row_count": n,
            "structure_fields": list(STRUCTURE_FIELDS),
            "text_fields": list(TEXT_FIELDS),
            "captured_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        print(f">>> Wrote {n} golden rows -> {out_path}")
        print(f"    meta -> {meta_path}")


if __name__ == "__main__":
    main()
