"""The VLM task list (``paths.valid_tasks``), with shared-registry facts filled in.

An entry may name ``shared_task``: a vista_bench task whose facts come from the registry
snapshot vista-eval vendors (``vistabench_eval.tasks``, the same one its probes read), not from
this repo's JSON. Filled in: ``mapping`` / ``label_meaning`` (class names), ``choices`` (the
scored classes, in label order), ``unscored_labels`` (the negative labels), ``is_binary``,
``grain`` and ``inputs``. The entry keeps only what is the VLM's own: the table, the row key,
the loader settings and (in the prompts file) the question wording. An entry that also writes
one of the filled fields by hand is refused, so the two lists cannot drift apart.

Plan: planning/vista_eval_vlm/vlm-radiology-progression-codex.md (on the mount).
"""

from __future__ import annotations

import json

SHARED_FIELDS = ("mapping", "label_meaning", "choices", "unscored_labels", "is_binary", "grain", "inputs")


def _manifest():
    # Imported lazily: only registries with a `shared_task` entry need the package, so the
    # board-task runs on the GPU boxes keep working without it.
    from vistabench_eval import tasks as vb_tasks
    return vb_tasks.get_manifest()


def resolve_shared(entry, manifest=None):
    """One entry with its ``shared_task`` facts filled in (entries without one pass through)."""
    name = entry.get("shared_task")
    if not name:
        return entry
    hand = [f for f in SHARED_FIELDS if f in entry]
    if hand:
        raise ValueError(f"{entry.get('task_name')}: {hand} come from shared task {name!r}; "
                         "remove them from the VLM registry entry")
    manifest = manifest if manifest is not None else _manifest()
    if name not in manifest.get("tasks", {}):
        raise ValueError(f"{entry.get('task_name')}: shared task {name!r} is not in the vista_bench registry snapshot")
    shared = manifest["tasks"][name]
    mapping = {str(k): v for k, v in shared["label_mapping"].items()}
    unscored = sorted(int(k) for k in mapping if int(k) < 0)
    scored = sorted(int(k) for k in mapping if int(k) >= 0)
    out = dict(entry)
    out.update({
        "mapping": mapping,
        "label_meaning": dict(mapping),
        "choices": [mapping[str(k)] for k in scored],
        "unscored_labels": unscored,
        "is_binary": bool(shared["is_binary"]),
        "grain": shared.get("grain", "person"),
        "inputs": shared.get("inputs", "current"),
    })
    if out["grain"] == "per_ct_landmark":
        out.setdefault("row_key", "ct_accession_number")
    if out["inputs"] == "current_and_prior" and not out.get("prior_ct"):
        raise ValueError(f"{entry.get('task_name')}: {name!r} is scored on current and prior CT, "
                         "but the entry has no prior_ct link")
    return out


def load_valid_tasks(path, manifest=None):
    """The registry list at ``path``, each ``shared_task`` entry resolved."""
    with open(path) as f:
        entries = json.load(f)
    return [resolve_shared(e, manifest) for e in entries]
