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
    out.setdefault("question", shared.get("question"))
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


# Settings a panel's members take from the panel entry.
_PANEL_INHERITED = ("task_source_csv", "dataset", "loader", "label_from_own_report", "group_name")


def resolve_panel(entry, manifest=None):
    """A panel entry (``panel_tasks``: shared tasks asked together, one call per scan) -> (panel, members).

    Each member is resolved like any ``shared_task`` entry and carries ``panel_of``, so scoring treats it
    as an ordinary task once the panel's answers are split per task. The panel itself holds the members
    in order (``panel``) and samples its scans on ``sample_task``'s labels.
    """
    manifest = manifest if manifest is not None else _manifest()
    members = []
    for name in entry["panel_tasks"]:
        m = {"task_name": name, "shared_task": name, "panel_of": entry["task_name"]}
        m.update({k: entry[k] for k in _PANEL_INHERITED if k in entry})
        members.append(resolve_shared(m, manifest))
    sample_task = entry.get("sample_task", entry["panel_tasks"][0])
    if sample_task not in entry["panel_tasks"]:
        raise ValueError(f"{entry['task_name']}: sample_task {sample_task!r} is not one of its panel_tasks")
    if len({m["row_key"] for m in members if "row_key" in m}) > 1:
        raise ValueError(f"{entry['task_name']}: panel members key their rows differently")
    panel = dict(entry)
    panel.update({"panel": members, "sample_task": sample_task, "row_key": members[0].get("row_key", "person_id")})
    return panel, members


def load_valid_tasks(path, manifest=None):
    """The registry list at ``path``: each ``shared_task`` entry resolved, each panel followed by its members."""
    with open(path) as f:
        entries = json.load(f)
    out = []
    for e in entries:
        if e.get("panel_tasks"):
            panel, members = resolve_panel(e, manifest)
            out += [panel] + members
        else:
            out.append(resolve_shared(e, manifest))
    return out
