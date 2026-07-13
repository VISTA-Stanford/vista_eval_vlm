"""
Task-mapping predicates — the single source of truth for "what kind of task is this".

Deliberately DEPENDENCY-FREE (no numpy / pandas / results_analyzer / context.normalize):
it is imported on the inference hot-path (`vista_run.run_bq._process_single_task_with_data`)
and by the pre-weights preflight (`eval/run_rung0_gpu.sh`), so it must import cheaply and
never pull heavy modules. `results.final_metrics` re-exports `is_binary_yes_no_task` so
existing `from results.final_metrics import is_binary_yes_no_task` callers keep working.

Why this exists: the rung-0 0b decoding no-op (2026-07-09) came from keying the Yes/No
decode constraint off a registry `is_binary` bool that a 3-class-with-Yes/No task sets to
False. The constraint must key off the task's *mapping* instead — the semantics whose
duplication caused that bug now live in one place.
"""


def is_binary_yes_no_task(mapping):
    """
    Return True if the task uses binary Yes/No answers with mapping "1" -> "Yes", "0" -> "No".

    A task can be Yes/No-scorable even when its registry marks `is_binary: False` — e.g. PFS's
    genuine 3-class mapping {"1":"Yes","0":"No","-1":"Insufficient follow-up or missing data"}
    still reads "Yes"/"No" for the "1"/"0" keys, so it IS Yes/No-mappable. This predicate keys
    only off the "1"/"0" keys; the presence of a "-1" (or any other) class does not disqualify it.
    """
    if not mapping or not isinstance(mapping, dict):
        return False
    yes_str = (mapping.get("1") or "").strip().lower()
    no_str = (mapping.get("0") or "").strip().lower()
    return yes_str == "yes" and no_str == "no"
