"""Gate the Task-3b dissolution by diffing two golden-output dumps.

Consumes two JSONL files produced by ``golden_harness.py`` (a BEFORE / legacy
baseline and an AFTER / post-3b capture) for the *same* (task, experiment, model),
joins them on ``index``, and applies the plan's verification gates:

- **Structure (hard, always):** ``image_hashes``, ``selected_indices``,
  ``image_count``, ``path_tile_count``, ``assembly_mode`` must be byte-identical.
  This is Gate 1 (imaging + selection + assembly no-op) AND Gate 2 (image bytes
  identical *per model* — you run this diff once per model file pair). Any drift
  here is a hard stop.
- **Text (mode-dependent):** ``dynamic_prompt`` + ``adapter_prompt_string``.
  - ``--mode strict`` (default; use for the passthrough post-3b run, where LUMIA-live
    is OFF and everything should be byte-identical): text must be byte-identical too.
  - ``--mode allowlist`` (use for the LUMIA-live run): text is compared after a
    normaliser; a normalised match is a *declared delta* (OQ-K — within-line field
    order / whitespace only, event order still strict) and passes. A residual
    difference is reported for human adjudication and fails unless ``--lenient``.

Exit code 0 = all gates pass; 1 = a gate failed.

Run once per (experiment, model) pair, e.g.::

    cd src && python -m vista_run.diff_golden \
        "$RESULTS/golden/.../TASK_path_full_legacy_golden.jsonl" \
        "$RESULTS/golden/.../TASK_path_full_post3b_golden.jsonl" \
        --mode strict

The ``normalize_text`` hook below is intentionally minimal (trailing-whitespace
only). The real within-line field-order allowlist gets filled in once the VM shows
the actual LUMIA-render vs ``patient_string`` diffs — do not speculate it now.
"""

import argparse
import json
import sys

# Fields whose drift is ALWAYS a hard failure (imaging / selection / assembly). In lockstep
# with golden_harness.STRUCTURE_FIELDS.
STRUCTURE_FIELDS = (
    "selected_indices",
    "image_hashes",
    "image_count",
    "path_tile_count",
    "assembly_mode",
)
# Fields governed by --mode. strict = byte-identical (this is where the plan's Gate-1 "full-string
# dynamic_prompt byte-identical" is enforced, for the passthrough post-3b run); allowlist =
# normalised (Gate 3, for the LUMIA-live run, where dynamic_prompt's embedded timeline +
# adapter_prompt_string legitimately change). In lockstep with golden_harness.TEXT_FIELDS.
TEXT_FIELDS = (
    "dynamic_prompt",
    "adapter_prompt_string",
)


def normalize_text(value):
    """Allowlist normaliser for the EHR/prompt strings (Gate 3, OQ-K).

    Currently strips trailing whitespace per line only — a safe, event-order-preserving
    normalisation. The declared within-line field-order allowlist is added here once the
    VM surfaces the real LUMIA-render vs patient_string deltas; keeping it minimal avoids
    baking in a speculative equivalence that could mask a genuine regression.
    """
    if value is None:
        return None
    return "\n".join(line.rstrip() for line in str(value).splitlines())


def _load(path):
    """Load a golden JSONL into {index: record}; error on duplicate indices."""
    rows = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("index")
            if idx in rows:
                raise SystemExit(f"{path}:{line_no}: duplicate index {idx!r} — golden must be unique per index")
            rows[idx] = rec
    return rows


def _preview(value, width=160):
    s = json.dumps(value, ensure_ascii=False)
    return s if len(s) <= width else s[:width] + "…"


def _constant_field(rows, field, path):
    """Return the single value of `field` across all rows, or error if it varies."""
    values = {rec.get(field) for rec in rows.values()}
    if len(values) > 1:
        raise SystemExit(
            f"{path}: golden rows disagree on '{field}' ({sorted(map(str, values))}) — "
            "not a single (task, experiment, model) dump"
        )
    return next(iter(values)) if values else None


def _check_compatible(before_rows, after_rows, before_path, after_path):
    """Refuse to diff two dumps that are not the same task / experiment / model.

    The join is on `index` alone, so an accidental cross-experiment (or cross-model) pair would
    otherwise produce a confidently-wrong gate report. Fail fast instead.
    """
    for field in ("task", "experiment", "model_type"):
        b = _constant_field(before_rows, field, before_path)
        a = _constant_field(after_rows, field, after_path)
        if b != a:
            raise SystemExit(
                f"metadata mismatch on '{field}': BEFORE={b!r} AFTER={a!r} — "
                "these dumps are not the same (task, experiment, model); refusing to diff."
            )


def diff(before, after, mode, lenient, max_report):
    before_rows = _load(before)
    after_rows = _load(after)
    _check_compatible(before_rows, after_rows, before, after)

    before_ids = set(before_rows)
    after_ids = set(after_rows)
    missing = before_ids - after_ids
    extra = after_ids - before_ids
    shared = sorted(before_ids & after_ids, key=lambda x: (str(type(x)), x))

    structure_fail = []
    text_fail = []
    text_declared = []

    for idx in shared:
        b = before_rows[idx]
        a = after_rows[idx]
        for field in STRUCTURE_FIELDS:
            if b.get(field) != a.get(field):
                structure_fail.append((idx, field, b.get(field), a.get(field)))
        for field in TEXT_FIELDS:
            bv, av = b.get(field), a.get(field)
            if bv == av:
                continue
            if mode == "allowlist" and normalize_text(bv) == normalize_text(av):
                text_declared.append((idx, field))
            else:
                text_fail.append((idx, field, bv, av))

    # ---- report ----
    print(f"BEFORE: {before}  ({len(before_rows)} rows)")
    print(f"AFTER : {after}  ({len(after_rows)} rows)")
    print(f"mode  : {mode}    shared indices: {len(shared)}")
    print("-" * 72)

    ok = True

    if missing or extra:
        ok = False
        print(f"[FAIL] index-set mismatch: {len(missing)} only-in-BEFORE, {len(extra)} only-in-AFTER")
        for idx in list(sorted(missing, key=str))[:max_report]:
            print(f"       only in BEFORE: index={idx!r}")
        for idx in list(sorted(extra, key=str))[:max_report]:
            print(f"       only in AFTER : index={idx!r}")

    if structure_fail:
        ok = False
        print(f"[FAIL] structure drift (Gate 1/2, hard): {len(structure_fail)} field mismatches")
        for idx, field, bv, av in structure_fail[:max_report]:
            print(f"       index={idx!r} field={field}")
            print(f"         BEFORE: {_preview(bv)}")
            print(f"         AFTER : {_preview(av)}")
    else:
        print("[PASS] structure (image_hashes / selected_indices / counts / assembly_mode) byte-identical")

    if text_declared:
        print(f"[NOTE] {len(text_declared)} declared text deltas (Gate 3 allowlist-normalised equal)")

    if text_fail:
        label = "FAIL" if not lenient else "WARN"
        if not lenient:
            ok = False
        print(f"[{label}] text drift ({mode}): {len(text_fail)} field mismatches "
              f"({'lenient — not failing' if lenient else 'hard'})")
        for idx, field, bv, av in text_fail[:max_report]:
            print(f"       index={idx!r} field={field}")
            print(f"         BEFORE: {_preview(bv)}")
            print(f"         AFTER : {_preview(av)}")
    else:
        print(f"[PASS] text (dynamic_prompt / adapter_prompt_string) within {mode} gate")

    if not shared:
        ok = False
        print("[WARN] no shared indices — nothing was compared; a no-op cannot be proven from an "
              "empty intersection (check the BEFORE/AFTER pair and their sort keys)")

    print("-" * 72)
    print("RESULT: " + ("ALL GATES PASS" if ok else "GATE FAILURE"))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="legacy/baseline golden JSONL")
    parser.add_argument("after", help="post-3b golden JSONL")
    parser.add_argument("--mode", choices=("strict", "allowlist"), default="strict",
                        help="strict = byte-identical text (passthrough run); allowlist = Gate-3 normalised text (LUMIA-live run)")
    parser.add_argument("--lenient", action="store_true",
                        help="report residual text deltas as WARNINGS instead of failing — a local aid "
                             "while building the Gate-3 allowlist, NOT for the committed/VM gate run "
                             "(it never masks structure drift, but it can exit 0 despite text drift)")
    parser.add_argument("--max-report", type=int, default=20, help="max mismatches to print per section")
    args = parser.parse_args()

    ok = diff(args.before, args.after, args.mode, args.lenient, args.max_report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
