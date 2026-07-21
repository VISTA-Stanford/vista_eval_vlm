"""Gate the Task-3b dissolution by diffing two golden-output dumps.

Consumes two JSONL files produced by ``golden_harness.py`` (a BEFORE / legacy
baseline and an AFTER / post-3b capture) for the *same* (task, experiment, model),
joins them on ``person_id``, and applies the plan's verification gates:

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

``--exclude-line-patterns`` is a **separate, distinct mechanism** from ``normalize_text``
(added for the Step-5 LUMIA-live re-plan, ``docs/plans/vlm-step5-lumia-demographics-flowsheet-replan.md``):
``normalize_text``'s allowlist is deliberately line-count- and event-order-preserving
(formatting-only normalization). ``--exclude-line-patterns`` instead drops whole lines
containing a declared substring from **both** sides before any comparison — for a
*permanent, event-count-changing* divergence between the legacy and live renders (e.g.
``STANFORD_OBS/Flowsheet`` observations, upstream-excluded from the live LUMIA source but
still present in the legacy ``patient_string`` baseline). Default empty, so existing
non-EHR callers are unaffected. Do not confuse the two: a normalizer match means "same
content, different formatting"; an excluded-line match means "this class is declared
out of scope for this gate, not verified here at all."

``--exclude-if-legacy-missing`` is a **third, per-row-conditional** variant, for classes
that legacy sometimes has and sometimes doesn't (unlike ``STANFORD_OBS``, which legacy
*always* has). For each row independently: if BEFORE contains none of the given patterns,
those patterns are stripped from **both** BEFORE and AFTER for that row only (AFTER's
extra content is accepted as a richer/correct render with no baseline to check against);
if BEFORE *does* contain a matching class, nothing is stripped and the normal comparison
verifies it byte-for-byte. Added for the demographics fix (``MEDS_BIRTH``/``Ethnicity``/
``Race``/``Gender``): some legacy golden rows predate the demographic fields entirely (a
legacy-vintage omission, not a bug), so those rows can't prove a byte match for a class
that was never captured in the baseline — but rows where legacy *does* carry it still get
verified.

The join key is ``person_id``, not ``index``. ``index`` (``golden_harness``'s ``ex.index``)
is a positional BQ row enumeration (``df['index'] = df.index`` in ``run_bq.py``) — stable
only when BEFORE/AFTER share the exact same query row count and order. A dataset-version
bump that adds/reorders source rows shifts every downstream ``index`` even for patients
whose content is unchanged, so an index join across such a bump yields a spuriously empty
intersection (found the hard way diffing the v1_5→v1_6 bump, Step 6 Phase 0 Step 0b:
content was proven byte-identical, but the index join produced 0 shared rows). Every task
table is one-row-per-person (``vista_bench``'s task-registry invariant — see
``docs/plans/vlm-step6-pathology-live-substrate.md``), so ``person_id`` is the actual
identity key and is stable regardless of row-count/order changes upstream; ``_load`` still
errors loudly on a duplicate ``person_id`` within one file, so this doesn't silently paper
over a real violation of that invariant.
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


def strip_excluded_lines(value, patterns):
    """Drop any line containing a declared-excluded substring — a permanent, accepted
    divergence (e.g. ``STANFORD_OBS/Flowsheet``), not a formatting delta. See the module
    docstring for how this differs from ``normalize_text``.
    """
    if value is None or not patterns:
        return value
    lines = str(value).splitlines()
    kept = [line for line in lines if not any(p in line for p in patterns)]
    return "\n".join(kept)


def strip_if_legacy_missing(bv, av, patterns):
    """Per-row conditional variant of ``strip_excluded_lines`` — see the module docstring's
    ``--exclude-if-legacy-missing`` section. Only strips when BEFORE has no matching class;
    when BEFORE has it, both sides pass through unchanged so the class is still verified.
    """
    if not patterns:
        return bv, av
    before_has_class = bv is not None and any(p in bv for p in patterns)
    if before_has_class:
        return bv, av
    return strip_excluded_lines(bv, patterns), strip_excluded_lines(av, patterns)


def _load(path):
    """Load a golden JSONL into {person_id: record}; error on duplicate person_ids."""
    rows = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("person_id")
            if pid in rows:
                raise SystemExit(f"{path}:{line_no}: duplicate person_id {pid!r} — golden must be one row per person")
            rows[pid] = rec
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

    The join is on `person_id` alone, so an accidental cross-experiment (or cross-model) pair
    would otherwise produce a confidently-wrong gate report. Fail fast instead.
    """
    for field in ("task", "experiment", "model_type"):
        b = _constant_field(before_rows, field, before_path)
        a = _constant_field(after_rows, field, after_path)
        if b != a:
            raise SystemExit(
                f"metadata mismatch on '{field}': BEFORE={b!r} AFTER={a!r} — "
                "these dumps are not the same (task, experiment, model); refusing to diff."
            )


def diff(before, after, mode, lenient, max_report, exclude_line_patterns=(), exclude_if_legacy_missing=()):
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

    for pid in shared:
        b = before_rows[pid]
        a = after_rows[pid]
        for field in STRUCTURE_FIELDS:
            if b.get(field) != a.get(field):
                structure_fail.append((pid, field, b.get(field), a.get(field)))
        for field in TEXT_FIELDS:
            bv, av = b.get(field), a.get(field)
            bv = strip_excluded_lines(bv, exclude_line_patterns)
            av = strip_excluded_lines(av, exclude_line_patterns)
            bv, av = strip_if_legacy_missing(bv, av, exclude_if_legacy_missing)
            if bv == av:
                continue
            if mode == "allowlist" and normalize_text(bv) == normalize_text(av):
                text_declared.append((pid, field))
            else:
                text_fail.append((pid, field, bv, av))

    # ---- report ----
    print(f"BEFORE: {before}  ({len(before_rows)} rows)")
    print(f"AFTER : {after}  ({len(after_rows)} rows)")
    print(f"mode  : {mode}    shared person_ids: {len(shared)}")
    if exclude_line_patterns:
        print(f"exclude-line-patterns (declared-excluded, stripped from both sides): {list(exclude_line_patterns)}")
    if exclude_if_legacy_missing:
        print(f"exclude-if-legacy-missing (stripped per-row only where BEFORE lacks the class): {list(exclude_if_legacy_missing)}")
    print("-" * 72)

    ok = True

    if missing or extra:
        ok = False
        print(f"[FAIL] person_id-set mismatch: {len(missing)} only-in-BEFORE, {len(extra)} only-in-AFTER")
        for pid in list(sorted(missing, key=str))[:max_report]:
            print(f"       only in BEFORE: person_id={pid!r}")
        for pid in list(sorted(extra, key=str))[:max_report]:
            print(f"       only in AFTER : person_id={pid!r}")

    if structure_fail:
        ok = False
        print(f"[FAIL] structure drift (Gate 1/2, hard): {len(structure_fail)} field mismatches")
        for pid, field, bv, av in structure_fail[:max_report]:
            print(f"       person_id={pid!r} field={field}")
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
        for pid, field, bv, av in text_fail[:max_report]:
            print(f"       person_id={pid!r} field={field}")
            print(f"         BEFORE: {_preview(bv)}")
            print(f"         AFTER : {_preview(av)}")
    else:
        print(f"[PASS] text (dynamic_prompt / adapter_prompt_string) within {mode} gate")

    if not shared:
        ok = False
        print("[WARN] no shared person_ids — nothing was compared; a no-op cannot be proven from an "
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
    parser.add_argument("--exclude-line-patterns", nargs="*", default=(), metavar="SUBSTRING",
                        help="declared-excluded class(es) — lines containing any of these substrings "
                             "are dropped from BOTH before/after text before comparison. A permanent, "
                             "accepted divergence (e.g. STANFORD_OBS/Flowsheet), NOT a formatting delta "
                             "— distinct from --mode allowlist's normalize_text. Default: none.")
    parser.add_argument("--exclude-if-legacy-missing", nargs="*", default=(), metavar="SUBSTRING",
                        help="per-row conditional variant: lines matching these substrings are "
                             "stripped from BOTH sides only for rows where BEFORE has none of them "
                             "(no baseline to check against); rows where BEFORE has the class are "
                             "verified normally. For classes legacy sometimes has and sometimes "
                             "doesn't (e.g. MEDS_BIRTH Ethnicity/ Race/ Gender/ — a legacy-vintage "
                             "gap, not a bug). Default: none.")
    args = parser.parse_args()

    ok = diff(args.before, args.after, args.mode, args.lenient, args.max_report,
              args.exclude_line_patterns, args.exclude_if_legacy_missing)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
