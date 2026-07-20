Reference: docs/claude_ops.md

# vista_eval_vlm — Step 5 re-plan: window-scope crop for live LUMIA `timeline` rendering

**Status: Draft** (2026-07-20)

**2026-07-20 update — Approach #1-4 (the window crop) confirmed valid, VM Phase 1 hit band-3
STOP.** The VM readback (`docs/vm-status/2026-07-20-a58f5f9.md`) confirmed the 24mo window crop
is a large partial fix (live/legacy ratio 1.93→1.13, excess 7990→2779) but `total_excess_lines`
still exceeds the band-3 threshold — residual is 98% LOINC. This plan's code fixes remain landed
and valid; only the residual investigation continues, in
`docs/plans/vlm-step5-lumia-loinc-provenance-replan.md`.

**2026-07-20 update #2 — Phase 2 (below) is now un-gated.** Round 4's LOINC-declared closing
verification still failed the byte-diff gate (systemic, non-enumerable divergence, not
LOINC-specific — see `docs/plans/vlm-step5-lumia-gate-retirement-replan.md`), which retires the
byte-diff gate as blocking. Phase 2's own text below still says "once Phase 1 above is clean" —
that precondition is superseded; Phase 2 runs regardless of Phase 1's byte-diff status. No changes
to Phase 2's command or its Expected/Stop criteria.

## Context

This continues the `feat/lumia-live-ehr-adapter` branch's Step 5 work (replacing the frozen
`patient_string` CSV column with a live LUMIA-XML render for the `no_image`/`axial_all_image`
experiments). Two prior class-3 deviations were root-caused and fixed (demographics gap,
`VALUE:`/`NOTE:` field-label mismatch) — but the 2026-07-20 VM run
(`docs/vm-status/2026-07-20-7ed0248.md`) found live still emits ~2× legacy's event volume, and
the leading hypothesis (`omop_split_interval_events` splitting non-instantaneous intervals into
start/end pairs) explains only 4.0% of the excess (321/7,990 lines) — a `<50%` STOP, refuting it
as the primary cause.

Planner-side git archaeology this session (meds2text transform-stack audit, ruling out every
other fan-out candidate) found the real cause: **legacy and live were never scoped to the same
calendar window.** `src/data_tools/csv_helper/subsampled_retrieval_csv.py:180` generates the
frozen `patient_string` with `start_times = embed_times - pd.DateOffset(months=24)` — a
**24-month lookback from `embed_time`** (git-blamed to a deliberate 2026-02-18 change by Ryan
D'Cunha, `6a4c2ead`; the neighboring comment says "6 months" and was simply never updated — dead
history, not the real behavior). Meanwhile `presets.py`'s `timeline` variant (used by
`no_image`/`axial_all_image`) sets `select = []` — **no window at all, full unrestricted
lifetime history**. For an oncology PFS-1yr cohort, essentially every patient has clinical
activity older than 24 months, so live legitimately renders far more events than legacy across
almost every domain — matching the broad, not-concentrated-in-one-domain pattern already visible
in the 4%-attributed subset's 17-domain spread.

Phil confirmed this explanation and made two calls: (1) apply the same crop to live LUMIA
rendering **in production**, not just to make this diff gate pass, and (2) make the window
length **a config option** rather than a second hardcoded literal, defaulting to **24 months**
(matching what legacy has actually been producing since 2026-02-18, not the stale 6-month
comment).

## Goal

Add a configurable time-window crop to the `timeline` variant, defaulting to 24 months before
`embed_time`, then re-run the existing byte-diff gate to confirm this explains (or substantially
resolves) the previously-unattributed ~96% excess.

## Approach

**1. `presets.py` — `_ehr_block("timeline")` gets a real window, matching `no_img_report`'s
existing shape.** Change:
```python
elif variant == "timeline":
    select = []
```
to:
```python
elif variant == "timeline":
    select = [{"fn": "window", "before": "24mo", "after": "0d"}]
```
Reuses the existing `window` selector fn (`src/context/selectors/ehr_filters.py:65`, already
takes `before`/`after` string offsets — no new mechanism). Update the module docstring: this is
now a resolved default (not "pending Phase 1 gate"), note the config-override path (below), and
explicitly keep the *separate* `code_filter`/STANFORD-exclusion question for `timeline` flagged
as still open — unrelated to this fix, not resolved by it.

**2. `run_bq.py` — `_apply_ehr_adapter` reads an optional config override.** After
`ehr_block = next(...)`, before constructing `EHRAdapter`, add:
```python
if ehr_block["config"].get("variant") == "timeline":
    window_before = self.cfg.get("ehr_timeline_window_before")
    if window_before:
        ehr_block["config"]["select"] = [{"fn": "window", "before": window_before, "after": "0d"}]
```
New top-level flat config key `ehr_timeline_window_before` (mirrors the existing flat-key style
of `timeline_truncation`, already read in this same function) — set it in
`configs/all_tasks.vm.yaml` to override the 24mo default without a code change. Absent config ->
presets.py's built-in 24mo default applies untouched.

**3. Docs** — `docs/04-running-the-pipeline.md` and `docs/00-data-setup.md` currently describe
the `timeline`/`no_image` EHR source as full unrestricted history; correct to state the 24mo
default crop + the `ehr_timeline_window_before` config knob. `src/context/selectors/ehr_filters.py`'s
module docstring also still says "`presets.py` currently carries a cohort-source `variant`
marker, not the resolved `select` chain" (line 15-17) — stale now that both `no_img_report` and
`timeline` carry resolved chains; update in the same pass.

**4. Scoping rationale (Codex review — modularity/YAGNI).** `no_img_report`'s `"6mo"` stays
hardcoded rather than gaining the same knob: it reproduces a *separate* legacy comparator's
contract (`get_described_events_window`'s own historical 6mo scope), not the `patient_string`/24mo
one this fix targets. Don't generalize until a real request shows up — note this in `presets.py`'s
docstring so it reads as a decision, not an oversight.

## Files to Modify

- `src/context/presets.py` — `_ehr_block("timeline")` window default + docstring, including the
  `no_img_report` scoping-rationale note (Approach #1, #4).
- `src/vista_run/run_bq.py` — `_apply_ehr_adapter` config-override read (Approach #2).
- `docs/04-running-the-pipeline.md`, `docs/00-data-setup.md`,
  `src/context/selectors/ehr_filters.py` (docstring only) — correct EHR-source description
  (Approach #3).
- `docs/plans/vlm-step5-lumia-window-scope-replan.md` — this plan (new file).
- `docs/plans/vlm-step5-lumia-render-alignment-replan.md` — add a header note that its Approach
  #3 (interval-split) was tested and refuted at 4.0% attribution; point to this plan as the next
  attempt (same pattern as the demographics-flowsheet -> render-alignment supersession already on
  this branch). Not a full supersede banner — its landed `ehr.py` fixes (VALUE:/NOTE:,
  `start|end` leak) remain valid and unrelated to this fix.
- `docs/plans/README.md` — add a row for this plan.

## Open Questions

- The `timeline` variant's `code_filter` (STANFORD-exclusion) question, left open in the original
  Step-5 plan, stays open — this fix is about `window` scope only, a different config axis.
- **Phase 1's quantitative decision-gate bands** (`total_excess_lines <= ~1000` /
  `live_legacy_ratio <= ~1.15` to confirm; `> ~2500` / `> ~1.3` to STOP) are Claude's proposed
  defaults, grounded in the already-measured 321-line interval-split signature + the 569-line
  zero-valued-numeric tail — not yet Phil-reviewed. Fine to adjust before/at the VM run if the
  real numbers suggest a tighter or looser band.

## Codex review

Reviewed 2026-07-20 (`docs/plans/reviews/vlm-step5-lumia-window-scope-replan-feedback.md`),
verdict Revise: Approach #1/#2 held up; the VM handoff's original draft leaned on a
non-existent restricted-input artifact, an ephemeral `$CLAUDE_JOB_DIR/tmp/` script, a wrong CLI
shape, and unquantified residual bands. All findings applied below; none disputed.

## Verification & VM handoff

**What runs on the VM** — all on **Claude-Code CPU `phil-sllm-01`** (weight-free rendering +
BQ/GCS reads only — same posture as every prior handoff on this branch). Two phases: re-verify
the byte-diff gate with the fix applied, then the still-outstanding human-QA landing gate from
the original Step-5 plan.

### Phase 1 — re-verify the byte-diff gate with the window crop applied

**Precondition — no separate restricted-input artifact exists.** The 2026-07-18 readback
(`docs/vm-status/2026-07-18-phase1-demographics-fix-rerun.md:141-143`) confirmed the "restricted
scratch config" mentioned in earlier plan drafts was just the plain
`configs/all_tasks.viewer.vm.yaml` — coverage was 1238/1238 (100%), so `--limit 20` already drew
an identical index set on both sides with no separate CSV. `legacy_small_v2` needs no re-bank
(legacy code/data unchanged, already banked + provenance-confirmed 2026-07-20). Re-bank only the
**live** arm at this branch's new HEAD, with the same plain config:

```bash
cd <repo-root>/vista_eval_vlm
git fetch origin && git checkout feat/lumia-live-ehr-adapter && git pull --ff-only
git rev-parse --short HEAD   # must show this plan's commit

cd src
python -m vista_run.golden_harness \
  --config <ABS_PATH_TO_configs/all_tasks.viewer.vm.yaml> \
  --type gemma3 --name google/medgemma-1.5-4b-it \
  --task progression_recurrence_free_survival_1_yr \
  --experiments no_image \
  --tag lumia_live_windowed --limit 20
cd ..
```

**Precondition checks (cheap, before the expensive bank):**
- **Resolved filter config:** construct one `EHRAdapter` from `get_preset("no_image")`'s `ehr`
  block (post `_apply_ehr_adapter`'s override) and assert `self.filters` includes
  `window(before="24mo", after="0d")` by default / the override value when
  `ehr_timeline_window_before` is set — catches the override landing after
  `EHRAdapter(config=...)` is already constructed (`EHRAdapter.__init__` resolves filters
  immediately, `src/context/adapters/ehr.py:264`).
- **embed_time anchor coverage:** `window()` returns the DataFrame **unchanged** when
  `embed_time` is missing (`ehr_filters.py:73`) — a null anchor silently preserves the bug this
  fix targets. Confirm all 20 `lumia_live_windowed` rows have a non-null, parseable `embed_time`
  (count only, no PHI). **Stop:** any missing `embed_time` — fix before proceeding, don't average
  it into the residual.

**Locate + diff**, reusing the same `.meta.json` provenance check pattern as the 2026-07-20 doc:

```bash
LEGACY=<results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_legacy_small_v2_golden.jsonl
LIVE=<results_dir>/golden/.../progression_recurrence_free_survival_1_yr_no_image_lumia_live_windowed_golden.jsonl
# verify both .meta.json: task/experiment/model_type/model_name/tag/limit match; STOP on mismatch

cd src
python -m vista_run.diff_golden "$LEGACY" "$LIVE" --mode strict \
  --exclude-line-patterns STANFORD_OBS/Flowsheet \
  --exclude-if-legacy-missing MEDS_BIRTH Ethnicity/ Race/ Gender/
```

Also re-run the masked characterization script below against `$LEGACY`/`$LIVE`, for an
apples-to-apples comparison against the 2026-07-20 run's
`total_legacy_lines=6987 / total_live_lines=13488 / total_excess_lines=7990`. Pasted here in
full (not `$CLAUDE_JOB_DIR/tmp/`-dependent — that scratch dir is per-job and not durable across
sessions) with the 2026-07-20 readback's nonnumeric-`VALUE:` crash fix already applied:

```python
#!/usr/bin/env python3
"""Phase-1 characterization script, vlm-step5-lumia-window-scope-replan.md. Reads two
already-banked golden JSONLs and reports masked, aggregate counts only -- no raw timeline
text, person_ids, or dates are printed.

Usage: python3 phase1_window_scope_characterize.py <legacy_jsonl> <live_jsonl>
"""
import json
import re
import sys
from collections import Counter, defaultdict

EXCLUDE_LINE_PATTERNS = ("STANFORD_OBS/Flowsheet",)
EXCLUDE_IF_LEGACY_MISSING = ("MEDS_BIRTH", "Ethnicity/", "Race/", "Gender/")

LINE_RE = re.compile(r"^\[(?P<time>[^\]]+)\]\s*\|\s*(?P<code>[^\s(|]+)(?:\s*\((?P<desc>[^)]*)\))?")
VALUE_RE = re.compile(r"\|\s*VALUE:\s*([0-9.]+)")


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec["index"]] = rec
    return rows


def strip_excluded(text, patterns):
    if text is None:
        return text
    return "\n".join(l for l in text.splitlines() if not any(p in l for p in patterns))


def strip_if_legacy_missing(before, after, patterns):
    before_has = bool(before) and any(p in before for p in patterns)
    if before_has:
        return before, after
    return strip_excluded(before, patterns), strip_excluded(after, patterns)


def apply_exclusions(before, after):
    before = strip_excluded(before, EXCLUDE_LINE_PATTERNS)
    after = strip_excluded(after, EXCLUDE_LINE_PATTERNS)
    before, after = strip_if_legacy_missing(before, after, EXCLUDE_IF_LEGACY_MISSING)
    return before, after


def parse_lines(text):
    out = []
    for line in (text or "").splitlines():
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            out.append((None, None, None, line))
            continue
        out.append((m.group("code"), m.group("desc") or "", m.group("time"), line))
    return out


def domain_of(code):
    if not code:
        return "UNKNOWN"
    return code.split("/")[0] if "/" in code else code


def has_value_or_note(line):
    return ("VALUE:" in line) or ("NOTE:" in line)


def main():
    legacy_path, live_path = sys.argv[1], sys.argv[2]
    legacy_rows = load(legacy_path)
    live_rows = load(live_path)

    legacy_ids = set(legacy_rows)
    live_ids = set(live_rows)
    shared = sorted(legacy_ids & live_ids, key=str)
    only_legacy = legacy_ids - live_ids
    only_live = live_ids - legacy_ids

    print(f"legacy_rows={len(legacy_rows)} live_rows={len(live_rows)} shared={len(shared)} "
          f"only_legacy={len(only_legacy)} only_live={len(only_live)}")
    if not shared:
        print("STOP: no shared indices between the two banks -- precondition failure, hand back.")
        sys.exit(1)

    total_legacy = 0
    total_live = 0
    total_excess = 0
    attributed_excess = 0
    domain_attributed = Counter()
    domain_total_excess = Counter()
    zero_value_legacy_lines = 0
    zero_value_collapsed = 0
    zero_value_no_live_counterpart = 0
    nonnumeric_value_tokens_skipped = 0

    for idx in shared:
        b = legacy_rows[idx].get("adapter_prompt_string")
        a = live_rows[idx].get("adapter_prompt_string")
        b, a = apply_exclusions(b, a)

        legacy_lines = parse_lines(b)
        live_lines = parse_lines(a)
        total_legacy += len(legacy_lines)
        total_live += len(live_lines)

        excess = len(live_lines) - len(legacy_lines)
        if excess > 0:
            total_excess += excess

            groups = defaultdict(set)
            for code, desc, time, _line in live_lines:
                groups[(code, desc)].add(time)
                domain_total_excess[domain_of(code)] += 0  # ensure key exists

            pair_excess = 0
            pair_excess_by_domain = Counter()
            for (code, desc), times in groups.items():
                if len(times) == 2:
                    pair_excess += 1
                    pair_excess_by_domain[domain_of(code)] += 1

            this_attributed = min(pair_excess, excess)
            attributed_excess += this_attributed
            for dom, cnt in pair_excess_by_domain.items():
                domain_attributed[dom] += cnt

            # domain breakdown of the FULL excess (not just the interval-paired subset) --
            # the 2026-07-20 run only reported this for the attributed slice, leaving the
            # dominant unattributed excess's domain shape unknown. Approximate: attribute
            # excess lines to whichever domains appear in live-only lines for this row.
            legacy_line_texts = {l for *_, l in legacy_lines}
            for code, _desc, _time, line in live_lines:
                if line not in legacy_line_texts:
                    domain_total_excess[domain_of(code)] += 1

        # zero-valued-numeric residual: legacy "VALUE: 0(.0)" lines whose live counterpart
        # at the same (code, desc, time) key has collapsed to no VALUE:/NOTE: segment at all.
        live_by_key = {}
        for code, desc, time, line in live_lines:
            live_by_key[(code, desc, time)] = line
        for code, desc, time, line in legacy_lines:
            vm = VALUE_RE.search(line)
            if vm:
                try:
                    is_zero = float(vm.group(1)) == 0.0
                except ValueError:
                    nonnumeric_value_tokens_skipped += 1
                    continue
                if is_zero:
                    zero_value_legacy_lines += 1
                    live_line = live_by_key.get((code, desc, time))
                    if live_line is None:
                        zero_value_no_live_counterpart += 1
                    elif not has_value_or_note(live_line):
                        zero_value_collapsed += 1

    pct = (100.0 * attributed_excess / total_excess) if total_excess else 0.0
    print(f"total_legacy_lines={total_legacy}")
    print(f"total_live_lines={total_live}")
    print(f"total_excess_lines={total_excess}")
    print(f"live_legacy_ratio={(total_live / total_legacy) if total_legacy else float('nan'):.2f}")
    print(f"attributed_excess_lines(split_interval_signature)={attributed_excess}")
    print(f"attributed_excess_pct={pct:.1f}")
    print(f"unattributed_excess_lines={total_excess - attributed_excess}")
    print("attributed_excess_by_domain=" + json.dumps(dict(domain_attributed), sort_keys=True))
    print("full_excess_by_domain=" + json.dumps(dict(domain_total_excess), sort_keys=True))
    print(f"zero_valued_legacy_VALUE_lines={zero_value_legacy_lines}")
    print(f"zero_valued_collapsed_on_live={zero_value_collapsed}")
    print(f"zero_valued_no_live_counterpart={zero_value_no_live_counterpart}")
    print(f"nonnumeric_VALUE_tokens_skipped={nonnumeric_value_tokens_skipped}")


if __name__ == "__main__":
    main()
```

**Expected/decision gate — quantitative bands** (legacy and live now cover the same ~24-month
window, so remaining excess should be limited to already-known causes: the 321-line
interval-split signature + the zero-valued-numeric tail, bounded by the 569 zero-value legacy
lines already observed 2026-07-20):
- `total_excess_lines <= ~1000` **and** `live_legacy_ratio <= ~1.15` -> windowing was the
  (near-)complete explanation; confirmed; proceed to Phase 2.
- `total_excess_lines` in `~1000-2500` or `live_legacy_ratio` in `~1.15-1.3` -> ambiguous middle
  band; STOP, hand back to Phil explicitly (same convention as the original Phase 1's 50-80%
  band) rather than auto-resolving.
- `total_excess_lines > ~2500` or `live_legacy_ratio > ~1.3` -> a second, still-unidentified cause
  remains; STOP; report `full_excess_by_domain` (now computed for the *entire* excess, not just
  the interval-paired subset) and hand back to the Mac for a narrower follow-up investigation, not
  a full re-plan from scratch.

**Stop:** index-set mismatch against `legacy_small_v2` (precondition failure — don't
re-interpret as a verdict); the embed_time anchor check failing; the config-resolution smoke
failing; or the decision-gate bands above.
**Destructive:** no — new tag under the git-ignored results tree only; clean up any worktree
used, same as prior phases on this branch.

### Phase 2 — human visual-QA render + landing gate

Reprised here in full so this plan is a self-contained handoff source (the original Step-5 plan's
Phase 3, `docs/plans/vlm-step5-lumia-live-ehr-adapter.md`, never yet completed on this branch
because Phase 1 kept re-opening). Once Phase 1 above is clean, render N≈5 each:

```bash
cd src
python -m results.context_viewer --config ../configs/all_tasks.viewer.vm.yaml --type gemma3 \
  --name google/medgemma-1.5-4b-it --task progression_recurrence_free_survival_1_yr \
  --experiment no_image --limit 5          # repeat for no_report/timeline_only, axial_all_image
```

**Expected:** each HTML exists, non-empty, self-contained (no external URLs, no local filesystem
paths in `<img src>`), exactly 5 cards; every card has non-empty rendered prompt/timeline text and
a non-empty token-count/bar; `axial_all_image` cards show the expected slice/thumbnail count,
`no_image`/`no_report` cards show 0 images where expected; no `STOP:` or traceback text anywhere.
**Report back:** file paths, existence, card counts, exit codes only — **never paste rendered
content** (PHI). Phil opens the files himself on `phil-sllm-01` to read the rendered text — the
manual QA step no agent can do for him.
**Stop:** missing/empty HTML, non-zero exit, the self-containment grep fails, a card missing
text/token content, or a `STOP:`/traceback string appears where it shouldn't.
**Destructive:** no.

## Landing & cleanup

- **Branch:** `feat/lumia-live-ehr-adapter` (continuing, no new branch).
- **Landing gate:** Phase 1 clean (or its residual fully explained), Phase 2 rendered and Phil
  has actually opened and read the HTML.
- **Merge sequence:** single branch, `/land` at the end -> `main`, prune branch.
- **Cleanup on land:** mark this plan and the original Step-5 plan `Status: Completed`; update
  `docs/next.md` and `docs/plans/README.md` (the row added under Files to Modify); confirm the
  render-alignment-replan.md header note (also under Files to Modify) is in place; note the
  resolved window default directly in `presets.py`'s docstring (done in Approach #1, not left as
  a follow-up); carry the remaining Open Question above into `docs/next.md` as a backlog item.
