"""EHR filter chain: DataFrame ops over the parsed-LUMIA event table.

Applied LIVE at inference (no re-materialization). Each filter takes the event
DataFrame + a small ``ctx`` (carries ``embed_time``, the window anchor) and
returns a filtered DataFrame. The chain runs before flat-render + truncate in the
EHR adapter.

Byte-identity (golden gate 3): the legacy no-imaging-report string was produced
by ``get_described_events_window`` (window = 6 months before ``embed_time``) +
``get_llm_event_string(exclude_report=True)`` (drops any code containing
"STANFORD"). We reproduce that as ``window(before=6mo, after=0)`` +
``code_filter(exclude_stanford=True)`` and then render with
``exclude_report=False`` — the ``code_filter`` predicate is the *same* substring
test the renderer used, so identical rows drop.

Event DataFrame columns (produced by the EHR adapter's LUMIA ingest, matching
``get_llm_event_string``): ``time`` (datetime), ``code``, ``description``,
``numeric_value``, ``unit``, ``text_value``, plus ``type`` (LUMIA event type) and
``speciality`` (provider) for note-type filtering.
"""

from __future__ import annotations

import re

import pandas as pd

# --- duration parsing ---------------------------------------------------------
# Supports d (days), w (weeks), mo (months, calendar-aware), y (years), h (hours).
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mo|[dwyh])\s*$", re.IGNORECASE)


def _as_offset(value):
    """Parse a duration string/number into a pandas offset.

    ``"365d"`` -> ``Timedelta(days=365)``; ``"6mo"`` -> ``DateOffset(months=6)``
    (calendar-aware, matching legacy ``pd.DateOffset(months=...)``); ``"1y"`` ->
    ``DateOffset(years=1)``. A bare number is treated as days.
    """
    if value is None:
        return pd.Timedelta(0)
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=value)
    m = _DURATION_RE.match(str(value))
    if not m:
        raise ValueError(f"Unparseable duration {value!r}; use e.g. '365d', '6mo', '1y', '12h'")
    qty, unit = float(m.group(1)), m.group(2).lower()
    if unit == "d":
        return pd.Timedelta(days=qty)
    if unit == "w":
        return pd.Timedelta(weeks=qty)
    if unit == "h":
        return pd.Timedelta(hours=qty)
    # calendar-aware units require integer counts
    if unit == "mo":
        return pd.DateOffset(months=int(qty))
    if unit == "y":
        return pd.DateOffset(years=int(qty))
    raise ValueError(f"Unsupported duration unit in {value!r}")


def window(df: pd.DataFrame, ctx: dict, before="6mo", after="0d") -> pd.DataFrame:
    """Keep events with ``time`` in ``[embed_time - before, embed_time + after]``.

    ``embed_time`` comes from ``ctx`` (the row's anchor). Rows with a null
    ``time`` are dropped (they cannot be placed on the window).
    """
    if df.empty or "time" not in df.columns:
        return df
    embed_time = ctx.get("embed_time")
    if embed_time is None or pd.isna(embed_time):
        # No anchor: cannot window; return as-is (adapter marks a declared-delta).
        return df
    embed_time = pd.Timestamp(embed_time)
    start = embed_time - _as_offset(before)
    end = embed_time + _as_offset(after)
    times = pd.to_datetime(df["time"], errors="coerce")
    keep = times.notna() & (times >= start) & (times <= end)
    return df[keep]


def note_type_filter(df: pd.DataFrame, ctx: dict, keep=None) -> pd.DataFrame:
    """Keep events whose type / code / provider specialty matches any ``keep`` term.

    ``keep`` is a list of category terms (e.g. ``["radiology", "pathology"]``).
    Matching is case-insensitive substring against ``type``, ``code`` and
    ``speciality`` — the union of the fields the plan maps note-type onto
    (``<event type>`` + provider ``speciality``). Exact category vocabulary is
    VM-refinable; keep this predicate the single place to adjust.
    """
    if df.empty or not keep:
        return df
    terms = [str(t).lower() for t in keep]
    cols = [c for c in ("type", "code", "speciality") if c in df.columns]
    if not cols:
        return df
    haystack = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = haystack.apply(lambda s: any(t in s for t in terms))
    return df[mask]


def code_filter(df: pd.DataFrame, ctx: dict, exclude_stanford=False,
                include=None, exclude=None) -> pd.DataFrame:
    """Drop / keep events by ``code`` substring.

    ``exclude_stanford=True`` drops any row whose ``code`` contains "STANFORD" —
    the exact predicate the legacy renderer applied via ``exclude_report=True``
    (so rendering afterward must pass ``exclude_report=False``). ``include`` /
    ``exclude`` are optional extra substring allow/deny lists.
    """
    if df.empty or "code" not in df.columns:
        return df
    codes = df["code"].fillna("").astype(str)
    mask = pd.Series(True, index=df.index)
    if exclude_stanford:
        mask &= ~codes.str.contains("STANFORD", regex=False)
    if include:
        inc = [str(t) for t in include]
        mask &= codes.apply(lambda c: any(t in c for t in inc))
    if exclude:
        exc = [str(t) for t in exclude]
        mask &= codes.apply(lambda c: not any(t in c for t in exc))
    return df[mask]


def summarize(df: pd.DataFrame, ctx: dict, model=None, **kwargs) -> pd.DataFrame:
    """Model-backed timeline summarization — NOT wired in Phase 1.

    Lifting ``_summarize_timeline_for_context`` from ``retrieval/iterative_
    retrieval.py`` drags a VLM call cluster (``_run_vlm_text_only``,
    ``_extract_answer``, ``TIMELINE_SUMMARY_TEMPLATE``, and a raw-timeline-on-
    exception fallback). It is model-backed (not weight-free), so the config-
    context viewer rejects/marks it, and Phase 1 leaves it as a fail-loud seam.

    Chosen fallback = FAIL LOUD (per claude_ops: fallbacks can mask upstream
    errors; escalate when uncertain). Flip to swallow only on explicit request.
    """
    raise NotImplementedError(
        "EHR 'summarize' filter is a deferred model-backed step (Phase 1 seam only). "
        "It requires a VLM and is not available on the weight-free path. Remove it from "
        "the config's EHR `select` chain, or wire the iterative_retrieval summarizer and "
        "decide its failure mode before enabling."
    )


EHR_FILTERS = {
    "window": window,
    "note_type_filter": note_type_filter,
    "code_filter": code_filter,
    "summarize": summarize,
}

# Filters that need model weights (rejected on the weight-free viewer path).
MODEL_BACKED_FILTERS = {"summarize"}


def resolve_ehr_filter(spec: dict):
    """Bind a config filter dict to a ``(df, ctx) -> df`` callable.

    ``spec`` is e.g. ``{"fn": "window", "before": "365d", "after": "0d"}``.
    """
    if not isinstance(spec, dict) or "fn" not in spec:
        raise ValueError(f"EHR filter spec must be a dict with an 'fn' key, got {spec!r}")
    fn_name = spec["fn"]
    if fn_name not in EHR_FILTERS:
        raise ValueError(
            f"Unknown EHR filter {fn_name!r}; valid: {sorted(EHR_FILTERS)}"
        )
    fn = EHR_FILTERS[fn_name]
    params = {k: v for k, v in spec.items() if k != "fn"}

    def _apply(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        return fn(df, ctx, **params)

    _apply.fn_name = fn_name
    _apply.params = params
    _apply.model_backed = fn_name in MODEL_BACKED_FILTERS
    return _apply
