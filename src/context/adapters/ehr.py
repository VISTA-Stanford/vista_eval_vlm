"""EHR modality adapter: given LUMIA timeline -> filtered, flat-rendered text.

Fully inference-time and symmetric with CT/pathology (no offline prep, no
``meds_reader``/``meds_tools``/ontology):

    ingest(raw, ctx)         parse the given LUMIA .xml -> event DataFrame
                             (or pass through a pre-rendered timeline string)
    contextualize(norm, ctx) filter chain -> flat-render -> truncate -> text block

The flat renderer and the truncator are the **canonical** functions from
``meds_timeline_utils`` (imported, not copied) so the string is byte-identical to
today's ``patient_string`` by construction — this is the "no third formatter
copy" discipline. The STANFORD/report skip is applied as a ``code_filter`` row
drop, so we render with ``exclude_report=False``.

LUMIA field coverage (VM-gated, see ``_lumia_event_to_row``): ``markup.md``
documents ``<event>`` with only ``type/code/name/note_id/provider_id/care_site_id``
(element text = a state token). The flat renderer additionally wants
``numeric_value``, ``unit``, ``text_value`` and a ``description`` distinct from
``name``. Those are read from ``<event>`` attributes *if the corpus carries them*
and default to ``None`` otherwise; every fully-absent field is recorded on the
block metadata (``lumia_missing_fields``) so gate 3 degrades to a declared-delta
(OQ-K) instead of failing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Canonical byte-for-byte flat renderer + truncator (imported, never copied).
from data_tools.utils.meds_timeline_utils import get_llm_event_string, truncate_timeline

from ..block import ContextBlock, TEXT
from ..selectors.ehr_filters import resolve_ehr_filter
from .base import ModalityAdapter

# Flat-renderer fields beyond the documented LUMIA <event> attrs; tracked for the
# VM field-coverage gate.
_RENDERER_EXTRA_FIELDS = ("numeric_value", "unit", "text_value", "description")

# Event <event> element text tokens that are pure state markers (not note bodies).
_STATE_TOKENS = {"start", "end", "", None}


def lumia_path_for(person_id, corpus_dir) -> Path:
    """Resolve the per-patient LUMIA file: ``<corpus_dir>/<person_id>.xml``."""
    return Path(corpus_dir) / f"{person_id}.xml"


def _lumia_event_to_row(entry_ts, event_el, provider_speciality: dict) -> dict:
    """Map one LUMIA ``<event>`` to a flat-renderer row.

    THE field-coverage seam (VM-gated). Documented fields map directly; the
    renderer's extra fields are read from attributes if present, else ``None``.
    Keep this the single place to adjust when a real ``.xml`` is inspected.
    """
    attrib = event_el.attrib
    code = attrib.get("code")
    # description: prefer an explicit `description` attr, else fall back to `name`.
    description = attrib.get("description")
    if description is None or description == "":
        description = attrib.get("name")

    # note body: an explicit `text_value` attr, else element text when it is a
    # real body (notes) rather than a state token like "start".
    text_value = attrib.get("text_value")
    if text_value is None:
        el_text = (event_el.text or "").strip()
        if el_text and el_text.lower() not in _STATE_TOKENS:
            text_value = el_text

    def _num(v):
        try:
            return float(v) if v is not None and str(v) != "" else None
        except (TypeError, ValueError):
            return None

    return {
        "time": entry_ts,
        "code": code,
        "description": description,
        "numeric_value": _num(attrib.get("numeric_value")),
        "unit": attrib.get("unit"),
        "text_value": text_value,
        # non-renderer fields, used by filters:
        "type": attrib.get("type"),
        "note_id": attrib.get("note_id"),
        "provider_id": attrib.get("provider_id"),
        "speciality": provider_speciality.get(attrib.get("provider_id")),
    }


def parse_lumia(source) -> pd.DataFrame:
    """Parse a LUMIA event stream into an event DataFrame.

    ``source`` may be an XML string, a path to an ``.xml`` file, or an
    ``ElementTree`` root. Returns a DataFrame with the columns
    ``get_llm_event_string`` consumes (``time``/``code``/``description``/
    ``numeric_value``/``unit``/``text_value``) plus ``type``/``speciality`` for
    filtering, sorted chronologically by ``time``.
    """
    if source is None:
        return pd.DataFrame()
    if isinstance(source, ET.Element):
        root = source
    else:
        s = str(source)
        if s.lstrip().startswith("<"):
            root = ET.fromstring(s)
        else:
            root = ET.parse(s).getroot()

    rows: list[dict] = []
    for encounter in root.iter("encounter"):
        # provider_id -> speciality, for note_type_filter
        provider_speciality = {
            p.attrib.get("provider_id"): p.attrib.get("speciality")
            for p in encounter.iter("provider")
        }
        for entry in encounter.iter("entry"):
            entry_ts = pd.to_datetime(entry.attrib.get("timestamp"), errors="coerce")
            for event_el in entry.findall("event"):
                rows.append(_lumia_event_to_row(entry_ts, event_el, provider_speciality))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # Stable chronological sort. LUMIA is generated from the MEDS timeline, so
        # document order is already chronological and this is a no-op in the
        # expected case (stable => ties keep document order, matching the legacy
        # renderer's row order). Kept as a safety net; golden gate 3 (VM) confirms
        # the resulting event order is byte-identical to legacy — if the corpus is
        # NOT chronologically ordered, revisit (preserve source order instead).
        df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return df


def lumia_missing_fields(event_df: pd.DataFrame) -> list[str]:
    """Which renderer-extra fields are entirely absent/empty across events."""
    missing = []
    for f in _RENDERER_EXTRA_FIELDS:
        if f not in event_df.columns or event_df[f].isna().all():
            missing.append(f)
    return missing


class EHRAdapter(ModalityAdapter):
    modality = "text"

    def __init__(self, block_id: str = "ehr", config: dict | None = None):
        super().__init__(block_id, config)
        self.filters = [resolve_ehr_filter(f) for f in (self.config.get("select") or [])]
        self.serialize_cfg = self.config.get("serialize", {}) or {}
        self.truncation_cfg = self.config.get("truncation")

    def ingest(self, raw, ctx: dict):
        """Normalize the raw EHR source.

        Accepts a LUMIA source (XML string / ``.xml`` path / ``person_id`` +
        ``ctx['lumia_corpus_dir']``) -> event DataFrame, or a pre-rendered
        timeline string -> passthrough (returned as-is).
        """
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, pd.DataFrame):
            return raw
        # person_id + corpus dir
        corpus_dir = ctx.get("lumia_corpus_dir")
        if corpus_dir is not None and not isinstance(raw, (ET.Element,)) and "\n" not in str(raw):
            candidate = lumia_path_for(raw, corpus_dir)
            if candidate.exists():
                return parse_lumia(candidate)
        s = str(raw)
        stripped = s.lstrip()
        if stripped.startswith("<") or (s.endswith(".xml") and Path(s).exists()):
            return parse_lumia(raw)
        # pre-rendered timeline text -> passthrough
        return s

    def _render_events(self, event_df: pd.DataFrame, ctx: dict) -> tuple[str, dict]:
        meta: dict = {}
        embed_time = ctx.get("embed_time")
        fctx = {"embed_time": embed_time, "model_type": ctx.get("model_type")}
        applied = []
        for f in self.filters:
            event_df = f(event_df, fctx)
            applied.append({"fn": f.fn_name, **f.params})
        meta["filters"] = applied
        meta["event_count"] = int(len(event_df))
        meta["lumia_missing_fields"] = lumia_missing_fields(event_df) if not event_df.empty else list(_RENDERER_EXTRA_FIELDS)
        include_text = bool(self.serialize_cfg.get("include_text", True))
        max_text_len = self.serialize_cfg.get("max_text_len")
        # STANFORD skip handled by code_filter -> render with exclude_report=False.
        text = get_llm_event_string(
            event_df, include_text=include_text, max_text_len=max_text_len, exclude_report=False
        )
        return text, meta

    def contextualize(self, normalized, ctx: dict) -> ContextBlock:
        meta: dict = {"serialize": self.serialize_cfg.get("style", "flat_timeline")}
        if isinstance(normalized, pd.DataFrame):
            text, rmeta = self._render_events(normalized, ctx)
            meta.update(rmeta)
        else:
            # passthrough: already-rendered timeline string (legacy cohorts)
            text = "" if normalized is None else str(normalized)
            meta["passthrough"] = True
            if self.filters:
                meta["filters_skipped"] = "select filters ignored on pre-rendered timeline text"

        # Truncate (verbatim canonical function), applied at inference.
        truncation_cfg = self.truncation_cfg
        if truncation_cfg is None:
            truncation_cfg = ctx.get("timeline_truncation")
        if truncation_cfg is not None:
            text = truncate_timeline(text, truncation_cfg)
            meta["truncation"] = truncation_cfg

        return ContextBlock(
            id=self.block_id,
            modality=self.modality,
            representation=TEXT,
            payload=text,
            metadata=meta,
        )
