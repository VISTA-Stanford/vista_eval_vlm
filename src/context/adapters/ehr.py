"""EHR modality adapter: given LUMIA timeline -> filtered, flat-rendered text.

Fully inference-time and symmetric with CT/pathology (no offline prep, no
``meds_reader``/``meds_tools``/ontology):

    ingest(raw, ctx)         parse the given LUMIA .xml -> event DataFrame
                             (or pass through a pre-rendered timeline string)
    contextualize(norm, ctx) filter chain -> flat-render -> truncate -> text block

The flat renderer and the truncator are the **canonical** functions from
``meds_timeline_utils`` (imported, not copied) so the string is byte-identical to
today's ``patient_string`` by construction — this is the "no third formatter
copy" discipline. The adapter **always** renders with ``exclude_report=False``;
the STANFORD/report skip is delegated to a ``code_filter(exclude_stanford=True)``
step in the EHR ``select`` chain (same substring predicate). NOTE: the concrete
per-preset ``select`` chains (e.g. ``window(6mo)`` + ``code_filter`` for the
no-imaging-report cohort) are wired when the hot-path pass consumes the presets
and validates gate 3 on the VM — the mechanism lives here; the preset wiring does
not yet (see ``presets.py``).

LUMIA field coverage (VM-gated, see ``_lumia_event_to_row``): ``markup.md``
documents ``<event>`` with only ``type/code/name/note_id/provider_id/care_site_id``
(element text = a state token, *or* the rendered numeric/note value — see below).
The flat renderer additionally wants ``numeric_value``, ``unit``, ``text_value`` and
a ``description`` distinct from ``name``. ``unit``/``description`` are read from
``<event>`` attributes *if the corpus carries them* and default to ``None``
otherwise; every fully-absent field is recorded on the block metadata
(``lumia_missing_fields``) so gate 3 degrades to a declared-delta instead of
failing. ``numeric_value``/``text_value`` are instead reconstructed from element
text — this resolves the 2026-07-06 field-coverage OQ-K fork ("accept the declared
delta vs. parse numeric from text") in favor of parsing; see ``_lumia_event_to_row``.

Demographics (VM-gated, see ``_demographic_rows``): ``<person>`` is a sibling of
``<events>`` under every ``<encounter>``, not an ``<event>`` itself, so the walk
above never sees it — ``parse_lumia`` separately reads one ``<person>`` block per
patient (demographics are identical across encounters; deduped by taking the
first) and synthesizes ``MEDS_BIRTH``/``Ethnicity``/``Race``/``Gender`` pseudo-event
rows so they flow through the same renderer as everything else. Conventions
(order, bare vs. described codes, birthdate anchor) confirmed against the real
schema + the legacy ``patient_string`` render — see
``docs/vm-status/2026-07-17-4b5239e.md``.
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
# "start|end" is omop_split_interval_events's combined token for near-instantaneous
# intervals (start ≈ end within 1 minute) — same state-marker, not a note body.
_STATE_TOKENS = {"start", "end", "start|end", "", None}

# <person>/<demographics> child tags to synthesize as pseudo-event rows, in the fixed order
# confirmed against the legacy render (13/13 patients, VM Phase 0.5): Ethnicity, Race, Gender.
_DEMOGRAPHIC_TAGS = ("ethnicity", "race", "gender")


def lumia_path_for(person_id, corpus_dir) -> Path | None:
    """Resolve the per-patient LUMIA file: ``<corpus_dir>/<person_id>.xml``.

    Returns ``None`` (rather than raising) when ``corpus_dir`` is unset, so callers
    that probe reachability (e.g. ``context_viewer.py``) get a clean falsy result
    instead of a ``TypeError`` from ``Path(None)``.
    """
    if corpus_dir is None:
        return None
    return Path(corpus_dir) / f"{person_id}.xml"


def _lumia_event_to_row(entry_ts, event_el, provider_speciality: dict) -> dict:
    """Map one LUMIA ``<event>`` to a flat-renderer row.

    THE field-coverage seam (VM-gated). Documented fields map directly; the
    renderer's extra fields are read from attributes if present, else ``None``.
    Keep this the single place to adjust when a real ``.xml`` is inspected.

    ``numeric_value``/``text_value`` resolution (resolves the 2026-07-06 field-coverage
    OQ-K fork — "accept the declared delta vs. parse numeric from text" — in favor of
    parsing): LUMIA ``<event>`` attributes never actually carry ``numeric_value``/
    ``text_value``. Confirmed against meds2text's real generator
    (``textify.py::event_to_xml`` ~L1248-1340, sibling repo ``../meds2text``): it picks one
    winning value via ``value = attributes.get("numeric_value") or attributes.get("text_value")
    or None``, writes it only into element text (``f"{float(value):.2f}"``/bare int for
    numerics, raw string otherwise), then unconditionally pops both attributes — so no
    attribute-level signal survives into the ``.xml``. This function instead ``float()``-parses
    element text to reconstruct ``numeric_value`` when neither attribute is populated.
    **Cross-repo contract:** if meds2text ever starts emitting explicit ``numeric_value``/
    ``text_value`` attributes, or changes its formatting, prefer those attributes over this
    text-parse heuristic and re-run the golden gate.

    Three declared, byte-level residual classes are accepted as permanent divergence (not
    adapter bugs — structurally forced by ``event_to_xml``'s export design; see
    ``docs/plans/vlm-step5-lumia-render-alignment-replan.md`` Approach #1):
      1. Dual-value rows (legacy ``VALUE: D.D | NOTE: D.D`` on one line) collapse to one
         value — ``event_to_xml`` keeps only the winner; the loser never reaches the ``.xml``.
      2. Zero-valued numerics: the ``or``-chain above treats a genuine ``0.0`` as falsy, so a
         zero-valued lab can lose to ``text_value``/``None`` and render with empty text.
      3. Ontology-override collapse: ``event_to_xml`` overwrites the winning text with an OMOP
         concept description when the value's string form resolves as a concept code — the
         number is unrecoverable once that happens.
    """
    attrib = event_el.attrib
    code = attrib.get("code")
    # description: prefer an explicit `description` attr, else fall back to `name`.
    description = attrib.get("description")
    if description is None or description == "":
        description = attrib.get("name")

    def _num(v):
        try:
            return float(v) if v is not None and str(v) != "" else None
        except (TypeError, ValueError):
            return None

    # text_value/numeric_value attrs are dead in practice (meds2text never emits them —
    # see docstring); reconstruct numeric_value by parsing element text when both are absent.
    text_value = attrib.get("text_value")
    numeric_value = _num(attrib.get("numeric_value"))
    if text_value is None and numeric_value is None:
        el_text = (event_el.text or "").strip()
        if el_text and el_text.lower() not in _STATE_TOKENS:
            parsed = _num(el_text)
            if parsed is not None:
                numeric_value = parsed
            else:
                text_value = el_text

    # unit: LUMIA carries it as `unit_source_value` (OMOP source unit), not `unit`.
    return {
        "time": entry_ts,
        "code": code,
        "description": description,
        "numeric_value": numeric_value,
        "unit": attrib.get("unit") or attrib.get("unit_source_value"),
        "text_value": text_value,
        # non-renderer fields, used by filters:
        # meds2text's textify.py (event_to_xml) actually emits this as `table=`, not `type=` —
        # markup.md's documented attribute name is wrong (confirmed via meds2text source + the
        # Step-5 re-plan's git-blame investigation, 2026-07-17).
        "type": attrib.get("table"),
        "note_id": attrib.get("note_id"),
        "provider_id": attrib.get("provider_id"),
        "speciality": provider_speciality.get(attrib.get("provider_id")),
    }


def _demographic_rows(person_el: ET.Element) -> list[dict]:
    """Synthesize MEDS_BIRTH/Ethnicity/Race/Gender pseudo-event rows from one ``<person>`` block.

    Conventions confirmed against the real LUMIA schema + the already-banked legacy
    ``patient_string`` render (VM Phase 0.5, ``docs/vm-status/2026-07-17-4b5239e.md``):
    ``<demographics>/<tag>``'s ``code`` attribute is already in ``Class/<omop_concept_id>`` form
    (e.g. ``"Gender/8532"``) — used verbatim, no re-prefixing, no description. All rows anchor at
    the ``<birthdate>`` timestamp; ``MEDS_BIRTH`` (bare code, no description) is emitted twice, then
    Ethnicity/Race/Gender in that order — this exact sequence matched legacy 13/13 patients.
    """
    birthdate_el = person_el.find("birthdate")
    birthdate = pd.to_datetime((birthdate_el.text or "").strip(), errors="coerce") if birthdate_el is not None else pd.NaT
    if pd.isna(birthdate):
        return []

    def _row(code):
        return {
            "time": birthdate,
            "code": code,
            "description": None,
            "numeric_value": None,
            "unit": None,
            "text_value": None,
            "type": None,
            "note_id": None,
            "provider_id": None,
            "speciality": None,
        }

    rows = [_row("MEDS_BIRTH"), _row("MEDS_BIRTH")]
    demographics_el = person_el.find("demographics")
    if demographics_el is not None:
        for tag in _DEMOGRAPHIC_TAGS:
            tag_el = demographics_el.find(tag)
            code = tag_el.attrib.get("code") if tag_el is not None else None
            if code:
                rows.append(_row(code))
    return rows


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
    person_el = None
    for encounter in root.iter("encounter"):
        # <person> repeats identically per-encounter (VM-confirmed) — dedup by taking the first.
        if person_el is None:
            person_el = encounter.find("person")
        # provider_id -> speciality, for note_type_filter
        provider_speciality = {
            p.attrib.get("provider_id"): p.attrib.get("speciality")
            for p in encounter.iter("provider")
        }
        for entry in encounter.iter("entry"):
            entry_ts = pd.to_datetime(entry.attrib.get("timestamp"), errors="coerce")
            for event_el in entry.findall("event"):
                rows.append(_lumia_event_to_row(entry_ts, event_el, provider_speciality))
    if person_el is not None:
        rows.extend(_demographic_rows(person_el))

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
