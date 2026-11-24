"""Pneumonitis pattern and lobes from the VAL scans' reports, for the oracle sets (spec
planning/vista_eval_vlm/2026-11-24-oracle-matched-examples.md). The findings extraction has no pneumonitis field.

One call per VAL report (the 190 scored scans), Sonnet 5.5 on Vertex (som-nero-plevriti-deidbdf, global; the route the
small-finding study uses on this host; build-time change from Gemini, 2026-11-24, same job and cost). It returns
whether the report describes non-infectious pneumonitis seen on this scan, the lobes it involves and a short pattern
word. Output goes to the private folder only (it is derived from VAL reports): ct-findings-5-v1_7-private/
kb_oracle/pneumonitis_val.json, keyed by the opaque scan id. The tester never sees it.

    cd optimizer/ct-findings-5
    uv run -q --no-project --with google-cloud-bigquery --with "anthropic[vertex]" python -B scripts/pneumonitis_pass.py

Prints counts and dollars only.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402

OUT = bd.PRIVATE_DIR / "kb_oracle" / "pneumonitis_val.json"
MODEL = "claude-sonnet-5-5"
LOBES = ("RUL", "RML", "RLL", "LUL", "LLL")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["present", "lobes", "pattern"],
          "properties": {"present": {"type": "boolean"},
                         "lobes": {"type": "array", "items": {"type": "string", "enum": list(LOBES)}},
                         "pattern": {"type": "string", "enum": ["ground glass", "consolidation", "organizing",
                                                                "reticular", "mixed", "none"]}}}
PROMPT = ("Read this CT report. Does it describe non-infectious pneumonitis (radiation-, drug- or immunotherapy-"
          "related, or organizing pneumonia) seen on this scan? Infection and episodes mentioned only in the history "
          "do not count. If yes, list the lobes involved (RUL, RML, RLL, LUL, LLL; all five if diffuse or bilateral "
          "without detail; the lingula counts as LUL) and the main pattern. If no, present false, no lobes, pattern "
          "none.\n\nREPORT:\n")


def val_scans() -> dict[str, str]:
    """{opaque VAL id: accession} for the VAL scans with organ ranges (the 190 every arm is scored on)."""
    ids = [i for i in json.loads((bd.DATA_DIR / "pools.json").read_text())["val"]
           if (bd.DATA_DIR / "organ_ranges" / i / "target.json").is_file()]
    m = {r["id"]: r["accession_number"] for r in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if r}
    return {i: m[i] for i in ids}


def reports(acc: list[str]) -> dict[str, str]:
    from google.cloud import bigquery

    sql = f"""select accession_number, array_agg(distinct report_note_text ignore nulls) as texts
              from `{bd.REPORT_TABLE}` where accession_number in unnest(@acc) group by accession_number"""
    job = bigquery.Client(project=bd.PROJECT).query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted(set(acc)))]))
    return {r["accession_number"]: "\n\n".join(t for t in r["texts"] if t and t.strip()) for r in job.result()}


def main() -> int:
    import anthropic

    scans = val_scans()
    done = json.loads(OUT.read_text()) if OUT.is_file() else {}
    todo = {i: a for i, a in scans.items() if i not in done}
    text = reports(list(todo.values())) if todo else {}
    client = anthropic.AnthropicVertex(project_id="som-nero-plevriti-deidbdf", region="global", max_retries=4)
    cost = [0.0]

    def one(item):
        sid, acc = item
        t = text.get(acc)
        if not t:
            return sid, {"present": None, "lobes": [], "pattern": "none", "status": "no report"}
        r = client.messages.create(model=MODEL, max_tokens=400, messages=[{"role": "user", "content": PROMPT + t}],
                                   output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        cost[0] += (r.usage.input_tokens * 2.0 + r.usage.output_tokens * 10.0) / 1e6
        return sid, {**json.loads(next(b.text for b in r.content if b.type == "text")), "status": "ok"}

    with ThreadPoolExecutor(8) as ex:
        for sid, ans in ex.map(one, sorted(todo.items())):
            done[sid] = ans
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(done, indent=1))
    pos = sum(1 for v in done.values() if v.get("present"))
    print(f"VAL scans {len(scans)}; read {len(todo)} now (${cost[0]:.2f}); pneumonitis described in {pos}; "
          f"no report {sum(v.get('status') == 'no report' for v in done.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
