"""Infection confirming pass (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md, Done when 1).

Reads every thoracic CT report whose wording calls pneumonia or infection (the 545 the grill counted:
'consistent with | compatible with | concerning for | suggestive of' + 'pneumonia | infection', or
'multifocal | bilateral | lobar | atypical pneumonia'; the negation filter is left to the model, which sees the
whole report) and asks Gemini 2.5 Pro on Vertex in the BAA project, once per report, whether this scan shows an
active lung infection: yes / no / unsure. Only yes counts (spec, Decided).

Output is private (derived from reports, keyed by accession): pneumonitis-variants-v1_7-private/infection_pass.json.
Resumable: a rerun reads only reports not yet answered. Prints counts and dollars only.

    cd optimizer/pneumonitis-variants
    uv run -q --no-project --with google-cloud-bigquery --with google-genai python -B scripts/infection_pass.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

OUT = paths.PRIVATE_DIR / "infection_pass.json"
MODEL = "gemini-2.5-pro"
LOCATION = "us-central1"
NOTES = f"{paths.PROJECT}.radiology.thoracic_ct_june26_clinical_notes"
#: The grill's wording match (BigQuery RE2). Exactly 545 reports on 2026-11-24.
MATCH = (r"(?i)((consistent with|compatible with|concerning for|suggestive of) (pneumonia|infection))"
         r"|((multifocal|bilateral|lobar|atypical) pneumonia)")
#: Gemini 2.5 Pro list price per million tokens (<=200k prompt): input $1.25, output incl. thinking $10.
PRICE_IN, PRICE_OUT = 1.25, 10.0
SCHEMA = {"type": "OBJECT", "required": ["answer"],
          "properties": {"answer": {"type": "STRING", "enum": ["yes", "no", "unsure"]}}}
PROMPT = (
    "Read this chest CT report. Does it say that THIS scan shows an active lung infection (pneumonia, including "
    "aspiration pneumonia, or another infectious process in the lung parenchyma)?\n"
    "- yes: the impression or findings attribute a current lung abnormality to infection or pneumonia, as the main "
    "or favoured explanation.\n"
    "- no: infection is denied ('no pneumonia'), mentioned only in the history or as a prior episode, mentioned only "
    "outside the lungs, or the lung change is attributed to something else (radiation, drug or immunotherapy "
    "pneumonitis, organizing pneumonia, edema, tumour, atelectasis).\n"
    "- unsure: infection is one of several equal possibilities, e.g. 'infection versus pneumonitis'.\n"
    "Answer with the JSON only.\n\nREPORT:\n")


def reports() -> dict[str, str]:
    from google.cloud import bigquery

    sql = f"""select accession_number, array_agg(distinct report_note_text ignore nulls) texts
              from `{NOTES}` where regexp_contains(report_note_text, r'{MATCH}') group by accession_number"""
    rows = bigquery.Client(project=paths.PROJECT).query(sql).result()
    return {r["accession_number"]: "\n\n".join(t for t in r["texts"] if t and t.strip()) for r in rows}


def main() -> int:
    from google import genai
    from google.genai import types

    text = reports()
    done = json.loads(OUT.read_text()) if OUT.is_file() else {}
    todo = sorted(a for a in text if a not in done)
    client = genai.Client(vertexai=True, project=paths.PROJECT, location=LOCATION)
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
                                      thinking_config=types.ThinkingConfig(thinking_budget=1024))
    cost = [0.0]

    def one(acc):
        for attempt in range(5):
            try:
                r = client.models.generate_content(model=MODEL, contents=PROMPT + text[acc], config=cfg)
                u = r.usage_metadata
                cost[0] += ((u.prompt_token_count or 0) * PRICE_IN
                            + ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) * PRICE_OUT) / 1e6
                return acc, {"answer": json.loads(r.text)["answer"], "status": "ok"}
            except Exception as e:  # noqa: BLE001 - retried, then recorded as an error (never a label)
                err = type(e).__name__
                time.sleep(2 ** attempt * 3)
        return acc, {"answer": None, "status": f"error {err}"}

    with ThreadPoolExecutor(8) as ex:
        for acc, ans in ex.map(one, todo):
            done[acc] = ans
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(done, indent=1))
    counts = {k: sum(1 for v in done.values() if v["answer"] == k) for k in ("yes", "no", "unsure")}
    errors = sum(1 for v in done.values() if v["answer"] is None)
    print(f"matching reports {len(text)}; read {len(todo)} now (${cost[0]:.2f}); {counts}; errors {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
