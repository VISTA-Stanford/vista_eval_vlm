#!/usr/bin/env python3
"""Pull the clinicians' own words out of the notes into one short digest.

Usage: python3 -B digest.py /path/to/record

Prints (newest last within each part, about 8-10 KB in total):
  1. the assessment / plan sections of the 3 most recent substantial clinician notes about the cancer
     (oncology, radiation oncology, thoracic surgery, consults, H&P);
  2. the impression of the most recent staging imaging reports (CT, PET, MRI) from the last 120 days;
  3. the final diagnosis lines of the most recent pathology reports.
Uses note.parquet (falls back to notes.txt). Read it in full: it is short. "Now" is the record's cutoff
(meta.json), so the 180- and 120-day windows end a week before the tumor board.

Copied from the died-2yr-ehr summary arm (program-v6, tools/digest.py, 2026-11-20) for the tumor board record agent;
the only change is reading the cutoff from meta.json instead of taking the latest note's date.
"""
import json
import os
import re
import sys

import pandas as pd

AP_RE = re.compile(r'(?i)(assessment\s*(?:and|&|/)\s*\n?\s*plan|impression\s*(?:and|&|/)\s*plan|'
                   r'assessment and recommendations|\ba/p\b|\bassessment\s*:|\bimpression\s*:|'
                   r'\brecommendations?\s*:|\bplan\s*:)')
CANCER_RE = re.compile(r'(?i)oncolog|carcinoma|cancer|malignan|nsclc|sclc|small cell|\bstage (?:i|1|2|3|4)|sbrt|sabr|'
                       r'lobectomy|mesothelioma|thymoma|metasta|immunotherapy|chemoradiation|tumou?r board|ecog')
CLIN_CLASSES = {'Progress note', 'Consult note', 'History and physical note', 'Evaluation + Plan note',
                'Plan of care note', 'No matching concept', 'Transplant surgery Note'}
IMG_RE = re.compile(r'(?i)\b(ct|pet|pet/ct|mri|mr)\b')
RADHEAD_RE = re.compile(r'\s*[A-Z0-9 /,\-\(\)&+]{4,120}:\s*\d\d/\d\d/\d{4}')
IMPRESSION_RE = re.compile(r'(?i)\b(impression|conclusion)\s*:?')


def clean(s):
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip()


def load(rec):
    p = os.path.join(rec, 'note.parquet')
    if os.path.exists(p):
        cols = ['note_id', 'note_date', 'note_datetime', 'note_text', 'note_class_concept_name', 'note_type']
        try:
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)
        df['cls'] = df.get('note_class_concept_name', pd.Series(['']*len(df))).fillna('')
        df['typ'] = df.get('note_type', pd.Series(['']*len(df))).fillna('')
        df['text'] = df['note_text'].fillna('').astype(str)
        df['date'] = df['note_date'].astype(str).str[:10]
        df['sortkey'] = df['note_datetime'].astype(str) if 'note_datetime' in df else df['date']
        return df.sort_values(['sortkey', 'note_id'])[['date', 'cls', 'typ', 'text']]
    p = os.path.join(rec, 'notes.txt')
    if not os.path.exists(p):
        return None
    rows = []
    cur = None
    with open(p, errors='replace') as fh:
        for line in fh:
            m = re.match(r'===== NOTE (\S+) \| (\S+)[^|]*\| (.*) =====', line)
            if m:
                cur = {'date': m.group(2)[:10], 'cls': '', 'typ': m.group(3).strip(), 'text': ''}
                rows.append(cur)
            elif cur is not None:
                cur['text'] += line
    return pd.DataFrame(rows) if rows else None


def ap_section(t, n=1900):
    hits = [h for h in AP_RE.finditer(t) if len(t) - h.start() > 200]
    if not hits:
        return clean(t[-1200:])
    # prefer a combined "assessment and plan" heading, then the last "assessment", then the last "plan"/other
    combined = [h for h in hits if re.search(r'(?i)plan|recommend', h.group(0)) and
                re.search(r'(?i)assess|impress', h.group(0))]
    assess = [h for h in hits if re.search(r'(?i)assess', h.group(0))]
    plan = [h for h in hits if re.search(r'(?i)plan|recommend', h.group(0))]
    start = (combined[0] if combined else assess[-1] if assess else plan[-1] if plan else hits[-1]).start()
    return clean(t[start:start + n])


def main(rec):
    df = load(rec)
    if df is None or df.empty:
        print('No notes in this record.')
        return
    try:
        cutoff = json.load(open(os.path.join(rec, 'meta.json')))['cutoff'][:10]
    except (OSError, ValueError, KeyError, TypeError):
        cutoff = df['date'].max()

    print('== RECENT CLINICIAN ASSESSMENT / PLAN (newest last)')
    rad_text = df['text'].map(lambda t: bool(RADHEAD_RE.match(t)) or ('COMPARISON' in t[:1500] and 'IMPRESSION' in t))
    is_rad = df['cls'].str.contains('Radiology|Pathology', regex=True) | (df['typ'] == 'Radiology Study Results') | rad_text
    clin = df[~is_rad & (df['text'].str.len() > 1500) &
              (df['cls'].isin(CLIN_CLASSES) | (df['cls'] == ''))]
    lo = (pd.Timestamp(cutoff) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    clin = clin[clin['date'] >= lo]
    clin = clin[clin['text'].map(lambda t: len(CANCER_RE.findall(t)) >= 4 and bool(AP_RE.search(t)))]
    seen = set()
    picked = []
    for _, r in clin.iloc[::-1].iterrows():
        sec = ap_section(r['text'])
        key = sec[:200]
        if key in seen:
            continue
        seen.add(key)
        picked.append((r, sec))
        if len(picked) == 3:
            break
    if not picked:
        print('  (no clinician note about the cancer with an assessment/plan in the last 180 days)')
    for r, sec in picked[::-1]:
        print(f'-- [{r["date"]}] {r["cls"] or r["typ"]}')
        print(sec)

    print('\n== IMAGING IMPRESSIONS, last 120 days (newest last)')
    rad = df[is_rad & ~df['cls'].str.contains('Pathology')]
    if 'date' in rad and len(rad):
        lo = (pd.Timestamp(cutoff) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        rad = rad[rad['date'] >= lo]
    out = []
    for _, r in rad.iloc[::-1].iterrows():
        t = r['text']
        head = clean(t[:160]).split('\n')[0][:110]
        if not IMG_RE.search(t[:600]) or re.search(r'(?i)^\s*(xr|x-ray|chest 1 view|chest 2 view)', head):
            continue
        m = None
        for m in IMPRESSION_RE.finditer(t):
            pass
        imp = clean(t[m.end():m.end() + 650]) if m else clean(t[-500:])
        if imp[:150] in seen:
            continue
        seen.add(imp[:150])
        out.append(f'-- [{r["date"]}] {head}\n{imp}')
        if len(out) == 4:
            break
    print('\n'.join(out[::-1]) if out else '  (none)')

    print('\n== PATHOLOGY DIAGNOSIS (newest last)')
    path = df[df['cls'].str.contains('Pathology')]
    out = []
    for _, r in path.iloc[::-1].iterrows():
        t = r['text']
        m = re.search(r'(?i)(final )?(pathologic |cytologic )?diagnosis\s*:?', t)
        seg = clean(t[m.start():m.start() + 400]) if m else clean(t[:300])
        out.append(f'-- [{r["date"]}] {seg}')
        if len(out) == 2:
            break
    print('\n'.join(out[::-1]) if out else '  (none)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/workspace/call/record')
