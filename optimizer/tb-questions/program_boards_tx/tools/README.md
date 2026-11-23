A tool here runs inside the session's sealed container (Python 3.11 with numpy, scipy, Pillow, nibabel, pandas, pyarrow
and duckdb; no network).

- digest.py: the notes reader. `python3 -B digest.py <record folder>` prints a fact sheet (tumor board year, registry
  cancer at first diagnosis with stage and EGFR/ALK/PD-L1, registry treatment events, cancer drugs with first/last
  dates, passages of the last year that mention a tumor board or presenting the case, a one-line list of the notes of
  the last 30 days), then the assessment/plan sections of the 3 most recent substantial clinician notes about the
  cancer (last 180 days before the cutoff), the impressions of the latest CT/PET/MRI reports (last 120 days) and the
  latest pathology diagnoses, about 15-18 KB. The fact sheet is printed even when the record has no notes.
