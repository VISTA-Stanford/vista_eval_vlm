A tool here runs inside the session's sealed container (Python 3.11 with numpy, scipy, Pillow, nibabel, pandas, pyarrow
and duckdb; no network).

- digest.py: the notes reader. `python3 -B digest.py <record folder>` prints the assessment/plan sections of the 3 most
  recent substantial clinician notes about the cancer (last 180 days before the cutoff), the impressions of the latest
  CT/PET/MRI reports (last 120 days) and the latest pathology diagnoses, about 8-10 KB. Copied from the died-2yr-ehr
  summary arm; its patterns were written for thoracic cancer notes.
