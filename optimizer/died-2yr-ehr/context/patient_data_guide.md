# Where things are in this record

One patient's record, cut 7 days before their first cancer treatment. This says what each file holds and where
each kind of fact lives, so you can go straight to it. Files are read-only; a table with no rows is absent.
Query parquet with duckdb or pandas (`duckdb.sql("select ... from 'record/measurement.parquet'")`); select the
columns you need, never `select *` on a wide table.

## The files

- `notes.txt`: every note, oldest first. Each starts `===== NOTE <note_id> | <datetime> | <type> =====`. The
  type is blank ("None") for about half, so the header alone won't find radiology or pathology reports. A median
  patient has about 55 notes (130 KB); some have thousands (up to 20 MB). Grep it; don't read it whole.
- `note.parquet`: the same notes as a table, with the labels the header lacks. `note_class_concept_name`
  ("Progress note", "Radiology Note", "Pathology Note", "Consult note", "History and physical note",
  "Emergency department Note", "Telephone encounter Note", "Letter", "Adverse event note", "Plan of care note",
  "Instructions"; "No matching concept" for about 40%), `note_type`, `order_type`, `note_date`, `note_text`.
  `note_id` is the number in the `notes.txt` header.
- OMOP tables, one parquet each. Every `<x>_concept_id` has its name beside it as `<x>_concept_name`; read the
  names. `..._source_value` columns are JSON text (`json_extract_string(col, '$.key')`).
- `registry_*.parquet`: the cancer registry's abstract of this cancer, one row (events: one row per event).
- `meta.json`: the patient's opaque id and the cutoff date.

Dates are shifted for privacy by one offset per patient: intervals inside the record are exact, a calendar date
is right to within about a month.

## Where each kind of fact lives

| Fact | Look here first | Also |
|---|---|---|
| Cancer site, histology | `registry_diagnoses`: `primarysite`, `histologictypeicdo3` (ICD-O-3), each with a `...description` label; `texthistologytitle`, `textprimarysitetitle` | pathology notes; `condition_occurrence` diagnosis names |
| Stage at diagnosis | `registry_diagnoses` clinical stage group: `ajcctnmclinstagegroup` (AJCC 8th era, about 6 in 10 patients) or `tnmclinstagegroup` (older editions, about 4 in 10); one of the two is filled for 99% of patients. With T, N, M in `ajcctnmclint/n/m` or `tnmclint/n/m`; `tnmeditionnumber` says which edition | notes (consult, radiology, tumour board). Pathologic stage is removed. There are no OMOP stage rows in this record |
| Spread at diagnosis | `registry_diagnoses` `metsatdxbone`, `metsatdxbrain`, `metsatdxliver`, `metsatdxlung`, `metsatdxdistantln`, `metsatdxother` (about 3 in 4 patients); older cases `csmetsatdx`, `csmetsatdxbone/brain/liver/lung`. `tumorsizeclinical` (mm) | radiology reports |
| EGFR, ALK, PD-L1, KRAS | `registry_diagnoses` `egfrmutationalanalysis`, `alkrearrangement`, `stanfordpdl1` (each about 3 in 10 patients), `kras` (fewer) | pathology and oncology notes (grep `egfr`, `l858r`, `exon 19`, `alk`, `pd-?l1`, `tps`) |
| Planned first treatment | `registry_treatments` first-course codes (no dates): surgery `rxsummsurgprimsite` (from 2023 `rxsummsurgprimsite2023`), radiation `phase1radiationtreatmentmodality`, `radlocationofrx`, `phase1totaldose`, `phase1numberoffractions`, systemic `rxsummchemo`, `rxsummbrm` (immunotherapy), `rxsummhormone`, `rxsummpalliativeproc`, `intentoftreatment` (about 6 in 10), `reasonfornosurgery`, `reasonfornoradiation`, `rxsummsurgradseq`, `rxsummsystemicsurseq`, `neoadjuvanttherapy` | consult and plan-of-care notes; `episode` (treatment plan text, about 1 in 10 patients) |
| Drugs given before the cutoff | `drug_exposure` (`drug_concept_name`, start date), `drug_era` | notes |
| Procedures, biopsies, prior surgery | `procedure_occurrence` (`procedure_concept_name`; about 4 in 10 rows have concept 0: read `procedure_source_value`) | registry `rxsummdxstgproc` (diagnostic/staging procedure) |
| Other conditions | `condition_occurrence` (`condition_concept_name`, start date; source JSON has `dx_name`) | history and physical, consult notes |
| Imaging | `image_occurrence`: one row per series (many per scan). Modality in `modality_source_value` JSON `$.modality`, upper case (`CT`, `PT` = PET, `MR`, `CR`/`DX` = x-ray; `SR`, `PR`, `KO`, `OT` are not images); `_study_description`, `image_occurrence_date`. `_note_id` is the radiology report's `note_id` (present for most rows; a report written after the cutoff is not here) | notes with class "Radiology Note" or type "Radiology Study Results" |
| Pathology | notes with class "Pathology Note" | registry `diagnosticconfirmation` |
| Labs, vitals, weight | `measurement` (`measurement_concept_name`, `value_as_number`, `unit_concept_name`, `range_low`/`range_high`, date). Units are missing on about a quarter of numeric rows; check before comparing. Registry `height`, `weight` at diagnosis | |
| Flowsheet entries | `observation`: most rows (about 85%) have concept 0; their meaning is in `observation_source_value` JSON `$.flo_meas_name` (BMI, oxygen device, pain score and so on) | |
| Performance status (ECOG, Karnofsky) | notes (grep `ecog`, `karnofsky`, `performance status`); almost never a structured row | |
| Smoking | `observation` with `observation_concept_id` 43054909 (about 8 in 10 patients); registry `registry_misc_details` `tobaccousecigarettes`, `tobaccousesmokingstatus` | notes |
| Age, sex | `registry_misc_details` `ageatdiagnosis`, `sex`; `person` `year_of_birth`, `gender_concept_name` | |
| Visits, admissions, ED | `visit_occurrence` (`visit_concept_name`, start and end dates), `visit_detail` | ED and discharge notes |
| Where diagnosed or treated | registry `classofcase`, `textplaceofdiagnosis`; `rxsumm*` = given anywhere, `rxhosp*` = given here | notes with type "Outside Organization Document" |
| Registry dates | `dateofdiagnosis_date`, `datefirstpositivebx_date`, `dateof1stcontact_date` (parsed); the column without `_date` is the raw `YYYYMMDD` text | |

## Reading registry codes

- Every value is text: compare `'01'`, not `1`. A `...description` column beside a code is its plain-English
  label; read it when filled. `99` usually means unknown and `88` or `98` not applicable: neither means "no".
- Surgery `rxsummsurgprimsite`: `00` none, `10`-`90` surgery of the primary site, `98` not applicable, `99`
  unknown. From 2023 `rxsummsurgprimsite2023`: `X000` none; codes ending `980`, `990` or `999` not applicable or
  unknown.
- Systemic `rxsummchemo`, `rxsummbrm`, `rxsummhormone`: `00` none, `01`-`03` given, `82`-`87` recommended but not
  given (the description says why; `87` refused), `88` recommended, unknown if given, `99` unknown.
- Radiation `phase1radiationtreatmentmodality`: `00` none, `99` unknown, anything else a modality.
  `radlocationofrx` `4` = all given elsewhere.
- The registry records the first course only, wherever it was given; OMOP records only what happened at this
  centre. A treatment in the registry with nothing in OMOP was usually given elsewhere, not missing.

## Sources that mislead

- A word in a note is a mention, not a fact: negations ("no evidence of"), rule-outs, family history, templates
  and text copied forward all match a grep. Read the sentence.
- A drug named in a note is not proof it was given: check `drug_exposure` or the registry codes.
- Stage words in OMOP diagnosis names are mostly other diseases' stages (kidney disease, pressure injuries).
- "EGFR" in labs is kidney function (estimated GFR), not the gene.
- `observation_period` does not say when the patient was seen; use visits and notes.
- No record of something is not proof it never happened: care elsewhere often leaves nothing here, and a patient
  referred in shortly before treatment may have few notes.
