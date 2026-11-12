"""Settings for the death-agent-over-time task, and the one place it borrows the died-2yr-ehr task's code.

Spec (approved 2026-11-11): planning/vista_eval_vlm/2026-11-11-died-ehr-agent-over-time.md on the mount
(/mnt/su-vista-uscentral1/chaudhari_lab/phil/). A Haiku 5.5 Claude Code session reads a patient's record cut at a
decision point (treatment start, first restaging CT, 6, 12, 24 months) and gives a 0-100 chance of death within 1 and
2 years of that day, with its reasons. Two arms: fresh (each point alone) and memory (also shown its own earlier
risks and reasons). One pass per point and arm, in order; no hill-climb.

Reuse, not copies: this module sets the died-2yr-ehr task's settings (probability answers, record only, Haiku, this
task's data folder) through the environment and then imports that task's ``config``, ``agentic`` and ``adapters``, so
staging, the record whitelist, answer parsing, the AUROC and the engine pin are that task's own code. Import this
module before anything that imports ``config``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIED_2YR_EHR = HERE.parent / "died-2yr-ehr"

# --------------------------------------------------------------------------------------------------
# Fixed choices (spec, Decided) and where things live
# --------------------------------------------------------------------------------------------------
#: The patient sessions run on Haiku 5.5 (Phil 2026-11-11). Set here, not read from the shell, so a stray
#: PROGRAM_MODEL cannot change the run.
PROGRAM_MODEL = "claude-haiku-5-5"
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
#: Everything with a person_id, an opaque id next to a date, a label or record text: the patient list, the
#: decision-point table, the exporter's cohort, the record archive, labels, probe scores, per-row answers.
PRIVATE_DIR = Path(os.environ.get("OVER_TIME_PRIVATE_DIR", MOUNT / "died-ehr-over-time-v1_7-private"))
#: Aggregates only (counts, score tables, cost): what may be read without opening the private folder.
RESULTS_DIR = Path(os.environ.get("OVER_TIME_RESULTS_DIR", MOUNT / "died-ehr-over-time-v1_7-runs"))
#: The local working copy the sessions are staged from: patients/<record id>/ (the exporter's layout), rows.jsonl,
#: labels.jsonl, probes.parquet, seed_ids.jsonl. Rebuilt from PRIVATE_DIR/archive by died-2yr-ehr's build_data.py rehydrate.
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", Path.home() / "data" / "died-ehr-over-time-v1_7")).resolve()
WORKDIR = Path(os.environ.get("TASK_WORKDIR", Path.home() / ".cache" / "alo-tasks" / "died-ehr-over-time")).resolve()

#: The decision points, in the order they run. (key, words). Windows in days from first treatment (K29's day 0).
POINTS = (("start", "treatment start"), ("restaging", "first restaging CT"), ("m6", "6 months"),
          ("m12", "12 months"), ("m24", "24 months"))
POINT_KEYS = tuple(k for k, _w in POINTS)
#: Treatment start: the chest-containing CT closest to day 0 within -60..+30 (ties: the earlier one).
START_WINDOW = (-60, 30)
#: First restaging: the first chest-containing CT 28-120 days after treatment start.
RESTAGING_WINDOW = (28, 120)
#: 6/12/24 months: the latest chest-containing CT within 3 months (91 days) before the month mark.
MONTH_MARKS = {"m6": 183, "m12": 365, "m24": 730}
MONTH_LOOKBACK = 91
#: K29 exam types that contain the chest (chest, chest-abdomen-pelvis, skull-to-thigh body).
CHEST_EXAMS = ("chest", "chest_abdomen_pelvis", "whole_body")
#: First treatment categories (K29 subsetting) in the cohort: IV chemotherapy or an oral targeted drug.
SUBSETTINGS = ("iv_chemotherapy", "oral_targeted")
#: The two questions: (answer key in output.json, gold field in labels.jsonl, words in the prompt). The answer keys
#: are the died-2yr-ehr task's, so its answer parsing reads them unchanged.
HORIZONS = (("died_within_1_year", "label_1yr", "1 year"), ("died_within_2_years", "label_2yr", "2 years"))

#: Input tables on the mount.
K29_PHASE_TABLE = Path("/mnt/su-vista-uscentral1/vistabench/results/longitudinal_2026_11/k29/phase_table.parquet")
K16_DIR = Path("/mnt/su-vista-uscentral1/vistabench/results/v1_7_slim_20261030_k16/cross_modality/"
               "per_ct_landmark_labels")
#: The label source (spec: K16 per-CT death labels); every other K16 file holding the same CT must agree.
K16_LABELS = K16_DIR / "K16-PERCT-ehr" / "clinical__clinical_pretx_intersection_per_example.parquet"
K16_TASKS = {"label_1yr": "per_ct_died_any_cause_1_yr", "label_2yr": "per_ct_died_any_cause_2_yr"}
#: The K16 per-CT probes compared on the same rows. The stacked model has TEST rows only in K16 (it is fit on VAL),
#: so it has no score on these VAL rows; it stays listed so the score table says so instead of dropping it.
K16_PROBES = {
    "stacked": K16_DIR / "K16-PERCT-cross" / "late_stacked__clinical+ct+ehr+progress+rad_per_example.parquet",
    "motor": K16_DIR / "K16-PERCT-ehr" / "ehr_code__motor-arm-E-mlp-per-type_512_medspace_v1_tokfix_intersection_per_example.parquet",
    "ct_merlinonc": K16_DIR / "K16-PERCT-ct" / "ct__merlinonc_top_centered_pancancer_postdx_ep99_intersection_per_example.parquet",
    "clinical_pretx": K16_DIR / "K16-PERCT-ehr" / "clinical__clinical_pretx_intersection_per_example.parquet",
}
#: The died-2yr-ehr task's own cohort (person_id -> index tumour), reused for its patients.
DIED_2YR_EHR_COHORT = MOUNT / "died-2yr-ehr-v1_7-private" / "cohort.parquet"
DIED_2YR_EHR_SPLITS = MOUNT / "died-2yr-ehr-v1_7" / "splits"
DIED_2YR_EHR_ID_MAP = MOUNT / "died-2yr-ct-v1_7-private" / "id_map.jsonl"

# --------------------------------------------------------------------------------------------------
# Spend (spec: one full pass, $250 hard stop; stop if cost per call passes $0.10 in the first 100 calls)
# --------------------------------------------------------------------------------------------------
#: The same setting name as the died-2yr-ehr task's run cap. Here it is a hard ceiling: a chunk of calls starts only
#: when the spend so far plus every call in it hitting its own cap stays at or under it, so it can never be passed.
RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "250"))
EARLY_CALLS = 100
EARLY_MAX_COST_PER_CALL = 0.10
#: Calls per pass. Small passes let the dollar checks run often; a finished pass is never re-run (the engine serves
#: it from its archive), so a resume after a network stop repeats nothing.
CHUNK = int(os.environ.get("OVER_TIME_CHUNK", "25"))

# --------------------------------------------------------------------------------------------------
# The died-2yr-ehr task's code, under this task's settings
# --------------------------------------------------------------------------------------------------
os.environ["ANSWER"] = "probability"
os.environ["WITH_SUMMARY"] = "0"
os.environ["SUMMARY_ONLY"] = "0"
os.environ["PROGRAM_MODE"] = "agentic"
os.environ["PROGRAM_MODEL"] = PROGRAM_MODEL
os.environ["TASK_DATA_DIR"] = str(DATA_DIR)
os.environ["TASK_WORKDIR"] = str(WORKDIR)
if str(DIED_2YR_EHR) not in sys.path:
    sys.path.insert(0, str(DIED_2YR_EHR))

import config  # noqa: E402  (died-2yr-ehr's; puts the engine on sys.path)

assert config.PROBABILITY and not config.WITH_SUMMARY and config.PROGRAM_MODEL == PROGRAM_MODEL
assert config.DATA_DIR == DATA_DIR, "config was imported before common set the data folder"
