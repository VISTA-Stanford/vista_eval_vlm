"""Shared locations for the pneumonitis-variants task (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = "som-nero-plevriti-deidbdf"
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
#: Opaque ids, splits, labels by variant, staged volumes. Nothing keyed by a patient or accession.
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "pneumonitis-variants-v1_7"))
#: Id map and anything derived from reports (the infection pass).
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "pneumonitis-variants-v1_7-private"))
#: Run outputs (spec, Deliverables).
RUNS_DIR = MOUNT / "ct-findings-5-v1_7-runs" / "pneumonitis-variants"
