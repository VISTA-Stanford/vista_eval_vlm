"""Export every agent and LLM answer run the VISTA-Bench paper reads to one table per run (paper-code stage 2).

Writes /mnt/su-vista-uscentral1/vistabench/results/paper_agent_answers_v1_7/<run_id>/answers.parquet (+ provenance)
and MANIFEST.json (agent key -> run id, program version, role, rows, patients). Inside the secure environment.

Which version a run is: for a hill-climbed agent the export checks the version named below against the climb's
run_summary.json (``winner`` = the best validation score it measured, ``seed`` = the version it started from), and
refuses on a mismatch, so a typed version can no longer silently differ from what the climb picked. A run kept on
purpose that is neither (the Haiku ladder's round 2) is marked ``other``. Which answers file holds which version still
rests on its file or folder name (the answers carry no version metadata); the climb's version hash is recorded.
Decision points the over-time task has no label for are kept (``label`` empty) and counted in the manifest
(``rows_without_label``); the figure readers drop them.
Not exported here: the CT findings agent (the mount copy has no scan number; its raw answers are on phil-sllm-02,
see scripts/paper/copy_agent_programs.sh).

  cd <vista_eval_vlm checkout> && UV_PYTHON=3.11 uv run python scripts/paper/export_paper_answers.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalset import paper_export as PE

M = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
AVS = M / "planning/vista_bench/paper/results/agents-vs-our-models-subgroups"
F5 = M / "planning/vista_bench/paper/results/fig5-vlm-agent/data"
EHR_IDMAP = M / "died-2yr-ct-v1_7-private/id_map.jsonl"
EHR_SPLIT = M / "died-2yr-ehr-v1_7/splits/test.jsonl"
SONNET = M / "died-2yr-ehr-v1_7-private-test-sonnet"
OT_PRIV = M / "died-ehr-over-time-v1_7-private-test"
OT_RUNS = OT_PRIV / "runs"
CLIMB = {
    "ehr_sonnet": M
    / "died-2yr-ehr-v1_7-runs/phil-sllm-02/prob-summary-r1/run_summary.json",
    "ehr_haiku_A": M
    / "died-2yr-ehr-v1_7-runs/phil-sllm-02/prob-summary-haiku-r1/run_summary.json",
    "ehr_haiku_R": M
    / "died-2yr-ehr-v1_7-runs/phil-sllm-02/prob-haiku-r1/run_summary.json",
    "ehr_haiku_L": M
    / "died-2yr-ehr-v1_7-runs/phil-sllm-02/prob-summaryonly-bare-r1/run_summary.json",
    "ct_death": M / "died-2yr-ct-v1_7-runs/phil-sllm-02/prob-r2/run_summary.json",
    "ct_death_first": M / "died-2yr-ct-v1_7-runs/phil-sllm-02/prob-r1/run_summary.json",
    "ot_haiku": M
    / "died-ehr-over-time-v1_7-runs/phil-sllm-01/climb-r1/run_summary.json",
    "ot_sonnet": M
    / "died-ehr-over-time-v1_7-runs/phil-sllm-02/climb-sonnet-restaging-r1/run_summary.json",
}
EHR_LABELS = f"{PE.STEP_LABEL_TABLE[1]} via the died-2yr-ehr task key (died-2yr-ehr-v1_7/splits/test.jsonl)"
OT_LABELS = f"{PE.BENCH}.per_ct_landmark_labels via the died-ehr-over-time task key (died-ehr-over-time-v1_7-private-test/labels.jsonl)"
CT_LABELS = (
    f"{PE.STEP_LABEL_TABLE[1]} via evalset_v1 step 1 (runs/v1/astra/step1_ct.csv)"
)

# key: (reader, reader args, climb, version tag, role, label table, run id, paper elements)
AGENTS = {
    "ehr_sonnet_v5": (
        "ehr",
        SONNET / "sonnet-program-v5.test-program-v5-test-a95450d6.answers.json",
        "ehr_sonnet",
        "program-v5",
        "winner",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-summary-r1/program-v5/test",
        "Fig 2b, Fig 4, T1, C5, D5, T3",
    ),
    "ehr_sonnet_v0": (
        "ehr",
        SONNET / "sonnet-program-v0.test-test-program-v0-test-dfa15c5d.answers.json",
        "ehr_sonnet",
        "program-v0",
        "seed",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-summary-r1/program-v0/test",
        "Fig 2b (untuned), T1, D5, T3",
    ),
    "ehr_haiku_A_v0": (
        "ehr",
        AVS / "ladder-test-answers/A-v0.answers.json",
        "ehr_haiku_A",
        "program-v0",
        "winner",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-summary-haiku-r1/program-v0/test",
        "T1, D2",
    ),
    "ehr_haiku_A_v2": (
        "ehr",
        AVS / "ladder-test-answers/A-v2.answers.json",
        "ehr_haiku_A",
        "program-v2",
        "other",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-summary-haiku-r1/program-v2/test",
        "T3 crossover (round-2 error per the appendix audit)",
    ),
    "ehr_haiku_R_v0": (
        "ehr",
        AVS / "ladder-test-answers/R-v0.answers.json",
        "ehr_haiku_R",
        "program-v0",
        "seed",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-haiku-r1/program-v0/test",
        "T1",
    ),
    "ehr_haiku_R_v8": (
        "ehr",
        AVS / "ladder-test-answers/R-v8.answers.json",
        "ehr_haiku_R",
        "program-v8",
        "winner",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-haiku-r1/program-v8/test",
        "T1, T3",
    ),
    "ehr_haiku_L_v0": (
        "ehr",
        AVS / "ladder-test-answers/L-v0.answers.json",
        "ehr_haiku_L",
        "program-v0",
        "winner",
        EHR_LABELS,
        "agents/died-2yr-ehr/prob-summaryonly-bare-r1/program-v0/test",
        "T1",
    ),
    "over_time_haiku_v5": (
        "ot",
        OT_RUNS / "over-time-test-climbed-r1/answers",
        "ot_haiku",
        "program-v5",
        "winner",
        OT_LABELS,
        "agents/died-ehr-over-time/climb-r1/program-v5/test",
        "Fig 3a, ED2",
    ),
    "over_time_haiku_v0": (
        "ot",
        OT_RUNS / "over-time-test-r1/answers",
        "ot_haiku",
        "program-v0",
        "seed",
        OT_LABELS,
        "agents/died-ehr-over-time/climb-r1/program-v0/test",
        "Fig 3a (zero-shot), ED2",
    ),
    "over_time_sonnet_v0": (
        "ot",
        OT_RUNS / "over-time-test-sonnet-zeroshot-r1/answers",
        "ot_sonnet",
        "program-v0",
        "seed",
        OT_LABELS,
        "agents/died-ehr-over-time/climb-sonnet-restaging-r1/program-v0/test",
        "Extended Data Fig 3a",
    ),
    "over_time_sonnet_restaging_v8": (
        "ot",
        OT_RUNS / "over-time-test-sonnet-restaging-tuned-r1/answers",
        "ot_sonnet",
        "program-v8",
        "winner",
        OT_LABELS,
        "agents/died-ehr-over-time/climb-sonnet-restaging-r1/program-v8/test",
        "Extended Data Fig 3a",
    ),
    "ct_death_v4": (
        "ct",
        F5 / "agent_ct_death_r2_v4_test_rows.csv",
        "ct_death",
        "program-v4",
        "winner",
        CT_LABELS,
        "agents/died-2yr-ct/prob-r2/program-v4/test",
        "Fig 2b, D4, T1, T3",
    ),
    "ct_death_v0": (
        "ct",
        F5 / "agent_ct_death_r2_v0_test_rows.csv",
        "ct_death_first",
        "program-v0",
        "seed",
        CT_LABELS,
        "agents/died-2yr-ct/prob-r1/program-v0/test",
        "Fig 2b (untuned), D4; the first climb's seed, not prob-r2's own round 0",
    ),
}
#: General-purpose LLM answer files Fig 2b reads (results/fig3-v12-interim/build_c_auroc_v15.py); exported when present.
LLM_FILES = (
    [
        f"v1/{m}/{s}.csv"
        for m in ("astra", "argon", "opus", "medgemma")
        for s in ("step1_ct", "step1_summary", "step2_ct")
    ]
    + ["v1/medgemma/step3_ct.csv"]
    + [
        f"fair_full/qwen3vl/{s}.csv"
        for s in ("step1_ct", "step1_summary", "step2_ct", "step3_ct")
    ]
    + [
        f"prob-v1/{st}/step1_summary.csv"
        for st in (
            "astra__astra_prob",
            "argon__argon_prob",
            "opus__opus_prob",
            "opus__sonnet_prob",
            "opus__haiku_prob",
        )
    ]
    + [
        f"prob-v1/{st}/step{n}_ct.csv"
        for st in (
            "astra__astra_prob",
            "argon__argon_prob",
            "opus__opus_prob",
            "opus__sonnet_prob",
            "opus__haiku_prob",
        )
        for n in (1, 2)
    ]
)


#: Fig 2's rule (build_c_auroc_v15.complete_prob): a probability CT answers file counts once every case was asked.
COMPLETE_ROWS = {"step1_ct": 1720, "step2_ct": 5000}


def complete(p: Path, rel: str) -> bool:
    need = COMPLETE_ROWS.get(p.stem) if rel.startswith("prob-v1/") else None
    if need is None:
        return True
    import pandas as pd

    return len(pd.read_csv(p, usecols=["p_yes"])) >= need


def check_version(climb_key, tag, role) -> str:
    return PE.check_climb_version(CLIMB[climb_key], tag, role)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PE.OUT_ROOT)
    a = ap.parse_args()
    manifest = {}
    for key, (kind, src, climb, tag, role, labels, run_id, used) in AGENTS.items():
        how = check_version(climb, tag, role)
        if kind == "ehr":
            rows = PE.ehr_agent_rows(src, EHR_IDMAP, EHR_SPLIT)
            inputs = [str(src), str(EHR_IDMAP), str(EHR_SPLIT)]
        elif kind == "ot":
            rows = PE.over_time_rows(OT_PRIV, src)
            inputs = sorted(str(p) for p in Path(src).glob("*-fresh.json")) + [
                str(OT_PRIV / "points.parquet"),
                str(OT_PRIV / "labels.jsonl"),
            ]
        else:
            rows = PE.ct_rows_file(src)
            inputs = [str(src)]
        version = f"{Path(CLIMB[climb]).parent.name}/{tag}"
        df = PE.export_agent_answers(rows, run_id, version, labels, inputs, how)
        f = PE.write_run(df, a.out)
        manifest[key] = {
            "run_id": run_id,
            "program_version": version,
            "role": role,
            "version_source": how,
            "rows": len(df),
            "patients": int(df.person_id.nunique()),
            "rows_without_label": int(df["label"].isna().sum()) if "label" in df else 0,
            "used_by": used,
        }
        print(
            f"{key}: {len(df):,} rows, {df.person_id.nunique():,} patients -> {f.parent.name} ({role})"
        )
    for rel in LLM_FILES:
        p = PE.EVALSET_RUNS / rel
        if not p.exists():
            print(f"skip (absent): {rel}")
            continue
        if not complete(p, rel):
            print(
                f"skip (still being written; Fig 2 uses a probability CT run only once complete): {rel}"
            )
            continue
        df = PE.export_evalset_file(p)
        f = PE.write_run(df, a.out)
        key = "llm/" + rel[: -len(".csv")]
        manifest[key] = {
            "run_id": df.run_id.iloc[0],
            "program_version": df.program_version.iloc[0],
            "role": "llm",
            "version_source": df.attrs["version_source"],
            "rows": len(df),
            "patients": int(df.person_id.nunique()),
            "used_by": "Fig 2b",
        }
        print(f"{key}: {len(df):,} rows, {df.person_id.nunique():,} patients")
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(manifest)} runs to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
