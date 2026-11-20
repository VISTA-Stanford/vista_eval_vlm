"""Build the probe tool's per-patient files (config.PROBES; spec planning/vista_eval_vlm/2026-11-09-died-2yr-ehr-probe-tool.md).

Six published v1.7 death probes, one per kind of input. For each patient a session can see:
  - the patient's score from each probe at 1 and 2 years: TRAIN patients get the out-of-fold score (a fit that never
    saw their label; vista-eval ``python -m vistaeval.eval.oof``, files in MOUNT_DATA_DIR/probes/oof/), VAL patients
    the published full-TRAIN score; "not available" where the probe has no input for the patient;
  - each probe's TRAIN facts, all from this task's own TRAIN labels and the out-of-fold scores: AUROC, coverage and a
    5-band calibration table (score band -> how many died). No VAL label is used anywhere in this script.
TEST patients get no file: TEST stays sealed (built only with --with-test, on its own go).

Also fits the headline bar: the six-probe combination, a logistic regression on the six scores plus a missing flag
each, fit on TRAIN out-of-fold scores and this task's TRAIN labels (default settings, never tuned on VAL), applied to
VAL. Written to PRIVATE_DIR/probes/combination_val.parquet (id, horizon, score) for scripts/score_baselines.py.

Checks before writing (exit 1 on any): every patient's split equals the probe's split; every published TRAIN/VAL
patient of this task found in the out-of-fold or published file has one row. Prints, per probe and horizon, the
out-of-fold TRAIN vs published VAL score spread (10th/50th/90th percentile and the two-sample KS statistic), so a
calibration table built on TRAIN can be trusted on VAL.

    uv run --with scipy python scripts/build_probe_tool.py              (mount + local copy; prints aggregates only)
    uv run python scripts/build_probe_tool.py --local-only              (copy the mount's files to config.PROBES_DIR)

Writes MOUNT_DATA_DIR/probes/files/<id>.json (durable copy, inside the secure environment) and config.PROBES_DIR/<id>.json
(local working copy), read-only, plus MOUNT_DATA_DIR/probes/probe_stats.json (aggregates). Never prints an id or row.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RESULTS = Path("/mnt/su-vista-uscentral1/vistabench/results/v1_7")
DATASET = "progression_recurrence_survival"
ID_MAP = config.ID_MAP_DIR / "id_map.jsonl"
#: Out-of-fold scores with every fold at the published fit's C (Phil 2026-11-10); probes/oof/ holds the first run.
OOF_DIR = config.MOUNT_DATA_DIR / "probes" / "oof_published_c"
MOUNT_FILES = config.MOUNT_DATA_DIR / "probes" / "files"
STATS = config.MOUNT_DATA_DIR / "probes" / "probe_stats.json"
COMBINATION = config.PRIVATE_DIR / "probes" / "combination_val.parquet"
#: (key, what the session is told it reads, results folder, out-of-fold folder, result stem). Spec, Decided: the menu.
PROBES = (
    ("radiology_report_text", "the patient's radiology report text before the cutoff (Gemini text embedding, linear "
     "probe)", "text_radiology", "text_radiology", "gemini_emb_001"),
    ("pathology_report_text", "the patient's pathology report text before the cutoff (Gemini text embedding, linear "
     "probe)", "text_pathology", "text_pathology", "gemini_emb_001"),
    ("chest_ct", "the latest chest CT before the cutoff (MerlinOnc pan-cancer CT model, linear probe)", "ct", "image",
     "merlinonc_top_centered_pancancer_postdx_ep99"),
    ("clinical_risk_factors", "structured clinical risk factors before treatment: age, sex, stage, histology, smoking "
     "and similar (XGBoost)", "clinical", "clinical", "clinical_pretx_xgb"),
    ("patient_summary_text", "a language-model summary of the record at the cutoff (Gemini text embedding, linear "
     "probe)", "text_llm_summary", "text_llm_summary", "gemini_emb_001"),
    ("coded_ehr_history", "the coded EHR history (diagnoses, drugs, procedures, labs) before the cutoff (MOTOR "
     "foundation model run-vista-02, linear probe)", "ehr_code", "ehr_code", "run-vista-02_512_medspace_v1_tokfix"),
)
#: (words, vista_bench task, gold field in splits/*.jsonl)
HORIZONS = (("1 year", "died_any_cause_1_yr", "label_1yr"), ("2 years", "died_any_cause_2_yr", "label"))
N_BANDS = 5
ABOUT = (
    "Risk of death from six prediction models (probes), each trained on a different part of the record of other "
    "patients in this cohort. A probe's score is its own scale, not a calibrated chance: read it through that probe's "
    "calibration table, which says how many patients in each score band actually died. Each probe's AUROC (0.5 is "
    "chance, 1.0 perfect) and coverage are measured on the training patients. A probe that could not see this "
    "patient's input says not available."
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_ro(path: Path, text: str) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o444)


def load_scores(splits: dict[str, list[dict]], to_pid: dict[str, str], with_test: bool) -> pd.DataFrame:
    """One row per (probe, horizon, task id): split, this task's label, score (NaN when the probe has no input)."""
    rows = []
    for key, _what, res_folder, oof_folder, stem in PROBES:
        oof = pd.read_parquet(OOF_DIR / oof_folder / f"{stem}_oof.parquet")
        pub = pd.read_parquet(RESULTS / res_folder / DATASET / f"{stem}_per_example.parquet",
                              columns=["task", "split", "person_id", "score"])
        for words, task, field in HORIZONS:
            o = oof[oof["task"] == task].assign(person_id=lambda d: d["person_id"].astype(str), split="train")
            p = pub[(pub["task"] == task) & pub["split"].isin(["val", "test"])]
            p = p.assign(person_id=p["person_id"].astype(str))
            all_pub = pub[pub["task"] == task].assign(person_id=lambda d: d["person_id"].astype(str))
            for frame, name in ((o, "out-of-fold"), (p, "published")):
                if frame["person_id"].duplicated().any():
                    raise SystemExit(f"{key} {task}: {name} file has more than one row per patient")
            score = dict(zip(o["person_id"], o["score"])) | dict(zip(p["person_id"], p["score"]))
            probe_split = dict(zip(all_pub["person_id"], all_pub["split"]))
            for split, recs in splits.items():
                if split == "test" and not with_test:
                    continue
                for r in recs:
                    pid = to_pid[r["id"]]
                    if pid in probe_split and probe_split[pid] != split:
                        raise SystemExit(f"{key} {task}: a patient's split differs between this task and the probe")
                    rows.append({"probe": key, "horizon": words, "id": r["id"], "split": split, "label": r[field],
                                 "score": score.get(pid, np.nan)})
    return pd.DataFrame(rows)


def probe_facts(df: pd.DataFrame) -> dict:
    """Per probe and horizon, from TRAIN only: AUROC, coverage, calibration bands; plus the TRAIN-vs-VAL spread."""
    from scipy.stats import ks_2samp

    facts = {}
    for key, *_ in PROBES:
        facts[key] = {}
        for words, *_ in HORIZONS:
            d = df[(df["probe"] == key) & (df["horizon"] == words)]
            tr = d[(d["split"] == "train") & d["score"].notna()]
            va = d[(d["split"] == "val") & d["score"].notna()]
            n_tr = int((d["split"] == "train").sum())
            edges = np.quantile(tr["score"], np.linspace(0, 1, N_BANDS + 1))
            band = np.clip(np.searchsorted(edges, tr["score"], side="right") - 1, 0, N_BANDS - 1)
            bands = []
            for b in range(N_BANDS):
                m = band == b
                bands.append({"score_from": round(float(edges[b]), 4), "score_to": round(float(edges[b + 1]), 4),
                              "patients": int(m.sum()), "died_percent": round(100 * float(tr["label"][m].mean()), 1)})
            q = lambda s: [round(float(x), 4) for x in np.quantile(s, [0.1, 0.5, 0.9])]  # noqa: E731
            facts[key][words] = {
                "train_auroc": round(float(roc_auc_score(tr["label"], tr["score"])), 3),
                "coverage_percent": round(100 * len(tr) / n_tr, 0),
                "calibration": bands,
                "_spread": {"train_p10_p50_p90": q(tr["score"]), "val_p10_p50_p90": q(va["score"]),
                            "ks": round(float(ks_2samp(tr["score"], va["score"]).statistic), 3),
                            "n_train": len(tr), "n_val": len(va)},
            }
    return facts


def combination(df: pd.DataFrame) -> pd.DataFrame:
    """Six-probe logistic regression per horizon: fit on TRAIN (out-of-fold scores, this task's labels), score VAL."""
    keys = [k for k, *_ in PROBES]
    out = []
    for words, *_ in HORIZONS:
        wide = df[df["horizon"] == words].pivot(index="id", columns="probe", values="score")[keys]
        meta = df[(df["horizon"] == words) & (df["probe"] == keys[0])].set_index("id")[["split", "label"]]
        tr_ids = meta.index[meta["split"] == "train"]
        fill = wide.loc[tr_ids].mean()
        X = pd.concat([wide.fillna(fill), wide.isna().astype(float).add_suffix("_missing")], axis=1)
        sc = StandardScaler().fit(X.loc[tr_ids])
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(X.loc[tr_ids]), meta.loc[tr_ids, "label"])
        va_ids = meta.index[meta["split"] == "val"]
        out.append(pd.DataFrame({"id": va_ids, "horizon": words,
                                 "score": clf.predict_proba(sc.transform(X.loc[va_ids]))[:, 1]}))
    return pd.concat(out, ignore_index=True)


def patient_file(df: pd.DataFrame, facts: dict, i: str) -> str:
    mine = df[df["id"] == i].set_index(["probe", "horizon"])["score"]
    probes = []
    for key, what, *_ in PROBES:
        entry = {"probe": key, "reads": what, "horizons": {}}
        for words, *_ in HORIZONS:
            f = facts[key][words]
            s = mine.get((key, words), np.nan)
            entry["horizons"][words] = {
                "this_patient_score": None if pd.isna(s) else round(float(s), 4),
                "available": not pd.isna(s),
                "train_auroc": f["train_auroc"], "coverage_percent": f["coverage_percent"],
                "calibration": f["calibration"],
            }
        probes.append(entry)
    return json.dumps({"about": ABOUT, "probes": probes}, indent=1)


def build(with_test: bool) -> int:
    splits = {s: _jsonl(config.MOUNT_DATA_DIR / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    to_pid = {m["id"]: str(m["person_id"]) for m in _jsonl(ID_MAP)}
    df = load_scores(splits, to_pid, with_test)
    facts = probe_facts(df)
    print("probe                   horizon  TRAIN AUROC  coverage  spread TRAIN (p10/p50/p90) vs VAL, KS")
    for key, *_ in PROBES:
        for words, *_ in HORIZONS:
            f = facts[key][words]
            sp = f["_spread"]
            print(f"{key:<23} {words:<8} {f['train_auroc']:.3f}        {f['coverage_percent']:>3.0f}%     "
                  f"{sp['train_p10_p50_p90']} vs {sp['val_p10_p50_p90']}  KS {sp['ks']:.3f}")
    comb = combination(df)
    COMBINATION.parent.mkdir(parents=True, exist_ok=True)
    comb.to_parquet(COMBINATION, index=False)
    print(f"six-probe combination: {comb['id'].nunique()} VAL patients x {comb['horizon'].nunique()} horizons written")
    MOUNT_FILES.mkdir(parents=True, exist_ok=True)
    config.PROBES_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(df["id"].unique())
    for i in ids:
        text = patient_file(df, facts, i)
        write_ro(MOUNT_FILES / f"{i}.json", text)
        write_ro(config.PROBES_DIR / f"{i}.json", text)
    STATS.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    avail = df.groupby("split")["score"].apply(lambda s: round(100 * s.notna().mean(), 1)).to_dict()
    print(f"probe files: {len(ids)} patients ({df.groupby('split')['id'].nunique().to_dict()}); "
          f"share of (probe, horizon) scores available by split, %: {avail}")
    return 0


def local_only() -> int:
    config.PROBES_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(MOUNT_FILES.glob("*.json")):
        write_ro(config.PROBES_DIR / src.name, src.read_text(encoding="utf-8"))
        n += 1
    print(f"local copies in PROBES_DIR: {n}")
    return 0 if n else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-only", action="store_true", help="copy the mount's probe files to config.PROBES_DIR")
    p.add_argument("--with-test", action="store_true", help="also write TEST patients' files (own go; TEST is sealed)")
    a = p.parse_args()
    return local_only() if a.local_only else build(a.with_test)


if __name__ == "__main__":
    sys.exit(main())
