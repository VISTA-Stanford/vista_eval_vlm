"""Tumor board comparison rows: embedding probes vs Argon and Opus asked directly, on the same TEST patients.

Rows scored (spec planning/vista_eval_vlm/2026-11-20-tb-agent-questions.md; hand-off
planning/vista_eval_vlm/session/2026-11-20-tb-handoff-comparison-rows.md):
  step 6 (treatment, summary at the tumor board): Argon, Opus, Gemini summary-embedding probe
  step 4 (progression, CT): Argon, Opus (current + prior CT), MerlinOnc v2.2 probe on the single CT and on the
         current + prior CT pair

Every model is scored with the eval set's own functions (``evalset.score``: one-vs-rest macro AUROC over the options
present, top pick balanced accuracy, tumor board pick in top two; binary AUROC with ties half). One set of patient
bootstrap draws per step is shared by every model, so the paired difference of two models moves with the same
patients. A probe's answer for a patient is its class probabilities (one per option, codes 0-4 as on the cards).

Checks that fail loudly: a probe's label must equal the eval set's label for every patient it covers; Argon's and
Opus's point AUROC recomputed here must equal the published scores file.

    uv run python scripts/tb_comparison_rows.py [--n-boot 2000] [--out-dir <mount>/runs/tb-direct-test/scores]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evalset.score import _option_metrics, _weighted_auroc

VB = Path("/mnt/su-vista-uscentral1/vistabench")
EVALSET = VB / "vlm/evalset_v1"
RUN = EVALSET / "runs/tb-direct-test"
LLMS = {"Argon": "argon__argon_tb", "Opus": "opus__opus_tb"}
TREATMENT = {"board_early_stage": "early_stage_management_answer", "board_systemic": "optimal_systemic_therapy_answer"}
PROGRESSION = {"board_progression": "progression_assessment_answer"}
CODES = [0, 1, 2, 3, 4]
SUMMARY_PROBE = VB / "results/v1_7/text_llm_summary/tb_classification/gemini_emb_001_per_example.parquet"
CT_PROBES = {
    "MerlinOnc single CT": VB / "results/v1_7/ct/tb_classification/merlinonc_10crops_v2.2_per_example.parquet",
    "MerlinOnc CT + prior CT": VB / "results/v1_7/ct/tb_classification/merlinonc_10crops_v2.2_prior_pair_per_example.parquet",
}


# -- answers: {question_id: frame(person_id, label, P or score)} per model --------------------------------------------

def llm_answers(recipe: str, step: int, arm: str) -> dict:
    df = pd.read_csv(RUN / recipe / f"step{step}_{arm}.csv", dtype={"person_id": str})
    out = {}
    for qid, q in df.groupby("question_id"):
        q = q[q["label"].astype(int) >= 0]
        if step == 6:
            parsed = [json.loads(a) if isinstance(a, str) and a.startswith("{") else None for a in q["answer"]]
            P = np.array([[p[str(c)] for c in CODES] if p else [np.nan] * len(CODES) for p in parsed])
            out[qid] = pd.DataFrame({"person_id": q["person_id"].to_numpy(), "label": q["label"].astype(int).to_numpy(),
                                     "P": list(P), "readable": [p is not None for p in parsed]})
        else:
            s = pd.to_numeric(q["answer"], errors="coerce")
            out[qid] = pd.DataFrame({"person_id": q["person_id"].to_numpy(), "label": q["label"].astype(int).to_numpy(),
                                     "score": s.to_numpy(), "readable": s.notna().to_numpy()})
    return out


def probe_answers(path: Path, tasks: dict, sample: pd.DataFrame, multiclass: bool) -> tuple[dict, dict]:
    """(answers, coverage) for the probe's TEST rows of the sample's patients; coverage = {qid: (covered, asked)}."""
    pe = pd.read_parquet(path)
    pe = pe[pe["split"] == "test"].copy()
    pe["person_id"] = pe["person_id"].astype(str)
    out, cov = {}, {}
    for qid, task in tasks.items():
        col = f"label__{task}"
        want = sample[sample[col].notna() & (sample[col] >= 0)][["person_id", col]].rename(columns={col: "elabel"})
        p = pe[pe["task"] == task]
        if p["person_id"].duplicated().any():
            raise ValueError(f"{path.name} {task}: a patient has two TEST rows")
        m = want.merge(p, on="person_id", how="inner")
        bad = (m["label"].astype(int) != m["elabel"].astype(int)).sum()
        if bad:
            raise ValueError(f"{path.name} {task}: {bad} patients' probe label differs from the eval set's label")
        cov[qid] = (len(m), len(want))
        if multiclass:
            P = []
            for cls, pr in zip(m["classes"], m["proba"]):
                d = dict(zip([int(c) for c in cls], pr))
                P.append([float(d.get(c, 0.0)) for c in CODES])
            out[qid] = pd.DataFrame({"person_id": m["person_id"], "label": m["label"].astype(int), "P": P,
                                     "readable": True})
        else:
            out[qid] = pd.DataFrame({"person_id": m["person_id"], "label": m["label"].astype(int),
                                     "score": m["score"].astype(float), "readable": True})
    return out, cov


# -- metrics ----------------------------------------------------------------------------------------------------------

def q_metrics(a: pd.DataFrame, w: np.ndarray) -> tuple:
    """(AUROC, top-pick BA, top two) of one question's rows with patient weights ``w`` (BA/top two NaN if binary)."""
    r = a["readable"].to_numpy(dtype=bool)
    lab = a["label"].to_numpy()
    if "P" in a:
        P = np.stack(a["P"].to_numpy()).astype(float)
        m, _, ba, t2 = _option_metrics(P, r, lab, CODES, w)
        return m, ba, t2
    s = a["score"].to_numpy(dtype=float)[r]
    u, inv = np.unique(s, return_inverse=True)
    return _weighted_auroc(inv, len(u), lab[r].astype(float), w[r]), np.nan, np.nan


def score_step(models: dict, n_boot: int, seed: int, patients: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model rows (own covered patients) and paired differences (patients both models answered)."""
    pidx = {p: i for i, p in enumerate(patients)}
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients)).astype(float)
             for _ in range(n_boot)]
    qids = sorted({q for ans in models.values() for q in ans})

    def series(ans: dict, keep: dict | None = None):
        """(point, boot array) per question + MACRO, rows restricted to ``keep[qid]`` patients if given."""
        res = {}
        for qid in qids:
            a = ans[qid]
            if keep is not None:
                a = a[a["person_id"].isin(keep[qid])]
            pc = a["person_id"].map(pidx).to_numpy()
            pt = q_metrics(a, np.ones(len(a)))
            bt = np.array([q_metrics(a, d[pc]) for d in draws])
            res[qid] = (len(a), pt, bt)
        if len(qids) > 1:
            res["MACRO"] = (sum(v[0] for v in res.values()), tuple(np.nanmean([res[q][1] for q in qids], axis=0)),
                            np.nanmean(np.stack([res[q][2] for q in qids]), axis=0))
        return res

    rows, pairs = [], []
    own = {name: series(ans) for name, ans in models.items()}
    for name, res in own.items():
        for qid, (n, pt, bt) in res.items():
            rows.append({"model": name, "question_id": qid, "n": n, "auroc": pt[0],
                         "auroc_lo": np.nanpercentile(bt[:, 0], 2.5), "auroc_hi": np.nanpercentile(bt[:, 0], 97.5),
                         "top_pick_ba": pt[1], "top2": pt[2]})
    names = list(models)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keep = {q: set(models[a][q]["person_id"]) & set(models[b][q]["person_id"]) for q in qids}
            ra, rb = series(models[a], keep), series(models[b], keep)
            for qid in ra:
                d = ra[qid][2][:, 0] - rb[qid][2][:, 0]
                pairs.append({"model_a": a, "model_b": b, "question_id": qid, "n": ra[qid][0],
                              "auroc_a": ra[qid][1][0], "auroc_b": rb[qid][1][0],
                              "diff": ra[qid][1][0] - rb[qid][1][0],
                              "diff_lo": np.nanpercentile(d, 2.5), "diff_hi": np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def check_published(rows: pd.DataFrame, step: int) -> None:
    pub = pd.read_csv(RUN / "scores/tb_direct_test.csv")
    for name, recipe in LLMS.items():
        for _, r in rows[rows["model"] == name].iterrows():
            p = pub[(pub["step"] == step) & (pub["recipe"] == recipe) & (pub["question_id"] == r["question_id"])]
            if len(p) != 1 or not np.isclose(p["auroc"].iloc[0], r["auroc"], atol=1e-9):
                raise AssertionError(f"{name} step {step} {r['question_id']}: recomputed AUROC {r['auroc']:.6f} "
                                     f"!= published {p['auroc'].tolist()}")
    print(f"  check: Argon and Opus step {step} AUROC equal the published scores file")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(RUN / "scores"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)

    s6 = pd.read_parquet(EVALSET / "samples/step6_test.parquet")
    s6["person_id"] = s6["person_id"].astype(str)
    models6 = {n: llm_answers(r, 6, "summary") for n, r in LLMS.items()}
    models6["Summary embedding probe"], cov6 = probe_answers(SUMMARY_PROBE, TREATMENT, s6, multiclass=True)
    rows6, pairs6 = score_step(models6, a.n_boot, a.seed, sorted(s6["person_id"]))
    check_published(rows6, 6)

    s4 = pd.read_parquet(EVALSET / "samples/step4_test.parquet")
    s4["person_id"] = s4["person_id"].astype(str)
    models4 = {n: llm_answers(r, 4, "ct") for n, r in LLMS.items()}
    cov4 = {}
    for name, path in CT_PROBES.items():
        models4[name], c = probe_answers(path, PROGRESSION, s4, multiclass=False)
        cov4[name] = c
    rows4, pairs4 = score_step(models4, a.n_boot, a.seed, sorted(s4["person_id"]))
    check_published(rows4, 4)

    rows = pd.concat([rows6.assign(step=6), rows4.assign(step=4)])
    pairs = pd.concat([pairs6.assign(step=6), pairs4.assign(step=4)])
    rows.to_csv(out / "tb_comparison_rows.csv", index=False)
    pairs.to_csv(out / "tb_comparison_rows_paired.csv", index=False)
    print("coverage (probe patients / eval-set patients):",
          {"summary probe": cov6, **{k: v for k, v in cov4.items()}})
    pd.set_option("display.width", 200)
    print(rows.round(3).to_string(index=False))
    print(pairs.round(3).to_string(index=False))
    print(f"wrote {out / 'tb_comparison_rows.csv'} and {out / 'tb_comparison_rows_paired.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
