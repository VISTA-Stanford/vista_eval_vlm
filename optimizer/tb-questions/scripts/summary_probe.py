"""The summary-embedding classifier row for the board questions agent (spec planning/vista_eval_vlm/2026-11-22-tb-
board-questions-agent.md, Comparators): a linear classifier on the benchmark's Gemini embedding of the tumor board-time
summary, trained on TRAIN first tumor boards with the new answer key, scored on VAL first tumor boards with this task's
own scorer (run.score_pass: per-question mean one-vs-rest AUROC, balanced accuracy, patient bootstrap intervals).

The embedding (vistabench embeddings v1_7 text_llm_summary/tb_classification gemini_emb_001, the same file the
tb comparison rows' probe used) is one per patient, of the summary cut 7 days before the patient's first tumor board
in the old table (tb_classification_tasks). A board is used only when it is the patient's first confirmed tumor board
and its date is within MATCH_DAYS of that old first-board date, so the summary was cut at about the same day as the
agent's record. Per question: standardised features, multinomial logistic regression, C picked by 5-fold
cross-validation on TRAIN (log loss), fit on the options with at least MIN_PER_OPTION TRAIN boards; rarer options get
probability 0 (a single-board option would otherwise leave no stratified folds). Counts only on screen.

    cd optimizer/tb-questions && TB_AGENT=boards-record uv run -q --project $AGENTIC_LABEL_OPT --with scikit-learn \\
        --with pandas --with pyarrow --with google-cloud-bigquery --with db-dtypes python -B scripts/summary_probe.py
"""

from __future__ import annotations

import json
import socket
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import adapters as ad  # noqa: E402
import run  # noqa: E402

EMB = Path("/mnt/su-vista-uscentral1/vistabench/embeddings/v1_7/text_llm_summary/tb_classification/gemini_emb_001_embeddings.parquet")
OLD = "som-nero-plevriti-deidbdf.vista_bench_v1_7.tb_classification_tasks"
MIN_PER_OPTION = 2
MATCH_DAYS = 3  # DECIDE: build 2026-11-22 (the finder's recall check used the same 3-day match)
C_GRID = (1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0)  # the first run picked 0.001, the old edge, everywhere


def main() -> int:
    import pandas as pd
    from google.cloud import bigquery
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not config.BOARDS:
        raise SystemExit("set TB_AGENT=boards-record")
    id_map = {m["id"]: m for m in map(json.loads, (config.ID_MAP_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    strata = ad.read_strata()
    old = bigquery.Client(project="som-nero-plevriti-deidbdf").query(
        f"select distinct person_id, date(first_tb_encounter_ts) d from `{OLD}`").to_dataframe()
    if old["person_id"].duplicated().any():
        raise SystemExit("STOP: two old first-board dates for one patient")
    old_date = dict(zip(old["person_id"].astype(int), old["d"]))
    emb = pd.read_parquet(EMB)
    X_of = {int(p): np.asarray(e, dtype=np.float32) for p, e in zip(emb["person_id"], emb["embedding"])}

    def usable(i):
        m = id_map[i]
        p = int(m["person_id"])
        return (strata[i]["first_board"] and p in X_of and p in old_date
                and abs((date.fromisoformat(m["board_date"]) - old_date[p]).days) <= MATCH_DAYS)

    split_rows = {s: [r for r in ad.read_split(s) if usable(r["id"])] for s in ("train", "val")}
    first = {s: sum(strata[r["id"]]["first_board"] for r in ad.read_split(s)) for s in ("train", "val")}
    print(f"first tumor boards with a matching summary embedding: TRAIN {len(split_rows['train'])} of {first['train']}, "
          f"VAL {len(split_rows['val'])} of {first['val']}")
    answers = {r["id"]: {q.key: None for q in config.QUESTIONS} | {"error": None} for r in split_rows["val"]}
    fits = {}
    for q in config.QUESTIONS:
        tr_all = [r for r in split_rows["train"] if r.get(q.key) is not None]
        n_of = {c: sum(r[q.key] == c for r in tr_all) for c in {r[q.key] for r in tr_all}}
        tr = [r for r in tr_all if n_of[r[q.key]] >= MIN_PER_OPTION]
        va = [r for r in split_rows["val"] if r.get(q.key) is not None]
        if not va:
            continue
        y = np.array([r[q.key] for r in tr])
        X = np.stack([X_of[int(id_map[r["id"]]["person_id"])] for r in tr])
        classes, counts = np.unique(y, return_counts=True)
        folds = int(min(5, counts.min())) if len(classes) > 1 else 0
        codes = [c for _k, c, _t in q.options]
        Xv = np.stack([X_of[int(id_map[r["id"]]["person_id"])] for r in va])
        if folds >= 2:
            # Options with fewer than `folds` TRAIN boards are kept; CV folds are stratified on what exists.
            gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                              {"logisticregression__C": list(C_GRID)}, scoring="neg_log_loss",
                              cv=StratifiedKFold(folds, shuffle=True, random_state=0))
            gs.fit(X, y)
            P = gs.predict_proba(Xv)
            cls = list(gs.best_estimator_.classes_)
            fits[q.key] = {"train": len(tr), "train_left_out_rare": len(tr_all) - len(tr), "val": len(va),
                          "C": gs.best_params_["logisticregression__C"], "cv_folds": folds}
        else:  # one TRAIN class (or too few): the TRAIN shares for everyone
            cls, P = list(classes), np.tile(counts / counts.sum(), (len(va), 1))
            fits[q.key] = {"train": len(tr), "val": len(va), "C": None, "cv_folds": folds}
        for r, p in zip(va, P):
            answers[r["id"]][q.key] = {str(c): (float(p[cls.index(c)]) if c in cls else 0.0) for c in codes}
    res = run.score_pass(answers, "val")
    res.update({"row": "summary-embedding classifier (gemini_emb_001, logistic regression), first tumor boards",
                "match_days": MATCH_DAYS, "fits": fits, "embedding": str(EMB)})
    out = run.RESULTS_ROOT / socket.gethostname() / config.AGENT / "summary-probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_probe_val.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"climb_score": res["climb_score"], "boards": res["boards"], "fits": fits,
                      "questions": {k: {m: v.get(m) for m in ("n", "auroc", "auroc_ci", "auroc_a_to_e",
                                                               "top_pick_balanced_accuracy", "always_most_common")}
                                    for k, v in res["questions"].items()}}, indent=1))
    print(f"written {out / 'summary_probe_val.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
