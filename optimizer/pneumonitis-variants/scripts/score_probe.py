"""The MerlinOnc probe on the same VAL scans the reader reads (spec, Done when: "MerlinOnc probe on the same scans
beside it"). Ported from the 2026-11-24 scratch probes summarised in chaudhari_lab/phil/pneumonitis-subtype-probe/:
MerlinOnc pan-cancer ep99 embeddings, one-vs-rest logistic regression per variant, StandardScaler, class_weight
balanced, C chosen by 5-fold GroupKFold on patient over TRAIN, scored on VAL with a patient bootstrap (2,000). Chance
on these sizes swings (sd 0.06-0.11), so each variant also gets a 30-shuffle null (TRAIN labels permuted, the chosen
C, refit, scored on VAL): judge a number against that distribution, not against 0.5.

TRAIN = every eligible TRAIN scan with an embedding (probe/train.jsonl, 599: no volume needed); VAL = exactly the
reader's VAL scans (probe/val.jsonl). Here infection is the fifth variant, like the reader's. Writes aggregates to
RUNS_DIR/probe.json and per-scan probe probabilities to PRIVATE_DIR/probe_val_predictions.json (for the comparison
table). Prints counts and AUROC only.

    cd optimizer/pneumonitis-variants
    uv run -q --no-project --with numpy --with scikit-learn python -B scripts/score_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

VARIANTS = ("radiation", "immunotherapy", "other_drug", "cop", "infection")
CS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_BOOT, N_NULL = 2000, 30


def load(split: str):
    import numpy as np

    rows = [json.loads(x) for x in (paths.DATA_DIR / "probe" / f"{split}.jsonl").read_text().splitlines() if x]
    with np.load(paths.DATA_DIR / "embeddings.npz") as z:
        idx = {str(i): n for n, i in enumerate(z["ids"])}
        emb = z["emb"]
    X = np.stack([emb[idx[r["id"]]] for r in rows]).astype("float32")
    return rows, X, np.array([r["group"] for r in rows]), np.array([r["variant"] for r in rows])


def fit(X, y, groups, C=None):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    best = (C, None)
    if C is None:
        best = None
        for c in CS:
            s = []
            for a, b in GroupKFold(5).split(X, y, groups):
                if len(set(y[b])) < 2 or len(set(y[a])) < 2:
                    continue
                sc = StandardScaler().fit(X[a])
                m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(sc.transform(X[a]), y[a])
                s.append(roc_auc_score(y[b], m.predict_proba(sc.transform(X[b]))[:, 1]))
            if best is None or np.mean(s) > best[1]:
                best = (c, float(np.mean(s)))
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=best[0], max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
    return (lambda Z: m.predict_proba(sc.transform(Z))[:, 1]), best


def boot(y, p, groups, rng, n=N_BOOT):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    pats = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in pats}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(pats, len(pats))])
        if len(set(y[s])) == 2:
            out.append(roc_auc_score(y[s], p[s]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main() -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    tr_rows, Xtr, gtr, vtr = load("train")
    va_rows, Xva, gva, vva = load("val")
    if set(gtr) & set(gva):
        raise SystemExit("a patient is in both the probe's TRAIN and VAL rows")
    rng = np.random.default_rng(0)
    res = {"embedding": "merlinonc_top_centered_pancancer_postdx_ep99", "train_scans": len(tr_rows),
           "val_scans": len(va_rows), "val_patients": int(len(set(gva))), "variants": {}}
    preds = {r["id"]: {} for r in va_rows}
    for v in VARIANTS:
        ytr, yva = (vtr == v).astype(int), (vva == v).astype(int)
        predict, (C, cv) = fit(Xtr, ytr, gtr)
        p = predict(Xva)
        for r, q in zip(va_rows, p):
            preds[r["id"]][v] = round(float(q), 5)
        nulls = []
        for j in range(N_NULL):
            perm = np.random.default_rng(1000 + j).permutation(ytr)
            pn, _ = fit(Xtr, perm, gtr, C=C)
            nulls.append(roc_auc_score(yva, pn(Xva)))
        auc = roc_auc_score(yva, p)
        res["variants"][v] = {"train_pos": int(ytr.sum()), "val_pos": int(yva.sum()), "val_neg": int((1 - yva).sum()),
                              "C": C, "train_cv_auroc": round(cv, 3), "val_auroc": round(float(auc), 3),
                              "ci95": boot(yva, p, gva, rng),
                              "null30": {"mean": round(float(np.mean(nulls)), 3), "sd": round(float(np.std(nulls)), 3),
                                         "p95": round(float(np.percentile(nulls, 95)), 3)}}
        print(v, json.dumps(res["variants"][v]), flush=True)
    res["mean_val_auroc"] = round(float(np.mean([res["variants"][v]["val_auroc"] for v in VARIANTS])), 4)
    paths.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RUNS_DIR / "probe.json").write_text(json.dumps(res, indent=1))
    (paths.PRIVATE_DIR / "probe_val_predictions.json").write_text(json.dumps(preds))
    print(f"mean VAL AUROC {res['mean_val_auroc']}; written {paths.RUNS_DIR / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
