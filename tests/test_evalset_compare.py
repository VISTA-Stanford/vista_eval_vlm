"""evalset.compare: AUROC, the missing-answer rule and the paired difference, checked against scikit-learn on made-up
data, and the step-3 test1000 list name (spec planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md)."""

import numpy as np
import pandas as pd
import pytest

from evalset import compare, samples


def _cases(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "person_id": rng.integers(0, n // 2, n).astype(str),
                         "label__radiology_progression": rng.integers(0, 2, n),
                         "prior_source": rng.choice(["report_named", "previous_no_usable_date"], n),
                         "prior_ambiguous": rng.integers(0, 2, n).astype(bool)})


def test_table_matches_sklearn_and_fills_missing_with_half():
    from sklearn.metrics import roc_auc_score
    cases = _cases()
    y = cases["label__radiology_progression"].to_numpy()
    rng = np.random.default_rng(2)
    ref = pd.Series(np.round(y * 0.3 + rng.random(len(y)), 1), index=cases["case_id"])  # tie-heavy
    arm = pd.Series(rng.random(len(y)), index=cases["case_id"])
    arm.iloc[:40] = np.nan
    t = compare.table({"Pillar0": ref, "Arm": arm}, cases, n_boot=200).set_index("row")
    assert t.loc["Pillar0", "auroc"] == pytest.approx(roc_auc_score(y, ref))
    filled = arm.fillna(0.5)
    assert t.loc["Arm", "auroc"] == pytest.approx(roc_auc_score(y, filled))
    ok = arm.notna().to_numpy()
    assert t.loc["Arm", "auroc_readable_only"] == pytest.approx(roc_auc_score(y[ok], arm[ok]))
    assert t.loc["Arm", "unreadable_or_missing"] == 40 and t.loc["Arm", "answered"] == len(y) - 40
    assert t.loc["Arm", "diff_vs_pillar0"] == pytest.approx(roc_auc_score(y, filled) - roc_auc_score(y, ref))
    assert t.loc["Arm", "diff_lo"] <= t.loc["Arm", "diff_vs_pillar0"] <= t.loc["Arm", "diff_hi"]
    assert t.loc["Pillar0", "auroc_lo"] <= t.loc["Pillar0", "auroc"] <= t.loc["Pillar0", "auroc_hi"]


def test_breakout_groups_cover_every_case():
    cases = _cases()
    s = pd.Series(np.random.default_rng(3).random(len(cases)), index=cases["case_id"])
    b = compare.breakout({"Pillar0": s}, cases)
    for col in ("prior_source", "prior_ambiguous"):
        assert b[b["by"] == col]["n"].sum() == len(cases)


def test_test1000_list_name():
    assert samples._name(3, "test1000") == "step3_test1000"
    with pytest.raises(ValueError):
        samples._name(3, "nope")
