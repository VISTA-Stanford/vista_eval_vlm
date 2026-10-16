"""Score split by board-report agreement (scripts/tb_scores_by_crosscheck.py). Synthetic rows only."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "tb_scores_by_crosscheck", Path(__file__).resolve().parents[1] / "scripts" / "tb_scores_by_crosscheck.py")
sx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sx)


def test_strata_maps_tags():
    xc = pd.DataFrame({"person_id": [1, 2, 3, 4], "task": [sx.TASK] * 4,
                       "board_report_agreement": ["agree", "disagree", None, "board_indeterminate"],
                       "report_source": ["ct_call", "petct_call", "no_report", "ct_call"]})
    st = sx.strata(xc).set_index("person_id")
    assert list(st["agreement"]) == ["agree", "disagree", "no_call", "no_call"]
    assert list(st["source"]) == ["ct_report", "petct_report", "none", "ct_report"]


def test_score_strata_counts_and_scores():
    ans = pd.DataFrame({"label": [1, 1, 0, 0, 1, 0], "pred": [1, 1, 0, 0, 0, 1],
                        "agreement": ["agree"] * 4 + ["disagree"] * 2, "source": ["ct_report"] * 6})
    rows = {(r["stratum_kind"], r["stratum"]): r for r in sx.score_strata(ans, [0, 1], n_boot=50)}
    assert rows[("agreement", "agree")]["n"] == 4 and rows[("agreement", "agree")]["balanced_accuracy"] == 1.0
    assert rows[("agreement", "disagree")]["balanced_accuracy"] == 0.0
    assert rows[("agreement_x_source", "agree|ct_report")]["n"] == 4


def _row(arm, kind, stratum, n):
    return {"task": sx.TASK, "model": "m", "arm": arm, "arm_pooled_n": 0, "stratum_kind": kind,
            "stratum": stratum, "n": n, "balanced_accuracy": 0.5}


def test_suppress_hides_small_groups_and_their_complements():
    rows = [_row("a", "agreement", "agree", 17), _row("a", "agreement", "disagree", 6),
            _row("a", "agreement", "mixed", 25), _row("a", "agreement", "no_call", 48),
            _row("a", "agreement_x_source", "agree|ct_report", 14), _row("a", "agreement_x_source", "agree|petct_report", 3),
            _row("a", "agreement_x_source", "disagree|ct_report", 4), _row("a", "agreement_x_source", "disagree|petct_report", 2)]
    out = {r["stratum"]: r for r in sx.suppress(rows)}
    assert out["disagree"]["n"] == "<11" and out["disagree"]["balanced_accuracy"] is None
    # one hidden among the arm's agreement groups (whose total is printed) -> the next smallest hides too
    assert out["agree"]["n"] == "hidden" and out["mixed"]["n"] == 25 and out["no_call"]["n"] == 48
    # agree's parts: 3 is small; 14 would give it back from agree's total, but agree is hidden now -> still one must hide
    assert out["agree|petct_report"]["n"] == "<11"
    assert all(isinstance(out[k]["n"], str) for k in ("disagree|ct_report", "disagree|petct_report"))
