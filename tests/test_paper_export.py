"""Paper-code stage 2: the answer export to the paper's one-table-per-run layout."""

import json

import numpy as np
import pandas as pd
import pytest

from evalset import paper_export as PE
from evalset.score import score_rows


def _answers(tmp_path, prob=True, step=1):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(40):
        for q, task in (
            ("death_1yr", "died_any_cause_1_yr"),
            ("death_2yr", "died_any_cause_2_yr"),
        ):
            lab = int(rng.integers(0, 2))
            p = float(np.clip(0.3 + 0.4 * lab + rng.normal(0, 0.2), 0, 1))
            rows.append(
                {
                    "case_id": str(100 + i),
                    "person_id": str(100 + i),
                    "split": "test" if i < 30 else "val",
                    "step": step,
                    "question_id": q,
                    "task": task,
                    "arm": "summary",
                    "recipe": "opus__sonnet_prob",
                    "model": "m",
                    "label": lab,
                    "response_text": "x",
                    "answer": round(p, 2) if prob else ("yes" if p > 0.5 else "no"),
                    "refused": False,
                    "error": None,
                    "p_yes": round(p, 2) if prob else None,
                }
            )
    d = tmp_path / "runs" / "prob-v1" / "opus__sonnet_prob"
    d.mkdir(parents=True)
    f = d / f"step{step}_summary.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    return f, tmp_path / "runs"


def test_probability_run_export_and_rescore_equal_the_original(tmp_path):
    f, root = _answers(tmp_path)
    d = PE.export_evalset_file(f, runs_root=root)
    PE.check_layout(d)
    assert d.run_id.iloc[0] == "evalset_v1/prob-v1/opus__sonnet_prob/step1_summary"
    assert (d.split == "test").all() and d.person_id.nunique() == 30
    assert d.horizon_years.tolist()[:2] == [1, 2]
    assert d.label_table.iloc[0].endswith(
        "progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr"
    )
    # independent: score the original file with the eval set's own scorer and the export rebuilt into its columns
    from evalset.score import prepare

    orig = prepare(pd.read_csv(f, dtype={"case_id": str, "person_id": str}))
    orig = orig[orig.split == "test"]
    a = score_rows(orig, n_boot=50)
    re_ = d.assign(
        recipe=d.program_version,
        arm="summary",
        score=d.probability,
        pred=None,
        yesno=False,
    )
    b = score_rows(
        re_[
            [
                "step",
                "recipe",
                "arm",
                "question_id",
                "person_id",
                "label",
                "pred",
                "refused",
                "score",
                "yesno",
            ]
        ],
        n_boot=50,
    )
    assert a.set_index("question_id").auroc.to_dict() == pytest.approx(
        b.set_index("question_id").auroc.to_dict()
    )


def test_yes_no_run_fills_answer_not_probability(tmp_path):
    f, root = _answers(tmp_path, prob=False)
    d = PE.export_evalset_file(f, runs_root=root)
    assert d.probability.isna().all() and set(d.answer) <= {"yes", "no"}
    PE.check_layout(d)


def test_check_layout_refuses_bad_tables(tmp_path):
    f, root = _answers(tmp_path)
    d = PE.export_evalset_file(f, runs_root=root)
    with pytest.raises(ValueError, match="duplicate"):
        PE.check_layout(pd.concat([d, d.iloc[:1]]))
    with pytest.raises(ValueError, match="outside 0-1"):
        PE.check_layout(d.assign(probability=d.probability * 100))
    with pytest.raises(ValueError, match="missing"):
        PE.check_layout(d.drop(columns="label_table"))
    with pytest.raises(ValueError, match="one run"):
        PE.check_layout(d.assign(run_id=["a", "b"] * (len(d) // 2)))


def test_winner_tag_is_best_measured_version_seed_included(tmp_path):
    p = tmp_path / "run_summary.json"
    rs = lambda regr, iters: p.write_text(
        json.dumps({"regrades": regr, "iters": iters})
    )
    rs(
        [{"tag": "program-v0", "metric": 0.81}],
        [{"tag": "program-v1", "auroc": 0.80}, {"tag": "program-v2", "auroc": 0.83}],
    )
    assert PE.winner_tag(p) == ("program-v2", 0.83)
    rs(
        [{"tag": "program-v0", "metric": 0.816}],
        [{"tag": "program-v1", "auroc": 0.797}, {"tag": "program-v2", "auroc": 0.803}],
    )
    assert PE.winner_tag(p) == (
        "program-v0",
        0.816,
    )  # the seed wins when no round beats it
    rs(
        [],
        [{"tag": "program-v8", "auroc": 0.831}, {"tag": "program-v10", "auroc": 0.831}],
    )
    assert PE.winner_tag(p)[0] == "program-v8"  # earliest on a tie
    rs([], [{"tag": "program-v1"}])
    with pytest.raises(ValueError):
        PE.winner_tag(p)


def test_write_run_layout_and_provenance(tmp_path):
    f, root = _answers(tmp_path)
    d = PE.export_evalset_file(f, runs_root=root)
    out = PE.write_run(d, out_root=tmp_path / "out")
    back = pd.read_parquet(out)
    assert len(back) == len(d) and out.parent.name == "step1_summary"
    prov = json.loads((out.parent / "answers.provenance.json").read_text())
    assert (
        prov["inputs"][0]["path"] == str(f) and len(prov["inputs"][0]["sha256"]) == 64
    )
    assert prov["program_version"] == "opus__sonnet_prob" and prov["patients"] == 30


def test_ehr_agent_rows_failed_answers_score_half(tmp_path):
    (tmp_path / "id.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "person_id": 100 + i, "split": "test"})
            for i in range(3)
        )
    )
    (tmp_path / "test.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "label_1yr": i % 2, "label": 1})
            for i in range(3)
        )
    )
    ans = {
        "c0": {"died_within_1_year": 0.2, "died_within_2_years": 0.6},
        "c1": {
            "died_within_1_year": True,
            "died_within_2_years": 0.9,
        },  # a yes/no-typed answer -> 0.5
        "c2": {"error": "timeout"},
    }
    (tmp_path / "a.json").write_text(json.dumps({"answers": ans}))
    d = PE.ehr_agent_rows(
        tmp_path / "a.json", tmp_path / "id.jsonl", tmp_path / "test.jsonl"
    ).set_index(["person_id", "task"])
    assert d.loc[("100", "died_any_cause_1_yr"), "probability"] == 0.2
    assert d.loc[
        ("101", "died_any_cause_1_yr"), ["probability", "failed"]
    ].tolist() == [0.5, True]
    assert d.loc[("101", "died_any_cause_2_yr"), "probability"] == 0.9
    assert d.loc[
        ("102", "died_any_cause_2_yr"), ["probability", "failed"]
    ].tolist() == [0.5, True]
    assert (
        d.loc[("101", "died_any_cause_1_yr"), "label"] == 1
        and d.loc[("100", "died_any_cause_2_yr"), "label"] == 1
    )


def test_over_time_rows_join_points_and_labels(tmp_path):
    pd.DataFrame(
        {
            "row": ["r1", "r2", "r3"],
            "person_id": [1, 1, 2],
            "point": ["start", "m6", "start"],
            "split": ["test", "test", "val"],
        }
    ).to_parquet(tmp_path / "points.parquet")
    (tmp_path / "labels.jsonl").write_text(
        "\n".join(
            json.dumps({"row": r, "label_1yr": 0, "label_2yr": 1})
            for r in ("r1", "r2", "r3")
        )
    )
    a = tmp_path / "answers"
    a.mkdir()
    (a / "start-fresh.json").write_text(
        json.dumps(
            {
                "answers": {
                    "r1": {"died_within_1_year": 0.1, "died_within_2_years": 0.3},
                    "r3": {"died_within_1_year": 0.9, "died_within_2_years": 0.9},
                }
            }
        )
    )
    (a / "m6-fresh.json").write_text(
        json.dumps(
            {"answers": {"r2": {"died_within_1_year": 0.4, "died_within_2_years": 0.5}}}
        )
    )
    d = PE.over_time_rows(tmp_path, a)
    assert sorted(d.case_id.unique()) == [
        "r1",
        "r2",
    ]  # the val row is not exported with the test split
    assert d[(d.case_id == "r2") & (d.task == "died_any_cause_2_yr")][
        ["probability", "label", "point"]
    ].values.tolist() == [[0.5, 1, "m6"]]


def test_ct_rows_file_refuses_rows_without_scan(tmp_path):
    f = tmp_path / "rows.csv"
    pd.DataFrame(
        {
            "person_id": [1, 1],
            "question_id": ["death_1yr", "death_2yr"],
            "label": [0, 1],
            "pred": [0, 1],
            "refused": [False, False],
            "prob": [0.2, 0.7],
        }
    ).to_csv(f, index=False)
    assert PE.ct_rows_file(f).task.tolist() == [
        "died_any_cause_1_yr",
        "died_any_cause_2_yr",
    ]
    pd.DataFrame(
        {
            "person_id": [1, 1],
            "question_id": ["lesion_liver"] * 2,
            "label": [0, 1],
            "pred": [0, 1],
            "refused": [False, False],
            "prob": [0.2, 0.7],
        }
    ).to_csv(f, index=False)
    with pytest.raises(ValueError, match="scan number"):
        PE.ct_rows_file(f)


def test_check_climb_version_refuses_wrong_roles(tmp_path):
    p = tmp_path / "run_summary.json"
    p.write_text(
        json.dumps(
            {
                "start": {"tag": "program-v0", "version_sha": "abc"},
                "regrades": [
                    {"tag": "program-v0", "metric": 0.70, "version_sha": "abc"}
                ],
                "iters": [
                    {"tag": "program-v1", "auroc": 0.75, "version_sha": "def"},
                    {"tag": "program-v2", "auroc": 0.72, "version_sha": "ghi"},
                ],
            }
        )
    )
    assert "winner" in PE.check_climb_version(p, "program-v1", "winner")
    assert "hash def" in PE.check_climb_version(p, "program-v1", "winner")
    assert "starting version" in PE.check_climb_version(p, "program-v0", "seed")
    assert "not its winner" in PE.check_climb_version(p, "program-v2", "other")
    with pytest.raises(ValueError, match="winner is program-v1"):
        PE.check_climb_version(p, "program-v2", "winner")
    with pytest.raises(ValueError, match="started from program-v0"):
        PE.check_climb_version(p, "program-v1", "seed")
    with pytest.raises(ValueError, match="never measured"):
        PE.check_climb_version(p, "program-v9", "other")
