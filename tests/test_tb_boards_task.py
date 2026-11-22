"""tb-questions boards-record mode (spec planning/vista_eval_vlm/2026-11-22-tb-board-questions-agent.md): the board
option tables and boards_select on synthetic rows. No engine, no network, no patient data."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TASK = Path(__file__).resolve().parents[1] / "optimizer" / "tb-questions"


@pytest.fixture(scope="module")
def bd():
    os.environ["TB_AGENT"] = "boards-record"
    sys.path.insert(0, str(TASK))
    spec = importlib.util.spec_from_file_location("tb_build_data", TASK / "scripts" / "build_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_option_tables_cover_every_label_v1_value_once():
    sys.path.insert(0, str(TASK))
    import board_options as bo

    for key, opts in bo.BOARD_OPTION_TABLES.items():
        letters = [k for k, _c, _t, _v in opts]
        assert letters == list("ABCDEFGHI"[: len(opts)]), key
        assert [c for _k, c, _t, _v in opts] == list(range(len(opts))), key
        assert len({v for *_x, v in opts}) == len(opts), key
    # T3 / T4: the old cards' A-E word for word, then F, other, more workup first
    for key, old in (("early_stage_plan", bo.EARLY_STAGE_OPTIONS), ("first_line_regimen", bo.SYSTEMIC_OPTIONS)):
        assert [o[:3] for o in bo.BOARD_OPTION_TABLES[key][:5]] == list(old)
        assert bo.BOARD_ANSWER_CODES[key]["more_workup_first"] == len(bo.BOARD_OPTION_TABLES[key]) - 1
    assert bo.BOARD_ANSWER_CODES["after_surgery_plan"]["more_workup_first"] == 5
    # the vista_bench labeller's values (label-v1) all have a code
    assert set(bo.BOARD_ANSWER_CODES["decision_type"]) == {
        "more_workup", "start_new_treatment", "continue_current", "switch_or_stop", "add_local_to_systemic",
        "surveillance", "refer", "supportive_or_hospice", "deferred"}


def _rows():
    def board(pid, d, split, first, treated, extra=()):
        base = [("decision_type", "refer"), ("intent", "not_applicable"), *extra]
        cut = (pd.Timestamp(d) - pd.Timedelta(days=7)).date()
        return [{"person_id": pid, "encounter_index": 0, "board_date": pd.Timestamp(d).date(), "cutoff": cut,
                 "split": split, "question": q, "answer": a, "first_board": first, "already_treated": treated}
                for q, a in base]
    return pd.DataFrame(board(1, "2020-01-08", "train", True, False, [("early_stage_plan", "more_workup_first")])
                        + board(1, "2020-05-08", "train", False, True)
                        + board(2, "2021-01-08", "val", True, False, [("first_line_regimen", "B")])
                        + board(3, "2021-02-08", "test", True, True))


def test_boards_select_writes_codes_seals_test_and_splits_by_patient(bd, tmp_path):
    src = tmp_path / "rows.parquet"
    _rows().to_parquet(src, index=False)
    data, priv = tmp_path / "data", tmp_path / "private"
    assert bd.boards_select(src, data_dir=data, private_dir=priv, expected={"train": 2, "val": 1, "test": 1}) == 0
    train = [json.loads(x) for x in (data / "splits" / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (data / "splits" / "val.jsonl").read_text().splitlines()]
    assert not (data / "splits" / "test.jsonl").exists()  # sealed until Phil's go
    assert len([json.loads(x) for x in (priv / "test_split.jsonl").read_text().splitlines()]) == 1
    assert {r["early_stage_plan"] for r in train} == {7, None}  # 'more workup first' is H = 7
    assert val[0]["first_line_regimen"] == 1 and val[0]["decision_type"] == 6 and val[0]["intent"] == 4
    assert val[0]["early_stage_plan"] is None and val[0]["after_surgery_plan"] is None
    strata = {r["id"]: r for r in map(json.loads, (data / "strata.jsonl").read_text().splitlines())}
    assert len({strata[r["id"]]["patient"] for r in train}) == 1  # one patient, two boards
    assert sorted(strata[r["id"]]["first_board"] for r in train) == [False, True]
    assert "person_id" not in json.dumps(train + val) and "person_id" not in (data / "strata.jsonl").read_text()
    pools = json.loads((data / "pools.json").read_text())
    assert set(pools["train"]) == {r["id"] for r in train} and "test" not in pools


def test_boards_select_refuses_wrong_counts_and_unknown_answers(bd, tmp_path):
    src = tmp_path / "rows.parquet"
    _rows().to_parquet(src, index=False)
    with pytest.raises(SystemExit, match="differ"):
        bd.boards_select(src, data_dir=tmp_path / "d", private_dir=tmp_path / "p")
    bad = _rows()
    bad.loc[0, "answer"] = "not_an_option"
    bad.to_parquet(src, index=False)
    with pytest.raises(SystemExit, match="no option"):
        bd.boards_select(src, data_dir=tmp_path / "d", private_dir=tmp_path / "p", expected={"train": 2, "val": 1, "test": 1})


def test_patient_split_leak_is_refused(bd, tmp_path):
    r = _rows()
    r.loc[r["person_id"] == 2, "person_id"] = 1  # the VAL board's patient is also a TRAIN patient
    src = tmp_path / "rows.parquet"
    r.to_parquet(src, index=False)
    with pytest.raises(SystemExit, match="two splits"):
        bd.boards_select(src, data_dir=tmp_path / "d", private_dir=tmp_path / "p", expected={"train": 2, "val": 1, "test": 1})
