"""Radiology progression as a VLM question (plan: planning/vista_eval_vlm/vlm-radiology-progression-codex.md).

Shared-registry facts, per-scan loader, image-only arms, whole-volume slices, chest windows,
the Codex adapter (through a fake ``codex`` program), and the VLM-vs-probe comparison.
No network, no patient data: every input is synthetic.
"""

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

import vista_run.run_bq as run_bq_mod
from context.adapters.ct import CTAdapter
from context.selectors.ct_selectors import evenly_spaced_k, up_to_k
from context.windowing import CHEST_WINDOWS, chest_rgb
from models.codex_exec import CodexExecAdapter, build_command, hidden_paths, parse_events
from results.final_metrics import compare_to_probe, score_tb
from tasks.registry import load_valid_tasks, resolve_shared
from vista_run.run_bq import PRIOR_COLUMNS, TaskOrchestrator, per_class_sample
from vqa_dataset import IMAGES_LINE_TOKEN, PromptDataset

CHOICES = ["Stable", "Progression", "Response", "Mixed Response"]
MANIFEST = {"tasks": {"radiology_progression": {
    "task_group": "radiology_progression", "question": "q", "is_binary": False, "scope": "cohort",
    "grain": "per_ct_landmark", "inputs": "current_and_prior",
    "label_mapping": {"0": "Stable", "1": "Progression", "2": "Response", "3": "Mixed Response",
                      "-1": "Indeterminate / Does not apply / No Prior"}}}}
PRIOR = {"table": "t", "key": "ct_accession_number", "row_key": "ct_accession_number", "arms": ["ct_pair"]}
ENTRY = {"task_name": "radiology_progression", "shared_task": "radiology_progression",
         "task_source_csv": "radiology_progression_tasks", "loader": "bq_table",
         "label_from_own_report": True, "prior_ct": PRIOR}


# --- shared registry -----------------------------------------------------------------------
def test_shared_task_fills_class_names_options_and_row_key():
    t = resolve_shared(ENTRY, MANIFEST)
    assert t["choices"] == CHOICES
    assert t["unscored_labels"] == [-1] and t["mapping"]["3"] == "Mixed Response"
    assert t["row_key"] == "ct_accession_number" and t["inputs"] == "current_and_prior"


def test_real_vendored_registry_has_this_task():
    # The other side of the contract: vista-eval's own vendored snapshot, not a fixture.
    t = resolve_shared(ENTRY)
    assert t["choices"] == CHOICES and t["grain"] == "per_ct_landmark"


def test_hand_written_shared_fields_are_refused():
    with pytest.raises(ValueError, match="come from shared task"):
        resolve_shared({**ENTRY, "choices": ["Stable", "Worse"]}, MANIFEST)


def test_current_and_prior_task_needs_a_prior_link():
    with pytest.raises(ValueError, match="prior_ct"):
        resolve_shared({k: v for k, v in ENTRY.items() if k != "prior_ct"}, MANIFEST)


def test_entries_without_shared_task_pass_through(tmp_path):
    board = {"task_name": "b", "mapping": {"1": "Yes", "0": "No"}, "choices": ["Yes", "No"]}
    (tmp_path / "r.json").write_text(json.dumps([board]))
    assert load_valid_tasks(tmp_path / "r.json", MANIFEST) == [board]


# --- slices and windows --------------------------------------------------------------------
def test_up_to_k_keeps_every_slice_of_a_short_scan_and_spaces_a_long_one():
    assert up_to_k(180, 250) == list(range(180))
    long = up_to_k(669, 250)
    assert len(long) == 250 == len(set(long)) and long[0] == 0 and long[-1] == 668
    assert long == evenly_spaced_k(669, 250)


def test_chest_windows_hand_worked():
    # -600 HU: lung (−1350..150) -> 750/1500*255 = 127.5; soft tissue (−160..240) clips to 0; bone 0.
    # 40 HU: lung 1390/1500*255 = 236.3; soft tissue 200/400*255 = 127.5; bone 540/1800*255 = 76.5.
    px = chest_rgb(np.array([[-600.0, 40.0]]))
    assert px[0, 0].tolist() == pytest.approx([127.5, 0.0, 0.0])
    assert px[0, 1].tolist() == pytest.approx([236.3, 127.5, 76.5], abs=0.1)
    assert CHEST_WINDOWS[0] == (-1350, 150)


def test_ct_adapter_chest_rgb_at_native_512():
    vol = np.full((512, 512, 300), 40.0)
    block = CTAdapter(config={"select": {"fn": "up_to_k", "k": 250},
                              "preprocess": {"windowing": "chest_rgb", "target_size": 512}}).contextualize(vol, {"model_type": "codex_exec"})
    assert block.metadata["count"] == 250 and block.metadata["windowing"] == "chest_rgb"
    im = block.payload[0]
    assert im.mode == "RGB" and im.size == (512, 512) and im.getpixel((256, 256)) == (236, 128, 76)


# --- sampling and the per-scan loader ------------------------------------------------------
def _scan_rows(n_per_class=(30, 20, 15, 12)):
    rows = []
    for label, n in enumerate(n_per_class):
        for i in range(n):
            rows.append({"person_id": 1000 + len(rows) // 3, "ct_accession_number": f"{label}{i:05d}",
                         "label": label, "split": "test" if i % 4 else "valid",
                         "image_study_uid": f"st{label}{i}", "image_series_uid": f"se{label}{i}",
                         "prior_image_study_uid": f"pst{label}{i}", "prior_image_series_uid": f"pse{label}{i}",
                         "prior_gap_days": 90, "prior_source": "report_named"})
    rows.append({**rows[0], "ct_accession_number": "X1", "label": -1})
    return pd.DataFrame(rows)


def test_per_class_sample_is_nested_and_capped():
    df = _scan_rows()
    small = per_class_sample(df, "ct_accession_number", 3, seed=7)
    big = per_class_sample(df, "ct_accession_number", 10, seed=7)
    assert small["label"].value_counts().to_dict() == {0: 3, 1: 3, 2: 3, 3: 3, -1: 1}
    assert big["label"].value_counts().to_dict() == {0: 10, 1: 10, 2: 10, 3: 10, -1: 1}
    assert set(small["ct_accession_number"]) <= set(big["ct_accession_number"])
    # Same draw however the input rows happen to be ordered.
    again = per_class_sample(df.sample(frac=1, random_state=1), "ct_accession_number", 3, seed=7)
    assert set(again["ct_accession_number"]) == set(small["ct_accession_number"])


def _stub_orch(monkeypatch, rows, cfg=None):
    o = TaskOrchestrator.__new__(TaskOrchestrator)
    o.cfg = {"runtime": {"max_new_tokens": 5}, **(cfg or {})}
    o.project_id, o.bq_client = "p", None
    o.model = o.processor = None
    monkeypatch.setattr(run_bq_mod, "fetch_task_data_from_bq", lambda client, table, task: rows.copy())
    return o


def test_loader_keys_on_the_scan_and_samples_the_test_split(monkeypatch):
    task = resolve_shared(ENTRY, MANIFEST)
    cfg = {"row_sample": {"per_class": 2, "seed": 0, "splits": ["test"], "require_ct": True}}
    df, _, _ = _stub_orch(monkeypatch, _scan_rows(), cfg)._load_bq_table_task_data(task)
    assert df["label"].value_counts().to_dict() == {0: 2, 1: 2, 2: 2, 3: 2}  # -1 dropped as unscored
    assert (df["split"] == "test").all()
    assert df["index"].tolist() == df["ct_accession_number"].tolist() == sorted(df["ct_accession_number"])


def test_loader_refuses_two_rows_for_one_scan(monkeypatch):
    rows = _scan_rows()
    rows = pd.concat([rows, rows.iloc[:1]])
    with pytest.raises(ValueError, match="ct_accession_number"):
        _stub_orch(monkeypatch, rows)._load_bq_table_task_data(resolve_shared(ENTRY, MANIFEST))


def _prompt_orch(monkeypatch, link):
    o = _stub_orch(monkeypatch, _scan_rows())
    o.prompts_map = {"radiology_progression::ct_pair": f"pair {IMAGES_LINE_TOKEN} Q",
                     "radiology_progression::ct_current": f"current {IMAGES_LINE_TOKEN} Q"}
    o._prior_links = {"t": link}
    return o


def _link(df):
    return df[["ct_accession_number", "prior_image_study_uid", "prior_image_series_uid", "prior_gap_days",
               "prior_source"]].assign(prior_image_study_uid=lambda d: "L" + d["prior_image_study_uid"])


def test_current_only_arm_never_carries_the_prior_even_when_the_table_does(monkeypatch):
    task = resolve_shared(ENTRY, MANIFEST)
    rows = _scan_rows().query("label >= 0")
    o = _prompt_orch(monkeypatch, _link(rows))
    cur = o._build_prompts_for_experiment(rows, task, "ct_current", "patient_string")
    assert not set(PRIOR_COLUMNS) & set(cur.columns)
    assert cur["dynamic_prompt"].iloc[0].startswith("current")
    pair = o._build_prompts_for_experiment(rows, task, "ct_pair", "patient_string")
    # The pair arm's prior comes from the link table (prefixed "L"), not the table's own columns.
    assert pair["prior_image_study_uid"].str.startswith("L").all() and len(pair) == len(rows)


def test_report_labelled_task_refuses_any_arm_with_patient_text(monkeypatch):
    task = resolve_shared(ENTRY, MANIFEST)
    o = _prompt_orch(monkeypatch, _link(_scan_rows()))
    for arm in ("no_image", "axial_all_image", "summary"):
        with pytest.raises(ValueError, match="image-only"):
            o._build_prompts_for_experiment(_scan_rows(), task, arm, "patient_string")


def test_image_arm_prompt_must_carry_the_images_line(monkeypatch):
    task = resolve_shared(ENTRY, MANIFEST)
    o = _prompt_orch(monkeypatch, _link(_scan_rows()))
    o.prompts_map = {"radiology_progression": "no line here"}
    with pytest.raises(ValueError, match="IMAGES_LINE"):
        o._build_prompts_for_experiment(_scan_rows(), task, "ct_current", "patient_string")


def test_resume_matches_accessions_read_back_as_numbers(tmp_path):
    o = TaskOrchestrator.__new__(TaskOrchestrator)
    o.results_base = tmp_path
    o.file_model_name = "m"
    task = {"task_source_csv": "s", "task_name": "t"}
    d = tmp_path / "s" / "t" / "m"
    d.mkdir(parents=True)
    pd.DataFrame({"index": ["0012345"], "model_response": ["Stable"]}).to_csv(d / "t_results_ct_pair.csv", index=False)
    _, existing = o._setup_output_and_resume(task, "ct_pair")
    # Read as a number, "0012345" would come back as 12345 and the scan would be asked again.
    assert existing == {"0012345"}
    item = {"raw_row": pd.Series({"index": "0012345"}), "load_error": None}
    assert o._prepare_batch_for_inference([item], existing, None) is None  # nothing left to send


def test_overflow_counts_the_arm_cap_per_scan(tmp_path):
    o = TaskOrchestrator.__new__(TaskOrchestrator)
    o.adapter = CodexExecAdapter("gpt-test")
    o.cfg = {"runtime": {"max_new_tokens": 5}, "ct_arms": {"select": {"fn": "up_to_k", "k": 250}}}
    df = pd.DataFrame({"index": ["a", "b"], "dynamic_prompt": ["abcd", "abcd"], "prior_image_study_uid": ["s", None]})
    kept = o._split_overflow(df, {"task_name": "t"}, "ct_pair", tmp_path / "t_results_ct_pair.csv")
    assert kept["prompt_token_count"].tolist() == [1 + 500 * 330, 1 + 250 * 330]


# --- the Codex adapter, through a fake `codex` program ---------------------------------------
FAKE_CODEX = r'''
import json, os, sys
from PIL import Image
args = sys.argv[1:]
out = args[args.index("-o") + 1]
images = args[args.index("-i") + 1].split(",") if "-i" in args else []
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_CODEX_MODE", "order")
log = os.environ.get("FAKE_CODEX_LOG")
if log:
    open(log, "a").write(json.dumps({"n_images": len(images), "args": args}) + "\n")
def ev(d): print(json.dumps(d))
if mode == "limit":
    ev({"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}); sys.exit(1)
if mode == "credits":
    ev({"type": "turn.failed", "error": {"message": "Your workspace is out of credits. Ask your workspace owner to refill in order to continue."}}); sys.exit(1)
if mode == "tool":
    ev({"type": "item.completed", "item": {"type": "command_execution", "command": "ls /mnt"}})
# The image's red value is its tag: 100 + scan (0 prior, 1 current), green = slice number % 256.
tags = [Image.open(p).getpixel((0, 0)) for p in images]
answer = "Stable"
if mode == "order":
    import re
    m = re.search(r"first (\d+) images are axial slices of the PRIOR CT.*?the next (\d+)", prompt)
    n_prior, n_cur = int(m.group(1)), int(m.group(2))
    ok = (len(tags) == n_prior + n_cur and all(t[0] == 100 for t in tags[:n_prior])
          and all(t[0] == 101 for t in tags[n_prior:]))
    answer = "Progression" if ok else "Response"
if mode == "offlist":
    answer = "Worse"
ev({"type": "error", "message": "stream disconnected; retrying"})
ev({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps({"answer": answer})}})
ev({"type": "turn.completed", "usage": {"input_tokens": 1000 + 330 * len(images), "cached_input_tokens": 0,
                                         "output_tokens": 20, "reasoning_output_tokens": 5}})
open(out, "w").write(json.dumps({"answer": answer}))
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    path = tmp_path / "codex"
    path.write_text(f"#!{sys.executable}\n" + FAKE_CODEX)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "calls.jsonl"))
    a = CodexExecAdapter("gpt-test", cache_dir=str(tmp_path / "cache"))
    a.configure({"codex_bin": str(path), "seal": False, "retries": 1})
    a.load()
    return a


def _tagged_ds(rows, swap=False):
    ds = PromptDataset(pd.DataFrame(rows), experiment="ct_pair", model_type="codex_exec", fail_on_ct_error=True,
                       ct_select={"fn": "up_to_k", "k": 250}, ct_preprocess={"windowing": "chest_rgb"})

    def load(study, series, row, idx):
        scan = 0 if str(study).startswith("pri") else 1
        if swap:
            scan = 1 - scan
        n = 7 if scan == 0 else 5
        return [Image.new("RGB", (16, 16), (100 + scan, i, 0)) for i in range(n)], list(range(n))
    ds._load_ct = load
    return ds


ROW = {"dynamic_prompt": f"pair {IMAGES_LINE_TOKEN} Q", "image_study_uid": "cur", "image_series_uid": "x",
       "prior_image_study_uid": "pri", "prior_image_series_uid": "y", "prior_gap_days": 91, "index": "A1"}


def test_prior_slices_reach_codex_first_matching_the_prompt_line(fake_codex):
    item = _tagged_ds([ROW])[0]
    assert "first 7 images are axial slices of the PRIOR CT (91 days before the current CT)" in item["question"]
    assert "the next 5 are axial slices of the CURRENT CT" in item["question"] and "red = lung" in item["question"]
    out = fake_codex.infer(None, None, [fake_codex.create_template(item)], 100, CHOICES)
    assert out[0]["text"] == "Progression"  # the fake answers Progression only when the order matches
    assert out[0]["usage"]["image_count"] == 12 and out[0]["usage"]["prompt_tokens"] == 1000 + 330 * 12


def test_swapped_order_fails_the_order_check(fake_codex):
    # Negative control: the same check must catch current-before-prior.
    item = _tagged_ds([ROW], swap=True)[0]
    out = fake_codex.infer(None, None, [fake_codex.create_template(item)], 100, CHOICES)
    assert out[0]["text"] == "Response"


def test_current_only_line_names_only_the_current_scan():
    row = {k: v for k, v in ROW.items() if not k.startswith("prior")}
    row["dynamic_prompt"] = f"cur {IMAGES_LINE_TOKEN} Q"
    q = _tagged_ds([row])[0]["question"]
    assert "the 5 images are axial slices of the CURRENT CT" in q and "PRIOR" not in q


def test_answers_are_cached_and_rerun_free(fake_codex, tmp_path):
    item = _tagged_ds([ROW])[0]
    msg = fake_codex.create_template(item)
    first = fake_codex.infer(None, None, [msg], 100, CHOICES)[0]
    second = fake_codex.infer(None, None, [msg], 100, CHOICES)[0]
    assert not first["cached"] and second["cached"] and second["text"] == first["text"]
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1


def test_off_list_answer_is_an_error_retried_once_and_not_cached(fake_codex, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "offlist")
    monkeypatch.setattr("models.codex_exec.time.sleep", lambda s: None)
    msg = {"question": "q", "images": [Image.new("RGB", (8, 8))]}
    out = fake_codex.infer(None, None, [msg], 100, CHOICES)[0]
    assert "not in options" in out["error"]
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2  # first try + one retry
    assert not any((tmp_path / "cache").rglob("*.json"))


def test_tool_use_in_a_sealed_call_discards_the_answer(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "tool")
    monkeypatch.setattr("models.codex_exec.time.sleep", lambda s: None)
    out = fake_codex.infer(None, None, [{"question": "q", "images": [Image.new("RGB", (8, 8))]}], 100, CHOICES)[0]
    assert "used a tool" in out["error"]


@pytest.mark.parametrize("mode", ["limit", "credits"])
def test_usage_limit_stops_every_later_call(fake_codex, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    msgs = [{"question": f"q{i}", "images": [Image.new("RGB", (8, 8))]} for i in range(3)]
    out = fake_codex.infer(None, None, msgs, 100, CHOICES)
    assert all(o["error"] for o in out) and "usage limit" in out[2]["error"]
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1  # no retry, no later calls


def test_token_cap_stops_new_calls(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "plain")
    # The cap stops new calls once reached, so a run can pass it by at most one call.
    fake_codex.max_total_tokens = 1300
    msgs = [{"question": f"q{i}", "images": [Image.new("RGB", (8, 8))]} for i in range(2)]
    out = fake_codex.infer(None, None, msgs, 100, CHOICES)
    assert out[0]["text"] == "Stable"  # 0 used before it; it uses 1000 + 330 + 20 = 1,350
    assert out[1]["error"].startswith("token cap")


def test_parse_events_recovered_reconnect_is_not_a_failure():
    ok = "\n".join(json.dumps(e) for e in [{"type": "error", "message": "stream disconnected"},
                                           {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}])
    assert parse_events(ok, '{"answer": "Stable"}', CHOICES)[:3] == ("Stable", {
        "prompt_tokens": 5, "cached_prompt_tokens": None, "output_tokens": 1, "thinking_tokens": None, "total_tokens": 6}, None)
    no_turn = json.dumps({"type": "error", "message": "stream disconnected"})
    assert parse_events(no_turn, "", CHOICES)[2].startswith("codex failed")


def test_sealed_command_hides_the_mount_and_switches_tools_off(tmp_path):
    home = tmp_path / "home"
    for p in ("code", ".config/gcloud", ".claude"):
        (home / p).mkdir(parents=True)
    cmd = build_command("m", [tmp_path / "a.png", tmp_path / "b.png"], "s.json", "o.txt", tmp_path, home=home)
    assert cmd[:4] == ["bwrap", "--dev-bind", "/", "/"]
    hidden = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--tmpfs"]
    assert str(home / "code") in hidden and str(home / ".config" / "gcloud") in hidden and str(home / ".claude") in hidden
    if Path("/mnt").exists():
        assert "/mnt" in hidden
    exec_part = cmd[cmd.index("--") + 1:]
    for feat in ("shell_tool", "apps", "browser_use", "image_generation", "plugins", "memories"):
        assert exec_part[exec_part.index(feat) - 1] == "--disable"
    assert exec_part[-3:] == ["-i", f"{tmp_path / 'a.png'},{tmp_path / 'b.png'}", "-"]
    assert "-s" in exec_part and exec_part[exec_part.index("-s") + 1] == "read-only"
    assert "--ephemeral" in exec_part
    assert exec_part[exec_part.index('web_search="disabled"') - 1] == "-c"
    assert set(hidden_paths(home)) >= {"/mnt", str(home / "code")}


# --- scoring: the image arms and the probe comparison --------------------------------------
def _scoring_setup(tmp_path):
    base, res = tmp_path / "base", tmp_path / "res"
    (base / "tasks").mkdir(parents=True)
    (base / "tasks" / "reg.json").write_text(json.dumps([ENTRY]))
    d = res / "radiology_progression_tasks" / "radiology_progression" / "gpt-test"
    d.mkdir(parents=True)
    # 8 test scans, two per class. Pair answers: right on 6 (misses one Response, one Mixed).
    acc = [f"00{i}" for i in range(8)]
    lab = [0, 0, 1, 1, 2, 2, 3, 3]
    pair = ["Stable", "Stable", "Progression", "Progression", "Response", "Stable", "Mixed Response", "Progression"]
    cur = ["Stable"] * 8
    for arm, ans in (("ct_pair", pair), ("ct_current", cur)):
        pd.DataFrame({"index": acc, "ct_accession_number": acc, "label": lab, "split": "test",
                      "model_response": ans}).to_csv(d / f"radiology_progression_results_{arm}.csv", index=False)
    return base, res, acc, lab


def _probe_file(path, acc, lab, test_pred, val_pred):
    val_acc = [f"9{i}" for i in range(4)]
    df = pd.DataFrame({
        "task": "radiology_progression", "split": ["test"] * len(acc) + ["val"] * 4,
        "accession_number": acc + val_acc, "label": lab + [0, 1, 2, 3],
        "pred_label": test_pred + val_pred})
    df.to_parquet(path)
    return str(path)


def test_compare_to_probe_hand_worked(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.registry._manifest", lambda: MANIFEST)
    base, res, acc, lab = _scoring_setup(tmp_path)
    # Probe A: val 4/4 right; on test always Stable -> balanced accuracy 0.25.
    a = _probe_file(tmp_path / "probe_a_per_example.parquet", acc, lab, [0] * 8, [0, 1, 2, 3])
    # Probe B: val 1/4; on test perfect (it must NOT be chosen: selection is on validation).
    b = _probe_file(tmp_path / "probe_b_per_example.parquet", acc, lab, lab, [0, 0, 0, 0])
    cfg = {"paths": {"base_dir": str(base), "results_dir": str(res), "valid_tasks": "tasks/reg.json"},
           "probe_comparison": {"task": "radiology_progression", "key": "ct_accession_number",
                                "arms": {"ct_pair": [a, b]}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = compare_to_probe(tmp_path / "c.yaml", tmp_path / "cmp.csv", n_boot=50)
    head = out[out.headline].iloc[0]
    assert head.probe == "probe_a" and head.n_matched == 8
    # VLM recalls: Stable 1, Progression 1, Response 1/2, Mixed 1/2 -> 0.75; probe A 0.25.
    assert head.vlm_balanced_accuracy == pytest.approx(0.75)
    assert head.probe_balanced_accuracy == pytest.approx(0.25)
    assert head.balanced_accuracy_diff == pytest.approx(0.5)
    other = out[~out.headline].iloc[0]
    assert other.probe == "probe_b" and other.probe_balanced_accuracy == pytest.approx(1.0)


def test_compare_to_probe_refuses_label_disagreement(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.registry._manifest", lambda: MANIFEST)
    base, res, acc, lab = _scoring_setup(tmp_path)
    bad = _probe_file(tmp_path / "p_per_example.parquet", acc, [1, 0, 1, 1, 2, 2, 3, 3], [0] * 8, [0, 1, 2, 3])
    cfg = {"paths": {"base_dir": str(base), "results_dir": str(res), "valid_tasks": "tasks/reg.json"},
           "probe_comparison": {"task": "radiology_progression", "arms": {"ct_pair": [bad]}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="different label"):
        compare_to_probe(tmp_path / "c.yaml", tmp_path / "cmp.csv", n_boot=10)


def test_score_tb_scores_the_image_arms_and_the_prior_gain(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.registry._manifest", lambda: MANIFEST)
    base, res, _, _ = _scoring_setup(tmp_path)
    cfg = {"paths": {"base_dir": str(base), "results_dir": str(res), "valid_tasks": "tasks/reg.json"},
           "tasks": ["radiology_progression"], "models": [{"type": "codex_exec", "name": "gpt-test"}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = score_tb(tmp_path / "c.yaml", tmp_path / "scores.csv", n_boot=50)
    pair = out[(out.arm == "ct_pair") & (out.split == "test")].iloc[0]
    cur = out[(out.arm == "ct_current") & (out.split == "test")].iloc[0]
    assert pair.balanced_accuracy == pytest.approx(0.75) and cur.balanced_accuracy == pytest.approx(0.25)
    cmp = pd.read_csv(tmp_path / "scores_matched_comparisons.csv")
    c = cmp[(cmp.arm == "ct_pair") & (cmp.versus == "ct_current")].iloc[0]
    assert c.balanced_accuracy_diff == pytest.approx(0.5)


# --- the AI Hub adapter (GPT-6 Astra), no network ------------------------------------------
import models.aihub_openai as aihub_mod  # noqa: E402
from models.aihub_openai import AIHubOpenAIAdapter, build_request as hub_request, parse_response as hub_parse, read_key  # noqa: E402

WAF_PAGE = "<html><head><title>Request Rejected</title></head><body>The requested URL was rejected.</body></html>"


def _hub_ok(answer, prompt=100):
    return json.dumps({"choices": [{"message": {"content": json.dumps({"answer": answer})}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": 30, "total_tokens": prompt + 30,
                                 "completion_tokens_details": {"reasoning_tokens": 12}}})


def test_hub_request_puts_images_first_and_forces_the_options():
    body = hub_request("Q?", [b"a", b"b"], CHOICES, 500)
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"] and content[-1]["text"] == "Q?"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    sch = body["response_format"]["json_schema"]
    assert sch["strict"] is True and sch["schema"]["properties"]["answer"]["enum"] == CHOICES
    assert "reasoning_effort" not in body and "logprobs" not in body and body["max_completion_tokens"] == 500


def test_hub_firewall_page_is_an_error_and_not_retried():
    ans, _, err, retry = hub_parse(200, "text/html; charset=utf-8", WAF_PAGE, CHOICES)
    assert ans is None and "rejected" in err and retry is False


def test_hub_parse_reads_answer_and_usage_and_refuses_off_list():
    ans, usage, err, _ = hub_parse(200, "application/json", _hub_ok("Response"), CHOICES)
    assert (ans, err) == ("Response", None)
    assert usage["prompt_tokens"] == 100 and usage["thinking_tokens"] == 12 and usage["total_tokens"] == 130
    assert "not in options" in hub_parse(200, "application/json", _hub_ok("Worse"), CHOICES)[2]
    assert hub_parse(429, "application/json", "{}", CHOICES)[3] is True
    assert hub_parse(400, "application/json", "{}", CHOICES)[3] is False


def _hub_adapter(tmp_path, monkeypatch, replies):
    monkeypatch.setenv("AI_HUB_API_KEY", "test-key")
    monkeypatch.setattr(aihub_mod.time, "sleep", lambda s: None)
    a = AIHubOpenAIAdapter("gpt-test", cache_dir=str(tmp_path))
    a.configure({"max_retries": 2})
    a.load()
    calls = []

    def fake_post(data):
        calls.append(json.loads(data))
        return replies.pop(0) if replies else (200, "application/json", _hub_ok("Stable"))
    a._post = fake_post
    return a, calls


def test_hub_over_fifty_images_is_refused_before_any_call(tmp_path, monkeypatch):
    a, calls = _hub_adapter(tmp_path, monkeypatch, [])
    out = a.infer(None, None, [{"question": "q", "images": [Image.new("RGB", (8, 8))] * 51}], 100, CHOICES)[0]
    assert "at most 50" in out["error"] and calls == []


def test_hub_rejection_names_the_size_and_answers_are_cached(tmp_path, monkeypatch):
    a, calls = _hub_adapter(tmp_path, monkeypatch, [(200, "text/html", WAF_PAGE)])
    msg = {"question": "q", "images": [Image.new("RGB", (8, 8))]}
    rejected = a.infer(None, None, [msg], 100, CHOICES)[0]
    assert "rejected" in rejected["error"] and "bytes" in rejected["error"] and len(calls) == 1  # no retry
    first = a.infer(None, None, [msg], 100, CHOICES)[0]
    again = a.infer(None, None, [msg], 100, CHOICES)[0]
    assert first["text"] == "Stable" and not first["cached"] and again["cached"] and len(calls) == 2
    assert first["usage"]["image_count"] == 1 and first["usage"]["request_bytes"] > 0


def test_hub_retries_a_rate_limit_then_answers(tmp_path, monkeypatch):
    a, calls = _hub_adapter(tmp_path, monkeypatch, [(429, "application/json", "{}"), (200, "application/json", _hub_ok("Progression"))])
    out = a.infer(None, None, [{"question": "q", "images": []}], 100, CHOICES)[0]
    assert out["text"] == "Progression" and len(calls) == 2


def test_hub_key_from_file_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_HUB_API_KEY", raising=False)
    f = tmp_path / "aihub.env"
    f.write_text("OTHER=1\nAI_HUB_API_KEY='abc123'\n")
    assert read_key(str(f)) == "abc123"
    with pytest.raises(RuntimeError, match="no AI Hub key"):
        read_key(str(tmp_path / "missing.env"))
