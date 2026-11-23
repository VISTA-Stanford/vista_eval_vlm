"""The knowledge-base pilot's own code: the tester's tool (program_agentic_kb/tools/kb.py), its refactored viewer, the
loop's usage counter (scripts/run_kb_loop.usage) and the round dealing (scripts/build_kb_data.deal). Synthetic volumes
and entries only; every expected value is worked out by hand from the rule it tests.

    cd optimizer/ct-findings-5 && uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with pillow --with numpy \\
        python -m pytest -q tests/test_kb.py
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "program_agentic_kb" / "tools"))
sys.path.insert(0, str(HERE / "scripts"))
import kb  # noqa: E402


def make_ct(folder: Path, nz: int = 40, seed: int = 0) -> Path:
    """A small volume whose slice k is filled with k * 20 - 1000 HU plus noise, so slices are told apart by value."""
    rng = np.random.default_rng(seed)
    hu = np.zeros((32, 28, nz), dtype=np.int16)
    for k in range(nz):
        hu[:, :, k] = k * 20 - 1000 + rng.integers(-5, 5, size=(32, 28))
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "ct.npz", hu=hu)
    (folder / "meta.json").write_text(json.dumps({"spacing_mm": [1.0, 1.2, 2.5], "orientation": "RAS", "shape": list(hu.shape)}))
    return folder


ORGANS = {"available": True, "slices": 40, "organs": [
    {"organ": "lungs", "present": True, "from_fraction": 0.5, "centre_fraction": 0.75, "to_fraction": 1.0},
    {"organ": "liver", "present": True, "from_fraction": 0.1, "centre_fraction": 0.3, "to_fraction": 0.5},
    {"organ": "brain", "present": False}]}


def test_viewer_refactor_draws_the_same_bytes(tmp_path):
    """The KB program's view_ct.py (grid factored out) writes the same PNG as the seed's for the same command."""
    ct = make_ct(tmp_path / "ct")
    outs = []
    for prog in ("program_agentic", "program_agentic_kb"):
        out = tmp_path / f"{prog}.png"
        subprocess.run([sys.executable, "-B", str(HERE / prog / "tools" / "view_ct.py"), str(ct), "--plane", "axial",
                        "--from", "0.2", "--to", "0.9", "--count", "7", "--window", "lung", "--crop", "0.1,0.1,0.9,0.8",
                        "--out", str(out)], check=True, capture_output=True)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_axis_range_maps_through_the_organ():
    # lungs span 0.5..1.0 of the axis: the organ's 0.2..0.8 is 0.5 + 0.2 * 0.5 = 0.6 to 0.5 + 0.8 * 0.5 = 0.9
    assert kb.axis_range({"organ": "lungs", "plane": "axial", "from": 0.2, "to": 0.8}, ORGANS) == pytest.approx((0.6, 0.9))
    # whole scan, and coronal views, take from/to as given
    assert kb.axis_range({"organ": "whole", "from": 0.1, "to": 0.3}, ORGANS) == (0.1, 0.3)
    assert kb.axis_range({"organ": "lungs", "plane": "coronal", "from": 0.4, "to": 0.6}, ORGANS) == (0.4, 0.6)
    # an organ absent or not in the scan's ranges: None; no ranges at all: None
    assert kb.axis_range({"organ": "brain", "from": 0, "to": 1}, ORGANS) is None
    assert kb.axis_range({"organ": "lungs", "from": 0, "to": 1}, {"available": False}) is None


def test_render_lands_on_the_mapped_slices(tmp_path):
    """liver 0..1 = axis 0.1..0.5 of 40 slices = slices round(0.1*39)=4 .. round(0.5*39)=20 (19.5 rounds to 20);
    count 3 -> 4, 12, 20. The first tile's mean value must be slice 4's HU (4*20-1000 = -920) under a wide window."""
    ct = make_ct(tmp_path / "ct")
    view = {"organ": "liver", "plane": "axial", "window": "-1100,-100", "from": 0.0, "to": 1.0, "count": 3}
    img, why = kb.render(ct, ORGANS, view, size=4096)
    assert why is None
    hu, meta = kb.view_ct.load(ct)
    want = kb.view_ct.grid(hu, meta["spacing_mm"], "axial", [4, 12, 20], "-1100,-100", "", 4096)
    assert np.array_equal(np.asarray(img), np.asarray(want))


def entry(**over):
    e = {"finding": "pneumonitis", "answer": "yes", "title": "Peripheral ground glass", "words": ["ground", "glass"],
         "note": "Look at both lower lobes.", "report_sentence": "patchy ground glass", "source": "s1",
         "views": [{"organ": "lungs", "plane": "axial", "window": "lung", "from": 0.0, "to": 0.5, "count": 4,
                    "image": "view1.png"}], "text_only": False}
    e.update(over)
    return e


@pytest.fixture
def library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    (lib / "entries").mkdir(parents=True)
    monkeypatch.setattr(kb, "LIBRARY", lib)
    monkeypatch.setattr(kb, "ENTRIES", lib / "entries")

    def add(eid, e, image=True):
        d = lib / "entries" / eid
        d.mkdir()
        (d / "entry.json").write_text(json.dumps(e))
        if image:
            Image.new("L", (64, 48), color=128).save(d / "view1.png")
    return add


def test_entry_problems(library):
    library("good", entry())
    library("text", entry(views=[], text_only=True, source=""), image=False)
    library("mismatch", entry(text_only=True))
    library("noimage", entry(), image=False)
    library("Bad_Name", entry())
    library("badview", entry(views=[{"organ": "lungs", "from": 0.8, "to": 0.2, "count": 4, "image": "view1.png"}]))
    found = {eid: kb.entry_problems(eid, e, {"s1"}) for eid, e in kb.entries()}
    assert found["good"] == [] and found["text"] == []
    assert any("text_only" in p for p in found["mismatch"])
    assert any("image" in p for p in found["noimage"])
    assert any("lowercase" in p for p in found["Bad_Name"])
    assert any("from and to" in p for p in found["badview"])
    assert any("source" in p for p in kb.entry_problems("good", entry(source="tester-scan"), {"s1"}))
    assert any("answer" in p for p in kb.entry_problems("good", entry(answer="maybe"), {"s1"}))


def test_list_sorts_by_similarity_and_filters(library, tmp_path, capsys):
    library("a-node", entry(finding="lesion_lymph_nodes", source="s1", title="node A"))
    library("b-node", entry(finding="lesion_lymph_nodes", source="s2", title="node B", words=["paratracheal"]))
    library("c-pneu", entry(source="s3"))
    library("d-general", entry(finding="general", answer=None, source="s4", title="windows"))
    call = tmp_path / "call"
    call.mkdir()
    (call / "kb_similarity.json").write_text(json.dumps({"s1": 0.91, "s2": 0.95, "s3": 0.99, "s4": 0.5}))
    kb.main(["list", "--finding", "lesion_lymph_nodes", "--scan", str(call)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "finding=" in ln]
    # pneumonitis is filtered out; "general" always matches; most similar first: s2 (0.95), s1 (0.91), s4 (0.5)
    assert [ln.split()[0] for ln in lines] == ["b-node", "a-node", "d-general"]
    kb.main(["list", "--words", "PARATRACHEAL"])  # words match case-blind, anywhere in the text
    assert [ln.split()[0] for ln in capsys.readouterr().out.splitlines() if "finding=" in ln] == ["b-node"]


def test_open_draws_example_beside_own_scan(library, tmp_path, capsys):
    library("c-pneu", entry())
    call = tmp_path / "call"
    make_ct(call / "ct")
    (call / "organs.json").write_text(json.dumps(ORGANS))
    assert kb.main(["open", "c-pneu", "--scan", str(call), "--out", str(tmp_path / "out")]) == 0
    out = Image.open(tmp_path / "out" / "c-pneu-view1.png")
    own, _ = kb.render(call / "ct", ORGANS, entry()["views"][0])
    # both halves at the own render's height: the 64x48 example scales to 64*h/48 wide; plus the 12 px gap
    h = max(48, own.height)
    assert out.size == (round(64 * h / 48) + 12 + round(own.width * h / own.height), h + 18)
    assert "right: your scan" in capsys.readouterr().out


def test_open_without_the_organ_shows_the_example_only(library, tmp_path, capsys):
    library("c-pneu", entry())
    call = tmp_path / "call"
    make_ct(call / "ct")
    (call / "organs.json").write_text(json.dumps({"available": False}))
    kb.main(["open", "c-pneu", "--scan", str(call), "--out", str(tmp_path / "out")])
    assert Image.open(tmp_path / "out" / "c-pneu-view1.png").size == (64, 48)
    assert "example only" in capsys.readouterr().out


def test_usage_counts_opens_reads_and_returns(tmp_path):
    import kb_usage

    ev = tmp_path / "events.jsonl"

    def bash(cmd):
        return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}

    def read(p):
        return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": p}}]}}
    rows = [bash("python3 -B tools/view_ct.py ../call/ct --count 9"),          # own view before any open: not a return
            bash("python3 -B tools/kb.py list --finding pneumonitis --scan ../call"),
            bash("python3 -B tools/kb.py open c-pneu --scan ../call"), read("/tmp/kb/c-pneu-view1.png"),
            bash("python3 -B tools/view_ct.py ../call/ct --from 0.6 --to 0.9"),  # a return
            bash("python3 -B tools/kb.py open a-node --scan ../call && python3 -B tools/kb.py open c-pneu --scan ../call"),
            read("/tmp/kb/a-node-view1.png"), read("/tmp/scan.png"),
            bash("python3 -B tools/view_ct.py ../call/ct --plane coronal")]     # a second return
    ev.write_text("\n".join(json.dumps(r) for r in rows))
    u = kb_usage.usage(ev)
    assert u["entries_opened"] == ["a-node", "c-pneu"]
    assert u["example_images_read"] == 2 and u["own_views_after_an_open"] == 2 and u["list_calls"] == 1


def test_deal_gives_every_round_the_same_mix():
    import build_kb_data as bkd

    ids = {"P": [f"p{i}" for i in range(75)], "N": [f"n{i}" for i in range(175)], "O": [f"o{i}" for i in range(250)]}
    rounds = bkd.deal(ids, 5, random.Random(0))
    assert [len(r) for r in rounds] == [100] * 5
    assert sorted(x for r in rounds for x in r) == sorted(x for v in ids.values() for x in v)
    for r in rounds:
        assert sum(x[0] == "p" for x in r) == 15 and sum(x[0] == "n" for x in r) == 35
    assert bkd.klass({"pneumonitis": 1, "lesion_lymph_nodes": 0}) == "P"
    assert bkd.klass({"pneumonitis": None, "lesion_lymph_nodes": 1}) == "N"
    assert bkd.klass({"pneumonitis": 0, "lesion_lymph_nodes": None}) == "O"
