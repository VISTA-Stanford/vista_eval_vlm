"""The library arm (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-library.md): the reader's library tool
(search, page, lit, example, check), own-patient hiding, example CT linking, merged "any drug" scoring, usage parsing
and discovery spend. Synthetic data from test_variants' fixtures; expected values worked out by hand.

    cd optimizer/pneumonitis-variants && uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with numpy --with pillow \
        --with pandas python -m pytest -q tests/
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_variants import DATA, VARIANTS, world  # noqa: F401  (world is a fixture)

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

import adapters as ad  # noqa: E402
import agentic  # noqa: E402
import compare  # noqa: E402
import config  # noqa: E402
import stage_examples as se  # noqa: E402

SEED = HERE / "program_agentic_library"
VIEW = {"organ": "lungs", "plane": "axial", "window": "lung", "from": 0.2, "to": 0.8, "count": 4}


def caches(world_ids):
    """The two caches the library sessions link from, under the test work folder."""
    from PIL import Image

    lit = config.LIT_ASSETS_DIR / "lit"
    if lit.exists():
        shutil.rmtree(config.LIT_ASSETS_DIR)
    lit.mkdir(parents=True)
    Image.new("L", (64, 64), 128).save(lit / "oa-fig1.jpg")
    (lit / "index.json").write_text(json.dumps([{"id": "oa-fig1", "file": "oa-fig1.jpg", "cause": "cop",
                                                 "pattern": "reversed halo", "caption": "a reversed halo sign",
                                                 "source": {"title": "t", "journal": "j", "year": 2020}}]))
    (lit / "sheets").mkdir()
    Image.new("RGB", (64, 64), (90, 90, 90)).save(lit / "sheets" / "cop--reversed-halo.jpg")
    (lit / "sheets.json").write_text(json.dumps([{"sheet": "sheets/cop--reversed-halo.jpg", "cause": "cop",
                                                  "pattern_group": "reversed-halo", "ids": ["oa-fig1"]}]))
    tr = config.TRAIN_SCANS_DIR
    if tr.exists():
        shutil.rmtree(tr)
    for i in world_ids:
        d = tr / i
        (d / "ct").mkdir(parents=True)
        for n in ("ct.npz", "meta.json"):
            shutil.copy2(DATA / "patients" / i / n, d / "ct" / n)
        shutil.copy2(DATA / "organs" / f"{i}.json", d / "organs.json")
        shutil.copy2(DATA / "masks" / f"{i}.npz", d / "organ_mask.npz")
    causes = {"s1": "cop", "s2": "radiation", "s5": "cop", "s3": "infection", "s4": "infection", "s6": "cop", "c1": "cop"}
    (tr / "index.json").write_text(json.dumps([{"id": i, "cause": causes[i], "group": "g"} for i in world_ids]))


def library_tree(tmp_path: Path, entries: dict[str, dict], pages: dict[str, str]) -> Path:
    import run_loop

    tree = tmp_path / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SEED, tree)
    tool = run_loop.load_atlas_tool(tree)  # the atlas drawing lives in tools/atlas.py of this program too
    for eid, e in entries.items():
        d = tree / "library" / "examples" / eid
        d.mkdir(parents=True)
        organs = json.loads((DATA / "organs" / f"{e['source']}.json").read_text())
        img, why = tool.draw(DATA / "patients" / e["source"], organs, VIEW, [{"box": [0.1, 0.1, 0.5, 0.5]}])
        assert img is not None, why
        tool.save(img, d / "1.jpg")
        (d / "entry.json").write_text(json.dumps({
            "title": eid, "patterns": ["reversed-halo"], "report_sentence": "", "images": [{"image": "1.jpg", "view": VIEW}],
            "findings": [{"text": "a ring of consolidation around ground glass", "slices": "5-9", "place": "right lower lobe"}],
            **e}))
    for name, text in pages.items():
        (tree / "library" / "pages" / f"{name}.md").write_text(text)
    return tree


def run_tool(tree: Path, *args) -> str:
    out = subprocess.run([sys.executable, "-B", str(tree / "tools" / "library.py"), *args], capture_output=True, text=True,
                         check=False)
    return out.stdout + out.stderr


@pytest.fixture()
def staged(world, tmp_path):
    caches(world)
    tree = library_tree(tmp_path, {"ex-a": {"cause": "cop", "source": "s1"}, "ex-own": {"cause": "cop", "source": "s5"}},
                        {"index": "# Start\n\nSee [[reversed-halo]] and [[ex:ex-a]] and [[lit:oa-fig1]].\n",
                         "reversed-halo": "# Reversed halo\n\nA ring of consolidation around ground glass; cop.\n"})
    folder = tmp_path / "call"
    agentic.stage_scan("c1", folder)
    counts = se.stage_library(tree, folder, "c1")
    return tree, folder, counts


def test_stage_links_assets_and_hides_own_patient(staged):
    tree, folder, counts = staged
    # c1's patient is P1; s5 is P1 too (test_variants' groups): ex-own is hidden, its CT not linked
    assert counts == {"hidden_examples": 1, "example_scans": 1}
    assert json.loads((folder / "library_assets" / "exclude.json").read_text()) == ["ex-own"]
    assert (folder / "library_scans" / "s1" / "ct" / "ct.npz").is_file()
    assert not (folder / "library_scans" / "s5").exists()
    # hard links, not copies
    assert os.stat(folder / "library_assets" / "lit" / "oa-fig1.jpg").st_ino == os.stat(config.LIT_ASSETS_DIR / "lit" / "oa-fig1.jpg").st_ino


def test_staging_reads_no_label(world, tmp_path, monkeypatch):
    caches(world)
    tree = library_tree(tmp_path, {"ex-a": {"cause": "cop", "source": "s1"}}, {"index": "# Start\n"})

    def boom(*_a, **_k):
        raise AssertionError("a label was read")

    monkeypatch.setattr(ad, "read_split", boom)
    folder = tmp_path / "call2"
    agentic.stage_scan("c1", folder)
    assert se.stage_library(tree, folder, "c1") == {"hidden_examples": 0, "example_scans": 1}


def test_tool_search_page_lit_example_and_log(staged):
    tree, folder, _c = staged
    out = run_tool(tree, "search", "reversed halo", "--scan", str(folder))
    assert "page reversed-halo" in out and "lit oa-fig1" in out and "example ex-a" in out
    assert "ex-own" not in out  # hidden from this scan
    assert "# Reversed halo" in run_tool(tree, "page", "reversed-halo", "--scan", str(folder))
    lit = run_tool(tree, "lit", "oa-fig1", "--scan", str(folder))
    assert "a reversed halo sign" in lit and str(folder / "library_assets" / "lit" / "oa-fig1.jpg") in lit
    out = run_tool(tree, "example", "ex-a", "--scan", str(folder), "--out", str(folder.parent / "lib"))
    assert "slices 5-9, right lower lobe" in out and "left: the example" in out
    assert str(folder / "library_scans" / "s1" / "ct") in out and "not available" not in out
    from PIL import Image

    with Image.open(folder.parent / "lib" / "ex-a-1.jpg") as im:
        assert im.width > im.height  # side by side
    assert "no example" in run_tool(tree, "example", "ex-own", "--scan", str(folder))
    import run_loop

    use = run_loop.library_use(folder / "library_log.jsonl")
    assert use["searches"] == 1 and use["pages_read"] == ["reversed-halo"] and use["lit_images_opened"] == 1
    assert use["examples_opened"] == 1 and use["library_commands"] == 5


def test_suggest_staged_by_similarity_without_own_patient_or_labels(world, tmp_path, monkeypatch):
    caches(world)
    # cop examples from s1 (cos 0.995 to c1), s6 (0.958), s5 (c1's own patient); one infection example from s4
    tree = library_tree(tmp_path, {"cop-s6": {"cause": "cop", "source": "s6"}, "cop-s1": {"cause": "cop", "source": "s1"},
                                   "cop-own": {"cause": "cop", "source": "s5"}, "inf-s4": {"cause": "infection", "source": "s4"}},
                        {"index": "# Start\n"})

    def boom(*_a, **_k):
        raise AssertionError("a label was read")

    monkeypatch.setattr(ad, "read_split", boom)
    folder = tmp_path / "call3"
    agentic.stage_scan("c1", folder)
    se.stage_library(tree, folder, "c1")
    sug = json.loads((folder / "library_assets" / "suggest.json").read_text())
    assert [x["id"] for x in sug["cop"]] == ["cop-s1", "cop-s6"]  # most similar first; the own-patient one never
    assert [x["id"] for x in sug["infection"]] == ["inf-s4"] and sug["radiation"] == []
    out = run_tool(tree, "suggest", "--scan", str(folder), "--out", str(tmp_path / "sug"))
    assert "[cop] cop-s1" in out and "[cop] cop-s6" in out and "cop-own" not in out
    assert "cop--reversed-halo.jpg" in out  # the literature sheet for the cause
    from PIL import Image

    with Image.open(tmp_path / "sug" / "suggest-cop-cop-s1.jpg") as im:
        assert im.width > im.height
    import run_loop

    use = run_loop.library_use(folder / "library_log.jsonl")
    assert use["examples_opened"] == 3  # the two cop and the one infection example, logged as opens
    sheets = run_tool(tree, "lit", "--sheets", "--cause", "cop", "--scan", str(folder))
    assert "reversed-halo (1 figures)" in sheets


def test_drawn_folder_per_cause(world, tmp_path):
    caches(world)
    tree = library_tree(tmp_path, {"cop-s6": {"cause": "cop", "source": "s6"}, "cop-s1": {"cause": "cop", "source": "s1"},
                                   "cop-own": {"cause": "cop", "source": "s5"}, "inf-s4": {"cause": "infection", "source": "s4"}},
                        {"index": "# Start\n"})
    folder = tmp_path / "call4"
    agentic.stage_scan("c1", folder)
    se.stage_library(tree, folder, "c1")
    names = se.stage_library_drawn(tree, folder, "c1", k=1)
    # cop: the most similar visible example (s1) and the cop literature sheet; infection: its one example; no own patient
    assert names == ["01-cop-cop-s1.jpg", "02-cop-literature-reversed-halo.jpg", "03-infection-inf-s4.jpg"]
    idx = (folder / "library_drawn" / "index.md").read_text()
    assert "## cop" in idx and "slices 5-9, right lower lobe" in idx and "cop-own" not in idx
    import usage as us

    ev = tmp_path / "ev.jsonl"
    ev.write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": f"/workspace/call/library_drawn/{names[0]}"}}]}}))
    assert us.usage(ev)["drawn_opened"] == 1


def test_hide_similarity(world, tmp_path, monkeypatch):
    caches(world)
    tree = library_tree(tmp_path, {"cop-s6": {"cause": "cop", "source": "s6"}, "cop-s1": {"cause": "cop", "source": "s1"}},
                        {"index": "# Start\n"})
    monkeypatch.setattr(config, "LIBRARY_HIDE_SIM", True)
    folder = tmp_path / "call5"
    agentic.stage_scan("c1", folder)
    se.stage_library(tree, folder, "c1")
    sug = json.loads((folder / "library_assets" / "suggest.json").read_text())
    assert sug["cop"] == [{"id": "cop-s1"}, {"id": "cop-s6"}]  # same pick and order, no scores
    out = run_tool(tree, "suggest", "--scan", str(folder), "--out", str(tmp_path / "s5"))
    assert "(similarity " not in out and "[cop] cop-s1" in out
    se.stage_library_drawn(tree, folder, "c1", k=1)
    assert "(similarity to your scan" not in (folder / "library_drawn" / "index.md").read_text()


def test_tool_finds_the_call_folder_itself(staged):
    tree, folder, _c = staged
    out = subprocess.run([sys.executable, "-B", str(tree / "tools" / "library.py"), "page", "index"], capture_output=True,
                         text=True, cwd=folder / "ct").stdout
    assert "# Start" in out and (folder / "library_log.jsonl").read_text().count('"page"') >= 1


def test_library_check(world, tmp_path):
    import run_loop

    caches(world)
    tree = library_tree(tmp_path, {"ex-a": {"cause": "cop", "source": "s1"}, "ex-b": {"cause": "radiation", "source": "s1"}},
                        {"index": "# Start\n[[nowhere]] [[lit:missing]]\n"})
    probs = run_loop.library_version_problems(tree)
    assert any("[[nowhere]] has no page" in p for p in probs)
    assert any("[[lit:missing]] has no literature image" in p for p in probs)
    assert any("ex-b: cause is not its scan's answer" in p for p in probs)  # s1 is a cop scan
    assert not any("ex-a" in p for p in probs)
    assert any("planning session adds no" in p for p in run_loop.library_version_problems(tree, examples_allowed=False))
    (tree / "library" / "pages" / "index.md").unlink()
    assert any("start-here page" in p for p in run_loop.library_version_problems(tree))


def test_merged_drug_auroc_by_hand():
    # 4 scans: other_drug, immunotherapy, radiation, cop. "drug" = either, probability = max of the two.
    gold = {"a": {"other_drug": 1, "immunotherapy": 0, "radiation": 0, "cop": 0, "infection": 0},
            "b": {"other_drug": 0, "immunotherapy": 1, "radiation": 0, "cop": 0, "infection": 0},
            "c": {"other_drug": 0, "immunotherapy": 0, "radiation": 1, "cop": 0, "infection": 0},
            "d": {"other_drug": 0, "immunotherapy": 0, "radiation": 0, "cop": 1, "infection": 0}}
    # the reader swaps the two drugs on a and b: five-cause other_drug/immunotherapy AUROC suffers, merged drug does not
    probs = {"a": {"other_drug": 0.1, "immunotherapy": 0.9, "radiation": 0.0, "cop": 0.0, "infection": 0.0},
             "b": {"other_drug": 0.9, "immunotherapy": 0.1, "radiation": 0.0, "cop": 0.0, "infection": 0.0},
             "c": {"other_drug": 0.0, "immunotherapy": 0.0, "radiation": 0.9, "cop": 0.0, "infection": 0.0},
             "d": {"other_drug": 0.0, "immunotherapy": 0.0, "radiation": 0.0, "cop": 0.9, "infection": 0.0}}
    ids = sorted(gold)
    _m, per = compare.mean_auroc(gold, probs, ids)
    assert per["other_drug"] == pytest.approx(2 / 3)  # a (0.1) above c and d (0.0) but below b (0.9)
    mm, mper = compare.merged_auroc(gold, probs, ids)
    assert mper["drug"] == 1.0 and mper["radiation"] == 1.0 and mper["cop"] == 1.0
    assert mper["infection"] == 0.5  # no infection scan
    assert mm == pytest.approx((1 + 1 + 1 + 0.5) / 4)


def test_spend_counts_discovery(tmp_path):
    w = tmp_path / "work"
    (w / "private" / "loop" / "library-r1").mkdir(parents=True)
    (w / "private" / "loop" / "library-r1" / "state.json").write_text(json.dumps({
        "rounds": {"2": {"builder_usd": 4.0, "reader_usd": 10.0}},
        "discovery": {"director-plan": {"builder_usd": 3.5}, "radiation": {"builder_usd": 12.25}}}))
    out = subprocess.run([sys.executable, "-I", str(HERE / "scripts" / "spend.py"), str(w), "--before", "1"],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    assert d["builder_usd"] == 19.75 and d["loop_usd"] == 19.75 and d["total_usd"] == 20.75
