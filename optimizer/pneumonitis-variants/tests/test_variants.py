"""pneumonitis-variants (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md, Code layout: "tests/ -
staging, scoring, example picking never reads the case's label"). Synthetic data in a temporary data folder; expected
values worked out by hand.

    cd optimizer/pneumonitis-variants && uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with numpy --with pillow \
        python -m pytest -q tests/
"""

import json
import os
import random
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="pv-test-"))
os.environ["TASK_DATA_DIR"] = str(_TMP / "data")
os.environ["TASK_PRIVATE_DIR"] = str(_TMP / "private")
os.environ["TASK_WORKDIR"] = str(_TMP / "work")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import adapters as ad  # noqa: E402
import agentic  # noqa: E402
import build_data as bd  # noqa: E402
import compare  # noqa: E402
import config  # noqa: E402
import stage_examples as se  # noqa: E402
import usage as us  # noqa: E402

DATA = Path(os.environ["TASK_DATA_DIR"])
VARIANTS = tuple(k for k, _t, _q in config.QUESTIONS)


# --- fixtures: a tiny data folder and a program version ----------------------------------------------------------


def write_scan(sid: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    d = DATA / "patients" / sid
    d.mkdir(parents=True, exist_ok=True)
    hu = rng.integers(-1000, 200, size=(32, 32, 20)).astype(np.int16)
    np.savez_compressed(d / "ct.npz", hu=hu)
    (d / "meta.json").write_text(json.dumps({"shape": list(hu.shape), "spacing_mm": [1.0, 1.0, 2.5],
                                             "orientation": "RAS", "slab_averaged": 1}))
    (DATA / "organs").mkdir(parents=True, exist_ok=True)
    (DATA / "masks").mkdir(parents=True, exist_ok=True)
    (DATA / "organs" / f"{sid}.json").write_text(json.dumps(bd.ranges_json({"lungs": list(range(4, 18))}, 20)))
    np.savez_compressed(DATA / "masks" / f"{sid}.npz", bits=np.zeros(hu.shape, np.uint8))


@pytest.fixture(scope="module")
def world():
    """Scans: case c1 (patient P1), sources s1..s6 (s5 is patient P1 too). Embeddings in 2D so cosines are known."""
    ids = ["c1", "s1", "s2", "s3", "s4", "s5", "s6"]
    vec = {"c1": (1, 0), "s1": (1, 0.1), "s2": (1, 0.5), "s3": (0, 1), "s4": (1, 0.2), "s5": (1, 0), "s6": (1, 0.3)}
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "embeddings.npz", ids=np.array(ids), emb=np.array([vec[i] for i in ids], np.float32))
    (DATA / "groups.json").write_text(json.dumps({"c1": "P1", "s1": "P2", "s2": "P3", "s3": "P4", "s4": "P5",
                                                  "s5": "P1", "s6": "P6"}))
    for n, i in enumerate(ids):
        write_scan(i, n)
    se._embeddings.cache_clear()
    se._groups.cache_clear()
    return ids


def make_tree(root: Path, entries: dict[str, dict]) -> Path:
    """A program version: the seed's tools, and examples/entries/<eid>/ with one rendered view each."""
    import shutil

    tree = root / f"tree-{random.random():.6f}"
    shutil.copytree(config.SEED_PROGRAM_DIR, tree)
    tool = se.load_tool(tree)
    for eid, e in entries.items():
        d = tree / "examples" / "entries" / eid
        d.mkdir(parents=True)
        view = {"organ": "lungs", "plane": "axial", "window": "lung", "from": 0.2, "to": 0.8, "count": 4}
        organs = json.loads((DATA / "organs" / f"{e['source']}.json").read_text())
        img, why = tool.render(DATA / "patients" / e["source"], organs, view, size=256)
        assert img is not None, why
        img.save(d / "view1.png")
        (d / "entry.json").write_text(json.dumps({"title": eid, "note": "n", "report_sentence": "", "words": [],
                                                  "views": [{**view, "image": "view1.png"}], **e}))
    return tree


# --- example picking ---------------------------------------------------------------------------------------------


def test_pick_by_hand(world):
    entries = [("a", {"variant": "radiation", "source": "s2", "views": [1]}),
               ("b", {"variant": "radiation", "source": "s1", "views": [1]}),
               ("c", {"variant": "radiation", "source": "s3", "views": [1]}),
               ("d", {"variant": "radiation", "source": "s5", "views": [1]}),   # same patient as the case: never
               ("e", {"variant": "radiation", "source": "c1", "views": [1]}),   # the case itself: never
               ("f", {"variant": "radiation", "source": "s4", "views": []}),    # no views: never
               ("g", {"variant": "infection", "source": "s1", "views": [1]})]
    sims = se.similarities("c1", {"s1", "s2", "s3", "s4", "s5", "c1"})
    # cosines with (1, 0): s1 0.995, s4 0.981, s2 0.894, s3 0.0
    assert sims["s1"] > sims["s4"] > sims["s2"] > sims["s3"]
    groups = json.loads((DATA / "groups.json").read_text())
    assert se.pick(entries, sims, groups, "c1", "radiation", 2) == ["b", "a"]
    assert se.pick(entries, sims, groups, "c1", "radiation", 5) == ["b", "a", "c"]
    assert se.pick(entries, sims, groups, "c1", "infection", 2) == ["g"]
    assert se.pick(entries, sims, groups, "c1", "cop", 2) == []


def test_picking_never_reads_a_label(world, tmp_path, monkeypatch):
    """No split file exists here, and reading one raises: the pick and the staging still work, and they give the same
    picks whatever the case's answer would be."""
    tree = make_tree(tmp_path, {"r1": {"variant": "radiation", "source": "s1"}, "r2": {"variant": "radiation", "source": "s2"},
                                "i1": {"variant": "infection", "source": "s4"}})
    assert not (DATA / "splits").exists()

    def boom(*_a, **_k):
        raise AssertionError("a label was read")

    monkeypatch.setattr(ad, "read_split", boom)
    assert se.planned(tree, "c1") == ["r1", "r2", "i1"]
    folder = tmp_path / "call"
    agentic.stage_scan("c1", folder)
    written = se.stage(tree, folder, "c1")
    assert written == ["01-radiation-r1.png", "02-radiation-r2.png", "03-infection-i1.png"]


def test_stage_writes_read_only_side_by_sides(world, tmp_path):
    tree = make_tree(tmp_path, {"r1": {"variant": "radiation", "source": "s1"}})
    folder = tmp_path / "call"
    agentic.stage_scan("c1", folder)
    se.stage(tree, folder, "c1")
    out = folder / "drawn_examples"
    from PIL import Image

    im = Image.open(out / "01-radiation-r1.png")
    assert im.width > im.height  # two halves side by side
    assert "cause on the example's scan (its answer key): radiation" in (out / "index.md").read_text()
    for f in out.iterdir():
        assert not f.stat().st_mode & stat.S_IWUSR
    assert {p.name for p in folder.iterdir()} == {"ct", "organs.json", "organ_mask.npz", "drawn_examples"}


def test_empty_library_is_the_zero_shot_arm(world, tmp_path):
    tree = make_tree(tmp_path, {})
    folder = tmp_path / "call"
    agentic.stage_scan("c1", folder)
    assert se.stage(tree, folder, "c1") == []
    assert "The library has no examples" in (folder / "drawn_examples" / "index.md").read_text()
    assert list((folder / "drawn_examples").glob("*.png")) == []


def test_stage_needs_a_tree():
    agentic._TREE = None
    spec = agentic.program_spec(forbidden=())
    with pytest.raises(RuntimeError):
        spec.stage(type("C", (), {"payload": {"id": "c1"}})(), _TMP / "nowhere", "p")


# --- the library check -------------------------------------------------------------------------------------------


def test_library_check(world, tmp_path):
    import run_loop

    entries = {f"{v.replace('_', '-')}-{n}": {"variant": v, "source": f"s{n}"} for v in VARIANTS for n in (1, 2)}
    tree = make_tree(tmp_path, entries)
    tool = se.load_tool(tree)
    assert tool.library_problems() == []
    (tree / "guide" / "README.md").write_text("# Guide\n\nRadiation: straight edges.\n")
    allowed = {"s1": None, "s2": None}
    probs = run_loop.version_problems(tree, {s: "radiation" for s in allowed})
    assert any("variant is not its scan's answer" in p for p in probs)  # e.g. infection-1 from s1, a radiation scan
    assert not any("guide" in p for p in probs)
    probs = run_loop.version_problems(tree, {"s1": "radiation"})
    assert any("source is not one of your scans" in p for p in probs)
    # one variant short of 2 entries
    tree2 = make_tree(tmp_path, {k: v for k, v in entries.items() if k != "cop-2"})
    assert any("variant cop has 1 entries" in p for p in se.load_tool(tree2).library_problems())
    # the empty placeholder guide is refused
    assert any("placeholder" in p for p in run_loop.version_problems(tree2, {"s1": "radiation", "s2": "radiation"}))


# --- scoring -----------------------------------------------------------------------------------------------------


def test_auroc_by_hand():
    assert ad.auroc([(1, 0.9), (1, 0.8), (0, 0.1), (0, 0.2)]) == 1.0
    assert ad.auroc([(1, 0.5), (0, 0.5)]) == 0.5
    assert ad.auroc([(1, 0.9), (0, 0.95), (1, 0.3), (0, 0.1)]) == 0.5  # pairs: 0.9>0.1, 0.9<0.95, 0.3>0.1, 0.3<0.95
    assert ad.auroc([(1, 0.9), (0, 0.5), (1, 0.3), (0, 0.1)]) == 0.75  # 0.9>0.5, 0.9>0.1, 0.3<0.5, 0.3>0.1


def test_failed_answer_scores_half():
    assert ad.prob({"error": "no_output", "radiation": 0.9}, "radiation") == 0.5
    assert agentic.percent(101) is None and agentic.percent(True) is None and agentic.percent(40) == 0.4


def test_paired_bootstrap_identical_arms_is_zero():
    gold = {f"x{i}": {v: int(v == VARIANTS[i % 5]) for v in VARIANTS} for i in range(20)}
    a = {i: {v: random.Random(i + v).random() for v in VARIANTS} for i in gold}
    groups = {i: f"p{n // 2}" for n, i in enumerate(sorted(gold))}
    obs, ci = compare.paired_boot(gold, a, a, groups, sorted(gold), n=200)
    assert obs == 0 and ci == [0, 0]
    perfect = {i: {v: float(gold[i][v]) for v in VARIANTS} for i in gold}
    obs, ci = compare.paired_boot(gold, a, perfect, groups, sorted(gold), n=200)
    assert obs > 0 and ci[0] > 0
    assert compare.mean_auroc(gold, perfect, sorted(gold))[0] == 1.0


def test_ranked_first():
    import run_loop

    assert run_loop.ranked_first({"error": None, "radiation": 0.2, "immunotherapy": 0.7, "other_drug": 0.7,
                                  "cop": 0.1, "infection": 0.0}) == "immunotherapy"  # a tie goes to the earlier key
    assert run_loop.ranked_first({"error": "no_output"}) is None


# --- the TRAIN draw ----------------------------------------------------------------------------------------------


def test_draw_train_rules():
    import pandas as pd

    rows, n = [], 0
    for v, patients in (("radiation", 40), ("immunotherapy", 40), ("other_drug", 40), ("cop", 30), ("infection", 40)):
        for p in range(patients):
            rows.append({"id": f"{v}{p}", "group": f"{v}-g{p}", "variant": v})
            n += 1
    train = pd.DataFrame(rows)
    rounds, stats = bd.draw_train(train, random.Random("t"))
    groups = dict(zip(train.id, train.group))
    src = {groups[i] for i in rounds["source"]}
    assert len(rounds["source"]) == 60
    for k in range(1, 6):
        ids = rounds[str(k)]
        assert len(ids) == 80 and len(set(ids)) == 80
        assert not {groups[i] for i in ids} & src
        assert all(sum(i.startswith(v) for i in ids) == 16 for v in VARIANTS)
    # cop: 30 patients, 12 to the source -> 18 fresh, 62 reused across the 5 rounds, no round short
    assert stats["reused_cop"] == 80 - 18 and not any(k.startswith("short") and v for k, v in stats.items())


def test_draw_train_counts_a_short_round():
    import pandas as pd

    rows = [{"id": f"{v}{p}", "group": f"{v}-g{p}", "variant": v}
            for v, n in (("radiation", 40), ("immunotherapy", 40), ("other_drug", 40), ("cop", 15), ("infection", 40))
            for p in range(n)]
    rounds, stats = bd.draw_train(pd.DataFrame(rows), random.Random("t"))
    # cop: 3 fresh; round 1 can only have those 3 (nothing earlier to reuse): 13 short, counted
    assert sum(i.startswith("cop") for i in rounds["1"]) == 3 and stats["short_cop"] >= 13


# --- usage -------------------------------------------------------------------------------------------------------


def test_usage_counts_distinct_drawn_images(tmp_path):
    ev = tmp_path / "events.jsonl"
    reads = ["/w/call/drawn_examples/01-radiation-a.png", "/w/call/drawn_examples/01-radiation-a.png",
             "/w/call/drawn_examples/02-radiation-b.png", "/w/prog/guide/README.md", "/w/call/drawn_examples/index.md"]
    ev.write_text("\n".join(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": p}}]}}) for p in reads))
    assert us.usage(ev) == {"drawn_opened": 2, "read_guide": True, "transcript": True}
    assert us.usage(None)["transcript"] is False
