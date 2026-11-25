"""The atlas arm (spec planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-atlas-arm.md): the atlas tool's drawing and
checks, the literature check, atlas staging (every entry, never the scan's own patient, no label read), the usage
counter and the three-arm comparison. Synthetic data from test_variants' fixtures.

    cd optimizer/pneumonitis-variants && uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with numpy --with pillow \
        --with pandas python -m pytest -q tests/
"""

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

from test_variants import DATA, VARIANTS, world  # noqa: F401  (world is a fixture)

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

import adapters as ad  # noqa: E402
import compare  # noqa: E402
import stage_examples as se  # noqa: E402
import usage as us  # noqa: E402

ATLAS_SEED = HERE / "program_agentic_atlas"
VIEW = {"organ": "lungs", "plane": "axial", "window": "lung", "from": 0.2, "to": 0.8, "count": 4}


def atlas_tree(tmp_path: Path, per: int = 10, sources=("s1", "s2", "s5")) -> tuple[Path, object]:
    import run_loop

    tree = tmp_path / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(ATLAS_SEED, tree)
    tool = run_loop.load_atlas_tool(tree)
    organs = json.loads((DATA / "organs" / "s1.json").read_text())
    for v in VARIANTS:
        for n in range(per):
            src = sources[n % len(sources)]
            d = tree / "atlas" / "entries" / f"{v.replace('_', '-')}-{n}"
            d.mkdir(parents=True)
            img, why = tool.draw(DATA / "patients" / src, organs, VIEW, [{"arrow": [0.1, 0.1, 0.4, 0.4], "label": "x"}])
            assert img is not None, why
            tool.save(img, d / "image.jpg")
            (d / "entry.json").write_text(json.dumps({
                "variant": v, "source": src, "title": f"{v} {n}", "image": "image.jpg", "view": VIEW,
                "marks": [{"arrow": [0.1, 0.1, 0.4, 0.4]}], "patterns": ["p"], "report_sentence": "",
                "findings": [{"text": "subpleural band", "slices": "5-9", "place": "right lower lobe"}]}))
    return tree, tool


def write_literature(tree: Path, n_sources: int = 5, cite_all: bool = True) -> None:
    srcs = "\n".join(f"- [L{i}] Author {i}, a review, Journal, 2020. https://example.org/{i}" for i in range(1, n_sources + 1))
    cites = " ".join(f"[L{i}]" for i in range(1, n_sources + 1))
    body = "".join(f"## {v}\n\nPatterns of {v}: {cites}\n\n" for v in VARIANTS)
    (tree / "guide" / "literature.md").write_text(f"# Literature\n\n{body}## Sources\n\n{srcs}\n")
    (tree / "guide" / "README.md").write_text("# Guide\n\n## Patterns\n\n- straight-edged band [L1]\n"
                                              + ("" if cite_all else "- reversed halo, no citation\n"))


def test_draw_size_and_marks(world, tmp_path):
    tree, tool = atlas_tree(tmp_path, per=1)
    organs = json.loads((DATA / "organs" / "s1.json").read_text())
    img, why = tool.draw(DATA / "patients" / "s1", organs, VIEW, [{"box": [0.2, 0.2, 0.5, 0.5]}])
    assert img is not None and max(img.size) == 768 and img.mode == "RGB"
    assert tool.draw(DATA / "patients" / "s1", organs, {**VIEW, "count": 9}, [])[0] is None  # more than 4 slices
    assert tool.mark_problem({"box": [0.5, 0.5, 0.2, 0.2]}) and tool.mark_problem({"arrow": [0, 0, 2, 0]})
    assert tool.mark_problem({"arrow": [0, 0, 1, 1], "box": [0, 0, 1, 1]})


def test_atlas_check_needs_exactly_ten_and_slices(world, tmp_path):
    tree, tool = atlas_tree(tmp_path)
    assert tool.atlas_problems() == []
    shutil.rmtree(tree / "atlas" / "entries" / "cop-9")
    assert any("variant cop has 9 entries" in p for p in tool.atlas_problems())
    e = tree / "atlas" / "entries" / "cop-0" / "entry.json"
    d = json.loads(e.read_text())
    d["findings"] = [{"text": "band", "slices": "lower lobes"}]
    e.write_text(json.dumps(d))
    assert any("slices" in p for p in tool.atlas_problems())


def test_literature_check(world, tmp_path):
    tree, tool = atlas_tree(tmp_path, per=1)
    assert any("missing" in p for p in tool.literature_problems())
    write_literature(tree, n_sources=5)
    assert tool.literature_problems() == [] and tool.cited_sources() == 5
    write_literature(tree, n_sources=4)
    assert any("cites 4 listed sources; at least 5" in p for p in tool.literature_problems())
    write_literature(tree, n_sources=5, cite_all=False)
    assert any("pattern bullet cites no listed source" in p for p in tool.literature_problems())
    (tree / "guide" / "literature.md").write_text((tree / "guide" / "literature.md").read_text()
                                                  .replace("https://example.org/3", "no link"))
    assert any("L3 has no link" in p for p in tool.literature_problems())


def test_round_zero_and_round_one_rules(world, tmp_path):
    import run_loop

    tree, _tool = atlas_tree(tmp_path)
    write_literature(tree)
    allowed = {"s1": "radiation", "s2": "radiation", "s5": "radiation"}
    probs = run_loop.atlas_version_problems(tree, allowed, literature_only=True)
    assert any("round 0 builds no atlas entries" in p for p in probs)
    assert any("variant is not its scan's answer" in p for p in probs)  # e.g. a cop entry from a radiation scan
    probs = run_loop.atlas_version_problems(tree, {"s1": "radiation"}, literature_only=False)
    assert any("source is not one of your scans" in p for p in probs)


def test_stage_atlas_every_entry_not_own_patient_no_label(world, tmp_path, monkeypatch):
    tree, _tool = atlas_tree(tmp_path)  # sources s1, s2, s5; s5 is the case c1's patient

    def boom(*_a, **_k):
        raise AssertionError("a label was read")

    monkeypatch.setattr(ad, "read_split", boom)
    planned = se.planned_atlas(tree, "c1")
    by_src = {eid: json.loads((tree / "atlas" / "entries" / eid / "entry.json").read_text())["source"] for eid in planned}
    assert "s5" not in by_src.values()
    # 10 per cause, every third (n = 2, 5, 8) from s5: 7 per cause, 35 in all
    assert len(planned) == 35
    import agentic

    folder = tmp_path / "call"
    agentic.stage_scan("c1", folder)
    names = se.stage_atlas(tree, folder, "c1")
    assert len(names) == 35 and names[0].startswith("01-radiation-") and names[-1].split("-")[1] == "infection"
    idx = (folder / "atlas_images" / "index.md").read_text()
    assert "## cop" in idx and "slices 5-9, right lower lobe: subpleural band" in idx
    for f in (folder / "atlas_images").iterdir():
        assert not f.stat().st_mode & stat.S_IWUSR


def test_empty_atlas_stages_nothing(world, tmp_path):
    tree = tmp_path / "empty"
    shutil.copytree(ATLAS_SEED, tree)
    folder = tmp_path / "call0"
    import agentic

    agentic.stage_scan("c1", folder)
    assert se.stage_atlas(tree, folder, "c1") == []
    assert "The atlas has no examples" in (folder / "atlas_images" / "index.md").read_text()


def test_usage_counts_atlas_images_and_shell_guide_reads(tmp_path):
    ev = tmp_path / "events.jsonl"
    blocks = [{"type": "tool_use", "name": "Read", "input": {"file_path": "/w/call/atlas_images/01-radiation-a.png"}},
              {"type": "tool_use", "name": "Read", "input": {"file_path": "/w/call/atlas_images/02-radiation-b.png"}},
              {"type": "tool_use", "name": "Bash", "input": {"command": "cat /w/prog/guide/literature.md | head"}}]
    ev.write_text(json.dumps({"type": "assistant", "message": {"content": blocks}}))
    assert us.usage(ev) == {"drawn_opened": 2, "read_guide": True, "transcript": True}


def test_compare_three_arms(world, tmp_path, monkeypatch):
    gold = {f"x{i}": {v: int(v == VARIANTS[i % 5]) for v in VARIANTS} for i in range(30)}
    (DATA / "splits").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ad, "read_split", lambda s: [{"id": i, **g} for i, g in gold.items()])
    groups = json.loads((DATA / "groups.json").read_text())
    (DATA / "groups.json").write_text(json.dumps({**groups, **{i: f"g{n // 2}" for n, i in enumerate(sorted(gold))}}))
    arms = {}
    for name, sharp in (("zero", 0.0), ("guide", 0.5), ("atlas", 1.0)):
        ans = {i: {"error": None, **{v: 0.5 + sharp * (g[v] - 0.5) * 0.9 + 0.01 * (hash((i, v)) % 7) for v in VARIANTS}}
               for i, g in gold.items()}
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"answers": ans}))
        arms[name] = p
    monkeypatch.setattr(sys, "argv", ["compare", "--zero", str(arms["zero"]), "--guide", str(arms["guide"]),
                                      "--arm", f"atlas={arms['atlas']}", "--pair", "atlas:zero-shot",
                                      "--pair", "atlas:guide", "--out", str(tmp_path / "cmp")])
    assert compare.main() == 0
    out = json.loads((tmp_path / "cmp" / "comparison.json").read_text())
    assert set(out["arms"]) >= {"zero-shot", "guide", "atlas"}
    assert out["arms"]["atlas"]["mean"] == 1.0
    assert out["pairs"]["atlas - zero-shot"]["mean"] > 0
    (DATA / "groups.json").write_text(json.dumps(groups))
    shutil.rmtree(DATA / "splits")
