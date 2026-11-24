"""The organ-zoom arms (spec planning/vista_eval_vlm/2026-11-24-agent-organ-zoom.md): the zoom viewer
(program_agentic_zoom/tools/zoom_organ.py), the mask staging (scripts/build_masks.py) and the usage counter
(scripts/zoom_usage.py). Synthetic volumes only; expected values worked out by hand from the rules.

    cd optimizer/ct-findings-5 && uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with numpy --with pillow \
        python -m pytest -q tests/test_zoom.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE / "program_agentic_zoom" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "small_finding"))
import zoom_organ as zo  # noqa: E402
import build_masks as bm  # noqa: E402
import zoom_usage as zu  # noqa: E402


# --- the box rule is render.py's ---------------------------------------------------------------------------------

def test_organ_box_by_hand():
    m = np.zeros((100, 200), bool)
    m[20:60, 40:120] = True  # rows 20..59, cols 40..119: box (40, 20, 120, 60), w 80 h 40, margin 8 x 4
    assert zo.organ_box(m, "liver") == (32, 16, 128, 64)
    # lymph nodes: middle half across first (x 60..100, w 40), then the margin (4 x 4)
    assert zo.organ_box(m, "lymph_nodes") == (56, 16, 104, 64)
    assert zo.organ_box(np.zeros((5, 5), bool), "liver") is None


def test_organ_box_matches_render():
    render = pytest.importorskip("render")
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = np.zeros((120, 90), bool)
        y0, x0 = rng.integers(0, 60), rng.integers(0, 45)
        m[y0:y0 + rng.integers(1, 60), x0:x0 + rng.integers(1, 45)] = True
        assert zo.organ_box(m, "liver") == render.organ_box(m, "liver")
        assert zo.organ_box(m, "lymph_nodes") == render.organ_box(m, "lymph_node")


def test_sub_box_by_hand():
    # box 100 wide x 50 high at (10, 20); right half, top half -> x 60..110, y 20..45
    assert zo.sub_box((10, 20, 110, 70), "0.5,0,1,0.5") == (60, 20, 110, 45)
    with pytest.raises(ValueError):
        zo.sub_box((0, 0, 10, 10), "0.6,0,0.5,1")


# --- mask staging ------------------------------------------------------------------------------------------------

def test_slab_labels_union_by_hand():
    lab = np.zeros((4, 4, 6), np.int16)  # 6 NIfTI slices, k = 2 -> 3 slabs
    lab[0, 0, 1] = 5    # liver in slice 1 -> slab 0
    lab[1, 1, 2] = 12   # right upper lobe in slice 2 -> slab 1
    lab[1, 1, 3] = 5    # liver in slice 3, same voxel -> slab 1 has both bits there
    lab[2, 2, 5] = 51   # heart in slice 5 -> slab 2, no bit
    bits, present = bm.slab_labels(lab, 2, 3)
    assert bits.shape == (4, 4, 3)
    assert bits[0, 0, 0] == 1 and bits[0, 0].sum() == 1
    assert bits[1, 1, 1] == 1 | 8
    assert bits[:, :, 2].sum() == 0
    assert present["liver"] == [0, 1] and present["lung_upper_lobe_right"] == [1] and present["lungs"] == [1]
    assert present["heart"] == [2] and present["lung_lower_lobe_left"] == []


def test_ranges_json_by_hand():
    r = bm.ranges_json({"liver": [2, 3, 4, 5, 6]}, 11)
    liver = next(o for o in r["organs"] if o["organ"] == "liver")
    assert (liver["from_slice"], liver["centre_slice"], liver["to_slice"]) == (2, 4, 6)
    assert (liver["from_fraction"], liver["to_fraction"]) == (0.2, 0.6)
    assert next(o for o in r["organs"] if o["organ"] == "heart") == {"organ": "heart", "present": False}
    assert r["slices"] == 11


# --- the zoom viewer, end to end ----------------------------------------------------------------------------------

def scan(tmp_path: Path, log_lines=()) -> Path:
    """64 x 64 x 10 slabs; the 'liver' is x 48..63 (the patient's right), y 16..31, slabs 2..5."""
    ct = tmp_path / "ct"
    ct.mkdir()
    hu = np.full((64, 64, 10), -1000, np.int16)
    hu[48:64, 16:32, 2:6] = 60
    np.savez_compressed(ct / "ct.npz", hu=hu)
    (ct / "meta.json").write_text(json.dumps({"spacing_mm": [1.0, 1.0, 2.5], "orientation": "RAS", "shape": [64, 64, 10]}))
    bits = np.zeros_like(hu, dtype=np.uint8)
    bits[48:64, 16:32, 2:6] = 1
    np.savez_compressed(tmp_path / "organ_mask.npz", bits=bits)
    if log_lines:
        (tmp_path / "views.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log_lines))
    return ct


def run(ct: Path, *args):
    return subprocess.run([sys.executable, "-B", str(TOOLS / "zoom_organ.py"), str(ct), *args],
                          capture_output=True, text=True)


def log_of(ct: Path) -> list[dict]:
    return [json.loads(l) for l in (ct.parent / "views.jsonl").read_text().splitlines()]


def test_zoom_draws_the_organ_on_the_image_left(tmp_path):
    ct = scan(tmp_path)
    out = tmp_path / "z.png"
    r = run(ct, "--organ", "liver", "--slab", "3", "--out", str(out))
    assert r.returncode == 0, r.stderr
    from PIL import Image
    img = np.asarray(Image.open(out))
    assert max(img.shape[1], img.shape[0] - zo.STRIP_PX) == 1024
    # the crop is the liver box plus 10% margin, so the middle is liver: 60 HU in the liver window (-40..160) is
    # (60 + 40) / 200 = half way, 127; the corner is margin outside the organ, -1000 HU, black
    body = img[zo.STRIP_PX:]
    assert abs(int(body[body.shape[0] // 2, body.shape[1] // 2]) - 127) <= 2
    assert body[2, 2] == 0
    # display box: patient right (high x) is the image left
    m = zo.slab_mask(np.load(tmp_path / "organ_mask.npz")["bits"], [1.0, 1.0, 2.5], 3, "liver")
    x0, _y0, x1, _y1 = zo.organ_box(m, "liver")
    assert x1 <= 32 + 2  # x 48..63 -> display columns 0..15 (+ margin)
    e = log_of(ct)[-1]
    assert (e["tool"], e["organ"], e["slab"], e["window"]) == ("zoom_organ", "liver", 3, "liver")
    assert e["bytes"] == out.stat().st_size


def test_zoom_refuses_a_slab_without_the_organ(tmp_path):
    ct = scan(tmp_path)
    r = run(ct, "--organ", "liver", "--slab", "8", "--out", str(tmp_path / "z.png"))
    assert r.returncode == 2 and "slabs 2-5" in r.stdout
    assert not (tmp_path / "z.png").exists()
    assert log_of(ct)[-1]["refused"] == "organ not on slab"


def test_zoom_refuses_after_40_images(tmp_path):
    ct = scan(tmp_path, [{"tool": "zoom_organ", "out": f"/tmp/{i}.png", "bytes": 1000} for i in range(40)])
    r = run(ct, "--organ", "liver", "--slab", "3", "--out", str(tmp_path / "z.png"))
    assert r.returncode == 3 and "REFUSED" in r.stdout


def test_zoom_refuses_past_18_mb(tmp_path):
    ct = scan(tmp_path, [{"tool": "zoom_organ", "out": "/tmp/a.png", "bytes": 18_000_000}])
    r = run(ct, "--organ", "liver", "--slab", "3", "--out", str(tmp_path / "z.png"))
    assert r.returncode == 3


def test_info_lists_slab_ranges(tmp_path):
    ct = scan(tmp_path)
    r = run(ct, "--info")
    d = json.loads(r.stdout)
    assert d["organs"]["liver"] == {"from_slab": 2, "to_slab": 5, "slabs_with_organ": 4}
    assert d["organs"]["lungs"] is None and d["images_left"] == 40


# --- the usage counter --------------------------------------------------------------------------------------------

RANGES = {"liver": (0, 10), "lungs": (8, 40), "lymph_nodes": (15, 40)}


def test_count_one_by_hand():
    log = [
        {"tool": "zoom_organ", "organ": "liver", "slab": 5, "out": "/tmp/a.png"},              # opened
        {"tool": "zoom_organ", "organ": "right_upper_lobe", "slab": 30, "out": "/tmp/b.png"},  # opened: lungs
        {"tool": "zoom_organ", "organ": "lymph_nodes", "slab": 20, "out": "/tmp/c.png"},       # drawn, never opened
        {"tool": "zoom_organ", "refused": "image limit"},
        {"tool": "zoom_organ", "crash": "ValueError"},
        {"tool": "view_ct", "plane": "axial", "slices": [9, 12], "crop": "0,0,1,1", "out": "/tmp/g.png"},  # liver+lungs
        {"tool": "view_ct", "plane": "axial", "slices": [20], "crop": "", "out": "/tmp/h.png"},          # no crop
    ]
    c = zu.count_one(log, {"/tmp/a.png", "/tmp/b.png", "/tmp/g.png", "/tmp/h.png"}, RANGES,
                     {"track": "lung_solid", "bin": "5-9", "slab": 31})
    assert (c["zoom_drawn"], c["zoom_opened"], c["grid_drawn"], c["grid_opened"]) == (3, 2, 2, 2)
    assert (c["refused"], c["crashes"]) == (1, 1)
    assert (c["zoom_opened_liver"], c["zoom_opened_lungs"], c["zoom_opened_lymph_nodes"]) == (1, 1, 0)
    assert c["zoom_regions"] == 2
    assert (c["crop_opened_liver"], c["crop_opened_lungs"], c["crop_opened_lymph_nodes"]) == (1, 1, 0)
    assert c["found_zoom"] is True  # lungs zoom at slab 30, lesion at 31
    lesion_far = {"track": "lymph_node", "bin": "5-9", "slab": 20}
    c2 = zu.count_one(log, {"/tmp/a.png", "/tmp/h.png"}, RANGES, lesion_far)
    assert c2["found_zoom"] is False and c2["found_any"] is True  # the uncropped grid at slab 20


def test_region_ranges_nodes_from_heart_to_lung_top():
    organs = {"organs": [{"organ": "heart", "present": True, "from_slice": 12, "to_slice": 30},
                         {"organ": "lungs", "present": True, "from_slice": 8, "to_slice": 40},
                         {"organ": "liver", "present": False}]}
    assert zu.region_ranges(organs) == {"lungs": (8, 40), "lymph_nodes": (12, 40)}
