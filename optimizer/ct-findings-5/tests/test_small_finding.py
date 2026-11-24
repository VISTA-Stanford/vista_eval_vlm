"""Small-finding perception item rules (small_finding/items.py) and the layer 2 gate (small_finding/score.py).
Synthetic inputs only; every expected value is worked out by hand from the rule, not by running the code.

    cd optimizer/ct-findings-5 && uv run -q --no-project --with pytest --with numpy python -m pytest -q tests/test_small_finding.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_finding"))
import items as it  # noqa: E402
import score as sc  # noqa: E402


@pytest.mark.parametrize("text,want", [
    ("4/56", (4, 56)), ("(4/56)", (4, 56)), ("3:120", (3, 120)),
    ("Series 3, image 120", (3, 120)), ("series 2 image 45", (2, 45)),
    ("image 45 of series 2", (2, 45)),
    ("56", None), ("image 56", None), ("", None), (None, None),
])
def test_parse_cite(text, want):
    assert it.parse_cite(text) == want


@pytest.mark.parametrize("text,want", [
    ("8 mm", [8.0]), ("1.2 x 0.9 cm", [12.0, 9.0]), ("4-mm", [4.0]), ("12 mm x 8 mm", [12.0, 8.0]),
    ("1.2 x 0.9 x 0.8 cm", [12.0, 9.0, 8.0]),
    ("3-4 mm", None), ("up to 6 mm", None), ("subcentimeter", None), ("<5 mm", None),
    ("12 mm x 8 cm", None),   # mixed units: refuse rather than guess
    ("8", None), ("", None), (None, None),
])
def test_parse_size(text, want):
    got = it.parse_size(text)
    assert got == (pytest.approx(want) if want else want)


def test_lesion_mm_node_short_axis():
    assert it.lesion_mm([15.0, 9.0], "lymph_node", "long_x_short") == 9.0   # short axis given
    assert it.lesion_mm([15.0, 9.0], "lung_solid", "long_x_short") == 15.0  # other tracks: largest
    assert it.lesion_mm([12.0], "lymph_node", "long") == 12.0               # no short axis: largest
    assert it.lesion_mm([7.0], "lymph_node", "short") == 7.0


@pytest.mark.parametrize("mm,want", [(4.9, "<5"), (5.0, "5-9"), (9.9, "5-9"), (10.0, "10-19"),
                                     (19.9, "10-19"), (20.0, ">=20")])
def test_bin_of(mm, want):
    assert it.bin_of(mm) == want


def test_nifti_z_runs_head_to_foot():
    # 101 slices: centre is z = 50, where the middle header (instance 50) sits
    assert it.nifti_z(101, 50, 50) == 50
    assert it.nifti_z(101, 60, 50) == 40   # a higher image number is further toward the feet: lower z
    assert it.nifti_z(101, 40, 50) == 60


@pytest.mark.parametrize("dz,k", [(1.25, 2), (0.625, 4), (2.5, 1), (5.0, 1), (1.0, 2)])
def test_slab_factor_matches_converter(dz, k):
    # build_data.convert: k = round(2.5 / dz) when dz < 2.5 (Python rounds 2.5 to 2), else 1
    assert it.slab_factor(dz) == k


def test_centre_offset():
    # 1.25 mm slices. Odd slice count: header on the centre (0); even: half a slice above (+0.5)
    assert it.centre_offset(100.0, 100.0, 1.25) == pytest.approx(0.0)
    assert it.centre_offset(100.625, 100.0, 1.25) == pytest.approx(0.5)
    assert it.centre_offset(105.0, 100.0, 1.25) is None   # four slices off: a missing end, dropped
    assert it.centre_offset(None, 100.0, 1.25) is None
    # an even-count scan: image 60 vs middle instance 50 -> 10 slices toward the feet from the header, which sits
    # half a slice above the centre of 100 slices (49.5): 49.5 + 0.5 - 10 = 40
    assert it.nifti_z(100, 60, 50) + it.centre_offset(100.625, 100.0, 1.25) == pytest.approx(40.0)


def test_npz_z():
    assert it.npz_z(41.4, 2) == 20
    assert it.npz_z(41.6, 2) == 21
    assert it.npz_z(41.6, 1) == 42


@pytest.mark.parametrize("lesion,want", [
    ({"organ": "lungs", "ct_descriptor": "ground-glass nodule"}, "lung_ground_glass"),
    ({"organ": "lungs", "ct_descriptor": "part-solid ground-glass nodule"}, "lung_solid"),
    ({"organ": "lungs", "ct_descriptor": "solid nodule"}, "lung_solid"),
    ({"organ": "lungs", "ct_descriptor": "thin-walled cyst"}, None),
    ({"organ": "liver", "ct_descriptor": "hypodensity"}, "liver"),
    ({"organ": "lymph nodes", "location": "right paratracheal"}, "lymph_node"),
    ({"organ": "lymph nodes", "location": "retroperitoneal"}, None),
    ({"organ": "bone", "ct_descriptor": "lytic lesion"}, None),
])
def test_track_of(lesion, want):
    assert it.track_of(lesion) == want


@pytest.mark.parametrize("loc,want", [("right upper lobe", "right"), ("LLL", "left"), ("lingula", "left"),
                                      ("bilateral lower lobes", None), ("", None)])
def test_side_of(loc, want):
    assert it.side_of(loc) == want


@pytest.mark.parametrize("text,track,want", [
    ("lungs: no pulmonary nodules.", "lung_solid", True),
    ("no suspicious pulmonary nodules or masses", "lung_ground_glass", True),
    ("the liver is unremarkable.", "liver", True),
    ("no focal hepatic lesions", "liver", True),
    ("no mediastinal or hilar lymphadenopathy", "lymph_node", True),
    ("no thoracic lymphadenopathy", "lymph_node", True),
    ("new pulmonary nodules", "lung_solid", False),
    ("mediastinal lymphadenopathy", "lymph_node", False),
])
def test_negative_phrases(text, track, want):
    assert bool(it.NEG_RX[track].search(text)) is want


def test_slab_mapping_on_a_real_conversion(tmp_path):
    # Run ct-findings-5's own converter on a synthetic 1.25 mm volume whose slice z holds the value z, and check
    # that slab npz_z(z, k) holds slice z (each slab is the mean of k = 2 slices: slab s = mean(2s, 2s+1)).
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_data as bd
    n = 41  # odd: the last slice falls in a dropped partial slab
    vol = np.broadcast_to(np.arange(n, dtype=np.float32), (4, 4, n)).copy()
    src = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([0.7, 0.7, 1.25, 1.0])), src)
    status, _ = bd.convert(("x", str(src), str(tmp_path)))
    assert status == "converted"
    hu = np.load(tmp_path / "x" / "ct.npz")["hu"]
    k = it.slab_factor(1.25)
    assert k == 2 and hu.shape[2] == n // k
    for z in (0, 1, 10, 25, 39):
        s = it.npz_z(z, k)
        assert hu[0, 0, s] == 2 * s  # mean of 2s and 2s+1 is 2s + 0.5, which rint rounds to the even 2s
        assert 2 * s <= z <= 2 * s + 1
    assert it.npz_z(40, k) >= hu.shape[2]  # slice 40 is in the dropped partial slab: items.py drops it


def test_layer2_job_selection():
    import run as rn
    items = ([{"item": f"p{j}", "track": "liver", "bin": "5-9", "role": "positive"} for j in range(2)]
             + [{"item": f"n{j}", "track": "liver", "bin": "5-9", "role": "negative"} for j in range(2)]
             + [{"item": f"e{j}", "track": "liver", "bin": "5-9", "role": "example"} for j in range(4)]
             + [{"item": "q0", "track": "liver", "bin": "<5", "role": "positive"}]          # passes, no examples
             + [{"item": "r0", "track": "lung_solid", "bin": "<5", "role": "positive"}])   # fails for both
    gate = {"bins": [{"model": "claude-opus-5-5", "track": "liver", "bin": "5-9", "pass": True},
                     {"model": "claude-sonnet-5-5", "track": "liver", "bin": "5-9", "pass": False},
                     {"model": "claude-sonnet-5-5", "track": "liver", "bin": "<5", "pass": True},
                     {"model": "claude-opus-5-5", "track": "lung_solid", "bin": "<5", "pass": False}]}
    jobs, passed, skipped = rn.layer2_jobs(items, gate)
    # only Opus, only liver 5-9, all four test items, three distinct examples each from that bin's pool
    assert {(j["model"], j["item"]["item"]) for j in jobs} == {("claude-opus-5-5", x) for x in ("p0", "p1", "n0", "n1")}
    assert all(len({e["item"] for e in j["examples"]}) == 3 and all(e["item"].startswith("e") for e in j["examples"])
               for j in jobs)
    assert skipped == {("liver", "<5")} and ("lung_solid", "<5") not in passed


def test_wilson():
    # 0 of 12: Wilson 95% upper end = z^2/(n+z^2) = 3.8416/15.8416 = 0.2425 (the normal approximation gives 0)
    lo, hi = sc.wilson(0, 12)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(0.2425, abs=1e-3)


def test_net_change():
    # positives: 4/10 -> 8/10 (+0.4); negatives: 1/10 -> 5/10 (+0.4): the examples only moved the threshold -> net 0
    p = ([1] * 4 + [0] * 6, [1] * 8 + [0] * 2)
    q = ([1] * 1 + [0] * 9, [1] * 5 + [0] * 5)
    assert sc.net_boot(p, q)[0] == pytest.approx(0.0)
    # same positives, negatives unchanged: net +0.4
    assert sc.net_boot(p, (q[0], q[0]))[0] == pytest.approx(0.4)


def test_gate():
    # Gate: sensitivity - false-positive rate > 0.1 and the 95% normal-approximation interval's low end > 0.
    # 20/40 vs 4/40: diff 0.4, se = sqrt(.5*.5/40 + .1*.9/40) = sqrt(.0085) = .0922, low = .4 - 1.96*.0922 = .2193
    r = sc.gate([1] * 20 + [0] * 20, [1] * 4 + [0] * 36)
    assert r["diff"] == pytest.approx(0.4) and r["lo"] == pytest.approx(0.2193, abs=1e-3) and r["pass"] is True
    # 9/40 vs 4/40: diff 0.125 > 0.1, but se = sqrt(.225*.775/40 + .0025*36/40... ) -> .0813, low = -.0343: fails
    r = sc.gate([1] * 9 + [0] * 31, [1] * 4 + [0] * 36)
    assert r["diff"] == pytest.approx(0.125) and r["lo"] == pytest.approx(-0.0343, abs=1e-3) and r["pass"] is False
    # 6/40 vs 4/40: diff 0.05, under 0.1: fails whatever the interval
    assert sc.gate([1] * 6 + [0] * 34, [1] * 4 + [0] * 36)["pass"] is False
    # 180/1000 vs 100/1000: diff 0.08, se = sqrt(.18*.82/1000 + .1*.9/1000) = .0154, low = .0498 > 0, but the
    # difference is under 0.1, so only the 0.1 rule fails it
    r = sc.gate([1] * 180 + [0] * 820, [1] * 100 + [0] * 900)
    assert r["lo"] == pytest.approx(0.0498, abs=1e-3) and r["pass"] is False
    # nothing answered: no gate
    assert sc.gate([], [])["pass"] is False


def test_organ_box():
    np = pytest.importorskip("numpy")
    import render as rd
    m = np.zeros((100, 200), bool)
    m[20:60, 40:160] = True  # organ: x 40..160, y 20..60 (exclusive ends)
    # 10% margin: x by 12 px, y by 4 px
    assert rd.organ_box(m, "liver") == (28, 16, 172, 64)
    # lymph node: middle half across (40+30 .. 160-30), margins 6 and 4
    assert rd.organ_box(m, "lymph_node") == (64, 16, 136, 64)
    assert rd.organ_box(np.zeros((10, 10), bool), "liver") is None
    # the margin never runs off the image
    edge = np.zeros((50, 50), bool); edge[0:50, 0:10] = True
    assert rd.organ_box(edge, "liver") == (0, 0, 11, 50)


def test_auroc_and_prob():
    # 3 positives, 2 negatives: pairs (p > q) = 0.9>0.2, 0.9>0.4, 0.6>0.2, 0.6>0.4, 0.4=0.4 (half), 0.4>0.2 -> 5.5/6
    assert sc.auroc([0.9, 0.6, 0.4], [0.2, 0.4]) == pytest.approx(5.5 / 6)
    assert sc.prob({"status": "ok", "answer": "present", "confidence": 80}) == pytest.approx(0.8)
    assert sc.prob({"status": "ok", "answer": "absent", "confidence": 80}) == pytest.approx(0.2)
    assert sc.prob({"status": "refusal"}) == 0.0
