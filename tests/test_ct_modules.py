"""The shared CT modules (context.windows, context.slices, context.nifti). Spec:
planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md (Code layout)."""

import nibabel as nib
import numpy as np
import pytest

from context import nifti, slices, windows
from context.selectors import ct_selectors


# -- windows ------------------------------------------------------------------------------------

def test_medgemma_rgb_hand_computed():
    hu = np.array([[-2000.0, -1024.0, 40.0, 1024.0, 3000.0]])
    out = windows.rgb_u8(hu, "medgemma")
    # 40 HU: R = 1064/2048*255 = 132.47 -> 132; G = 175/350*255 = 127.5 -> 128 (half to even); B = 40/80*255 = 127.5 -> 128
    assert out[0, 2].tolist() == [132, 128, 128]
    assert out[0, 0].tolist() == [0, 0, 0] and out[0, 1].tolist() == [0, 0, 0]
    assert out[0, 3].tolist() == [255, 255, 255] and out[0, 4].tolist() == [255, 255, 255]


def test_viewer_window_truncates():
    # lung (-1350..150) at -600 HU: 750/1500*255 = 127.5 -> 127 (truncated, the viewer's rule)
    assert windows.window_u8(np.array([-600.0]), "lung").tolist() == [127]
    assert windows.window_u8(np.array([-600.0]), "-1350,150").tolist() == [127]


def test_legacy_names_point_at_the_one_home():
    from context import windowing
    assert windowing.multi_window_rgb is windows.multi_window_rgb
    assert windowing.CHEST_WINDOWS == windows.RGB_SETS["chest"]
    assert ct_selectors.evenly_spaced_k is slices.evenly_spaced_k
    assert windows.describe("medgemma") == "wide, soft-tissue and brain windows"


# -- slices -------------------------------------------------------------------------------------

def test_medgemma_uniform_matches_notebook_rule_on_known_values():
    idx = slices.medgemma_uniform(200, 85)
    assert len(idx) == 85
    assert idx[0] == 2  # round(1/85 * 199) = round(2.34)
    assert idx[-1] == 199  # i = 85 always lands on the last slice
    assert idx == sorted(set(idx))
    assert slices.medgemma_uniform(60, 85) == list(range(60))  # short volume: every slice


def test_fraction_range_viewer_examples():
    assert slices.fraction_range(100, 16)[0] == 0 and slices.fraction_range(100, 16)[-1] == 99
    assert slices.fraction_range(101, 3, 0.4, 0.7) == [40, 55, 70]


def test_selector_registry_has_new_samplers():
    sel = slices.resolve_ct_selector({"fn": "medgemma_uniform", "k": 50})
    assert len(sel(300)) == 50


# -- nifti: radiology display from any stored orientation ---------------------------------------

def _write(tmp_path, name, data, affine):
    p = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(p))
    return p


def _volume_with_markers(order, signs, shape=(6, 5, 4)):
    """A volume stored with world axes permuted (``order``) and flipped (``signs``), carrying
    marker 100 at the patient's right-anterior-superior corner and 200 at left-posterior-inferior.

    World is nibabel's RAS+ (coordinates grow toward Right, Anterior, Superior), so the
    right-anterior-superior corner is the voxel with the largest world x, y and z."""
    world_shape = shape  # (x, y, z) extent in voxels
    stored_shape = tuple(world_shape[o] for o in order)
    data = np.zeros(stored_shape, dtype=np.int16)
    affine = np.zeros((4, 4))
    affine[3, 3] = 1
    for stored_axis, (world_axis, sign) in enumerate(zip(order, signs)):
        affine[world_axis, stored_axis] = sign
    # world corner -> stored index: world coord = sign * index, so max world needs index n-1 if
    # sign = +1, else index 0.
    def stored_index(world_corner_max):
        idx = []
        for stored_axis, (world_axis, sign) in enumerate(zip(order, signs)):
            n = stored_shape[stored_axis]
            want_max = world_corner_max[world_axis]
            idx.append(n - 1 if (want_max == (sign > 0)) else 0)
        return tuple(idx)
    data[stored_index((True, True, True))] = 100   # right, anterior, superior
    data[stored_index((False, False, False))] = 200  # left, posterior, inferior
    return data, affine


@pytest.mark.parametrize("order,signs", [
    ((0, 1, 2), (1, 1, 1)),      # RAS (the feb26 store)
    ((0, 1, 2), (-1, -1, 1)),    # LPS (DICOM order)
    ((2, 1, 0), (1, -1, -1)),    # scan axis stored first, flipped
    ((1, 0, 2), (-1, 1, -1)),    # x and y swapped, feet first
])
def test_axial_stack_puts_anatomy_where_radiologists_expect(tmp_path, order, signs):
    data, affine = _volume_with_markers(order, signs)
    stack = nifti.AxialStack.open(_write(tmp_path, "v", data, affine))
    assert len(stack) == 4
    top, bottom = stack[0], stack[len(stack) - 1]
    assert top.shape == (5, 6)  # rows = anterior-posterior extent, columns = right-left extent
    # Head slice: right-anterior marker at the top-left (patient's right on the image's left).
    assert top[0, 0] == 100 and (top == 100).sum() == 1
    # Feet slice: left-posterior marker at the bottom-right.
    assert bottom[-1, -1] == 200 and (bottom == 200).sum() == 1


def test_axial_stack_on_ras_is_antitranspose_of_raw_slice(tmp_path):
    """What changes for the eval set vs the raw slices earlier runs used, on a RAS store."""
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(7, 5, 3)).astype(np.int16)
    img = nifti.load(_write(tmp_path, "r", data, np.diag([0.7, 0.8, 2.5, 1.0])))
    stack = nifti.AxialStack(img)
    for k in range(3):
        raw = np.asarray(img.dataobj[:, :, 2 - k], dtype=np.float64)
        assert np.array_equal(stack[k], raw[::-1, ::-1].T)
        assert stack.voxel_index(k) == 2 - k
    assert stack.spacing_mm == pytest.approx((0.7, 0.8, 2.5))


def test_ct_path_and_blob():
    assert nifti.ct_blob(None, "s") is None
    assert nifti.ct_blob("a", "b")[1] == "a__b.nii.gz"
    assert str(nifti.ct_path("a", "b", mount_root="/m")).endswith("su-vista-uscentral1/" + nifti.DEFAULT_CT_SNAPSHOT_PREFIX + "/a__b.nii.gz")
