"""CT NIfTI reading: the one home for finding a scan on the mount and reading its axial slices.

Replaces two readers that oriented the volume differently: main's dataset loader (``vqa_dataset``),
which hands models the stored array as is, and the agent's data builder
(``optimizer/died-2yr-ct/scripts/build_data.py``), which reorients to RAS and draws slices the
radiology way. Both views are here:

  * ``raw`` (``load(path).dataobj``): the stored array, sliced ``[:, :, idx]``. What every run before
    the eval set saw. The feb26 store is RAS (40 of 40 sampled 2026-10-27), so a raw slice has the
    patient's left at the top and anterior on the right, and index 0 is the lowest slice.
  * ``AxialStack``: radiology display. Patient's right on the image's left, anterior at the top,
    slices ordered head to feet. Works for any stored orientation (read from the affine) and reads
    one plane at a time, so a 2,000-slice CT is never loaded whole.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CT_SNAPSHOT_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/feb26"
CT_BUCKET = "su-vista-uscentral1"


def ct_blob(study_uid, series_uid, prefix=DEFAULT_CT_SNAPSHOT_PREFIX):
    """(blob_path, filename) for a (study, series) UID pair, or None when either is null/empty.

    The store's layout is ``{prefix}/{study}__{series}.nii.gz`` in bucket ``su-vista-uscentral1``.
    """
    import pandas as pd
    if study_uid is None or series_uid is None:
        return None
    if pd.isna(study_uid) or pd.isna(series_uid):
        return None
    study, series = str(study_uid).strip(), str(series_uid).strip()
    if not study or not series:
        return None
    filename = f"{study}__{series}.nii.gz"
    return f"{prefix}/{filename}", filename


def ct_path(study_uid, series_uid, mount_root="/mnt", prefix=DEFAULT_CT_SNAPSHOT_PREFIX, bucket=CT_BUCKET):
    """The scan's path under the gcsfuse mount (``<mount_root>/<bucket>/<blob>``), or None."""
    b = ct_blob(study_uid, series_uid, prefix)
    return None if b is None else Path(mount_root) / bucket / b[0]


def load(path):
    """Lazy nibabel image: nothing is read until a slice is taken from ``.dataobj``."""
    import nibabel as nib
    return nib.load(str(path))


class AxialStack:
    """Axial slices of one CT in radiology display, head to feet, as float64 Hounsfield units.

    ``stack[k]`` is slice k (0 = most superior); rows run anterior to posterior, columns run the
    patient's right to left. ``voxel_index(k)`` is the stored array index that slice came from.
    """

    def __init__(self, img):
        import nibabel as nib
        self.img = img
        ornt = nib.io_orientation(img.affine)  # per stored axis: (RAS axis it runs along, +1/-1)
        self._axis_of = {int(ras): i for i, (ras, _) in enumerate(ornt)}
        self._flip = {int(ras): int(f) for ras, f in ornt}
        self._ndim = len(img.shape)
        self.n = int(img.shape[self._axis_of[2]])
        zooms = img.header.get_zooms()
        self.spacing_mm = tuple(float(zooms[self._axis_of[a]]) for a in (0, 1, 2))

    @classmethod
    def open(cls, path):
        return cls(load(path))

    def __len__(self):
        return self.n

    def voxel_index(self, k: int) -> int:
        """Stored index along the scan axis for display slice ``k`` (0 = most superior)."""
        cz = self.n - 1 - k  # index counting upward (toward superior)
        return cz if self._flip[2] == 1 else self.n - 1 - cz

    def __getitem__(self, k: int) -> np.ndarray:
        if not 0 <= k < self.n:
            raise IndexError(k)
        zax = self._axis_of[2]
        idx = [slice(None)] * 3
        idx[zax] = self.voxel_index(k)
        if self._ndim == 4:
            idx.append(0)
        plane = np.asarray(self.img.dataobj[tuple(idx)], dtype=np.float64)
        # plane's axes are the two stored in-plane axes, in stored order; make them (x, y) = (R, A).
        a, _ = [i for i in range(3) if i != zax]
        canon = plane if a == self._axis_of[0] else plane.T
        if self._flip[0] == -1:
            canon = canon[::-1, :]
        if self._flip[1] == -1:
            canon = canon[:, ::-1]
        # Display: rows anterior -> posterior (y descending), columns right -> left (x descending).
        return np.ascontiguousarray(canon[::-1, ::-1].T)
