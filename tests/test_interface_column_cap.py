"""The unit-power column cap on interface scattering matrices
(``single_eme.cap_columns``, ``SingleEME.INTERFACE_COLUMN_CAP``)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.propagator.single_propagator.single_eme import SingleEME, cap_columns  # noqa: E402


def test_cap_scales_only_columns_above_unit_power():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((3, 6, 6)) + 1j * rng.standard_normal((3, 6, 6))
    S[:, :, :2] *= 0.1                                  # two columns well below unit power
    C = cap_columns(S)
    power = np.sum(np.abs(C) ** 2, axis=1)
    assert np.all(power <= 1.0 + 1e-12)
    assert np.allclose(C[:, :, :2], S[:, :, :2])         # untouched
    # a capped column keeps its direction
    j = 3
    ratio = C[:, :, j] / S[:, :, j]
    assert np.allclose(ratio, ratio[:, :1])


def test_cap_leaves_a_unitary_matrix_alone():
    q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((8, 8)) + 0j)
    assert np.allclose(cap_columns(q[None]), q[None])


def test_auto_means_lossy_only():
    class Fake(SingleEME):
        def __init__(self, lossless):
            self._lossless = lossless
    assert Fake(True)._column_cap_enabled() is False
    assert Fake(False)._column_cap_enabled() is True
    Fake.INTERFACE_COLUMN_CAP = False
    assert Fake(False)._column_cap_enabled() is False
    Fake.INTERFACE_COLUMN_CAP = "auto"
