"""A cached dataset must refuse to serve a different problem.

The pickles hold effective indices and overlap matrices for one specific mode
problem.  Change the index model, the wavelength, the mesh, the window or the
grid and every stored number belongs to a different problem - but the files
still load and the results still look plausible.  ``dataset_identity`` writes
the identity next to the cache and checks it on open.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.data_updater.dataset_identity import (  # noqa: E402
    FINGERPRINT_FILE,
    DatasetIdentityError,
    fingerprint,
    verify,
)


class FakeInfo:
    """Minimal stand-in for a dataset's ``DatasetInfo``."""

    def __init__(self, wavelength=1.55e-6, modes=6, widths=None):
        self._wavelength = wavelength
        self._modes = modes
        self._widths = widths if widths is not None else np.linspace(0.4e-6, 1.5e-6, 56)

    def get_wavelength(self):
        return self._wavelength

    def get_mode_numbers(self):
        return self._modes

    def get_parameter_names(self):
        return ["top_width", "curvature"]

    def get_parameter_grid(self):
        return {
            "top_width": self._widths,
            "curvature": np.linspace(-2e5, 2e5, 81),
        }


def test_first_open_writes_the_fingerprint(tmp_path):
    identity = fingerprint(FakeInfo())
    verify(str(tmp_path), identity, has_cached_data=False)

    written = json.loads((tmp_path / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    assert written == identity


def test_matching_identity_passes(tmp_path):
    identity = fingerprint(FakeInfo())
    verify(str(tmp_path), identity, has_cached_data=False)
    verify(str(tmp_path), identity, has_cached_data=True)


def test_changed_wavelength_is_rejected(tmp_path):
    verify(str(tmp_path), fingerprint(FakeInfo()), has_cached_data=False)
    with pytest.raises(DatasetIdentityError, match="wavelength_m"):
        verify(
            str(tmp_path),
            fingerprint(FakeInfo(wavelength=1.31e-6)),
            has_cached_data=True,
        )


def test_changed_mode_count_is_rejected(tmp_path):
    verify(str(tmp_path), fingerprint(FakeInfo()), has_cached_data=False)
    with pytest.raises(DatasetIdentityError, match="mode_numbers"):
        verify(str(tmp_path), fingerprint(FakeInfo(modes=8)), has_cached_data=True)


def test_a_refined_grid_is_rejected(tmp_path):
    """Same endpoints, finer step - the classic silent mismatch."""
    verify(str(tmp_path), fingerprint(FakeInfo()), has_cached_data=False)
    finer = FakeInfo(widths=np.linspace(0.4e-6, 1.5e-6, 111))
    with pytest.raises(DatasetIdentityError, match="parameter_grid"):
        verify(str(tmp_path), fingerprint(finer), has_cached_data=True)


def test_an_empty_cache_is_adopted_rather_than_rejected(tmp_path):
    """Nothing is at risk when there is no cached data, so just re-stamp it."""
    verify(str(tmp_path), fingerprint(FakeInfo()), has_cached_data=False)
    changed = fingerprint(FakeInfo(wavelength=1.31e-6))
    verify(str(tmp_path), changed, has_cached_data=False)

    written = json.loads((tmp_path / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    assert written["wavelength_m"] == pytest.approx(1.31e-6)


def test_material_change_is_caught_through_the_cross_section(tmp_path):
    """The case this was written for: a swapped index model."""
    from em_simulation.fde import FullEtchStrip
    from em_simulation.fde.materials import SELLMEIER_COEFFICIENTS, SellmeierMaterial

    class FakeBackend:
        def __init__(self, cross_section):
            self.cross_section = cross_section
            self.window = (1.6e-6, -0.8e-6, 0.8e-6)

        def solve_grid(self):
            return np.zeros(160), np.zeros(80)

    info = FakeInfo()
    constant = FakeBackend(FullEtchStrip(core_index=3.4757, clad_index=1.444))
    verify(str(tmp_path), fingerprint(info, constant), has_cached_data=False)

    dispersive = FakeBackend(
        FullEtchStrip(
            core=SellmeierMaterial(**SELLMEIER_COEFFICIENTS["Si"]),
            cladding=SellmeierMaterial(**SELLMEIER_COEFFICIENTS["SiO2"]),
        )
    )
    with pytest.raises(DatasetIdentityError, match="cross_section"):
        verify(str(tmp_path), fingerprint(info, dispersive), has_cached_data=True)


def test_missing_fingerprint_file_is_adopted(tmp_path):
    """A dataset from before fingerprinting existed must still open."""
    identity = fingerprint(FakeInfo())
    verify(str(tmp_path), identity, has_cached_data=True)
    assert os.path.exists(tmp_path / FINGERPRINT_FILE)
