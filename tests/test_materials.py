"""Material models and the wavelength path through the cross section.

The PyOptik-backed tests skip when the database snapshot is not installed, so
the suite still runs on a machine that has never fetched it.  The offline
Sellmeier formulas and the coercion/fingerprint logic are always exercised.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.fde import FullEtchStrip  # noqa: E402
from em_simulation.fde import materials as M  # noqa: E402

WL = 1.55e-6


def _has_database():
    try:
        M.get_catalog()
        return True
    except M.PyOptikUnavailable:
        return False


needs_database = pytest.mark.skipif(
    not _has_database(), reason="PyOptik material snapshot not installed"
)


# ------------------------------------------------------------- offline models


def test_builtin_silica_matches_malitson():
    """Malitson 1965 gives n = 1.44402 for fused silica at 1550 nm."""
    silica = M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["SiO2"])
    assert abs(silica.n(WL) - 1.44402) < 1e-5


def test_builtin_silicon_is_close_to_the_reference_value():
    """Salzberg 1957 puts Si at 3.4777, about 2e-3 above the Li 1980 value.

    The two are different fits to different measurements; the gap is why
    swapping between them has to invalidate a dataset rather than pass
    unnoticed.
    """
    silicon = M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["Si"])
    assert abs(silicon.n(WL) - 3.4777) < 1e-3
    assert abs(silicon.n(WL) - 3.4757) > 1e-4


def test_sellmeier_is_dispersive():
    silicon = M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["Si"])
    assert silicon.n(1.31e-6) > silicon.n(1.55e-6) > silicon.n(1.625e-6)


def test_sellmeier_accepts_arrays():
    silica = M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["SiO2"])
    values = silica.n(np.array([1.31e-6, 1.55e-6]))
    assert values.shape == (2,)


def test_constant_index_is_flat():
    material = M.ConstantIndex(3.4757)
    assert material.n(1.31e-6) == pytest.approx(material.n(1.65e-6))
    assert material.n(WL) == pytest.approx(3.4757)


# ------------------------------------------------------------------ coercion


def test_as_material_accepts_numbers_and_materials():
    assert isinstance(M.as_material(1.444), M.ConstantIndex)
    existing = M.ConstantIndex(2.0)
    assert M.as_material(existing) is existing
    assert isinstance(M.as_material("main/Si/Li-293K"), M.PyOptikMaterial)


def test_as_material_rejects_nonsense():
    with pytest.raises(TypeError):
        M.as_material(["not", "a", "material"])


def test_fingerprints_distinguish_models():
    """Two index models must never share a fingerprint."""
    fingerprints = {
        M.ConstantIndex(3.4757).fingerprint(),
        M.ConstantIndex(3.4777).fingerprint(),
        M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["Si"]).fingerprint(),
        M.PyOptikMaterial("main/Si/Li-293K").fingerprint(),
        M.PyOptikMaterial("main/Si/Salzberg").fingerprint(),
    }
    assert len(fingerprints) == 5


# ------------------------------------------------------------------ database


@needs_database
def test_database_reproduces_the_shipped_constants():
    """The numbers this project shipped as literals come from these entries."""
    assert M.PyOptikMaterial(M.SI_PAGE).n(WL) == pytest.approx(3.4757, abs=1e-4)
    assert M.PyOptikMaterial(M.SIO2_PAGE).n(WL) == pytest.approx(1.4440, abs=1e-4)


@needs_database
def test_database_reports_a_validity_range_in_metres():
    low, high = M.PyOptikMaterial(M.SI_PAGE).validity_range()
    assert 1e-7 < low < 1e-5 and low < high
    assert low <= WL <= high


@needs_database
def test_silicon_dispersion_is_significant_across_the_bands():
    """Si moves by ~0.04 in index from O to L band - not a rounding error."""
    silicon = M.silicon()
    span = silicon.n(1.26e-6) - silicon.n(1.625e-6)
    assert span > 0.03


@needs_database
def test_absorbing_material_is_flagged():
    """Gold must not pass quietly: the EME layer assumes a real index."""
    gold = M.PyOptikMaterial("main/Au/Johnson")
    with pytest.warns(RuntimeWarning, match="absorbing material"):
        offenders = M.check_lossless([gold], WL)
    assert offenders and offenders[0][1] > 1


def test_transparent_materials_are_not_flagged():
    silica = M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["SiO2"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert M.check_lossless([silica], WL) == []


# ------------------------------------------------------- cross-section wiring


def test_cross_section_still_accepts_plain_numbers():
    """The pre-materials keyword form must keep working."""
    cross_section = FullEtchStrip(core_index=3.4757, clad_index=1.444)
    assert cross_section.core_index == pytest.approx(3.4757)
    assert cross_section.clad_index == pytest.approx(1.444)
    assert cross_section.cladding_index == pytest.approx(1.444)


def test_cross_section_rejects_both_spellings_of_one_layer():
    with pytest.raises(TypeError):
        FullEtchStrip(core=3.4757, core_index=3.4757)


def test_index_profile_follows_the_requested_wavelength():
    """A dispersive stack must respond to `wavelength` in the parameter dict."""
    cross_section = FullEtchStrip(
        core=M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["Si"]),
        cladding=M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["SiO2"]),
    )
    x = np.linspace(-1.6e-6, 1.6e-6, 81)
    y = np.linspace(-0.8e-6, 0.8e-6, 41)
    params = {"top_width": 500e-9, "curvature": 0.0}

    at_1550 = cross_section.index(x, y, {**params, "wavelength": 1.55e-6})
    at_1310 = cross_section.index(x, y, {**params, "wavelength": 1.31e-6})
    assert at_1310.max() > at_1550.max()

    # No wavelength given falls back to the cross section's reference.
    cross_section.reference_wavelength = 1.31e-6
    assert np.allclose(cross_section.index(x, y, params), at_1310)


def test_constant_stack_ignores_wavelength():
    cross_section = FullEtchStrip(core_index=3.4757, clad_index=1.444)
    x = np.linspace(-1.6e-6, 1.6e-6, 41)
    y = np.linspace(-0.8e-6, 0.8e-6, 21)
    params = {"top_width": 500e-9, "curvature": 0.0}
    a = cross_section.index(x, y, {**params, "wavelength": 1.55e-6})
    b = cross_section.index(x, y, {**params, "wavelength": 1.31e-6})
    assert np.allclose(a, b)


def test_cross_section_fingerprint_tracks_its_materials():
    constant = FullEtchStrip(core_index=3.4757, clad_index=1.444)
    sellmeier = FullEtchStrip(
        core=M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["Si"]),
        cladding=M.SellmeierMaterial(**M.SELLMEIER_COEFFICIENTS["SiO2"]),
    )
    assert constant.fingerprint() != sellmeier.fingerprint()
    assert constant.fingerprint() == FullEtchStrip(
        core_index=3.4757, clad_index=1.444
    ).fingerprint()
