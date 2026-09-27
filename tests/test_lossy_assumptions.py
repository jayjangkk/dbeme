"""Phase 0 of the PML task: removing the "the medium is real" assumptions.

Four places hard-code a real index.  All four are no-ops on the datasets this
project ships, which is exactly why they are dangerous: the first lossy run
would report zero loss and look healthy.  These tests pin both halves - that
the lossy branches are *equivalent* on real-index data, and that they actually
differ once a mode is complex.

See ``tasks/02_pml_backend.md`` and ``CLAUDE.md`` 5.16a / 5.16b.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.fde.base import FDEBackend, ModeData  # noqa: E402
from dbeme.fde.materials import ConstantIndex, silica, silicon  # noqa: E402
from dbeme.propagator.propagator import (  # noqa: E402
    LOSSY_NEFF_TOLERANCE,
    _reject_unitary_projection_on_lossy_modes,
)

WL = 1.55e-6


# ------------------------------------------------------------------ fixtures


def _mode_data(n_modes=2, nx=8, ny=6, lossy=False, seed=0):
    """A ModeData with the reality structure a lossless solve produces.

    ``_pin_gauge`` rotates each mode so its transverse field is real at the
    peak, which leaves the transverse components real and the longitudinal one
    imaginary.  That is the structure under which time reversal and the
    reciprocal backward partner coincide.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1e-6, 1e-6, nx)
    y = np.linspace(-0.5e-6, 0.5e-6, ny)
    E = np.zeros((n_modes, 3, nx, ny), dtype=complex)
    H = np.zeros((n_modes, 3, nx, ny), dtype=complex)
    shape = (n_modes, nx, ny)
    # The power integral has to come out *positive real*, as it does for a
    # power-normalised solve.  Left arbitrary it can be negative, and then
    # sqrt() is imaginary, the normalisation rotates every component by 90
    # degrees, and the reality structure this test is about is destroyed.
    E[:, 0] = 1.0 + 0.5 * rng.random(shape)
    H[:, 1] = 1.0 + 0.5 * rng.random(shape)
    E[:, 1] = 0.1 * rng.random(shape)
    H[:, 0] = -0.1 * rng.random(shape)
    E[:, 2] = 1j * rng.normal(size=shape)
    H[:, 2] = 1j * rng.normal(size=shape)
    # Make the modes *distinct*.  Built from the same positive random field
    # they overlap by ~1.0, i.e. they are one mode written twice - a set with
    # no biorthogonal basis at all, which `assemble` now (correctly) rejects.
    # A sign flip across x gives the second mode odd parity, as TE1 has.
    if n_modes > 1:
        E[1] = E[1] * np.sign(x)[None, :, None]
        H[1] = H[1] * np.sign(x)[None, :, None]
    neff = np.array([2.4, 1.8][:n_modes], dtype=complex)
    if lossy:
        neff = neff + 1j * np.array([1e-4, 3e-5][:n_modes])
    return ModeData(x=x, y=y, E=E, H=H, neff=neff,
                    TE_pol=np.array([0.9, 0.1][:n_modes]))


# --------------------------------------------- 5.16a backward construction


def test_the_two_backward_bases_agree_on_a_real_index_mode():
    """The whole reason Phase 0 is safe.

    Time reversal (``E- = conj(E+)``) and the reciprocal partner
    (``E_t, -E_z``, ``-H_t, H_z``, no conjugation) are the same thing when the
    transverse components are real and the longitudinal one is imaginary.  If
    this ever fails, every existing dataset changed meaning.
    """
    modes = [_mode_data()]
    _, _, neff_a, _, E_a, H_a = assemble(modes, 2, 2, lossless=True)
    _, _, neff_b, _, E_b, H_b = assemble(modes, 2, 2, lossless=False)

    assert E_a == pytest.approx(E_b, abs=1e-6)
    assert H_a == pytest.approx(H_b, abs=1e-6)
    assert neff_a == pytest.approx(neff_b, abs=1e-9)


def test_the_two_backward_bases_disagree_once_the_mode_is_complex():
    """And the reason it matters: conjugation turns loss into gain."""
    E = _mode_data().E.copy()
    H = _mode_data().H.copy()
    # A genuinely complex transverse field, as a leaky or metallic mode has.
    E[:, 0] = E[:, 0] * np.exp(0.4j)
    modes = [ModeData(x=_mode_data().x, y=_mode_data().y, E=E, H=H,
                      neff=np.array([2.4, 1.8], dtype=complex),
                      TE_pol=np.array([0.9, 0.1]))]

    _, _, _, _, E_a, _ = assemble(modes, 2, 2, lossless=True)
    _, _, _, _, E_b, _ = assemble(modes, 2, 2, lossless=False)
    assert not np.allclose(E_a[:, 2:], E_b[:, 2:], atol=1e-6)


def test_backward_modes_flip_only_the_longitudinal_component():
    modes = [_mode_data()]
    _, _, _, _, E, H = assemble(modes, 2, 2, lossless=False)
    forward, backward = E[:, :2], E[:, 2:]
    assert backward[:, :, 0] == pytest.approx(forward[:, :, 0])
    assert backward[:, :, 1] == pytest.approx(forward[:, :, 1])
    assert backward[:, :, 2] == pytest.approx(-forward[:, :, 2])
    assert H[:, 2:, 0] == pytest.approx(-H[:, :2, 0])
    assert H[:, 2:, 2] == pytest.approx(H[:, :2, 2])


# ----------------------------------------------------- 5.16b the gain clip


def test_negative_imaginary_neff_is_clipped_only_when_lossless():
    modes = [_mode_data(lossy=True)]
    modes[0].neff = np.array([2.4 - 1e-4j, 1.8 - 3e-5j])

    _, _, clipped, _, _, _ = assemble(modes, 2, 2, lossless=True)
    _, _, kept, _, _, _ = assemble(modes, 2, 2, lossless=False)

    assert np.imag(clipped[0, :2]) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert np.imag(kept[0, :2]) == pytest.approx([-1e-4, -3e-5], rel=1e-3)


def test_positive_imaginary_neff_survives_either_way():
    modes = [_mode_data(lossy=True)]
    for lossless in (True, False):
        _, _, neff, _, _, _ = assemble(modes, 2, 2, lossless=lossless)
        assert np.imag(neff[0, :2]) == pytest.approx([1e-4, 3e-5], rel=1e-3)


# ------------------------------------------------------ the overlap metric


def test_overlap_metric_survives_complex_stretched_coordinates():
    """Under a PML the integration measure is complex and must stay complex.

    The old code cast the coordinates with ``dtype=float``, which throws away
    the stretch.  The symptom would have been a mildly non-unitary S-matrix
    that looks exactly like truncation error.
    """
    modes = [_mode_data()]
    _, _, _, _, E, H = assemble(modes, 2, 2)
    x = np.linspace(-1e-6, 1e-6, 8).astype(complex)
    y = np.linspace(-0.5e-6, 0.5e-6, 6).astype(complex)
    x[:2] = x[:2] * (1 + 2j)  # a PML layer on one edge

    stretched = overlap_matrix(E[0], H[0], x, y, 2)
    plain = overlap_matrix(E[0], H[0], np.real(x), np.real(y), 2)

    assert np.iscomplexobj(stretched)
    assert not np.allclose(stretched, plain), "the stretch was discarded"


def test_real_coordinates_are_unchanged_by_dropping_the_cast():
    modes = [_mode_data()]
    _, _, _, _, E, H = assemble(modes, 2, 2)
    x = np.linspace(-1e-6, 1e-6, 8)
    y = np.linspace(-0.5e-6, 0.5e-6, 6)
    assert overlap_matrix(E[0], H[0], x, y, 2) == pytest.approx(
        overlap_matrix(E[0], H[0], x.astype(complex), y.astype(complex), 2)
    )


# --------------------------------------------------- force_unitary guarding


def test_unitary_projection_is_refused_on_lossy_modes():
    neff = np.array([[2.4 + 1e-5j, 1.8 + 0j]])
    with pytest.raises(ValueError, match="force_unitary=True with lossy modes"):
        _reject_unitary_projection_on_lossy_modes(neff, True)


def test_unitary_projection_is_allowed_on_a_real_index_solve():
    neff = np.array([[2.4 + 1e-17j, 1.8 - 1e-17j]])
    _reject_unitary_projection_on_lossy_modes(neff, True)  # must not raise


def test_the_guard_only_applies_when_projection_is_requested():
    neff = np.array([[2.4 + 1e-3j, 1.8 + 0j]])
    _reject_unitary_projection_on_lossy_modes(neff, False)  # must not raise


def test_the_tolerance_sits_below_anything_physical():
    """1 dB/cm is Im(n_eff) = 2.8e-6 at 1550 nm."""
    one_db_per_cm = 1.0 / (4.343 * 2 * (2 * np.pi / WL) * 0.01)
    assert LOSSY_NEFF_TOLERANCE < one_db_per_cm / 100


# ----------------------------------------------------------- the flag itself


def test_the_abc_defaults_to_lossless():
    assert FDEBackend.lossless.fget(None) is True


def test_emepy_reports_lossless_for_a_transparent_stack():
    backend = EmepyFDE(
        cross_section=FullEtchStrip(
            thickness=220e-9,
            core=silicon(out_of_range="raise"),
            cladding=silica(out_of_range="raise"),
        ),
        num_modes=2,
        wavelength=WL,
        window=(2.0e-6, -0.8e-6, 0.8e-6),
        mesh=40,
    )
    assert backend.lossless is True


def test_emepy_reports_lossy_for_an_absorbing_stack():
    with pytest.warns(RuntimeWarning, match="absorbing material"):
        backend = EmepyFDE(
            cross_section=FullEtchStrip(
                thickness=220e-9,
                core=ConstantIndex(3.48 + 0.02j, name="lossy Si"),
                cladding=silica(out_of_range="raise"),
            ),
            num_modes=2,
            wavelength=WL,
            window=(2.0e-6, -0.8e-6, 0.8e-6),
            mesh=40,
        )
    assert backend.lossless is False


def test_the_real_solver_produces_the_reality_structure_the_switch_assumes():
    """Ground the equivalence claim in the actual backend, not the fixture.

    ``test_the_two_backward_bases_agree_on_a_real_index_mode`` only means
    something if a real solve really does come back with real transverse
    components and an imaginary longitudinal one - that is what ``_pin_gauge``
    plus power normalisation is for.  Check it, then check the two backward
    bases agree on those fields.
    """
    backend = EmepyFDE(
        cross_section=FullEtchStrip(
            thickness=220e-9,
            core=silicon(out_of_range="raise"),
            cladding=silica(out_of_range="raise"),
        ),
        num_modes=2,
        wavelength=WL,
        window=(2.0e-6, -0.8e-6, 0.8e-6),
        mesh=60,
    )
    modes = [backend.solve((500e-9, 0.0))]

    _, _, _, _, E, H = assemble(modes, 2, 2, lossless=True)
    forward_E, forward_H = E[0, :2], H[0, :2]
    for field in (forward_E, forward_H):
        transverse = np.abs(np.imag(field[:, :2])).max()
        longitudinal = np.abs(np.real(field[:, 2])).max()
        scale = np.abs(field).max()
        assert transverse / scale < 1e-3, "transverse components are not real"
        assert longitudinal / scale < 1e-3, "longitudinal component is not imaginary"

    _, _, _, _, E_b, H_b = assemble(modes, 2, 2, lossless=False)
    assert E_b == pytest.approx(E, rel=1e-3, abs=1e-3 * np.abs(E).max())
    assert H_b == pytest.approx(H, rel=1e-3, abs=1e-3 * np.abs(H).max())


# --------------------------------------------------- radiation-mode mask


def test_radiation_mask_keeps_lossy_guided_modes():
    """The 100 dB/cm rule is a lossless-model heuristic.  A Si wire beside
    gold (Im ~ 5e-3) and a gap plasmon (2e-2) are guided modes of a lossy
    basis; only the cutoff may reject them there."""
    from dbeme.geometry.geometry import (
        RADIATION_IMAG_NEFF, radiation_mode_mask,
    )

    neffs = np.array([2.20 + 4.5e-3j, 1.47 + 2.0e-2j, 1.30 + 1.0e-2j, 2.40 + 0.0j])
    lossy = radiation_mode_mask(neffs, 1.444, lossless=False)
    assert lossy.tolist() == [False, False, True, False]
    lossless = radiation_mode_mask(neffs, 1.444, lossless=True)
    assert lossless.tolist() == [True, True, True, False]
    assert radiation_mode_mask(np.array([2.0 + 0.5 * RADIATION_IMAG_NEFF * 1j]), 1.444)[0] is np.False_


def test_radiation_mask_asks_the_dataset_whether_it_is_lossless():
    from types import SimpleNamespace

    from dbeme.geometry.geometry import _data_is_lossless

    assert _data_is_lossless(SimpleNamespace(_is_lossless=lambda: False)) is False
    assert _data_is_lossless(SimpleNamespace(_is_lossless=lambda: True)) is True
    assert _data_is_lossless(SimpleNamespace()) is True, "no probe means the old lossless default"
