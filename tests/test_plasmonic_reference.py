"""The plasmonic ruler: gold, the single-interface SPP, and the MIM gap plasmon.

These are the closed-form answers the lossy solver is checked against
(``dbeme/reference/plasmonic.py``).  Nothing here solves a mode.

Sign convention throughout the project: fields go as ``e^{i beta z}``, so an
absorbing metal has ``Im(n) > 0`` and a lossy mode has ``Im(n_eff) > 0``.
Getting either sign backwards turns loss into gain silently.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde.materials import gold  # noqa: E402
from dbeme.reference.plasmonic import (  # noqa: E402
    MIMSlab,
    attenuation_db_per_um,
    propagation_length,
    spp_index,
)

WL = 1.55e-6


@pytest.fixture(scope="module")
def eps_au():
    return complex(gold(out_of_range="raise").index(WL)) ** 2


# ------------------------------------------------------------------- gold


def test_gold_is_johnson_and_christy_at_1550():
    n = complex(gold(out_of_range="raise").index(WL))
    assert n.real == pytest.approx(0.524, abs=0.01)
    assert n.imag == pytest.approx(10.74, abs=0.05)


def test_gold_absorbs_in_this_projects_sign_convention(eps_au):
    """``Im(eps) > 0`` is loss for ``e^{i beta z}``; the negative sign is gain."""
    assert eps_au.real < -100
    assert eps_au.imag > 0


# -------------------------------------------------------------------- SPP


def test_spp_index_formula(eps_au):
    n = spp_index(eps_au, 1.0)
    expected = np.sqrt(eps_au * 1.0 / (eps_au + 1.0))
    assert n == pytest.approx(expected) or n == pytest.approx(-expected)
    assert n.real > 1.0 and n.imag > 0


def test_spp_lies_just_above_the_dielectric_line(eps_au):
    """A weakly bound SPP on a good metal: n barely above n_d, long-lived."""
    n = spp_index(eps_au, 1.0)
    assert 1.0 < n.real < 1.02
    assert propagation_length(n, WL) > 100e-6


def test_spp_in_silica_is_bound_more_tightly(eps_au):
    assert spp_index(eps_au, 1.444**2).real > spp_index(eps_au, 1.0).real


# ---------------------------------------------------------------- MIM slab


@pytest.fixture(scope="module")
def slab(eps_au):
    return MIMSlab(WL, eps_au, 1.0)


@pytest.fixture(scope="module")
def sweep(slab):
    gaps = np.array([2000, 500, 200, 100, 50, 25, 20]) * 1e-9
    return gaps, slab.sweep(gaps)


def test_roots_actually_satisfy_the_dispersion(slab, sweep):
    gaps, neff = sweep
    for gap, value in zip(gaps, neff):
        residual = abs(slab.dispersion(slab.k0 * value, gap))
        assert residual < 1e-5 * slab.k0


def test_wide_gap_limit_is_the_single_interface_spp(slab, eps_au):
    """Two far-apart walls are two independent SPPs."""
    wide = slab.solve(10e-6)
    assert wide == pytest.approx(spp_index(eps_au, 1.0), abs=2e-3)


def test_gap_plasmon_index_rises_as_the_gap_closes(sweep):
    _, neff = sweep
    assert np.all(np.diff(neff.real) > 0)


def test_gap_plasmon_loss_rises_as_the_gap_closes(sweep):
    """More of the field in the metal, more absorption."""
    _, neff = sweep
    assert np.all(neff.imag > 0)
    assert np.all(np.diff(neff.imag) > 0)


def test_the_deep_subwavelength_slot_matches_the_ntt_figure(slab):
    """NTT Technical Review 16(7): 'loss can rise to about 1 dB/um for cores
    in the deep-subwavelength regime'.  The 20 nm air core of the fabricated
    device is exactly that regime."""
    n = slab.solve(20e-9)
    assert 0.7 < attenuation_db_per_um(n, WL) < 1.6
    assert 1.6 < n.real < 2.1


def test_the_antisymmetric_branch_has_a_cutoff_and_the_fundamental_does_not(slab):
    """Two walls far apart carry both parities; a 20 nm slot carries only the
    even-E_x gap plasmon.  The odd mode is cut off there, and the secant has
    nothing to converge to - which is the physics, not a solver failure."""
    wide = 2e-6
    even = slab.solve(wide, branch="fundamental")
    odd = slab.solve(wide, neff_guess=even, branch="antisymmetric")
    # At 2 um the odd branch sits at n ~ 0.95, below the air line: it is
    # already leaky.  The point is only that it is a different solution.
    assert abs(odd - even) > 1e-3
    assert slab.solve(20e-9, branch="fundamental").real > 1.5
    with pytest.raises(RuntimeError):
        slab.solve(20e-9, neff_guess=even, branch="antisymmetric")


def test_an_unknown_branch_is_rejected(slab):
    with pytest.raises(ValueError, match="branch"):
        slab.solve(100e-9, branch="even")


# ------------------------------------------------------------ conversions


def test_propagation_length_and_attenuation_are_consistent():
    n = 1.8 + 0.03j
    alpha_db_per_m = attenuation_db_per_um(n, WL) * 1e6
    # power e-folding over L_prop is 4.343 dB
    assert alpha_db_per_m * propagation_length(n, WL) == pytest.approx(4.343, rel=1e-3)
