"""The Phase 1 ruler: bend loss from Airy functions, and a PML that matches it.

``tasks/02_pml_backend.md`` Phase 1 builds a reference *before* any backend, so
that a PML implementation can be checked against something that shares none of
its machinery.  These tests pin the reference itself (against WKB, which knows
nothing about Airy functions), then pin the finite-difference PML against the
reference.

Everything here is one-dimensional and runs in milliseconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.reference import (  # noqa: E402
    BentSlab,
    attenuation_db_per_cm,
    bend_loss_db_per_90deg,
    imag_neff_from_db_per_cm,
)
from em_simulation.reference.fd1d_pml import PMLGrid, solve_bent_slab  # noqa: E402

WL = 1.55e-6

#: Weakly guiding on purpose.  A 220 nm Si slab confines so hard that its bend
#: loss is below double precision at every radius where the linearised
#: conformal map still holds - the barrier action grows as 3.3e7 * R.
WEAK = dict(core_index=1.50, clad_index=1.444, half_width=0.75e-6)


@pytest.fixture(scope="module")
def slab():
    return BentSlab(**WEAK)


@pytest.fixture(scope="module")
def sweep(slab):
    radii = np.array([150e-6, 200e-6, 250e-6, 300e-6, 400e-6, 500e-6])
    return radii, slab.sweep(radii)


# ------------------------------------------------------------- conversions


def test_one_db_per_cm_is_the_textbook_imaginary_index():
    assert imag_neff_from_db_per_cm(1.0, WL) == pytest.approx(2.84e-6, rel=1e-2)
    assert attenuation_db_per_cm(2.84e-6, WL) == pytest.approx(1.0, rel=1e-2)


def test_conversions_round_trip():
    for value in (1e-9, 1e-6, 1e-3):
        assert imag_neff_from_db_per_cm(
            attenuation_db_per_cm(value, WL), WL
        ) == pytest.approx(value, rel=1e-12)


def test_quarter_turn_loss_scales_with_arc_length():
    """Twice the radius is twice the arc, so twice the loss at equal alpha."""
    a = bend_loss_db_per_90deg(1e-6, WL, 100e-6)
    b = bend_loss_db_per_90deg(1e-6, WL, 200e-6)
    assert b == pytest.approx(2 * a, rel=1e-12)


# ------------------------------------------------------------- the straight limit


def test_straight_neff_lies_between_cladding_and_core(slab):
    n = slab.straight_neff()
    assert slab.clad_index < n < slab.core_index


def test_a_looser_bend_approaches_the_straight_guide(slab):
    """Re(n_eff) -> straight value and Im -> 0 as the bend opens out."""
    tight = slab.solve(200e-6)
    loose = slab.solve(500e-6, neff_guess=tight)
    straight = slab.straight_neff()
    assert abs(loose.real - straight) < abs(tight.real - straight)
    assert loose.imag < tight.imag


def test_tm_is_less_confined_than_te(slab):
    """A TM slab mode sees a weaker effective well, so it sits lower."""
    tm = BentSlab(**WEAK, polarization="TM")
    assert tm.straight_neff() < slab.straight_neff()


def test_polarization_is_validated():
    with pytest.raises(ValueError, match="TE"):
        BentSlab(polarization="TEM")


# ------------------------------------------------------------------ the physics


def test_loss_falls_monotonically_as_the_bend_opens(sweep):
    _, neff = sweep
    assert np.all(np.diff(neff.imag) < 0)


def test_the_sweep_spans_at_least_three_decades(sweep):
    """The Phase 1 gate is stated over three decades of Im(n_eff)."""
    _, neff = sweep
    assert np.log10(neff.imag.max() / neff.imag.min()) > 3.0


def test_loss_is_positive_in_this_projects_sign_convention(sweep):
    """``e^{i beta z}`` means a decaying mode has ``Im(n_eff) > 0``.

    The opposite branch is the incoming-wave solution and would report gain.
    ``correct_gain_modified`` clips negative values as round-off, so getting
    this backwards would silently produce a lossless answer.
    """
    _, neff = sweep
    assert np.all(neff.imag > 0)


def test_the_turning_point_moves_outward_with_radius(slab, sweep):
    """The tunnelling barrier widening is *why* the loss falls."""
    radii, neff = sweep
    turning = np.array([slab.turning_point(n, r) for n, r in zip(neff, radii)])
    assert np.all(np.diff(turning) > 0)


def test_loss_follows_the_wkb_tunnelling_exponent(slab, sweep):
    """The strongest check on the reference: it agrees with WKB on the slope.

    WKB knows nothing about Airy functions - it is just the barrier integral -
    yet it must predict how fast the loss falls with radius.  It does not fix
    the prefactor, so only the slope is compared, plus the requirement that the
    leftover offset is radius independent.
    """
    radii, neff = sweep
    exponent = np.array([slab.wkb_exponent(n, r) for n, r in zip(neff, radii)])
    measured = -np.log(neff.imag)

    slope_measured = np.polyfit(radii, measured, 1)[0]
    slope_wkb = np.polyfit(radii, exponent, 1)[0]
    assert slope_measured / slope_wkb == pytest.approx(1.0, rel=0.01)

    offset = measured - exponent
    assert offset.std() / abs(offset.mean()) < 0.02


def test_the_shipped_reference_stays_inside_the_linearisation(slab, sweep):
    """The Airy form drops O((a/R)^2); the reference must not rely on it."""
    radii, _ = sweep
    for radius in radii:
        assert slab.linearisation_error(radius) < 0.01


def test_a_seed_outside_the_guided_range_is_rejected_clearly(slab):
    """Outside the guided range the core 2x2 goes singular.

    Without an up-front check that surfaces as ``LinAlgError: Singular
    matrix``, which says nothing useful.  The result is guarded too - an
    unguarded secant once converged to n_eff *above* the core index.
    """
    with pytest.raises(ValueError, match="outside"):
        slab.solve(250e-6, neff_guess=slab.core_index + 0.5)
    with pytest.raises(ValueError, match="outside"):
        slab.solve(250e-6, neff_guess=slab.clad_index - 0.1)


def test_a_seed_well_inside_the_range_still_converges(slab):
    """The guard must not be so tight that continuation cannot move."""
    straight = slab.straight_neff()
    far = slab.clad_index + 0.25 * (slab.core_index - slab.clad_index)
    assert slab.solve(250e-6, neff_guess=far).real == pytest.approx(
        slab.solve(250e-6, neff_guess=straight).real, abs=1e-6
    )


# --------------------------------------------------------------- the PML itself


def test_pml_grid_is_real_until_the_stretch_begins():
    grid = PMLGrid(inner=2e-6, pml_start=4e-6, thickness=3e-6, dx=50e-9)
    physical = grid.u[: grid.n_physical + 1]
    assert np.allclose(np.imag(physical), 0.0)
    assert np.any(np.imag(grid.u[grid.n_physical + 1 :]) != 0.0)


def test_pml_stretch_is_graded_not_a_step():
    """A step in the stretch reflects; the ramp must start from unity."""
    grid = PMLGrid(inner=2e-6, pml_start=4e-6, thickness=3e-6, dx=50e-9,
                   factor=1 + 2j, order=2)
    imaginary = np.abs(np.imag(grid.increments[grid.n_physical:]))
    assert imaginary[0] < imaginary[-1]
    assert np.all(np.diff(imaginary) > 0)


def test_fd_pml_reproduces_the_airy_reference(slab, sweep):
    """The Phase 1 gate itself: within 10 % over more than three decades."""
    radii, exact = sweep
    errors = []
    for radius, reference in zip(radii, exact):
        value, _ = solve_bent_slab(slab, radius, neff_target=reference.real)
        errors.append(abs(value.imag / reference.imag - 1) * 100)
    assert max(errors) < 10.0


def test_fd_pml_agrees_on_the_real_part_too(slab):
    reference = slab.solve(250e-6)
    value, _ = solve_bent_slab(slab, 250e-6, neff_target=reference.real)
    assert value.real == pytest.approx(reference.real, abs=2e-4)


def test_the_pml_starts_outside_the_turning_point(slab):
    """Inside it the field is evanescent, so a PML there absorbs nothing."""
    _, info = solve_bent_slab(slab, 250e-6)
    assert info["pml_start"] > info["turning_point"]


@pytest.mark.parametrize(
    "override",
    [
        {"pml_thickness": 6e-6},   # thickness x1.5
        {"factor": 1 + 4j},        # stretch factor x2
        {"standoff": 2.25e-6},     # standoff x1.5
    ],
)
def test_loss_is_stable_under_the_named_pml_perturbations(slab, override):
    """Phase 3's criterion, checked early because it is cheap here.

    Every reported loss must hold to better than 5 % under PML thickness x1.5,
    stretch factor x2 and standoff x1.5.  A number that moves under those is
    not a result.
    """
    reference = slab.solve(250e-6)
    base, _ = solve_bent_slab(slab, 250e-6, neff_target=reference.real)
    moved, _ = solve_bent_slab(slab, 250e-6, neff_target=reference.real,
                               **override)
    assert abs(moved.imag / base.imag - 1) < 0.05


def test_a_thin_pml_is_detectably_not_converged(slab):
    """The probe that shows where convergence ends.

    Halving the PML to 2 um shifts the loss by ~14 %, which is why 4 um is the
    shipped default.  Pinning this stops someone "optimising" the thickness
    down without noticing the answer moved.
    """
    reference = slab.solve(250e-6)
    base, _ = solve_bent_slab(slab, 250e-6, neff_target=reference.real)
    thin, _ = solve_bent_slab(slab, 250e-6, neff_target=reference.real,
                              pml_thickness=2e-6)
    assert abs(thin.imag / base.imag - 1) > 0.05
