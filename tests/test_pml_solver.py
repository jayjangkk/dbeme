"""Phase 2: the PML mode solver.

``tasks/02_pml_backend.md`` Route A.  Three things had to be right, and each
failed silently rather than loudly the first time:

* the ``stretchmesh`` axis convention, which emepy's own recipe gets wrong;
* mode selection, which cannot be by effective index once a PML is present;
* the core mask used by the confinement filter, which must survive the
  conformal index ramp of a bend.

The solves here are on a coarse grid to keep the suite quick; the quantitative
validation against the Airy reference lives in
``examples/study_bend_loss_reference.py`` and ``reports/06_pml_phase1_gate.md``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.fde import FullEtchStrip  # noqa: E402
from em_simulation.fde.materials import ConstantIndex, silica  # noqa: E402
from em_simulation.fde.pml import (  # noqa: E402
    PMLModeSolver,
    stretched_grid,
    turning_point,
)

WL = 1.55e-6


def _sin_cross_section():
    """Si3N4, which is weakly guiding enough to radiate measurably."""
    return FullEtchStrip(
        thickness=0.4e-6,
        core=ConstantIndex(1.996, name="Si3N4"),
        cladding=silica(out_of_range="raise"),
    )


@pytest.fixture(scope="module")
def solver():
    return PMLModeSolver(
        _sin_cross_section(),
        window=(5.0e-6, -1.6e-6, 1.6e-6),
        mesh=180,
        pml_thickness=2.0e-6,
        num_modes=10,
    )


# ------------------------------------------------------------ the axis bug


@pytest.mark.parametrize(
    "kwargs,axis,edge",
    [
        (dict(layers_plus_x=3), "x", "end"),
        (dict(layers_minus_x=3), "x", "start"),
        (dict(layers_plus_y=3), "y", "end"),
        (dict(layers_minus_y=3), "y", "start"),
    ],
)
def test_stretch_lands_on_the_named_edge(kwargs, axis, edge):
    """The bug that cost the most, pinned.

    ``stretchmesh`` orders ``nlayers`` as ``[+y, -y, +x, -x]`` while emepy
    slices its fields as though 0 and 1 were x.  Putting the PML on the wrong
    axis does not raise - it absorbs the guided mode, and the symptom is a
    large, radius-independent "loss".
    """
    x = np.linspace(0.0, 10.0, 11)
    y = np.linspace(0.0, 20.0, 21)
    gx, gy = stretched_grid(x, y, **kwargs)

    moved = np.abs(np.imag(np.diff(gx if axis == "x" else gy)))
    still = np.abs(np.imag(np.diff(gy if axis == "x" else gx)))
    assert moved.sum() > 0, "the named axis was not stretched"
    assert still.sum() == 0, "the other axis was stretched"
    if edge == "end":
        assert moved[-1] > moved[0]
    else:
        assert moved[0] > moved[-1]


def test_both_ends_of_an_axis_absorb_in_the_same_sense():
    """``stretchmesh`` ends with ``abs(Im)``, which makes the ``-x``/``-y``
    layers gain.  The operator consumes ``diff(x)``: its imaginary part must
    have one sign in every layer, and on a symmetric window the stretch is
    antisymmetric."""
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-2.0, 2.0, 41)
    gx, gy = stretched_grid(x, y, layers_plus_x=4, layers_minus_x=4,
                            layers_plus_y=6, layers_minus_y=6)
    for g in (gx, gy):
        step = np.imag(np.diff(g))
        assert step[0] > 0 and step[-1] > 0, "an outer cell is not stretched"
        assert np.all(step >= 0), "a layer stretches with the wrong sign (gain)"
        assert np.allclose(g, -g[::-1]), "mirror-image layers are not mirror images"
    assert np.allclose(np.real(gx), x) and np.allclose(np.real(gy), y)


def test_a_low_side_pml_never_produces_gain():
    """The physical statement of the same bug, on raw eigenvalues.

    ``solve(return_all=True)`` reports ``n_eff`` before the ``Im >= 0`` fold,
    so a gain layer shows as ``Im(n_eff) < 0`` on the radiating members of
    the spectrum.  With the ``abs`` in place the ``-x`` run here returned
    the complex conjugate of the ``+x`` spectrum.
    """
    spectra = {}
    for edge in ("+x", "-x"):
        solver = PMLModeSolver(
            _sin_cross_section(),
            window=(3.0e-6, -1.2e-6, 1.2e-6),
            mesh=90,
            pml_thickness=1.0e-6,
            pml_edges=(edge,),
            num_modes=8,
        )
        _, info = solver.solve({"top_width": 1.0e-6, "curvature": 0.0},
                               1.5, return_all=True)
        spectra[edge] = info["neff"]
        # The discretised stretched operator is not exactly passive: on this
        # coarse mesh one continuum member sits at Im = -1.5e-4 even on the
        # validated +x edge (which is why ``solve`` folds the sign).  The bug
        # conjugates the *whole* spectrum, Im ~ -2e-2, an order of magnitude
        # beyond that noise.
        assert np.all(np.imag(info["neff"]) > -1e-3), f"gain on the {edge} edge"
        assert np.imag(info["neff"]).max() > 1e-3, "no radiating mode to test on"
    # a mirror-symmetric guide sees the same spectrum from either side;
    # a conjugated one differs by 2 Im(n_eff) ~ 0.04
    left = np.sort_complex(spectra["+x"])
    right = np.sort_complex(spectra["-x"])
    assert np.allclose(left, right, atol=1e-4)


def test_no_layers_means_no_stretch():
    x = np.linspace(0.0, 1.0, 5)
    gx, gy = stretched_grid(x, x)
    assert np.allclose(np.imag(gx), 0.0)
    assert np.allclose(np.imag(gy), 0.0)


# --------------------------------------------------------- the turning point


def test_turning_point_uses_the_exponential_conformal_form():
    """``u_t = R ln(n_eff / n_clad)``, not the linearised version.

    The Airy reference linearises the map, which puts the turning point ~70 %
    further out.  Using that here would place the PML in the wrong region.
    """
    assert turning_point(2.4, 1.444, 5e-6) == pytest.approx(
        5e-6 * np.log(2.4 / 1.444)
    )
    linearised = 5e-6 * (2.4**2 - 1.444**2) / (2 * 1.444**2)
    assert turning_point(2.4, 1.444, 5e-6) < linearised


def test_the_turning_point_moves_out_as_the_bend_opens():
    a = turning_point(1.64, 1.444, 15e-6)
    b = turning_point(1.64, 1.444, 40e-6)
    assert b > a


# ------------------------------------------------------------- the solver


def test_the_solver_declares_itself_lossy(solver):
    """`assemble` switches its backward basis on this, never on Im(n_eff)."""
    assert solver.lossless is False


def test_the_pml_starts_inside_the_window(solver):
    assert 0 < solver.pml_start < solver.window[0]


def test_a_straight_guide_is_undisturbed_by_the_pml(solver):
    """The PML must be invisible to a bound mode.

    A distant PML absorbs nothing from a guided mode, so the straight-guide
    answer has to be the same with and without it.  When the layers were on the
    wrong axis this failed hard: the mode vanished and every eigenvalue came
    back with ``Im ~ 0.3``.
    """
    bare = PMLModeSolver(
        _sin_cross_section(),
        window=solver.window,
        mesh=180,
        pml_thickness=0.0,
        pml_edges=(),
        num_modes=10,
    )
    params = {"top_width": 1.0e-6, "curvature": 0.0}
    with_pml, info = solver.solve(params, 1.70)
    without, _ = bare.solve(params, 1.70)

    assert np.real(with_pml) == pytest.approx(np.real(without), abs=2e-4)
    assert with_pml.imag < 1e-7
    assert info["selected_confinement"] > 0.5


def test_bend_loss_falls_as_the_radius_opens(solver):
    """The physics: a wider tunnelling barrier means exponentially less loss."""
    straight, _ = solver.solve({"top_width": 1.0e-6, "curvature": 0.0}, 1.70)
    tight, _ = solver.solve(
        {"top_width": 1.0e-6, "curvature": 1 / 15e-6}, np.real(straight)
    )
    loose, _ = solver.solve(
        {"top_width": 1.0e-6, "curvature": 1 / 30e-6}, np.real(straight)
    )
    assert tight.imag > loose.imag
    assert tight.imag / loose.imag > 100, "loss should fall by decades, not a bit"


def test_loss_is_positive_in_this_projects_sign_convention(solver):
    """``e^{i beta z}``: a decaying mode has ``Im(n_eff) > 0``."""
    value, _ = solver.solve({"top_width": 1.0e-6, "curvature": 1 / 15e-6}, 1.64)
    assert value.imag > 0


def test_the_confinement_filter_holds_through_a_bend(solver):
    """The core mask must survive the conformal index ramp.

    ``cross_section.index`` returns ``n exp(kappa x)`` under a bend, so a global
    threshold against ``max(index)`` selects the far outer *cladding* - and a
    Berenger mode then scores confinement 1.0 and gets picked.  Comparing
    within each column is ramp invariant.
    """
    straight, info0 = solver.solve({"top_width": 1.0e-6, "curvature": 0.0}, 1.70)
    _, info = solver.solve(
        {"top_width": 1.0e-6, "curvature": 1 / 15e-6}, np.real(straight)
    )
    assert info["selected_confinement"] == pytest.approx(
        info0["selected_confinement"], abs=0.05
    )
    assert info["selected_confinement"] < 0.99, "a mask that selects everything"


def test_the_spectrum_really_does_contain_modes_to_reject(solver):
    """The filter is not decoration: most of what comes back is spurious.

    With a PML the shift-invert window is full of Berenger modes of the
    discretised continuum.  If they were rare, sorting by effective index would
    have been good enough - it is not.
    """
    _, info = solver.solve(
        {"top_width": 1.0e-6, "curvature": 1 / 15e-6}, 1.64, return_all=True
    )
    confinement = info["confinement"]
    assert confinement.max() > 0.5, "no guided mode found at all"
    assert (confinement < solver.confinement_threshold).sum() >= 1, (
        "nothing was rejected, so this test proves nothing"
    )


def test_the_guard_fires_when_nothing_clears_the_threshold(solver):
    """Better a clear failure than a confidently returned Berenger mode."""
    strict = PMLModeSolver(
        _sin_cross_section(),
        window=solver.window,
        mesh=180,
        pml_thickness=2.0e-6,
        num_modes=10,
        confinement_threshold=0.999,
    )
    with pytest.raises(RuntimeError, match="confinement"):
        strict.solve({"top_width": 1.0e-6, "curvature": 0.0}, 1.70)


def test_guided_modes_are_ordered_ahead_of_spurious_ones(solver):
    """Neither ordering rule works alone, and this is why.

    Ordering by confinement flips between two guided modes sitting at 0.7650
    and 0.7643.  Ordering by ``Re(n_eff)`` fails the other way: a PML spectrum
    is not bounded by the guided modes, and at N = 40 two Berenger modes with
    zero confinement came back at ``Re(n_eff) = 2.3414`` on the narrow guide,
    above its real TE0 at 2.2333.  Mode 0 was then spurious on one side of an
    interface and physical on the other, and the junction transmitted 0.006
    instead of 1.000.

    Grouping by guidedness first and sorting by ``Re(n_eff)`` inside each group
    fixes it, because the confinement gap *between* the groups is large even
    where the gap inside the guided group is not.
    """
    params = {"top_width": 1.0e-6, "curvature": 0.0}
    _, info = solver.solve(params, 1.70, return_all=True)
    if (info["confinement"] < solver.confinement_threshold).sum() == 0:
        pytest.skip("no spurious modes in this basis to be ordered behind")

    data, confinement = solver.mode_data(params, 1.70)
    guided = confinement >= solver.confinement_threshold
    # every guided mode precedes every spurious one
    assert not np.any(np.diff(guided.astype(int)) > 0), (
        "a spurious mode was ordered ahead of a guided one"
    )
    # and the guided block is sorted by descending Re(n_eff)
    guided_neff = np.real(data.neff[: int(guided.sum())])
    assert np.all(np.diff(guided_neff) <= 1e-9)


def test_mode_zero_is_the_best_confined_mode(solver):
    """The label the whole EME cascade is built on."""
    data, confinement = solver.mode_data(
        {"top_width": 1.0e-6, "curvature": 0.0}, 1.70
    )
    assert confinement[0] == pytest.approx(confinement.max(), abs=1e-9)
