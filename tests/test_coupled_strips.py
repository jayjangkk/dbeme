"""The two-core cross section and the anti-crossing it produces.

``CoupledStrips`` is what demos 1-3 of the plan all rest on, so its geometry,
its guide-localisation measure and the supermode physics it produces are worth
pinning down.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import EmepyFDE  # noqa: E402
from dbeme.fde.cross_section import CoupledStrips  # noqa: E402

WL = 1.55e-6
WINDOW = (2.125e-6, -0.8e-6, 0.8e-6)


@pytest.fixture(scope="module")
def backend():
    return EmepyFDE(
        cross_section=CoupledStrips(gap=200e-9),
        parameter_names=("w1", "w2"),
        num_modes=4,
        wavelength=WL,
        window=WINDOW,
        mesh=160,
    )


# ---------------------------------------------------------------- geometry


def test_cores_are_placed_around_the_gap_centre():
    """Guide 1 sits at x < 0, guide 2 at x > 0, separated by exactly `gap`."""
    cross_section = CoupledStrips(gap=200e-9)
    (lo1, hi1), (lo2, hi2) = cross_section.edges({"w1": 850e-9, "w2": 200e-9})

    assert hi1 == pytest.approx(-100e-9)
    assert lo2 == pytest.approx(+100e-9)
    assert lo2 - hi1 == pytest.approx(200e-9)
    assert hi1 - lo1 == pytest.approx(850e-9)
    assert hi2 - lo2 == pytest.approx(200e-9)


def test_gap_can_come_from_the_parameters():
    """A swept gap overrides the constructor default."""
    cross_section = CoupledStrips(gap=200e-9, swept_parameters=("w1", "w2", "gap"))
    (_, hi1), (lo2, _) = cross_section.edges(
        {"w1": 500e-9, "w2": 500e-9, "gap": 350e-9}
    )
    assert lo2 - hi1 == pytest.approx(350e-9)


def test_index_profile_has_two_separated_cores():
    cross_section = CoupledStrips(gap=200e-9)
    x = np.linspace(-2.125e-6, 2.125e-6, 426)
    y = np.linspace(-0.8e-6, 0.8e-6, 81)
    n = cross_section.index(x, y, {"w1": 850e-9, "w2": 500e-9})

    core = n.max(axis=1) > 2.5
    # Two runs of core pixels, i.e. one gap between them.
    edges = np.diff(core.astype(int))
    assert np.count_nonzero(edges == 1) == 2, "expected two separate cores"
    assert np.count_nonzero(edges == -1) == 2


def test_fixed_gap_is_in_the_fingerprint():
    """Two datasets differing only in gap must not share an identity."""
    a = CoupledStrips(gap=200e-9)
    b = CoupledStrips(gap=250e-9)
    assert a.fingerprint() != b.fingerprint()

    # When the gap is swept it belongs to the parameter grid instead, and the
    # cross-section fingerprint should not pin it.
    swept = CoupledStrips(gap=200e-9, swept_parameters=("w1", "w2", "gap"))
    other = CoupledStrips(gap=250e-9, swept_parameters=("w1", "w2", "gap"))
    assert swept.fingerprint() == other.fingerprint()


# ----------------------------------------------------------------- physics


def test_power_fractions_localise_the_modes(backend):
    """A strongly asymmetric pair puts its lowest modes in the broad guide."""
    modes = backend.solve((850e-9, 200e-9))
    fractions = backend.cross_section.power_fractions(modes)

    assert 0.0 <= fractions.min() and fractions.max() <= 1.0
    # TE0 and TE1 of the 850 nm guide; the 200 nm guide holds nothing.
    assert fractions[0] > 0.95
    assert fractions[1] > 0.9


def test_the_pair_anticrosses(backend):
    """The whole point of the adiabatic coupler.

    Sweeping the widths past each other, the two guides exchange which one
    holds the second TE branch, and the branches repel rather than cross.
    """
    def te_branches(w1, w2):
        modes = backend.solve((round(w1, 12), round(w2, 12)))
        neff = np.real(modes.neff)
        keep = [
            i for i in range(len(neff))
            if modes.TE_pol[i] > 0.5 and neff[i] > 1.6
        ]
        return neff[keep], backend.cross_section.power_fractions(modes)[keep]

    n_start, f_start = te_branches(850e-9, 200e-9)
    n_end, f_end = te_branches(650e-9, 500e-9)

    # Second TE branch starts in the broad guide and ends in the narrow one.
    assert f_start[1] > 0.8, f_start
    assert f_end[1] < 0.2, f_end
    # The fundamental never leaves the broad guide.
    assert f_start[0] > 0.9 and f_end[0] > 0.9

    # And it rises rather than falling - it has become the narrow guide's TE0.
    assert n_end[1] > n_start[1]


def test_branch_separation_stays_open_through_the_crossing(backend):
    """An anti-crossing repels: the two branches never actually touch."""
    separations = []
    for fraction in np.linspace(0.2, 0.8, 7):
        w1 = 850e-9 + (650e-9 - 850e-9) * fraction
        w2 = 200e-9 + (500e-9 - 200e-9) * fraction
        modes = backend.solve((round(w1, 12), round(w2, 12)))
        neff = np.real(modes.neff)
        te = [
            neff[i] for i in range(len(neff))
            if modes.TE_pol[i] > 0.5 and neff[i] > 1.6
        ]
        if len(te) >= 3:
            separations.append(te[1] - te[2])

    assert separations, "no crossing region sampled"
    assert min(separations) > 0.02, f"branches came too close: {separations}"
