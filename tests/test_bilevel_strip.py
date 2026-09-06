"""The rib cross section, and the symmetry argument it rests on.

A polarization rotator only works if the cross section has no horizontal mirror
plane.  These tests check that ``BiLevelStrip`` builds the right geometry, that
a symmetric strip really does give zero TE-TM coupling (so the failure mode
``CLAUDE.md`` §5.11 warns about is caught rather than mistaken for a bug), and
that adding the slab produces the TM0/TE1 hybridisation the device needs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.fde import BiLevelStrip, EmepyFDE, FullEtchStrip  # noqa: E402

WL = 1.55e-6
WINDOW = (2.2e-6, -0.8e-6, 0.8e-6)


@pytest.fixture(scope="module")
def backend():
    return EmepyFDE(
        cross_section=BiLevelStrip(thickness=220e-9, slab_thickness=90e-9),
        parameter_names=("w_core", "w_slab"),
        num_modes=4,
        wavelength=WL,
        window=WINDOW,
        mesh=170,
    )


# ---------------------------------------------------------------- geometry


def test_rib_has_a_full_core_and_a_thinner_slab():
    cross_section = BiLevelStrip(thickness=220e-9, slab_thickness=90e-9)
    x = np.linspace(-2.2e-6, 2.2e-6, 441)
    y = np.linspace(-0.8e-6, 0.8e-6, 161)
    n = cross_section.index(x, y, {"w_core": 550e-9, "w_slab": 1550e-9})
    silicon = n > 2.5

    def height(x_query):
        column = silicon[int(np.argmin(np.abs(x - x_query)))]
        rows = np.flatnonzero(column)
        return y[rows[0]], y[rows[-1]]

    core_lo, core_hi = height(0.0)
    slab_lo, slab_hi = height(600e-9)

    assert core_hi - core_lo == pytest.approx(220e-9, abs=1.5e-8)
    assert slab_hi - slab_lo == pytest.approx(90e-9, abs=1.5e-8)
    # Both sit on the same floor: the etch removes material from the top.
    assert slab_lo == pytest.approx(core_lo, abs=1.5e-8)
    assert slab_hi < core_hi


def test_equal_widths_give_a_plain_strip():
    """How a bi-level taper starts and finishes."""
    cross_section = BiLevelStrip()
    x = np.linspace(-2.2e-6, 2.2e-6, 441)
    y = np.linspace(-0.8e-6, 0.8e-6, 161)
    n = cross_section.index(x, y, {"w_core": 450e-9, "w_slab": 450e-9})

    strip = FullEtchStrip(
        thickness=220e-9,
        core=cross_section.core,
        cladding=cross_section.cladding,
    )
    reference = strip.index(x, y, {"top_width": 450e-9, "curvature": 0.0})
    assert np.allclose(n, reference, atol=1e-9)


def test_slab_thickness_must_be_a_real_partial_etch():
    with pytest.raises(ValueError, match="strictly between"):
        BiLevelStrip(thickness=220e-9, slab_thickness=220e-9)
    with pytest.raises(ValueError, match="strictly between"):
        BiLevelStrip(thickness=220e-9, slab_thickness=0.0)


def test_slab_thickness_is_in_the_fingerprint():
    assert (
        BiLevelStrip(slab_thickness=90e-9).fingerprint()
        != BiLevelStrip(slab_thickness=130e-9).fingerprint()
    )


# ----------------------------------------------------------------- physics


#: Well above the 1.444 cladding index.  Near cutoff a mode's field is mostly
#: in the cladding and its TE fraction stops meaning anything, so both tests
#: below restrict themselves to modes that are genuinely guided.
GUIDED = 1.6


def _guided_te_fractions(modes):
    return {
        i: float(modes.TE_pol[i])
        for i in range(len(modes.neff))
        if np.real(modes.neff[i]) > GUIDED
    }


def test_symmetric_strip_has_no_hybrid_modes():
    """The failure mode CLAUDE.md 5.11 warns about, pinned as a test.

    A buried strip has a horizontal mirror plane.  Its modes split into two
    classes under that reflection and nothing mixes them, so every guided mode
    is cleanly quasi-TE or quasi-TM.  A rotator built on this cross section
    returns identically zero conversion, and it would look like a bug.

    Note the modes are not *purely* TE or TM even so - the symmetry makes ``Ey``
    odd in y rather than zero, so quasi-TE lands near 0.98 rather than exactly
    1.  What the symmetry forbids is the in-between.
    """
    backend = EmepyFDE(
        cross_section=FullEtchStrip(thickness=220e-9),
        num_modes=4,
        wavelength=WL,
        window=WINDOW,
        mesh=170,
    )
    fractions = _guided_te_fractions(backend.solve((550e-9, 0.0)))
    assert fractions
    for index, value in fractions.items():
        assert value > 0.9 or value < 0.1, (
            f"mode {index} is hybrid (TE fraction {value:.3f}) in a cross "
            "section that has a horizontal mirror plane"
        )


def test_rib_hybridises_tm0_and_te1(backend):
    """With the slab, genuinely mixed modes appear - that is the rotation.

    Same measurement, same cutoff, opposite answer: removing the horizontal
    mirror plane lets TM0 and TE1 mix, and at the anti-crossing they are close
    to a 50/50 superposition.
    """
    # Near the anti-crossing found by examples/demo_polarization_rotator.py.
    fractions = _guided_te_fractions(backend.solve((488e-9, 870e-9)))
    hybrid = {i: v for i, v in fractions.items() if 0.2 < v < 0.8}
    assert hybrid, (
        f"no hybrid mode at the anti-crossing; TE fractions {fractions}"
    )


def test_polarization_rotates_across_the_taper(backend):
    """The branch that starts TM must end TE.

    Checked at the endpoints of the published width schedule, without any
    tracking machinery: at the input the second guided mode is TM0, and at the
    output the second is TE1.
    """
    entry = backend.solve((450e-9, 450e-9))
    exit_ = backend.solve((850e-9, 850e-9))

    def second_guided(modes):
        order = [
            i for i in np.argsort(-np.real(modes.neff))
            if np.real(modes.neff[i]) > 1.5
        ]
        return order[1]

    assert backend.cross_section is not None
    assert entry.TE_pol[second_guided(entry)] < 0.2, "input should be TM0"
    assert exit_.TE_pol[second_guided(exit_)] > 0.8, "output should be TE1"
