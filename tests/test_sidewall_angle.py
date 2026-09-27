"""Trapezoidal sidewalls.

A slanted etch is both a fabrication tolerance and a symmetry-breaking
mechanism: a trapezoid has no horizontal mirror plane, which is the same
symmetry a partial etch is introduced to break.  These tests pin the geometry,
check that a vertical wall reproduces the rectangle *exactly* (including the
dataset fingerprint, so existing caches stay valid), and confirm the physics.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import BiLevelStrip, EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.cross_section import _sidewall_run  # noqa: E402

WL = 1.55e-6
WINDOW = (2.2e-6, -0.8e-6, 0.8e-6)


# ------------------------------------------------------------------- run


def test_sidewall_run_matches_trigonometry():
    assert _sidewall_run(220e-9, 90.0) == 0.0
    assert _sidewall_run(130e-9, 87.0) == pytest.approx(130e-9 / np.tan(np.radians(87)))
    assert _sidewall_run(90e-9, 85.0) == pytest.approx(7.874e-9, rel=1e-3)


def test_shallower_angle_runs_out_further():
    runs = [_sidewall_run(220e-9, a) for a in (89.0, 85.0, 80.0)]
    assert runs[0] < runs[1] < runs[2]


def test_impossible_angles_are_rejected():
    with pytest.raises(ValueError, match="must be in"):
        _sidewall_run(220e-9, 0.0)
    with pytest.raises(ValueError, match="must be in"):
        _sidewall_run(220e-9, -10.0)


# -------------------------------------------------------------- geometry


def _half_widths(cross_section, params, heights):
    x = np.linspace(-1.2e-6, 1.2e-6, 2401)  # 1 nm cells
    y = np.linspace(-0.16e-6, 0.16e-6, 641)
    n = cross_section.index(x, y, params)
    silicon = n > 2.5
    out = []
    for height in heights:
        column = silicon[:, int(np.argmin(np.abs(y - height)))]
        rows = np.flatnonzero(column)
        out.append((x[rows[-1]] - x[rows[0]]) / 2 if rows.size else 0.0)
    return out


def test_vertical_strip_is_unchanged_by_the_trapezoid_code():
    """A 90 degree wall must reproduce the rectangle bit for bit."""
    x = np.linspace(-2.2e-6, 2.2e-6, 441)
    y = np.linspace(-0.8e-6, 0.8e-6, 161)
    params = {"top_width": 550e-9, "curvature": 0.0}
    a = FullEtchStrip(thickness=220e-9).index(x, y, params)
    b = FullEtchStrip(thickness=220e-9, sidewall_angle=90.0).index(x, y, params)
    assert np.array_equal(a, b)


def test_strip_widens_towards_its_base():
    """Width is top-referenced, so the base is wider."""
    cross_section = FullEtchStrip(thickness=220e-9, sidewall_angle=85.0)
    top, bottom = _half_widths(
        cross_section, {"top_width": 640e-9, "curvature": 0.0}, [105e-9, -105e-9]
    )
    assert top == pytest.approx(320e-9, abs=4e-9)
    expected = 320e-9 + _sidewall_run(220e-9, 85.0)
    assert bottom == pytest.approx(expected, abs=4e-9)
    assert bottom > top


def test_rib_uses_a_different_angle_per_etch_step():
    """The full etch shapes the slab, the partial etch shapes the core."""
    cross_section = BiLevelStrip(
        thickness=220e-9,
        slab_thickness=90e-9,
        bottom_sidewall_angle=85.0,
        top_sidewall_angle=87.0,
    )
    params = {"w_core": 550e-9, "w_slab": 900e-9}
    core_top, core_base, slab_top, slab_base = _half_widths(
        cross_section, params, [105e-9, -15e-9, -25e-9, -105e-9]
    )

    assert core_top == pytest.approx(275e-9, abs=4e-9)
    assert core_base - core_top == pytest.approx(
        _sidewall_run(130e-9, 87.0), abs=4e-9
    )
    assert slab_top == pytest.approx(450e-9, abs=4e-9)
    assert slab_base - slab_top == pytest.approx(
        _sidewall_run(90e-9, 85.0), abs=4e-9
    )


# ----------------------------------------------------------- fingerprint


def test_vertical_walls_leave_the_fingerprint_untouched():
    """Existing datasets must stay valid.

    A 90 degree wall *is* the rectangular geometry, so recording a redundant
    angle term would change the identity string without changing the geometry
    and invalidate every dataset built before angles existed.
    """
    assert (
        FullEtchStrip(thickness=220e-9).fingerprint()
        == FullEtchStrip(thickness=220e-9, sidewall_angle=90.0).fingerprint()
    )
    assert (
        BiLevelStrip().fingerprint()
        == BiLevelStrip(bottom_sidewall_angle=90.0, top_sidewall_angle=90.0).fingerprint()
    )


def test_slanted_walls_change_the_fingerprint():
    base = BiLevelStrip().fingerprint()
    assert BiLevelStrip(bottom_sidewall_angle=85.0).fingerprint() != base
    assert BiLevelStrip(top_sidewall_angle=87.0).fingerprint() != base
    assert (
        BiLevelStrip(bottom_sidewall_angle=85.0, top_sidewall_angle=87.0).fingerprint()
        != BiLevelStrip(bottom_sidewall_angle=87.0, top_sidewall_angle=85.0).fingerprint()
    )


# --------------------------------------------------------------- physics


def _second_and_third(backend, width):
    modes = backend.solve((round(float(width), 12), 0.0))
    neff = np.real(modes.neff)
    keep = sorted(
        [i for i in range(len(neff)) if neff[i] > 1.6], key=lambda i: -neff[i]
    )[1:3]
    return [neff[i] for i in keep], [float(modes.TE_pol[i]) for i in keep]


@pytest.mark.parametrize("angle,expect_hybrid", [(90.0, False), (85.0, True)])
def test_slant_turns_a_crossing_into_an_anticrossing(angle, expect_hybrid):
    """The whole mechanism, on a plain strip with no partial etch.

    TE1 and TM0 of a 220 nm strip cross near 640-660 nm.  With vertical walls
    the horizontal mirror plane keeps them in different symmetry classes and
    they pass straight through each other, staying cleanly polarised.  Slant
    the walls and the same crossing opens into an anti-crossing with genuinely
    mixed modes.
    """
    backend = EmepyFDE(
        cross_section=FullEtchStrip(thickness=220e-9, sidewall_angle=angle),
        num_modes=4,
        wavelength=WL,
        window=WINDOW,
        mesh=180,
    )
    hybrid_found = False
    for width in np.arange(620, 681, 10) * 1e-9:
        _, te = _second_and_third(backend, width)
        if len(te) == 2 and all(0.2 < value < 0.8 for value in te):
            hybrid_found = True
            break
    assert hybrid_found is expect_hybrid, (
        f"at {angle} degrees, hybrid modes {'were not' if expect_hybrid else 'were'} "
        "found across the TE1/TM0 crossing"
    )
