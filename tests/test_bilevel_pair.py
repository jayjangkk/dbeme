"""The two-arm, two-height cross section of the double-tip edge coupler.

``BiLevelPair`` carries the whole Wan & Wang edge-coupler path in one family
(``tasks/18_edge_coupler_oband.md``): 150 nm tips, L-shaped arms in the
height converter, 220 nm strips, and - at ``gap = 0`` - the merged MMI and
output strip.  These tests pin each of those limits to an independently
written cross section, and the stack (Si substrate, BOX, top oxide, air)
to its layer boundaries.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import BiLevelPair, CoupledStrips, FullEtchStrip  # noqa: E402

X = np.linspace(-3.0e-6, 3.0e-6, 601)       # 10 nm cells, x = 0 on a node
Y = np.linspace(-1.0e-6, 1.0e-6, 201)


def buried(**kwargs):
    """The comparison stack: oxide above and below, nothing else."""
    return BiLevelPair(substrate=None, top=None, **kwargs)


# ---------------------------------------------------------------- limits


def test_full_height_arms_are_a_coupled_pair():
    pair = buried()
    n = pair.index(X, Y, {"w_low": 0.0, "w_high": 450e-9, "gap": 420e-9})
    ref = CoupledStrips(thickness=220e-9, core=pair.core, cladding=pair.cladding,
                        swept_parameters=("w1", "w2", "gap"), reference_wavelength=1.31e-6)
    m = ref.index(X, Y, {"w1": 450e-9, "w2": 450e-9, "gap": 420e-9})
    assert np.allclose(n, m, atol=1e-9)


def test_shelf_only_arms_are_a_thin_pair_on_the_same_floor():
    """The 150 nm tip: the partial etch removes silicon from the top, so the
    thin strip sits on the BOX, centred 35 nm below the 220 nm layer's centre."""
    pair = buried()
    n = pair.index(X, Y, {"w_low": 130e-9, "w_high": 0.0, "gap": 1.6e-6})
    ref = CoupledStrips(thickness=150e-9, core=pair.core, cladding=pair.cladding,
                        swept_parameters=("w1", "w2", "gap"), reference_wavelength=1.31e-6)
    shift = -110e-9 + 75e-9                  # centre of the 150 nm layer
    m = ref.index(X, Y - shift, {"w1": 130e-9, "w2": 130e-9, "gap": 1.6e-6})
    assert np.allclose(n, m, atol=1e-9)


def test_merged_arms_are_one_strip():
    """gap = 0: the MMI and the output guide."""
    pair = buried()
    n = pair.index(X, Y, {"w_low": 0.0, "w_high": 800e-9, "gap": 0.0})
    strip = FullEtchStrip(thickness=220e-9, core=pair.core, cladding=pair.cladding,
                          reference_wavelength=1.31e-6)
    m = strip.index(X, Y, {"top_width": 1.6e-6, "curvature": 0.0})
    assert np.allclose(n, m, atol=1e-9)


def test_l_shaped_arm_has_its_full_height_part_on_the_inner_edge():
    pair = buried()
    n = pair.index(X, Y, {"w_low": 200e-9, "w_high": 100e-9, "gap": 1.6e-6})
    si = n > 2.5

    def rows(x_query):
        col = np.flatnonzero(si[int(np.argmin(np.abs(X - x_query)))])
        return Y[col[0]], Y[col[-1]]

    inner = rows(0.8e-6 + 50e-9)             # inside w_high, next to the gap
    outer = rows(0.8e-6 + 100e-9 + 100e-9)   # on the shelf
    assert inner[1] - inner[0] == pytest.approx(220e-9, abs=1e-12)   # half-filled edge nodes count
    assert outer[1] - outer[0] == pytest.approx(150e-9, abs=1e-12)
    assert inner[0] == pytest.approx(outer[0])                        # same floor
    assert not si[int(np.argmin(np.abs(X - 1.11e-6)))].any()          # beyond the arm


def test_mirror_symmetric():
    pair = BiLevelPair()
    for p in ({"w_low": 130e-9, "w_high": 0.0, "gap": 1.6e-6},
              {"w_low": 150e-9, "w_high": 150e-9, "gap": 1.6e-6},
              {"w_low": 0.0, "w_high": 450e-9, "gap": 420e-9}):
        n = pair.index(X, Y, p)
        assert np.allclose(n, n[::-1, :], atol=1e-12, rtol=0)


# ---------------------------------------------------------------- stack


def test_stack_layers():
    pair = BiLevelPair()
    y = np.linspace(-3.0e-6, 3.0e-6, 601)
    n = pair.index(np.array([4e-6]), y, {"w_low": 130e-9, "w_high": 0.0, "gap": 1.6e-6})[0]
    n_si = pair.core_index_at(pair.reference_wavelength)
    n_ox = pair.cladding_index_at(pair.reference_wavelength)
    assert pair.y_substrate == pytest.approx(-2.11e-6)
    assert pair.y_top_oxide == pytest.approx(1.89e-6)
    assert n[y < -2.12e-6] == pytest.approx(n_si)
    assert n[(y > -2.1e-6) & (y < 1.88e-6)] == pytest.approx(n_ox)
    assert n[y > 1.9e-6] == pytest.approx(1.0)


def test_cutoff_is_the_oxide_not_the_substrate():
    """A max(cladding, substrate) cut-off would flag every mode as radiation."""
    pair = BiLevelPair()
    wl = 1.31e-6
    assert pair.cladding_index_at(wl) == pytest.approx(float(np.real(pair.cladding.index(wl))))
    assert pair.cladding_index_at(wl) < 1.5


def test_core_mask_excludes_the_substrate():
    pair = BiLevelPair(core_mask_margin=0.1e-6)
    x = np.linspace(-5e-6, 5e-6, 1001)
    y = np.linspace(-2.8e-6, 2.8e-6, 561)
    mask = pair.core_mask(x, y, {"w_low": 130e-9, "w_high": 0.0, "gap": 1.6e-6})
    assert mask.shape == (x.size, y.size)
    assert not mask[:, y < pair.y_substrate].any()
    assert not mask[:, y > pair.y_top_oxide].any()
    assert mask[np.argmin(np.abs(x - 0.865e-6)), np.argmin(np.abs(y + 0.035e-6))]
    assert not mask[np.argmin(np.abs(x)), np.argmin(np.abs(y))]           # the gap


def test_polygons_paint_the_same_silicon_as_index():
    pair = BiLevelPair()
    p = {"w_low": 150e-9, "w_high": 150e-9, "gap": 1.6e-6}
    regions = pair.polygons(p)
    assert list(regions) == ["substrate", "top", "core"]
    core_area = regions["core"][0].area
    expected = 2 * (300e-9 * 150e-9 + 150e-9 * 70e-9)
    assert core_area == pytest.approx(expected, rel=1e-9)

    # the same area from the index: fill fraction of Si in the device layer
    n = buried().index(X, Y, p)
    n_si = pair.core_index_at(pair.reference_wavelength)
    n_ox = pair.cladding_index_at(pair.reference_wavelength)
    fill = (n**2 - n_ox**2) / (n_si**2 - n_ox**2)
    area = fill.sum() * (X[1] - X[0]) * (Y[1] - Y[0])
    assert area == pytest.approx(expected, rel=1e-6)


def test_fingerprint_separates_stacks():
    assert BiLevelPair().fingerprint() != buried().fingerprint()
    assert BiLevelPair(box=2e-6).fingerprint() != BiLevelPair(box=3e-6).fingerprint()
    assert BiLevelPair(thickness_low=150e-9).fingerprint() != BiLevelPair(thickness_low=130e-9).fingerprint()


def test_rejects_negative_or_empty_arms():
    pair = BiLevelPair()
    with pytest.raises(ValueError):
        pair.index(X, Y, {"w_low": -1e-9, "w_high": 100e-9, "gap": 0.0})
    with pytest.raises(ValueError):
        pair.index(X, Y, {"w_low": 0.0, "w_high": 0.0, "gap": 1e-6})
    with pytest.raises(ValueError):
        BiLevelPair(thickness_low=0.0)
