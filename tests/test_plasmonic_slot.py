"""The plasmonic slot cross section and the dataset-facing PML backend.

Geometry and contract only - the physics (does the solver find the gap
plasmon at the slab reference's index) lives in ``examples/`` and
``reports/12``, because a resolved metal solve is too slow for the suite.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.fde.base import FDEBackend, ModeData  # noqa: E402
from em_simulation.fde.materials import air, gold, silica  # noqa: E402
from em_simulation.fde.pml import PMLBackend  # noqa: E402
from em_simulation.fde.slot_converter import PlasmonicSlotConverter  # noqa: E402

WL = 1.55e-6


@pytest.fixture(scope="module")
def section():
    return PlasmonicSlotConverter(gap=20e-9, metal_thickness=20e-9)


@pytest.fixture(scope="module")
def grid():
    return np.linspace(-0.5e-6, 0.5e-6, 501), np.linspace(-0.3e-6, 0.3e-6, 301)


def _at(n, x, y, xx, yy):
    return n[np.argmin(np.abs(x - xx)), np.argmin(np.abs(y - yy))]


# --------------------------------------------------------------- geometry


def test_index_is_complex_because_gold_is(section, grid):
    x, y = grid
    n = section.index(x, y, {"w_si": 0.4e-6})
    assert np.iscomplexobj(n)
    assert np.all(np.imag(n) >= 0), "a negative Im(n) would be gain"


def test_the_layout_matches_the_paper(section, grid):
    """Si core centred; gold plates in the Si plane with an air gap; SiO2
    below, air above."""
    x, y = grid
    n = section.index(x, y, {"w_si": 0.4e-6})
    assert _at(n, x, y, 0.0, 0.0).real == pytest.approx(3.476, abs=0.01)   # Si
    assert _at(n, x, y, 0.0, 0.25e-6).real == pytest.approx(1.0, abs=1e-6)  # air above
    assert _at(n, x, y, 0.0, -0.25e-6).real == pytest.approx(1.444, abs=0.01)  # oxide below
    # air gap between the Si edge (200 nm) and the metal (220 nm)
    assert _at(n, x, y, 0.21e-6, -0.1e-6).real == pytest.approx(1.0, abs=1e-6)
    # gold from 220 nm outward, bottom on the substrate plane, 20 nm tall
    au = _at(n, x, y, 0.3e-6, -0.1e-6)
    assert au.imag > 9 and au.real < 1
    assert _at(n, x, y, 0.3e-6, 0.0).real == pytest.approx(1.0, abs=1e-6)  # air above the film


def test_metal_follows_the_taper_with_a_constant_gap(section, grid):
    x, y = grid
    for width in (0.4e-6, 0.2e-6, 0.05e-6):
        n = section.index(x, y, {"w_si": width})
        inner = 0.5 * width + section.gap
        assert _at(n, x, y, inner - 5e-9, -0.1e-6).real == pytest.approx(1.0, abs=1e-6)
        assert _at(n, x, y, inner + 8e-9, -0.1e-6).imag > 9


def test_no_silicon_leaves_a_slot_two_gaps_wide(section, grid):
    x, y = grid
    n = section.index(x, y, {"w_si": 0.0})
    assert _at(n, x, y, 0.0, -0.1e-6).real == pytest.approx(1.0, abs=1e-6)
    assert _at(n, x, y, 0.015e-6, -0.1e-6).real == pytest.approx(1.0, abs=1e-6)
    assert _at(n, x, y, 0.03e-6, -0.1e-6).imag > 9


def test_permittivity_not_index_is_averaged(section):
    """A half-filled metal cell must average eps, which for gold is far from
    averaging n: (n_Au^2 + 1)/2 has |n| ~ 7.5, while (n_Au + 1)/2 has |n| ~ 5.4."""
    x = np.array([0.0])                    # one cell straddling the metal edge
    y = np.array([-0.1e-6])
    thin = PlasmonicSlotConverter(gap=0.0, metal_thickness=20e-9)
    # a 1-cell grid gets fill fraction 1 or 0, so probe eps via the formula
    eps_au = complex(gold().index(WL)) ** 2
    expected_half = np.sqrt(0.5 * eps_au + 0.5 * 1.0)
    assert abs(expected_half) > 7.0 and abs((complex(gold().index(WL)) + 1) / 2) < 6.0
    assert thin.index(x, y, {"w_si": 0.0}).shape == (1, 1)


# ------------------------------------------------------------- core mask


def test_core_mask_covers_the_slot_and_the_silicon(section, grid):
    x, y = grid
    mask = section.core_mask(x, y, {"w_si": 0.4e-6})
    assert mask.shape == (x.size, y.size)
    assert mask[np.argmin(np.abs(x)), np.argmin(np.abs(y))]                 # Si core
    assert mask[np.argmin(np.abs(x - 0.21e-6)), np.argmin(np.abs(y + 0.1e-6))]  # the gap
    assert not mask[np.argmin(np.abs(x - 0.4e-6)), np.argmin(np.abs(y + 0.1e-6))]  # metal
    assert not mask[np.argmin(np.abs(x)), np.argmin(np.abs(y - 0.25e-6))]   # air above


def test_core_mask_is_the_slot_when_the_silicon_is_gone(section, grid):
    """Where the gap plasmon lives - an *air* region a high-index heuristic
    would call cladding."""
    x, y = grid
    mask = section.core_mask(x, y, {"w_si": 0.0})
    columns = np.flatnonzero(mask.any(axis=1))
    assert x[columns].min() == pytest.approx(-section.gap, abs=3e-9)
    assert x[columns].max() == pytest.approx(section.gap, abs=3e-9)


# --------------------------------------------------------------- identity


def test_fingerprint_tracks_every_geometric_parameter():
    base = PlasmonicSlotConverter().fingerprint()
    assert PlasmonicSlotConverter(gap=40e-9).fingerprint() != base
    assert PlasmonicSlotConverter(metal_thickness=30e-9).fingerprint() != base
    assert PlasmonicSlotConverter(si_thickness=200e-9).fingerprint() != base
    assert PlasmonicSlotConverter(plate_reach=0.4e-6).fingerprint() != base
    assert PlasmonicSlotConverter(substrate=air()).fingerprint() != base
    assert PlasmonicSlotConverter(metal_bottom=-0.2e-6).fingerprint() != base


def test_walls_can_be_placed_off_the_substrate_plane(grid):
    x, y = grid
    section = PlasmonicSlotConverter(metal_thickness=400e-9, metal_bottom=-200e-9, gap=20e-9)
    n = section.index(x, y, {"w_si": 0.0})
    assert _at(n, x, y, 0.1e-6, 0.19e-6).imag > 9      # gold well above the Si top
    assert _at(n, x, y, 0.1e-6, -0.19e-6).imag > 9     # and below its bottom
    assert _at(n, x, y, 0.1e-6, 0.21e-6).real == pytest.approx(1.0, abs=1e-6)
    mask = section.core_mask(x, y, {"w_si": 0.0})
    rows = np.flatnonzero(mask.any(axis=0))
    assert y[rows].min() == pytest.approx(-0.2e-6, abs=3e-9)
    assert y[rows].max() == pytest.approx(0.2e-6, abs=3e-9)


def test_cutoff_is_the_oxide_not_the_air(section):
    assert section.cladding_index_at(WL) == pytest.approx(1.444, abs=1e-3)


def test_default_window_holds_the_widest_wire_and_both_plates(section):
    half, y_min, y_max = section.default_window({"w_si": 0.4e-6})
    _, _, outer = section.edges(0.4e-6)
    assert half > outer
    assert y_min < -0.5 * section.si_thickness and y_max > 0.5 * section.si_thickness


def test_plates_default_to_the_silicon_height():
    """A lateral slot binds only between full-height walls."""
    section = PlasmonicSlotConverter(si_thickness=200e-9)
    assert section.metal_thickness == pytest.approx(200e-9)
    assert PlasmonicSlotConverter(metal_thickness=20e-9).metal_thickness == pytest.approx(20e-9)


def test_platform_is_suspended_with_full_height_plates_and_four_pml_edges():
    from em_simulation.platforms import plasmonic_converter_dataset_info

    info = plasmonic_converter_dataset_info(cell=20e-9)()
    assert info.cladding_index == pytest.approx(1.0)
    section = info.get_cross_section()
    assert section.metal_thickness == pytest.approx(400e-9)
    assert section.metal_bottom == pytest.approx(-200e-9), "walls centred on the Si"
    half, y_min, y_max = section.default_window({"w_si": 0.4e-6})
    assert y_max == -y_min and y_max >= 0.75e-6 - 1e-12
    backend = info.get_fde_backend()
    assert set(backend.pml_edges) == {"+x", "-x", "+y", "-y"}
    assert backend.lossless is False
    on_oxide = plasmonic_converter_dataset_info(cell=20e-9, substrate="silica")()
    assert on_oxide.cladding_index == pytest.approx(1.444, abs=1e-3)


def test_platform_grid_is_aligned_with_the_geometry():
    """Pitch exactly the cell, origin on a grid point, and every axis width
    puts the Si edge and the metal inner edge on grid points - so all cross
    sections are discretised the same way (a metal edge moving inside a cell
    moved the hybrid mode's index by 0.1)."""
    from em_simulation.platforms import plasmonic_converter_dataset_info

    cell = 10e-9
    info = plasmonic_converter_dataset_info(cell=cell)()
    backend = info.get_fde_backend()
    x, y = np.real(backend.x), np.real(backend.y)       # the stretch is imaginary
    assert np.allclose(np.diff(x), cell) and np.allclose(np.diff(y), cell)
    assert np.abs(x).min() < 1e-15 and np.abs(y).min() < 1e-15
    section = info.get_cross_section()
    widths = np.asarray(info.parameters["w_si"], dtype=float)
    assert widths[0] == 0.0 and widths[-1] == pytest.approx(0.4e-6)
    assert np.allclose(np.diff(widths), 2 * cell)
    for w in widths:
        _, inner, _ = section.edges(w)
        assert np.abs(x - 0.5 * w).min() < 1e-15
        assert np.abs(x - inner).min() < 1e-15


# ------------------------------------------------------------ the backend


@pytest.fixture(scope="module")
def backend():
    """Deliberately coarse and small: this checks the contract, not the physics."""
    section = PlasmonicSlotConverter(gap=25e-9, metal_thickness=200e-9,
                                     plate_reach=0.2e-6, substrate=air())
    return PMLBackend(
        section, target_neff=1.7,
        wavelength=WL, window=(-0.4e-6, 0.4e-6, -0.3e-6, 0.3e-6), mesh=100,
        pml_thickness=0.1e-6, pml_edges=("+x", "-x", "+y", "-y"), num_modes=6,
    )


def test_backend_satisfies_the_abc(backend):
    assert isinstance(backend, FDEBackend)
    assert backend.lossless is False
    assert backend.num_modes == 6
    assert backend.wavelength == pytest.approx(WL)
    assert backend.parameter_names == ("w_si",)


def test_backend_grid_is_the_complex_stretched_one(backend):
    x, y = backend.grid()
    assert np.iscomplexobj(x) and np.iscomplexobj(y)
    assert np.abs(np.imag(x)).max() > 0
    sx, sy = backend.solve_grid()
    assert np.array_equal(sx, x) and np.array_equal(sy, y)


def test_backend_solve_returns_gauge_pinned_mode_data(backend):
    data = backend.solve((0.0,))
    assert isinstance(data, ModeData)
    assert data.E.shape == (6, 3, len(backend.x), len(backend.y))
    assert np.all(np.imag(data.neff) >= 0), "a lossy basis must not contain gain"
    # gauge, as `_pin_gauge` defines it: the transverse field is real at its
    # peak, and the *first* raster point above half-maximum is positive (the
    # first rather than the peak, so equal-magnitude lobes cannot flip it).
    for i in range(6):
        transverse = data.E[i, :2]
        magnitude = np.abs(transverse)
        peak = transverse.flat[int(np.argmax(magnitude))]
        assert abs(np.imag(peak)) < 1e-6 * abs(peak)
        real = np.real(transverse).ravel()
        first = np.flatnonzero(np.abs(real) > 0.5 * magnitude.max())[0]
        assert real[first] > 0


def test_backend_recipe_enters_the_dataset_identity(backend):
    """Same grid, different PML or target: a different mode problem, and the
    identity check must say so.  The recipe has to survive a JSON round trip
    unchanged, or every reopen would spuriously fail."""
    import json
    from types import SimpleNamespace

    from em_simulation.data_updater.dataset_identity import fingerprint

    recipe = backend.fingerprint()
    assert json.loads(json.dumps(recipe)) == recipe
    assert recipe["stretch_convention"] >= 1

    info = SimpleNamespace(
        get_wavelength=lambda: WL, get_mode_numbers=lambda: 6,
        get_parameter_names=lambda: ("w_si",),
        get_parameter_grid=lambda: {"w_si": np.array([0.0, 1e-7])},
    )
    assert fingerprint(info, backend)["backend"] == recipe
    other = PMLBackend(
        backend.cross_section, target_neff=1.9, wavelength=WL,
        window=backend.window, mesh=100, pml_thickness=0.1e-6,
        pml_edges=("+x", "-x"), num_modes=6,
    )
    assert other.fingerprint() != recipe
    # a lossless backend has no recipe, so lossless datasets keep their identity
    assert "backend" not in fingerprint(info, SimpleNamespace(window=None))


def test_backend_is_deterministic_across_solves(backend):
    """The dataset persists overlaps but not fields: two solves of the same
    point must agree in gauge, not just in n_eff."""
    a = backend.solve((0.0,))
    b = backend.solve((0.0,))
    assert a.neff == pytest.approx(b.neff, rel=1e-6)
    assert np.abs(a.E[0] - b.E[0]).max() < 1e-4 * np.abs(a.E[0]).max()


# ------------------------------------------------------------ rounding


def test_sharp_plates_are_unchanged_by_the_rounding_code(grid):
    x, y = grid
    a = PlasmonicSlotConverter(gap=20e-9, metal_thickness=400e-9, metal_bottom=-200e-9)
    b = PlasmonicSlotConverter(gap=20e-9, metal_thickness=400e-9, metal_bottom=-200e-9,
                               corner_radius=0.0)
    assert np.array_equal(a.index(x, y, {"w_si": 0.0}), b.index(x, y, {"w_si": 0.0}))
    assert a.fingerprint() == b.fingerprint()


def test_rounded_corners_remove_metal_only_at_the_corners():
    """1 nm grid: the corner cell of a 20 nm-rounded plate is air, a cell on
    the straight face is gold, and the removed area is the 4 (1 - pi/4) r^2 of
    four quarter-circle notches."""
    x = np.linspace(-0.5e-6, 0.5e-6, 1001)
    y = np.linspace(-0.3e-6, 0.3e-6, 601)
    r = 20e-9
    sharp = PlasmonicSlotConverter(gap=20e-9, metal_thickness=400e-9, metal_bottom=-200e-9)
    rounded = PlasmonicSlotConverter(gap=20e-9, metal_thickness=400e-9, metal_bottom=-200e-9,
                                     corner_radius=r)
    ns, nr = sharp.index(x, y, {"w_si": 0.0}), rounded.index(x, y, {"w_si": 0.0})
    inner, top = 20e-9, 200e-9
    assert _at(nr, x, y, inner + 2e-9, top - 2e-9).real == pytest.approx(1.0, abs=1e-6)  # notch
    assert _at(nr, x, y, inner + 2e-9, top - 40e-9).imag > 9                              # face
    assert _at(nr, x, y, inner + 40e-9, top - 2e-9).imag > 9                              # top
    metal_s = (np.imag(ns) > 5).sum()
    metal_r = (np.imag(nr) > 5).sum()
    removed = (metal_s - metal_r) * 1e-9 * 1e-9                      # 1 nm cells
    expected = 2 * 4 * (1 - np.pi / 4) * r * r                        # two plates, four notches each
    assert removed == pytest.approx(expected, rel=0.15)
    assert rounded.fingerprint() != sharp.fingerprint()


def test_rounded_fill_is_quantised_away_from_epsilon_near_zero():
    from em_simulation.fde.slot_converter import _rounded_rect_fill

    x = np.arange(-100, 101, 5) * 1e-9
    y = np.arange(-100, 101, 5) * 1e-9
    f = _rounded_rect_fill(x, y, -50e-9, 50e-9, -50e-9, 50e-9, 20e-9)
    partial = f[(f > 0) & (f < 1)]
    assert partial.size > 0
    assert np.allclose(partial * 64, np.round(partial * 64))
    assert partial.min() >= 1 / 64 - 1e-12
