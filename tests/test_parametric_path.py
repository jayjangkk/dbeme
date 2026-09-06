"""Snapping arbitrary parameter functions onto a dataset grid.

``ParametricPath`` generalises the path construction that ``SingleWaveguide``
hard-codes for one-core datasets, so the properties that matter are the ones
that were bugs in the original: the device must come out the right length, must
start and end on the requested values, and must place an interface wherever any
parameter steps.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.geometry.parametric_path import ParametricPath  # noqa: E402


class FakeUpdater:
    """Just enough of a DataUpdater for path construction."""

    def __init__(self, grid):
        self.parameter_grid = grid
        self.parameter_names = list(grid)
        self.wavelength = 1.55e-6


def make_path(functions, length, grid=None, **kwargs):
    grid = grid or {
        "w1": np.round(np.arange(600, 920, 20) * 1e-9, 12),
        "w2": np.round(np.arange(180, 560, 20) * 1e-9, 12),
    }
    return ParametricPath(FakeUpdater(grid), functions, length, **kwargs)


def linear(start, stop, length):
    return lambda z: start + (stop - start) * (np.asarray(z) / length)


def test_path_length_is_preserved():
    """The classic bug: the path stopped at the last parameter *change*."""
    L = 100e-6
    path = make_path(
        {"w1": linear(860e-9, 660e-9, L), "w2": linear(200e-9, 500e-9, L)}, L
    )
    _, delta_zs = path.calc_simulation_parameters()
    assert np.sum(delta_zs) == pytest.approx(L, rel=1e-12)


def test_endpoints_land_on_the_requested_values():
    L = 100e-6
    path = make_path(
        {"w1": linear(860e-9, 660e-9, L), "w2": linear(200e-9, 500e-9, L)}, L
    )
    points, _ = path.calc_simulation_parameters()
    assert points[0] == pytest.approx((860e-9, 200e-9))
    assert points[-1] == pytest.approx((660e-9, 500e-9))


def test_a_constant_device_becomes_two_identical_sections():
    """A uniform waveguide still needs an input and an output."""
    L = 20e-6
    path = make_path({"w1": 700e-9, "w2": 300e-9}, L, max_section_length=1e3)
    points, delta_zs = path.calc_simulation_parameters()
    assert len(points) == 2
    assert points[0] == points[1]
    assert np.sum(delta_zs) == pytest.approx(L)


def test_every_grid_step_gets_its_own_section():
    """One interface per parameter change, and no more."""
    L = 100e-6
    # w1 alone, crossing five 20 nm grid steps.
    path = make_path({"w1": linear(700e-9, 800e-9, L), "w2": 300e-9}, L)
    points, _ = path.calc_simulation_parameters()
    widths = [p[0] for p in points]
    assert widths[0] == pytest.approx(700e-9)
    assert widths[-1] == pytest.approx(800e-9)
    # Monotonic and stepping one grid point at a time.
    steps = np.diff(widths)
    assert np.all(steps >= 0)
    assert np.max(steps) == pytest.approx(20e-9)


def test_long_uniform_runs_are_subdivided():
    L = 100e-6
    path = make_path(
        {"w1": 700e-9, "w2": 300e-9}, L, max_section_length=10e-6
    )
    points, delta_zs = path.calc_simulation_parameters()
    assert len(points) >= 10
    assert np.max(delta_zs) <= 10e-6 + 1e-12
    assert np.sum(delta_zs) == pytest.approx(L)
    # Subdividing a uniform run must not invent geometry.
    assert len(set(points)) == 1


def test_missing_parameter_function_is_rejected():
    with pytest.raises(ValueError, match="w2"):
        make_path({"w1": 700e-9}, 100e-6)


def test_constants_are_accepted_in_place_of_callables():
    L = 50e-6
    path = make_path({"w1": linear(700e-9, 740e-9, L), "w2": 300e-9}, L)
    points, _ = path.calc_simulation_parameters()
    assert all(p[1] == pytest.approx(300e-9) for p in points)


def test_non_uniform_axis_is_honoured():
    """A refined region must actually be used where the path passes through."""
    grid = {
        "w1": np.round(
            np.unique(np.concatenate([
                np.arange(600, 700, 20), np.arange(700, 805, 5),
                np.arange(820, 920, 20),
            ])) * 1e-9, 12
        ),
        "w2": np.round(np.arange(180, 560, 20) * 1e-9, 12),
    }
    L = 100e-6
    path = make_path(
        {"w1": linear(700e-9, 800e-9, L), "w2": 300e-9}, L, grid=grid
    )
    points, _ = path.calc_simulation_parameters()
    steps = np.diff([p[0] for p in points])
    assert np.max(steps) == pytest.approx(5e-9), "fine region was not used"
