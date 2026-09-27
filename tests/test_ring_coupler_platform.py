"""Fast invariants of the ring-coupler platform and the roughness model (`tasks/11` §2).

No mode solving: geometry, grid alignment, fingerprints, and the closed-form
pieces of the loss budget.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from em_simulation.circuit.roughness import (  # noqa: E402
    payne_lacey_alpha,
    roughness_loss_db_per_cm,
    sidewall_factor,
    slab_te_index,
)
from em_simulation.fde.base import ModeData  # noqa: E402
from em_simulation.platforms import BusRingStrips, ring_coupler_dataset_info  # noqa: E402


def test_bus_stays_fixed_as_gap_changes():
    info = ring_coupler_dataset_info()()
    cs = info.get_cross_section()
    assert isinstance(cs, BusRingStrips)
    bus_at = set()
    for gap in info.get_parameter_grid()["gap"]:
        bus, ring = cs.edges({"w1": 500e-9, "w2": 500e-9, "gap": gap})
        bus_at.add(tuple(np.round(bus, 12)))
        assert abs((ring[0] - bus[1]) - gap) < 1e-15
        assert abs((ring[1] - ring[0]) - 500e-9) < 1e-15
    assert len(bus_at) == 1, "the bus must not move with the gap"


def test_gap_axis_edges_sit_on_cell_boundaries():
    for kw in ({}, dict(cell=20e-9, fine_step=20e-9, coarse_step=40e-9)):
        info = ring_coupler_dataset_info(**kw)()
        cell = kw.get("cell", 10e-9)
        half, y0, y1 = info._window
        x0 = -half
        cs = info.get_cross_section()
        for gap in info.get_parameter_grid()["gap"]:
            for width in info.get_parameter_grid()["w1"]:
                bus, ring = cs.edges({"w1": width, "w2": width, "gap": gap})
                for edge in (*bus, *ring):
                    k = (edge - x0) / cell
                    assert abs(k - round(k)) < 1e-6, (kw, gap, width, edge)


def test_cell_must_divide_window():
    with pytest.raises(ValueError):
        ring_coupler_dataset_info(cell=30e-9)


def test_fingerprint_carries_the_frame_and_gap_max():
    a = ring_coupler_dataset_info()().get_cross_section().fingerprint()
    b = ring_coupler_dataset_info(gap_max=0.6e-6)().get_cross_section().fingerprint()
    assert "bus_fixed" in a and a != b


def test_slab_index_is_the_textbook_value():
    # 220 nm Si slab in silica at 1550 nm: TE0 ~ 2.83
    n = slab_te_index(220e-9, 3.476, 1.444, 1.55e-6)
    assert 2.80 < n < 2.86


def test_payne_lacey_scales_as_sigma_squared_and_is_order_db_per_cm():
    args = (500e-9, 220e-9, 3.476, 1.444, 1.55e-6)
    l1, _ = roughness_loss_db_per_cm(*args, 2e-9, 50e-9)
    l2, _ = roughness_loss_db_per_cm(*args, 4e-9, 50e-9)
    assert abs(l2 / l1 - 4.0) < 1e-9
    assert 0.5 < l1 < 20.0          # the model is order-of-magnitude; 500 x 220 strips show 2-3 dB/cm
    assert payne_lacey_alpha(500e-9, 2.83, 1.444, 2.45, 1.55e-6, 0.0, 50e-9) == 0.0


def test_sidewall_factor_on_a_gaussian_is_the_boundary_intensity_fraction():
    x = np.linspace(-1e-6, 1e-6, 401)
    y = np.linspace(-0.5e-6, 0.5e-6, 101)
    sx, sy = 0.25e-6, 0.12e-6
    field = np.exp(-x[:, None] ** 2 / (2 * sx ** 2)) * np.exp(-y[None, :] ** 2 / (2 * sy ** 2))
    E = np.zeros((1, 3, len(x), len(y)), dtype=complex)
    E[0, 0] = field
    md = ModeData(x=x, y=y, E=E, H=np.zeros_like(E), neff=np.array([2.4 + 0j]), TE_pol=np.array([1.0]))
    F, per = sidewall_factor(md, 0, (-0.25e-6, 0.25e-6), (-0.11e-6, 0.11e-6))
    # analytic: two lines at x = +-sx, |E|^2 = exp(-1) there, over |y| < 0.11 um
    intensity = field ** 2
    total = np.trapezoid(np.trapezoid(intensity, y, axis=1), x)
    rows = np.abs(y) <= 0.11e-6
    line = np.exp(-1.0) * np.exp(-y[rows] ** 2 / sy ** 2)
    expected = 2 * np.trapezoid(line, y[rows]) / total
    assert abs(F / expected - 1.0) < 0.02
    assert abs(per[0] / per[1] - 1.0) < 1e-12
