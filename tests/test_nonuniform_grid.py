"""Piecewise-refined solve grids (``PMLModeSolver(refine_x=..., refine_y=...)``).

The finite-difference operator always took per-cell spacings; what a
refinement needed fixing was everything that inferred cell geometry from the
cell *centres* alone - the cross section's fill fractions, the rounded-corner
fill, and the unweighted power sums behind confinement and the TE fraction.
Each of those is pinned here, on a uniform grid (must be unchanged) and on a
refined one (must be exact), and then the whole thing is used the way the
Kocabas tip platform uses it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.fde.cross_section import FullEtchStrip, _cell_edges, _fill_fraction  # noqa: E402
from em_simulation.fde.materials import ConstantIndex, silica  # noqa: E402
from em_simulation.fde.pml import PMLBackend, PMLModeSolver, refined_axis  # noqa: E402
from em_simulation.fde.slot_converter import _rounded_rect_fill  # noqa: E402


# ------------------------------------------------------------- the axis


def test_refined_axis_keeps_every_base_node_and_cuts_the_region():
    base = np.linspace(-10.0, 10.0, 21)                 # 1.0 pitch
    z = refined_axis(base, [(-2.0, 2.0, 0.25)])
    assert np.all(np.isin(base, z))
    inside = z[(z >= -2.0) & (z <= 2.0)]
    assert np.allclose(np.diff(inside), 0.25)
    outside = np.diff(z[z < -2.0])
    assert np.allclose(outside, 1.0)
    assert z.size == 21 + 4 * 3                          # four base cells, three new nodes each
    assert np.all(np.diff(z) > 0)


def test_refined_axis_with_no_regions_is_the_base():
    base = np.linspace(0.0, 1.0, 11)
    assert refined_axis(base, []) is not None
    assert np.array_equal(refined_axis(base, []), base)


@pytest.mark.parametrize("regions,message", [
    ([(-2.3, 2.0, 0.5)], "base-grid nodes"),
    ([(-2.0, 2.0, 0.3)], "integer multiple"),
    ([(-9.0, 2.0, 0.5)], "two base cells"),
    ([(-2.0, 9.0, 0.5)], "two base cells"),
    ([(-3.0, 1.0, 0.5), (0.0, 3.0, 0.5)], "overlap"),
    ([(2.0, 2.0, 0.5)], "empty"),
])
def test_refined_axis_rejects_what_would_break_alignment(regions, message):
    base = np.linspace(-10.0, 10.0, 21)
    with pytest.raises(ValueError, match=message):
        refined_axis(base, regions)


# ---------------------------------------------------- cell geometry


def _nodes_and_centres():
    nodes = refined_axis(np.linspace(-1.0, 1.0, 21), [(-0.2, 0.2, 0.025)])
    return nodes, 0.5 * (nodes[:-1] + nodes[1:])


def test_cell_edges_recover_the_nodes_of_a_refined_grid():
    nodes, centres = _nodes_and_centres()
    assert np.allclose(_cell_edges(centres), nodes, atol=1e-12)


def test_cell_edges_are_the_midpoints_on_a_uniform_grid():
    centres = np.linspace(0.05, 0.95, 10)
    edges = _cell_edges(centres)
    assert np.allclose(edges, np.linspace(0.0, 1.0, 11))


def test_cell_edges_refuse_an_unequal_outermost_pair():
    nodes = np.array([0.0, 0.5, 1.5, 2.5, 3.5])          # first cell half-size
    centres = 0.5 * (nodes[:-1] + nodes[1:])
    with pytest.raises(ValueError, match="outermost"):
        _cell_edges(centres)


def test_fill_fraction_is_exact_across_a_refinement_boundary():
    """A feature edge on the transition node fills its cells 0 or 1, never a
    fraction - which the midpoint-of-centres rule got wrong by a fifth of a
    cell there."""
    nodes, centres = _nodes_and_centres()
    f = _fill_fraction(centres, -0.2, 0.2)               # edges on the transition nodes
    assert set(np.round(f, 9)) <= {0.0, 1.0}
    assert np.isclose(f[(centres > -0.2) & (centres < 0.2)].min(), 1.0)
    assert np.isclose(f[(centres < -0.2) | (centres > 0.2)].max(), 0.0)
    # and an edge inside a fine cell fills that fine cell by its own width
    f = _fill_fraction(centres, -0.2, 0.0125)            # half of one fine cell
    i = int(np.argmin(np.abs(centres - 0.0125)))
    assert np.isclose(f[i], 0.5)


def test_fill_fraction_is_unchanged_on_a_uniform_grid():
    centres = np.linspace(-0.95, 0.95, 20)
    lo, hi = -0.37, 0.52
    edges = np.r_[centres[0] - 0.05, 0.5 * (centres[:-1] + centres[1:]), centres[-1] + 0.05]
    want = np.clip((np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)) / 0.1, 0, 1)
    assert np.allclose(_fill_fraction(centres, lo, hi), want)


def test_rounded_fill_uses_each_cells_own_width():
    """A rectangle with corners rounded by less than a fine cell is, on the
    straight faces, exactly the flat fill; the arcs are sub-sampled."""
    nodes, centres = _nodes_and_centres()
    y_nodes = np.linspace(-0.5, 0.5, 11)
    y_centres = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    sharp = np.outer(_fill_fraction(centres, -0.2, 0.2), _fill_fraction(y_centres, -0.3, 0.3))
    rounded = _rounded_rect_fill(centres, y_centres, -0.2, 0.2, -0.3, 0.3, radius=0.0)
    assert np.allclose(rounded, sharp)


# ----------------------------------------------------------- the solver


def _sin_strip():
    return FullEtchStrip(thickness=0.4e-6, core=ConstantIndex(1.996, name="Si3N4"),
                         cladding=silica(out_of_range="raise"))


_KW = dict(window=(2.0e-6, -1.0e-6, 1.0e-6), pml_thickness=0.5e-6,
           pml_edges=("+x", "-x"), num_modes=3, colocate=True)


def test_refinement_enters_the_fingerprint_only_when_used():
    plain = PMLBackend(_sin_strip(), target_neff=1.6, mesh=41, **_KW)
    fine = PMLBackend(_sin_strip(), target_neff=1.6, mesh=41,
                      refine_x=((-0.5e-6, 0.5e-6, 0.05e-6),), **_KW)
    assert "refine_x" not in plain.fingerprint() and "refine_y" not in plain.fingerprint()
    assert fine.fingerprint()["refine_x"] == [[-0.5e-6, 0.5e-6, 0.05e-6]]
    assert len(fine.x) == 41 + 10 * 1                    # ten base cells split in two
    assert fine.pml_start == plain.pml_start


def test_refinement_may_not_reach_the_pml():
    with pytest.raises(ValueError, match="PML"):
        PMLModeSolver(_sin_strip(), mesh=41, refine_x=((-1.9e-6, 0.0, 0.05e-6),), **_KW)


def test_uniform_grid_results_are_unaffected():
    """The area weights are a constant there and cancel; two ARPACK runs
    differ in their last digit, so the comparison is to solver tolerance."""
    a = PMLModeSolver(_sin_strip(), mesh=41, **_KW)
    b = PMLModeSolver(_sin_strip(), mesh=41, refine_x=(), **_KW)
    da, ca = a.mode_data({"top_width": 1.0e-6}, 1.6)
    db, cb = b.mode_data({"top_width": 1.0e-6}, 1.6)
    assert np.allclose(da.neff, db.neff, rtol=1e-6, atol=1e-9)
    assert np.allclose(ca, cb, atol=1e-6) and np.allclose(da.TE_pol, db.TE_pol, atol=1e-6)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)


def test_refining_the_core_moves_the_index_toward_the_fine_answer():
    """Coarse everywhere, fine everywhere, and fine only where the core is:
    the third must sit much nearer the second than the first does, at a
    fraction of the unknowns.  The core strip is what a refinement is for."""
    kw = dict(_KW, num_modes=2)
    width = 0.8e-6
    coarse = PMLModeSolver(_sin_strip(), mesh=41, mesh_y=21, **kw)            # 0.1 um
    fine = PMLModeSolver(_sin_strip(), mesh=81, mesh_y=41, **kw)              # 0.05 um
    refined = PMLModeSolver(_sin_strip(), mesh=41, mesh_y=21,                 # 0.05 um in the core only
                            refine_x=((-0.6e-6, 0.6e-6, 0.05e-6),),
                            refine_y=((-0.5e-6, 0.5e-6, 0.05e-6),), **kw)
    n = {}
    for label, solver in (("coarse", coarse), ("fine", fine), ("refined", refined)):
        data, conf = solver.mode_data({"top_width": width}, 1.6)
        n[label] = data.neff[int(np.argmax(conf))].real
    assert len(refined.x) < len(fine.x)
    assert abs(n["refined"] - n["fine"]) < 0.4 * abs(n["coarse"] - n["fine"])
    assert abs(n["refined"] - n["fine"]) < 2e-3


def test_confinement_is_area_weighted_on_a_refined_grid():
    """A mode's core fraction must not depend on how many cells the core is
    cut into.  Coarse-vs-refined confinement of the same guided mode agree
    to a few percent; an unweighted sum would over-count the fine cells."""
    kw = dict(_KW, num_modes=2)
    coarse = PMLModeSolver(_sin_strip(), mesh=41, mesh_y=21, **kw)
    refined = PMLModeSolver(_sin_strip(), mesh=41, mesh_y=21,
                            refine_x=((-0.6e-6, 0.6e-6, 0.025e-6),), **kw)
    _, c0 = coarse.mode_data({"top_width": 0.8e-6}, 1.6)
    _, c1 = refined.mode_data({"top_width": 0.8e-6}, 1.6)
    # the mask itself moves by half a coarse cell between node and centre
    # sampling (index_profile), which is worth ~0.1 on this small grid; an
    # unweighted sum over a core holding four cells per coarse one would
    # have pushed the fraction toward 1
    assert abs(c0.max() - c1.max()) < 0.15
    assert c1.max() < 0.9


# --------------------------------------------------- the Kocabas tip


def test_kocabas_tip_platform_keeps_every_edge_on_a_node():
    from em_simulation.platforms import kocabas_converter_dataset_info, kocabas_path

    cell, fine, extent = 25e-9, 5e-9, 50e-9
    info = kocabas_converter_dataset_info(set_number=2, cell=cell, tip_refine=(extent, fine))()
    assert tuple(info.parameter_names) == ("w_si", "half_slot")
    backend = info.get_fde_backend()
    x = np.real(backend.x)
    assert backend.fingerprint()["refine_x"] == [[-extent, extent, fine]]
    assert np.isclose(np.diff(x)[np.abs(x[:-1]) < extent - 1e-12], fine).all()
    # the Si edge of every width and the wall of every half-slot are nodes,
    # so every point a path can visit - detours included - is aligned
    for w in info.parameters["w_si"]:
        assert np.abs(x - 0.5 * w).min() < 1e-15
    for h in info.parameters["half_slot"]:
        assert np.abs(x - h).min() < 1e-15
    # the width axis is fine inside the strip and coarse outside it
    w = info.parameters["w_si"]
    assert np.allclose(np.diff(w[w <= 2 * extent + 1e-15]), 2 * fine)
    assert np.allclose(np.diff(w[w >= 2 * extent - 1e-15]), 2 * cell)
    # the two path parameterisations describe the same device
    a, length = kocabas_path(2)
    b, length_b = kocabas_path(2, half_slot=True)
    z = np.linspace(0, length, 50)
    assert length == length_b
    assert np.allclose(b["half_slot"](z) - 0.5 * b["w_si"](z), a["gap"](z))
    section = info.get_cross_section()
    assert np.isclose(section.index(x, np.real(backend.y), {"w_si": 40e-9, "half_slot": 125e-9})[
        int(np.argmin(np.abs(x - 0.0))), len(backend.y) // 2].real, np.sqrt(12.085), atol=0.05)


def test_kocabas_wall_strip_refines_the_wall_axis_and_leaves_the_width_axis_coarse():
    """The strip the gold wall moves through: fine wall steps, 10 nm width
    steps as before, every edge on a node - including the strip's own
    boundaries and the slot end."""
    from em_simulation.platforms import kocabas_converter_dataset_info

    cell, fine = 25e-9, 5e-9
    info = kocabas_converter_dataset_info(set_number=2, cell=cell, refine=((100e-9, 300e-9, fine),))()
    assert tuple(info.parameter_names) == ("w_si", "half_slot")
    backend = info.get_fde_backend()
    x = np.real(backend.x)
    assert backend.fingerprint()["refine_x"] == [[-300e-9, -100e-9, fine], [100e-9, 300e-9, fine]]
    mid = np.abs(0.5 * (x[:-1] + x[1:]))                             # cells by their midpoint
    assert np.allclose(np.diff(x)[(mid > 100e-9) & (mid < 300e-9)], fine)
    assert np.allclose(np.diff(x)[(mid < 100e-9) | (mid > 300e-9)], cell)
    w, h = info.parameters["w_si"], info.parameters["half_slot"]
    assert np.allclose(np.diff(w), 2 * cell)                       # width axis untouched
    fine_part = h[(h >= 100e-9 - 1e-12) & (h <= 300e-9 + 1e-12)]
    assert np.allclose(np.diff(fine_part), fine)                   # wall axis fine inside the strip
    assert np.allclose(np.diff(h[h >= 300e-9 - 1e-12]), cell)     # and coarse beyond it
    for v in w:
        assert np.abs(x - 0.5 * v).min() < 1e-15
    for v in h:
        assert np.abs(x - v).min() < 1e-15


def test_kocabas_tip_refine_is_the_zero_strip():
    from em_simulation.platforms import kocabas_converter_dataset_info

    a = kocabas_converter_dataset_info(set_number=2, cell=25e-9, tip_refine=(50e-9, 5e-9))()
    b = kocabas_converter_dataset_info(set_number=2, cell=25e-9, refine=((0.0, 50e-9, 5e-9),))()
    assert np.array_equal(a.parameters["w_si"], b.parameters["w_si"])
    assert np.array_equal(a.parameters["half_slot"], b.parameters["half_slot"])
    assert a.get_fde_backend().fingerprint() == b.get_fde_backend().fingerprint()
    with pytest.raises(ValueError, match="not both"):
        kocabas_converter_dataset_info(set_number=2, cell=25e-9, tip_refine=(50e-9, 5e-9),
                                       refine=((0.0, 50e-9, 5e-9),))()


def test_kocabas_refine_rejects_off_grid_strips():
    from em_simulation.platforms import kocabas_converter_dataset_info

    with pytest.raises(ValueError, match="base grid"):
        kocabas_converter_dataset_info(set_number=2, cell=25e-9, refine=((110e-9, 300e-9, 5e-9),))()
    with pytest.raises(ValueError, match="integer multiple"):
        kocabas_converter_dataset_info(set_number=2, cell=25e-9, refine=((100e-9, 300e-9, 7e-9),))()
