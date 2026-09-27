"""The finite-element backend (``dbeme/fde/femwell_fde.py``).

What has to be true for a FEM dataset to be trusted downstream:

* it satisfies the ``FDEBackend`` contract and its fields land on the
  declared uniform grid, co-located, gauge-pinned, ordered guided-first;
* on a geometry the finite-difference grid *does* converge - a dielectric
  strip - the FEM modes agree with the PML backend's, in ``n_eff`` and, more
  to the point, in the overlap between the two solvers' fields evaluated on
  the same grid.  That overlap is the one quantity the method rests on, and
  the point-evaluation onto the grid is the one error this backend adds;
* on the plasmonic cross section the rounded gold corner is a curve: the
  slot mode is found, complex, with a physical loss, and the corner field is
  bounded - no anisotropic-cell blow-up, because there are no cells.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("femwell")

from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.fde.base import FDEBackend  # noqa: E402
from dbeme.fde.cross_section import FullEtchStrip  # noqa: E402
from dbeme.fde.femwell_fde import FemwellBackend  # noqa: E402
from dbeme.fde.materials import ConstantIndex  # noqa: E402
from dbeme.fde.pml import PMLBackend  # noqa: E402


def _si_strip():
    return FullEtchStrip(thickness=0.22e-6, core=ConstantIndex(3.476, name="Si"),
                         cladding=ConstantIndex(1.444, name="SiO2"), substrate=ConstantIndex(1.444, name="SiO2"))


_WINDOW = (1.0e-6, -0.7e-6, 0.7e-6)


@pytest.fixture(scope="module")
def fem():
    be = FemwellBackend(_si_strip(), target_neff=2.4, parameter_names=("top_width",),
                        wavelength=1.55e-6, window=_WINDOW, cell=10e-9, num_modes=4,
                        resolution={"core": (12e-9, 0.1e-6), "default": (60e-9, 0.0)})
    return be, be.solve((0.5e-6,))


@pytest.fixture(scope="module")
def fd():
    be = PMLBackend(_si_strip(), target_neff=2.4, parameter_names=("top_width",),
                    wavelength=1.55e-6, window=_WINDOW, mesh=201, mesh_y=141, pml_thickness=0.2e-6,
                    pml_edges=("+x", "-x", "+y", "-y"), num_modes=4, colocate=True)
    return be, be.solve((0.5e-6,))


def test_contract_and_grid(fem):
    be, data = fem
    assert isinstance(be, FDEBackend) and be.lossless is False
    x, y = be.grid()
    assert data.E.shape == (4, 3, x.size, y.size) and data.H.shape == data.E.shape
    assert np.allclose(np.diff(x), 10e-9) and x[0] == -1.0e-6 and x[-1] == 1.0e-6
    assert data.neff.shape == (4,) and np.all(np.imag(data.neff) >= 0)
    fp = be.fingerprint()
    assert fp["solver"] == "femwell" and fp["cell_m"] == 10e-9 and "resolution" in fp


def test_modes_are_guided_first_and_gauge_pinned(fem):
    be, data = fem
    # the Si strip's TE0 leads, at the textbook value for 500 x 220 nm in oxide
    assert abs(data.neff[0].real - 2.449) < 0.02
    assert data.TE_pol[0] > 0.9
    # deterministic gauge: solving again gives the same field, not its negative
    again = be.solve((0.5e-6,))
    assert np.abs(again.E[0] - data.E[0]).max() < 1e-6 * np.abs(data.E[0]).max()


def test_agrees_with_the_finite_difference_backend_on_a_dielectric(fem, fd):
    """Same strip, same grid; the FD grid is converged here, so the two
    solvers must agree - and their *fields* must overlap as the same mode."""
    (fe, dfe), (fdb, dfd) = fem, fd
    # The two discretisations converge to the common limit from opposite
    # sides - first-order FEM from below (2.4446 at 12 nm elements, 2.4451 at
    # 6 nm), the FD grid from above (2.4476 at 5 nm) - so at these cheap
    # settings they sit 3-4e-3 apart on TE0 and ~8e-3 on the weakly bound TM0.
    # The bar is that gap, not a solver-tolerance number.
    assert abs(dfe.neff[0].real - dfd.neff[0].real) < 6e-3
    assert abs(dfe.neff[1].real - dfd.neff[1].real) < 1.2e-2
    # both on the same (x, y): FD fields are on the PML solver's stretched grid,
    # whose real part is the FEM grid
    assert np.allclose(np.real(fdb.x), fe.x) and np.allclose(np.real(fdb.y), fe.y)
    x, y, neff, te, E, H = assemble([dfe, dfd], 2, lossless=False)
    O = overlap_matrix(E[0], H[1], x, y)[:2, :2]           # FEM modes x FD modes
    assert abs(O[0, 0]) > 0.99 and abs(O[1, 1]) > 0.98
    assert abs(O[0, 1]) < 0.03 and abs(O[1, 0]) < 0.03


def test_grid_sampling_reproduces_skfem_and_averages_shared_points(fem):
    """Interior grid points reproduce skfem's own point evaluation exactly;
    a point on an element edge or vertex is the mean over the elements that
    share it - so a constant field is reproduced at every grid point, the
    Lagrange weights of every point sum to one, and the window boundary,
    which lies on the grid, is such a shared set."""
    from skfem import ElementTriN1, ElementTriP1

    be, _ = fem
    params = {"top_width": 0.5e-6, "wavelength": 1.55e-6}
    mesh, basis0, eps = be._mesh_and_epsilon(params)
    basis = basis0.with_element(ElementTriN1() * ElementTriP1())      # what femwell solves on
    P, Q, shared = be._sampling(basis)
    n = be.x.size * be.y.size
    assert P.shape == (2 * n, basis.split_indices()[0].size) and Q.shape == (n, basis.split_indices()[1].size)
    assert shared.size > 2 * (be.x.size + be.y.size) - 4          # at least the window's boundary points
    assert np.allclose(np.asarray(Q.sum(axis=1)).ravel(), 1.0)
    bt, bz = basis.split_bases()
    const = bt.project(lambda x: np.stack([np.ones_like(x[0]), np.zeros_like(x[0])]))
    ex, ey = (P @ const).reshape(2, n)
    assert np.abs(ex - 1.0).max() < 1e-8 and np.abs(ey).max() < 1e-8
    rng = np.random.default_rng(0)
    interior = rng.choice(np.setdiff1d(np.arange(n), shared), 300, replace=False)
    dofs = rng.standard_normal(bt.N) + 1j * rng.standard_normal(bt.N)
    X, Y = np.meshgrid(be.x * 1e6, be.y * 1e6, indexing="ij")
    pts = np.vstack([X.ravel(), Y.ravel()])[:, interior]
    ref = np.asarray(bt.probes(pts) @ dofs).reshape(2, -1)
    mine = (P @ dofs)[np.concatenate([interior, n + interior])].reshape(2, -1)
    assert np.abs(mine - ref).max() < 1e-12 * np.abs(ref).max()
    dofz = rng.standard_normal(bz.N)
    assert np.abs((Q @ dofz)[interior] - np.asarray(bz.probes(pts) @ dofz)).max() < 1e-12


def test_axis_steps_free_the_wall_from_the_cell():
    """On the FEM backend the half_slot axis may step finer than the field
    grid; the set's slot start and end stay on it; the FD grid refuses."""
    from dbeme.platforms import kocabas_converter_dataset_info

    info = kocabas_converter_dataset_info(set_number=2, cell=5e-9, corner_radius=20e-9, solver="femwell",
                                          axes="half_slot", axis_steps={"half_slot": 1e-9})()
    hs, w = info.parameters["half_slot"], info.parameters["w_si"]
    assert np.allclose(np.diff(hs), 1e-9) and np.allclose(np.diff(w), 5e-9)
    assert np.isclose(hs, 275e-9).any() and np.isclose(hs, 125e-9).any() and np.isclose(w, 400e-9).any()
    with pytest.raises(ValueError, match="tied to the cell"):
        kocabas_converter_dataset_info(set_number=2, axes="half_slot", axis_steps={"half_slot": 1e-9})()
    with pytest.raises(ValueError, match="not on the axes"):
        kocabas_converter_dataset_info(set_number=2, solver="femwell", axes="half_slot",
                                       axis_steps={"half_slot": 4e-9})()


def test_plasmonic_slot_on_a_conforming_mesh():
    from dbeme.platforms import kocabas_converter_dataset_info

    info = kocabas_converter_dataset_info(set_number=2, cell=5e-9, corner_radius=20e-9,
                                          solver="femwell", mode_numbers=6, axes="half_slot")()
    assert info.parameters["w_si"][1] - info.parameters["w_si"][0] == pytest.approx(5e-9)
    be = info.get_fde_backend()
    data, conf = be.mode_data({"w_si": 200e-9, "half_slot": 200e-9}, be.target_neff)
    phys = np.imag(data.neff) < 0.05
    te_like = [n for n, t, p in zip(data.neff, data.TE_pol, phys) if p and t > 0.5]
    assert te_like, "no TE-like physical mode found"
    n = max(te_like, key=lambda v: v.real)
    # the rounded-corner FD answer at 5 nm was 2.036; FEM at 6 nm elements on the
    # gold sat at 2.049 in the feasibility probe; either way a bound, lossy TE mode
    assert 1.95 < n.real < 2.10 and 0 < n.imag < 0.01
    # the corner field is bounded: no cell anisotropy, no ENZ mix
    i = int(np.argmin(np.abs(data.neff - n)))
    ey = np.abs(data.E[i, 1]); ex = np.abs(data.E[i, 0])
    assert ey.max() < 3 * ex.max()
    assert be.last_solve_info["triangles"] > 5000
    with pytest.raises(ValueError, match="no meaning"):
        kocabas_converter_dataset_info(set_number=2, solver="femwell", refine=((100e-9, 300e-9, 1e-9),))()
