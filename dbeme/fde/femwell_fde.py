r"""A finite-element mode solver behind :class:`~dbeme.fde.base.FDEBackend`.

Why a second lossy backend exists.  :mod:`pml` solves on a uniform
finite-difference grid, and on a plasmonic cross section that grid *is* the
accuracy floor: a rounded gold corner is a staircase of 5 nm cells, and the
gap plasmon - which lives on the ~23 nm skin depth and the corner's curvature -
moves by 0.03-0.05 in ``n_eff`` between 5, 2.5 and 1 nm cells without
converging (report 13 section 7).  A body-conforming mesh describes the arc as
a curve; the mode then converges with element size, and the dataset's
parameter axes are no longer tied to a cell pitch, because a wall at 202.5 nm
is just another geometry rather than an edge inside a cell.

What this backend does, in order:

1. asks the cross section for its geometry as shapely polygons
   (:meth:`CrossSection.polygons`), one per material region, and meshes them
   with gmsh through femwell (``mesh_from_OrderedDict``) with per-region
   resolutions - fine on the metal, coarse in the cladding;
2. solves the vectorial eigenproblem with femwell's ``compute_modes``
   (Nedelec N1 x Lagrange P1 elements, shift-invert about ``target_neff``);
   the permittivity is complex where the material is, so gold works as it
   should;
3. absorbs outgoing radiation in a **lossy outer ring** rather than a
   coordinate-stretch PML - femwell has no stretched coordinates, and an
   absorber of a few tenths in ``epsilon''`` over 0.3 um does the same job for
   a mode whose tail has decayed by the window edge (the same standoff rule as
   the PML: the ring must sit outside the field);
4. **evaluates E and H on the dataset's uniform** ``(x, y)`` **grid** by
   point evaluation of the finite-element functions.  Every grid point is
   located in the triangulation once per solve (a trapezoid map, O(log n)
   per point), the two probe matrices - Nedelec for the transverse
   components, Lagrange for the longitudinal one - are built once and
   applied to all modes at once, and a grid point that lies on an element
   edge, which every point on a material interface along a grid line does,
   takes the mean over the elements sharing that edge (all elements at a
   vertex): the Nedelec element's normal component is discontinuous across
   edges by the discretisation error, so a one-sided value would depend on
   which element a locator happened to pick.  Everything downstream -
   normalisation, biorthogonalisation, the overlap integrals, the cascade -
   is unchanged, because it only ever sees fields on the common grid.  That
   sampling is the one thing this backend adds to the error budget, and it
   is what the tests pin: the FEM and FD backends must agree on a dielectric
   strip, where the FD grid *is* converged.

The fields femwell returns are co-located by construction (both E and H are
finite-element functions evaluated at the same points), so there is no
``colocate`` question here.  Modes are ordered as :mod:`pml` orders them -
guided first, then by decreasing ``Re n_eff`` - and gauge-pinned the same way,
so a FEM dataset behaves like a PML one in every downstream test.

Units: femwell works in micrometres; this module converts at the boundary.
"""

import time
import warnings
from collections import OrderedDict

import numpy as np

from .base import FDEBackend, ModeData
from .emepy_fde import _pin_gauge

__all__ = ["FemwellBackend", "FemwellModeSolver"]

_M_TO_UM = 1e6


class FemwellModeSolver:
    """Complex modes of a cross section on a boundary-conforming FEM mesh.

    :param cross_section: A :class:`~dbeme.fde.cross_section.CrossSection`
        that implements :meth:`polygons`.
    :param wavelength: Metres.
    :param window: ``(half_width, y_min, y_max)`` or ``(x_min, x_max, y_min,
        y_max)`` in metres - the *inner* window the fields are returned on.
    :param cell: Pitch of the uniform output grid, metres.  This is the grid
        the dataset stores overlaps on; it need not resolve the geometry, only
        the field, since the geometry is resolved by the mesh.
    :param absorber_thickness: Depth of the lossy ring outside the window,
        metres.
    :param absorber_loss: ``epsilon''`` added in the ring.  A few tenths
        absorbs a radiating tail over 0.3 um without reflecting a bound mode.
    :param resolution: ``{region: (element_size_m, distance_m)}`` per named
        region, plus ``"default"`` for the maximum element size elsewhere.
    :param order: Finite-element order, 1 or 2.
    :param num_modes: Eigenpairs to request about the target.
    :param confinement_threshold: Minimum core power fraction for a mode to
        count as guided - same discriminator as :mod:`pml`.
    """

    def __init__(
        self,
        cross_section,
        wavelength=1.55e-6,
        window=(1.2e-6, -0.85e-6, 0.85e-6),
        cell=5e-9,
        absorber_thickness=0.3e-6,
        absorber_loss=0.5,
        resolution=None,
        order=1,
        num_modes=10,
        confinement_threshold=0.05,
    ):
        self.cross_section = cross_section
        self._wavelength = float(wavelength)
        self.window = tuple(float(v) for v in window)
        self.cell = float(cell)
        self.absorber_thickness = float(absorber_thickness)
        self.absorber_loss = float(absorber_loss)
        self.resolution = dict(resolution or {})
        self.order = int(order)
        self._num_modes = int(num_modes)
        self.confinement_threshold = float(confinement_threshold)
        self.last_solve_info = {}

        if len(self.window) == 4:
            x_min, x_max, y_min, y_max = self.window
        else:
            half_width, y_min, y_max = self.window
            x_min, x_max = -half_width, half_width
        self.x_min, self.x_max, self.y_min, self.y_max = (float(v) for v in (x_min, x_max, y_min, y_max))
        # a uniform grid whose nodes include the window edges; the stored
        # overlaps live on it, so it is fixed for the life of a dataset
        nx = int(round((self.x_max - self.x_min) / self.cell)) + 1
        ny = int(round((self.y_max - self.y_min) / self.cell)) + 1
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.y = np.linspace(self.y_min, self.y_max, ny)

    # ------------------------------------------------------------------ API

    @property
    def num_modes(self):
        return self._num_modes

    @num_modes.setter
    def num_modes(self, value):
        self._num_modes = int(value)

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def lossless(self):
        """Never: the absorber alone makes every mode complex."""
        return False

    @property
    def cladding_index(self):
        return self.cross_section.cladding_index_at(self._wavelength)

    def grid(self):
        return self.x, self.y

    def solve_grid(self):
        """Same grid: the fields are *evaluated* on it, not solved on it."""
        return self.x, self.y

    def index_profile(self, params):
        """Real index on the output grid, for plotting and the core mask."""
        return self.cross_section.index(self.x, self.y, params)

    # ------------------------------------------------------------- meshing

    def _mesh_and_epsilon(self, params):
        """A gmsh mesh of the cross section plus the absorbing ring, and the
        piecewise-constant complex permittivity on it."""
        import shapely
        from shapely.geometry import box
        from skfem import Basis, ElementTriP0
        from skfem.io.meshio import from_meshio
        from femwell.mesh import mesh_from_OrderedDict

        wl = self.cross_section.wavelength_of(params)
        regions = self.cross_section.polygons(params)          # OrderedDict name -> (polygon [m], eps)
        s = _M_TO_UM
        inner = box(self.x_min * s, self.y_min * s, self.x_max * s, self.y_max * s)
        t = self.absorber_thickness * s
        outer = box(self.x_min * s - t, self.y_min * s - t, self.x_max * s + t, self.y_max * s + t)
        shapes, eps_of = OrderedDict(), {}
        for name, (poly, eps) in regions.items():
            scaled = shapely.affinity.scale(poly, xfact=s, yfact=s, origin=(0, 0))
            shapes[name] = shapely.intersection(scaled, inner)
            eps_of[name] = complex(eps)
        # the background: what the cross section says fills the rest, and the
        # same medium with added loss in the ring
        eps_bg = complex(self.cross_section.background_epsilon(params))
        shapes["background"] = inner
        shapes["absorber"] = outer
        eps_of["background"] = eps_bg
        eps_of["absorber"] = eps_bg + 1j * self.absorber_loss

        res = {}
        for name in regions:
            if name in self.resolution:
                size, dist = self.resolution[name]
                res[name] = dict(resolution=size * s, distance=dist * s)
        default_max = self.resolution.get("default", (0.08e-6, 0.0))[0] * s
        mesh = from_meshio(mesh_from_OrderedDict(shapes, resolutions=res, default_resolution_max=default_max))
        basis0 = Basis(mesh, ElementTriP0())
        eps = basis0.zeros(dtype=complex)
        for name, value in eps_of.items():
            dofs = basis0.get_dofs(elements=name)
            eps[dofs] = value
        return mesh, basis0, eps

    # ------------------------------------------------------------- solving

    def mode_data(self, params, target_neff, num_modes=None):
        """The ``n`` best-confined modes near the target, evaluated on the grid.

        :returns: ``(ModeData, confinement)`` ordered guided-first then by
            decreasing ``Re n_eff``, exactly as :mod:`pml` does.
        """
        from femwell.maxwell.waveguide import compute_modes

        wanted = int(num_modes or self.num_modes)
        t0 = time.time()
        mesh, basis0, eps = self._mesh_and_epsilon(params)
        t_mesh = time.time() - t0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modes = compute_modes(
                basis0, eps, wavelength=self._wavelength * _M_TO_UM, num_modes=wanted,
                order=self.order, n_guess=float(np.real(target_neff)), solver="scipy",
            )
        t_solve = time.time() - t0 - t_mesh
        neff = np.array([complex(m.n_eff) for m in modes.modes])
        # e^{i beta z}: an absorbing mode has Im > 0
        neff = np.array([complex(n.real, abs(n.imag)) for n in neff])

        nx, ny = self.x.size, self.y.size
        E = np.zeros((wanted, 3, nx, ny), dtype=np.complex128)
        H = np.zeros((wanted, 3, nx, ny), dtype=np.complex128)
        found = list(modes.modes)
        shared = np.zeros(0, dtype=int)
        if found:
            basis = found[0].basis
            if any(m.basis is not basis for m in found):
                raise RuntimeError("femwell returned modes on different bases")
            P, Q, shared = self._sampling(basis)
            it, iz = basis.split_indices()
            k = len(found)
            for field, dofs in ((E, np.stack([m.E for m in found], axis=1)),
                                (H, np.stack([m.H for m in found], axis=1))):
                field[:k, :2] = (P @ dofs[it]).T.reshape(k, 2, nx, ny)
                field[:k, 2] = (Q @ dofs[iz]).T.reshape(k, nx, ny)
        t_eval = time.time() - t0 - t_mesh - t_solve

        confinement = self._confinement(E, H, params)
        guided = confinement >= self.confinement_threshold
        order = np.lexsort((-np.real(neff), ~guided))
        E, H, neff, confinement = E[order], H[order], neff[order], confinement[order]
        te = np.zeros(wanted)
        for i in range(wanted):
            total = np.sum(np.abs(E[i, 0]) ** 2 + np.abs(E[i, 1]) ** 2)
            te[i] = float(np.sum(np.abs(E[i, 0]) ** 2) / total) if total else 0.0
            _pin_gauge(E[i], H[i])
        self.last_solve_info = dict(nodes=int(mesh.p.shape[1]), triangles=int(mesh.t.shape[1]),
                                    shared_points=int(shared.size),
                                    t_mesh=t_mesh, t_solve=t_solve, t_eval=t_eval)
        return ModeData(x=self.x, y=self.y, E=E, H=H, neff=neff, TE_pol=te), confinement

    # ------------------------------------------------------------ sampling

    def _sampling(self, basis):
        """The probe matrices ``(P, Q)`` that carry a solution's transverse
        (Nedelec) and longitudinal (Lagrange) dofs onto the uniform grid,
        rows component-major then grid point (``P`` has ``2 nx ny`` rows), and
        the indices of the grid points that were averaged over several
        elements.

        Location is a trapezoid map of the triangulation, exact and O(log n)
        per point, built once per solve.  skfem's own ``probes`` locates with
        a finder that tests every point of a batch against every candidate
        element - quadratic in the batch - and had to be called per mode and
        per 4000-point chunk: 88 s of a 160 s solve on the Kocabas mesh
        (report 13 section 8).  This path costs half a second for all modes.

        A grid point on an element edge (one vanishing barycentric weight)
        or vertex (two) is averaged over every element that contains it.
        The Nedelec element's normal component jumps across edges by the
        discretisation error - 0.2 % on a guided mode, 10 % on a continuum
        mode of the same solve - so the one-sided value skfem's finder used
        to return depended on which element it happened to pick; the mean is
        also what the finite-difference backend's co-location gives at an
        interface node.  Every material edge that coincides with a grid line
        is such a set of points.
        """
        from matplotlib.tri import Triangulation

        mesh = basis.mesh
        bt, bz = basis.split_bases()
        X, Y = np.meshgrid(self.x * _M_TO_UM, self.y * _M_TO_UM, indexing="ij")
        pts = np.vstack([X.ravel(), Y.ravel()])
        n = pts.shape[1]
        cells = np.asarray(Triangulation(mesh.p[0], mesh.p[1], mesh.t.T).get_trifinder()(pts[0], pts[1]))
        if (cells < 0).any():
            raise RuntimeError(f"{np.count_nonzero(cells < 0)} grid points lie outside the mesh")
        loc = bt.mapping.invF(pts[:, :, None], tind=cells)[:, :, 0]
        bary = np.vstack([1.0 - loc[0] - loc[1], loc[0], loc[1]])   # weights of vertices t[0], t[1], t[2]
        touch = np.abs(bary) > 1e-9
        shared = np.flatnonzero(touch.sum(axis=0) < 3)
        keep = np.ones(n, dtype=bool)
        keep[shared] = False
        point, cell, weight = [np.flatnonzero(keep)], [cells[keep]], [np.ones(int(keep.sum()))]
        for p in shared:
            verts = mesh.t[touch[:, p], cells[p]]
            elems = np.flatnonzero(np.isin(mesh.t, verts).sum(axis=0) == verts.size)
            point.append(np.full(elems.size, p))
            cell.append(elems)
            weight.append(np.full(elems.size, 1.0 / elems.size))
        point, cell, weight = (np.concatenate(v) for v in (point, cell, weight))
        return self._probe_matrix(bt, pts, point, cell, weight), self._probe_matrix(bz, pts, point, cell, weight), shared

    @staticmethod
    def _probe_matrix(basis, pts, point, cell, weight):
        """skfem's ``CellBasis.probes`` with the containing element supplied
        per entry and a weight per entry, so that one grid point may draw on
        several elements.  Rows are component-major, then grid point."""
        from scipy.sparse import coo_matrix

        loc = basis.mapping.invF(pts[:, point][:, :, None], tind=cell)
        phis = np.array([basis.elem.gbasis(basis.mapping, loc, k, tind=cell)[0] for k in range(basis.Nbfun)])
        comp = int(np.prod(basis._base_tensor_order))
        phis = (phis.reshape(basis.Nbfun, comp, -1) * weight).reshape(-1)
        n = pts.shape[1]
        rows = np.tile(np.concatenate([point + c * n for c in range(comp)]), basis.Nbfun)
        cols = basis.element_dofs[:, np.tile(cell, comp)].reshape(-1)
        return coo_matrix((phis, (rows, cols)), shape=(comp * n, basis.N)).tocsr()

    def _confinement(self, E, H, params):
        """Fraction of transverse magnetic power inside the core, on the grid."""
        region = getattr(self.cross_section, "core_mask", None)
        if region is not None:
            core = np.asarray(region(self.x, self.y, params), dtype=bool)
        else:
            index = np.real(self.index_profile(params))
            column_max = index.max(axis=1, keepdims=True)
            column_min = index.min(axis=1, keepdims=True)
            core = index > 0.5 * (column_max + column_min)
        out = np.empty(E.shape[0])
        for i in range(E.shape[0]):
            power = np.abs(H[i, 0]) ** 2 + np.abs(H[i, 1]) ** 2
            total = power.sum()
            out[i] = power[core].sum() / total if total else 0.0
        return out


class FemwellBackend(FemwellModeSolver, FDEBackend):
    """:class:`FemwellModeSolver` behind the :class:`FDEBackend` contract.

    :param target_neff: Shift-invert target for every point (CLAUDE.md
        section 5.13a: among the physical branches, not below them).
    """

    #: Bumped when stored FEM modes change meaning with no parameter changing.
    #: 1: fields sampled one-sided at element edges, whichever element skfem's
    #: finder returned.  2 (2026-09-18): grid points on an edge or vertex are
    #: averaged over the elements sharing it (:meth:`_sampling`).
    MESH_CONVENTION = 2

    def __init__(self, cross_section, target_neff, parameter_names=None, **kwargs):
        super().__init__(cross_section, **kwargs)
        self.target_neff = float(np.real(target_neff))
        self.parameter_names = tuple(
            parameter_names if parameter_names is not None else cross_section.parameter_names
        )

    def fingerprint(self):
        """The mesh and absorber recipe, for the dataset identity."""
        return {
            "solver": "femwell",
            "mesh_convention": self.MESH_CONVENTION,
            "order": self.order,
            "cell_m": self.cell,
            "absorber_thickness_m": self.absorber_thickness,
            "absorber_loss": self.absorber_loss,
            "resolution": {k: [float(v) for v in vals] for k, vals in sorted(self.resolution.items())},
            "target_neff": self.target_neff,
            "confinement_threshold": self.confinement_threshold,
        }

    def solve(self, parameter_point):
        params = dict(zip(self.parameter_names, parameter_point))
        params.setdefault("wavelength", self._wavelength)
        data, _ = self.mode_data(params, self.target_neff)
        return data
