r"""A PML mode solver: complex ``n_eff``, and therefore radiation loss.

Phase 2 Route A of ``tasks/02_pml_backend.md``.  ``EmepyFDE`` returns a real
``n_eff``, so bend, substrate and metal loss are all identically zero (5.9).
This module adds the missing piece by giving ``EMpy``'s vectorial solver a
complex-stretched grid.

:class:`PMLModeSolver` is the solver; :class:`PMLBackend` puts it behind
:class:`~em_simulation.fde.base.FDEBackend`, so a *dataset* can be built on a
lossy basis (``datasets/Si_plasmonic_slot_1550``, report 12).  That wiring
waited for Phase 3's mode-count study - a PML-EME basis needs tens of modes
rather than six and DBEME stores ``(2N x 2N)`` overlaps per adjacent
grid-point pair - which the scattering cascade settled (report 06 sections
7-8): usable at N = 4...40 on the direct route, and ``SMATRIX_METHOD =
"auto"`` picks that route whenever the backend declares itself lossy.

What had to be got right
------------------------
**The axis convention is not what emepy assumes.**  ``stretchmesh``'s
``nlayers`` is ordered ``[+y, -y, +x, -x]``, but ``MSEMpy`` slices its fields
``[nlayers[1]:-nlayers[0], nlayers[3]:-nlayers[2]]`` with **x** as the first
axis, and its commented-out PML recipe (``fd.py:400``) builds
``[layer_xp, layer_xn, 0, 0]`` and then calls
``stretchmesh(self.x, np.zeros(1), ...)``, which sends the stretch to a dummy
array.  emepy's intended PML was axis-swapped as written.  Putting the layers on
the wrong axis does not fail loudly - it quietly absorbs the guided mode, and
the symptom is a large, radius-independent "loss".

**Mode selection cannot be by effective index.**  ``VFDModeSolver.solve`` calls
ARPACK with ``which='LR'``.  With a PML the spectrum fills with Berenger modes
of the discretised continuum, and on a 500 nm Si strip that returns
``n_eff = 4.73 - 2.64j`` - above the core index, so not a guided mode at all.
Selection here is shift-invert about a target (``which='LM'`` with ``sigma``)
followed by a **core-confinement filter**, which is what actually picks the
physical mode.

**Both ends of an axis have to absorb in the same sense.**  ``stretchmesh``
takes the absolute value of the stretched coordinate's imaginary part, which
turns the ``-x`` and ``-y`` layers into gain.  Nothing in the ``+x``-only
validation could see that; the plasmonic converter, which needs three edges,
returned nothing but gain/loss corner modes until it was found.  See
:func:`stretched_grid`.

**The PML must sit outside the turning point.**  Inside it the field is
evanescent and there is nothing to absorb.  For a bend the conformal transform
puts the turning point at ``u_t = R ln(n_eff / n_clad)``.

Validation
----------
See ``reports/06_pml_phase1_gate.md``.  Against the independent Airy reference
in :mod:`em_simulation.reference`, at matched ``n_eff``, the loss rate agrees to
**3 %** over seven decades on a SiN guide; against Vlasov & McNab (2004) the
bend-loss trend of a 220 nm Si wire is reproduced (report 06 section 6).  The
first device - a Si-wire-to-gold-slot converter, report 12 - found two more
things that had to be got right: the low-side PML layers were gain
(:func:`stretched_grid`) and a metal edge inside a cell aliases ``n_eff``
(``platforms.plasmonic_converter_dataset_info`` snaps the grid).
"""

import numpy as np
import scipy.sparse.linalg as spla

from . import _compat  # noqa: F401  - numpy/scipy shims must load first
from .base import FDEBackend, ModeData
from .emepy_fde import _pin_gauge

_M_TO_UM = 1e6

__all__ = ["PMLBackend", "PMLModeSolver", "stretched_grid", "turning_point"]


def _crop(field, nx, ny):
    """``compute_other_fields`` returns some components one cell short."""
    out = np.zeros((nx, ny), dtype=complex)
    rows, cols = min(nx, field.shape[0]), min(ny, field.shape[1])
    out[:rows, :cols] = field[:rows, :cols]
    return out


def turning_point(neff, clad_index, radius):
    r"""Where a bent guide starts to radiate, from the conformal transform.

    The map replaces a bend with a straight guide of index
    :math:`n\,e^{u/R}`, so the cladding reaches ``n_eff`` at

    .. math::  u_t = R \ln(n_{eff} / n_{clad})

    Note this is the *exponential* form.  Linearising - as the Airy reference
    does - puts it at ``R(n_eff^2 - n_clad^2)/(2 n_clad^2)``, which is roughly
    70 % further out and will place a PML in the wrong region.
    """
    return float(radius) * np.log(np.real(neff) / float(clad_index))


def stretched_grid(x, y, layers_plus_x=0, layers_minus_x=0,
                   layers_plus_y=0, layers_minus_y=0, factor=1 + 2j):
    r"""Complex-stretched grid, with the axis convention spelled out.

    Parabolic stretch, the ``'P'`` method of EMpy's ``stretchmesh``: over the
    ``n`` outermost cells of an edge, from the inner boundary ``q1`` to the
    window edge ``q2``,

    .. math::  	ilde z = z + (f - 1)\,rac{(z - q1)^3}{(q2 - q1)^2}

    which reaches ``f (q2 - q1)`` at the edge with a zero slope at ``q1``.

    Why this is not a call to ``stretchmesh``: that routine ends with
    ``xx.real + 1j*abs(xx.imag)``.  The stretch is *antisymmetric* by
    construction - ``(z - q1)^3`` is negative on the ``-x``/``-y`` side - and
    it has to be, because what the finite-difference operator consumes is
    ``diff(x)``, whose imaginary part must carry the **same** sign in both
    layers for both to absorb.  The ``abs`` makes ``Im(diff)`` negative on the
    low side, so ``-x`` and ``-y`` come out as gain layers.  A guided mode
    barely notices (it is evanescent there), but paired with an absorbing
    layer the spectrum turns PT-symmetric: the loss cancels (``Im n_eff = 0``
    on a radiating mode), and in the gain/loss corners near-real Berenger
    modes appear that shift-invert then returns instead of the physical
    mode.  ``tests/test_pml_solver.py`` pins both.

    ``stretchmesh`` orders ``nlayers`` as ``[+y, -y, +x, -x]``; the keyword
    names here exist so a caller cannot silently put the PML on the wrong
    axis, which absorbs the guided mode instead of the radiation.

    :returns: ``(x, y)`` complex arrays of the same length as the inputs.
    """
    x = np.array(x, dtype=complex)
    y = np.array(y, dtype=complex)
    f = complex(factor)
    for z, low, high in ((x, layers_minus_x, layers_plus_x),
                         (y, layers_minus_y, layers_plus_y)):
        _stretch_end(z, int(high), f, outer=-1)
        _stretch_end(z, int(low), f, outer=0)
    return x, y


def _stretch_end(z, n, f, outer):
    """Stretch ``n`` cells at one end of ``z`` in place; ``outer`` is -1 or 0."""
    if n <= 0 or f == 1:
        return
    if n >= len(z) // 2:
        raise ValueError(f"{n} PML layers on a {len(z)}-point axis")
    if outer == -1:
        cells, q1, q2 = slice(len(z) - 1 - n, len(z)), z[-1 - n], z[-1]
    else:
        cells, q1, q2 = slice(0, n), z[n], z[0]
    z[cells] = z[cells] + (f - 1) * (z[cells] - q1) ** 3 / (q2 - q1) ** 2


class PMLModeSolver:
    """Complex modes of a cross section, on a complex-stretched grid.

    :param cross_section: Any :class:`~em_simulation.fde.cross_section.CrossSection`.
    :param wavelength: Metres.
    :param window: ``(half_width, y_min, y_max)``, or ``(x_min, x_max, y_min,
        y_max)`` for an asymmetric one.  A bend radiates **outward only**, so
        the inner side needs perhaps a micron while the outer side has to reach
        past the turning point and hold the PML.  Forcing symmetry there
        doubles the x span, and at a fixed mesh count that halves the
        resolution - which is enough to under-resolve a high-contrast Si core
        and report a badly wrong ``n_eff``.
    :param mesh: Points across ``x``; ``mesh_y`` defaults to a matching pitch.
    :param pml_thickness: PML depth on the ``+x`` edge, metres.  Phase 1 found
        2 um under-converged and 4 um converged for a weak guide; scale with
        the wavelength in the cladding.
    :param pml_factor: Complex stretch reached at the outer edge.  ``1 + 2j`` is
        emepy's own intended value; ``1 + 4j`` is safer.
    :param pml_edges: Which edges to absorb on.  A bend radiates outward only,
        so ``("+x",)`` is the default; add ``"-y"``/``"+y"`` for a stack with
        substrate leakage.
    :param num_modes: Eigenpairs to request around the target.
    :param confinement_threshold: Minimum core power fraction for a mode to
        count as guided.
    """

    def __init__(
        self,
        cross_section,
        wavelength=1.55e-6,
        window=(2.5e-6, -1.2e-6, 1.2e-6),
        mesh=200,
        mesh_y=None,
        pml_thickness=1.5e-6,
        pml_factor=1 + 2j,
        pml_edges=("+x",),
        num_modes=10,
        confinement_threshold=0.05,
        accuracy=1e-8,
        boundary="0000",
    ):
        self.cross_section = cross_section
        self._wavelength = float(wavelength)
        self.window = tuple(float(v) for v in window)
        self.pml_thickness = float(pml_thickness)
        self.pml_factor = complex(pml_factor)
        self.pml_edges = tuple(pml_edges)
        self._num_modes = int(num_modes)
        self.confinement_threshold = float(confinement_threshold)
        self.accuracy = float(accuracy)
        self.boundary = boundary

        if len(self.window) == 4:
            x_min, x_max, y_min, y_max = self.window
        else:
            half_width, y_min, y_max = self.window
            x_min, x_max = -half_width, half_width
        self.x_min, self.x_max = float(x_min), float(x_max)
        mesh = int(mesh)
        if mesh_y is None:
            mesh_y = max(
                16, int(round(mesh * (y_max - y_min) / (x_max - x_min)))
            )
        self._x = np.linspace(x_min, x_max, mesh)
        self._y = np.linspace(y_min, y_max, int(mesh_y))

        step_x = self._x[1] - self._x[0]
        step_y = self._y[1] - self._y[0]
        depth_x = int(round(self.pml_thickness / step_x))
        depth_y = int(round(self.pml_thickness / step_y))
        self._layers = dict(
            layers_plus_x=depth_x if "+x" in self.pml_edges else 0,
            layers_minus_x=depth_x if "-x" in self.pml_edges else 0,
            layers_plus_y=depth_y if "+y" in self.pml_edges else 0,
            layers_minus_y=depth_y if "-y" in self.pml_edges else 0,
        )
        self.x, self.y = stretched_grid(
            self._x, self._y, factor=self.pml_factor, **self._layers
        )

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
        """Never. That is the whole point of this backend (5.16a, 5.16b)."""
        return False

    @property
    def cladding_index(self):
        """Radiation cut-off index at this solver's wavelength."""
        return self.cross_section.cladding_index_at(self._wavelength)

    def grid(self):
        """The complex-stretched ``(x, y)`` the fields come back on."""
        return self.x, self.y

    def solve_grid(self):
        """Same grid: the eigenproblem and the fields share it here."""
        return self.x, self.y

    @property
    def pml_start(self):
        """Where the ``+x`` stretch begins, in metres."""
        return self.x_max - self._layers["layers_plus_x"] * (
            self._x[1] - self._x[0]
        )

    def index_profile(self, params):
        """Real index on the solve grid, for plotting and the core mask."""
        return self.cross_section.index(np.real(self.x), np.real(self.y), params)

    def solve(self, params, target_neff, return_all=False):
        """Modes near ``target_neff``, selected by confinement.

        :param params: Cross-section parameter dict.
        :param target_neff: Shift-invert target - the straight-guide answer, or
            the previous point of a sweep.
        :param return_all: Return every eigenpair with its confinement, rather
            than just the best guided one.
        :returns: ``ModeData`` for the selected mode, plus an info dict.
        """
        from EMpy_gpu.modesolvers.FD import VFDModeSolver

        cross_section = self.cross_section

        def epsfunc(x_um, y_um):
            n = cross_section.index(
                np.real(np.asarray(x_um, dtype=complex)) / _M_TO_UM,
                np.real(np.asarray(y_um, dtype=complex)) / _M_TO_UM,
                params,
            )
            return n.astype(complex) ** 2

        solver = VFDModeSolver(
            self.wavelength * _M_TO_UM,
            self.x * _M_TO_UM,
            self.y * _M_TO_UM,
            epsfunc,
            self.boundary,
        )
        solver.nmodes, solver.tol = self.num_modes, self.accuracy
        matrix = solver.build_matrix()

        k0 = 2 * np.pi / (self.wavelength * _M_TO_UM)
        values, vectors = spla.eigs(
            matrix,
            k=self.num_modes,
            which="LM",                       # nearest the shift, not "largest"
            sigma=(float(np.real(target_neff)) * k0) ** 2,
            tol=self.accuracy,
            ncv=min(8 * self.num_modes, matrix.shape[0] - 1),
        )
        neff = np.sqrt(values) / k0
        neff = np.where(np.real(neff) < 0, -neff, neff)

        confinement = self._confinement(solver, vectors, params)
        info = {
            "neff": neff,
            "confinement": confinement,
            "pml_start": self.pml_start,
        }
        if return_all:
            return None, info

        guided = np.flatnonzero(confinement >= self.confinement_threshold)
        if not guided.size:
            raise RuntimeError(
                f"no mode with core confinement >= {self.confinement_threshold} "
                f"near n_eff = {np.real(target_neff):.4f}; best was "
                f"{confinement.max():.4f}. The PML is probably absorbing the "
                "guided mode - check the edge it is on and its standoff."
            )
        best = guided[np.argmax(confinement[guided])]
        info["selected"] = int(best)
        # e^{i beta z}: a decaying mode has Im > 0.  The conjugate branch is the
        # incoming-wave solution and would report gain.
        winner = complex(np.real(neff[best]), abs(np.imag(neff[best])))
        info["selected_neff"] = winner
        info["selected_confinement"] = float(confinement[best])
        return winner, info

    def mode_data(self, params, target_neff, num_modes=None):
        """The ``n`` best-confined modes near the target, as :class:`ModeData`.

        This is what an EME basis is built from, and it is the reason Phase 3
        exists: with a PML the basis has to include enough of the discretised
        radiation continuum to be approximately complete, and DBEME stores
        ``(2N x 2N)`` overlaps per adjacent grid-point pair.

        The grid returned is the **complex stretched** one.  That is deliberate:
        the overlap integrals must use the stretched metric or biorthogonality
        breaks (Phase 0.3).  Use ``np.real`` on it for plotting.

        :returns: ``ModeData`` ordered by decreasing confinement.
        """
        from EMpy_gpu.modesolvers.FD import VFDModeSolver

        wanted = int(num_modes or self.num_modes)
        cross_section = self.cross_section

        def epsfunc(x_um, y_um):
            n = cross_section.index(
                np.real(np.asarray(x_um, dtype=complex)) / _M_TO_UM,
                np.real(np.asarray(y_um, dtype=complex)) / _M_TO_UM,
                params,
            )
            return n.astype(complex) ** 2

        solver = VFDModeSolver(
            self.wavelength * _M_TO_UM,
            self.x * _M_TO_UM,
            self.y * _M_TO_UM,
            epsfunc,
            self.boundary,
        )
        solver.nmodes, solver.tol = wanted, self.accuracy
        matrix = solver.build_matrix()
        k0 = 2 * np.pi / (self.wavelength * _M_TO_UM)
        values, vectors = spla.eigs(
            matrix,
            k=wanted,
            which="LM",
            sigma=(float(np.real(target_neff)) * k0) ** 2,
            tol=self.accuracy,
            ncv=min(max(4 * wanted, 40), matrix.shape[0] - 1),
        )
        neff = np.sqrt(values) / k0
        neff = np.where(np.real(neff) < 0, -neff, neff)

        confinement = self._confinement(solver, vectors, params)
        # Guided modes first, then everything else; descending Re(n_eff) within
        # each group.  Neither half of that rule is optional.
        #
        # Ordering by confinement alone is unstable: two guided modes can sit
        # at 0.7650 and 0.7643, the order flips on numerical noise, and "mode
        # 0" then means different things at adjacent grid points.  The cascade
        # is built on those labels agreeing, and the symptom was a junction
        # transmitting 0.0004 instead of 0.9988.
        #
        # Ordering by Re(n_eff) alone fails differently: a PML spectrum is not
        # bounded by the guided modes.  At N = 40 two Berenger modes with zero
        # confinement came back at Re(n_eff) = 2.3414 on the narrow guide,
        # *above* its real TE0 at 2.2333 - so mode 0 was spurious on one side of
        # the interface and physical on the other, and T_00 collapsed to 0.006.
        #
        # Grouping by guidedness first makes the labels agree, because the
        # confinement gap between the two groups is enormous (0.74 against
        # 0.00) even where the gap *within* the guided group is not.
        guided = confinement >= self.confinement_threshold
        order = np.lexsort((-np.real(neff), ~guided))

        nx, ny = solver.nx, solver.ny
        hx = [vectors[: nx * ny, i].reshape(nx, ny) for i in order]
        hy = [vectors[nx * ny :, i].reshape(nx, ny) for i in order]
        ordered = np.array(
            [complex(np.real(neff[i]), abs(np.imag(neff[i]))) for i in order]
        )
        hz, ex, ey, ez = solver.compute_other_fields(ordered, hx, hy)

        E = np.zeros((wanted, 3, nx, ny), dtype=np.complex128)
        H = np.zeros((wanted, 3, nx, ny), dtype=np.complex128)
        te_fraction = np.zeros(wanted)
        for i in range(wanted):
            E[i, 0], E[i, 1], E[i, 2] = (
                _crop(ex[i], nx, ny), _crop(ey[i], nx, ny), _crop(ez[i], nx, ny)
            )
            H[i, 0], H[i, 1], H[i, 2] = (
                _crop(hx[i], nx, ny), _crop(hy[i], nx, ny), _crop(hz[i], nx, ny)
            )
            total = np.abs(E[i, 0]) ** 2 + np.abs(E[i, 1]) ** 2
            te_fraction[i] = (
                float(np.sum(np.abs(E[i, 0]) ** 2) / np.sum(total))
                if np.sum(total)
                else 0.0
            )
            # A dataset persists overlaps but not fields, so the same cross
            # section solved in two sessions must come back in the same gauge
            # (CLAUDE.md 5.6).  `_pin_gauge` rotates the transverse field real
            # and positive at its peak; for a complex (lossy) mode the field is
            # not globally real afterwards, but the peak itself is, which is
            # all the rule needs to be deterministic.
            _pin_gauge(E[i], H[i])

        return ModeData(
            x=self.x, y=self.y, E=E, H=H, neff=ordered, TE_pol=te_fraction
        ), confinement[order]

    # -------------------------------------------------------------- internals

    def _confinement(self, solver, vectors, params):
        """Fraction of transverse magnetic power inside the core.

        The discriminator between a guided mode and a Berenger mode of the
        discretised continuum.  Sorting by effective index cannot do this - a
        PML mode can sit above the core index.
        """
        nx, ny = solver.nx, solver.ny
        region = getattr(self.cross_section, "core_mask", None)
        if region is not None:
            # A plasmonic slot keeps its power in an *air* gap between plates
            # whose Re(n) sits below air, so "the high-index region" is the
            # wrong question there.  Let the cross section answer it.
            core = np.asarray(
                region(np.real(self.x), np.real(self.y), params), dtype=bool
            )[:nx, :ny]
        else:
            index = np.real(self.index_profile(params))[:nx, :ny]
            # The mask has to be ramp invariant.  Under a bend the conformal
            # transform multiplies the whole profile by exp(kappa x), so a
            # global threshold against max(index) selects the far outer
            # *cladding* rather than the core - and then a Berenger mode
            # scores confinement 1.0.  The ramp is constant within a column,
            # so compare column by column.
            column_max = index.max(axis=1, keepdims=True)
            column_min = index.min(axis=1, keepdims=True)
            core = index > 0.5 * (column_max + column_min)

        out = np.empty(vectors.shape[1])
        for i in range(vectors.shape[1]):
            hx = vectors[: nx * ny, i].reshape(nx, ny)
            hy = vectors[nx * ny :, i].reshape(nx, ny)
            power = np.abs(hx) ** 2 + np.abs(hy) ** 2
            total = power.sum()
            out[i] = power[core].sum() / total if total else 0.0
        return out


class PMLBackend(PMLModeSolver, FDEBackend):
    """:class:`PMLModeSolver` behind the :class:`FDEBackend` contract.

    This is what lets a *dataset* be built on a lossy basis - ``DataUpdater``
    and ``DataExtractor`` only ever call ``solve(point)``, ``solve_grid()`` and
    ``cross_section``.  It was deliberately not written until Phase 3 had
    answered whether a 20-40 mode PML basis could be carried at all
    (``reports/06``, section 7); it can, on the scattering cascade.

    :param target_neff: Shift-invert target for every point.  A dataset solves
        its points in whatever order devices visit them, so the target cannot
        be "the previous answer"; it is one number, chosen to sit between the
        indices of the modes the device converts between.  For the Si-to-slot
        converter that is ~2.0, below Si TE0 (~2.3) and above the gap plasmon
        (~1.5-1.9), so shift-invert returns both.
    """

    #: Bumped when stored PML modes change meaning with no parameter changing.
    #: 1: the stretch-sign fix of 2026-09-06 (``stretched_grid``); anything
    #: solved before it had gain layers on ``-x``/``-y``.
    STRETCH_CONVENTION = 1

    def __init__(self, cross_section, target_neff, parameter_names=None, **kwargs):
        super().__init__(cross_section, **kwargs)
        self.target_neff = float(np.real(target_neff))
        self.parameter_names = tuple(
            parameter_names if parameter_names is not None
            else cross_section.parameter_names
        )

    def fingerprint(self):
        """The PML recipe, for the dataset identity (JSON-stable types only).

        The grid alone cannot tell two PML bases apart: the same ``(x, y)``
        with a different stretch, edge set or target is a different mode
        problem, and stored overlaps from one are meaningless in the other.
        """
        return {
            "stretch_convention": self.STRETCH_CONVENTION,
            "pml_edges": sorted(self.pml_edges),
            "pml_thickness_m": float(self.pml_thickness),
            "pml_factor": [float(self.pml_factor.real), float(self.pml_factor.imag)],
            "target_neff": float(self.target_neff),
            "confinement_threshold": float(self.confinement_threshold),
        }

    def solve(self, parameter_point):
        """One parameter point as :class:`ModeData`, gauge pinned."""
        params = dict(zip(self.parameter_names, parameter_point))
        params.setdefault("wavelength", self._wavelength)
        data, _ = self.mode_data(params, self.target_neff)
        return data
