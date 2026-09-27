"""emepy-backed FDE solver: the open-source replacement for Lumerical MODE.

The original dataset-based-EME implementation drove Ansys Lumerical MODE
through ``lumapi`` to fill its dataset.  This backend produces the same
quantities - ``neff``, the TE polarisation fraction, and the ``E``/``H``
profiles on a common grid - from emepy's finite-difference vectorial mode
solver (:class:`emepy.fd.MSEMpy`, which wraps EMpy's ``VFDModeSolver``).

Everything downstream of :class:`~dbeme.fde.base.ModeData` - field
normalisation, overlap integrals, mode tracking, the EME transfer matrices -
is unchanged from the Lumerical-based version.
"""

import warnings

import numpy as np

from . import _compat

_compat.apply()

from emepy.fd import MSEMpy  # noqa: E402  (import must follow the shims)

from .base import FDEBackend, ModeData  # noqa: E402
from .cross_section import CrossSection, FullEtchStrip  # noqa: E402
from .materials import check_lossless  # noqa: E402

#: emepy/EMpy are unit-agnostic as long as the wavelength and the grid share a
#: unit.  We keep the public API in metres (matching ``dataset_info.py``) and
#: solve in micrometres, which is EMpy's conventional scale and better
#: conditioned for the sparse eigensolver.
_M_TO_UM = 1e6


class EmepyFDE(FDEBackend):
    """Solve waveguide cross sections with emepy on a fixed grid.

    :param cross_section: Geometry description, e.g.
        :class:`~dbeme.fde.cross_section.FullEtchStrip`.
    :param parameter_names: Dataset parameter order.  Defaults to the cross
        section's own order.
    :param num_modes: Number of forward modes per parameter point.
    :param wavelength: Free-space wavelength in metres.
    :param window: ``(half_width, y_min, y_max)`` in metres.  Defaults to
        ``cross_section.default_window(max_params)``.
    :param max_params: Largest parameter values in the sweep, used to size the
        default window.  Required when ``window`` is not given.
    :param mesh: Number of grid points along x (y is scaled to keep cells
        roughly square, unless ``mesh_y`` is given).
    :param mesh_y: Number of grid points along y.
    :param accuracy: Eigensolver tolerance handed to EMpy.
    :param boundary: EMpy boundary code, four characters for N/S/E/W, each one
        of ``'0'`` (electric wall), ``'S'`` (symmetric), ``'A'``
        (antisymmetric).
    """

    def __init__(
        self,
        cross_section: CrossSection = None,
        parameter_names=None,
        num_modes=6,
        wavelength=1.55e-6,
        window=None,
        max_params=None,
        mesh=120,
        mesh_y=None,
        accuracy=1e-8,
        boundary="0000",
    ):
        self.cross_section = (
            cross_section if cross_section is not None else FullEtchStrip()
        )
        self.parameter_names = tuple(
            parameter_names
            if parameter_names is not None
            else self.cross_section.parameter_names
        )
        self._num_modes = int(num_modes)
        self._wavelength = float(wavelength)
        self.accuracy = float(accuracy)
        self.boundary = boundary

        if window is None:
            if max_params is None:
                raise ValueError("provide either `window` or `max_params`")
            window = self.cross_section.default_window(max_params)
        half_width, y_min, y_max = window
        self.window = (float(half_width), float(y_min), float(y_max))

        # Fixed grid, shared by every parameter point (metres).
        mesh = int(mesh)
        if mesh_y is None:
            span_x, span_y = 2 * half_width, (y_max - y_min)
            mesh_y = max(16, int(round(mesh * span_y / span_x)))
        self._x = np.linspace(-half_width, half_width, mesh)
        self._y = np.linspace(y_min, y_max, int(mesh_y))

        # Grid actually carried by the returned fields; emepy's `get_mode`
        # re-derives it, so we learn it from the first solve and then check
        # that it never changes.
        self._field_x = None
        self._field_y = None
        self._ramp_warned = False

        # Keep the cross section's own reference wavelength in step, so that
        # `cross_section.cladding_index` and `backend.cladding_index` cannot
        # disagree, then check the stack is usable here.
        self.cross_section.reference_wavelength = self._wavelength
        self._check_materials()

    # ------------------------------------------------------------------ API

    @property
    def num_modes(self):
        return self._num_modes

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def cladding_index(self):
        """Radiation cut-off index, evaluated at this backend's wavelength."""
        return self.cross_section.cladding_index_at(self._wavelength)

    @property
    def lossless(self):
        """True unless the stack absorbs - this solver has no PML (5.9)."""
        return getattr(self, "_lossless", True)

    def wavelength_of(self, params):
        """The wavelength a parameter point asks for, defaulting to our own."""
        return float(params.get("wavelength", self._wavelength))

    def _check_materials(self):
        """Warn if the stack is absorbing here, or used outside its data range.

        Both are silent failure modes: an absorbing material breaks the
        lossless assumptions in the backward-mode basis, and a dispersion
        formula evaluated past its fitted range returns a plausible-looking
        number with no support behind it.
        """
        materials = getattr(self.cross_section, "materials", None)
        if materials is None:
            return
        stack = materials()
        # An absorbing stack makes this backend's modes complex, which the
        # conjugated backward basis and the Im(n_eff) clip cannot represent.
        # Record it rather than only warning, so `assemble` can switch.
        self._lossless = not check_lossless(stack, self._wavelength)

        for material in stack:
            bounds = material.validity_range()
            if bounds is None:
                continue
            low, high = bounds
            if not (low <= self._wavelength <= high):
                warnings.warn(
                    f"{material.name} is only characterised from "
                    f"{low*1e9:.0f} to {high*1e9:.0f} nm, but this dataset is "
                    f"being built at {self._wavelength*1e9:.0f} nm. The index "
                    "is an extrapolation.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def grid(self):
        if self._field_x is None:
            raise RuntimeError("grid is only known after the first solve()")
        return self._field_x, self._field_y

    def solve_grid(self):
        """The ``(x, y)`` grid the eigenproblem is discretised on, in metres.

        One cell coarser than :meth:`grid`, which is the grid the fields come
        back on.  Use this to evaluate the index profile for plotting.
        """
        return self._x, self._y

    def solve(self, parameter_point) -> ModeData:
        params = dict(zip(self.parameter_names, parameter_point))
        # A dispersive cross section needs to know where to evaluate. Passing
        # it alongside the geometry parameters means that when a wavelength
        # axis is added to a dataset it arrives here by the same route.
        params.setdefault("wavelength", self._wavelength)
        solver = self._build_solver(params)
        solver.solve()

        modes = [solver.get_mode(i) for i in range(self._num_modes)]
        # Lumerical FDE reports modes in order of decreasing effective index;
        # EMpy's eigensolver ordering is not guaranteed, so sort explicitly.
        modes.sort(key=lambda m: -np.real(m.neff))

        x = np.real(np.asarray(modes[0].x, dtype=complex)) / _M_TO_UM
        y = np.real(np.asarray(modes[0].y, dtype=complex)) / _M_TO_UM
        self._check_grid(x, y)

        nx, ny = len(x), len(y)
        E = np.zeros((self._num_modes, 3, nx, ny), dtype=np.complex128)
        H = np.zeros((self._num_modes, 3, nx, ny), dtype=np.complex128)
        neff = np.zeros(self._num_modes, dtype=np.complex128)
        TE_pol = np.zeros(self._num_modes, dtype=float)

        for i, m in enumerate(modes):
            E[i, 0], E[i, 1], E[i, 2] = m.Ex, m.Ey, m.Ez
            H[i, 0], H[i, 1], H[i, 2] = m.Hx, m.Hy, m.Hz
            neff[i] = m.neff
            TE_pol[i] = _te_fraction(m.Ex, m.Ey, x, y)
            _pin_gauge(E[i], H[i])

        self._check_conformal_ramp(params, np.real(neff[0]))

        solver.clear()
        return ModeData(x=x, y=y, E=E, H=H, neff=neff, TE_pol=TE_pol)

    def _check_conformal_ramp(self, params, neff_max):
        """Warn when the bend transform outgrows the solve window.

        The conformal map replaces a bend of curvature ``kappa`` with a
        straight guide whose index is multiplied by ``exp(kappa * x)`` - the
        index therefore grows without bound towards the outside of the bend.
        Physically that is the leakage region; numerically, once the ramped
        cladding at the window edge reaches the mode index, the solver starts
        returning spurious modes pinned against the wall, which would outrank
        the real guided modes in the neff ordering.

        With no PML available there is no way to absorb them, so the honest
        thing is to say when the window is too wide for the radius asked for.
        """
        curvature = abs(float(params.get("curvature", 0.0)))
        if curvature == 0.0:
            return
        half_width = self.window[0]
        n_clad = self.cross_section.cladding_index_at(self.wavelength_of(params))
        n_edge = n_clad * np.exp(curvature * half_width)
        if n_edge > neff_max and not self._ramp_warned:
            self._ramp_warned = True
            radius_um = 1e6 / curvature
            warnings.warn(
                f"conformal bend transform: at curvature {curvature:.3g} 1/m "
                f"(R = {radius_um:.1f} um) the ramped cladding reaches "
                f"n = {n_edge:.2f} at the window edge, above the guided-mode "
                f"index {neff_max:.2f}. Spurious modes pinned to the window "
                f"wall may appear. Narrow the solve window or keep the "
                f"curvature grid below about "
                f"{np.log(neff_max / n_clad) / half_width:.3g} 1/m.",
                RuntimeWarning,
                stacklevel=3,
            )

    # -------------------------------------------------------------- internals

    def _build_solver(self, params):
        """Construct an :class:`MSEMpy` bound to our own index function."""
        cs = self.cross_section

        def epsfunc(x_um, y_um):
            n = cs.index(
                np.real(np.asarray(x_um, dtype=complex)) / _M_TO_UM,
                np.real(np.asarray(y_um, dtype=complex)) / _M_TO_UM,
                params,
            )
            return n.astype(complex) ** 2

        # MSEMpy only keeps a caller-supplied `epsfunc` when `width` is not
        # None (otherwise its constructor rebuilds one from width/thickness),
        # so the real width and thickness are passed through as well.
        wavelength = self.wavelength_of(params)
        core_index = (
            cs.core_index_at(wavelength)
            if hasattr(cs, "core_index_at")
            else getattr(cs, "core_index", 3.4757)
        )
        return MSEMpy(
            wl=wavelength * _M_TO_UM,
            width=float(params[self.parameter_names[0]]) * _M_TO_UM,
            thickness=getattr(cs, "thickness", 0.22e-6) * _M_TO_UM,
            num_modes=self._num_modes,
            core_index=core_index,
            cladding_index=cs.cladding_index_at(wavelength),
            x=self._x * _M_TO_UM,
            y=self._y * _M_TO_UM,
            epsfunc=epsfunc,
            accuracy=self.accuracy,
            boundary=self.boundary,
            subpixel=False,
        )

    def _check_grid(self, x, y):
        if self._field_x is None:
            self._field_x, self._field_y = x, y
            return
        if not (np.allclose(x, self._field_x) and np.allclose(y, self._field_y)):
            raise RuntimeError(
                "the FDE grid changed between parameter points; overlap "
                "integrals require one common grid"
            )


def _pin_gauge(E, H):
    """Fix a mode's global phase deterministically, in place.

    An eigenvector is only defined up to a complex scale, and ARPACK's starting
    vector is random, so solving the *same* cross section twice can return a
    mode with the opposite sign.  Normalising to unit power does not remove
    that freedom: sending ``(E, H) -> (-E, -H)`` leaves ``(E x H)_z`` unchanged.

    That matters because the dataset persists overlap matrices but not fields.
    An overlap ``<E_P, H_A>`` computed in one session and ``<E_P, H_B>``
    computed in another would then be in different gauges for the same point
    ``P``, and cascading them would put a spurious sign on a row - silently,
    and depending on the order devices happened to be run in.

    So the gauge is pinned here, at the only place that sees the fields:

    1. Rotate the global phase so the transverse field is real, using the
       location of the largest ``|E_t|``.
    2. That still leaves a sign.  Fix it by requiring the first grid point (in
       raster order) where ``|E_t|`` exceeds half its maximum to be positive.
       Picking the *first* such point rather than the largest is what makes
       this stable for a mode with equal-magnitude lobes - a TE1 mode's two
       lobes differ only in sign, so an ``argmax`` between them could flip on
       round-off, but their raster order cannot.

    :param E: One mode's electric field, shape ``(3, nx, ny)``, modified in place.
    :param H: The matching magnetic field, modified in place.
    """
    transverse = E[:2]
    magnitude = np.abs(transverse)
    peak = magnitude.max()
    if peak == 0:
        return

    flat_peak = int(np.argmax(magnitude))
    phase = np.angle(transverse.flat[flat_peak])
    E *= np.exp(-1j * phase)
    H *= np.exp(-1j * phase)

    real_part = np.real(E[:2])
    significant = np.flatnonzero(np.abs(real_part).ravel() > 0.5 * peak)
    if significant.size and real_part.ravel()[significant[0]] < 0:
        E *= -1
        H *= -1


def _te_fraction(Ex, Ey, x, y):
    """TE polarisation fraction, matching Lumerical's definition.

    ``int |Ex|^2 / int (|Ex|^2 + |Ey|^2)`` over the cross section, with ``x``
    the in-plane transverse direction.
    """

    def integrate(f):
        return np.trapezoid(np.trapezoid(f, y, axis=1), x)

    ex = integrate(np.abs(Ex) ** 2)
    ey = integrate(np.abs(Ey) ** 2)
    total = ex + ey
    return float(ex / total) if total > 0 else 0.0
