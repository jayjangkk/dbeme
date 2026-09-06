r"""Waveguide cross sections, expressed as a refractive-index function.

A cross section turns a point in parameter space (top width, curvature, ...)
into ``n(x, y)`` on an arbitrary sampling grid.  It replaces the ``.lms``
Lumerical model file of the original dataset-based-EME implementation: the
``.lms`` file held the parameterised geometry, and the FDE solver set its
parameters through ``mode.set(param_name, value)``.  Here the geometry is a
few lines of Python instead.

All lengths are in **metres** and curvature in **1/m**, matching the SI
convention of ``dataset_info.py``.

Bends
-----
A finite-difference solver finds modes of a straight, z-invariant waveguide.
A bend of radius :math:`R = 1/\kappa` is handled with the standard conformal
transformation, which maps the bend onto an equivalent straight guide with a
graded index

.. math::  n_{eq}(x, y) = n(x, y) \, e^{\kappa x}

where ``x`` is measured outwards from the bend centre of curvature side.  The
resulting mode is the true bend mode (shifted outwards, with the correct
:math:`n_{eff}`) to the accuracy of the conformal map.
"""

import abc

import numpy as np

from .materials import Material, as_material, silica, silicon


class CrossSection(metaclass=abc.ABCMeta):
    """Parameterised refractive-index profile of a waveguide cross section."""

    #: Parameter names this cross section understands, in dataset tuple order.
    parameter_names = ()

    #: Wavelength used when ``params`` carries none, in metres.  A dataset is
    #: built at one wavelength today, so this is normally just that wavelength.
    reference_wavelength = 1.55e-6

    @abc.abstractmethod
    def index(self, x, y, params: dict) -> np.ndarray:
        """Return ``n(x, y)`` with shape ``(len(x), len(y))``.

        :param x: Lateral coordinates in metres.
        :param y: Vertical coordinates in metres.
        :param params: Mapping of parameter name to value.  ``EmepyFDE`` adds
            ``"wavelength"`` (metres) to this, so a dispersive material knows
            where to evaluate.  When a wavelength axis is eventually added to a
            dataset it lands here as an ordinary parameter and nothing else has
            to change.
        """

    def wavelength_of(self, params):
        """The wavelength ``params`` asks for, falling back to the reference."""
        return float(params.get("wavelength", self.reference_wavelength))

    @abc.abstractmethod
    def cladding_index_at(self, wavelength) -> float:
        """Radiation-mode cut-off index at ``wavelength`` (metres)."""

    @abc.abstractmethod
    def materials(self) -> list:
        """Every :class:`~em_simulation.fde.materials.Material` in the stack."""

    def fingerprint(self) -> str:
        """Identity of the geometry *and* its index models.

        Included in the dataset fingerprint so a cache built with, say,
        ``Si/Li-293K`` is never silently reused with ``Si/Salzberg`` - the two
        differ by 2e-3 in index, which is far more than the accuracy the method
        otherwise delivers.
        """
        parts = [type(self).__name__]
        parts += [m.fingerprint() for m in self.materials()]
        return "|".join(parts)

    @abc.abstractmethod
    def default_window(self, max_params: dict):
        """Return ``(half_width, y_min, y_max)`` in metres for the solve window.

        The EME overlap integrals require every parameter point to share one
        grid, so this is called **once** with the largest values in the sweep
        and the resulting window is then held fixed.
        """

    @property
    def cladding_index(self) -> float:
        """Radiation-mode cut-off index, at the reference wavelength."""
        return self.cladding_index_at(self.reference_wavelength)


def _fill_fraction(centers, lo, hi):
    """Fraction of each 1-D cell that lies inside ``[lo, hi]``.

    ``centers`` are cell centres; cell edges are taken half-way between
    neighbours.  This is the sub-pixel averaging that lets a moderate mesh
    resolve a 20 nm change in waveguide width, which the dataset grid needs.
    """
    centers = np.asarray(np.real(centers), dtype=float)
    if centers.size == 1:
        return np.array([1.0 if lo <= centers[0] <= hi else 0.0])
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    left, right = edges[:-1], edges[1:]
    overlap = np.minimum(right, hi) - np.maximum(left, lo)
    return np.clip(overlap / (right - left), 0.0, 1.0)


def _sidewall_run(height, angle_degrees):
    """Lateral run of a sidewall of given height, per side, in metres.

    ``angle_degrees`` is measured from the horizontal in the usual fab
    convention: 90 is a vertical wall, and anything less leans outwards going
    down, so the feature is wider at its base than at its top.  A 130 nm step
    at 87 degrees runs out by 6.8 nm on each side.

    :returns: 0.0 for a vertical wall, so the trapezoid degenerates exactly to
        the rectangle and nothing changes for existing datasets.
    """
    if angle_degrees >= 90.0:
        return 0.0
    if angle_degrees <= 0.0:
        raise ValueError(f"sidewall angle {angle_degrees} must be in (0, 90]")
    return float(height) / np.tan(np.radians(angle_degrees))


def _trapezoid_fill(x, y, y_lo, y_hi, half_width_top, run):
    """Area fraction of a trapezoidal layer, per cell.

    The layer spans ``[y_lo, y_hi]`` and is ``2 * half_width_top`` wide at its
    top face, widening linearly to ``2 * (half_width_top + run)`` at its base.
    Evaluating the half-width at each row's centre and multiplying by that
    row's vertical fill is exact for a vertical wall and second-order accurate
    for a slanted one, which is the same order as the sub-pixel averaging
    already used everywhere else.

    :returns: ``(len(x), len(y))`` array of fill fractions.
    """
    vertical = _fill_fraction(y, y_lo, y_hi)
    centres = np.clip(np.asarray(np.real(y), dtype=float), y_lo, y_hi)
    # 0 at the top face, 1 at the base.
    depth = (y_hi - centres) / (y_hi - y_lo)
    half_widths = half_width_top + run * depth

    fill = np.empty((x.size, y.size))
    for j, half_width in enumerate(half_widths):
        if vertical[j] <= 0.0 or half_width <= 0.0:
            fill[:, j] = 0.0
        else:
            fill[:, j] = _fill_fraction(x, -half_width, half_width) * vertical[j]
    return fill


class FullEtchStrip(CrossSection):
    """Fully etched (strip) waveguide: a rectangular core on a substrate.

    ``top_width`` is the core width, the core height is fixed at
    ``thickness``, and because the etch is full there is no remaining slab -
    everything beside the core is cladding.

    ::

                        |<-- top_width -->|
          top cladding  +-----------------+   } thickness      y
        ----------------|      core       |------------------  ^
          substrate     +-----------------+                    |
        ==========================================             +--> x

    :param thickness: Core height in metres (e.g. ``220e-9``).
    :param core: Core material.  A :class:`~em_simulation.fde.materials.Material`,
        a number, or a ``"shelf/book/page"`` database identifier.  Defaults to
        crystalline Si from refractiveindex.info.
    :param cladding: Material above and beside the core.  Defaults to SiO2.
    :param substrate: Material below the core.  Defaults to ``cladding``, i.e.
        a symmetric buried guide.
    :param reference_wavelength: Wavelength in metres at which the plain
        ``core_index`` / ``cladding_index`` properties are reported.
    :param curvature_sign: ``+1`` puts the centre of curvature at ``-x``, so a
        positive curvature pushes the mode towards ``+x``.

    The older ``core_index`` / ``clad_index`` / ``substrate_index`` keywords
    still work and are treated as constant-index materials.
    """

    parameter_names = ("top_width", "curvature")

    def __init__(
        self,
        thickness=220e-9,
        core=None,
        cladding=None,
        substrate=None,
        reference_wavelength=1.55e-6,
        curvature_sign=1,
        core_index=None,
        clad_index=None,
        substrate_index=None,
        sidewall_angle=90.0,
    ):
        if core is not None and core_index is not None:
            raise TypeError("pass either `core` or `core_index`, not both")
        if cladding is not None and clad_index is not None:
            raise TypeError("pass either `cladding` or `clad_index`, not both")
        if substrate is not None and substrate_index is not None:
            raise TypeError("pass either `substrate` or `substrate_index`, not both")

        self.thickness = float(thickness)
        self.reference_wavelength = float(reference_wavelength)
        self.sidewall_angle = float(sidewall_angle)

        core_spec = core if core is not None else core_index
        clad_spec = cladding if cladding is not None else clad_index
        sub_spec = substrate if substrate is not None else substrate_index

        self.core = as_material(core_spec) if core_spec is not None else silicon()
        self.cladding = as_material(clad_spec) if clad_spec is not None else silica()
        self.substrate = as_material(sub_spec) if sub_spec is not None else self.cladding

        self.curvature_sign = int(np.sign(curvature_sign)) or 1

    def materials(self):
        return [self.core, self.cladding, self.substrate]

    def cladding_index_at(self, wavelength):
        """Whichever of cladding and substrate is higher - the guiding cut-off."""
        return float(
            max(
                np.real(self.cladding.index(wavelength)),
                np.real(self.substrate.index(wavelength)),
            )
        )

    def core_index_at(self, wavelength):
        return float(np.real(self.core.index(wavelength)))

    @property
    def core_index(self):
        """Core index at the reference wavelength."""
        return self.core_index_at(self.reference_wavelength)

    @property
    def clad_index(self):
        """Cladding index at the reference wavelength."""
        return float(np.real(self.cladding.index(self.reference_wavelength)))

    @property
    def substrate_index(self):
        """Substrate index at the reference wavelength."""
        return float(np.real(self.substrate.index(self.reference_wavelength)))

    def index(self, x, y, params):
        width = float(params["top_width"])
        curvature = float(params.get("curvature", 0.0))
        wavelength = self.wavelength_of(params)

        n_core = self.core_index_at(wavelength)
        n_clad = float(np.real(self.cladding.index(wavelength)))
        n_sub = float(np.real(self.substrate.index(wavelength)))

        x = np.asarray(np.real(x), dtype=float)
        y = np.asarray(np.real(y), dtype=float)

        # Sub-pixel fill fractions for the core.  With a slanted sidewall the
        # core is a trapezoid, wider at its base than the quoted top width -
        # which also destroys the horizontal mirror plane, so an angled strip
        # can convert TE to TM with no partial etch at all.
        run = _sidewall_run(self.thickness, self.sidewall_angle)
        fy = _fill_fraction(y, -0.5 * self.thickness, 0.5 * self.thickness)

        # Background: substrate below the core layer, cladding above/beside.
        f_sub = _fill_fraction(y, y.min() - 1.0, -0.5 * self.thickness)
        n_bg = np.sqrt(f_sub * n_sub**2 + (1.0 - f_sub) * n_clad**2)
        n_bg = np.broadcast_to(n_bg[np.newaxis, :], (x.size, y.size))

        # Blend the core in by area fraction (eps averaging, not n averaging).
        f_core = _trapezoid_fill(
            x, y, -0.5 * self.thickness, 0.5 * self.thickness, 0.5 * width, run
        )
        n = np.sqrt(f_core * n_core**2 + (1.0 - f_core) * n_bg**2)

        if curvature != 0.0:
            n = n * np.exp(self.curvature_sign * curvature * x[:, np.newaxis])

        return n

    def fingerprint(self):
        parts = [type(self).__name__, f"t={self.thickness:.6g}"]
        # As BiLevelStrip: a 90 degree wall *is* the rectangle, so recording a
        # redundant angle term would invalidate datasets built before angles
        # existed without any change of geometry.
        if self.sidewall_angle < 90.0:
            parts.append(f"wall={self.sidewall_angle:.4g}")
        parts += [m.fingerprint() for m in self.materials()]
        return "|".join(parts)

    def default_window(self, max_params):
        """Window sized from the widest core in the sweep, plus a margin."""
        width = float(max_params["top_width"])
        half_width = 0.5 * width + 1.5e-6
        y_half = 0.5 * self.thickness + 1.0e-6
        return half_width, -y_half, y_half


class CoupledStrips(CrossSection):
    """Two fully etched strips side by side, on a common substrate.

    The cross section the adiabatic-coupler family needs.  Both cores sit in
    the same 220 nm device layer, separated by ``gap``, and the pair is centred
    on ``x = 0`` so that the guides stay inside the window as their widths
    change::

              |<-- w1 -->| gap |<-- w2 -->|
        clad  +----------+     +----------+   } thickness       y
        ------|  core 1  |-----|  core 2  |---------------      ^
        subs  +----------+     +----------+                     |
        ===========================================             +--> x
                          ^
                       x = 0

    ``w1`` is the guide at negative ``x``.  In the Sacher-style adiabatic
    coupler ``w1`` is the *broad* guide, which keeps TE0, and ``w2`` the narrow
    one, which collects TE1.

    ``gap`` may be swept as a dataset parameter or fixed once here; whichever
    is not in ``params`` falls back to the constructor value.  Keeping it out
    of the grid is much cheaper - every extra axis adds two neighbour solves
    per point - so sweep it only when a device actually varies it.

    :param thickness: Device-layer thickness in metres.
    :param gap: Default edge-to-edge separation in metres.
    :param core: Core material (see :mod:`em_simulation.fde.materials`).
    :param cladding: Material above and between the cores.
    :param substrate: Material below.  Defaults to ``cladding``.
    :param swept_parameters: Names this cross section expects in the dataset
        grid.  Defaults to ``("w1", "w2")``; use ``("w1", "w2", "gap")`` to put
        the gap on the grid too.
    :param reference_wavelength: Where the plain index properties are reported.
    :param curvature_sign: As :class:`FullEtchStrip`.
    """

    def __init__(
        self,
        thickness=220e-9,
        gap=200e-9,
        core=None,
        cladding=None,
        substrate=None,
        swept_parameters=("w1", "w2"),
        reference_wavelength=1.55e-6,
        curvature_sign=1,
        w2=None,
        x_offset=0.0,
    ):
        self.thickness = float(thickness)
        self.gap = float(gap)
        # Defaults for whatever is not on the dataset grid.  Holding w2 or the
        # gap fixed keeps a device that does not vary them off the axis list,
        # and every axis costs two extra neighbour solves per path point.
        self.w2 = None if w2 is None else float(w2)
        self.x_offset = float(x_offset)
        self.parameter_names = tuple(swept_parameters)
        self.reference_wavelength = float(reference_wavelength)

        self.core = as_material(core) if core is not None else silicon()
        self.cladding = as_material(cladding) if cladding is not None else silica()
        self.substrate = (
            as_material(substrate) if substrate is not None else self.cladding
        )
        self.curvature_sign = int(np.sign(curvature_sign)) or 1

    # ------------------------------------------------------------- materials

    def materials(self):
        return [self.core, self.cladding, self.substrate]

    def cladding_index_at(self, wavelength):
        return float(
            max(
                np.real(self.cladding.index(wavelength)),
                np.real(self.substrate.index(wavelength)),
            )
        )

    def core_index_at(self, wavelength):
        return float(np.real(self.core.index(wavelength)))

    @property
    def core_index(self):
        return self.core_index_at(self.reference_wavelength)

    def fingerprint(self):
        """Identity, including the gap when it is *not* a swept parameter.

        A fixed gap is part of the geometry, so two datasets that differ only
        in it must not share a fingerprint.  When the gap is on the grid the
        parameter-grid half of the dataset fingerprint already covers it.
        """
        parts = [type(self).__name__, f"t={self.thickness:.6g}"]
        if "gap" not in self.parameter_names:
            parts.append(f"gap={self.gap:.6g}")
        if "w2" not in self.parameter_names and self.w2 is not None:
            parts.append(f"w2={self.w2:.6g}")
        if "x_offset" not in self.parameter_names and self.x_offset:
            parts.append(f"dx={self.x_offset:.6g}")
        parts += [m.fingerprint() for m in self.materials()]
        return "|".join(parts)

    # ---------------------------------------------------------------- geometry

    def edges(self, params):
        """``((x1_lo, x1_hi), (x2_lo, x2_hi))`` of the two cores, in metres.

        ``x_offset`` rigidly translates the pair.  A translation changes nothing
        about a single cross section - the modes are simply shifted - but it is
        exactly what a *tilted* coupler does between one section and the next,
        and the overlap between relatively displaced modes is what the rapid
        adiabatic coupler manipulates.  See ``reports/05_rapid_adiabatic_coupler.md``.
        """
        w1 = float(params["w1"])
        w2 = float(params.get("w2", self.w2))
        gap = float(params.get("gap", self.gap))
        shift = float(params.get("x_offset", self.x_offset))
        return (
            (shift - 0.5 * gap - w1, shift - 0.5 * gap),
            (shift + 0.5 * gap, shift + 0.5 * gap + w2),
        )

    def nominal_width(self, params):
        """A representative width, for solvers that want one."""
        return float(params["w1"]) + float(params.get("w2", self.w2 or 0.0))

    def index(self, x, y, params):
        wavelength = self.wavelength_of(params)
        curvature = float(params.get("curvature", 0.0))

        n_core = self.core_index_at(wavelength)
        n_clad = float(np.real(self.cladding.index(wavelength)))
        n_sub = float(np.real(self.substrate.index(wavelength)))

        x = np.asarray(np.real(x), dtype=float)
        y = np.asarray(np.real(y), dtype=float)

        (x1_lo, x1_hi), (x2_lo, x2_hi) = self.edges(params)

        # Sub-pixel fill fractions.  The two cores never overlap (gap > 0), so
        # their lateral fractions simply add.
        fx = _fill_fraction(x, x1_lo, x1_hi) + _fill_fraction(x, x2_lo, x2_hi)
        fx = np.clip(fx, 0.0, 1.0)
        fy = _fill_fraction(y, -0.5 * self.thickness, 0.5 * self.thickness)

        f_sub = _fill_fraction(y, y.min() - 1.0, -0.5 * self.thickness)
        n_bg = np.sqrt(f_sub * n_sub**2 + (1.0 - f_sub) * n_clad**2)
        n_bg = np.broadcast_to(n_bg[np.newaxis, :], (x.size, y.size))

        f_core = fx[:, np.newaxis] * fy[np.newaxis, :]
        n = np.sqrt(f_core * n_core**2 + (1.0 - f_core) * n_bg**2)

        if curvature != 0.0:
            n = n * np.exp(self.curvature_sign * curvature * x[:, np.newaxis])

        return n

    def power_fractions(self, mode_data, split_x=0.0):
        """Fraction of each mode's transverse power on the ``w1`` side.

        The single number that says *which physical waveguide* a supermode
        lives in, which is what an adiabatic coupler is judged on.  A value
        near 1 means the mode sits in guide 1 (the broad one, at ``x < 0``),
        near 0 means guide 2.  Around 0.5 the two guides are hybridised - that
        is the anti-crossing.

        :param mode_data: A :class:`~em_simulation.fde.base.ModeData`.
        :param split_x: Where to cut, in metres.  The gap centre is at 0 by
            construction, so the default is the obvious choice.
        :returns: Array of length ``num_modes``.
        """
        x, y = mode_data.x, mode_data.y
        left = x < split_x
        fractions = np.empty(len(mode_data.neff))
        for i in range(len(mode_data.neff)):
            intensity = (
                np.abs(mode_data.E[i, 0]) ** 2 + np.abs(mode_data.E[i, 1]) ** 2
            )
            total = np.trapezoid(np.trapezoid(intensity, y, axis=1), x)
            in_guide1 = np.trapezoid(
                np.trapezoid(intensity[left], y, axis=1), x[left]
            )
            fractions[i] = in_guide1 / total if total > 0 else np.nan
        return fractions

    def default_window(self, max_params):
        """Window sized from the widest pair in the sweep, plus a margin."""
        w1 = float(max_params["w1"])
        w2 = float(max_params.get("w2", self.w2 or 0.0))
        gap = float(max_params.get("gap", self.gap))
        half_width = 0.5 * (w1 + gap + w2) + 1.3e-6
        y_half = 0.5 * self.thickness + 0.9e-6
        return half_width, -y_half, y_half


class BiLevelStrip(CrossSection):
    """A rib: a full-height core on a wider, partially etched slab.

    The cross section a polarization rotator needs.  A buried strip with
    symmetric cladding has a horizontal mirror plane, and TE and TM modes sit
    in different symmetry classes under it, so they cannot couple at all - a
    rotator built on one returns exactly zero conversion.  Leaving a slab on
    the *bottom* of the device layer destroys that mirror plane and lets TM0
    and TE1 hybridise::

               |<------- w_slab ------->|
                    |<-- w_core -->|
        clad        +--------------+                } thickness      y
        ------+-----+     core     +-----+--------   } slab_thickness ^
        subs  |            slab          |                            |
        ======+==========================+=========                   +--> x

    The slab is *centred* on the core, so the vertical mirror plane at ``x=0``
    survives.  That is deliberate and it is not a problem: under ``x -> -x``,
    TM0 and TE1 fall in the *same* symmetry class (the odd profile of TE1's
    ``Ex`` cancels against the sign flip that ``Ex`` itself picks up), so they
    are free to couple.  It is only the horizontal mirror that has to go.

    Setting ``w_slab == w_core`` recovers a plain full-etch strip, which is how
    a bi-level taper starts and finishes.

    :param thickness: Full device-layer thickness in metres.
    :param slab_thickness: Remaining silicon after the partial etch.
    :param core: Core material; :param cladding: above and beside;
        :param substrate: below, defaulting to ``cladding``.
    :param swept_parameters: Dataset axis names, default ``("w_core", "w_slab")``.
    """

    def __init__(
        self,
        thickness=220e-9,
        slab_thickness=90e-9,
        core=None,
        cladding=None,
        substrate=None,
        swept_parameters=("w_core", "w_slab"),
        reference_wavelength=1.55e-6,
        curvature_sign=1,
        bottom_sidewall_angle=90.0,
        top_sidewall_angle=90.0,
    ):
        if not 0.0 < slab_thickness < thickness:
            raise ValueError(
                f"slab_thickness ({slab_thickness}) must lie strictly between 0 "
                f"and thickness ({thickness}); equal values give a plain strip, "
                "which has the horizontal mirror plane this class exists to break"
            )
        self.thickness = float(thickness)
        self.slab_thickness = float(slab_thickness)
        self.parameter_names = tuple(swept_parameters)
        self.reference_wavelength = float(reference_wavelength)
        self.bottom_sidewall_angle = float(bottom_sidewall_angle)
        self.top_sidewall_angle = float(top_sidewall_angle)

        self.core = as_material(core) if core is not None else silicon()
        self.cladding = as_material(cladding) if cladding is not None else silica()
        self.substrate = (
            as_material(substrate) if substrate is not None else self.cladding
        )
        self.curvature_sign = int(np.sign(curvature_sign)) or 1

    # ------------------------------------------------------------- materials

    def materials(self):
        return [self.core, self.cladding, self.substrate]

    def cladding_index_at(self, wavelength):
        return float(
            max(
                np.real(self.cladding.index(wavelength)),
                np.real(self.substrate.index(wavelength)),
            )
        )

    def core_index_at(self, wavelength):
        return float(np.real(self.core.index(wavelength)))

    @property
    def core_index(self):
        return self.core_index_at(self.reference_wavelength)

    def fingerprint(self):
        parts = [
            type(self).__name__,
            f"t={self.thickness:.6g}",
            f"slab={self.slab_thickness:.6g}",
        ]
        # Only recorded when the walls are actually slanted.  A 90 degree wall
        # *is* the rectangular geometry, so adding a redundant "angle=90" term
        # would change the identity string without changing the geometry and
        # invalidate every dataset built before angles existed.
        if self.bottom_sidewall_angle < 90.0 or self.top_sidewall_angle < 90.0:
            parts.append(
                f"walls={self.bottom_sidewall_angle:.4g}/"
                f"{self.top_sidewall_angle:.4g}"
            )
        parts += [m.fingerprint() for m in self.materials()]
        return "|".join(parts)

    # ---------------------------------------------------------------- geometry

    def nominal_width(self, params):
        return float(params["w_slab"])

    def index(self, x, y, params):
        wavelength = self.wavelength_of(params)
        curvature = float(params.get("curvature", 0.0))
        w_core = float(params["w_core"])
        w_slab = float(params["w_slab"])

        n_core = self.core_index_at(wavelength)
        n_clad = float(np.real(self.cladding.index(wavelength)))
        n_sub = float(np.real(self.substrate.index(wavelength)))

        x = np.asarray(np.real(x), dtype=float)
        y = np.asarray(np.real(y), dtype=float)

        bottom = -0.5 * self.thickness
        top = +0.5 * self.thickness
        slab_top = bottom + self.slab_thickness

        # Two etch steps, two sidewalls.  The *top* wall is the partial etch
        # that defines the core above the slab; the *bottom* wall is the full
        # etch that defines the slab itself.  Widths are top-referenced, so
        # each layer widens downwards from the quoted value.
        core_run = _sidewall_run(top - slab_top, self.top_sidewall_angle)
        slab_run = _sidewall_run(slab_top - bottom, self.bottom_sidewall_angle)

        # The two layers occupy disjoint height ranges, so their per-cell area
        # fractions add rather than needing a max().  In the row that straddles
        # the slab surface both contribute, which is the correct decomposition.
        f_core = _trapezoid_fill(x, y, slab_top, top, 0.5 * w_core, core_run)
        f_core = f_core + _trapezoid_fill(
            x, y, bottom, slab_top, 0.5 * w_slab, slab_run
        )
        f_core = np.clip(f_core, 0.0, 1.0)

        f_sub = _fill_fraction(y, y.min() - 1.0, bottom)
        n_bg = np.sqrt(f_sub * n_sub**2 + (1.0 - f_sub) * n_clad**2)
        n_bg = np.broadcast_to(n_bg[np.newaxis, :], (x.size, y.size))

        n = np.sqrt(f_core * n_core**2 + (1.0 - f_core) * n_bg**2)

        if curvature != 0.0:
            n = n * np.exp(self.curvature_sign * curvature * x[:, np.newaxis])

        return n

    def default_window(self, max_params):
        half_width = 0.5 * float(max_params["w_slab"]) + 1.3e-6
        y_half = 0.5 * self.thickness + 0.9e-6
        return half_width, -y_half, y_half
