r"""Si wire beside a plasmonic slot: the cross section of a lateral mode converter.

The device is Ono et al., *Optica* **3**, 999 (2016) / NTT Technical Review
**16**(7), 2018: a 400 x 200 nm Si wire feeding a gold/air/gold slot whose air
core is 50 x 20 nm, through a 600 nm laterally tapered converter.  A follow-up
that reproduces it (arXiv:1801.00833) is explicit about the layout: the gold
plates sit **left and right** of the slot, in the plane of the silicon, and the
Si width tapers linearly to a point with a constant air gap ``w_gap`` between
the Si edge and the metal - so the metal's inner edge follows the taper and the
slot that remains when the Si has gone is ``2 w_gap`` wide.

This cross section is that family, parameterised by the one thing that varies
along the converter:

    w_si       Si core width (the swept axis, 400 nm -> 0)
    gap        air gap between the Si edge and the metal inner edge
    metal_thickness   gold plate height; default the Si height (see the class)
    si_thickness      220 nm here (200 nm in the paper)

Layout, y up, x lateral, origin at the Si core centre::

        air
   Au ██   gap  ┌── Si ──┐  gap   ██ Au        <- Au bottom at metal_bottom
   ─────────────┴────────┴──────────────       (default: y = -si_thickness / 2)
        SiO2

Where this model departs from the paper, on purpose: the vertical dimension is
*not* tapered, so the gold plates span the Si height and the slot is a lateral
gap plasmon rather than the device's vertical 20 nm one; the Si is 220 nm
rather than 200 nm to match every other dataset in this project; and the
platform suspends the structure in air, because over oxide the lateral slot's
mode (1.18 + 0.058j at a 40 nm slot) sits below the substrate index and leaks -
the device avoids that with vertical confinement this model does not have.

Metals need complex permittivity, and averaging must be done in
:math:`\varepsilon`, not :math:`n`.  So this ``index()`` returns a complex
index, :math:`n = \sqrt{\varepsilon}` with :math:`\mathrm{Im} \ge 0` - which
the PML solver squares back into the complex :math:`\varepsilon` map it needs.
"""

import numpy as np

from .cross_section import CrossSection, _fill_fraction
from .materials import air, as_material, gold, silica, silicon

__all__ = ["PlasmonicSlotConverter"]


class PlasmonicSlotConverter(CrossSection):
    """A Si core between two gold plates, with an air gap either side.

    :param si_thickness: Si core height, metres.
    :param metal_thickness: Gold plate height, metres, bottom on the
        substrate plane; default the Si height.  The device's 50 x 20 nm core
        is a *vertical* MIM - its 20 nm sits between gold above and below - and
        a lateral model cannot reproduce that.  20 nm-tall plates here leave a
        slot open to the dielectric above and below, and nothing binds in it
        (best core confinement 0.03); plates spanning the Si height make it a
        proper lateral gap plasmon.
    :param gap: Air gap between the Si edge and the metal, metres.
    :param plate_reach: How far outward each gold plate extends from its
        inner edge, metres.  The gap plasmon decays into gold over the ~23 nm
        skin depth, so 250 nm is generous; beyond it is cladding, so the PML
        never sits inside the metal.
    :param metal_bottom: Where the plates start vertically, metres; default
        the substrate plane (the Si bottom).  A suspended structure has no
        plane to sit on, so the platform centres them on the Si.  It matters
        because the *height* of the walls is what binds a lateral slot mode:
        on the 5 nm grid a 40 nm slot reads 0.869 (leaky) between 220 nm
        walls and 1.146 + 0.036j between 400 nm ones.
    :param core, metal, cladding, substrate: Materials; default Si, Au
        (Johnson & Christy), air, SiO2.
    """

    parameter_names = ("w_si",)

    def __init__(
        self,
        si_thickness=220e-9,
        metal_thickness=None,
        gap=20e-9,
        plate_reach=0.25e-6,
        metal_bottom=None,
        core=None,
        metal=None,
        cladding=None,
        substrate=None,
        reference_wavelength=1.55e-6,
    ):
        self.si_thickness = float(si_thickness)
        self.metal_thickness = (
            float(metal_thickness) if metal_thickness is not None else self.si_thickness
        )
        self.gap = float(gap)
        self.plate_reach = float(plate_reach)
        self.metal_bottom = (
            float(metal_bottom) if metal_bottom is not None else -0.5 * self.si_thickness
        )
        self.reference_wavelength = float(reference_wavelength)
        self.core = as_material(core) if core is not None else silicon()
        self.metal = as_material(metal) if metal is not None else gold()
        self.cladding = as_material(cladding) if cladding is not None else air()
        self.substrate = as_material(substrate) if substrate is not None else silica()

    # ----------------------------------------------------------- materials

    def materials(self):
        return [self.core, self.metal, self.cladding, self.substrate]

    def _eps(self, material, wavelength):
        n = complex(np.asarray(material.index(wavelength)).ravel()[0])
        return n * n

    def cladding_index_at(self, wavelength):
        """Radiation cutoff: the larger of the two dielectrics bounding the core."""
        return float(max(
            np.real(self.cladding.index(wavelength)),
            np.real(self.substrate.index(wavelength)),
        ))

    def core_index_at(self, wavelength):
        return float(np.real(self.core.index(wavelength)))

    # ------------------------------------------------------------ geometry

    def edges(self, w_si):
        """``(si_half, metal_inner, metal_outer)`` in metres."""
        si_half = 0.5 * float(w_si)
        inner = si_half + self.gap
        return si_half, inner, inner + self.plate_reach

    def index(self, x, y, params):
        w_si = float(params["w_si"])
        wavelength = self.wavelength_of(params)
        eps_si = self._eps(self.core, wavelength)
        eps_au = self._eps(self.metal, wavelength)
        eps_cl = self._eps(self.cladding, wavelength)
        eps_sub = self._eps(self.substrate, wavelength)

        x = np.asarray(np.real(x), dtype=float)
        y = np.asarray(np.real(y), dtype=float)
        y_lo = -0.5 * self.si_thickness

        # background: substrate below the device plane, cladding above
        f_sub = _fill_fraction(y, y.min() - 1.0, y_lo)
        eps = np.broadcast_to(
            (f_sub * eps_sub + (1.0 - f_sub) * eps_cl)[np.newaxis, :],
            (x.size, y.size),
        ).astype(complex).copy()

        si_half, inner, outer = self.edges(w_si)

        # silicon core
        if w_si > 0:
            fx = _fill_fraction(x, -si_half, si_half)
            fy = _fill_fraction(y, y_lo, y_lo + self.si_thickness)
            f = np.outer(fx, fy)
            eps = f * eps_si + (1.0 - f) * eps

        # gold plates
        fy_m = _fill_fraction(y, self.metal_bottom, self.metal_bottom + self.metal_thickness)
        for lo, hi in ((-outer, -inner), (inner, outer)):
            f = np.outer(_fill_fraction(x, lo, hi), fy_m)
            eps = f * eps_au + (1.0 - f) * eps

        n = np.sqrt(eps)
        # sqrt's principal branch already puts Im >= 0 for Im(eps) >= 0;
        # pin it anyway so a round-off -0j never reads as gain.
        return np.where(np.imag(n) < 0, np.conj(n), n)

    def core_mask(self, x, y, params):
        """Where a guided mode of this structure keeps its power.

        The Si core, and the air between the metal plates over the metal's
        height - which is where the gap plasmon lives.  Used by the PML
        solver's confinement filter in place of its high-index heuristic,
        which would look at the metal (Re n = 0.5, *below* air) and conclude
        the slot is cladding.
        """
        w_si = float(params["w_si"])
        x = np.asarray(np.real(x), dtype=float)
        y = np.asarray(np.real(y), dtype=float)
        y_lo = min(-0.5 * self.si_thickness, self.metal_bottom)
        top = max(0.5 * self.si_thickness, self.metal_bottom + self.metal_thickness)
        si_half, inner, _ = self.edges(w_si)
        between_plates = (np.abs(x) <= inner)[:, None] & ((y >= y_lo) & (y <= top))[None, :]
        return between_plates

    # ------------------------------------------------------------ identity

    def fingerprint(self):
        parts = [
            type(self).__name__,
            f"t_si={self.si_thickness:.6g}",
            f"t_m={self.metal_thickness:.6g}",
            f"gap={self.gap:.6g}",
            f"reach={self.plate_reach:.6g}",
            f"m_bottom={self.metal_bottom:.6g}",
        ]
        parts += [m.fingerprint() for m in self.materials()]
        return "|".join(parts)

    def default_window(self, max_params):
        """Wide enough for the widest Si and both plates, plus a PML margin."""
        # The gap plasmon decays into gold over the ~23 nm skin depth, so the
        # plates need little reach; the Si TE0 needs ~0.4 um of cladding.
        _, _, outer = self.edges(float(max_params["w_si"]))
        half_width = outer + 0.45e-6
        # Symmetric, and tall enough that the weakly bound slot mode's tail
        # (n ~ 1.15 in air: 1/e over ~0.4 um) keeps ~0.3 um between the wall
        # ends and the +-y PML.  Symmetric so the grid samples the two plate
        # edges identically - a weak mode otherwise leans to the better
        # resolved edge.
        extent = max(0.5 * self.si_thickness,
                     abs(self.metal_bottom), abs(self.metal_bottom + self.metal_thickness))
        y_half = max(0.65e-6, extent + 0.55e-6)
        return half_width, -y_half, y_half
