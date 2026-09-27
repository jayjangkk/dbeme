r"""Exact complex ``n_eff`` of a bent slab, in Airy functions.

This is the *ruler* for the PML work (``tasks/02_pml_backend.md`` Phase 1): a
one-dimensional problem whose radiation loss is known semi-analytically, so a
PML implementation can be checked against something that does not share any of
its machinery.  It probes exactly what a PML has to get right - the radiating
tail beyond the turning point - and it is cheap enough to sweep over decades.

Nothing here imports DBEME.

The formulation
---------------
A bend is mapped to a straight guide with an equivalent index (Heiblum and
Harris, *IEEE J. Quantum Electron.* **11**, 75 (1975)):

.. math::  n_{eq}(u) = n(u)\, e^{u/R}

with ``u`` measured outward from the guide centre.  Linearising
:math:`n_{eq}^2 \approx n^2 (1 + 2u/R)` makes the Helmholtz equation

.. math::  \psi'' + \left[k_0^2 n_{eq}^2(u) - \beta^2\right]\psi = 0

piecewise **Airy**:

.. math::  \psi'' + (A_j + B_j u)\,\psi = 0, \quad
           A_j = k_0^2 n_j^2 - \beta^2, \quad B_j = 2 k_0^2 n_j^2 / R

which is Airy's equation in :math:`\zeta_j(u) = -(A_j + B_j u) / B_j^{2/3}`.

Inward, :math:`n_{eq}` falls without bound, so the mode is evanescent and the
decaying solution ``Ai`` applies.  Outward, :math:`n_{eq}` grows until it
reaches :math:`n_{eff}` at the **turning point**

.. math::  u_t = R\,(n_{eff}^2 - n_{clad}^2) / (2 n_{clad}^2)

beyond which the field radiates.  That tunnelling barrier is the bend loss, and
it is why the loss is exponential in ``R``.

Two numerical points, both essential
------------------------------------
**The outgoing solution must be a single Airy function.**  The textbook form
``Bi(z) + i Ai(z)`` cannot be evaluated: at a realistic barrier ``Bi ~ 1e10``
while ``Ai ~ 1e-11``, so the radiating part - which *is* the loss - sits
:math:`10^{21}` below the rounding error of the term beside it.  The identity

.. math::  \mathrm{Bi}(z) + i\,\mathrm{Ai}(z) = 2 e^{i\pi/6}\,
           \mathrm{Ai}\!\left(z\,e^{2\pi i/3}\right)

rewrites it as one function at a rotated argument, with no cancellation.

**Match logarithmic derivatives, not amplitudes.**  ``Ai`` and ``Bi`` span
enormous ranges across the barrier; their *ratio* does not.  ``scipy``'s scaled
``airye`` carries the same exponential factor on the function and its
derivative, so ``Aip/Ai`` is exact even where both would overflow.

Sign convention
---------------
Fields go as :math:`e^{i\beta z}`, so a decaying (lossy) mode has
:math:`\mathrm{Im}(n_{eff}) > 0`.  That is the convention the rest of this
project uses - ``correct_gain_modified`` treats negative ``Im(n_eff)`` as
round-off - and :meth:`BentSlab.solve` asserts it.
"""

import numpy as np
from scipy.special import airy, airye

__all__ = [
    "BentSlab",
    "attenuation_db_per_cm",
    "bend_loss_db_per_90deg",
    "imag_neff_from_db_per_cm",
]

#: exp(2 pi i / 3): rotates Ai into the outgoing-wave solution.
_ROT = np.exp(2j * np.pi / 3)


# --------------------------------------------------------------- conversions


def attenuation_db_per_cm(neff_imag, wavelength):
    """Power attenuation in dB/cm from ``Im(n_eff)``.

    Amplitude goes as :math:`e^{-k_0 n'' z}`, power as twice that, so
    :math:`\\alpha = 2 k_0 n''`.  At 1550 nm, 1 dB/cm is ``Im(n_eff) = 2.8e-6``.
    """
    k0 = 2 * np.pi / wavelength
    return 4.342944819 * 2 * k0 * np.asarray(neff_imag) / 100.0


def imag_neff_from_db_per_cm(db_per_cm, wavelength):
    """Inverse of :func:`attenuation_db_per_cm`."""
    k0 = 2 * np.pi / wavelength
    return np.asarray(db_per_cm) * 100.0 / (4.342944819 * 2 * k0)


def bend_loss_db_per_90deg(neff_imag, wavelength, radius):
    """Loss over a quarter turn, which is how bend loss is usually quoted."""
    return attenuation_db_per_cm(neff_imag, wavelength) * (
        np.pi * np.asarray(radius) / 2 * 100.0
    )


# ------------------------------------------------------------------- the slab


class BentSlab:
    """A three-layer slab bent to radius ``R``, solved in Airy functions.

    :param wavelength: Free-space wavelength, metres.
    :param core_index: Core index.
    :param clad_index: Cladding index, used on both sides unless
        ``inner_index`` is given.
    :param half_width: Half the core width, metres.  Note the model linearises
        the conformal map, so it needs ``half_width / radius`` small - keep it
        below ~0.05 and check :meth:`linearisation_error`.
    :param inner_index: Index on the inside of the bend, if it differs.
    :param polarization: ``"TE"`` matches ``psi`` and ``psi'``; ``"TM"``
        matches ``psi`` and ``psi'/n^2``.
    """

    def __init__(
        self,
        wavelength=1.55e-6,
        core_index=3.4757,
        clad_index=1.4440,
        half_width=250e-9,
        inner_index=None,
        polarization="TE",
    ):
        self.wavelength = float(wavelength)
        self.core_index = float(core_index)
        self.clad_index = float(clad_index)
        self.half_width = float(half_width)
        self.inner_index = float(
            clad_index if inner_index is None else inner_index
        )
        if polarization not in ("TE", "TM"):
            raise ValueError("polarization must be 'TE' or 'TM'")
        self.polarization = polarization

    # ------------------------------------------------------------ helpers

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    def _jump(self, index):
        """Weight on ``psi'`` at an interface: 1 for TE, ``1/n^2`` for TM."""
        return 1.0 if self.polarization == "TE" else 1.0 / index**2

    def linearisation_error(self, radius):
        """Relative error of ``e^{2u/R} ~ 1 + 2u/R`` at the core edge.

        The Airy form comes from linearising the conformal map, so it is only
        exact as ``half_width / radius -> 0``.  This is the size of what was
        dropped; a strongly guided slab tight enough to radiate measurably can
        easily violate it, which is why the shipped reference is weakly
        guiding.
        """
        ratio = 2 * self.half_width / radius
        return abs(np.expm1(ratio) - ratio) / abs(np.expm1(ratio))

    def wkb_exponent(self, neff, radius):
        r"""Tunnelling action across the barrier, :math:`\tfrac43 \zeta(a)^{3/2}`.

        The loss is set by the field that tunnels from the core edge out to the
        turning point, so WKB gives

        .. math::  \mathrm{Im}(n_{eff}) \propto
                   \exp\!\left(-\tfrac43 \zeta(a)^{3/2}\right)

        This is an *independent* handle on the same physics: the slope of
        ``log(Im n_eff)`` against radius must match ``-d(exponent)/dR`` without
        either calculation knowing about the other.  Used by the tests.
        """
        beta = self.k0 * np.real(neff)
        zeta = self._zeta(self.half_width, self.clad_index, beta, radius)
        return (4 / 3) * np.real(zeta) ** 1.5

    def turning_point(self, neff, radius):
        """Distance beyond the core centre at which the field starts radiating."""
        n_clad = self.clad_index
        return radius * (np.real(neff) ** 2 - n_clad**2) / (2 * n_clad**2)

    def _airy_ratio(self, z):
        """``Ai'(z) / Ai(z)``, safe where both would overflow.

        ``airye`` scales ``Ai`` and ``Ai'`` by the *same* exponential, so the
        ratio is exact even where the functions themselves are unrepresentable.

        Only valid in an evanescent region.  ``airye``'s real branch scales by
        ``exp(2/3 z**(3/2))``, which is NaN for ``z < 0``, so a real negative
        argument is promoted to complex - there the scaling is a unit-modulus
        phase shared by both, and the ratio is still exact.
        """
        if np.isrealobj(z) and np.real(z) < 0:
            z = complex(z)
        ai, aip, _, _ = airye(z)
        return aip / ai

    def _region(self, index, beta, radius):
        """``(A, B, B**(1/3), B**(2/3))`` for one layer."""
        a = self.k0**2 * index**2 - beta**2
        b = 2 * self.k0**2 * index**2 / radius
        return a, b, b ** (1 / 3), b ** (2 / 3)

    def _zeta(self, u, index, beta, radius):
        a, b, _, b23 = self._region(index, beta, radius)
        return -(a + b * u) / b23

    # ------------------------------------------------------- the dispersion

    def _dispersion(self, beta, radius):
        """Mismatch in ``psi'/psi`` at the outer interface. Zero at a mode.

        Shoot from the inner cladding, where the decaying solution ``Ai`` is
        the only admissible one, across the core, and compare against the
        outgoing solution on the outside.
        """
        a = self.half_width

        # --- inner cladding: decaying, so psi = Ai(zeta) -----------------
        _, _, b13_in, _ = self._region(self.inner_index, beta, radius)
        zeta_in = self._zeta(-a, self.inner_index, beta, radius)
        # d zeta / du = -B**(1/3), hence the sign.
        y_in = -b13_in * self._airy_ratio(zeta_in)
        y_in *= self._jump(self.core_index) / self._jump(self.inner_index)

        # --- core: propagate psi, psi' across on Ai and Bi ---------------
        _, _, b13_co, _ = self._region(self.core_index, beta, radius)
        zl = self._zeta(-a, self.core_index, beta, radius)
        zr = self._zeta(+a, self.core_index, beta, radius)
        # Unscaled on purpose.  The core is the oscillatory region, so zeta is
        # O(1) and nothing overflows - and the scaled forms would be *wrong*
        # here: Ai and Bi carry different exponential factors, and those
        # factors differ between the two interfaces, so a 2x2 solve on the left
        # and an evaluation on the right would not be in the same basis.
        ail, aipl, bil, bipl = airy(zl)
        air, aipr, bir, bipr = airy(zr)
        matrix = np.array(
            [[ail, bil], [-b13_co * aipl, -b13_co * bipl]], dtype=complex
        )
        coefficients = np.linalg.solve(matrix, np.array([1.0, y_in], dtype=complex))
        psi_r = coefficients @ np.array([air, bir])
        dpsi_r = coefficients @ np.array([-b13_co * aipr, -b13_co * bipr])
        y_core = dpsi_r / psi_r
        y_core *= self._jump(self.clad_index) / self._jump(self.core_index)

        # --- outer cladding: outgoing, so psi = Ai(zeta * exp(2 pi i/3)) --
        _, _, b13_out, _ = self._region(self.clad_index, beta, radius)
        zeta_out = self._zeta(+a, self.clad_index, beta, radius) * _ROT
        y_out = -b13_out * _ROT * self._airy_ratio(zeta_out)

        return y_core - y_out

    # --------------------------------------------------------- straight limit

    def straight_neff(self):
        """``n_eff`` of the unbent slab, as the continuation seed.

        Real, and found by bisection on the standard symmetric-slab
        transcendental equation, so it needs no starting guess of its own.
        """
        k0, a = self.k0, self.half_width
        n1, n2 = self.core_index, self.clad_index
        weight = 1.0 if self.polarization == "TE" else (n1 / n2) ** 2

        def residual(neff):
            kappa = k0 * np.sqrt(max(n1**2 - neff**2, 0.0))
            gamma = k0 * np.sqrt(max(neff**2 - n2**2, 0.0))
            return np.tan(kappa * a) - weight * gamma / kappa

        lo, hi = n2 + 1e-9, n1 - 1e-9
        # The fundamental mode is the largest root; walk down from the core
        # index until the residual changes sign, then bisect.
        grid = np.linspace(hi, lo, 20001)
        values = np.array([residual(v) for v in grid])
        crossings = np.flatnonzero(
            (np.sign(values[:-1]) != np.sign(values[1:]))
            & (np.abs(values[:-1] - values[1:]) < 1e3)
        )
        if not crossings.size:
            raise RuntimeError("no guided mode in the straight slab")
        left, right = grid[crossings[0]], grid[crossings[0] + 1]
        for _ in range(200):
            mid = 0.5 * (left + right)
            if residual(left) * residual(mid) <= 0:
                right = mid
            else:
                left = mid
        return 0.5 * (left + right)

    # ------------------------------------------------------------- solving

    def solve(self, radius, neff_guess=None, tol=1e-14, max_iter=100):
        """Complex ``n_eff`` at one bend radius.

        :param radius: Bend radius, metres, measured to the guide centre.
        :param neff_guess: Starting point; defaults to the straight-guide
            value.  Sweeps should pass the previous radius' answer.
        :returns: Complex ``n_eff`` with ``Im > 0`` for a lossy mode.
        """
        if neff_guess is None:
            neff_guess = self.straight_neff() + 1e-12j
        # Validate the seed, not just the answer.  Outside the guided range the
        # Airy arguments leave the regime the shooting assumes and the core
        # 2x2 goes singular, which surfaces as an opaque LinAlgError.
        seed = float(np.real(neff_guess))
        if not (self.clad_index < seed < self.core_index):
            raise ValueError(
                f"neff_guess = {seed:.6f} is outside "
                f"({self.clad_index}, {self.core_index}); a guided mode of "
                "this slab cannot be there"
            )

        # Secant iteration in the complex plane.  Newton would need a
        # derivative of a function built from Airy ratios; a secant step is
        # both simpler and, seeded by continuation, entirely adequate.
        beta0 = self.k0 * complex(neff_guess)
        beta1 = beta0 * (1 + 1e-9) + 1e-9j
        f0 = self._dispersion(beta0, radius)
        f1 = self._dispersion(beta1, radius)
        for _ in range(max_iter):
            if abs(f1 - f0) < 1e-300:
                break
            step = f1 * (beta1 - beta0) / (f1 - f0)
            beta0, f0 = beta1, f1
            beta1 = beta1 - step
            f1 = self._dispersion(beta1, radius)
            if abs(step) < tol * abs(beta1):
                break
        else:
            raise RuntimeError(f"no convergence at R = {radius:.3e} m")

        neff = beta1 / self.k0
        # A bound mode of the straight guide lies strictly between the cladding
        # and core indices.  The secant can otherwise land on a root of the
        # dispersion function that is not a guided mode at all - an unguarded
        # sweep returned n_eff *above* the core index.
        real = float(np.real(neff))
        if not (self.clad_index < real < self.core_index):
            raise RuntimeError(
                f"converged to n_eff = {real:.6f}, outside "
                f"({self.clad_index}, {self.core_index}) - not a guided mode"
            )
        # e^{i beta z} means a decaying mode has Im(beta) > 0.  The opposite
        # sign is the incoming-wave branch and would report gain.
        return complex(real, abs(np.imag(neff)))

    def sweep(self, radii, neff_guess=None):
        """Solve a descending or ascending radius sweep by continuation.

        Losses span decades across a useful radius range, so each solve is
        seeded with the previous answer; a cold guess fails at tight radii.

        :returns: Complex ``n_eff`` array, ordered as ``radii``.
        """
        radii = np.asarray(radii, dtype=float)
        # Walk from the loosest bend, which is closest to the straight guide.
        order = np.argsort(-radii)
        out = np.empty(len(radii), dtype=complex)
        guess = neff_guess
        for index in order:
            out[index] = self.solve(radii[index], neff_guess=guess)
            guess = out[index]
        return out
