"""sax models for the pieces DBEME describes by an index: straight, arc, ideal coupler.

A ring arc is not an S-matrix DBEME cascades - it is ``2 pi R`` of a uniform
bend guide whose ``n_eff(lambda, kappa = 1/R)`` the strip dataset holds - so
its model is analytic: ``exp(+i 2 pi n_eff L / lambda)`` with a loss.
``IndexModel`` carries ``n_eff(lambda)`` as a polynomial fitted through the
per-wavelength solves (never linear by default: degree 2 gives ``n_eff``,
``n_g`` and dispersion), and the models take ``wl`` in **microns** as every
sax model does (`em_simulation.circuit` docstring).
"""

import numpy as np
import jax.numpy as jnp
import sax

from .phase_fit import PolynomialFit

TWO_PI = 2.0 * np.pi
UM = 1e-6


def loss_db_per_cm_from_imag_neff(imag_neff, wavelength_m):
    """Power loss in dB/cm of a mode with ``Im(n_eff)`` (project sign: ``> 0`` absorbs)."""
    alpha = 2.0 * TWO_PI / wavelength_m * np.asarray(imag_neff, dtype=float)  # 2 k0 Im n, 1/m
    return 10.0 * np.log10(np.e) * alpha / 100.0


class IndexModel:
    """``n_eff(lambda)`` as a polynomial in ``(lambda - lambda0) / lambda0``."""

    def __init__(self, fit):
        self.fit = fit

    @classmethod
    def from_samples(cls, wavelengths_m, neff, degree=2):
        """Fit through per-wavelength solves; ``degree=2`` unless fewer samples."""
        return cls(PolynomialFit.fit(wavelengths_m, np.real(neff), degree))

    @classmethod
    def from_neff_ng(cls, neff, ng, wavelength_m):
        """First-order model ``n(lambda) = n_eff + (lambda - lambda0) dn/dlambda``
        with ``dn/dlambda = (n_eff - n_g) / lambda0`` - what sax's ``straight``
        assumes.  Fine over a few nm; use ``from_samples`` for a band."""
        wl0 = float(wavelength_m)
        return cls(PolynomialFit(wl0, [float(neff), float(neff) - float(ng)]))

    def neff(self, wavelengths_m):
        return self.fit(wavelengths_m)

    def ng(self, wavelengths_m):
        wl = jnp.asarray(wavelengths_m, dtype=jnp.float64)
        return self.fit(wl) - wl * self.fit.derivative()(wl)


def straight_model(index, length_m, loss_db_per_cm=0.0, ports=("o1", "o2")):
    """sax model of ``length_m`` of uniform guide with ``IndexModel`` *index*."""
    p_in, p_out = ports
    length_m = float(length_m)
    amplitude = 10.0 ** (-float(loss_db_per_cm) * length_m * 100.0 / 20.0)

    def model(*, wl: float = 1.55) -> sax.SDict:
        wl_m = jnp.asarray(wl, dtype=jnp.float64) * UM
        phase = TWO_PI * index.neff(wl_m) * length_m / wl_m
        t = amplitude * jnp.exp(1j * phase)
        return sax.reciprocal({(p_in, p_out): t})

    model.info = {"length_m": length_m, "loss_db_per_cm": float(loss_db_per_cm)}
    return model


def arc_model(index, radius_m, angle=TWO_PI, loss_db_per_cm=0.0, ports=("o1", "o2")):
    """A bend of ``radius_m`` over ``angle`` radians; ``index`` is the *bent*
    guide's ``n_eff(lambda)`` at ``kappa = 1/R`` (the strip dataset's
    curvature axis), the length is the arc at the ring's centre radius."""
    return straight_model(index, float(radius_m) * float(angle), loss_db_per_cm, ports)


def ideal_coupler(r=None, kappa=None, ports=("o1", "o2", "o3", "o4")):
    """Lossless, wavelength-flat, symmetric 4-port coupler.

    Ports: ``o1`` bus in, ``o2`` bus through, ``o3`` ring in, ``o4`` ring out.
    ``S(o1,o2) = S(o3,o4) = r`` and ``S(o1,o4) = S(o3,o2) = i kappa``; unitary
    for ``r**2 + kappa**2 = 1``.  A stand-in for the DBEME coupler while the
    ring algebra and the sax composition are being checked against each other.
    """
    if r is None and kappa is None:
        raise ValueError("give r or kappa")
    if r is None:
        r = float(np.sqrt(1.0 - float(kappa) ** 2))
    if kappa is None:
        kappa = float(np.sqrt(1.0 - float(r) ** 2))
    o1, o2, o3, o4 = ports
    r = jnp.asarray(float(r), dtype=jnp.complex128)
    ik = jnp.asarray(1j * float(kappa), dtype=jnp.complex128)

    def model(*, wl: float = 1.55) -> sax.SDict:
        ones = jnp.ones_like(jnp.asarray(wl, dtype=jnp.float64))
        return sax.reciprocal({(o1, o2): r * ones, (o3, o4): r * ones,
                               (o1, o4): ik * ones, (o3, o2): ik * ones})

    model.info = {"r": float(np.real(r)), "kappa": float(kappa)}
    return model
