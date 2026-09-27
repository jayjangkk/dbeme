"""Magnitude and phase between sparsely solved wavelengths.

DBEME solves one dataset per wavelength, so a component's S-matrix is known
at five to seven wavelengths across a band; a ring needs it at picometre
resolution.  Two rules (`tasks/11` §1.1):

* the magnitude is interpolated as ``|S|`` with a monotone cubic (PCHIP), so
  it never overshoots between samples;
* the phase is never interpolated wrapped, and never through a linear
  ``n_eff(lambda)`` (memory "S-param phase traps": that staircases the group
  delay).  It is unwrapped against a reference, turned into an *optical
  length* ``Lambda(lambda) = phi * lambda / (2 pi)`` (metres), and ``Lambda``
  is fitted with a low-order polynomial in ``(lambda - lambda0) / lambda0``.
  For a uniform guide ``Lambda = n_eff(lambda) * L``, so the coefficients are
  ``n_eff``, ``dn/dlambda`` (i.e. ``n_g``) and dispersion, and the fit is
  exact whenever ``n_eff`` is polynomial in ``lambda``.  (`tasks/11` wrote
  "a polynomial in 1/lambda of the phase"; that basis spends a degree on the
  trivial ``1/lambda`` carrier and is what this replaces - same family, one
  degree cheaper.)

Every evaluator is a ``jnp`` function of the wavelength in **metres**, so it
can be differentiated and traced; fitting is done once with numpy/scipy.
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import PchipInterpolator

TWO_PI = 2.0 * np.pi


def wrap(angle):
    """Into ``(-pi, pi]``."""
    return np.angle(np.exp(1j * np.asarray(angle, dtype=float)))


def unwrap_against(wavelengths, phase, reference=None):
    """Continuous phase from wrapped samples.

    With no ``reference`` this is ``np.unwrap`` along the wavelength axis,
    which is only unambiguous while the phase advances by less than ``pi``
    between samples: ``L * n_g * d_lambda / lambda**2 < 1/2``, i.e. about
    28 um of silicon strip at 10 nm steps.  Past that, pass a ``reference``
    - an array of the same shape or a callable of the wavelength - and the
    wrapped residual against it is unwrapped and added back
    (`em_simulation/propagator/sparams_phase.py`).

    What the reference must get right is the **slope**, not the offset: the
    residual's step between samples must stay under ``pi``, so the
    reference's *group* index has to be within ``lambda**2 / (2 L d_lambda)``
    of the truth (0.4 for 300 um at 10 nm steps) - a constant-``n_eff``
    reference fails on a long device.  An offset in ``n_eff`` only shifts the
    result by a multiple of ``2 pi``, which no S-parameter, and no delay,
    can see.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if reference is None:
        return np.unwrap(phase, axis=0)
    ref = reference(wavelengths) if callable(reference) else np.asarray(reference, dtype=float)
    ref = np.broadcast_to(ref, phase.shape)
    residual = np.unwrap(wrap(phase - ref), axis=0)
    return ref + residual


class PolynomialFit:
    """``y(lambda) = sum_k c_k x**k`` with ``x = (lambda - lambda0) / lambda0``.

    Centred and scaled so the fit is well conditioned over a 100 nm band; the
    evaluator is Horner's rule in ``jnp``.
    """

    def __init__(self, wl0, coefficients):
        self.wl0 = float(wl0)
        self.coefficients = np.asarray(coefficients, dtype=float)

    @classmethod
    def fit(cls, wavelengths, values, degree):
        wavelengths = np.asarray(wavelengths, dtype=float)
        values = np.asarray(values, dtype=float)
        if wavelengths.ndim != 1 or len(wavelengths) != len(values):
            raise ValueError("wavelengths and values must be 1-D of equal length")
        degree = int(min(degree, len(wavelengths) - 1))
        if degree < 0:
            raise ValueError("need at least one sample")
        wl0 = float(np.mean(wavelengths))
        x = (wavelengths - wl0) / wl0
        coefficients = np.polynomial.polynomial.polyfit(x, values, degree)
        fit = cls(wl0, coefficients)
        fit.residual = float(np.max(np.abs(np.asarray(fit(wavelengths)) - values)))
        fit.degree = degree
        return fit

    def __call__(self, wavelengths):
        x = (jnp.asarray(wavelengths, dtype=jnp.float64) - self.wl0) / self.wl0
        out = jnp.zeros_like(x) + self.coefficients[-1]
        for c in self.coefficients[-2::-1]:
            out = out * x + c
        return out

    def derivative(self):
        """``d y / d lambda`` as another PolynomialFit."""
        d = np.polynomial.polynomial.polyder(self.coefficients) / self.wl0
        if d.size == 0:
            d = np.zeros(1)
        return PolynomialFit(self.wl0, d)


class OpticalLengthFit:
    """Phase through its optical length ``Lambda = phi * lambda / (2 pi)``.

    ``phase`` must be *continuous* (see :func:`unwrap_against`).  ``degree``
    3 is the default in `tasks/11`; ``residual_rad`` is the largest misfit at
    the samples, the gate's check that a low-order fit is enough.
    """

    def __init__(self, wavelengths, phase, degree=3):
        wavelengths = np.asarray(wavelengths, dtype=float)
        phase = np.asarray(phase, dtype=float)
        self.length = PolynomialFit.fit(wavelengths, phase * wavelengths / TWO_PI, degree)
        self.wavelengths = wavelengths
        self.residual_rad = float(np.max(np.abs(np.asarray(self.phase(wavelengths)) - phase)))
        self.degree = self.length.degree

    def optical_length(self, wavelengths):
        """``Lambda(lambda)`` in metres (``n_eff * L`` for a uniform guide)."""
        return self.length(wavelengths)

    def phase(self, wavelengths):
        wl = jnp.asarray(wavelengths, dtype=jnp.float64)
        return TWO_PI * self.length(wl) / wl

    def group_optical_length(self, wavelengths):
        """``Lambda - lambda dLambda/dlambda`` (``n_g * L`` for a uniform guide).

        Group delay is this over ``c``: ``tau = +d(phase)/d(omega)`` in the
        project convention.
        """
        wl = jnp.asarray(wavelengths, dtype=jnp.float64)
        return self.length(wl) - wl * self.length.derivative()(wl)


class PchipJax:
    """Monotone cubic through the samples, evaluated in ``jnp``.

    Built once with scipy's PCHIP; the piecewise polynomial is applied with a
    ``searchsorted`` so the evaluator is a traceable function of the
    wavelength.  Outside the sampled range the end pieces are extrapolated,
    exactly as scipy does - a ring band should sit inside the solved band.
    """

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) == 1:
            self.constant = float(y[0])
            self.x = x
            return
        self.constant = None
        pchip = PchipInterpolator(x, y, extrapolate=True)
        self.x = pchip.x
        self.c = pchip.c  # (4, n-1)
        # jnp copies for tracing: a traced index into a numpy array fails under jit
        self._x = jnp.asarray(self.x, dtype=jnp.float64)
        self._c = jnp.asarray(self.c, dtype=jnp.float64)

    def __call__(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        if self.constant is not None:
            return jnp.zeros_like(x) + self.constant
        idx = jnp.clip(jnp.searchsorted(self._x, x, side="right") - 1, 0, len(self.x) - 2)
        dx = x - self._x[idx]
        c = self._c
        return ((c[0][idx] * dx + c[1][idx]) * dx + c[2][idx]) * dx + c[3][idx]
