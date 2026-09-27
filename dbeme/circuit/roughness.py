"""Sidewall-roughness scattering loss: Payne & Lacey with the effective-index method.

A PML mode solver absorbs the radiation of a *smooth* geometry; roughness
loss is statistical and needs a model with the roughness rms ``sigma`` and
correlation length ``L_c`` as declared inputs (`tasks/11` §2.2).  This is
the planar-waveguide result of

    F. P. Payne and J. P. R. Lacey, "A theoretical analysis of scattering
    loss from planar optical waveguides", Opt. Quantum Electron. 26, 977
    (1994)

for an exponential autocorrelation, applied laterally through the
effective-index method: the 220 nm slab's TE index is the "core" of a
symmetric slab of width ``w`` in silica.  It is an order-of-magnitude model
(its own authors say so); the repo treats it as such and checks it against
the 2-3 dB/cm a 500 x 220 nm strip typically shows.

    alpha = sigma^2 / (k0 d^4 n1) * g(V) * f(x, gamma)          [power, 1/m]
    g(V)  = U^2 V^2 / (1 + W)
    f     = x * sqrt( sqrt((1+x^2)^2 + 2 x^2 gamma^2) + 1 - x^2 )
              / sqrt((1+x^2)^2 + 2 x^2 gamma^2)
    x = W L_c / d,  gamma = n2 V / (n1 W sqrt(Delta)),  Delta = (n1^2 - n2^2) / (2 n1^2)
    U = d sqrt(k0^2 n1^2 - beta^2),  W = d sqrt(beta^2 - k0^2 n2^2),  V = k0 d sqrt(n1^2 - n2^2)

with ``d`` the half-width.  The bend enhancement is taken from the mode's
own field: ``alpha(R) = alpha(inf) * F(R) / F(inf)`` with ``F`` the sidewall
field factor (:func:`sidewall_factor`), because a bend pushes the mode onto
the outer wall.
"""

import numpy as np
from scipy.optimize import brentq

TWO_PI = 2.0 * np.pi


def slab_te_index(thickness, n_core, n_clad, wavelength):
    """Fundamental TE index of a symmetric slab (even mode)."""
    k0 = TWO_PI / wavelength

    def f(n):
        kx = k0 * np.sqrt(n_core ** 2 - n ** 2)
        g = k0 * np.sqrt(n ** 2 - n_clad ** 2)
        return np.tan(kx * thickness / 2.0) - g / kx

    lo, hi = n_clad + 1e-9, n_core - 1e-9
    # the first branch of tan: walk down from n_core until f changes sign
    grid = np.linspace(hi, lo, 4000)
    vals = np.array([f(n) for n in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0 \
                and abs(vals[i] - vals[i + 1]) < 50:
            return float(brentq(f, grid[i + 1], grid[i]))
    raise RuntimeError("no guided TE slab mode")


def payne_lacey_alpha(width, n1, n2, neff, wavelength, sigma, correlation_length):
    """Power attenuation (1/m) of a symmetric slab of width ``width`` and
    index ``n1`` in ``n2``, mode index ``neff``, roughness ``sigma`` and
    exponential correlation length ``correlation_length``."""
    k0 = TWO_PI / wavelength
    d = 0.5 * width
    beta = k0 * neff
    U = d * np.sqrt(max(k0 ** 2 * n1 ** 2 - beta ** 2, 0.0))
    W = d * np.sqrt(max(beta ** 2 - k0 ** 2 * n2 ** 2, 0.0))
    V = k0 * d * np.sqrt(n1 ** 2 - n2 ** 2)
    delta = (n1 ** 2 - n2 ** 2) / (2.0 * n1 ** 2)
    x = W * correlation_length / d
    gamma = n2 * V / (n1 * W * np.sqrt(delta))
    g = U ** 2 * V ** 2 / (1.0 + W)
    root = np.sqrt((1.0 + x ** 2) ** 2 + 2.0 * x ** 2 * gamma ** 2)
    f = x * np.sqrt(root + 1.0 - x ** 2) / root
    return sigma ** 2 / (k0 * d ** 4 * n1) * g * f


def roughness_loss_db_per_cm(width, thickness, n_core, n_clad, wavelength, sigma, correlation_length,
                             neff=None):
    """Straight-guide roughness loss in dB/cm via the effective-index method.

    ``neff`` may be given (the full 2-D mode's index); otherwise the lateral
    slab's own even-mode index is used, which keeps the model self-consistent.
    Returns ``(loss_db_per_cm, details)``.
    """
    n_slab = slab_te_index(thickness, n_core, n_clad, wavelength)
    n_lat = slab_te_index(width, n_slab, n_clad, wavelength)
    n_use = float(neff) if neff is not None else n_lat
    alpha = payne_lacey_alpha(width, n_slab, n_clad, n_use, wavelength, sigma, correlation_length)
    return 10.0 * np.log10(np.e) * alpha / 100.0, {"n_slab": n_slab, "n_lateral_eim": n_lat,
                                                     "neff_used": n_use, "alpha_per_m": float(alpha)}


def sidewall_factor(mode_data, mode, x_edges, y_range):
    """``sum |E|^2 on the sidewall lines / sum |E|^2 dA`` for one mode.

    ``x_edges`` are the sidewall positions (metres); each line is sampled as
    the mean of the columns just inside and just outside the edge (the normal
    field is discontinuous there), over ``y_range = (y_lo, y_hi)`` - the core
    thickness.  Returns ``(F_total, [F per edge])`` in 1/m.
    """
    x, y = np.real(mode_data.x), np.real(mode_data.y)
    E = mode_data.E[mode]
    intensity = np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2 + np.abs(E[2]) ** 2
    total = np.trapezoid(np.trapezoid(intensity, y, axis=1), x)
    rows = (y >= y_range[0]) & (y <= y_range[1])
    per_edge = []
    dx = float(np.min(np.diff(x)))
    for xe in x_edges:
        on = np.flatnonzero(np.abs(x - xe) < 1e-3 * dx)
        if on.size:                       # the edge is a grid point: take it
            line = intensity[int(on[0])]
        else:                             # otherwise the mean of its two neighbours
            i = int(np.searchsorted(x, xe))
            line = 0.5 * (intensity[max(i - 1, 0)] + intensity[min(i, len(x) - 1)])
        per_edge.append(float(np.trapezoid(line[rows], y[rows]) / total))
    return float(sum(per_edge)), per_edge
