r"""A fibre beam at a chip facet, projected onto a cross section's modes.

The launch for an edge coupler: a Gaussian of waist ``w0`` (``MFD / 2``) in
the facet plane, expanded in the forward modes of the first cross section of
a DBEME path.  The coefficient vector goes straight into the path's
S-matrix, ``out = S @ [a, 0]``.

Conventions are the dataset's (``dbeme.fde.assemble``):

* the overlap is **unconjugated**, :math:`\langle E_a, H_b \rangle =
  \tfrac12 \int (E_a \times H_b)_z\, dA`, on the solver's grid with its own
  (possibly complex-stretched) integration measure;
* the modes are normalised to :math:`\langle E_m, H_m \rangle = 1` and
  Löwdin-biorthogonalised, so ``M = <E_j, H_k>`` is the identity up to an
  antisymmetric residue;
* the backward partner of a mode keeps ``E_t`` and negates ``H_t``.

With :math:`A_m = \langle E_g, H_m\rangle`, :math:`B_m = \langle E_m,
H_g\rangle`, the beam's E expands as :math:`E_t = \sum s_m E_m` with
:math:`A = M^T s`, and its H as :math:`H_t = \sum d_m H_m` with
:math:`B = M d`.  At the facet the incident beam plus a reflection must
match the transmitted modes: :math:`(1 + r) E_g = \sum t_m E_m`,
:math:`(1 - r) H_g = \sum t_m H_m`.  Resolving that per mode gives the
launch

.. math::  t_m = \frac{2 s_m d_m}{s_m + d_m},

exact when the mode has the beam's shape (it reduces to the Fresnel
:math:`2 n_1/(n_1+n_2)` scaled to unit powers, so :math:`|t|^2 = 1 - |r|^2`),
and equal to :math:`s_m = d_m` - the plain overlap - when the beam's index
matches the mode's.  The naive expansion coefficient :math:`(s + d)/2` is
*not* a transmission: for a 1 -> 2 index step it gives
:math:`|t|^2 = 1.125`.  At an oxide-clad facet (beam in SiO2, tip mode at
1.449) the two differ by < 1e-5.

**Units of H.**  The EMpy-based backends (``EmepyFDE``, ``PMLBackend``)
return H scaled by the vacuum impedance, so a plane wave in a medium of
index ``n`` has :math:`H_t = n\, \hat z \times E_t`: pass ``z_units=1``.
``FemwellBackend`` returns SI fields: pass ``z_units=ETA0``.  Getting this
wrong weights the two halves of the projection 377x apart and nothing warns.

**Normalisation.**  The beam is scaled to unit power on the *infinite*
plane, analytically, and then cut to the physical interior of the window
(zero in the PML).  Power the window does not hold - the beam's tails in
the PML, which at a real facet land on the substrate or in air - is
therefore counted as lost, not renormalised away.
"""

import numpy as np

from ..data_updater import overlap_calculation_tool as oct
from ..fde.assemble import overlap_matrix

ETA0 = 376.730313668

__all__ = [
    "ETA0",
    "gaussian_fields",
    "gaussian_launch",
    "paper_overlap",
    "power_coupling",
    "physical_interior",
]


def physical_interior(x, y):
    """Boolean ``(nx, ny)`` mask of the nodes outside every PML layer."""
    return (np.imag(np.asarray(x)) == 0)[:, None] & (np.imag(np.asarray(y)) == 0)[None, :]


def _real_weights(x, y):
    dx = np.abs(oct.compute_differences(np.real(np.asarray(x, dtype=complex))))
    dy = np.abs(oct.compute_differences(np.real(np.asarray(y, dtype=complex))))
    return np.outer(dx, dy)


def gaussian_fields(x, y, w0, center=(0.0, 0.0), pol="TE", n_medium=1.447, z_units=1.0):
    """The beam's ``(E, H)``, each ``(1, 3, nx, ny)``, unit power on the infinite plane.

    :param w0: 1/e field radius in metres (``MFD / 2``).
    :param center: ``(x0, y0)`` of the beam axis, metres.
    :param pol: ``"TE"`` (E along x) or ``"TM"`` (E along y).
    :param n_medium: Index the beam arrives in; sets ``H = (n / z_units) z x E``.
    :param z_units: ``1`` for EMpy-based backends, :data:`ETA0` for femwell.
    """
    xr = np.real(np.asarray(x, dtype=complex)) - center[0]
    yr = np.real(np.asarray(y, dtype=complex)) - center[1]
    X, Y = np.meshgrid(xr, yr, indexing="ij")
    g = np.exp(-(X**2 + Y**2) / w0**2) * physical_interior(x, y)
    # 1/2 (n/Z) int |g|^2 dA over the plane = 1/2 (n/Z) pi w0^2 / 2
    p_inf = 0.5 * (n_medium / z_units) * np.pi * w0**2 / 2.0
    g = g / np.sqrt(p_inf)
    E = np.zeros((1, 3) + g.shape, dtype=complex)
    H = np.zeros_like(E)
    if pol.upper() == "TE":
        E[0, 0] = g
        H[0, 1] = (n_medium / z_units) * g      # z x x = +y
    elif pol.upper() == "TM":
        E[0, 1] = g
        H[0, 0] = -(n_medium / z_units) * g     # z x y = -x
    else:
        raise ValueError("pol must be 'TE' or 'TM'")
    return E, H


def gaussian_launch(x, y, E, H, w0, center=(0.0, 0.0), pol="TE", n_medium=1.447,
                    z_units=1.0, rcond=None):
    """Forward and backward coefficients of the beam in the modes ``(E, H)``.

    :param E, H: The first cross section's **forward** modes in the dataset
        basis (``assemble`` output, normalised and biorthogonal), each
        ``(N, 3, nx, ny)``.
    :param rcond: ``None`` solves ``M`` exactly; a number uses a pseudo-inverse
        with that cutoff, for a basis with near-null directions.
    :returns: dict with ``a`` (the forward launch ``t_m``, ``(N,)``), ``s``
        and ``d`` (the E- and H-expansion coefficients), ``A``, ``B`` (the
        raw projections), ``window_power`` (the beam's power inside the
        physical interior, of 1).
    """
    Eg, Hg = gaussian_fields(x, y, w0, center, pol, n_medium, z_units)
    A = overlap_matrix(Eg, H, x, y)[0]
    B = overlap_matrix(E, Hg, x, y)[:, 0]
    M = overlap_matrix(E, H, x, y)
    if rcond is None:
        s = np.linalg.solve(M.T, A)
        d = np.linalg.solve(M, B)
    else:
        s = np.linalg.pinv(M.T, rcond=rcond) @ A
        d = np.linalg.pinv(M, rcond=rcond) @ B
    den = s + d
    ok = np.abs(den) > 1e-9 * max(np.abs(den).max(), 1e-300)
    a = np.zeros_like(den)
    a[ok] = 2.0 * s[ok] * d[ok] / den[ok]
    window_power = float(np.real(overlap_matrix(Eg, Hg, x, y)[0, 0]))
    return {"a": a, "s": s, "d": d, "A": A, "B": B, "window_power": window_power}


def paper_overlap(x, y, E_mode, w0, center=(0.0, 0.0), pol="TE"):
    r"""The scalar overlap of Wan & Wang's eq. (1).

    :math:`|\int E_1 E_2\, dA|^2 / (\int |E_1|^2 dA \int |E_2|^2 dA)` with
    ``E_1`` the Gaussian and ``E_2`` the mode's dominant transverse component
    (``Ex`` for TE, ``Ey`` for TM), on the physical interior with real
    weights.  Gauge- and normalisation-free.

    :param E_mode: One mode's ``(3, nx, ny)`` electric field.
    """
    c = 0 if pol.upper() == "TE" else 1
    w = _real_weights(x, y) * physical_interior(x, y)
    xr = np.real(np.asarray(x, dtype=complex)) - center[0]
    yr = np.real(np.asarray(y, dtype=complex)) - center[1]
    X, Y = np.meshgrid(xr, yr, indexing="ij")
    g = np.exp(-(X**2 + Y**2) / w0**2)
    e = E_mode[c]
    num = np.abs(np.sum(g * e * w)) ** 2
    den = np.sum(g**2 * w) * np.sum(np.abs(e) ** 2 * w)
    return float(num / den)


def power_coupling(x, y, E_mode, H_mode, w0, center=(0.0, 0.0), pol="TE",
                   n_medium=1.447, z_units=1.0):
    r"""Butt-coupling power efficiency of the beam into one mode, conjugated.

    :math:`\eta = \mathrm{Re}\!\left[\frac{\langle E_g, H_m^*\rangle
    \langle E_m, H_g^*\rangle}{\langle E_m, H_m^*\rangle}\right] /
    \mathrm{Re}\,\langle E_g, H_g^* \rangle`, on the physical interior with
    real weights - the standard mode-overlap coupling, without the facet's
    reflection.  The beam's power is taken on the infinite plane, so power
    outside the window counts as lost.
    """
    Eg, Hg = gaussian_fields(x, y, w0, center, pol, n_medium, z_units)
    w = _real_weights(x, y) * physical_interior(x, y)

    def cross(Ea, Hb):
        return 0.5 * np.sum((Ea[0] * Hb[1] - Ea[1] * Hb[0]) * w)

    Em, Hm = E_mode, H_mode
    num = cross(Eg[0], np.conj(Hm)) * cross(Em, np.conj(Hg[0]))
    den = cross(Em, np.conj(Hm))
    return float(np.real(num / den))       # beam power on the infinite plane is 1
