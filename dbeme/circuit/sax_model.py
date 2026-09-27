"""A sax model from a DBEME S-matrix sampled at a few wavelengths.

``sax_model_from_samples`` takes what a per-wavelength DBEME run produces -
``S[m, i, j]`` at wavelengths ``wavelengths[m]`` (metres), the amplitude
*from* ``ports[i]`` *to* ``ports[j]`` - and returns a sax model
``model(wl=...)`` (``wl`` in microns) that interpolates between the solves the
way `tasks/11` §1.1 prescribes: ``|S|`` by a monotone cubic, the phase by a
polynomial fit of its optical length after unwrapping (:mod:`phase_fit`).

Ports are sax names; multimode ports are ``"o1@TE0"``, ``"o1@TE1"`` and so on.
The model is made reciprocal (``S = S^T``) and reports how far the samples
were from that in ``model.info["asymmetry"]`` - a §5.1 check for free.
"""

import numpy as np
import jax.numpy as jnp
import sax

from .phase_fit import OpticalLengthFit, PchipJax, unwrap_against

UM = 1e-6


def sax_model_from_samples(wavelengths, S, ports, *, phase_reference=None, degree=3,
                           symmetrize=True, zero_threshold=1e-12):
    """Build the model.

    :param wavelengths: ``(M,)`` metres, ascending.
    :param S: ``(M, P, P)`` complex, ``S[m, i, j]`` from ``ports[i]`` to ``ports[j]``.
    :param ports: ``P`` sax port names.
    :param phase_reference: ``None`` (plain ``np.unwrap`` along wavelength -
        valid while the phase moves < pi between samples), an ``(M, P, P)``
        array, or a callable of the wavelength returning ``(P, P)``: a
        continuous phase within pi of the truth (see
        :func:`phase_fit.unwrap_against`).
    :param degree: polynomial degree of the optical-length fit (<= M - 1).
    :param symmetrize: enforce reciprocity by averaging ``S`` and ``S^T``.
    :param zero_threshold: entries whose magnitude never exceeds this are
        left out of the SDict (sax reads a missing key as zero).
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    S = np.asarray(S, dtype=complex)
    ports = list(ports)
    M, P = len(wavelengths), len(ports)
    if S.shape != (M, P, P):
        raise ValueError(f"S must be (M, P, P) = {(M, P, P)}, got {S.shape}")
    if M > 1 and np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be ascending")

    asymmetry = float(np.max(np.abs(S - np.swapaxes(S, 1, 2)))) if P > 1 else 0.0
    if symmetrize:
        S = 0.5 * (S + np.swapaxes(S, 1, 2))

    if phase_reference is not None and not callable(phase_reference):
        phase_reference = np.asarray(phase_reference, dtype=float)
        if phase_reference.shape != (M, P, P):
            raise ValueError("phase_reference must be (M, P, P) or callable")

    entries = {}
    residuals = {}
    for i in range(P):
        for j in range(i, P) if symmetrize else range(P):
            samples = S[:, i, j]
            mag = np.abs(samples)
            if np.max(mag) <= zero_threshold:
                continue
            if phase_reference is None:
                ref = None
            elif callable(phase_reference):
                def ref(wl, i=i, j=j):
                    return np.asarray([np.asarray(phase_reference(w))[i, j] for w in np.atleast_1d(wl)])
            else:
                ref = phase_reference[:, i, j]
            phase = unwrap_against(wavelengths, np.angle(samples), ref)
            fit = OpticalLengthFit(wavelengths, phase, degree) if M > 1 else None
            entries[(ports[i], ports[j])] = (PchipJax(wavelengths, mag), fit, float(phase[0]))
            residuals[(ports[i], ports[j])] = fit.residual_rad if fit is not None else 0.0

    def model(*, wl: float = 1.55) -> sax.SDict:
        wl_m = jnp.asarray(wl, dtype=jnp.float64) * UM
        out = {}
        for key, (mag, fit, phase0) in entries.items():
            phase = fit.phase(wl_m) if fit is not None else phase0
            out[key] = mag(wl_m) * jnp.exp(1j * phase)
        return sax.reciprocal(out) if symmetrize else out

    model.info = {
        "ports": ports,
        "wavelength_range_m": (float(wavelengths[0]), float(wavelengths[-1])),
        "samples": int(M),
        "degree": int(min(degree, M - 1)),
        "asymmetry": asymmetry,
        "phase_fit_residual_rad": residuals,
    }
    return model


def sax_model_from_grid(param_values, wavelengths, S, ports, *, param_name, phase_reference=None,
                        degree=2, symmetrize=True, zero_threshold=1e-12):
    """A sax model with one continuous *design* parameter, from a 2-D sample grid.

    `tasks/15` §5.4 Route C: ``S[p, m, i, j]`` sampled at design values
    ``param_values[p]`` (e.g. a coupler's straight length, or its end width
    difference) and wavelengths ``wavelengths[m]``.  Along wavelength each
    entry is fitted as :func:`sax_model_from_samples` does (magnitude PCHIP,
    phase by an optical-length polynomial after unwrapping against
    ``phase_reference``); the fit's ingredients are then interpolated along
    the design axis with a monotone cubic, so the model
    ``model(wl=..., <param_name>=...)`` is differentiable in both.  The
    gradient is the interpolant's, over the cached grid; the mode solver is
    not differentiated.

    ``phase_reference`` is ``None``, an ``(P, M, P_ports, P_ports)`` array or a
    callable ``(param, wl) -> (P_ports, P_ports)``: a continuous phase within
    pi of the truth at every sample - on a long element it must carry the
    group index (`tasks/11` §1.1).
    """
    param_values = np.asarray(param_values, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    S = np.asarray(S, dtype=complex)
    ports = list(ports)
    Np, M, P = len(param_values), len(wavelengths), len(ports)
    if S.shape != (Np, M, P, P):
        raise ValueError(f"S must be (Np, M, P, P) = {(Np, M, P, P)}, got {S.shape}")
    if Np > 1 and np.any(np.diff(param_values) <= 0):
        raise ValueError("param_values must be ascending")
    if symmetrize:
        S = 0.5 * (S + np.swapaxes(S, 2, 3))

    def reference(p, i, j):
        if phase_reference is None:
            return None
        if callable(phase_reference):
            return np.asarray([np.asarray(phase_reference(param_values[p], w))[i, j] for w in wavelengths], dtype=float)
        return np.asarray(phase_reference)[p, :, i, j]

    entries = {}
    residuals = {}
    for i in range(P):
        for j in range(i, P) if symmetrize else range(P):
            if np.max(np.abs(S[:, :, i, j])) <= zero_threshold:
                continue
            mags = np.abs(S[:, :, i, j])                                          # (Np, M)
            phases = np.stack([unwrap_against(wavelengths, np.angle(S[p, :, i, j]), reference(p, i, j))
                               for p in range(Np)])                                # (Np, M)
            # make the unwrapped phases continuous along the design axis too
            for p in range(1, Np):
                k = np.round((phases[p] - phases[p - 1]) / (2 * np.pi))
                phases[p] -= 2 * np.pi * np.round(np.median(k))
            fits = [OpticalLengthFit(wavelengths, phases[p], degree) if M > 1 else None for p in range(Np)]
            coef = np.stack([f.length.coefficients for f in fits]) if M > 1 else phases[:, :1]     # (Np, degree + 1)
            mag_interp = [PchipJax(param_values, mags[:, m]) for m in range(M)]
            coef_interp = [PchipJax(param_values, coef[:, d]) for d in range(coef.shape[1])]
            entries[(ports[i], ports[j])] = (mag_interp, coef_interp, fits[0] if M > 1 else None)
            residuals[(ports[i], ports[j])] = float(max(f.residual_rad for f in fits)) if M > 1 else 0.0

    def model(*, wl: float = 1.55, **kwargs) -> sax.SDict:
        wl_m = jnp.asarray(wl, dtype=jnp.float64) * UM
        p = jnp.asarray(kwargs.get(param_name, param_values[0]), dtype=jnp.float64)
        out = {}
        for key, (mag_interp, coef_interp, fit0) in entries.items():
            # magnitude: PCHIP along the design axis at each sampled wavelength, then PCHIP in wavelength
            mags_at_p = jnp.stack([mi(p) for mi in mag_interp])                      # (M,)
            mag = _pchip_eval_jnp(wavelengths, mags_at_p, wl_m) if M > 1 else mags_at_p[0]
            if fit0 is not None:
                c = jnp.stack([ci(p) for ci in coef_interp])                          # (degree + 1,)
                x = (wl_m - fit0.length.wl0) / fit0.length.wl0
                length = c[-1]
                for d in range(c.shape[0] - 2, -1, -1):                              # Horner, traced coefficients
                    length = length * x + c[d]
                phase = 2 * jnp.pi * length / wl_m
            else:
                phase = coef_interp[0](p)
            out[key] = mag * jnp.exp(1j * phase)
        return sax.reciprocal(out) if symmetrize else out

    # sax reads settings off the signature: expose the design parameter explicitly
    import inspect

    model.__signature__ = inspect.Signature([
        inspect.Parameter("wl", inspect.Parameter.KEYWORD_ONLY, default=1.55, annotation=float),
        inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, default=float(param_values[0]), annotation=float)])
    model.info = {"ports": ports, "param_name": param_name, "param_range": (float(param_values[0]), float(param_values[-1])),
                  "wavelength_range_m": (float(wavelengths[0]), float(wavelengths[-1])), "samples": (int(Np), int(M)),
                  "phase_fit_residual_rad": residuals}
    return model


def _pchip_eval_jnp(x_nodes, y_nodes, x):
    """Linear interpolation in ``jnp`` through nodes whose *values* are traced (magnitudes at sampled wavelengths)."""
    x_nodes = jnp.asarray(x_nodes, dtype=jnp.float64)
    idx = jnp.clip(jnp.searchsorted(x_nodes, x, side="right") - 1, 0, len(x_nodes) - 2)
    x0, x1 = x_nodes[idx], x_nodes[idx + 1]
    t = (x - x0) / (x1 - x0)
    return y_nodes[idx] * (1 - t) + y_nodes[idx + 1] * t
