"""Lattice-form (cascaded Mach-Zehnder) filter cells: closed form and sax netlist.

A lattice cell is ``N + 1`` two-by-two couplers with ``N`` delay stages between
them (Jinguji & Kawachi 1995; Madsen & Zhao 1999).  The closed form here is the
2 x 2 transfer matrix of the cell - the *lattice theory* a sax composition of
the same blocks must reproduce (`tests/test_lattice.py`), and the cheap model
Phase 0 of `tasks/15` uses to compare topologies before any mode is solved.

Conventions match the rest of the circuit layer: a coupler with power cross
coupling ``kappa2`` is ``[[r, i k], [i k, r]]`` with ``r = sqrt(1 - kappa2)``
(`waveguide.ideal_coupler`: o1 in-1, o2 out-1, o3 in-2, o4 out-2), a delay
stage multiplies arm 1 by ``exp(i phi_1)`` and arm 2 by ``exp(i phi_2)``
(project sign ``exp(+i beta z)``), and the cell's ports are ``in1``, ``in2``
(coupler 0's o1, o3) and ``out1``, ``out2`` (coupler N's o2, o4).  Everything
is ``jnp`` so the transfer is differentiable in the couplings and the phases.
"""

import jax.numpy as jnp
import numpy as np

__all__ = ["coupler_matrix", "delay_matrix", "lattice_transfer", "lattice_netlist",
           "response_powers", "PORTS"]

PORTS = ("in1", "in2", "out1", "out2")


def coupler_matrix(kappa2):
    """``(..., 2, 2)`` coupler transfer for cross-power ``kappa2`` (scalar or array)."""
    k2 = jnp.asarray(kappa2, dtype=jnp.float64)
    r = jnp.sqrt(jnp.clip(1.0 - k2, 0.0, 1.0))
    k = jnp.sqrt(jnp.clip(k2, 0.0, 1.0))
    zero = jnp.zeros_like(r)
    row1 = jnp.stack([r + 0j, 1j * k + zero], axis=-1)
    row2 = jnp.stack([1j * k + zero, r + 0j], axis=-1)
    return jnp.stack([row1, row2], axis=-2)


def delay_matrix(phi1, phi2):
    """``(..., 2, 2)`` diagonal delay stage ``diag(exp(i phi1), exp(i phi2))``."""
    p1 = jnp.asarray(phi1, dtype=jnp.float64)
    p2 = jnp.asarray(phi2, dtype=jnp.float64)
    p1, p2 = jnp.broadcast_arrays(p1, p2)
    zero = jnp.zeros_like(p1)
    row1 = jnp.stack([jnp.exp(1j * p1), zero + 0j], axis=-1)
    row2 = jnp.stack([zero + 0j, jnp.exp(1j * p2)], axis=-1)
    return jnp.stack([row1, row2], axis=-2)


def lattice_transfer(kappa2s, phases):
    """Transfer matrix ``M`` of the cell, ``out = M @ in``.

    :param kappa2s: ``N + 1`` cross couplings (each scalar, or broadcastable
        to the phase batch shape).
    :param phases: ``(N, 2, ...)`` - for stage ``k`` the arm-1 and arm-2
        phases, with any trailing batch shape (e.g. a wavelength axis).
    :returns: ``(..., 2, 2)`` complex.
    """
    kappa2s = list(kappa2s)
    n_stages = len(phases)
    if len(kappa2s) != n_stages + 1:
        raise ValueError(f"{len(kappa2s)} couplers for {n_stages} stages; need N + 1")
    m = coupler_matrix(kappa2s[0])
    for k in range(n_stages):
        d = delay_matrix(phases[k][0], phases[k][1])
        m = coupler_matrix(kappa2s[k + 1]) @ (d @ m)
    return m


def response_powers(kappa2s, phases):
    """Powers ``|M|^2`` as a dict keyed by ``(in, out)`` port names."""
    m = lattice_transfer(kappa2s, phases)
    p = jnp.abs(m) ** 2
    return {("in1", "out1"): p[..., 0, 0], ("in2", "out1"): p[..., 0, 1],
            ("in1", "out2"): p[..., 1, 0], ("in2", "out2"): p[..., 1, 1]}


def lattice_netlist(n_stages, couplers="coupler", arms="arm"):
    """sax netlist of a lattice cell.

    Instances ``C0 .. CN`` (couplers) and ``A{k}a`` / ``A{k}b`` (arm 1 / arm 2
    of stage ``k``); ``couplers`` is one component name for all couplers or a
    sequence of ``N + 1`` names (a library with a different kind per
    position); ``arms`` likewise one name or ``N`` pairs.  Settings such as
    ``length_um`` are given per instance at call time, as sax does.
    """
    n = int(n_stages)
    if isinstance(couplers, str):
        couplers = [couplers] * (n + 1)
    couplers = list(couplers)
    if len(couplers) != n + 1:
        raise ValueError("need N + 1 coupler component names")
    if isinstance(arms, str):
        arms = [(arms, arms)] * n
    arms = [tuple(a) if not isinstance(a, str) else (a, a) for a in arms]
    instances = {f"C{k}": {"component": couplers[k]} for k in range(n + 1)}
    connections = {}
    for k in range(n):
        instances[f"A{k}a"] = {"component": arms[k][0]}
        instances[f"A{k}b"] = {"component": arms[k][1]}
        connections[f"C{k},o2"] = f"A{k}a,o1"
        connections[f"A{k}a,o2"] = f"C{k+1},o1"
        connections[f"C{k},o4"] = f"A{k}b,o1"
        connections[f"A{k}b,o2"] = f"C{k+1},o3"
    ports = {"in1": "C0,o1", "in2": "C0,o3", "out1": f"C{n},o2", "out2": f"C{n},o4"}
    return {"instances": instances, "connections": connections, "ports": ports}


def unitarity_defect(kappa2s, phases):
    """``max |M^H M - I|`` - zero for the lossless closed form (a test helper)."""
    m = np.asarray(lattice_transfer(kappa2s, phases))
    eye = np.eye(2)
    g = np.swapaxes(np.conj(m), -1, -2) @ m
    return float(np.max(np.abs(g - eye)))
