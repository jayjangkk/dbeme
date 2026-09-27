"""`sax_model_from_grid`: a design-parameter surrogate that reproduces its samples and differentiates.

A synthetic directional coupler ``S(L, lambda)`` - cross ``i sin(pi dn L /
lambda)``, bar ``cos``, common phase ``2 pi n L / lambda`` - sampled on a
grid of straight lengths and wavelengths.  The model must return the sample
values at the grid points, interpolate the coupling between them to the
accuracy of a monotone cubic, and give a ``jax.grad`` with respect to the
length that matches finite differences of the same interpolant.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dbeme.circuit.sax_model import sax_model_from_grid  # noqa: E402

N_EFF, N_G, DN, WL0 = 1.626, 1.991, 0.038, 1.31e-6
PORTS = ("o1", "o2", "o3", "o4")


def _S(L, wl):
    n = N_EFF + (N_EFF - N_G) * (wl - WL0) / WL0
    phi = np.pi * DN * L / wl
    common = np.exp(1j * 2 * np.pi * n * L / wl)
    S = np.zeros((4, 4), dtype=complex)
    S[0, 1] = S[1, 0] = np.cos(phi) * common
    S[2, 3] = S[3, 2] = np.cos(phi) * common
    S[0, 3] = S[3, 0] = 1j * np.sin(phi) * common
    S[2, 1] = S[1, 2] = 1j * np.sin(phi) * common
    return S


def _reference(L, wl):
    n = N_EFF + (N_EFF - N_G) * (wl - WL0) / WL0
    return np.full((4, 4), 2 * np.pi * n * L / wl)


def _build():
    lengths = np.linspace(2e-6, 20e-6, 10)
    wls = np.array([1.300e-6, 1.310e-6, 1.320e-6])
    S = np.array([[_S(L, wl) for wl in wls] for L in lengths])
    return lengths, wls, sax_model_from_grid(lengths, wls, S, PORTS, param_name="length_m", phase_reference=_reference)


def test_grid_model_reproduces_samples():
    lengths, wls, model = _build()
    for L in (lengths[0], lengths[4], lengths[-1]):
        for wl in wls:
            out = model(wl=wl * 1e6, length_m=L)
            ref = _S(L, wl)
            assert abs(complex(out[("o1", "o4")]) - ref[0, 3]) < 1e-6
            assert abs(complex(out[("o1", "o2")]) - ref[0, 1]) < 1e-6


def test_grid_model_interpolates_and_differentiates():
    lengths, wls, model = _build()
    L = 0.5 * (lengths[3] + lengths[4])
    k2 = float(abs(model(wl=1.31, length_m=L)[("o1", "o4")]) ** 2)
    assert abs(k2 - float(np.sin(np.pi * DN * L / WL0) ** 2)) < 2e-3

    def kappa2(L):
        return jnp.abs(model(wl=1.31, length_m=L)[("o1", "o4")]) ** 2

    g = float(jax.grad(kappa2)(L))
    h = 1e-9
    fd = (float(kappa2(L + h)) - float(kappa2(L - h))) / (2 * h)
    assert abs(g - fd) < 1e-3 * abs(fd)
