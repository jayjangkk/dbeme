"""The analytical ring (`em_simulation/circuit/ring.py`) against brute force.

`tasks/11` §1.3 and the Phase 1 gate: the amplitude and intensity forms agree,
a lossless ring conserves power, critical coupling nulls the through port,
and every closed-form figure of merit matches what a dense scan of the
transfer function measures.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from em_simulation.circuit.ring import (  # noqa: E402
    add_drop,
    add_drop_amplitude,
    all_pass,
    all_pass_amplitude,
    ring_metrics,
    round_trip_phase,
    single_pass_amplitude,
)

PHI = np.linspace(-np.pi, 3 * np.pi, 4001)
CASES = [(0.95, 0.99), (0.7, 0.9), (0.99, 0.999), (0.5, 0.5)]


@pytest.mark.parametrize("r,a", CASES)
def test_all_pass_amplitude_matches_intensity(r, a):
    assert np.max(np.abs(np.abs(all_pass_amplitude(r, a, PHI)) ** 2 - all_pass(r, a, PHI))) < 1e-12


@pytest.mark.parametrize("r1,r2,a", [(0.95, 0.95, 0.99), (0.9, 0.8, 0.95), (0.99, 0.97, 1.0)])
def test_add_drop_amplitude_matches_intensity(r1, r2, a):
    tp, td = add_drop_amplitude(r1, r2, a, PHI)
    Tp, Td = add_drop(r1, r2, a, PHI)
    assert np.max(np.abs(np.abs(tp) ** 2 - Tp)) < 1e-12
    assert np.max(np.abs(np.abs(td) ** 2 - Td)) < 1e-12


@pytest.mark.parametrize("r", [0.5, 0.9, 0.99])
def test_lossless_all_pass_is_unitary(r):
    assert np.max(np.abs(all_pass(r, 1.0, PHI) - 1.0)) < 1e-12


@pytest.mark.parametrize("r1,r2", [(0.9, 0.9), (0.95, 0.8), (0.99, 0.99)])
def test_lossless_add_drop_conserves_power(r1, r2):
    Tp, Td = add_drop(r1, r2, 1.0, PHI)
    assert np.max(np.abs(Tp + Td - 1.0)) < 1e-12


def test_critical_coupling_nulls_through_port():
    a = 0.97
    assert all_pass(a, a, 0.0) < 1e-30
    Tp, _ = add_drop(0.9 * a, 0.9, a, 0.0)
    assert Tp < 1e-30


def _scan(r, a, neff, ng, L, wl0, r2=None):
    """Dense wavelength scan with a first-order dispersive index."""
    wl = np.arange(wl0 - 12e-9, wl0 + 12e-9, 2e-13)
    n = neff + (wl - wl0) * (neff - ng) / wl0
    phi = round_trip_phase(n, L, wl)
    T = all_pass(r, a, phi) if r2 is None else add_drop(r, r2, a, phi)[0]
    return wl, T


def _minima(wl, T):
    idx = np.flatnonzero((T[1:-1] < T[:-2]) & (T[1:-1] < T[2:])) + 1
    return wl[idx], T[idx]


def _fwhm(wl, T, centre, t_min, t_max):
    half = 0.5 * (t_min + t_max)
    i = np.argmin(np.abs(wl - centre))
    lo = i
    while lo > 0 and T[lo] < half:
        lo -= 1
    hi = i
    while hi < len(T) - 1 and T[hi] < half:
        hi += 1
    return wl[hi] - wl[lo]


@pytest.mark.parametrize("r,a,r2", [(0.95, 0.99, None), (0.9, 0.995, None), (0.95, 0.99, 0.95)])
def test_metrics_against_dense_scan(r, a, r2):
    neff, ng, R, wl0 = 2.45, 4.2, 10e-6, 1550e-9
    L = 2 * np.pi * R
    wl, T = _scan(r, a, neff, ng, L, wl0, r2)
    dips, depths = _minima(wl, T)
    m = ring_metrics(r, a, ng, L, wl0, r2)
    # FSR: spacing of the two dips nearest wl0 (the identity is first order in dispersion)
    order = np.argsort(np.abs(dips - wl0))[:2]
    fsr_measured = abs(dips[order[0]] - dips[order[1]])
    assert abs(fsr_measured / m["FSR_m"] - 1.0) < 5e-3
    # FWHM and Q at the dip nearest wl0
    k = order[0]
    off = np.max(T)
    fwhm = _fwhm(wl, T, dips[k], depths[k], off)
    assert abs(fwhm / m["FWHM_m"] - 1.0) < 1e-2
    assert abs((wl0 / fwhm) / m["Q_loaded"] - 1.0) < 1e-2
    assert abs(depths[k] / m["T_pass_resonance"] - 1.0) < 1e-3
    # Q decomposition
    q_sum = 1.0 / m["Q_intrinsic"] + 1.0 / m["Q_coupling"]
    assert abs(q_sum * m["Q_loaded"] - 1.0) < 1e-2


def test_single_pass_amplitude():
    # 3 dB/cm over 1 cm is a power factor 0.5, amplitude 1/sqrt(2)
    assert abs(single_pass_amplitude(3.0103, 1e-2) - np.sqrt(0.5)) < 1e-4
    assert single_pass_amplitude(0.0, 1.0) == 1.0
