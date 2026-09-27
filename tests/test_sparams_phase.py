"""Invariants of `em_simulation/propagator/sparams_phase` - continuous phase and delay.

Each test encodes one of the three traps the module exists for, or one of the
contracts the S-parameter scripts rely on without checking at run time.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from em_simulation.propagator.sparams_phase import (  # noqa: E402
    C_LIGHT,
    aligned_gradient,
    cascade,
    dominant_delay,
    gauge_chain,
    match,
    port_signs,
    propagate_into,
    reconstruct,
    reorder_interface,
    sign_interface,
    smoothstep,
    star,
)


def test_match_recovers_a_shuffle():
    rng = np.random.default_rng(1)
    n = np.array([1.72, 1.66, 1.61, 1.55, 1.50])
    te = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    shuffle = rng.permutation(len(n))
    q = match(n, te, n[shuffle] + 1e-4, te[shuffle])
    assert np.array_equal(shuffle[q], np.arange(len(n)))


def test_gradient_follows_modes_through_slot_reordering():
    """Slots shuffled at every wavelength and section; the derivative must not care.

    Differencing slots instead of modes is the SiSNPR failure: 65 % of slots
    move across the band there.
    """
    rng = np.random.default_rng(7)
    lams = np.linspace(1260e-9, 1360e-9, 11)
    points, modes = 4, 5
    offset = np.array([1.72, 1.66, 1.60, 1.54, 1.48])      # 60 mdn apart
    slope = np.array([-2.0e5, -2.6e5, -3.1e5, -1.5e5, -2.9e5])  # <= 3e-3 / 10 nm
    te_mode = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    stack = np.empty((len(lams), points, modes))
    te = np.empty_like(stack)
    truth = np.empty_like(stack)
    for i, lam in enumerate(lams):
        for k in range(points):
            order = rng.permutation(modes)
            stack[i, k] = offset[order] + slope[order] * (lam - lams[0])
            te[i, k] = te_mode[order]
            truth[i, k] = slope[order]
    dn, stats = aligned_gradient(stack, te, lams)
    assert np.allclose(dn, truth, rtol=0, atol=1e-6 * np.abs(slope).max())
    assert stats["slots_reordered_fraction"] > 0.5
    assert stats["max_step_aligned"] < stats["max_step_unaligned"]


def test_reorder_interface_is_a_basis_relabelling():
    rng = np.random.default_rng(3)
    n = 4
    m = rng.normal(size=(2 * n, 2 * n)) + 1j * rng.normal(size=(2 * n, 2 * n))
    ql, qr = rng.permutation(n), rng.permutation(n)
    out = reorder_interface(m, ql, qr)
    for a in range(n):
        for b in range(n):
            assert out[a, b] == m[qr[a], ql[b]]                  # right fwd <- left fwd
            assert out[n + a, b] == m[n + ql[a], ql[b]]          # left bwd  <- left fwd
            assert out[a, n + b] == m[qr[a], n + qr[b]]          # right fwd <- right bwd
            assert out[n + a, n + b] == m[n + ql[a], n + qr[b]]  # left bwd  <- right bwd


def _aliased_device(nm, length=580e-6, n_eff=1.70, dn_dlam=-2.5e5):
    lam = np.asarray(nm, dtype=float) * 1e-9
    n = n_eff + dn_dlam * (lam - 1.31e-6)
    phi = 2.0 * np.pi / lam * n * length
    residual = 0.3 + 0.04 * (lam - 1.31e-6) * 1e9 / 10.0
    return lam, n, phi, residual


def test_reconstruct_recovers_an_aliased_phase_and_its_delay():
    nm = np.arange(1260, 1361, 10)
    lam, n, phi, residual = _aliased_device(nm)
    s = 0.97 * np.exp(1j * (phi + residual))
    assert np.abs(np.diff(phi)).min() > 2 * np.pi, "the test must actually alias"
    got = reconstruct(nm, s, phi)
    assert got["safe"] and got["gauge_flips"] == 0
    assert got["equals_arg_S_to_rad"] < 1e-9
    assert np.allclose(got["phase_rad"], phi + residual, atol=1e-9)
    ng = n - lam * (-2.5e5)
    tau_prop = ng * 580e-6 / C_LIGHT * 1e12
    assert np.allclose(got["tau_from_phase_ps"], tau_prop, rtol=2e-3)


def test_reconstruct_removes_gauge_sign_flips():
    """The SNRAC TE1 failure: same light, the solver's port sign flips."""
    nm = np.arange(1260, 1361, 10)
    lam, n, phi, residual = _aliased_device(nm)
    flips = np.array([1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1], dtype=float)
    s = 0.95 * flips * np.exp(1j * (phi + residual))
    got = reconstruct(nm, s, phi)
    assert got["gauge_flips"] == 4
    assert np.array_equal(got["gauge_signs"], flips * flips[0])
    assert got["safe"]
    assert np.allclose(got["phase_rad"], phi + residual, atol=1e-9)


def test_reconstruct_refuses_a_residual_that_drifts_too_fast():
    nm = np.arange(1260, 1361, 10)
    lam, n, phi, _ = _aliased_device(nm)
    drifting = np.exp(1j * (phi + 1.2 * np.arange(len(nm))))
    assert not reconstruct(nm, drifting, phi)["safe"]


def test_gauge_chain_recovers_sign_flips_along_the_light_path():
    """Signs flipped at random points at wavelength b; the chain finds them."""
    rng = np.random.default_rng(11)
    points, modes = 7, 3
    chain = np.array([0, 0, 1, 1, 1, 2, 2])          # the light changes slot twice
    interfaces_a = []
    for k in range(points - 1):
        m = 0.05 * (rng.normal(size=(2 * modes, 2 * modes))
                    + 1j * rng.normal(size=(2 * modes, 2 * modes)))
        m[chain[k + 1], chain[k]] = 0.9 * np.exp(1j * rng.uniform(-0.3, 0.3))
        interfaces_a.append(m)
    truth = rng.choice([-1.0, 1.0], size=(points, modes))
    truth[0] = 1.0                                   # the chain's reference
    interfaces_b = [sign_interface(interfaces_a[k], truth[k], truth[k + 1])
                    for k in range(points - 1)]
    v, port_sign, untrusted = gauge_chain(interfaces_a, interfaces_b, chain)
    assert untrusted == 0
    assert np.array_equal(v[np.arange(points), chain],
                          truth[np.arange(points), chain])
    assert port_sign == truth[0, chain[0]] * truth[-1, chain[-1]]
    realigned = [sign_interface(interfaces_b[k], v[k], v[k + 1])
                 for k in range(points - 1)]
    for k in range(points - 1):
        assert realigned[k][chain[k + 1], chain[k]] == pytest.approx(
            interfaces_a[k][chain[k + 1], chain[k]])


@pytest.mark.parametrize("branch", [0, 2])
def test_dominant_delay_is_the_single_branch_sum_when_power_stays_put(branch):
    rng = np.random.default_rng(branch)
    sections, modes, lam = 30, 4, 1.31e-6
    neff = 1.5 + 0.2 * rng.random((sections + 1, modes))
    dn = -2e5 * rng.random((sections + 1, modes))
    dz = 1e-6 + 1e-6 * rng.random(sections)
    weights = np.full((2 * sections, modes), 1e-3)
    weights[:, branch] = 0.99
    phase, tau, ng, ref = dominant_delay(neff, dz, weights, lam, dn)
    assert np.all(ref == branch)
    assert phase == pytest.approx(np.sum(2 * np.pi / lam * neff[:sections, branch] * dz))
    expect = np.sum((neff[:sections, branch] - lam * dn[:sections, branch]) * dz) / C_LIGHT
    assert tau == pytest.approx(expect)


def _random_smatrix(rng, n, scale=0.3):
    return scale * (rng.normal(size=(2 * n, 2 * n)) + 1j * rng.normal(size=(2 * n, 2 * n)))


def test_star_is_the_reference_product_batched():
    """The frequency-grid cascade must be the propagator's own algebra."""
    from em_simulation.matrix_calculation_tool import _redheffer_star_product

    rng = np.random.default_rng(5)
    n = 3
    a = np.stack([_random_smatrix(rng, n) for _ in range(4)])
    b = np.stack([_random_smatrix(rng, n) for _ in range(4)])
    got = star(a, b)
    for f in range(4):
        assert np.allclose(got[f], _redheffer_star_product(a[f], b[f]), rtol=0, atol=1e-12)
    assert np.allclose(star(a[0], b)[2], _redheffer_star_product(a[0], b[2]),
                       rtol=0, atol=1e-12)


def test_propagate_into_is_propagation_starred_with_the_interface():
    from em_simulation.matrix_calculation_tool import _redheffer_star_product

    rng = np.random.default_rng(9)
    n = 4
    interface = _random_smatrix(rng, n)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(3, n)))
    got = propagate_into(interface, phase)
    for f in range(3):
        prop = np.zeros((2 * n, 2 * n), dtype=complex)
        prop[:n, :n] = np.diag(phase[f])
        prop[n:, n:] = np.diag(phase[f])
        assert np.allclose(got[f], _redheffer_star_product(prop, interface),
                           rtol=0, atol=1e-12)


def test_cascade_is_the_propagators_section_order():
    """``prop_0, interface_0, prop_1, ...`` - `_find_Smatrix_new_length`'s order."""
    from em_simulation.matrix_calculation_tool import _redheffer_star_product

    rng = np.random.default_rng(13)
    n, sections, freqs = 3, 6, 2
    interfaces = np.stack([_random_smatrix(rng, n, 0.2) + np.eye(2 * n)
                           for _ in range(sections)])
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(freqs, sections, n)))
    got = cascade(interfaces, phases)
    for f in range(freqs):
        lumped = None
        for k in range(sections):
            prop = np.zeros((2 * n, 2 * n), dtype=complex)
            prop[:n, :n] = np.diag(phases[f, k])
            prop[n:, n:] = np.diag(phases[f, k])
            lumped = prop if lumped is None else _redheffer_star_product(lumped, prop)
            lumped = _redheffer_star_product(lumped, interfaces[k])
        assert np.allclose(got[f], lumped, rtol=0, atol=1e-10)


def test_smoothstep_is_flat_at_both_ends():
    assert np.allclose(smoothstep([-0.5, 0.0, 0.5, 1.0, 1.5]), [0.0, 0.0, 0.5, 1.0, 1.0])
    assert smoothstep(1e-6) < 1e-11
    assert 1.0 - smoothstep(1.0 - 1e-6) < 1e-11


def test_port_signs_recovers_row_and_column_flips():
    """Two snapshots of one device: port-mode signs flipped, plus a small real change.

    Rows 1 and 3 hold only weak entries, as a TM output row does on the SiSNPRS.
    """
    rng = np.random.default_rng(17)
    freqs = 40
    magnitude = np.array([[1.0, 1e-2], [1e-3, 3e-2], [2e-2, 1.0], [1e-3, 5e-3]])
    reference = magnitude * np.exp(1j * rng.uniform(-np.pi, np.pi, size=(freqs, 4, 2)))
    row_flip = np.array([1.0, -1.0, 1.0, -1.0])
    col_flip = np.array([-1.0, -1.0])
    drift = np.exp(1j * rng.normal(scale=0.05, size=reference.shape))
    other = reference * drift * row_flip[None, :, None] * col_flip[None, None, :]
    rows, cols, score = port_signs(reference, other)
    assert cols[0] == 1.0 and score > 0
    assert np.array_equal(rows[:, None] * cols[None, :],
                          row_flip[:, None] * col_flip[None, :])
