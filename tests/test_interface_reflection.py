"""The interface scattering matrix against exact references.

Transmission and reflection of an interface come from tangential continuity,
``E^a (I + R12) = E^b T12`` and ``H^a (I - R12) = H^b T12``, projected on one
section's modes (``SingleEME.INTERFACE_PROJECTION``).  In a *complete* basis
both projections are exact, so a model whose mode set is complete gives an
exact reference with nothing truncated:

* a Fresnel step between two plane waves (one mode each, complete);
* a discretised slab junction: ``M`` grid points, all ``M`` modes of each
  section including the evanescent ones, lossless and lossy (complex index,
  unconjugated biorthogonal normalisation, as in the PML datasets);
* a Fabry-Perot slab ``a | b | a`` cascaded with the production star product,
  against the field problem solved globally - no S-matrix convention involved.

Upstream's reflection block, ``1/2 (O_ab^T - O_ba) T12``, multiplied a
section-(k+1)-by-k matrix by another one; with the sign the T->S conversion
added to its (1,2) block, a lossless Fresnel step came out as
``[[t, r21], [-r12, t]]``, with ``|S^H S - I| = 2 t |r|`` (README, "What changed
relative to upstream").
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme import matrix_calculation_tool as mct  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME  # noqa: E402

M = 8                       # grid points = modes per section: a complete basis
H_GRID = 0.25e-6
K0 = 2 * np.pi / 1.55e-6
N_LOW, N_HIGH = 1.444, 3.48


# ------------------------------------------------------------------ models


def slab_modes(n):
    """All ``M`` modes of a discretised slab, ``(beta, E, H)`` with ``E``/``H``
    holding one mode per column and ``E^T H = I`` (unconjugated)."""
    d2 = (np.diag(-2.0 * np.ones(M)) + np.diag(np.ones(M - 1), 1) + np.diag(np.ones(M - 1), -1)) / H_GRID ** 2
    w, U = np.linalg.eig(d2 + np.diag((K0 * np.asarray(n, complex)) ** 2))
    order = np.argsort(-w.real)
    w, U = w[order], U[:, order]
    U = U / np.sqrt(np.sum(U * U, axis=0))[None, :]          # u_i^T u_i = 1 (complex symmetric)
    beta = np.sqrt(w)
    beta = np.where(beta.imag < 0, -beta, beta)              # decaying / forward
    return beta, U / np.sqrt(beta)[None, :], U * np.sqrt(beta)[None, :]


def exact_interface(Ea, Ha, Eb, Hb):
    """Field continuity solved directly: ``[[T12, R21], [R12, T21]]``."""
    def one_side(E1, H1, E2, H2):
        X, Y = np.linalg.solve(E1, E2), np.linalg.solve(H1, H2)
        T = 2 * np.linalg.inv(X + Y)
        return T, X @ T - np.eye(len(T))
    T12, R12 = one_side(Ea, Ha, Eb, Hb)
    T21, R21 = one_side(Eb, Hb, Ea, Ha)
    return np.block([[T12, R21], [R12, T21]])


def fresnel(n1, n2):
    """One plane-wave mode each side, normalised to unit power."""
    t = 2 * np.sqrt(n1 * n2) / (n1 + n2)
    r = (n1 - n2) / (n1 + n2)
    overlaps = np.array([[np.sqrt(n2 / n1)]]), np.array([[np.sqrt(n1 / n2)]])
    return overlaps, np.array([[t, -r], [r, t]])


PROFILE_A = [N_LOW] * 3 + [N_HIGH] * 2 + [N_LOW] * 3
PROFILE_B = [N_LOW] * 2 + [N_HIGH] * 4 + [N_LOW] * 2


def junction(lossy):
    loss = 0.02j if lossy else 0.0
    ba, Ea, Ha = slab_modes(np.asarray(PROFILE_A) + loss)
    bb, Eb, Hb = slab_modes(np.asarray(PROFILE_B) + loss)
    return (ba, Ea, Ha), (bb, Eb, Hb)


# ----------------------------------------------------------- the code path


def interface_smatrix(overlap_ab, overlap_ba, projection, route):
    """``SingleEME``'s interface scattering matrix for one interface."""
    s = object.__new__(SingleEME)
    s.overlap_forward_ab = np.asarray(overlap_ab, complex)[None]
    s.overlap_forward_ba = np.asarray(overlap_ba, complex)[None]
    s.section_count, s.mode_count = 2, overlap_ab.shape[0]
    s._lossless = projection == "input"
    s.INTERFACE_PROJECTION = projection
    s.INTERFACE_RCOND = None            # no truncation: the basis is complete
    s.INTERFACE_COLUMN_CAP = False
    if route == "direct":
        return s._calc_interface_Smatrix()[0]
    return mct._convert_3Dmatrix(s._calc_interface_Tmatrix())[0]


CASES = [(p, r) for p in ("input", "output") for r in ("direct", "transfer")]


# ------------------------------------------------------------------- tests


@pytest.mark.parametrize("projection,route", CASES)
def test_fresnel_step(projection, route):
    (oab, oba), exact = fresnel(N_LOW, N_HIGH)
    S = interface_smatrix(oab, oba, projection, route)
    assert np.allclose(S, exact, atol=1e-12)
    assert np.allclose(S.conj().T @ S, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("projection,route", CASES)
def test_complete_basis_junction(projection, route, lossy):
    (_, Ea, Ha), (_, Eb, Hb) = junction(lossy)
    S = interface_smatrix(Ea.T @ Hb, Eb.T @ Ha, projection, route)
    assert np.max(np.abs(S - exact_interface(Ea, Ha, Eb, Hb))) < 1e-9


@pytest.mark.parametrize("projection", ["input", "output"])
def test_reflection_is_gauge_covariant(projection):
    """A sign per mode per section moves ``S`` by ``D S D`` only, whatever the
    signs on the two sides - the reflection block included."""
    (_, Ea, Ha), (_, Eb, Hb) = junction(lossy=True)
    oab, oba = Ea.T @ Hb, Eb.T @ Ha
    rng = np.random.default_rng(0)
    da, db = rng.choice([-1.0, 1.0], M), rng.choice([-1.0, 1.0], M)
    S = interface_smatrix(oab, oba, projection, "direct")
    S_g = interface_smatrix(da[:, None] * oab * db[None, :], db[:, None] * oba * da[None, :], projection, "direct")
    D = np.concatenate([db, da])        # outputs [b2; b1]: section b, then section a
    Din = np.concatenate([da, db])      # inputs  [a1; a2]
    assert np.allclose(S_g, D[:, None] * S * Din[None, :], atol=1e-12)


def _propagation(beta, length):
    P = np.diag(np.exp(1j * beta * length))
    Z = np.zeros_like(P)
    return np.block([[P, Z], [Z, P]])


def _fabry_perot_field(a, b, length):
    """``a | b (length) | a`` solved as one linear system for every input mode:
    unknowns R (left, backward), F and B (in b, referred to z = 0), T (right)."""
    (_, Ea, Ha), (bb, Eb, Hb) = a, b
    P = np.diag(np.exp(1j * bb * length))
    Pi = np.diag(np.exp(-1j * bb * length))
    Z = np.zeros((M, M))
    A = np.block([[Ea, -Eb, -Eb, Z],
                  [-Ha, -Hb, Hb, Z],
                  [Z, Eb @ P, Eb @ Pi, -Ea],
                  [Z, Hb @ P, -Hb @ Pi, -Ha]])
    out = np.zeros((2 * M, M), complex)
    for j in range(M):
        e = np.zeros(M)
        e[j] = 1.0
        x = np.linalg.solve(A, np.concatenate([-Ea @ e, -Ha @ e, np.zeros(2 * M)]))
        out[:M, j], out[M:, j] = x[3 * M:], x[:M]
    return out


@pytest.mark.parametrize("projection", ["input", "output"])
@pytest.mark.parametrize("length", [0.30e-6, 0.35e-6, 0.50e-6])
def test_fabry_perot_cascade(projection, length):
    """The round trip between two interfaces is where a wrong reflection shows
    in the *transmission*: the production star product on the code's interface
    matrices must reproduce the global field solution."""
    a, b = junction(lossy=False)
    (_, Ea, Ha), (bb, Eb, Hb) = a, b
    S_ab = interface_smatrix(Ea.T @ Hb, Eb.T @ Ha, projection, "direct")
    S_ba = interface_smatrix(Eb.T @ Ha, Ea.T @ Hb, projection, "direct")
    S = mct._redheffer_star_product(mct._redheffer_star_product(S_ab, _propagation(bb, length)), S_ba)
    field = _fabry_perot_field(a, b, length)
    assert np.max(np.abs(S[:, :M] - field)) < 1e-8


def test_transfer_to_scattering_is_an_involution():
    """``[F2; B2] = M [F1; B1]`` -> ``[F2; B1] = S [F1; B2]`` maps S back to M."""
    (_, Ea, Ha), (_, Eb, Hb) = junction(lossy=True)
    S = exact_interface(Ea, Ha, Eb, Hb)
    assert np.allclose(mct._convert_2Dmatrix(mct._convert_2Dmatrix(S)), S, atol=1e-9)
    assert np.allclose(mct._convert_3Dmatrix(mct._convert_3Dmatrix(S[None]))[0], S, atol=1e-9)


def test_the_conversion_gives_the_interface_blocks():
    """Bienstman's transfer matrix of an exact interface converts to
    ``[[T12, R21], [R12, T21]]`` - for ``_convert_2Dmatrix`` (MultiPropagator's
    junctions) as for ``_convert_3Dmatrix``.  An involution alone would also
    admit ``n12 = -m12 inv(m22)`` with ``n21 = +inv(m22) m21``."""
    (_, Ea, Ha), (_, Eb, Hb) = junction(lossy=True)
    S = exact_interface(Ea, Ha, Eb, Hb)
    T12, R21, R12, T21 = S[:M, :M], S[:M, M:], S[M:, :M], S[M:, M:]
    iT = np.linalg.inv(T21)
    transfer = np.block([[T12 - R21 @ iT @ R12, R21 @ iT], [-iT @ R12, iT]])
    assert np.allclose(mct._convert_2Dmatrix(transfer), S, atol=1e-9)
    assert np.allclose(mct._convert_3Dmatrix(transfer[None])[0], S, atol=1e-9)
