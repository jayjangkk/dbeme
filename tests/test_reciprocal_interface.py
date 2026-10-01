"""Reciprocal projection and self-overlap correction of the interface matrix
(``SingleEME.INTERFACE_RECIPROCAL``, ``INTERFACE_SELF_OVERLAP``).

Lorentz reciprocity in this code's conventions - unconjugated normalisation
``1/2 int e_i x h_j = delta_ij``, backward partner ``(e_t, -h_t)``, interface
ordering ``[b2; b1] = S [a1; a2]`` - makes the channel-ordered matrix
``[[R12, T21], [T12, R21]]`` symmetric, and a one-sign-per-mode gauge acts on
it by congruence.  The production transmission blocks satisfy ``T21 = T12^T``
on any basis; the reflection blocks are symmetric only if each section's
self-overlap ``M`` is ``I``.  Biorthogonalisation forces only ``sym(M) = I``,
and with ``M = I + A`` an interface between a section and itself gives
``T = I - A^2`` (output side) and ``R = -A (I - A^2)``.

The models: the discretised slab of ``tests/test_interface_reflection.py``
(complete, so pointwise continuity is the exact reference), its truncation,
and the same slab with every overlap taken in a perturbed quadrature
``W = I + eps K`` and biorthogonalised as ``assemble.biorthogonalise`` does,
which leaves ``M = I + A`` with ``A != 0`` - a defect of the same form as
the FEM basis's (whose ``A`` may come from the overlap window rather than
from quadrature).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme import matrix_calculation_tool as mct  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME, cap_columns  # noqa: E402

K0 = 2 * np.pi / 1.55e-6
N_LOW, N_HIGH = 1.444, 3.48
GRID, SPAN = 60, 3.0e-6
LOSS = 0.02j


# ------------------------------------------------------------------ models


def slab_modes(n, h):
    """All modes of a discretised slab: ``(beta, E, H)``, one mode per column,
    ``E^T H = I`` (unconjugated)."""
    n = np.asarray(n, complex)
    m = len(n)
    d2 = (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / h ** 2
    w, U = np.linalg.eig(d2 + np.diag((K0 * n) ** 2))
    order = np.argsort(-w.real)
    w, U = w[order], U[:, order]
    U = U / np.sqrt(np.sum(U * U, axis=0))[None, :]
    beta = np.sqrt(w)
    beta = np.where(beta.imag < 0, -beta, beta)
    return beta, U / np.sqrt(beta)[None, :], U * np.sqrt(beta)[None, :]


def slab(width, loss=0.0):
    """A core of ``width`` in a ``SPAN`` window, edge cells averaged by fill
    fraction so that widths need not sit on the grid."""
    h = SPAN / GRID
    edges = np.linspace(-SPAN / 2, SPAN / 2, GRID + 1)
    fill = np.clip((np.minimum(edges[1:], width / 2) - np.maximum(edges[:-1], -width / 2)) / h, 0, 1)
    n = np.sqrt((fill * N_HIGH ** 2 + (1 - fill) * N_LOW ** 2).astype(complex)) + loss
    return slab_modes(n, h)


def exact_interface(Ea, Ha, Eb, Hb):
    """Field continuity solved pointwise on a complete basis."""
    def one_side(E1, H1, E2, H2):
        X, Y = np.linalg.solve(E1, E2), np.linalg.solve(H1, H2)
        T = 2 * np.linalg.inv(X + Y)
        return T, X @ T - np.eye(len(T))
    T12, R12 = one_side(Ea, Ha, Eb, Hb)
    T21, R21 = one_side(Eb, Hb, Ea, Ha)
    return np.block([[T12, R21], [R12, T21]])


PROFILE_A = [N_LOW] * 3 + [N_HIGH] * 2 + [N_LOW] * 3      # tests/test_interface_reflection.py
PROFILE_B = [N_LOW] * 2 + [N_HIGH] * 4 + [N_LOW] * 2


def complete_junction(lossy):
    loss = LOSS if lossy else 0.0
    _, Ea, Ha = slab_modes(np.asarray(PROFILE_A) + loss, 0.25e-6)
    _, Eb, Hb = slab_modes(np.asarray(PROFILE_B) + loss, 0.25e-6)
    return Ea, Ha, Eb, Hb


def truncated_overlaps(lossy, n_modes, widths=(0.50e-6, 0.52e-6)):
    """Consecutive-section overlaps of a slab path, first ``n_modes`` of each."""
    loss = LOSS if lossy else 0.0
    modes = [slab(w, loss) for w in widths]
    oab = np.array([(a[1].T @ b[2])[:n_modes, :n_modes] for a, b in zip(modes[:-1], modes[1:])])
    oba = np.array([(b[1].T @ a[2])[:n_modes, :n_modes] for a, b in zip(modes[:-1], modes[1:])])
    beta = np.array([m[0][:n_modes] for m in modes[:-1]])
    return oab, oba, beta


def antisymmetric(n, size, complex_, seed):
    """A random antisymmetric matrix of spectral norm ``size``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if complex_ else 0)
    A = X - X.T
    return A * (size / np.linalg.norm(A, 2))


def perturbed_quadrature(eps, dw=2e-9, seed=3):
    """The lossy slab junction 0.50 um -> 0.50 um + ``dw``, complete, with every
    overlap taken in ``W = I + eps K`` and each section biorthogonalised in it
    (symmetric part of ``E^T W H`` -> ``I``): returns the two overlap sets, the
    exact interface of the fields so mixed, and the self-overlaps ``M_a``,
    ``M_b``."""
    rng = np.random.default_rng(seed)
    K = np.diag(rng.standard_normal(GRID)) + 0.5 * (np.diag(rng.standard_normal(GRID - 1), 1)
                                                   + np.diag(rng.standard_normal(GRID - 1), -1))
    W = np.eye(GRID) + eps * K

    def loewdin(E, H):
        S = 0.5 * (E.T @ W @ H + (E.T @ W @ H).T)
        w, V = np.linalg.eig(S)
        L = V @ np.diag(w ** -0.5) @ np.linalg.inv(V)
        return E @ L, H @ L

    _, Ea, Ha = slab(0.50e-6, LOSS)
    _, Eb, Hb = slab(0.50e-6 + dw, LOSS)
    Ea, Ha = loewdin(Ea, Ha)
    Eb, Hb = loewdin(Eb, Hb)
    return Ea.T @ W @ Hb, Eb.T @ W @ Ha, exact_interface(Ea, Ha, Eb, Hb), Ea.T @ W @ Ha, Eb.T @ W @ Hb


# ----------------------------------------------------------- the code path


def stub(oab, oba, lossless, reciprocal="auto", self_overlap=False, projection="auto", rcond=1e-2,
         beta=None, dz=None):
    s = object.__new__(SingleEME)
    s.overlap_forward_ab = np.asarray(oab, complex).reshape((-1,) + np.shape(oab)[-2:])
    s.overlap_forward_ba = np.asarray(oba, complex).reshape((-1,) + np.shape(oba)[-2:])
    s.section_count = s.overlap_forward_ab.shape[0] + 1
    s.mode_count = s.overlap_forward_ab.shape[1]
    s._lossless = lossless
    s._force_unitary = s._force_passive = False
    s.INTERFACE_PROJECTION = projection
    s.INTERFACE_RCOND = rcond
    s.INTERFACE_COLUMN_CAP = False
    s.INTERFACE_RECIPROCAL = reciprocal
    s.INTERFACE_SELF_OVERLAP = self_overlap
    if beta is not None:
        s.beta_forward = np.asarray(beta)
        s.output_data = {"EME_delta_zs": np.asarray(dz, float)}
    return s


def interface(s, route="direct"):
    if route == "direct":
        return s._calc_interface_Smatrix()
    return mct._convert_3Dmatrix(s._calc_interface_Tmatrix())


def channel(S):
    """``[b2; b1] = S [a1; a2]`` -> ``[b1; b2] = S_ch [a1; a2]``."""
    n = S.shape[-1] // 2
    return np.concatenate([S[..., n:, :], S[..., :n, :]], axis=-2)


def nonreciprocity(S):
    C = channel(S)
    return np.abs(C - np.swapaxes(C, -1, -2)).max() / np.abs(S).max()


def blocks(S):
    n = S.shape[-1] // 2
    return S[..., :n, :n], S[..., :n, n:], S[..., n:, :n], S[..., n:, n:]   # T12, R21, R12, T21


def column_power(S):
    return np.sum(np.abs(S) ** 2, axis=-2)


# ------------------------------------------------------------ the switches


def test_the_defaults():
    assert SingleEME.INTERFACE_RECIPROCAL == "auto"
    assert SingleEME.INTERFACE_SELF_OVERLAP is False


@pytest.mark.parametrize("switch", [{"reciprocal": "sometimes"}, {"self_overlap": True},
                                    {"self_overlap": "exact"}])
def test_unknown_values_are_rejected(switch):
    oab, oba, _ = truncated_overlaps(True, 4)
    with pytest.raises(ValueError):
        interface(stub(oab, oba, lossless=False, **switch))


def test_auto_leaves_a_lossless_basis_untouched():
    """``"auto"`` is bit-for-bit production on a lossless basis, although its
    truncated reflection is not symmetric (forcing the projection does change
    it)."""
    oab, oba, _ = truncated_overlaps(False, 8)
    for route, rcond in (("direct", None), ("direct", 1e-2), ("transfer", None)):
        auto = interface(stub(oab, oba, lossless=True, rcond=rcond), route)
        off = interface(stub(oab, oba, lossless=True, reciprocal=False, rcond=rcond), route)
        on = interface(stub(oab, oba, lossless=True, reciprocal=True, rcond=rcond), route)
        assert np.array_equal(auto, off)
        assert np.abs(on - off).max() > 1e-8


def test_auto_acts_on_a_lossy_basis():
    oab, oba, _ = truncated_overlaps(True, 12)
    auto = interface(stub(oab, oba, lossless=False))
    on = interface(stub(oab, oba, lossless=False, reciprocal=True))
    off = interface(stub(oab, oba, lossless=False, reciprocal=False))
    assert np.array_equal(auto, on)
    assert np.abs(auto - off).max() > 1e-8


# ------------------------------------------------- exactness and invariance


@pytest.mark.parametrize("lossy", [False, True])
@pytest.mark.parametrize("projection", ["input", "output"])
@pytest.mark.parametrize("route", ["direct", "transfer"])
def test_complete_basis_exactness_is_unchanged(route, projection, lossy):
    """On a complete basis with ``M = I`` the production blocks are exact and
    reciprocal already: neither treatment changes them beyond round-off."""
    Ea, Ha, Eb, Hb = complete_junction(lossy)
    oab, oba = Ea.T @ Hb, Eb.T @ Ha
    exact = exact_interface(Ea, Ha, Eb, Hb)
    off = interface(stub(oab, oba, lossless=not lossy, reciprocal=False, projection=projection, rcond=None),
                    route)[0]
    for switches in ({"reciprocal": True}, {"reciprocal": True, "self_overlap": "estimate"},
                     {"reciprocal": False, "self_overlap": "estimate"}):
        S = interface(stub(oab, oba, lossless=not lossy, projection=projection, rcond=None, **switches), route)[0]
        assert np.abs(S - exact).max() < 1e-9
        assert np.abs(S - off).max() < 1e-11


@pytest.mark.parametrize("self_overlap", [False, "estimate"])
@pytest.mark.parametrize("projection", ["input", "output"])
def test_the_treatments_are_gauge_covariant(projection, self_overlap):
    """A sign per mode per section moves ``S`` by ``D S D``, on a truncated
    lossy basis where the treatments act."""
    oab, oba, _ = truncated_overlaps(True, 12)
    oab, oba = oab[0], oba[0]
    rng = np.random.default_rng(0)
    da, db = rng.choice([-1.0, 1.0], 12), rng.choice([-1.0, 1.0], 12)
    S = interface(stub(oab, oba, lossless=False, self_overlap=self_overlap, projection=projection))[0]
    Sg = interface(stub(da[:, None] * oab * db[None, :], db[:, None] * oba * da[None, :], lossless=False,
                        self_overlap=self_overlap, projection=projection))[0]
    Dout, Din = np.concatenate([db, da]), np.concatenate([da, db])
    assert np.abs(Sg - Dout[:, None] * S * Din[None, :]).max() < 1e-12


@pytest.mark.parametrize("projection", ["input", "output"])
def test_production_transmission_is_already_reciprocal(projection):
    """``T21 = T12^T`` holds for the production formulas on a truncated basis,
    so the projection's transmission average is a no-op there."""
    oab, oba, _ = truncated_overlaps(True, 12)
    off = interface(stub(oab, oba, lossless=False, reciprocal=False, projection=projection))[0]
    on = interface(stub(oab, oba, lossless=False, projection=projection))[0]
    T12, _, _, T21 = blocks(off)
    assert np.abs(T21 - T12.T).max() < 1e-12 * np.abs(off).max()
    assert np.abs(blocks(on)[0] - T12).max() < 1e-12 * np.abs(off).max()


@pytest.mark.parametrize("self_overlap", [False, "estimate"])
@pytest.mark.parametrize("projection", ["auto", "input"])
def test_a_truncated_lossy_cascade_is_reciprocal(projection, self_overlap):
    """Five interfaces of a lossy slab taper, 12 of 60 modes kept, cascaded with
    the production star product: reciprocal to round-off with the projection,
    and measurably not without it."""
    widths = np.linspace(0.40e-6, 0.60e-6, 6)
    oab, oba, beta = truncated_overlaps(True, 12, widths)
    dz = np.full(len(oab), 0.3e-6)

    def lumped(reciprocal):
        s = stub(oab, oba, lossless=False, reciprocal=reciprocal, self_overlap=self_overlap,
                 projection=projection, beta=beta, dz=dz)
        total = s._calc_Smatrix_direct()
        L = total[0]
        for X in total[1:]:
            L = mct._redheffer_star_product(L, X)
        return L, total

    L, total = lumped("auto")
    assert max(nonreciprocity(X) for X in total) < 1e-13
    assert nonreciprocity(L) < 1e-12
    assert nonreciprocity(lumped(False)[0]) > 1e-6


# ------------------------------------------------ the self-overlap defect


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("projection", ["auto", "input", "output"])
def test_a_zero_step_reflects_nothing(projection, complex_):
    """``O_ab = O_ba = M = I + A``: an interface between a section and itself.
    Production reflects ``-A (I - A^2)`` (output side) or ``-A`` (input side),
    both antisymmetric, so the reciprocal part is exactly zero."""
    M = np.eye(12) + antisymmetric(12, 0.2, complex_, seed=5)
    T12, R21, R12, T21 = blocks(interface(stub(M, M, lossless=False, projection=projection))[0])
    assert np.abs(R12).max() < 1e-12 and np.abs(R21).max() < 1e-12
    production = interface(stub(M, M, lossless=False, reciprocal=False, projection=projection))[0]
    assert np.abs(blocks(production)[2]).max() > 1e-2


@pytest.mark.parametrize("complex_", [False, True])
def test_the_estimate_removes_the_zero_step_floor(complex_):
    """The output-side projection transmits ``I - A^2`` at a zero step, which
    the reciprocal projection leaves alone (for real ``A`` it is a gain of
    ``2 |A e_j|^2`` per column).  With ``M`` estimated from the reflection
    asymmetry the step is the identity to ``O(A^3)``."""
    A = antisymmetric(12, 0.2, complex_, seed=7)
    M = np.eye(12) + A
    reciprocal = interface(stub(M, M, lossless=False))[0]
    estimate = interface(stub(M, M, lossless=False, self_overlap="estimate"))[0]
    floor = np.abs(reciprocal - np.eye(24)).max()
    assert floor > 0.2 * np.abs(A @ A).max()
    assert np.abs(estimate - np.eye(24)).max() < 0.05 * floor
    if not complex_:
        assert column_power(reciprocal).max() > 1 + np.linalg.norm(A, axis=0).max() ** 2
        assert column_power(estimate).max() < 1 + 1e-3


@pytest.mark.parametrize("projection", ["input", "output"])
def test_the_estimate_corrects_a_quadrature_error(projection):
    """A physical 2 nm step of the complete lossy slab, every overlap taken in a
    perturbed quadrature (median self-overlap column error 0.09; the FEM
    basis has 0.05).  The production blocks miss the exact interface by the
    zero-step artefacts; the projection removes the reflection part and the
    estimate the transmission part as well."""
    oab, oba, exact, Ma, _ = perturbed_quadrature(eps=0.1)
    A = 0.5 * (Ma - Ma.T)
    assert 0.05 < np.median(np.linalg.norm(A, axis=0)) < 0.15

    def error(**switches):
        S = interface(stub(oab, oba, lossless=False, projection=projection, rcond=None, **switches))[0]
        return np.abs(S - exact).max()

    production, reciprocal, estimate = error(reciprocal=False), error(), error(self_overlap="estimate")
    assert production > 1e-2
    assert estimate < 0.1 * production
    assert estimate < reciprocal


def test_the_estimate_refuses_a_large_asymmetry():
    """Beyond :attr:`SingleEME.SELF_OVERLAP_ESTIMATE_LIMIT` the reflection
    asymmetry is not a small self-overlap error (on report 22's FD-PML bases
    degenerate continuum pairs reach 6.6), and the estimate raises."""
    M = np.eye(12) + antisymmetric(12, 0.6, False, seed=11)
    with pytest.raises(ValueError, match="does not hold"):
        interface(stub(M, M, lossless=False, self_overlap="estimate"))
    small = np.eye(12) + antisymmetric(12, 0.2, False, seed=11)
    interface(stub(small, small, lossless=False, self_overlap="estimate"))


@pytest.mark.parametrize("projection", ["input", "output"])
def test_the_corrected_equations_are_exact_with_the_true_self_overlap(projection):
    """Fed the true ``M_a``, ``M_b`` of the perturbed-quadrature junction, the
    Galerkin equations reproduce the exact interface: the transforms are right
    for all four blocks, on either projection side."""
    oab, oba, exact, Ma, Mb = perturbed_quadrature(eps=0.1)
    A_a, A_b = 0.5 * (Ma - Ma.T), 0.5 * (Mb - Mb.T)
    s = stub(oab, oba, lossless=False, reciprocal=False, projection=projection, rcond=None)
    # the method takes the uncorrected reflection blocks, whose antisymmetric
    # part it reads as -A; hand it -A directly
    T12, T21, R12, R21 = (x[0] for x in s._self_overlap_corrected_blocks(-A_a[None], -A_b[None], None))
    assert np.abs(np.block([[T12, R21], [R12, T21]]) - exact).max() < 1e-12


def test_the_estimate_resolves_auto_like_the_interface():
    """On a lossy basis ``"auto"`` means the output side, in the correction as
    in the uncorrected blocks."""
    oab, oba, _, _, _ = perturbed_quadrature(eps=0.1)
    auto = interface(stub(oab, oba, lossless=False, self_overlap="estimate", rcond=None))
    output = interface(stub(oab, oba, lossless=False, self_overlap="estimate", projection="output", rcond=None))
    inp = interface(stub(oab, oba, lossless=False, self_overlap="estimate", projection="input", rcond=None))
    assert np.abs(auto - output).max() < 1e-14
    assert np.abs(auto - inp).max() > 1e-8


def test_the_estimate_checks_both_sections():
    """The limit applies to the estimate from either reflection block: here only
    ``A_b`` (from ``R21``) is out of range."""
    Ea, Ha, Eb, Hb = complete_junction(True)
    n = Ea.shape[1]
    oab = Ea.T @ Hb @ (np.eye(n) + antisymmetric(n, 0.72, False, seed=12))
    oba = Eb.T @ Ha
    s = stub(oab, oba, lossless=False, reciprocal=False, projection="input", rcond=None)
    _, R21, R12, _ = blocks(interface(s)[0])
    norm = lambda R: np.linalg.norm(0.5 * (R - R.T), 2)  # noqa: E731
    assert norm(R12) < SingleEME.SELF_OVERLAP_ESTIMATE_LIMIT < norm(R21)
    with pytest.raises(ValueError, match="does not hold"):
        interface(stub(oab, oba, lossless=False, reciprocal=False, self_overlap="estimate",
                       projection="input", rcond=None))


@pytest.mark.parametrize("lossless", [True, False])
def test_the_transfer_route_keeps_no_cutoff(lossless):
    """The transfer route inverts ``T21`` and was validated without the
    pseudo-inverse cutoff; it must still ignore :attr:`INTERFACE_RCOND` now
    that it builds its blocks through ``_calc_interface_blocks``."""
    rng = np.random.default_rng(4)
    U, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    O = 0.5 * U @ np.diag([1.0, 1.0, 0.9, 0.8, 0.5, 3e-3]) @ U.T
    oab = O + 1e-4 * rng.standard_normal((6, 6))
    oba = O.T
    s = stub(oab, oba, lossless=lossless, rcond=1e-2)
    with_cutoff = [np.asarray(x) for x in s._calc_interface_blocks(1e-2)]
    without = [np.asarray(x) for x in s._calc_interface_blocks(None)]
    assert max(np.abs(x - y).max() for x, y in zip(with_cutoff, without)) > 1e-5    # the cutoff acts here
    transfer = interface(s, "transfer")
    assert np.array_equal(transfer, interface(stub(oab, oba, lossless=lossless, rcond=None), "transfer"))


def test_the_cap_acts_after_the_projection():
    """With :attr:`INTERFACE_COLUMN_CAP` opted in, the reciprocal blocks are
    capped: a zero step with real ``A`` has columns above 1 (``I - A^2``)."""
    M = np.eye(12) + antisymmetric(12, 0.2, False, seed=7)
    s = stub(M, M, lossless=False)
    uncapped = interface(s)
    s.INTERFACE_COLUMN_CAP = True
    capped = interface(s)
    assert column_power(uncapped).max() > 1 + 1e-3
    assert column_power(capped).max() <= 1 + 1e-12
    assert np.abs(capped - cap_columns(uncapped)).max() < 1e-15
