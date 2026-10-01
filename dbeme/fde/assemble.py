r"""Turn raw solver output into the arrays the EME solver consumes.

This is the numerical core that used to sit inside ``DataUpdater``'s
Lumerical sweep post-processing.  It is backend independent: give it a list of
:class:`~dbeme.fde.base.ModeData` sampled on one common grid and it
produces the normalised bidirectional mode basis and the overlap matrices that
the dataset stores.

Conventions (unchanged from the Lumerical-based implementation)
---------------------------------------------------------------
* ``N = num_modes`` forward modes are extended to ``2N`` by appending the
  backward-propagating partners.  For a reciprocal medium the backward mode is
  :math:`(E_t, -H_t) \\to (E^*, -H^*)` with :math:`n_{eff} \\to -n_{eff}`.
* Each forward mode is normalised so that
  :math:`\\tfrac12 \\int (E \\times H)_z \\, dA = 1`.
* The overlap between parameter points ``a`` and ``b`` is
  :math:`O_{ij} = \\tfrac12 \\int (E_{a,i} \\times H_{b,j})_z \\, dA`.
"""

import numpy as np
from scipy.linalg import sqrtm

from ..data_updater import overlap_calculation_tool as oct


def assemble(mode_data_list, num_modes, prop_axis=2, lossless=True,
             biorthogonal=True):
    """Build the ``2N`` bidirectional mode basis for a set of parameter points.

    :param mode_data_list: One :class:`ModeData` per parameter point, all
        sharing the same grid.
    :param num_modes: Number of forward modes ``N``.
    :param prop_axis: Index of the propagation component (2 = z).
    :param lossless: Whether the medium is lossless by construction - take it
        from :attr:`~dbeme.fde.base.FDEBackend.lossless`, never from
        ``Im(n_eff)``.  It selects how the backward basis is built and whether
        a negative ``Im(n_eff)`` is treated as round-off.
    :param biorthogonal: Enforce ``½(M + Mᵀ) = I`` within each point's forward
        modes before mirroring them backwards - see
        :func:`biorthogonalise`.  Leave it on; the switch exists to reproduce
        pre-fix results, not because it is ever the better choice.
    :returns: ``(x, y, neff, TE_pol, E, H)`` where ``neff`` and ``TE_pol`` have
        shape ``(n_points, 2N)`` and the fields ``(n_points, 2N, 3, nx, ny)``.
    """
    n_pts = len(mode_data_list)
    x, y = mode_data_list[0].x, mode_data_list[0].y
    nx, ny = len(x), len(y)
    N = int(num_modes)

    neff = np.zeros((n_pts, 2 * N), dtype=np.complex64)
    TE_pol = np.zeros((n_pts, 2 * N), dtype=np.float32)
    E = np.zeros((n_pts, 2 * N, 3, nx, ny), dtype=np.complex64)
    H = np.zeros((n_pts, 2 * N, 3, nx, ny), dtype=np.complex64)

    for p, md in enumerate(mode_data_list):
        # A mode solver can return a slightly negative imaginary part from
        # round-off; that would show up as gain, so it is clipped to zero.
        # In a lossy model that clip is not a guard, it is data loss - the
        # sign of Im(n_eff) is the loss - so it is only applied when the
        # backend says the medium is lossless (5.16b).
        neff[p, :N] = (
            oct.correct_gain_modified(md.neff[:N]) if lossless else md.neff[:N]
        )
        TE_pol[p, :N] = md.TE_pol[:N]
        E[p, :N] = md.E[:N]
        H[p, :N] = md.H[:N]

    # Normalise the forward modes to unit power, then mirror them backwards.
    E[:, :N], H[:, :N] = oct.normalize_field(E[:, :N], H[:, :N], x, y, prop_axis)

    if biorthogonal:
        E[:, :N], H[:, :N] = biorthogonalise(E[:, :N], H[:, :N], x, y, prop_axis)

    neff[:, N:] = -neff[:, :N]
    TE_pol[:, N:] = TE_pol[:, :N]
    if lossless:
        # Time reversal.  Valid only for a real index: conjugating a decaying
        # mode produces a growing one (5.16a).
        E[:, N:] = np.conjugate(E[:, :N])
        H[:, N:] = -np.conjugate(H[:, :N])
    else:
        # The reciprocal backward partner, with no conjugation:
        # (E_t, -E_z, -H_t, H_z) at beta -> -beta.  For a real index the
        # transverse components are real and the longitudinal ones imaginary,
        # so this reduces to the conjugated form above - which is why the
        # switch is a no-op on every dataset in this project.
        E[:, N:] = E[:, :N]
        H[:, N:] = -H[:, :N]
        E[:, N:, prop_axis] *= -1
        H[:, N:, prop_axis] *= -1

    return x, y, neff, TE_pol, E, H


def biorthogonalise(E, H, x, y, prop_axis=2):
    r"""Make each point's forward modes biorthogonal - docs/validation_backlog.md §5.6.

    §5.6 records that ``_pin_gauge`` fixes one global phase **per mode** and
    cannot fix a **near-degenerate subspace**, where the residual gauge freedom
    is continuous rather than a sign; and it names the coupled-pair
    anti-crossing as the operating point where that would bite.  It does.

    The whole interface algebra rests on ``O_aa = I``:
    ``T12 = 2·inv(O_ab + O_baᵀ)`` reduces to the identity for ``a = b`` only if
    a cross section's modes are biorthogonal among themselves.  Measured on a
    coupled Si pair, two modes at ``n_eff`` 2.386468 and 2.376356
    (``Δn`` = 0.0101) had ``|⟨E₄,H₅⟩| = 0.0745`` - so an interface between a
    section and *itself* was not the identity, and a **constant-width** guide,
    where the answer must be ``T = 1`` exactly, transmitted **1.59**.

    Two distinct effects are corrected:

    * **near-degeneracy** - ARPACK returns an arbitrary rotation inside a
      near-degenerate subspace and no per-mode phase convention can pin it;
    * **discretisation** - even well-separated modes are only biorthogonal to
      the accuracy of the quadrature (the ~2 % noted in docs/validation_backlog.md §4).

    The correction is Löwdin symmetric orthogonalisation in the
    **unconjugated** metric - the one the method actually uses (§5.16), which
    stays valid for lossy and PML problems where a conjugated one would not.
    Writing :math:`M_{ij} = \langle E_i, H_j \rangle`, the interface formula
    consumes ``O + Oᵀ``, so what must equal the identity is the symmetric part
    :math:`S = \tfrac12 (M + M^\mathsf{T})`.  With :math:`A = S^{-1/2}` and
    :math:`E_i \leftarrow \sum_j E_j A_{ji}` (likewise ``H``),
    :math:`A^\mathsf{T} S A = I` exactly, since ``S`` is complex symmetric and
    so is its principal square root.

    Löwdin is the right choice because it is the *minimal* change: of all
    transformations that orthogonalise the set it is closest to the identity,
    so modes keep their character and their ``n_eff`` labels stay meaningful.
    Mixing modes of different ``beta`` is only free when they are degenerate;
    here the mixing is large only where ``Δbeta`` is small, and tiny elsewhere.

    No transformation of the modes can remove the antisymmetric part
    ``½(M − Mᵀ)`` (a congruence keeps it), and only the input-side ``T`` is
    blind to it: the output-side ``T`` and the reflection block see it as a
    zero-step gain and reflection. ``SingleEME.INTERFACE_RECIPROCAL`` and
    ``INTERFACE_SELF_OVERLAP`` deal with it at the interface (2026-10-01).

    :param E: ``(n_pts, N, 3, nx, ny)`` forward fields, already normalised.
    :returns: ``(E, H)`` transformed in place-compatible fashion.
    """
    E = np.asarray(E)
    H = np.asarray(H)
    for p in range(E.shape[0]):
        M = overlap_matrix(E[p], H[p], x, y, prop_axis)
        A = _inverse_sqrt(0.5 * (M + M.T))
        E[p] = np.einsum("ji,jcxy->icxy", A, E[p])
        H[p] = np.einsum("ji,jcxy->icxy", A, H[p])
    return E, H


def _inverse_sqrt(S, real_tol=1e-9):
    """``S**(-1/2)`` for the symmetric overlap, keeping it real where it is real.

    A lossless mode set in the canonical gauge - transverse components real,
    longitudinal imaginary, which is what ``_pin_gauge`` establishes - gives a
    **real** ``S``.  Taking the real branch there keeps the mixing matrix real,
    and a real mixing preserves that reality structure.  That matters beyond
    tidiness: it is exactly the condition under which the two backward-basis
    constructions coincide (§5.16a, ``E- = conj(E+)`` versus the reciprocal
    partner), so every existing dataset keeps its meaning.  ``sqrtm`` on the
    same matrix would take the principal complex branch and quietly break it.

    A non-positive eigenvalue means the modes are linearly dependent in the
    overlap metric - two solver outputs spanning the same field.  There is no
    orthogonalisation of such a set, so it is raised rather than papered over.
    """
    scale = float(np.abs(S).max())
    if scale <= 0.0:
        raise ValueError("overlap matrix is identically zero")
    if np.abs(S.imag).max() <= real_tol * scale:
        w, V = np.linalg.eigh(S.real)
        # Relative, not absolute: the overlap carries the field normalisation,
        # so an absolute threshold would misjudge an unnormalised set entirely.
        if w.min() <= real_tol * w.max():
            raise ValueError(
                "mode set is linearly dependent in the overlap metric "
                f"(eigenvalues {w.min():.3e} .. {w.max():.3e}); two modes span "
                "the same field, so no biorthogonal basis exists"
            )
        return (V / np.sqrt(w)) @ V.T
    return np.linalg.inv(sqrtm(S))


def overlap_matrix(E_a, H_b, x, y, prop_axis=2):
    """Overlap matrix between the modes of two parameter points.

    :param E_a: Normalised ``E`` of point ``a``, shape ``(2N, 3, nx, ny)``.
    :param H_b: Normalised ``H`` of point ``b``, same shape.
    :returns: ``(2N, 2N)`` complex matrix, element ``[i, j]`` being
        :math:`\\tfrac12 \\int (E_{a,i} \\times H_{b,j})_z \\, dA`.

    Only the ``prop_axis`` component of the cross product survives the area
    integral, and the integral itself is a contraction over the grid, so the
    whole thing is two matrix products.  Forming the full
    ``(2N, 2N, 3, nx, ny)`` cross product first and summing it away instead
    costs 12.5 GB per call on the 5 nm Kocabas grid and 46.5 GiB on the
    2.5 nm one, which is where it stops being a performance question.
    """
    # Do not cast to float.  Under a PML the transverse coordinates are
    # complex-stretched, and the integration measure has to be stretched with
    # them or biorthogonality breaks and O(a, a) stops being the identity.
    # The symptom of getting this wrong is a mildly non-unitary S-matrix that
    # looks exactly like truncation error.
    dx = oct.compute_differences(np.asarray(x))
    dy = oct.compute_differences(np.asarray(y))
    weight = np.outer(dx, dy).reshape(-1)

    # (A x B)_p = A_{p+1} B_{p+2} - A_{p+2} B_{p+1}, indices modulo 3.
    i1, i2 = (prop_axis + 1) % 3, (prop_axis + 2) % 3
    n_a, n_b = E_a.shape[0], H_b.shape[0]

    def flat(field, component):
        return field[:, component].reshape(field.shape[0], -1)

    ea1 = flat(E_a, i1) * weight
    ea2 = flat(E_a, i2) * weight
    out = ea1 @ flat(H_b, i2).T
    out -= ea2 @ flat(H_b, i1).T
    return out.reshape(n_a, n_b) / 2


def prop_axis_index(crosssection_x, crosssection_y):
    """Index of the axis normal to the cross section: 0 = x, 1 = y, 2 = z."""
    remaining = {"x", "y", "z"} - {crosssection_x, crosssection_y}
    if len(remaining) != 1:
        raise ValueError(
            f"cross-section axes {crosssection_x!r}/{crosssection_y!r} are not "
            "two distinct axes out of x/y/z"
        )
    axis = remaining.pop()
    return {"x": 0, "y": 1, "z": 2}[axis]
