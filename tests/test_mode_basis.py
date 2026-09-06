"""Basis invariants: biorthogonality (§5.6) and mode tracking (§5.13).

Both were found by `reports/07_sirac_optimization.md` §8 on a coupled Si pair
and fixed upstream.  Neither is device-specific: any coupled or symmetric
structure produces near-degenerate modes, and any anti-crossing defeats a
greedy tracker.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from em_simulation.data_updater import overlap_calculation_tool as oct  # noqa: E402
from em_simulation.fde.assemble import (  # noqa: E402
    assemble,
    biorthogonalise,
    overlap_matrix,
)
from em_simulation.geometry.mode_tracking import (  # noqa: E402
    hungarian_mode_links,
    link_quality,
)

WAVELENGTH = 1.310e-6


# ------------------------------------------------------------ §5.6

def _grid(nx=14, ny=11):
    return (np.linspace(-1.2e-6, 1.2e-6, nx), np.linspace(-0.8e-6, 0.8e-6, ny))


def _random_modes(rng, n, x, y):
    """Power-normalised random fields, as ``assemble`` hands them over.

    Normalisation is not cosmetic here: the overlap integral carries the field
    scale, and on a 1e-6 grid unnormalised O(1) fields give ``S ~ 1e-13``,
    which is entirely round-off.
    """
    shape = (1, n, 3, len(x), len(y))
    E = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    H = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return oct.normalize_field(E, H, x, y, 2)


def test_biorthogonalise_makes_the_symmetric_self_overlap_identity():
    """``½(M + Mᵀ) = I`` is what ``T12 = 2·inv(O_ab + O_baᵀ)`` presumes."""
    rng = np.random.default_rng(0)
    x, y = _grid()
    n = 5
    E, H = _random_modes(rng, n, x, y)

    M = overlap_matrix(E[0], H[0], x, y, 2)
    before = np.abs(0.5 * (M + M.T) - np.eye(n)).max()

    E2, H2 = biorthogonalise(E.copy(), H.copy(), x, y, 2)
    M2 = overlap_matrix(E2[0], H2[0], x, y, 2)
    after = np.abs(0.5 * (M2 + M2.T) - np.eye(n)).max()

    assert before > 1e-2
    assert after < 1e-9


def test_biorthogonalise_leaves_an_already_biorthogonal_set_alone():
    """It must be a no-op where there is nothing to fix (minimality)."""
    x, y = _grid()
    n = 4
    E, H = _random_modes(np.random.default_rng(1), n, x, y)
    E, H = biorthogonalise(E, H, x, y, 2)

    E2, H2 = biorthogonalise(E.copy(), H.copy(), x, y, 2)
    # Idempotent to the relative precision of the fields themselves.
    assert np.abs(E2 - E).max() < 1e-6 * np.abs(E).max()
    assert np.abs(H2 - H).max() < 1e-6 * np.abs(H).max()


def test_assemble_applies_biorthogonalisation_and_can_be_switched_off():
    """The flag exists to reproduce pre-fix results, and must actually do so."""
    from em_simulation.fde.base import ModeData

    rng = np.random.default_rng(2)
    x, y = _grid()
    n = 4
    shape = (n, 3, len(x), len(y))
    md = ModeData(
        x=x, y=y,
        E=(rng.normal(size=shape) + 1j * rng.normal(size=shape)),
        H=(rng.normal(size=shape) + 1j * rng.normal(size=shape)),
        neff=np.linspace(2.5, 2.0, n).astype(complex),
        TE_pol=np.full(n, 0.9),
    )

    _, _, _, _, E_on, H_on = assemble([md], n, prop_axis=2, biorthogonal=True)
    M_on = overlap_matrix(E_on[0, :n], H_on[0, :n], x, y, 2)
    # assemble stores the basis as complex64, so 1e-6 is the floor here.
    assert np.abs(0.5 * (M_on + M_on.T) - np.eye(n)).max() < 1e-6

    _, _, _, _, E_off, H_off = assemble([md], n, prop_axis=2, biorthogonal=False)
    M_off = overlap_matrix(E_off[0, :n], H_off[0, :n], x, y, 2)
    assert np.abs(0.5 * (M_off + M_off.T) - np.eye(n)).max() > 1e-3


# ------------------------------------------------------------ §5.13

def test_hungarian_mode_links_are_a_permutation():
    """A per-column argmax can claim one predecessor twice; this cannot.

    The fixture is the anti-crossing failure mode: branches 1 and 2 both have
    their largest overlap with predecessor 1, so greedy leaves predecessor 2
    unclaimed and a branch with no data (``n_eff = 0``).
    """
    n = 3
    block = np.array([[0.98, 0.02, 0.01],
                      [0.03, 0.71, 0.72],
                      [0.01, 0.66, 0.68]])
    overlaps = np.zeros((1, 2 * n, 2 * n), dtype=complex)
    overlaps[0, :n, :n] = block

    greedy = [int(np.argmax(np.abs(block[:, j]))) for j in range(n)]
    assert len(set(greedy)) < n, "fixture should defeat a per-column argmax"

    links, quality = hungarian_mode_links(overlaps)
    assert sorted(links[0].tolist()) == list(range(n))
    assert links.shape == (1, n)
    assert quality.shape == (1, n)


def test_hungarian_matches_greedy_when_greedy_is_already_a_permutation():
    n = 4
    block = np.eye(n) * 0.9 + np.full((n, n), 0.02)
    overlaps = np.zeros((1, 2 * n, 2 * n), dtype=complex)
    overlaps[0, :n, :n] = block
    links, _ = hungarian_mode_links(overlaps)
    assert links[0].tolist() == list(range(n))


def test_link_quality_reports_the_weakest_link():
    quality = np.array([[0.9, 0.8], [0.4, 0.95]])
    report = link_quality(quality)
    assert report["min_overlap"] == 0.4
    assert report["worst_section"] == 1
    assert report["sections_below_half"] == 1


def test_linearly_dependent_modes_are_rejected_not_mixed():
    """Two solver outputs spanning one field have no biorthogonal basis.

    Silently orthogonalising such a set produces a complex mixing matrix even
    for a real, lossless problem, which breaks the reality structure that makes
    the two backward-basis constructions agree (§5.16a).  Better to say so.
    """
    import pytest

    x, y = _grid()
    n, nx, ny = 2, len(x), len(y)
    E = np.zeros((1, n, 3, nx, ny), dtype=complex)
    H = np.zeros((1, n, 3, nx, ny), dtype=complex)
    E[0, :, 0] = 1.0
    H[0, :, 1] = 1.0          # both "modes" are the same field
    with pytest.raises(ValueError, match="linearly dependent"):
        biorthogonalise(E, H, x, y, 2)
