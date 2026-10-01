"""The sign gauge ``_equalize_overlap_phase`` puts on the overlap sets.

A mode's sign belongs to its field, so the equalisation must remove it from
``overlap_ab`` and ``overlap_ba`` alike.  Upstream took a second, independent
set of masks from ``diag(overlap_ba)``; on a lossy basis a branch can have
diagonals of opposite sign in the two sets, and from there on that branch
carried opposite signs in them (README, "What changed relative to upstream").

The overlaps here are synthetic: a known gauge ``G_i`` (one sign per mode,
section 0 the reference) is put on overlaps whose diagonals are canonical,
and the rule must take it off again - exactly, since only signs are involved -
whatever ``diag(overlap_ba)`` says.  Checked for ``Geometry`` and
``DirectGeometry``, which carry the same rule.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.geometry.direct_geometry import DirectGeometry  # noqa: E402
from dbeme.geometry.geometry import Geometry  # noqa: E402

N = 4   # forward modes; the matrices are 2N x 2N, forward block first


def canonical(rng, n_overlaps):
    """Overlaps in the canonical gauge: ``Re diag`` positive in the forward
    block, negative in the backward block (the backward mode is ``(E, -H)``),
    ``O_ba ~ O_ab^T`` as between neighbouring cross sections."""
    ab, ba = [], []
    for _ in range(n_overlaps):
        A = 0.1 * (rng.standard_normal((2 * N, 2 * N)) + 1j * rng.standard_normal((2 * N, 2 * N)))
        d = (0.8 + 0.2 * rng.random(N)) * np.exp(1j * rng.uniform(-1.0, 1.0, N))   # complex, Re > 0
        A[np.arange(N), np.arange(N)] = d
        A[np.arange(N, 2 * N), np.arange(N, 2 * N)] = -d
        B = A.T + 0.01 * (rng.standard_normal((2 * N, 2 * N)) + 1j * rng.standard_normal((2 * N, 2 * N)))
        ab.append(A)
        ba.append(B)
    return np.array(ab), np.array(ba)


def gauged(ab, ba, rng):
    """Put a random per-mode sign on every section but the first."""
    n = len(ab)
    g = [np.ones(N)] + [rng.choice([-1.0, 1.0], N) for _ in range(n)]
    G = [np.concatenate([s, s]) for s in g]          # a mode and its backward partner flip together
    ab_raw = np.array([G[i][:, None] * ab[i] * G[i + 1][None, :] for i in range(n)])
    ba_raw = np.array([G[i + 1][:, None] * ba[i] * G[i][None, :] for i in range(n)])
    return ab_raw, ba_raw


def equalize(cls, ab, ba, **kwargs):
    ab, ba = np.array(ab, copy=True), np.array(ba, copy=True)
    if cls is Geometry:
        if kwargs.pop("with_dict", True) and not kwargs:
            kwargs = {"overlap_dict": {"ab": {}, "ba": {}}, "index_mapping": {i: i for i in range(len(ab))}}
        out = Geometry._equalize_overlap_phase(None, ab, ba, **kwargs)
    else:
        out = DirectGeometry._equalize_overlap_phase(SimpleNamespace(_verbose=False), ab, ba)
    return out


RULES = [Geometry, DirectGeometry]


@pytest.mark.parametrize("cls", RULES)
@pytest.mark.parametrize("n_overlaps", [1, 2, 3, 12])
def test_the_gauge_comes_off_both_sets(cls, n_overlaps):
    """Every section's sign, the last one included, whatever the path length."""
    rng = np.random.default_rng(n_overlaps)
    ab, ba = canonical(rng, n_overlaps)
    out = equalize(cls, *gauged(ab, ba, rng))
    assert np.array_equal(out[0], ab)
    assert np.array_equal(out[1], ba)


@pytest.mark.parametrize("cls", RULES)
@pytest.mark.parametrize("k", [0, 3, 9])
def test_a_disagreeing_ba_diagonal_does_not_move_the_gauge(cls, k):
    """A branch whose ``O_ba`` diagonal has the other sign - what a radiating
    PML branch can do - keeps the ``O_ab`` sign in both sets, at that
    interface and at every one after it (first, interior and last interface)."""
    rng = np.random.default_rng(7)
    ab, ba = canonical(rng, 10)
    j = 2
    ba[k, j, j] = -ba[k, j, j]                       # the disagreement is in the data
    ba[k, N + j, N + j] = -ba[k, N + j, N + j]
    out = equalize(cls, *gauged(ab, ba, rng))
    assert np.array_equal(out[0], ab)
    assert np.array_equal(out[1], ba)


@pytest.mark.parametrize("cls", RULES)
@pytest.mark.parametrize("k", [2, 5])
def test_a_zero_diagonal_does_not_delete_the_mode(cls, k):
    """``np.sign(0)`` is 0: used as a mask it zeroed the mode's column, the next
    diagonal was then 0 too, and the mode vanished from the rest of the path.
    A zero now keeps the previous section's sign, for the mode and its
    backward partner alike (interior and last interface)."""
    rng = np.random.default_rng(11)
    ab, ba = canonical(rng, 6)
    ab[k, 1, 1] = 0.0
    ab[k, N + 1, N + 1] = -0.0
    out = equalize(cls, ab, ba)
    for i in range(k, 6):
        assert np.array_equal(out[0][i], ab[i])
        assert np.array_equal(out[1][i], ba[i])
    assert np.abs(out[0][-1][:, 1]).max() > 0


def test_geometry_without_an_overlap_dict():
    rng = np.random.default_rng(3)
    ab, ba = canonical(rng, 4)
    ab_out, ba_out, overlap_dict = equalize(Geometry, *gauged(ab, ba, rng), with_dict=False)
    assert overlap_dict == {"ab": {}, "ba": {}}
    assert np.array_equal(ab_out, ab) and np.array_equal(ba_out, ba)


def test_geometry_masks_the_corner_overlaps_like_the_main_sets():
    """With a one-to-one ``index_mapping``, an interior
    ``additional_overlap_dict`` entry gets the masks of the path overlap it
    sits beside, in ``ab`` and in ``ba`` alike.  (With a many-to-one mapping
    the dict takes upstream's cumulative masks; no propagator reads it.)"""
    rng = np.random.default_rng(5)
    ab, ba = canonical(rng, 8)
    ab_raw, ba_raw = gauged(ab, ba, rng)
    k = 4
    overlap_dict = {"ab": {k: [ab_raw[k].copy()]}, "ba": {k: [ba_raw[k].copy()]}}
    index_mapping = {i: i for i in range(len(ab))}
    ab_out, ba_out, overlap_dict = equalize(Geometry, ab_raw, ba_raw, overlap_dict=overlap_dict,
                                            index_mapping=index_mapping)
    assert np.array_equal(overlap_dict["ab"][k][0], ab_out[k])
    assert np.array_equal(overlap_dict["ba"][k][0], ba_out[k])
