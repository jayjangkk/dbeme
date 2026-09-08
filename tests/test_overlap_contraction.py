"""``overlap_matrix`` contracts; it must still be the same integral.

The implementation used to build ``np.cross(E_a[:, None], H_b[None, :])`` -
an array of shape ``(2N, 2N, 3, nx, ny)`` - and then sum it away.  Only the
``prop_axis`` component survives the area integral, so the whole thing is two
matrix products.  On the 5 nm Kocabas grid the old form needed 11.7 GiB of
temporary per call and on the 2.5 nm one 46.5 GiB, which is where a run dies
rather than merely being slow (report 13 section 7).

These tests pin the rewrite against a literal transcription of the old form,
including the two things that are easy to lose: the complex (PML-stretched)
integration measure, and the general ``prop_axis``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import em_simulation.data_updater.overlap_calculation_tool as oct  # noqa: E402
from em_simulation.fde.assemble import overlap_matrix  # noqa: E402


def reference(E_a, H_b, x, y, prop_axis=2):
    """The pre-2026-09-08 implementation, verbatim."""
    cross = np.cross(E_a[:, np.newaxis], H_b[np.newaxis, :], axis=2)
    dx = oct.compute_differences(np.asarray(x))
    dy = oct.compute_differences(np.asarray(y))
    weight = np.outer(dx, dy)[np.newaxis, np.newaxis, np.newaxis, :, :]
    return (weight * cross).sum(axis=(3, 4))[:, :, prop_axis] / 2


def fields(rng, n, nx, ny, dtype):
    shape = (n, 3, nx, ny)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)


def stretched(nx, ny):
    """Coordinates as a PML leaves them: complex, unequally spaced."""
    x = np.linspace(0.0, 1.0, nx) + 1j * np.linspace(0.0, 0.3, nx) ** 3
    y = np.linspace(0.0, 1.0, ny) - 1j * np.linspace(0.0, 0.2, ny) ** 2
    return x, y


@pytest.mark.parametrize("prop_axis", [0, 1, 2])
@pytest.mark.parametrize("dtype,tol", [(np.complex64, 2e-6), (np.complex128, 1e-12)])
def test_matches_the_old_form(prop_axis, dtype, tol):
    rng = np.random.default_rng(11 + prop_axis)
    nx, ny, n = 29, 23, 8
    E, H = fields(rng, n, nx, ny, dtype), fields(rng, n, nx, ny, dtype)
    x, y = stretched(nx, ny)
    want = reference(E, H, x, y, prop_axis)
    got = overlap_matrix(E, H, x, y, prop_axis)
    assert got.shape == (n, n)
    assert np.abs(got - want).max() / np.abs(want).max() < tol


def test_the_measure_stays_complex():
    """A real-cast measure would break biorthogonality under a PML."""
    rng = np.random.default_rng(3)
    nx, ny, n = 25, 21, 4
    E, H = fields(rng, n, nx, ny, np.complex128), fields(rng, n, nx, ny, np.complex128)
    x, y = stretched(nx, ny)
    complex_metric = overlap_matrix(E, H, x, y)
    real_metric = overlap_matrix(E, H, np.real(x) + 0j, np.real(y) + 0j)
    assert np.abs(complex_metric - real_metric).max() > 1e-6


def test_rectangular_mode_counts():
    """The two sides need not carry the same number of modes."""
    rng = np.random.default_rng(5)
    nx, ny = 19, 17
    E, H = fields(rng, 8, nx, ny, np.complex128), fields(rng, 3, nx, ny, np.complex128)
    x, y = stretched(nx, ny)
    got = overlap_matrix(E, H, x, y)
    assert got.shape == (8, 3)
    assert np.abs(got - reference(E, H, x, y)).max() / np.abs(got).max() < 1e-12


def test_no_pairwise_temporary(monkeypatch):
    """The point of the rewrite: no ``(2N, 2N, 3, nx, ny)`` array is formed.

    A size assertion cannot see this - numpy allocates outside tracemalloc -
    so pin the behaviour instead: with ``np.cross`` removed, an implementation
    that still builds the pairwise cross product cannot run, and a contracting
    one is unaffected.
    """
    rng = np.random.default_rng(17)
    nx, ny, n = 40, 36, 10
    E, H = fields(rng, n, nx, ny, np.complex128), fields(rng, n, nx, ny, np.complex128)
    x, y = stretched(nx, ny)
    want = reference(E, H, x, y)

    def refuse(*args, **kwargs):
        raise AssertionError("overlap_matrix must not form the pairwise cross product")

    monkeypatch.setattr(np, "cross", refuse)
    got = overlap_matrix(E, H, x, y)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12
