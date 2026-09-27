"""Gauge consistency as the *dataset* sees it.

``test_gauge_reproducibility.py`` covers the underlying question - whether two
independent solves of one cross section return the same fields - and explains
why the gauge is free in the first place.  This file checks the three
consequences that matter to ``DataUpdater``, which never sees fields and only
ever stores overlap matrices:

* an overlap must not depend on the order its two points were solved in, since
  one of them may come from the pickle and the other from a fresh solve;
* the overlap diagonal must come out canonically positive, so that
  ``Geometry._equalize_overlap_phase`` is a no-op rather than a correction;
* a point re-solved after the in-memory mode cache is dropped must reproduce
  itself, which is the cross-session case in miniature.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402

POINT = (0.9e-6, 0.0)
NEIGHBOUR = (0.92e-6, 0.0)

# The eigensolver runs at accuracy 1e-8 and the overlap integral roughly
# doubles that, so anything at 1e-7 or below is convergence noise.  A gauge
# flip moves these quantities by 2.0, so the test does not need a tight bound
# to be decisive.
REPRODUCIBILITY_TOL = 1e-6


@pytest.fixture(scope="module")
def backend():
    return EmepyFDE(
        cross_section=FullEtchStrip(),
        num_modes=5,
        wavelength=1.55e-6,
        window=(1.6e-6, -0.8e-6, 0.8e-6),
        mesh=110,
    )


def _overlap(backend, solve_order):
    """Overlap between POINT and NEIGHBOUR, solving them in a given order."""
    modes = {p: backend.solve(p) for p in solve_order}
    x, y, neff, te, E, H = assemble(
        [modes[POINT], modes[NEIGHBOUR]], backend.num_modes, prop_axis=2
    )
    return overlap_matrix(E[0], H[1], x, y, prop_axis=2)


def test_solve_order_does_not_change_the_overlap(backend):
    """A point re-solved later, against a different neighbour, must match."""
    forward = _overlap(backend, [POINT, NEIGHBOUR])
    reversed_ = _overlap(backend, [NEIGHBOUR, POINT])
    assert np.max(np.abs(forward - reversed_)) < REPRODUCIBILITY_TOL


def test_adjacent_overlap_diagonal_is_positive(backend):
    """The pinned gauge makes the overlap diagonal canonically positive."""
    O = _overlap(backend, [POINT, NEIGHBOUR])
    diagonal = np.real(np.diag(O))[: backend.num_modes]
    # Only modes that actually correspond between the two widths carry a
    # meaningful diagonal; an unguided mode at one width has no partner at the
    # other and its diagonal is ~0, of arbitrary sign.
    tracked = diagonal[np.abs(diagonal) > 0.5]
    assert tracked.size >= 3, diagonal
    assert np.all(tracked > 0), diagonal


def test_gauge_survives_the_mode_cache_being_dropped(backend):
    """Solve, discard the learned grid, solve again - fields must be identical."""
    first = backend.solve(POINT)
    backend._field_x = backend._field_y = None
    second = backend.solve(POINT)

    for name, a, b in (("E", first.E, second.E), ("H", first.H, second.H)):
        scale = np.max(np.abs(a))
        assert np.max(np.abs(a - b)) < REPRODUCIBILITY_TOL * scale, name
    assert np.allclose(first.neff, second.neff, atol=1e-8)
