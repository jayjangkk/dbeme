"""Curvature sign handling.

A left bend and a right bend are mirror images.  Folding them onto ``|kappa|``
is exact only while a path never changes direction; an S-bend does, and folding
would collapse its two halves onto the same grid points and erase the mode
conversion at their join.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.geometry.curvature import (  # noqa: E402
    grid_has_negative_curvature,
    map_curvature,
)

SIGNED_GRID = {"curvature": np.linspace(-1e5, 1e5, 41)}
FOLDED_GRID = {"curvature": np.linspace(0, 1e5, 21)}


def test_signed_grid_is_detected():
    assert grid_has_negative_curvature(SIGNED_GRID)
    assert not grid_has_negative_curvature(FOLDED_GRID)
    assert not grid_has_negative_curvature({})


def test_signed_grid_keeps_the_sign():
    kappa = np.array([-5e4, 0.0, 5e4])
    assert np.allclose(map_curvature(kappa, SIGNED_GRID), kappa)


def test_folded_grid_takes_the_magnitude():
    kappa = np.array([-5e4, 0.0, 5e4])
    assert np.allclose(map_curvature(kappa, FOLDED_GRID), [5e4, 0.0, 5e4])


def test_s_bend_halves_are_distinct_on_a_signed_grid():
    """The two halves of an S-bend must land on different grid points."""
    kappa = np.array([4e4, 4e4, -4e4, -4e4])
    mapped = map_curvature(kappa, SIGNED_GRID)
    assert len(np.unique(mapped)) == 2, "sign change collapsed"

    folded = map_curvature(kappa, FOLDED_GRID)
    assert len(np.unique(folded)) == 1, "folding is supposed to collapse it"
