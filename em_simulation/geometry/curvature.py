"""Mapping signed path curvature onto the dataset's curvature axis.

A bend that turns left and one that turns right are mirror images of each
other, so for a symmetric cross section they have the same modes up to a
reflection.  The upstream implementation used that to halve the dataset: it
stored non-negative curvature only and took ``|kappa|`` when building a path
(the sign was carried by ``rotation_angle``, which only exists for datasets
whose cross section is itself orientation-dependent).

That shortcut is exact for a bend that never changes direction.  It is *not*
exact across a sign change, which is what an S-bend does at its midpoint.  The
two half-bends have mirrored mode sets, and the overlap between a mode of one
and a mode of the other is a genuinely different integral - it cannot be
recovered from the folded data.  Folding an S-bend therefore makes the two
halves look identical and hides the mode conversion at the join entirely.

So: fold only when the dataset has no negative curvature to fold onto.  A
dataset whose ``curvature`` axis is symmetric about zero keeps the sign, treats
left and right bends as distinct grid points, and gets the join right.
"""

import numpy as np


def grid_has_negative_curvature(parameter_grid):
    """Whether the dataset's curvature axis extends below zero."""
    curvature = parameter_grid.get("curvature")
    return curvature is not None and float(np.min(curvature)) < 0.0


def map_curvature(curvatures, parameter_grid):
    """Map signed path curvature onto the dataset axis.

    :param curvatures: Signed curvature along the path, in 1/m.
    :param parameter_grid: The dataset's ``{name: values}`` grid.
    :returns: Curvature values as they should appear in the parameter tuples.
    """
    curvatures = np.asarray(curvatures, dtype=float)
    if grid_has_negative_curvature(parameter_grid):
        return curvatures
    return np.abs(curvatures)
