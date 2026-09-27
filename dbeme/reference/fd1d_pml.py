r"""A 1-D finite-difference mode solver with a complex-stretched PML.

This is the *thing being measured* in Phase 1 of ``tasks/02_pml_backend.md``.
:mod:`~dbeme.reference.bent_slab` gives the semi-analytic answer; this
module reproduces it with the machinery a real PML backend would use - complex
coordinate stretching, a non-Hermitian operator, and shift-invert selection of
the mode nearest a target index.  If the two disagree, the formulation is
wrong, and a 2-D solver would only hide it.

It deliberately solves the **same linearised profile** as the Airy reference,

.. math::  n_{eq}^2(u) = n^2(u)\,(1 + 2u/R)

so that any discrepancy is attributable to the discretisation and the PML, not
to a different physical model.

Three things this exercises that a lossless solver never does
------------------------------------------------------------
* **The PML must start outside the turning point.**  Inside it the field is
  evanescent and there is nothing to absorb; a PML placed there does nothing
  useful and can reflect.  ``standoff`` is measured from :math:`u_t`.
* **The index profile is continued to complex coordinates.**  The outward index
  ramp is what makes the field radiate, and it has to be evaluated at the
  stretched coordinate, not at its real part - otherwise the PML region is not
  the analytic continuation of the physical one.
* **Mode selection cannot be "largest real part".**  With a PML the spectrum
  fills with Berenger modes of the discretised continuum, so the wanted mode is
  found by shift-invert about a target, never by sorting.

Nothing here imports DBEME.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["PMLGrid", "solve_bent_slab"]


class PMLGrid:
    """A 1-D grid whose outer increments are complex-stretched.

    :param inner: Extent on the inside of the bend, metres (real grid).
    :param pml_start: Where the stretch begins, metres.
    :param thickness: PML thickness in unstretched metres.
    :param dx: Target step in the physical region.
    :param factor: Complex multiplier reached at the outer edge.  ``1 + 2j`` is
        the value emepy's own (commented-out) recipe intended.
    :param order: Polynomial grading exponent of the ramp.
    """

    def __init__(self, inner, pml_start, thickness, dx, factor=1 + 2j, order=2):
        self.factor = complex(factor)
        self.order = int(order)

        n_phys = max(int(round((pml_start + inner) / dx)), 8)
        n_pml = max(int(round(thickness / dx)), 4)
        self.n_physical, self.n_pml = n_phys, n_pml

        step_phys = (pml_start + inner) / n_phys
        step_pml = thickness / n_pml

        # Grade the stretch in from 1 so the PML does not present a step
        # discontinuity to the incoming wave.
        ramp = ((np.arange(1, n_pml + 1) - 0.5) / n_pml) ** self.order
        stretch = 1.0 + (self.factor - 1.0) * ramp

        increments = np.concatenate([
            np.full(n_phys, step_phys, dtype=complex),
            step_pml * stretch,
        ])
        self.u = -inner + np.concatenate([[0.0], np.cumsum(increments)])
        self.increments = increments
        self.pml_start = pml_start

    def __len__(self):
        return len(self.u)

    @property
    def physical_u(self):
        """Real part of the coordinate, for reporting and material bounds."""
        return np.real(self.u)


def _second_derivative(u):
    """Complex tridiagonal second-derivative operator on a non-uniform grid.

    Standard three-point formula, kept in complex arithmetic throughout so a
    stretched coordinate flows straight through.  Dirichlet at both ends.
    """
    n = len(u)
    h = np.diff(u)
    hm, hp = h[:-1], h[1:]
    lower = 2.0 / (hm * (hm + hp))
    centre = -2.0 / (hm * hp)
    upper = 2.0 / (hp * (hm + hp))

    interior = n - 2
    return sp.diags(
        [lower[1:], centre, upper[:-1]],
        offsets=[-1, 0, 1],
        shape=(interior, interior),
        dtype=complex,
    ).tocsc()


def solve_bent_slab(
    slab,
    radius,
    neff_target=None,
    dx=20e-9,
    inner=4e-6,
    standoff=1.5e-6,
    pml_thickness=4e-6,
    factor=1 + 2j,
    order=2,
    n_eigenvalues=12,
):
    """Complex ``n_eff`` of a bent slab, by finite difference with a PML.

    :param slab: A :class:`~dbeme.reference.bent_slab.BentSlab`, used
        for its geometry, indices and turning point - not for its solver.
    :param radius: Bend radius, metres.
    :param neff_target: Shift-invert target; defaults to the straight guide.
    :param dx: Step in the physical region, metres.
    :param inner: How far inward the domain extends, metres.
    :param standoff: Gap between the turning point and the PML, metres.
    :param pml_thickness: PML thickness, metres.
    :param factor: Complex stretch reached at the outer edge.
    :param order: Grading exponent.
    :returns: Complex ``n_eff``, ``Im > 0`` for a lossy mode.
    """
    k0 = slab.k0
    if neff_target is None:
        neff_target = slab.straight_neff()

    turning = slab.turning_point(neff_target, radius)
    grid = PMLGrid(
        inner=inner,
        pml_start=turning + standoff,
        thickness=pml_thickness,
        dx=dx,
        factor=factor,
        order=order,
    )
    u = grid.u

    # Piecewise index on the physical coordinate, then the conformal ramp
    # continued analytically to the complex coordinate.
    physical = grid.physical_u
    index = np.where(
        np.abs(physical) <= slab.half_width,
        slab.core_index,
        np.where(physical < 0, slab.inner_index, slab.clad_index),
    )
    n_eq_squared = index**2 * (1.0 + 2.0 * u / radius)

    operator = _second_derivative(u) + sp.diags(
        k0**2 * n_eq_squared[1:-1], dtype=complex
    ).tocsc()

    # Shift-invert about the target: with a PML the spectrum is dense with
    # Berenger modes, and "largest real part" would return one of those.
    sigma = (k0 * complex(neff_target)) ** 2
    values, vectors = spla.eigs(
        operator, k=min(n_eigenvalues, operator.shape[0] - 2), sigma=sigma,
        which="LM",
    )

    neff = np.sqrt(values) / k0
    neff = np.where(np.real(neff) < 0, -neff, neff)

    # Among the shift-invert results, take the one that is actually guided and
    # closest to the target.  A leaky mode still sits between the cladding and
    # core indices in its real part.
    guided = np.flatnonzero(
        (np.real(neff) > slab.clad_index) & (np.real(neff) < slab.core_index)
    )
    if not guided.size:
        raise RuntimeError(
            f"no guided eigenvalue near n_eff = {np.real(neff_target):.4f}"
        )
    best = guided[np.argmin(np.abs(neff[guided] - neff_target))]
    winner = neff[best]

    return complex(np.real(winner), abs(np.imag(winner))), {
        "grid_points": len(grid),
        "pml_points": grid.n_pml,
        "turning_point": turning,
        "pml_start": grid.pml_start,
        "field": vectors[:, best],
        "u": u,
    }
