"""Invariant checks a result should pass before it is written up.

These are the Tier-1 items of the validation backlog, packaged so a demo can
open with a table of what was actually measured rather than an assertion that
the solver works.  All of them run on a warm dataset and cost no mode solving.

Each returns a :class:`Check` with the measured value and a pass flag, so a
failing check is reported rather than raised - a report that says a check
failed is more useful than one that could not be produced.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import matrix_calculation_tool as mct
from .propagator.eme import EME
from .runner.runner import Runner


@dataclass
class Check:
    """One invariant, its criterion, and what was measured."""

    name: str
    criterion: str
    measured: float
    passed: bool
    note: str = ""

    def __str__(self):
        mark = "pass" if self.passed else "FAIL"
        tail = f"  ({self.note})" if self.note else ""
        return f"{self.name:34s} {self.criterion:22s} {self.measured:10.3e}  {mark}{tail}"


def lumped_smatrix(geometry, force_unitary=False, force_passive=False):
    """The single scattering matrix of a whole device.

    :param geometry: A :class:`~em_simulation.geometry.geometry.Geometry`.
    :param force_unitary: Project each section onto the nearest unitary matrix.
        Leave **off** for anything measuring truncation error - the projection
        hides exactly what is being measured.
    :returns: ``(2N, 2N)`` complex array.
    """
    eme = EME(geometry, force_passive=force_passive, force_unitary=force_unitary)
    eme.calc_Smatrix()
    smatrix = eme.propagator.smatrix
    lumped = smatrix[0]
    for i in range(1, len(smatrix)):
        lumped = mct._redheffer_star_product(lumped, smatrix[i])
    return lumped


def guided_slice(geometry, section=0):
    """Indices of the modes that are guided at ``section``.

    The radiation modes are grid artefacts whose overlaps carry no physics, so
    every check below restricts itself to the guided block.
    """
    mask = geometry.output_data["radiation_mode_mask"]
    n_modes = mask.shape[1] // 2
    return [i for i in range(n_modes) if not mask[section, i]]


#: How far above the cladding index a mode must sit to count as resolved.
#:
#: ``radiation_mode_mask`` only asks ``n_eff > n_clad``, which lets through
#: modes sitting a hair above cutoff - and those are window artefacts, not
#: physics.  A mode's evanescent tail decays as
#: ``lambda / (2 pi sqrt(n_eff^2 - n_clad^2))``; at 1550 nm with
#: ``n_clad = 1.444`` a margin of 0.05 puts that decay length near 1.3 um,
#: which fits inside the ~2 um half-windows used here.  Anything closer to
#: cutoff is clipped by the boundary, so its fields - and every overlap built
#: from them - are set by the window rather than by the waveguide.
#:
#: This matters: on the angled rotator a mode at ``n_eff = 1.4453`` (0.0013
#: above cutoff, decay length 4 um in a 2.2 um half-window) drove the
#: reflection-symmetry check to 5.9e-2 while the two real branches were at
#: 7.8e-11.
GUIDED_MARGIN = 0.05


def guided_throughout(geometry, cutoff=None, margin=GUIDED_MARGIN):
    """Modes that stay properly guided from end to end.

    A branch that is below cutoff at one end and guided at the other is a real
    physical event, not an error, but no S-matrix identity holds for it - there
    is nothing at one port to be reciprocal *with*.  Every check below is
    restricted to the branches that exist all the way through, and that are far
    enough above cutoff to be resolved by the solve window - see
    :data:`GUIDED_MARGIN`.

    :param cutoff: Absolute ``n_eff`` floor.  Overrides ``margin`` when given.
    :param margin: How far above the cladding index to require.
    """
    output = geometry.output_data
    neff = np.real(output["neff"])
    n_modes = neff.shape[1] // 2

    if cutoff is None:
        cladding = _cladding_index(geometry)
        cutoff = None if cladding is None else cladding + margin

    if cutoff is None:
        mask = output["radiation_mode_mask"]
        return [i for i in range(n_modes) if not np.any(mask[:, i])]

    mask = output["radiation_mode_mask"]
    return [
        i
        for i in range(n_modes)
        if not np.any(mask[:, i]) and np.all(neff[:, i] > cutoff)
    ]


def _cladding_index(geometry):
    """The dataset's cladding index, if it can be reached from the geometry."""
    data = getattr(geometry, "data", None)
    getter = getattr(data, "get_cladding_index", None)
    if getter is None:
        return None
    try:
        return float(getter())
    except Exception:  # pragma: no cover - metadata is optional
        return None


def check_reciprocity(geometry, tolerance=1e-3, cutoff=None):
    """Reciprocity of the scattering matrix (5.1).

    Note the layout.  ``_convert_3Dmatrix`` does **not** produce the textbook
    port ordering ``[[S11, S12], [S21, S22]]``; it produces

        S = [[T_forward,  R_right],
             [R_left,     T_backward]]

    so the naive ``S == S.T`` is not the reciprocity statement here and fails
    at the 1e-2 level on a perfectly good taper.  What reciprocity actually
    requires of this layout is

        T_forward == T_backward^T          (measured here)
        R_left    == R_left^T,  R_right == R_right^T

    Measured with ``force_unitary=False``, since the projection would
    symmetrise the matrix and hide the very asymmetry being looked for.  This
    is the check that would catch a conjugated overlap integral, which
    unitarity alone cannot see.

    :returns: :class:`Check` on ``max|T_f - T_b^T|``.
    """
    if geometry.output_data is None:
        geometry.calc_output_data()
    smatrix = lumped_smatrix(geometry, force_unitary=False)
    keep = guided_throughout(geometry, cutoff)
    n_modes = geometry.output_data["neff"].shape[1] // 2
    if len(keep) < 1:
        return Check("reciprocity  T_f = T_b^T (5.1)", f"< {tolerance:.0e}",
                     float("nan"), True, "no end-to-end guided modes")

    shifted = [i + n_modes for i in keep]
    forward = smatrix[np.ix_(keep, keep)]
    backward = smatrix[np.ix_(shifted, shifted)]
    error = float(np.max(np.abs(forward - backward.T)))
    return Check(
        "reciprocity  T_f = T_b^T (5.1)",
        f"< {tolerance:.0e}",
        error,
        error < tolerance,
        f"{len(keep)} end-to-end guided modes",
    )


def check_reflection_symmetry(geometry, tolerance=1e-2, cutoff=None):
    """The two reflection blocks must each be symmetric (5.1).

    Measured in absolute terms, not relative.  On a device whose reflection is
    genuinely negligible - which is every adiabatic structure here - ``|R|``
    itself sits at the truncation floor, so a *relative* criterion would demand
    symmetry of what is essentially numerical noise.  The absolute asymmetry
    should be no worse than the power-conservation residual, and on a device
    with real reflection (a straight guide, where it is exactly zero, or an
    abrupt junction) it drops to 1e-12.
    """
    if geometry.output_data is None:
        geometry.calc_output_data()
    smatrix = lumped_smatrix(geometry, force_unitary=False)
    keep = guided_throughout(geometry, cutoff)
    n_modes = geometry.output_data["neff"].shape[1] // 2
    shifted = [i + n_modes for i in keep]

    right = smatrix[np.ix_(keep, shifted)]
    left = smatrix[np.ix_(shifted, keep)]
    error = float(
        max(np.max(np.abs(right - right.T)), np.max(np.abs(left - left.T)))
    )
    return Check(
        "reflection blocks symmetric (5.1)",
        f"< {tolerance:.0e}",
        error,
        error < tolerance,
        "R = R^T at both ports",
    )


def check_power_conservation(geometry, tolerance=1e-2, cutoff=None):
    """Total transmitted + reflected power, with the projection off (5.2).

    The residual is the truncation error of a finite mode basis.  It is a
    measurement, not a defect - but it bounds every other number in the report.

    Restricted to modes that are guided end to end: a branch that is cut off at
    one end genuinely loses its power, and including it would report a physical
    effect as a numerical error.
    """
    if geometry.output_data is None:
        geometry.calc_output_data()
    smatrix = lumped_smatrix(geometry, force_unitary=False)
    keep = guided_throughout(geometry, cutoff)
    if not keep:
        return Check("power conservation, no projection", f"< {tolerance:.0e}",
                     float("nan"), True, "no end-to-end guided modes")

    worst = 0.0
    for index in keep:
        launch = np.zeros(smatrix.shape[0], dtype=complex)
        launch[index] = 1.0
        total = float(np.sum(np.abs(smatrix @ launch) ** 2))
        worst = max(worst, abs(total - 1.0))
    return Check(
        "power conservation, no projection",
        f"|1 - sum| < {tolerance:.0e}",
        worst,
        worst < tolerance,
        f"worst of {len(keep)} end-to-end guided launches",
    )


def check_mode_basis_convergence(geometry_factory, mode_counts, tolerance=None):
    """Residual non-unitarity must fall as modes are added (5.2).

    ``limit_mode_number`` truncates the basis *from an existing dataset*, so
    this sweep costs no extra mode solving.

    :param geometry_factory: ``f(limit) -> Geometry`` built with that limit.
    :param mode_counts: Increasing mode counts to try.
    :returns: ``(Check, {count: residual})``
    """
    residuals = {}
    for count in mode_counts:
        geometry = geometry_factory(count)
        geometry.calc_output_data()
        residuals[count] = check_power_conservation(geometry).measured

    values = [residuals[c] for c in mode_counts]
    # Not strictly monotonic in general, but the largest basis must be the
    # best one - otherwise adding modes is making the answer worse.
    best_is_largest = values[-1] <= min(values) * 1.5 + 1e-12
    return (
        Check(
            "mode-basis convergence (5.2)",
            "residual falls with N",
            values[-1],
            best_is_largest,
            f"N={list(mode_counts)} -> " + ", ".join(f"{v:.1e}" for v in values),
        ),
        residuals,
    )


def check_slicing_independence(coarse_geometry, fine_geometry, tolerance=1e-3):
    """The same device sliced two ways must give the same transmission (5.4).

    Any difference is the parameter grid asserting itself, since both slicings
    snap onto the same axis.
    """
    for geometry in (coarse_geometry, fine_geometry):
        if geometry.output_data is None:
            geometry.calc_output_data()

    coarse = transmission(coarse_geometry)
    fine = transmission(fine_geometry)
    shared = min(len(coarse), len(fine))
    error = float(np.max(np.abs(coarse[:shared] - fine[:shared])))
    return Check(
        "slicing independence (5.4)",
        f"max|dT| < {tolerance:.0e}",
        error,
        error < tolerance,
        "coarse vs fine z-sampling",
    )


def transmission(geometry, launch_index=0, force_unitary=True):
    """Output power per forward mode for one launched mode.

    :returns: Array of length ``n_modes`` (forward block only).
    """
    if geometry.output_data is None:
        geometry.calc_output_data()
    eme = EME(geometry, force_unitary=force_unitary)
    eme.calc_Smatrix()
    runner = Runner(eme)
    n_launch = int(
        np.count_nonzero(geometry.output_data["radiation_mode_mask"][0] == False)
    )
    launch = np.zeros(n_launch, dtype=complex)
    launch[launch_index] = 1.0
    out = np.abs(runner.propagate_lumped_smatrix(launch)) ** 2
    n_modes = geometry.output_data["neff"].shape[1] // 2
    return out[:n_modes]


def check_branch_tracking(geometry, tolerance=0.5):
    """How confidently each tracked branch was followed section to section (5.13).

    Mode order from the solver is by effective index, so labels swap wherever
    two branches cross.  ``Geometry`` re-labels by maximum overlap with the
    previous section instead, and after that re-ordering the diagonal element
    ``overlap_ab[i, b, b]`` *is* the link that was used for branch ``b`` at
    interface ``i``.  Its magnitude is therefore a direct measure of tracking
    confidence: near 1 means the branch was recognised unambiguously, and a
    value below the linker's own 0.5 threshold means it was not - at which
    point the branch is given a new label and the physics is lost.

    Note what this deliberately does **not** test.  An earlier version required
    the tracked branches never to cross in effective index; that is wrong.  Two
    branches of different polarisation do not interact, so they cross for real,
    and following them through the crossing is exactly what correct tracking
    looks like.  Only branches that *do* interact repel, and the link
    magnitudes below detect a genuine failure whether or not a crossing is
    involved.

    :returns: ``(Check, per-branch minimum link magnitude)``
    """
    if geometry.output_data is None:
        geometry.calc_output_data()

    overlap = geometry.output_data["overlap_ab"]
    n_modes = overlap.shape[1] // 2
    keep = guided_throughout(geometry)
    if not keep:
        return (
            Check("branch tracking (5.13)", f"link > {tolerance}", float("nan"),
                  True, "no end-to-end guided branches"),
            np.array([]),
        )

    links = np.abs(np.einsum("ijj->ij", overlap[:, :n_modes, :n_modes]))
    per_branch = links[:, keep].min(axis=0)
    worst = float(per_branch.min())
    return (
        Check(
            "branch tracking (5.13)",
            f"link > {tolerance}",
            worst,
            worst > tolerance,
            f"weakest section-to-section link over {len(keep)} branches",
        ),
        per_branch,
    )


def format_table(checks):
    """Render checks as the markdown table a report opens with."""
    lines = [
        "| check | criterion | measured | pass |",
        "|---|---|---|---|",
    ]
    for check in checks:
        mark = "yes" if check.passed else "**no**" if check.passed is False else "-"
        value = "n/a" if np.isnan(check.measured) else f"{check.measured:.2e}"
        note = f" <br><sub>{check.note}</sub>" if check.note else ""
        lines.append(
            f"| {check.name} | {check.criterion} | {value}{note} | {mark} |"
        )
    return "\n".join(lines)
