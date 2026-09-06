"""The mode gauge must be reproducible across independent solves.

Why this matters
----------------
An eigenvector is defined only up to a complex scale, and EMpy calls ARPACK
(``scipy.sparse.linalg.eigs``) without a fixed ``v0``, so the starting vector
is random and the *same* cross section can come back with a mode of opposite
sign on two different runs.

Power normalisation cannot remove that freedom, because
``P = 1/2 * int (E x H)_z`` is **quadratic** in the eigenvector:
``(-E) x (-H) = E x H``.  So P is identical for +psi and -psi and dividing by
sqrt(P) rescales without ever flipping.  The conjugated form is quadratic too,
so it is equally blind.

The overlap, by contrast, is **linear** in each mode separately, so a flip of
mode i at point a flips row i of ``O(a, b)``.

Within one consistently built chain that cancels: a gauge ``G_k = diag(+-1)``
at point k enters as ``O(k-1,k) G_k`` on one side and ``G_k O(k,k+1)`` on the
other, and ``G_k G_k = I``.  The cancellation requires *the same* ``G_k`` on
both sides -- i.e. every overlap touching point k must come from one and the
same field array.

The dataset persists overlaps but **not** fields.  So if point P is re-solved
in a later session to build its overlap with a new neighbour B, while
``O(A, P)`` is served from the pickle, the two are in different gauges and a
spurious sign survives on that row: silently, and depending on the order in
which devices happened to be run.

``emepy_fde._pin_gauge`` closes this by fixing each mode's global phase
deterministically.  This test is the regression guard for that.
"""

import numpy as np
import pytest

from em_simulation.fde import EmepyFDE, FullEtchStrip
from em_simulation.fde import emepy_fde as efde

POINT = (0.7e-6, 0.0)
NUM_MODES = 3


def _backend():
    return EmepyFDE(
        cross_section=FullEtchStrip(
            thickness=220e-9, core_index=3.4757, clad_index=1.444
        ),
        parameter_names=("top_width", "curvature"),
        num_modes=NUM_MODES,
        wavelength=1.55e-6,
        window=(1.2e-6, -0.6e-6, 0.6e-6),
        mesh=90,
    )


def _relative_field_difference(a, b):
    """max|Ea - Eb| / max|Ea|, per mode.  A sign flip gives ~2."""
    scale = np.abs(a).max(axis=(1, 2, 3))
    scale[scale == 0] = 1.0
    return np.abs(a - b).max(axis=(1, 2, 3)) / scale


def test_repeated_solve_returns_the_same_gauge():
    """Two independent solves of one cross section must agree field-for-field."""
    backend = _backend()
    first = backend.solve(POINT)
    second = backend.solve(POINT)

    np.testing.assert_allclose(first.neff, second.neff, rtol=1e-6)

    drift = _relative_field_difference(first.E, second.E)
    assert drift.max() < 1e-3, (
        "mode fields are not reproducible across solves "
        f"(per-mode relative difference {drift}). A value near 2 means the "
        "gauge flipped; _pin_gauge is not holding."
    )


def test_gauge_freedom_is_real_without_pinning(monkeypatch):
    """Sanity check on the test itself: without pinning the sign can move.

    Not a failure if it happens to agree -- ARPACK is random, not adversarial.
    The point is that ``_pin_gauge`` is what makes the result *guaranteed*,
    and this documents what it is protecting against.
    """
    monkeypatch.setattr(efde, "_pin_gauge", lambda E, H: None)
    backend = _backend()
    drift = _relative_field_difference(backend.solve(POINT).E, backend.solve(POINT).E)
    # Informational: report, do not assert a flip.
    print(f"\nunpinned per-mode relative field difference: {drift}")


@pytest.mark.xfail(
    reason="near-degenerate modes: ARPACK returns an arbitrary rotation "
    "within the degenerate subspace, which a per-mode phase cannot fix. "
    "Needs a coupled-pair dataset to exercise; see CLAUDE.md 5.6.",
    strict=False,
)
def test_degenerate_subspace_is_reproducible():
    raise NotImplementedError("requires the two-core cross section")
