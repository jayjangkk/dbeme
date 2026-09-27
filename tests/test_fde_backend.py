"""Physics sanity checks for the emepy FDE backend.

Run with:  python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402

WL = 1.55e-6


def make_backend(num_modes=4, mesh=90):
    cs = FullEtchStrip(thickness=220e-9, core_index=3.4757, clad_index=1.444)
    return EmepyFDE(
        cross_section=cs,
        num_modes=num_modes,
        wavelength=WL,
        window=(2.2e-6, -1.1e-6, 1.1e-6),
        mesh=mesh,
    )


@pytest.fixture(scope="module")
def backend():
    return make_backend()


def test_500nm_strip_mode_spectrum(backend):
    """A 500 nm x 220 nm SiO2-buried Si strip: TE0 ~ 2.45, TM0 ~ 1.79.

    Reference values come from a mesh- and window-convergence study of this
    same solver (20 nm cells, 8 x 4 um window), and agree with emepy's own
    built-in rectangle cross section to 1e-3.
    """
    md = backend.solve((500e-9, 0.0))
    neff = np.real(md.neff)

    assert np.all(np.diff(neff) <= 0), f"modes not sorted by neff: {neff}"
    assert 2.40 < neff[0] < 2.50, f"TE0 neff out of range: {neff[0]}"
    assert 1.75 < neff[1] < 1.85, f"TM0 neff out of range: {neff[1]}"
    assert md.TE_pol[0] > 0.9, "fundamental should be TE"
    assert md.TE_pol[1] < 0.2, "second mode should be TM"
    # Everything from the fourth mode on is unguided at this width.
    assert neff[3] < 1.444, f"unexpected guided mode: {neff[3]}"


def test_shipped_baseline_is_unchanged():
    """Pin the reference numbers the dataset and the README quote.

    Solved at the shipped dataset's own window and mesh, so this catches a
    change in the cross section, the mesh, the material indices or the solver
    that would silently move every result.
    """
    reference = EmepyFDE(
        cross_section=FullEtchStrip(
            thickness=220e-9, core_index=3.4757, clad_index=1.444
        ),
        num_modes=3,
        wavelength=WL,
        window=(1.6e-6, -0.8e-6, 0.8e-6),
        mesh=160,
    )
    neff = np.real(reference.solve((500e-9, 0.0)).neff)
    assert abs(neff[0] - 2.449) < 5e-3, f"TE0 baseline moved: {neff[0]}"
    assert abs(neff[1] - 1.788) < 5e-3, f"TM0 baseline moved: {neff[1]}"


def test_wider_waveguide_is_multimode(backend):
    """A 1 um strip supports TE0 and TE1; TE1 sits below TE0."""
    md = backend.solve((1.0e-6, 0.0))
    neff = np.real(md.neff)
    te = md.TE_pol

    te_modes = [neff[i] for i in range(len(neff)) if te[i] > 0.8 and neff[i] > 1.444]
    assert len(te_modes) >= 2, f"expected >=2 guided TE modes, got {te_modes}"
    assert te_modes[0] > te_modes[1]


def test_neff_increases_with_width(backend):
    """Confinement, hence neff, grows monotonically with core width."""
    widths = [400e-9, 500e-9, 700e-9, 900e-9]
    neffs = [np.real(backend.solve((w, 0.0)).neff[0]) for w in widths]
    assert all(b > a for a, b in zip(neffs, neffs[1:])), neffs


def test_bend_raises_neff(backend):
    """The conformal map shifts the mode outwards and raises its neff.

    For a bend of radius R the effective index of the equivalent straight guide
    satisfies neff_bend ~ neff_straight * (1 + <x>/R) with <x> > 0, so a
    positive curvature must increase neff.
    """
    straight = np.real(backend.solve((500e-9, 0.0)).neff[0])
    bent = np.real(backend.solve((500e-9, 100000.0)).neff[0])  # R = 10 um
    assert bent > straight, (straight, bent)
    # 10 um radius on a 500 nm guide: a sub-percent shift, not a blow-up.
    assert (bent - straight) / straight < 0.05


def test_grid_is_shared_between_parameter_points(backend):
    a = backend.solve((500e-9, 0.0))
    b = backend.solve((1.2e-6, 50000.0))
    assert np.allclose(a.x, b.x)
    assert np.allclose(a.y, b.y)


def test_self_overlap_is_identity(backend):
    """<E_i, H_j> of a point with itself must be the identity on forward modes.

    This is the normalisation and orthogonality check that the whole EME
    interface calculation rests on.
    """
    md = backend.solve((900e-9, 0.0))
    x, y, neff, te, E, H = assemble([md], backend.num_modes, prop_axis=2)
    O = overlap_matrix(E[0], H[0], x, y, prop_axis=2)

    N = backend.num_modes
    forward = O[:N, :N]
    assert np.allclose(np.diag(forward), 1.0, atol=2e-2), np.diag(forward)

    off = forward - np.diag(np.diag(forward))
    assert np.max(np.abs(off)) < 5e-2, np.max(np.abs(off))


def test_backward_modes_are_mirror_of_forward(backend):
    md = backend.solve((700e-9, 0.0))
    x, y, neff, te, E, H = assemble([md], backend.num_modes, prop_axis=2)
    N = backend.num_modes

    assert np.allclose(neff[0, N:], -neff[0, :N])
    assert np.allclose(te[0, N:], te[0, :N])
    assert np.allclose(E[0, N:], np.conjugate(E[0, :N]))
    assert np.allclose(H[0, N:], -np.conjugate(H[0, :N]))


def test_adjacent_widths_overlap_strongly(backend):
    """Neighbouring grid points must be well coupled, or the grid is too coarse.

    The mode-tracking step in ``Geometry`` links modes across sections by
    requiring |overlap| > 0.5, so this is the condition that makes a 20 nm
    width step a usable dataset spacing.
    """
    a = backend.solve((900e-9, 0.0))
    b = backend.solve((920e-9, 0.0))
    x, y, neff, te, E, H = assemble([a, b], backend.num_modes, prop_axis=2)
    O = overlap_matrix(E[0], H[1], x, y, prop_axis=2)

    N = backend.num_modes
    # Guided modes only: radiation modes are grid artefacts and do not track.
    guided = [i for i in range(N) if np.real(neff[0, i]) > 1.444]
    assert guided
    for i in guided:
        assert abs(O[i, i]) > 0.9, (i, abs(O[i, i]))
