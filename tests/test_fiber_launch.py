"""The fibre launch: a Gaussian projected onto a cross section's modes.

Analytic checks only - no mode solving.  A "mode" that *is* the beam must
take all of it, the paper's scalar overlap of the beam with itself is one,
the conjugated power coupling agrees, and the TM field's H has the sign that
makes its Poynting vector point forward.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.fde.assemble import overlap_matrix  # noqa: E402
from dbeme.propagator.fiber import (  # noqa: E402
    ETA0,
    gaussian_fields,
    gaussian_launch,
    paper_overlap,
    power_coupling,
)

W0 = 2.0e-6
X = np.linspace(-9e-6, 9e-6, 361)
Y = np.linspace(-9e-6, 9e-6, 361)


def as_mode(E, H):
    """Normalise a beam to <E, H> = 1 on this grid, as ``assemble`` does."""
    p = overlap_matrix(E, H, X, Y)[0, 0]
    return E / np.sqrt(p), H / np.sqrt(p)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_power_points_forward(pol):
    E, H = gaussian_fields(X, Y, W0, pol=pol, n_medium=1.45)
    p = overlap_matrix(E, H, X, Y)[0, 0]
    assert np.real(p) == pytest.approx(1.0, rel=2e-3)     # unit power on the plane


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_a_mode_equal_to_the_beam_takes_all_of_it(pol):
    Em, Hm = as_mode(*gaussian_fields(X, Y, W0, pol=pol, n_medium=1.45))
    out = gaussian_launch(X, Y, Em, Hm, W0, pol=pol, n_medium=1.45)
    # forward amplitude = sqrt(beam power on this grid / its power on the plane)
    assert abs(out["a"][0]) ** 2 == pytest.approx(out["window_power"], rel=1e-9)
    assert out["s"][0] == pytest.approx(out["d"][0], rel=1e-9)


def test_impedance_mismatch_is_the_fresnel_transmission():
    """Same shape, index 1 -> 2: |t|^2 = 1 - ((n1 - n2) / (n1 + n2))^2 = 8/9.

    The naive expansion coefficient (s + d) / 2 would give 9/8 here."""
    Em, Hm = as_mode(*gaussian_fields(X, Y, W0, pol="TE", n_medium=2.0))
    out = gaussian_launch(X, Y, Em, Hm, W0, pol="TE", n_medium=1.0)
    t2 = abs(out["a"][0]) ** 2 / out["window_power"]
    assert t2 == pytest.approx(8.0 / 9.0, rel=1e-6)
    assert abs(0.5 * (out["s"][0] + out["d"][0])) ** 2 / out["window_power"] == pytest.approx(9.0 / 8.0, rel=1e-6)


def test_orthogonal_polarisation_takes_nothing():
    Em, Hm = as_mode(*gaussian_fields(X, Y, W0, pol="TM", n_medium=1.45))
    out = gaussian_launch(X, Y, Em, Hm, W0, pol="TE", n_medium=1.45)
    assert abs(out["a"][0]) < 1e-12


def test_paper_overlap_of_the_beam_with_itself_is_one():
    E, _ = gaussian_fields(X, Y, W0, pol="TE")
    assert paper_overlap(X, Y, E[0], W0, pol="TE") == pytest.approx(1.0, rel=1e-12)
    # and a shifted beam gives the textbook exp(-d^2 / w0^2)
    E2, _ = gaussian_fields(X, Y, W0, center=(1e-6, 0.0), pol="TE")
    assert paper_overlap(X, Y, E2[0], W0, pol="TE") == pytest.approx(np.exp(-1.0 / 4.0), rel=1e-3)


def test_power_coupling_matches_the_scalar_overlap_for_matched_beams():
    E, H = gaussian_fields(X, Y, W0, center=(1e-6, 0.0), pol="TE", n_medium=1.45)
    eta = power_coupling(X, Y, E[0], H[0], W0, pol="TE", n_medium=1.45)
    assert eta == pytest.approx(np.exp(-1.0 / 4.0), rel=2e-3)


def test_femwell_units():
    """SI fields (femwell) need z_units = eta0 to give the same answer."""
    E, H = gaussian_fields(X, Y, W0, pol="TE", n_medium=1.45, z_units=ETA0)
    assert np.real(overlap_matrix(E, H, X, Y)[0, 0]) == pytest.approx(1.0, rel=2e-3)
    assert np.max(np.abs(H)) / np.max(np.abs(E)) == pytest.approx(1.45 / ETA0)
