"""Stage 1 of the S-matrix reformulation: the direct route must match.

``SingleEME`` builds a transfer matrix per interface, cascades those, and
converts to scattering form at the end.  That inherits transfer-matrix
instability, which is fatal for the two things this reformulation is for:

* **plasmonics** - the transfer form propagates the backward block as
  ``exp(-i beta dz)``, i.e. ``exp(+k0 n'' dz)`` for a lossy mode, which grows.
  A Au/air MIM SPP (``n'' ~ 0.08``) reaches e^2 over a 600 nm taper; the
  evanescent modes a sub-wavelength converter needs (``n'' ~ 1-2``) reach e^48.
* **rings** - a 5 um ring is 31 um of circumference, so any growing exponential
  is cascaded to destruction.

Stage 1 adds the scattering route *alongside* the transfer one and proves they
agree, so the default can be switched later without moving anything published.
These tests are that proof.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "examples"))

from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.fde.materials import silica, silicon  # noqa: E402
from dbeme.geometry.geometry import Geometry  # noqa: E402
from dbeme.matrix_calculation_tool import (  # noqa: E402
    _redheffer_star_product,
)
from dbeme.propagator.single_propagator.single_eme import (  # noqa: E402
    SingleEME,
)

WL = 1.55e-6

#: The two routes are algebraically identical, so they may differ only by
#: arithmetic.  The transfer route stores its matrices as complex64, which puts
#: the floor near 1e-7; the scattering route stays in complex128.
AGREEMENT = 1e-6


class _Stub(Geometry):
    def __init__(self, output_data):
        self._is_composite_geometry = False
        self._verbose = False
        self.output_data = output_data


def _output_data(mode_data_list, num_modes, length=2e-6, lossless=True):
    x, y, neff, te, E, H = assemble(
        mode_data_list, num_modes, prop_axis=2, lossless=lossless
    )
    sections = len(mode_data_list)
    width = 2 * num_modes
    ab = np.zeros((sections - 1, width, width), dtype=complex)
    ba = np.zeros((sections - 1, width, width), dtype=complex)
    for i in range(sections - 1):
        ab[i] = overlap_matrix(E[i], H[i + 1], x, y, 2)
        ba[i] = overlap_matrix(E[i + 1], H[i], x, y, 2)
    return {
        "neff": neff,
        "beta": 2 * np.pi / WL * neff,
        "overlap_ab": ab,
        "overlap_ba": ba,
        "delta_zs": np.full(sections - 1, length),
        "EME_delta_zs": np.full(sections - 1, length),
        "radiation_mode_mask": np.zeros((sections, width), dtype=bool),
    }


def _synthetic(sections=3, modes=2, loss=0.0):
    """An ``output_data`` with a chosen amount of loss, no solving involved."""
    width = 2 * modes
    neff = np.tile(
        np.array([2.4, 1.8, -2.4, -1.8][:width], dtype=complex), (sections, 1)
    )
    neff[:, :modes] += 1j * loss
    neff[:, modes:] -= 1j * loss
    overlap = np.tile(np.eye(width, dtype=complex), (sections - 1, 1, 1))
    return {
        "neff": neff,
        "beta": 2 * np.pi / WL * neff,
        "overlap_ab": overlap.copy(),
        "overlap_ba": overlap.copy(),
        "delta_zs": np.full(sections - 1, 2e-6),
        "EME_delta_zs": np.full(sections - 1, 2e-6),
        "radiation_mode_mask": np.zeros((sections, width), dtype=bool),
    }


def _both(output_data, **kwargs):
    a = SingleEME(_Stub(output_data), **kwargs)
    a.calc_Smatrix(method="transfer")
    b = SingleEME(_Stub(output_data), **kwargs)
    b.calc_Smatrix(method="direct")
    return np.asarray(a.smatrix), np.asarray(b.smatrix)


def _cascade(smatrix):
    lumped = smatrix[0]
    for step in smatrix[1:]:
        lumped = _redheffer_star_product(lumped, step)
    return lumped


@pytest.fixture(scope="module")
def taper():
    """A real four-section lossless taper, 6 modes."""
    backend = EmepyFDE(
        cross_section=FullEtchStrip(
            thickness=0.22e-6,
            core=silicon(out_of_range="raise"),
            cladding=silica(out_of_range="raise"),
        ),
        num_modes=6,
        wavelength=WL,
        window=(2.0e-6, -0.6e-6, 0.6e-6),
        mesh=160,
    )
    modes = [backend.solve((w, 0.0)) for w in (0.50e-6, 0.45e-6, 0.40e-6, 0.36e-6)]
    return _output_data(modes, 6)


# --------------------------------------------------------------- the gate


def test_the_two_routes_agree_on_a_real_taper(taper):
    """Stage 1's gate: nothing published may move when the default switches."""
    transfer, direct = _both(taper, force_unitary=False)
    assert transfer.shape == direct.shape
    assert np.abs(transfer - direct).max() < AGREEMENT


def test_the_two_routes_agree_block_by_block(taper):
    """A sign slip in one block would still leave the total small."""
    transfer, direct = _both(taper, force_unitary=False)
    modes = transfer.shape[1] // 2
    for name, rows, cols in (
        ("T_forward", slice(0, modes), slice(0, modes)),
        ("R_right", slice(0, modes), slice(modes, None)),
        ("R_left", slice(modes, None), slice(0, modes)),
        ("T_backward", slice(modes, None), slice(modes, None)),
    ):
        worst = np.abs(transfer[:, rows, cols] - direct[:, rows, cols]).max()
        assert worst < AGREEMENT, f"{name} differs by {worst:.2e}"


def test_the_two_routes_agree_after_cascading(taper):
    """Per-section agreement is not the same as device-level agreement."""
    transfer, direct = _both(taper, force_unitary=False)
    assert np.abs(_cascade(transfer) - _cascade(direct)).max() < AGREEMENT


@pytest.mark.parametrize("modes", [1, 2, 4])
@pytest.mark.parametrize("sections", [2, 3, 5])
def test_the_two_routes_agree_across_shapes(modes, sections):
    transfer, direct = _both(_synthetic(sections, modes), force_unitary=False)
    assert np.abs(transfer - direct).max() < AGREEMENT


def test_the_projection_flags_are_applied_by_both_routes():
    data = _synthetic()
    for flags in ({"force_unitary": True}, {"force_passive": True}):
        transfer, direct = _both(data, **flags)
        assert np.abs(transfer - direct).max() < AGREEMENT


# ------------------------------------------------------------ the algebra


def test_interface_smatrix_is_the_scattering_blocks(taper):
    """``[[T12, -R21], [R12, T21]]`` - the inv(T21) round trip cancels."""
    eme = SingleEME(_Stub(taper), force_unitary=False)
    modes = eme.mode_count
    T12 = eme._calc_transmission_matrix(eme.overlap_forward_ab, eme.overlap_forward_ba)
    T21 = eme._calc_transmission_matrix(eme.overlap_forward_ba, eme.overlap_forward_ab)
    R12 = eme._calc_reflection_matrix(
        eme.overlap_forward_ab, eme.overlap_forward_ba, T12)
    R21 = eme._calc_reflection_matrix(
        eme.overlap_forward_ba, eme.overlap_forward_ab, T21)

    built = eme._calc_interface_Smatrix()
    assert built[:, :modes, :modes] == pytest.approx(T12)
    assert built[:, :modes, modes:] == pytest.approx(-R21)
    assert built[:, modes:, :modes] == pytest.approx(R12)
    assert built[:, modes:, modes:] == pytest.approx(T21)


def test_propagation_smatrix_uses_the_same_block_twice():
    """The structural reason the scattering route cannot blow up.

    A transfer matrix carries ``exp(+i beta dz)`` forward and
    ``exp(-i beta dz)`` backward.  In scattering form both waves travel forward
    along their own direction, so both blocks are the *decaying* one.
    """
    eme = SingleEME(_Stub(_synthetic(modes=2, loss=0.05)), force_unitary=False)
    modes = eme.mode_count
    block = eme._calc_propagation_Smatrix()
    forward = block[:, :modes, :modes]
    backward = block[:, modes:, modes:]
    assert forward == pytest.approx(backward)
    assert np.abs(block[:, :modes, modes:]).max() == 0.0
    assert np.abs(block[:, modes:, :modes]).max() == 0.0


def test_propagation_never_grows_for_a_lossy_mode():
    """The transfer route's backward block does exactly this, and grows."""
    for loss in (0.0, 0.01, 0.1, 1.0, 2.0):
        eme = SingleEME(_Stub(_synthetic(modes=2, loss=loss)), force_unitary=False)
        block = eme._calc_propagation_Smatrix()
        assert np.abs(block).max() <= 1.0 + 1e-12, (
            f"propagation amplifies at Im(n_eff) = {loss}"
        )


def test_the_transfer_route_does_grow_for_a_lossy_mode():
    """Pin the failure being fixed, so it cannot be quietly reintroduced."""
    eme = SingleEME(_Stub(_synthetic(modes=2, loss=1.0)), force_unitary=False)
    transfer = eme._calc_phase_propagation_Tmatrix()
    assert np.abs(transfer).max() > 10.0


# ----------------------------------------------------------------- wiring


class _Lossy(_Stub):
    """A geometry whose data source declares its backend lossy."""

    class _Data:
        @staticmethod
        def _is_lossless():
            return False

    def __init__(self, output_data):
        super().__init__(output_data)
        self.data = self._Data()


def test_the_default_route_is_auto():
    assert SingleEME.SMATRIX_METHOD == "auto"


def test_auto_keeps_a_lossless_geometry_on_the_transfer_route():
    """Stage 4's promise: nothing published moves.

    Every dataset behind reports 01-05 is lossless by construction, and at an
    interface whose mode-matching matrix is near singular the two routes are
    not numerically identical (~2e-5 in the guided block on the README taper).
    So lossless data stays on the route it was published with.
    """
    eme = SingleEME(_Stub(_synthetic()), force_unitary=False)
    assert eme.resolve_method() == "transfer"
    eme.calc_Smatrix()
    assert eme._is_tmatrix_calculated is True


def test_auto_sends_a_lossy_geometry_down_the_direct_route():
    """The transfer route cannot carry a lossy basis at all - its backward
    propagation block grows as exp(+k0 n'' dz)."""
    eme = SingleEME(_Lossy(_synthetic(loss=0.1)), force_unitary=False)
    assert eme.resolve_method() == "direct"
    eme.calc_Smatrix()
    assert eme._is_smatrix_calculated is True
    assert eme.tmatrix is None


def test_an_explicit_method_overrides_auto():
    lossless = SingleEME(_Stub(_synthetic()), force_unitary=False)
    assert lossless.resolve_method("direct") == "direct"
    lossy = SingleEME(_Lossy(_synthetic(loss=0.1)), force_unitary=False)
    assert lossy.resolve_method("transfer") == "transfer"


def test_a_geometry_without_a_data_source_counts_as_lossless():
    """Stubs and test doubles must keep the historical behaviour."""
    eme = SingleEME(_Stub(_synthetic()), force_unitary=False)
    assert eme._lossless is True


def test_rescaling_a_lossy_geometry_never_builds_a_transfer_matrix():
    """``change_strucutre_length`` used to build one unconditionally, and for a
    lossy basis that is the matrix that overflows."""
    eme = SingleEME(_Lossy(_synthetic(sections=4, loss=0.5)), force_unitary=False)
    eme.change_strucutre_length(5e-6)
    assert eme.tmatrix is None
    assert np.abs(np.asarray(eme.smatrix)).max() <= 1.0 + 1e-9


def test_an_unknown_method_is_rejected():
    eme = SingleEME(_Stub(_synthetic()), force_unitary=False)
    with pytest.raises(ValueError, match="transfer"):
        eme.calc_Smatrix(method="pseudo")


def test_the_direct_route_never_builds_a_transfer_matrix():
    """That is the entire point - forming T is what goes unstable."""
    eme = SingleEME(_Stub(_synthetic()), force_unitary=False)
    eme.calc_Smatrix(method="direct")
    assert eme._is_smatrix_calculated is True
    assert eme.tmatrix is None
    assert eme._is_tmatrix_calculated is False


def test_the_direct_route_reports_section_lengths():
    """``_lengths_per_matrix`` must keep its meaning for length rescaling."""
    data = _synthetic(sections=4)
    eme = SingleEME(_Stub(data), force_unitary=False)
    eme.calc_Smatrix(method="direct")
    assert eme._lengths_per_matrix.shape == (2 * 4 - 2,)
    assert eme._lengths_per_matrix[0::2] == pytest.approx(data["EME_delta_zs"])
    assert eme._lengths_per_matrix[1::2] == pytest.approx(0.0)


def test_length_rescaling_agrees_between_routes(taper):
    """Every length sweep in ``examples/`` goes through this path.

    If it kept using transfer matrices, switching the default would leave the
    length sweeps - which is most of what this project reports - on the
    unstable route.
    """
    eme = SingleEME(_Stub(taper), force_unitary=False)
    for new_length in (2e-6, 6e-6, 20e-6, 60e-6):
        transfer = eme._find_Smatrix_new_length(new_length, method="transfer")
        direct = eme._find_Smatrix_new_length(new_length, method="direct")
        assert np.abs(np.asarray(transfer) - np.asarray(direct)).max() < AGREEMENT
        assert np.abs(_cascade(np.asarray(transfer))
                      - _cascade(np.asarray(direct))).max() < AGREEMENT


def test_length_rescaling_rejects_an_unknown_method(taper):
    eme = SingleEME(_Stub(taper), force_unitary=False)
    with pytest.raises(ValueError, match="transfer"):
        eme._find_Smatrix_new_length(5e-6, method="pseudo")


def test_rescaled_propagation_never_grows_for_a_lossy_mode():
    """The rescaled path must inherit the boundedness, not just the base one."""
    eme = SingleEME(_Stub(_synthetic(modes=2, loss=1.0)), force_unitary=False)
    for ratio in (0.5, 1.0, 5.0, 50.0):
        block = eme._calc_propagation_Smatrix(length_ratio=ratio)
        assert np.abs(block).max() <= 1.0 + 1e-12


# ------------------------------------- Stage 2: the analytic lossy answer


#: Above this, exp(-k0 n'' L) underflows to zero over the test length and the
#: comparison becomes vacuous rather than demanding.
@pytest.mark.parametrize("loss", [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.08, 0.3])
def test_a_uniform_lossy_guide_matches_the_closed_form(loss):
    """``|T| = exp(-k0 n'' L)`` exactly, for a guide that is uniform in z.

    With identity overlaps every interface is transparent, so the device is
    pure propagation and the answer is known in closed form.  0.08 is a Au/air
    MIM plasmonic mode.

    The length is taken from the data rather than assumed: getting it wrong
    gives a result that is the right answer *squared*, which looks plausible.
    """
    data = _synthetic(sections=6, modes=1, loss=loss)
    total_length = float(np.sum(data["EME_delta_zs"]))
    expected = np.exp(-(2 * np.pi / WL) * loss * total_length)
    assert expected > 1e-300, "test length makes the comparison vacuous"

    eme = SingleEME(_Stub(data), force_unitary=False)
    eme.calc_Smatrix(method="direct")
    lumped = _cascade(np.asarray(eme.smatrix))
    launch = np.zeros(2 * eme.mode_count, dtype=complex)
    launch[0] = 1.0
    assert abs((lumped @ launch)[0]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("loss", [0.0, 0.08, 1.0, 5.0])
def test_a_passive_structure_never_produces_gain(loss):
    """No amount of loss may make the output energy exceed the input."""
    eme = SingleEME(_Stub(_synthetic(sections=6, modes=2, loss=loss)),
                    force_unitary=False)
    eme.calc_Smatrix(method="direct")
    lumped = _cascade(np.asarray(eme.smatrix))
    launch = np.zeros(2 * eme.mode_count, dtype=complex)
    launch[0] = 1.0
    assert float(np.sum(np.abs(lumped @ launch) ** 2)) <= 1.0 + 1e-9


def test_loss_grows_monotonically_with_length():
    data = _synthetic(sections=6, modes=1, loss=0.1)
    eme = SingleEME(_Stub(data), force_unitary=False)
    previous = None
    for new_length in (1e-6, 5e-6, 20e-6, 100e-6):
        smatrix = np.asarray(
            eme._find_Smatrix_new_length(new_length, method="direct"))
        launch = np.zeros(2 * eme.mode_count, dtype=complex)
        launch[0] = 1.0
        value = abs((_cascade(smatrix) @ launch)[0])
        if previous is not None:
            assert value < previous, "a longer lossy guide must transmit less"
        previous = value


# ----------------------------- Stage 3: the conditioning cutoff, in place


def test_the_cutoff_does_not_touch_a_lossless_basis(taper):
    """Measured cond(O_ab + O_ba^T) is 1.2-35 on guided bases.

    That is a smallest-to-largest singular value ratio of 0.029, thirty times
    above the 1e-3 cutoff, so nothing is discarded and the scattering route
    still reproduces the transfer route.  The equivalence gate above would
    catch a regression, but state the margin explicitly.
    """
    eme = SingleEME(_Stub(taper), force_unitary=False)
    matrix = (eme.overlap_forward_ab
              + np.transpose(eme.overlap_forward_ba, (0, 2, 1)))
    for section in matrix:
        assert np.linalg.cond(section) < 1.0 / (10 * SingleEME.INTERFACE_RCOND)


def test_the_cutoff_is_off_for_the_transfer_route(taper):
    """Truncating before an inversion is worse than not truncating.

    ``_calc_interface_Tmatrix`` forms ``inv(T21)``; a truncated ``T21`` is rank
    deficient, and the measured output energy went from 2.8e3 to 2.7e8.  The
    transfer route must therefore pass ``rcond=None``.
    """
    eme = SingleEME(_Stub(taper), force_unitary=False)
    exact = eme._calc_transmission_matrix(
        eme.overlap_forward_ab, eme.overlap_forward_ba)
    default = eme._calc_transmission_matrix(
        eme.overlap_forward_ab, eme.overlap_forward_ba, None)
    assert np.abs(exact - default).max() == 0.0


# --------------------------------------------- interface projection side


def _T(form, oab, oba):
    from types import SimpleNamespace

    from dbeme.propagator.single_propagator.single_eme import SingleEME

    stub = SimpleNamespace(INTERFACE_PROJECTION=form)
    return SingleEME._calc_transmission_matrix(stub, oab, oba)


def test_projection_sides_coincide_on_a_perfect_interface():
    eye = np.eye(3, dtype=complex)[None]
    for form in ("input", "output"):
        assert np.allclose(_T(form, eye, eye), eye)


def test_output_side_projection_loses_the_unrepresented_mismatch():
    """One mode each side, overlap 0.99: the field the basis cannot hold is
    1 - 0.99^2.  Output-side projection loses it; input-side projection gains
    it (1/0.99^2).  The plasmonic taper of report 12 is forty of these."""
    o = np.array([[[0.99 + 0j]]])
    t_out = abs(_T("output", o, o)[0, 0, 0]) ** 2
    t_in = abs(_T("input", o, o)[0, 0, 0]) ** 2
    assert t_out == pytest.approx(0.99 ** 2)
    assert t_in == pytest.approx(1 / 0.99 ** 2)
    assert t_out < 1 < t_in


def test_output_side_projection_is_the_documented_matrix_form():
    rng = np.random.default_rng(3)
    oab = np.eye(4) + 0.05 * rng.standard_normal((4, 4)) + 0.01j * rng.standard_normal((4, 4))
    oba = np.eye(4) + 0.05 * rng.standard_normal((4, 4)) + 0.01j * rng.standard_normal((4, 4))
    expected = 2 * oab.T @ np.linalg.inv(oab.T + oba) @ oba
    assert np.allclose(_T("output", oab[None], oba[None])[0], expected)
    assert np.allclose(_T("input", oab[None], oba[None])[0], 2 * np.linalg.inv(oab + oba.T))


def test_auto_projection_follows_the_basis():
    """Lossless: upstream's input-side form (the transfer route needs its
    T21 invertible); lossy: the bounded output-side form."""
    from types import SimpleNamespace

    from dbeme.propagator.single_propagator.single_eme import SingleEME

    o = np.array([[[0.99 + 0j]]])
    lossless = SimpleNamespace(INTERFACE_PROJECTION="auto", _lossless=True)
    lossy = SimpleNamespace(INTERFACE_PROJECTION="auto", _lossless=False)
    assert abs(SingleEME._calc_transmission_matrix(lossless, o, o)[0, 0, 0]) ** 2 == pytest.approx(1 / 0.99 ** 2)
    assert abs(SingleEME._calc_transmission_matrix(lossy, o, o)[0, 0, 0]) ** 2 == pytest.approx(0.99 ** 2)
    assert SingleEME.INTERFACE_PROJECTION == "auto"


def test_an_unknown_projection_is_rejected():
    with pytest.raises(ValueError, match="INTERFACE_PROJECTION"):
        _T("sideways", np.eye(2)[None], np.eye(2)[None])
