"""The rapid adiabatic coupler platform and its trajectory extraction.

Two things went wrong while building demo 7, and both are cheap to pin:

* the ``x_offset`` axis step is not a free accuracy knob.  A RAC is defined by
  the *slope* of its outline in ``(w1, x)``, and a grid can only realise slopes
  that are ratios of its two axis steps, so an axis that is merely "fine" can
  still be unable to express the design;
* ``Re(O_01)`` has several roots in tilt over the first half of the device, and
  picking one by scan order takes the branch that makes the device worse.

Neither test solves a mode, so both are fast.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "examples"))

from dbeme.platforms import (  # noqa: E402
    RAC_GAP,
    RAC_W_BOT,
    RAC_W_EQUAL,
    RAC_W_TOP,
    rac_dataset_info,
)

os.environ.setdefault("RAC_CLAD", "air")
import demo_rapid_adiabatic_coupler as demo  # noqa: E402


# ------------------------------------------------------------------ platform


def test_offset_axis_expresses_the_design_slope():
    """The grid must be able to walk the outline the device actually has.

    Region III moves 502 nm laterally against 100 nm of width - a slope of 5.
    With a 5 nm offset step against the 2 nm width step near the symmetric
    point that is exactly representable; the original 20 nm step could only
    produce 2, 4 and 10, and the cached path zig-zagged around the design.
    """
    info = rac_dataset_info()()
    offsets = info.parameters["x_offset"]
    widths = info.parameters["w1"]

    offset_step = np.diff(offsets).min()
    fine_width_step = np.diff(widths[widths <= 400e-9]).min()
    assert offset_step == pytest.approx(5e-9)
    assert fine_width_step == pytest.approx(2e-9)

    slope = offset_step / fine_width_step
    assert 5.0 % slope == pytest.approx(0.0, abs=1e-9)


def test_offset_axis_step_is_configurable():
    coarse = rac_dataset_info(offset_step_nm=20)()
    assert np.diff(coarse.parameters["x_offset"]).min() == pytest.approx(20e-9)


def test_offset_axis_is_symmetric_about_zero():
    """A tilt has a sign, and folding to |x| would erase it."""
    offsets = rac_dataset_info()().parameters["x_offset"]
    assert offsets.min() < 0 < offsets.max()
    assert 0.0 in np.round(offsets, 12)


def test_air_and_oxide_are_different_datasets():
    """The fabricated device is unclad; the identity must say so.

    They are told apart by the cladding *material*, not by
    ``cladding_index`` - that field is the radiation cutoff, and both stacks
    share it because both sit on a buried oxide box.
    """
    oxide, air = rac_dataset_info()(), rac_dataset_info(clad="air")()
    assert oxide.get_cross_section().cladding.n(1.55e-6) == pytest.approx(
        1.4440, abs=1e-3
    )
    assert air.get_cross_section().cladding.n(1.55e-6) == pytest.approx(1.0)
    assert (
        oxide.get_cross_section().fingerprint()
        != air.get_cross_section().fingerprint()
    )


def test_air_clad_stack_is_vertically_asymmetric():
    """Air above, buried oxide below - as the real chip is."""
    air = rac_dataset_info(clad="air")().get_cross_section()
    assert air.cladding.n(1.55e-6) == pytest.approx(1.0)
    assert air.substrate.n(1.55e-6) == pytest.approx(1.4440, abs=1e-3)


def test_only_w1_is_swept_the_partner_guide_is_fixed():
    """Every extra axis costs two neighbour solves per path point."""
    info = rac_dataset_info()()
    assert tuple(info.parameter_names) == ("w1", "x_offset")
    cross_section = info.get_cross_section()
    assert cross_section.w2 == pytest.approx(RAC_W_TOP)
    assert cross_section.gap == pytest.approx(RAC_GAP)


# ------------------------------------------------------------------ schedule


def test_schedule_slope_is_the_tilt_and_is_centred():
    functions = demo.schedule(-2.5, length=10e-6)
    z = np.array([0.0, 5e-6, 10e-6])
    offset = functions["x_offset"](z)

    assert offset[1] == pytest.approx(0.0, abs=1e-15)
    assert offset[0] == pytest.approx(-offset[2])
    slope = (offset[2] - offset[0]) / 10e-6
    assert slope == pytest.approx(np.tan(np.radians(-2.5)))


def test_schedule_walks_the_published_widths():
    functions = demo.schedule(0.0)
    assert functions["w1"](0.0) == pytest.approx(RAC_W_BOT)
    assert functions["w1"](demo.LENGTH) == pytest.approx(RAC_W_EQUAL)


def test_tilted_schedule_integrates_the_slope():
    """theta is a slope, so the offset is its running integral."""
    functions = demo.tilted_schedule(lambda e: np.full_like(e, -2.0), length=10e-6)
    constant = demo.schedule(-2.0, length=10e-6)
    z = np.linspace(0, 10e-6, 11)
    assert functions["x_offset"](z) == pytest.approx(
        constant["x_offset"](z), abs=1e-12
    )


# ---------------------------------------------------------------- trajectory


def _signed_map(tilts, roots_per_position):
    """A coupling map whose columns have exactly the requested roots."""
    signed = np.zeros((len(tilts), len(roots_per_position)))
    for j, roots in enumerate(roots_per_position):
        column = np.ones(len(tilts))
        for root in roots:
            column = column * (tilts - root)
        signed[:, j] = column
    return signed


def test_continuation_follows_the_branch_not_the_scan_order():
    """The measured failure: two roots, and scan order takes the wrong one.

    Over the first half of region III ``Re(O_01)`` crosses zero twice.  The
    root near +1.7 degrees is not the rapid adiabatic condition - a positive
    tilt was the *worst* point of the whole constant-tilt scan - and following
    the branch from the symmetric end picks the physical one.
    """
    tilts = np.linspace(2.0, -5.0, 15)
    signed = _signed_map(tilts, [[1.7, -0.9], [-1.3], [-2.9]])
    theta = demo.rac_trajectory(np.array([480e-9, 430e-9, 380e-9]), tilts, signed)

    assert theta[2] == pytest.approx(-2.9, abs=0.05)
    assert theta[1] == pytest.approx(-1.3, abs=0.05)
    assert theta[0] == pytest.approx(-0.9, abs=0.05)
    assert theta[0] < 0.0, "took the spurious positive root"


def test_trajectory_is_nan_where_there_is_no_root():
    tilts = np.linspace(2.0, -5.0, 15)
    signed = np.ones((15, 2))  # never changes sign
    theta = demo.rac_trajectory(np.array([480e-9, 380e-9]), tilts, signed)
    assert np.all(np.isnan(theta))


def test_seed_uses_the_coupling_minimum_when_the_last_column_is_ambiguous():
    tilts = np.linspace(2.0, -5.0, 15)
    signed = _signed_map(tilts, [[-1.0], [1.5, -3.0]])
    magnitude = np.abs(signed)
    theta = demo.rac_trajectory(
        np.array([480e-9, 380e-9]), tilts, signed, magnitude
    )
    assert np.isfinite(theta).all()


# ------------------------------------------------------- radiation cutoff


def test_cutoff_is_the_substrate_not_the_cladding_when_they_differ():
    """An air-clad guide on a buried oxide box cuts off at 1.444, not 1.0.

    ``DatasetInfo.cladding_index`` is consumed as the *radiation cutoff* - the
    index above which a mode is bound - and for a vertically asymmetric stack
    that is the larger of cladding and substrate.  Reading the cladding alone
    counted three substrate-radiating box modes as guided, and the sanity gate
    then scored launches into window artefacts.
    """
    air = rac_dataset_info(clad="air")()
    cross_section = air.get_cross_section()

    assert cross_section.cladding.n(1.55e-6) == pytest.approx(1.0)
    assert air.cladding_index == pytest.approx(1.4440, abs=1e-3)
    assert air.cladding_index == pytest.approx(
        cross_section.cladding_index_at(1.55e-6)
    )


def test_symmetric_stacks_are_unaffected_by_the_cutoff_fix():
    """Every buried dataset in this project must keep the value it had."""
    from dbeme.platforms import (
        sacher_bilevel_dataset_info,
        sacher_coupler_dataset_info,
    )

    for info in (
        rac_dataset_info()(),
        sacher_coupler_dataset_info(1.55e-6)(),
        sacher_bilevel_dataset_info(1.55e-6)(),
    ):
        assert info.cladding_index == pytest.approx(1.4440, abs=1e-3)
