"""``MultiEME`` construction, and where its flags end up.

``MultiEME.__init__`` passed ``force_unitar=`` to a parent whose signature says
``force_unitary`` and which takes no ``**kwargs``, so building one raised
``TypeError`` immediately.  Nothing in the demos or the rest of the suite
constructs a ``MultiEME``, which is why it went unnoticed - so the point of
these tests is simply that the class can be built at all, and that both flags
arrive where they are meant to.

The geometries here are stubs.  The flags are plumbing, and driving a real
dataset through a cascade would test the physics instead of the wiring while
making the suite minutes slower.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbeme.geometry.composite_geometry import CompositeGeometry  # noqa: E402
from dbeme.geometry.geometry import Geometry  # noqa: E402
from dbeme.propagator.multi_propagator.multi_eme import MultiEME  # noqa: E402


class _StubGeometry(Geometry):
    """The smallest thing ``Propagator`` and ``SingleEME`` will accept.

    Deliberately does not call ``Geometry.__init__``: that wants a real
    ``DataUpdater``, and none of what it sets up is involved in passing two
    boolean flags down a constructor chain.
    """

    def __init__(self, sections=3, modes=2, loss=0.0):
        self._is_composite_geometry = False
        self._verbose = False
        width = 2 * modes
        neff = np.tile(
            np.array([2.4, 1.8, -2.4, -1.8][:width], dtype=complex), (sections, 1)
        )
        neff[:, 0] += 1j * loss
        self.output_data = {
            "neff": neff,
            "beta": 2 * np.pi / 1.55e-6 * neff,
            "overlap_ab": np.tile(np.eye(width, dtype=complex), (sections - 1, 1, 1)),
            "overlap_ba": np.tile(np.eye(width, dtype=complex), (sections - 1, 1, 1)),
            "delta_zs": np.full(sections - 1, 1e-6),
            "radiation_mode_mask": np.zeros((sections, width), dtype=bool),
        }

    def calc_simulation_parameters(self):  # pragma: no cover - never reached
        raise NotImplementedError


def _composite(count=2, **kwargs):
    return CompositeGeometry([_StubGeometry(**kwargs) for _ in range(count)])


# ------------------------------------------------------------- construction


def test_multi_eme_can_be_constructed():
    """The regression: this raised ``TypeError`` on the keyword name alone."""
    assert MultiEME(_composite()) is not None


def test_construction_builds_one_child_propagator_per_geometry():
    composite = _composite(count=3)
    propagator = MultiEME(composite)
    assert len(propagator.propagators) == 3
    assert len(propagator._geometries) == 3


# ------------------------------------------------------------------- flags


@pytest.mark.parametrize("force_unitary", [True, False])
def test_force_unitary_reaches_the_parent(force_unitary):
    propagator = MultiEME(_composite(), force_unitary=force_unitary)
    assert propagator._force_unitary is force_unitary


@pytest.mark.parametrize("force_passive", [True, False])
def test_force_passive_reaches_the_parent(force_passive):
    propagator = MultiEME(_composite(), force_passive=force_passive)
    assert propagator._force_passive is force_passive


def test_both_flags_reach_every_child_propagator():
    """The parent's flags are not the ones that do the work.

    ``MultiEME`` delegates to a ``SingleEME`` per geometry, and it is those
    that actually build scattering matrices, so the flags have to arrive there
    too - not only on the parent.
    """
    propagator = MultiEME(_composite(count=3), force_passive=True, force_unitary=True)
    for child in propagator.propagators:
        assert child._force_unitary is True
        assert child._force_passive is True


def test_the_flags_are_not_transposed():
    """Two adjacent booleans are easy to swap; pin them independently."""
    propagator = MultiEME(_composite(), force_passive=True, force_unitary=False)
    assert propagator._force_passive is True
    assert propagator._force_unitary is False
    for child in propagator.propagators:
        assert child._force_passive is True
        assert child._force_unitary is False


# ------------------------------------------------- the lossy-projection guard


def test_the_lossy_guard_reaches_multi_eme_through_its_children():
    """``MultiPropagator.__init__`` does not call ``Propagator.__init__``.

    So ``_reject_unitary_projection_on_lossy_modes`` is never invoked on the
    parent.  It does not need to be: every geometry is wrapped in a
    ``SingleEME``, which does call it, so a lossy geometry anywhere in the
    composite still refuses the projection.  Verified here rather than assumed.

    Adding the guard to ``MultiPropagator`` directly would also mean writing a
    second version of it: ``CompositeGeometry.output_data`` is a *list* of
    per-geometry dicts, not one dict, which is the reason that constructor
    cannot call its parent in the first place.
    """
    composite = CompositeGeometry([_StubGeometry(), _StubGeometry(loss=1e-4)])
    with pytest.raises(ValueError, match="force_unitary=True with lossy modes"):
        MultiEME(composite, force_unitary=True)


def test_a_lossy_composite_is_fine_without_the_projection():
    composite = CompositeGeometry([_StubGeometry(), _StubGeometry(loss=1e-4)])
    propagator = MultiEME(composite, force_unitary=False)
    assert propagator._force_unitary is False


def test_an_all_lossless_composite_still_allows_the_projection():
    propagator = MultiEME(_composite(), force_unitary=True)
    assert propagator._force_unitary is True
