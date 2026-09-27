"""`dbeme/circuit/cascade_jnp.py` against the production cascade on a shipped dataset.

The linear taper of `examples/demo_linear_taper.py` (0.5 -> 1.2 um over 10 um)
on `datasets/Si_fulletch_220nm`: every grid point it visits is cached, so
building it solves nothing - the mode solver is patched to raise, so a test
that would solve (and write to the dataset) fails instead.  Asserts:

* the ``jnp`` cascade equals `dbeme.validation.lumped_smatrix` on the
  scattering route at the path's own section lengths, to 1e-10;
* ``jax.grad`` of a transmitted power with respect to one section length
  matches a central finite difference.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATASET = os.path.join(ROOT, "datasets", "Si_fulletch_220nm")


@pytest.fixture(scope="module")
def path():
    pytest.importorskip("jax")
    from dbeme import DataUpdater, LinearTaper
    from dbeme.fde.emepy_fde import EmepyFDE

    def refuse(self, point):
        raise RuntimeError(f"cold grid point {point}: a test must not solve")

    solve, EmepyFDE.solve = EmepyFDE.solve, refuse
    try:
        taper = LinearTaper(DataUpdater(DATASET), input_width=0.5e-6,
                            output_width=1.2e-6, length=10e-6)
        taper.calc_output_data()
    finally:
        EmepyFDE.solve = solve
    return taper


def test_jnp_cascade_matches_production(path):
    import jax.numpy as jnp

    from dbeme.circuit.cascade_jnp import CascadeJnp
    from dbeme.validation import lumped_smatrix

    cascade, dz = CascadeJnp.from_path(path)
    S = np.asarray(cascade.smatrix(jnp.asarray(dz)))
    reference = lumped_smatrix(path, method="direct")
    assert S.shape == reference.shape
    assert np.max(np.abs(S - reference)) < 1e-10


def test_jnp_cascade_gradient(path):
    import jax
    import jax.numpy as jnp

    from dbeme.circuit.cascade_jnp import CascadeJnp

    cascade, dz = CascadeJnp.from_path(path)
    k = len(dz) // 2

    def power(dzs):
        return jnp.abs(cascade.transmission(dzs)[0, 0]) ** 2

    dz_j = jnp.asarray(dz)
    g = float(jax.grad(power)(dz_j)[k])
    h = 2e-10
    fd = (float(power(dz_j.at[k].add(h))) - float(power(dz_j.at[k].add(-h)))) / (2 * h)
    # a phase-oscillating function: the central difference is truncation-limited
    assert abs(g - fd) < 1e-4 * max(abs(fd), 1e-3)
