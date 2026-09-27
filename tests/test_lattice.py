"""`em_simulation/circuit/lattice.py` against sax and against known lattice designs.

* the closed-form transfer of a cell equals the sax composition of the same
  ideal couplers and straight arms (the netlist and the port convention are
  right);
* the closed form is unitary;
* the imec MZ4 (0.348 / 0.158 / 0.790 / 0.158 / 0.348, four equal delays) is
  flat-top: somewhere in its period the 0.5 dB bandwidth of the cross port
  exceeds a quarter of the FSR (the paper: 63 % of the channel spacing, i.e.
  31 % of the FSR) and the through port is power-complementary;
* a first-order MZI is the sinusoid it must be.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402

from em_simulation.circuit import IndexModel, ideal_coupler, straight_model  # noqa: E402
from em_simulation.circuit.lattice import (  # noqa: E402
    lattice_netlist,
    lattice_transfer,
    response_powers,
    unitarity_defect,
)

NEFF, NG, WL0 = 1.626, 1.991, 1.31e-6


def _phases(lengths_um, wls_um):
    """(N, 2, W) phases of straight arms of the given lengths at the given wavelengths."""
    index = IndexModel.from_neff_ng(NEFF, NG, WL0)
    wl = np.asarray(wls_um) * 1e-6
    n = np.asarray(index.neff(wl))
    out = []
    for la, lb in lengths_um:
        out.append([2 * np.pi * n * la * 1e-6 / wl, 2 * np.pi * n * lb * 1e-6 / wl])
    return np.asarray(out)


def test_closed_form_matches_sax_cell():
    kappa2s = (0.5, 0.29, 0.08)
    lengths = [(250.0 + 753.0, 250.0), (250.0 + 1506.0, 250.0)]
    wls = np.linspace(1.300, 1.320, 41)
    index = IndexModel.from_neff_ng(NEFF, NG, WL0)
    models = {f"c{k}": ideal_coupler(kappa=float(np.sqrt(v))) for k, v in enumerate(kappa2s)}
    for k, (la, lb) in enumerate(lengths):
        models[f"a{k}"] = straight_model(index, la * 1e-6)
        models[f"b{k}"] = straight_model(index, lb * 1e-6)
    net = lattice_netlist(2, couplers=[f"c{k}" for k in range(3)], arms=[(f"a{k}", f"b{k}") for k in range(2)])
    circuit, _ = sax.circuit(net, models=models)
    S = circuit(wl=jnp.asarray(wls))
    m = np.asarray(lattice_transfer(kappa2s, _phases(lengths, wls)))
    for (i, j), key in ((("in1", "out1"), (0, 0)), (("in2", "out1"), (0, 1)),
                        (("in1", "out2"), (1, 0)), (("in2", "out2"), (1, 1))):
        assert np.max(np.abs(np.asarray(S[(i, j)]) - m[..., key[0], key[1]])) < 1e-9


def test_closed_form_is_unitary():
    wls = np.linspace(1.300, 1.320, 11)
    phases = _phases([(1003.0, 250.0), (1003.0, 250.0), (1003.0, 250.0), (1003.0, 250.0)], wls)
    assert unitarity_defect((0.348, 0.158, 0.790, 0.158, 0.348), phases) < 1e-12


def _period_response(kappa2s, multiples, n_points=2000):
    """Powers over one period of the finest stage, dispersionless."""
    x = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phases = np.asarray([[2 * np.pi * m * x, np.zeros_like(x)] for m in multiples])
    return x, {k: np.asarray(v) for k, v in response_powers(kappa2s, phases).items()}


def _best_half_db_bandwidth(power, x):
    """Largest fraction of the period over which `power` stays within 0.5 dB of its local max."""
    best = 0.0
    n = len(x)
    thresh = 10 ** (-0.05)
    for i in range(n):
        if power[i] < 0.5:
            continue
        lo = power[i] * thresh
        width = 0
        j = i
        while width < n and power[j % n] >= lo:
            width += 1
            j += 1
        best = max(best, width / n)
    return best


def test_mz4_is_flat_top():
    """imec's couplings in this module's convention are their complements
    (``[[r, ik],[ik, r]]`` with the delay on arm 1: a 0.842 coupler is the
    0.158 coupler with its arms swapped); the cell is then flat-top with a
    0.5 dB bandwidth above a quarter of the FSR at both ports.  The paper:
    63 % of the channel spacing, i.e. 31 % of the FSR."""
    imec = (0.348, 0.158, 0.790, 0.158, 0.348)
    x, p = _period_response(tuple(1.0 - k for k in imec), (1, 1, 1, 1))
    cross = p[("in1", "out2")]
    through = p[("in1", "out1")]
    assert np.max(np.abs(cross + through - 1.0)) < 1e-12
    assert _best_half_db_bandwidth(cross, x) > 0.25
    assert _best_half_db_bandwidth(through, x) > 0.25
    # the uncomplemented set is a different (narrower) filter, not a flat-top
    x, q = _period_response(imec, (1, 1, 1, 1))
    assert _best_half_db_bandwidth(q[("in1", "out2")], x) < 0.25


def test_first_order_mzi_is_a_sinusoid():
    x, p = _period_response((0.5, 0.5), (1,))
    expected = np.sin(np.pi * x) ** 2
    assert np.max(np.abs(p[("in1", "out2")] - expected)) < 1e-12 or \
        np.max(np.abs(p[("in1", "out1")] - expected)) < 1e-12
