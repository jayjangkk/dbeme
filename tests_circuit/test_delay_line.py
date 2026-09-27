"""The circulax delay-line plugin (`em_simulation/circuit/circulax_ext.py`).

Runs in ``.venv-circuit`` only:  .venv-circuit/Scripts/python -m pytest tests_circuit -q

Gate of the 2026-09-20 plan: the line alone delays a step by ``tau`` at
amplitude ``a``; the delay-line ring's steady state equals the sax ring's
spectrum; the ring-down after the source is gated off decays with
``tau_E = Q_L lambda / (2 pi c)``; the build-up has the same time constant.
"""

import importlib.util
import os

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
import circulax  # noqa: E402
from circulax.components import electronic  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("circulax_ext", os.path.join(ROOT, "em_simulation", "circuit", "circulax_ext.py"))
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)

C = 299792458.0
NEFF, NG, WL0 = 2.4467, 4.178, 1.55
R_UM = 10.0
LENGTH_UM = 2 * np.pi * R_UM
LOSS = 4.4
R_COUPLER = 0.98


def coupler(*, wl: float = 1.55) -> sax.SDict:
    r = jnp.asarray(R_COUPLER, dtype=complex)
    ik = 1j * jnp.sqrt(1 - R_COUPLER ** 2)
    one = jnp.ones_like(jnp.asarray(wl, dtype=float))
    return sax.reciprocal({("o1", "o2"): r * one, ("o3", "o4"): r * one,
                           ("o1", "o4"): ik * one, ("o3", "o2"): ik * one})


def arc(*, wl: float = 1.55) -> sax.SDict:
    n = NEFF - (wl - WL0) * (NG - NEFF) / WL0
    amp = 10 ** (-LOSS * LENGTH_UM * 1e-4 / 20)
    return sax.reciprocal({("o1", "o2"): amp * jnp.exp(1j * 2 * jnp.pi * n * LENGTH_UM / wl)})


def closed_form():
    L = LENGTH_UM * 1e-6
    a = 10 ** (-LOSS * L * 100 / 20)
    ra = R_COUPLER * a
    fwhm = (1 - ra) * 1.55e-6 ** 2 / (np.pi * NG * L * np.sqrt(ra))
    q = 1.55e-6 / fwhm
    return q, q * 1.55e-6 / (2 * np.pi * C)


def _line_circuit(n_sections):
    net = {"instances": {"dl": {"component": "delay"}}, "connections": {},
           "ports": {"in": "dl,p1", "out": "dl,p2"}}
    bench = circulax.attach_testbench(
        net, sources={"in": {"component": "OpticalGate", "settings": {"power": 1.0, "t_on": 2e-12, "t_off": 1.0, "rise": 2e-14}}},
        loads={"out": {"component": "Resistor", "settings": {"R": 1.0}}})
    models = {"delay": ext.make_delay_line(n_sections), "OpticalGate": ext.OpticalGate, "Resistor": electronic.Resistor}
    return circulax.compile_circuit(bench, models, is_complex=True)


@pytest.mark.parametrize("n_sections", [8, 32])
def test_delay_line_delays_a_step_by_tau(n_sections):
    c = _line_circuit(n_sections)
    tau = NG * LENGTH_UM * 1e-6 / C
    ts = np.linspace(0, 2e-12 + 3 * tau, 601)
    sol = c.transient(t0=0.0, t1=float(ts[-1]), dt0=tau / 400, saveat=jnp.asarray(ts),
                      params={"dl.length_um": LENGTH_UM, "dl.ng": NG, "dl.neff": NEFF, "dl.loss_db_per_cm": LOSS})
    y = np.asarray(ext.complex_solution(c, sol.ys))
    v_out = y[:, c.port_map["dl,p2"]]
    amp = 10 ** (-LOSS * LENGTH_UM * 1e-4 / 20)
    # the mean delay (centroid of the step's derivative) is exact for the
    # chain at any N; the half-rise point is its median, below the mean
    mag = np.abs(v_out)
    dmag = np.gradient(mag, ts)
    t_mean = np.trapezoid(ts * dmag, ts) / np.trapezoid(dmag, ts)
    assert abs((t_mean - 2e-12) / tau - 1.0) < 0.01, (t_mean, tau)
    spread = np.sqrt(np.trapezoid((ts - t_mean) ** 2 * dmag, ts) / np.trapezoid(dmag, ts))
    assert spread < 1.2 * tau / np.sqrt(n_sections)
    assert abs(np.abs(v_out[-1]) / amp - 1.0) < 1e-3   # BDF2 step accuracy, not the model
    # the carrier phase of the settled output is the sax arc's
    expected = np.asarray(arc(wl=1.55)[("o1", "o2")])
    assert abs(np.angle(v_out[-1] / expected)) < 1e-3


def _ring_circuit(n_sections, t_on, t_off):
    bench = circulax.attach_testbench(
        ext.ring_netlist(),
        sources={"in": {"component": "OpticalGate", "settings": {"power": 1.0, "t_on": t_on, "t_off": t_off, "rise": 2e-14}}},
        loads={"out": {"component": "Resistor", "settings": {"R": 1.0}}})
    models = {"coupler": circulax.sax_component(coupler, name="coupler"), "delay": ext.make_delay_line(n_sections),
              "OpticalGate": ext.OpticalGate, "Resistor": electronic.Resistor}
    return circulax.compile_circuit(bench, models, is_complex=True)


def _resonance():
    circ, _ = sax.circuit({"instances": {"cp": {"component": "coupler"}, "arc": {"component": "arc"}},
                           "connections": {"cp,o4": "arc,o1", "arc,o2": "cp,o3"},
                           "ports": {"in": "cp,o1", "out": "cp,o2"}}, models={"coupler": coupler, "arc": arc})
    wl = np.linspace(1.545, 1.556, 44001)
    T = np.abs(np.asarray(circ(wl=jnp.asarray(wl))[("in", "out")])) ** 2
    return float(wl[int(np.argmin(T))]), circ


def test_delay_line_ring_steady_state_is_the_sax_ring():
    c = _ring_circuit(8, 1e-12, 1.0)
    _, circ = _resonance()
    wl = np.linspace(1.549, 1.554, 41)
    params = {"arc.length_um": LENGTH_UM, "arc.ng": NG, "arc.neff": NEFF, "arc.loss_db_per_cm": LOSS}
    t_cx = []
    for w in wl:  # steady state with the gate fully on: solve at t >> t_on via the DC of the on-state
        y = np.asarray(ext.complex_solution(c, c(params={**params, "wl": float(w), "src_in.t_on": -1.0})))
        t_cx.append(y[c.port_map["cp,o2"]] / y[c.port_map["cp,o1"]])
    t_ref = np.asarray(circ(wl=jnp.asarray(wl))[("in", "out")])
    assert np.max(np.abs(np.asarray(t_cx) - t_ref)) < 1e-6


@pytest.mark.parametrize("n_sections", [8, 16])
def test_ring_down_time_is_q_over_omega(n_sections):
    q_cf, tau_e = closed_form()
    wl_res, _ = _resonance()
    t_on, t_off = 1e-12, 1e-12 + 12 * tau_e
    t_end = t_off + 6 * tau_e
    c = _ring_circuit(n_sections, t_on, t_off)
    ts = np.linspace(0, t_end, 1201)
    sol = c.transient(t0=0.0, t1=t_end, dt0=tau_e / 2000, saveat=jnp.asarray(ts), max_steps=400000,
                      params={"arc.length_um": LENGTH_UM, "arc.ng": NG, "arc.neff": NEFF,
                              "arc.loss_db_per_cm": LOSS, "wl": wl_res})
    y = np.asarray(ext.complex_solution(c, sol.ys))
    energy = np.abs(y[:, c.port_map["arc,p1"]]) ** 2       # field entering the arc ~ stored energy
    # ring-down: fit log(energy) between t_off + tau_e and t_off + 4 tau_e
    sel = (ts > t_off + 1.0 * tau_e) & (ts < t_off + 4.0 * tau_e)
    slope = np.polyfit(ts[sel], np.log(energy[sel]), 1)[0]
    tau_fit = -1.0 / slope
    assert abs(tau_fit / tau_e - 1.0) < 0.02, (tau_fit, tau_e)
    # build-up: 1 - e^{-t/tau_E} in amplitude of the intracavity field
    sel = (ts > t_on + 0.2 * tau_e) & (ts < t_on + 3.0 * tau_e)
    amp = np.abs(y[:, c.port_map["arc,p1"]])
    a_inf = amp[(ts > t_off - 2 * tau_e) & (ts < t_off)].mean()
    slope_up = np.polyfit(ts[sel], np.log(1 - amp[sel] / a_inf), 1)[0]
    assert abs((-1.0 / slope_up) / (2 * tau_e) - 1.0) < 0.05   # amplitude builds with 2 tau_E
