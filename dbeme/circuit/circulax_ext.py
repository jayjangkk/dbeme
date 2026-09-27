"""circulax plugins: what a loop needs in the time domain (`tasks/11` Phase 5).

**Only for ``.venv-circuit``.**  This file imports circulax and is loaded by
path (``importlib.util.spec_from_file_location``), never through
``dbeme.circuit``, whose package chain imports emepy; it therefore
depends on nothing else in the repo.  It is a plugin against the pinned
circulax release, not a fork: circulax's ``@component`` / ``@source``
decorators take a plain physics function returning the DAE terms
``(f, q)`` with ``dq/dt + f = 0`` and ``f[port]`` the current *into* the
component, which is all a delay line needs (plan of 2026-09-20).

``OpticalDelayLine``
    An S-matrix component is memoryless, so a ring built from them has no
    round-trip time.  This one carries the complex *envelope* through a
    chain of ``N`` first-order sections with total delay ``tau = n_g L / c``
    and applies the carrier's ``a e^{i phi}`` at the output, in the same
    wave convention circulax stamps sax components with (``V = a + b``,
    ``I = a - b``): the input is a matched termination, the output a matched
    emitter.  In steady state every section equals its input, so the DC
    solve *is* the sax arc at that wavelength (``phi(wl)`` with a linear
    index, as ``sax.models.straight``).  The chain's response converges to
    a pure delay as ``N`` grows (spread ``tau / sqrt(N)``); what a ring-down
    needs is the phase at the linewidth, ``(tau / tau_E)^2 / N``, so ``N``
    of order ten is already enough - measured in ``tests_circuit``.

``OpticalGate``
    A CW source switched on at ``t_on`` and off at ``t_off`` (sigmoid edges),
    for build-up and ring-down.
"""

import jax.nn as jnn
import jax.numpy as jnp
from circulax.components.base_component import PhysicsReturn, Signals, States, component, source

C_LIGHT = 299792458.0


def make_delay_line(n_sections: int = 16):
    """Build an ``OpticalDelayLine`` component class with ``n_sections`` states."""
    states = tuple(f"x{k}" for k in range(1, n_sections + 1))

    @component(ports=("p1", "p2"), states=states, holomorphic=True)
    def OpticalDelayLine(
        signals: Signals,
        s: States,
        neff: float = 2.4467,
        ng: float = 4.178,
        length_um: float = 56.0,
        loss_db_per_cm: float = 0.0,
        wl: float = 1.55,
        wl0: float = 1.55,
    ) -> PhysicsReturn:
        """Envelope delay ``n_g L / c`` with the carrier's ``a e^{i phi(wl)}``."""
        n = neff - (wl - wl0) * (ng - neff) / wl0
        phase = 2.0 * jnp.pi * n * length_um / wl
        amp = 10.0 ** (-loss_db_per_cm * length_um * 1e-4 / 20.0)
        tau = ng * length_um * 1e-6 / C_LIGHT
        tau_k = tau / n_sections
        a_in = signals.p1                       # matched input: V = a (b = 0), I = V
        f = {"p1": a_in}
        q = {}
        prev = a_in
        for name in states:
            xk = getattr(s, name)
            f[name] = prev - xk                 # tau_k dx/dt = x_{k-1} - x_k
            q[name] = -tau_k * xk
            prev = xk
        out = amp * jnp.exp(1j * phase) * prev
        f["p2"] = signals.p2 - 2.0 * out        # I into the port = a2 - b2 = (V - out) - out
        return f, q

    OpticalDelayLine.n_sections = n_sections
    return OpticalDelayLine


@source(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
def OpticalGate(
    signals: Signals,
    s: States,
    t: float,
    power: float = 1.0,
    phase: float = 0.0,
    t_on: float = 1e-12,
    t_off: float = 1e-9,
    rise: float = 1e-13,
) -> PhysicsReturn:
    """CW field ``sqrt(power) e^{i phase}`` between ``t_on`` and ``t_off``."""
    envelope = jnn.sigmoid((t - t_on) / rise) * (1.0 - jnn.sigmoid((t - t_off) / rise))
    v_val = jnp.sqrt(power) * envelope * jnp.exp(1j * phase)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


def complex_solution(circuit, y):
    """A circulax solution vector (real-unrolled ``[re | im]``) as complex nodes."""
    n = circuit.sys_size
    y = jnp.asarray(y)
    if y.shape[-1] == 2 * n:
        return y[..., :n] + 1j * y[..., n:]
    return y


def ring_netlist():
    """All-pass ring: coupler o4 -> delay line -> coupler o3."""
    return {
        "instances": {"cp": {"component": "coupler"}, "arc": {"component": "delay"}},
        "connections": {"cp,o4": "arc,p1", "arc,p2": "cp,o3"},
        "ports": {"in": "cp,o1", "out": "cp,o2"},
    }
