"""Phase 5 of `tasks/11`: the ring in circulax, against the frequency domain.

Runs in ``.venv-circuit`` (circulax 0.2.3, jax 0.9.2, sax 0.18.2 - pinned
apart from the main venv, whose jax 0.11 / sax 0.14 circulax refuses).
The ring is the ideal-coupler all-pass ring of Phase 1 (analytic arc,
wavelength-flat coupler), so this check does not depend on the DBEME
coupler; it asks whether circulax, fed the *same sax models*, reproduces
the sax spectrum.

**What circulax 0.2.3 can and cannot do with these models (measured
2026-09-20).**  ``sax_component`` wraps an S-matrix as a static admittance
or wave stamp evaluated at the wavelength passed in; a steady-state ("DC")
solve per wavelength is therefore the frequency-domain spectrum, batched by
``jax.vmap`` when ``wl`` is an array.  Its transient solver, however, raises
for frequency-domain components ("time-domain convolution not supported",
`s_transforms.fdomain_component`), and an S-matrix component carries no
memory: a ring built from S-matrices has no round-trip delay in the time
domain.  The ring-down gate of `tasks/11` §5.2 is therefore met with a
*plugin*, ``em_simulation/circuit/circulax_ext.py`` - an envelope delay line
built on circulax's own ``@component`` API (no fork): the arc becomes a
chain of first-order sections with the exact mean delay ``n_g L / c`` and
the carrier's ``a e^{i phi}``, and the ring-down after gating the source off
decays with ``Q lambda / (2 pi c)`` (``tests_circuit/test_delay_line.py``).

Run:  .venv-circuit/Scripts/python studies/ring/circulax_ringdown.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
import circulax  # noqa: E402
from circulax.components import electronic, photonic  # noqa: E402

C_LIGHT = 299792458.0
OUT = os.path.join(ROOT, "reports", "output", "ring", "circulax_ringdown.json")

# the same models as em_simulation.circuit.waveguide, restated so this script
# has no dependency on the main venv's package (it imports emepy)
NEFF, NG, WL0 = 2.4467, 4.178, 1.55      # 500 x 220 nm strip, ring_loss.json
R_UM = 10.0
LOSS_DB_PER_CM = 4.4                      # roughness budget, ring_loss.json
R_COUPLER = 0.98                          # |t| of the coupler


def arc(*, wl: float = 1.55) -> sax.SDict:
    length = 2 * np.pi * R_UM
    n = NEFF - (wl - WL0) * (NG - NEFF) / WL0
    phase = 2 * jnp.pi * n * length / wl
    amp = 10 ** (-LOSS_DB_PER_CM * length * 1e-4 / 20)
    return sax.reciprocal({("o1", "o2"): amp * jnp.exp(1j * phase)})


def coupler(*, wl: float = 1.55) -> sax.SDict:
    r = jnp.asarray(R_COUPLER, dtype=complex)
    ik = 1j * jnp.sqrt(1 - R_COUPLER ** 2)
    one = jnp.ones_like(jnp.asarray(wl, dtype=float))
    return sax.reciprocal({("o1", "o2"): r * one, ("o3", "o4"): r * one,
                           ("o1", "o4"): ik * one, ("o3", "o2"): ik * one})


NETLIST = {
    "instances": {"cp": {"component": "coupler"}, "arc": {"component": "arc"}},
    "connections": {"cp,o4": "arc,o1", "arc,o2": "cp,o3"},
    "ports": {"in": "cp,o1", "out": "cp,o2"},
}


def closed_form_q(wl_m=1.55e-6):
    L = 2 * np.pi * R_UM * 1e-6
    a = 10 ** (-LOSS_DB_PER_CM * L * 100 / 20)
    ra = R_COUPLER * a
    fwhm = (1 - ra) * wl_m ** 2 / (np.pi * NG * L * np.sqrt(ra))
    return wl_m / fwhm, fwhm


def main():
    result = {"model": {"neff": NEFF, "ng": NG, "R_um": R_UM, "loss_db_per_cm": LOSS_DB_PER_CM, "r": R_COUPLER}}
    q, fwhm = closed_form_q()
    result["closed_form"] = {"Q_loaded": q, "FWHM_m": fwhm, "tau_energy_s": q * 1.55e-6 / (2 * np.pi * C_LIGHT)}
    print(f"closed form: Q = {q:.0f}, FWHM = {fwhm*1e12:.2f} pm, energy ring-down tau = {result['closed_form']['tau_energy_s']*1e12:.2f} ps")

    # sax spectrum (frequency domain) - the reference; one FSR is 9.1 nm
    circ, _ = sax.circuit(NETLIST, models={"coupler": coupler, "arc": arc})
    wl = np.linspace(1.545, 1.556, 22001)
    t_sax = np.asarray(circ(wl=jnp.asarray(wl))[("in", "out")])
    T = np.abs(t_sax) ** 2
    k = int(np.argmin(T))
    result["sax"] = {"lambda_res_um": float(wl[k]), "T_min": float(T[k])}
    print(f"sax: resonance at {wl[k]:.6f} um, T_min = {T[k]:.4f}")

    # circulax: same netlist, same sax models, source at 'in', matched load at 'out'
    info = {"circulax": getattr(circulax, "__version__", "0.2.3")}
    try:
        bench = circulax.attach_testbench(
            NETLIST,
            sources={"in": {"component": "OpticalSource", "settings": {"power": 1.0}}},
            loads={"out": {"component": "Resistor", "settings": {"R": 1.0}}},  # z0 = 1: matched
        )
        models = {"coupler": circulax.sax_component(coupler, name="coupler"),
                  "arc": circulax.sax_component(arc, name="arc"),
                  "OpticalSource": photonic.OpticalSource, "Resistor": electronic.Resistor}
        compiled = circulax.compile_circuit(bench, models, is_complex=True)
        info["compiled"] = True
        info["sys_size"] = int(compiled.sys_size)
        pm = {k: int(v) for k, v in compiled.port_map.items()}
        info["port_map"] = pm
        # steady state at every wavelength (vmap over the batched global 'wl')
        sub = wl[::10]
        y = np.asarray(compiled(wl=jnp.asarray(sub)))
        n = compiled.sys_size
        if y.shape[-1] == 2 * n:            # complex circuits come back unrolled [re | im]
            y = y[:, :n] + 1j * y[:, n:]
        v_in = y[:, pm["cp,o1"]]
        v_out = y[:, pm["cp,o2"]]
        t_cx = v_out / v_in
        t_ref = np.asarray(circ(wl=jnp.asarray(sub))[("in", "out")])
        mag_err = float(np.max(np.abs(np.abs(t_cx) - np.abs(t_ref))))
        ph_err = float(np.max(np.abs(np.angle(t_cx / t_ref))))
        info["dc_sweep"] = {"points": int(len(sub)), "max_abs_T_error": mag_err, "max_phase_error_rad": ph_err,
                            "T_min_circulax": float(np.min(np.abs(t_cx) ** 2)),
                            "lambda_res_circulax_um": float(sub[int(np.argmin(np.abs(t_cx)))])}
        print(f"circulax steady state vs sax over {len(sub)} wavelengths: max ||T| error| = {mag_err:.2e}, "
              f"max phase error = {ph_err:.2e} rad, T_min = {info['dc_sweep']['T_min_circulax']:.4f}")
    except Exception as exc:
        info["compiled"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
        print("circulax steady-state check failed:", info["error"])

    # the transient, with the delay-line plugin (em_simulation/circuit/circulax_ext.py):
    # the ring built from S-matrices alone is memoryless; the plugin gives the
    # arc its round-trip time and the ring-down follows.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "circulax_ext", os.path.join(ROOT, "em_simulation", "circuit", "circulax_ext.py"))
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    tau_e = result["closed_form"]["tau_energy_s"]
    wl_res = result["sax"]["lambda_res_um"]
    t_on, t_off = 1e-12, 1e-12 + 12 * tau_e
    t_end = t_off + 6 * tau_e
    ringdown = {}
    for n_sections in (8, 16):
        bench = circulax.attach_testbench(
            ext.ring_netlist(),
            sources={"in": {"component": "OpticalGate", "settings": {"power": 1.0, "t_on": t_on, "t_off": t_off, "rise": 2e-14}}},
            loads={"out": {"component": "Resistor", "settings": {"R": 1.0}}})
        models = {"coupler": circulax.sax_component(coupler, name="coupler"), "delay": ext.make_delay_line(n_sections),
                  "OpticalGate": ext.OpticalGate, "Resistor": electronic.Resistor}
        c = circulax.compile_circuit(bench, models, is_complex=True)
        ts = np.linspace(0, t_end, 1201)
        sol = c.transient(t0=0.0, t1=t_end, dt0=tau_e / 2000, saveat=jnp.asarray(ts), max_steps=400000,
                          params={"arc.length_um": 2 * np.pi * R_UM, "arc.ng": NG, "arc.neff": NEFF,
                                  "arc.loss_db_per_cm": LOSS_DB_PER_CM, "wl": wl_res})
        yc = np.asarray(ext.complex_solution(c, sol.ys))
        energy = np.abs(yc[:, c.port_map["arc,p1"]]) ** 2
        sel = (ts > t_off + tau_e) & (ts < t_off + 4 * tau_e)
        tau_fit = -1.0 / np.polyfit(ts[sel], np.log(energy[sel]), 1)[0]
        ringdown[str(n_sections)] = {"tau_fit_s": float(tau_fit), "tau_E_s": tau_e, "ratio": float(tau_fit / tau_e),
                                    "Q_from_ringdown": float(tau_fit * 2 * np.pi * C_LIGHT / (wl_res * 1e-6))}
        print(f"ring-down, {n_sections}-section delay line: tau = {tau_fit*1e12:.3f} ps vs Q lambda/(2 pi c) = {tau_e*1e12:.3f} ps "
              f"(ratio {tau_fit/tau_e:.4f}); Q from the decay {ringdown[str(n_sections)]['Q_from_ringdown']:.0f} vs {q:.0f}")
    info["transient"] = {"ran": True, "note": "S-matrix components are memoryless; the arc is the OpticalDelayLine plugin"}
    info["ring_down"] = ringdown
    info["ring_down_gate"] = "met: tau from the decay vs Q lambda / (2 pi c), see ring_down"
    result["circulax"] = info

    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
