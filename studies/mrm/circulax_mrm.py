"""Depletion-mode ring modulator in circulax, parametrised from DBEME - `reports/16_MRM.md`.

Follows circulax's ``ring_modulator`` example (t-CMT ring with an electrical
port, RC driver, optical CW source, 1 Ohm load) with every ring parameter
taken from this repo instead of typed in:

* ``n_g``, ``L`` and the roughness loss: `reports/output/ring/ring_loss.json`;
* ``gamma`` (through amplitude) of the point coupler at the chosen gap:
  `reports/output/ring/coupler_1550.json` (bus side, Task 11);
* ``dn_eff/dV``, ``d alpha/dV``, the doped-core free-carrier absorption and
  ``C_j``: `reports/output/mrm/eo_parameters.json` (the TE0 field over a
  lateral pn junction, `studies/mrm/eo_parameters.py`).

Analyses (circulax 0.2.3, ``.venv-circuit``):
  1. optical step on/off -> photon lifetime, against ``tau = Q lambda/(2 pi c)``
     from the same ``gamma``, ``a`` (and against the delay-line ring of
     Task 11 built from the same numbers);
  2. small-signal EO response 0.5-80 GHz by harmonic balance and by a
     transient sweep, against the analytic RC x optical (second-order) form;
  3. NRZ eye diagram at a symbol rate set by the photon lifetime, at two
     bias points.

Run:  .venv-circuit/Scripts/python studies/mrm/circulax_mrm.py [--gap 150] [--radius 10]
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.nn as jnn  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import circulax  # noqa: E402
from circulax.components import electronic, photonic  # noqa: E402
from circulax.components.base_component import PhysicsReturn, Signals, States, component, source  # noqa: E402

C_LIGHT = 299792458.0
OUT = os.path.join(ROOT, "reports", "output", "mrm")
_spec = importlib.util.spec_from_file_location("circulax_ext", os.path.join(ROOT, "dbeme", "circuit", "circulax_ext.py"))
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)


# ------------------------------------------------------------- components
@component(ports=("p1", "p2", "v_e"), states=("a", "i_out"), holomorphic=False)
def RingModulatorCMT(
    signals: Signals,
    s: States,
    ng: float = 4.178,
    L: float = 62.83e-6,
    gamma: float = 0.982,
    alpha0: float = 0.989,
    alpha1: float = 0.0,
    f_operating: float = 1.934e14,
    f_resonance: float = 1.934e14,
    v_to_wr: float = 0.0,
) -> PhysicsReturn:
    """Temporal coupled-mode ring with an electro-optic port (circulax example form).

    ``da/dt = -j sqrt(2/tau_e) E_in + j d_omega a - a/tau``, ``E_out = E_in - j sqrt(2/tau_e) a``,
    ``d_omega = 2 pi (f_op - f_res) + v_to_wr V``, round-trip amplitude
    ``alpha0 + alpha1 V``.  The electrical port is read at high impedance.
    """
    voltage = jnp.real(signals.v_e)
    tau_e = 2.0 * ng * L / ((1.0 - gamma ** 2) * C_LIGHT)
    alpha_v = alpha0 + alpha1 * voltage
    tau_l = 2.0 * ng * L / ((1.0 - alpha_v ** 2) * C_LIGHT)
    tau = 1.0 / (1.0 / tau_e + 1.0 / tau_l)
    coupling = jnp.sqrt(2.0 / tau_e)
    d_omega = 2.0 * jnp.pi * (f_operating - f_resonance) + v_to_wr * voltage
    rhs_a = -1j * coupling * signals.p1 + 1j * d_omega * s.a - s.a / tau
    e_out = signals.p1 - 1j * coupling * s.a
    f = {"p1": 0.0 + 0.0j, "p2": s.i_out, "i_out": signals.p2 - e_out, "a": -rhs_a, "v_e": 0.0 + 0.0j}
    q = {"a": s.a}
    return f, q


@source(ports=("p1", "p2"), states=("i_src",), amplitude_param="V_ac", holomorphic=True)
def BiasedAC(signals: Signals, s: States, t: float, V_bias: float = -1.0, V_ac: float = 0.05,
             freq: float = 1e9, t_start: float = 0.0) -> PhysicsReturn:
    v = V_bias + jnp.where(t >= t_start, V_ac * jnp.sin(2 * jnp.pi * freq * (t - t_start)), 0.0)
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": (signals.p1 - signals.p2) - v}, {}


def make_nrz(bits):
    bits = tuple(int(b) for b in bits)

    @source(ports=("p1", "p2"), states=("i_src",), holomorphic=True)
    def NRZ(signals: Signals, s: States, t: float, V_low: float = -1.5, V_high: float = -0.5,
            T_bit: float = 1e-10, rise: float = 5e-12, t_start: float = 0.0) -> PhysicsReturn:
        arr = jnp.asarray(bits, dtype=float)
        delta = arr - jnp.concatenate([jnp.zeros(1), arr[:-1]])
        times = t_start + jnp.arange(len(bits)) * T_bit
        v = V_low + (V_high - V_low) * jnp.sum(delta * jnn.sigmoid((t - times) / rise))
        return {"p1": s.i_src, "p2": -s.i_src, "i_src": (signals.p1 - signals.p2) - v}, {}
    return NRZ


# ---------------------------------------------------------------- inputs
def dbeme_parameters(gap_nm, radius_um):
    with open(os.path.join(ROOT, "reports", "output", "ring", "ring_loss.json")) as h:
        loss = json.load(h)
    with open(os.path.join(ROOT, "reports", "output", "ring", "coupler_1550.json")) as h:
        coup = json.load(h)
    with open(os.path.join(OUT, "eo_parameters.json")) as h:
        eo = json.load(h)
    wl = np.asarray(loss["lambdas_nm"]) * 1e-9
    n_s = np.asarray(loss["index"]["straight"]["neff"])
    ng = float(n_s[1] - wl[1] * (n_s[2] - n_s[0]) / (wl[2] - wl[0]))
    key = f"{radius_um:g}"
    neff_bend = loss["index"][key]["neff"][1] if key in loss["index"] else float(n_s[1])
    entry = min(coup["gaps"], key=lambda g: abs(g["g0_nm"] - gap_nm))
    if abs(coup["radius_um"] - radius_um) > 1e-9:
        raise SystemExit(f"coupler results are for R = {coup['radius_um']} um")
    gamma = float(np.sqrt(entry["t2"]))
    rough = loss["roughness"]["straight_db_per_cm_eim_index"] * loss["roughness"]["bend_enhancement"].get(key, 1.0)
    return {"ng": ng, "neff_bend": neff_bend, "L": 2 * np.pi * radius_um * 1e-6, "gamma": gamma,
            "gap_nm": entry["g0_nm"], "kappa2": 1 - entry["t2"], "rough_db_per_cm": rough,
            "fca_full_db_per_cm": eo["junction"]["alpha_FCA_full_core_db_per_cm"], "eo": eo}


def ring_parameters(p, v_bias):
    """t-CMT parameters at a reverse bias (volts, positive = reverse)."""
    rows = p["eo"]["bias"]
    V = np.array([r["V_reverse"] for r in rows])
    dn_dv = float(np.interp(v_bias, V, [r["dneff_dV_per_V"] for r in rows]))
    dalpha = float(np.interp(v_bias, V, [r["dalpha_db_per_cm"] for r in rows]))
    dalpha_dv = float(np.interp(v_bias, V, [r["dalpha_dV_db_per_cm_per_V"] for r in rows]))
    cj = float(np.interp(v_bias, V, [r["Cj_fF_per_um"] for r in rows])) * 1e-15 * 1e6 * p["L"]  # F for the ring
    loss_db = p["rough_db_per_cm"] + p["fca_full_db_per_cm"] + dalpha
    a = 10 ** (-loss_db * p["L"] * 100 / 20)
    alpha1 = a * (-np.log(10) / 20) * dalpha_dv * p["L"] * 100
    omega0 = 2 * np.pi * C_LIGHT / 1.55e-6
    v_to_wr = -omega0 * dn_dv / p["ng"]
    return {"a": a, "alpha1": alpha1, "v_to_wr": v_to_wr, "loss_db_per_cm": loss_db, "Cj_F": cj,
            "dn_dv": dn_dv, "dlambda_dV_pm": dn_dv * 1.55e-6 / p["ng"] * 1e12}


def lifetimes(p, r):
    tau_e = 2 * p["ng"] * p["L"] / ((1 - p["gamma"] ** 2) * C_LIGHT)
    tau_l = 2 * p["ng"] * p["L"] / ((1 - r["a"] ** 2) * C_LIGHT)
    tau = 1 / (1 / tau_e + 1 / tau_l)
    omega0 = 2 * np.pi * C_LIGHT / 1.55e-6
    return {"tau_e_ps": tau_e * 1e12, "tau_l_ps": tau_l * 1e12, "tau_ps": tau * 1e12,
            "Q_loaded": omega0 * tau / 2, "Q_coupling": omega0 * tau_e / 2, "Q_intrinsic": omega0 * tau_l / 2,
            "f3dB_optical_GHz": 1 / (2 * np.pi * tau) * 1e-9}


# --------------------------------------------------------------- netlists
def netlist_eo(v_bias, v_ac, freq, R_s, C_j, ring_settings):
    return {
        "instances": {
            "GND": {"component": "ground"},
            "OptSrc": {"component": "optical_cw", "settings": {"power": 1.0}},
            "Ring": {"component": "ring_eo", "settings": ring_settings},
            "Load": {"component": "resistor", "settings": {"R": 1.0}},
            "Vsrc": {"component": "biased_ac", "settings": {"V_bias": v_bias, "V_ac": v_ac, "freq": freq, "t_start": 0.0}},
            "Rs": {"component": "resistor", "settings": {"R": R_s}},
            "Cj": {"component": "capacitor", "settings": {"C": C_j}},
        },
        "connections": {
            "GND,p1": ("OptSrc,p2", "Load,p2", "Vsrc,p2", "Cj,p2"),
            "OptSrc,p1": "Ring,p1", "Ring,p2": "Load,p1", "Vsrc,p1": "Rs,p1", "Rs,p2": ("Cj,p1", "Ring,v_e"),
        },
        "ports": {"in": "Ring,p1", "out": "Ring,p2", "ve": "Ring,v_e"},
    }


def compile_eo(net, nrz=None):
    models = {"optical_cw": photonic.OpticalSource, "ring_eo": RingModulatorCMT, "resistor": electronic.Resistor,
              "capacitor": electronic.Capacitor, "biased_ac": BiasedAC}
    if nrz is not None:
        models["nrz"] = nrz
    return circulax.compile_circuit(net, models, is_complex=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=150.0)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--vbias", type=float, default=1.0, help="reverse bias, V")
    ap.add_argument("--rs", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = dbeme_parameters(args.gap, args.radius)
    r = ring_parameters(p, args.vbias)
    lt = lifetimes(p, r)
    print(f"DBEME ring: R = {args.radius} um, gap {p['gap_nm']:.0f} nm -> gamma = {p['gamma']:.4f} (kappa^2 {p['kappa2']:.4f}); "
          f"n_g {p['ng']:.3f}; loss {r['loss_db_per_cm']:.1f} dB/cm (rough {p['rough_db_per_cm']:.1f} + FCA) -> a = {r['a']:.4f}")
    print(f"  EO at {args.vbias} V reverse: dn/dV {r['dn_dv']:.2e}/V = {r['dlambda_dV_pm']:.1f} pm/V, alpha1 {r['alpha1']:.2e}/V, "
          f"C_j {r['Cj_F']*1e15:.1f} fF, R_s {args.rs} Ohm -> RC {args.rs*r['Cj_F']*1e12:.2f} ps")
    print(f"  lifetimes: tau_e {lt['tau_e_ps']:.1f} ps, tau_l {lt['tau_l_ps']:.1f} ps, tau {lt['tau_ps']:.1f} ps; "
          f"Q_L {lt['Q_loaded']:.0f}, optical f_3dB {lt['f3dB_optical_GHz']:.2f} GHz")
    result = {"inputs": {k: v for k, v in p.items() if k != "eo"}, "ring": r, "lifetimes": lt, "R_s": args.rs, "V_bias": args.vbias}

    f_res = C_LIGHT / 1.55e-6
    tau = lt["tau_ps"] * 1e-12
    # Voltage convention throughout: reverse bias positive (as eo_parameters.json).
    # The ring shifts its own resonance by v_to_wr * V, so the biased resonance is
    # f_res + v_to_wr V_bias / 2 pi and the laser sits one half-linewidth above it
    # (d_omega * tau = 1, the example's operating point).
    f_res_biased = f_res + r["v_to_wr"] * args.vbias / (2 * np.pi)
    settings = {"ng": p["ng"], "L": p["L"], "gamma": p["gamma"], "alpha0": r["a"], "alpha1": r["alpha1"],
                "f_operating": f_res_biased + 1 / (2 * np.pi * tau), "f_resonance": f_res, "v_to_wr": r["v_to_wr"]}

    # 1. photon lifetime: optical step on/off with the electrical bias fixed
    # 1. at zero detuning so the decay is a clean exponential (no beat)
    step_settings = dict(settings, f_operating=f_res_biased)
    net = netlist_eo(args.vbias, 0.0, 1e9, args.rs, r["Cj_F"], step_settings)
    net["instances"]["OptSrc"] = {"component": "optical_gate", "settings": {"power": 1.0, "t_on": 5e-12, "t_off": 5e-12 + 10 * tau, "rise": 2e-13}}
    models = {"optical_gate": ext.OpticalGate, "ring_eo": RingModulatorCMT, "resistor": electronic.Resistor,
              "capacitor": electronic.Capacitor, "biased_ac": BiasedAC}
    c1 = circulax.compile_circuit(net, models, is_complex=True)
    t_end = 5e-12 + 16 * tau
    ts = np.linspace(0, t_end, 1601)
    sol = c1.transient(t0=0.0, t1=t_end, dt0=tau / 500, saveat=jnp.asarray(ts), max_steps=400000)
    y = np.asarray(ext.complex_solution(c1, sol.ys))
    a_state = y[:, c1.port_map.get("Ring,a", -1)] if "Ring,a" in c1.port_map else None
    e_out = y[:, c1.port_map["Ring,p2"]]
    energy = np.abs(a_state) ** 2 if a_state is not None else np.abs(e_out - y[:, c1.port_map["Ring,p1"]]) ** 2
    t_off = 5e-12 + 10 * tau
    sel = (ts > t_off + 1.5 * tau) & (ts < t_off + 5 * tau)
    tau_fit = -1 / np.polyfit(ts[sel], np.log(energy[sel] + 1e-300), 1)[0]
    # the detuned ring rings at d_omega: fit the envelope through the log of |a|^2 smoothed over a beat period
    # t-CMT's tau is the *amplitude* lifetime (da/dt = -a/tau); the energy decays with tau/2
    result["photon_lifetime"] = {"energy_tau_fit_ps": tau_fit * 1e12, "amplitude_tau_cmt_ps": lt["tau_ps"],
                                 "ratio_2tau_fit_over_tau": 2 * tau_fit / tau}
    print(f"  1. optical step: energy decay {tau_fit*1e12:.2f} ps = tau/2 with tau = {2*tau_fit*1e12:.2f} ps vs t-CMT tau {lt['tau_ps']:.2f} ps (ratio {2*tau_fit/tau:.4f})")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ts * 1e12, np.abs(e_out) ** 2, label="|E_out|²")
    ax.plot(ts * 1e12, energy / max(energy.max(), 1e-30), label="ring energy (norm.)")
    ax.set_xlabel("time (ps)"); ax.set_ylabel("power"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"optical step on/off, τ = {lt['tau_ps']:.1f} ps", fontsize=9)
    fig.savefig(os.path.join(OUT, "mrm_1_photon_lifetime.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # 2. small-signal EO response: harmonic balance and transient, vs analytic
    freqs = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80]) * 1e9
    v_ac = 0.02
    net = netlist_eo(args.vbias, v_ac, 1e9, args.rs, r["Cj_F"], settings)
    c2 = compile_eo(net)
    y_dc = c2.dc()
    amps_hb, amps_tr = [], []
    for f in freqs:
        try:
            y_time, _ = c2.hb(freq=float(f), harmonics=5, y0=y_dc, params={"Vsrc.freq": float(f)})
            yc = np.asarray(ext.complex_solution(c2, y_time))
            P = np.abs(yc[:, c2.port_map["Ring,p2"]]) ** 2
            amps_hb.append(float(P.max() - P.min()))
        except Exception as exc:
            amps_hb.append(np.nan)
            print(f"     HB failed at {f/1e9:g} GHz: {type(exc).__name__}: {str(exc)[:120]}")
        n_per = 6
        t1 = n_per / f
        tt = np.linspace(0, t1, 60 * n_per + 1)
        sol = c2.transient(t0=0.0, t1=t1, dt0=1 / (f * 200), saveat=jnp.asarray(tt), max_steps=400000,
                           params={"Vsrc.freq": float(f)})
        yc = np.asarray(ext.complex_solution(c2, sol.ys))
        P = np.abs(yc[:, c2.port_map["Ring,p2"]]) ** 2
        last = tt > t1 * (1 - 2 / n_per)
        amps_tr.append(float(P[last].max() - P[last].min()))
    amps_hb, amps_tr = np.array(amps_hb), np.array(amps_tr)
    # analytic: RC pole x second-order optical response (circulax example form)
    omega_m = 2 * np.pi * freqs
    inv_tau = 1 / tau
    inv_tau_l = 1 / (lt["tau_l_ps"] * 1e-12)
    d_omega_dc = 2 * np.pi * (settings["f_operating"] - settings["f_resonance"])
    H_rc = 1 / (1 + 1j * omega_m * args.rs * r["Cj_F"])
    H_opt = (1j * omega_m + 2 * inv_tau_l) / (-(omega_m ** 2) + 1j * 2 * inv_tau * omega_m + d_omega_dc ** 2 + inv_tau ** 2)
    H = np.abs(H_rc * H_opt)
    ref = amps_tr[0] if np.isfinite(amps_tr[0]) else amps_hb[0]
    resp_tr = 20 * np.log10(amps_tr / ref)
    resp_hb = 20 * np.log10(amps_hb / ref)
    resp_an = 20 * np.log10(H / H[0])
    f3 = float(np.interp(-3.0, resp_an[::-1], freqs[::-1]) / 1e9) if resp_an.min() < -3 else None
    f3_tr = float(np.interp(-3.0, resp_tr[::-1], freqs[::-1]) / 1e9) if np.nanmin(resp_tr) < -3 else None
    result["eo_response"] = {"freqs_GHz": (freqs / 1e9).tolist(), "transient_dB": resp_tr.tolist(), "hb_dB": resp_hb.tolist(),
                             "analytic_dB": resp_an.tolist(), "f3dB_analytic_GHz": f3, "f3dB_transient_GHz": f3_tr,
                             "RC_pole_GHz": 1 / (2 * np.pi * args.rs * r["Cj_F"]) * 1e-9}
    print(f"  2. EO -3 dB: transient {f3_tr} GHz, analytic {f3} GHz; RC pole {result['eo_response']['RC_pole_GHz']:.0f} GHz")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.semilogx(freqs / 1e9, resp_tr, "o", label="transient")
    ax.semilogx(freqs / 1e9, resp_hb, "s", mfc="none", label="harmonic balance")
    ax.semilogx(freqs / 1e9, resp_an, "-", label="analytic RC × optical")
    ax.axhline(-3, color="k", lw=0.6, ls=":"); ax.set_xlabel("modulation frequency (GHz)"); ax.set_ylabel("EO response (dB)")
    ax.set_title(f"small-signal EO response, V_bias = {args.vbias} V, gap {p['gap_nm']:.0f} nm", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(OUT, "mrm_2_eo_response.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # 3. NRZ eye at a symbol rate set by the lifetime: T_bit = 3 tau (and 1.5 tau)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 96)
    eyes = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, factor in zip(axes, (3.0, 1.5)):
        T_bit = factor * tau
        rate = 1 / T_bit
        nrz = make_nrz(bits)
        net = netlist_eo(args.vbias, 0.0, 1e9, args.rs, r["Cj_F"], settings)
        net["instances"]["Vsrc"] = {"component": "nrz", "settings": {"V_low": args.vbias - 1.0, "V_high": args.vbias + 1.0,
                                                                       "T_bit": T_bit, "rise": T_bit / 10, "t_start": 2 * T_bit}}
        c3 = compile_eo(net, nrz=nrz)
        t_end = (len(bits) + 2) * T_bit
        tt = np.linspace(0, t_end, int(len(bits) * 40))
        sol = c3.transient(t0=0.0, t1=t_end, dt0=T_bit / 200, saveat=jnp.asarray(tt), max_steps=2000000)
        yc = np.asarray(ext.complex_solution(c3, sol.ys))
        P = np.abs(yc[:, c3.port_map["Ring,p2"]]) ** 2
        # fold into a 2-bit eye after the first 8 bits
        keep = tt > 10 * T_bit
        phase = np.mod(tt[keep], 2 * T_bit) / T_bit
        ax.plot(phase, P[keep], ",", alpha=0.5)
        ax.set_xlabel("time (bit periods)"); ax.set_ylabel("|E_out|²"); ax.set_title(f"NRZ at {rate*1e-9:.1f} GBaud (T_bit = {factor:g} τ)", fontsize=9)
        ax.grid(alpha=0.3)
        # eye opening: at the bit centre, min of 'one' levels minus max of 'zero' levels
        centre = np.abs(phase - 0.5) < 0.05
        levels = P[keep][centre]
        thresh = 0.5 * (levels.max() + levels.min())
        ones, zeros = levels[levels > thresh], levels[levels <= thresh]
        eyes[f"{rate*1e-9:.1f}_GBaud"] = {"T_bit_over_tau": factor, "eye_opening": float(ones.min() - zeros.max()) if len(ones) and len(zeros) else None,
                                          "high_mean": float(ones.mean()) if len(ones) else None, "low_mean": float(zeros.mean()) if len(zeros) else None}
    fig.savefig(os.path.join(OUT, "mrm_3_eye.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    result["eye"] = eyes
    print("  3. eyes:", eyes)

    with open(os.path.join(OUT, "mrm_results.json"), "w") as h:
        json.dump(result, h, indent=1, default=float)
    print("wrote", os.path.join(OUT, "mrm_results.json"))


if __name__ == "__main__":
    main()
