"""Electro-optic parameters of a depletion-mode ring modulator from the DBEME mode - `reports/16`.

What a t-CMT ring modulator (circulax's ``ring_modulator`` example) needs
and where each number comes from here:

| parameter | source |
|---|---|
| ``n_g``, ``L``, roughness loss | `reports/output/ring/ring_loss.json` (Task 11) |
| ``gamma`` = through amplitude of the coupler | `reports/output/ring/coupler_1550.json` (Task 11, bus side) |
| ``dn_eff/dV``, ``d alpha/dV``, ``alpha_FCA``, ``C_j(V)`` | **this script**: the TE0 field of the 500 x 220 nm strip over a lateral pn junction |

The junction model is the textbook one: an abrupt lateral p-n junction at
``x = x_j`` across the full core height, doping ``N_A``, ``N_D``, built-in
voltage ``V_bi = (kT/q) ln(N_A N_D / n_i^2)``, depletion width
``W(V) = sqrt(2 eps (V_bi + V_r) (N_A + N_D) / (q N_A N_D))`` split as
``W_p = W N_D/(N_A+N_D)``, ``W_n = W N_A/(N_A+N_D)``.  Removing carriers
from the depleted slab changes the silicon index and absorption by the
Soref-Bennett plasma-dispersion relations at 1550 nm (Soref & Bennett,
IEEE JQE 23, 123 (1987)):

    dn     = -(8.8e-22 dN + 8.5e-18 dP^0.8)          [dN, dP in cm^-3]
    dalpha =   8.5e-18 dN + 6.0e-18 dP                [cm^-1]

The mode sees them through its overlap with the depleted region,
``Gamma(V) = int_dep eps |E|^2 dA / int eps |E|^2 dA`` (the standard
first-order perturbation, ``dn_eff = (n_Si / n_eff) * Gamma * dn`` in the
dominant-field approximation).  The undepleted doped core carries the
static free-carrier absorption that sets ``alpha0``; the depletion's
voltage dependence gives ``v_to_wr = -omega_0 (dn_eff/dV) / n_g`` and
``alpha1``.  Junction capacitance per unit length ``eps h / W``.

Everything here is a *model with declared doping*; the DBEME content is the
field overlap.  Run (main venv):  .venv/Scripts/python studies/mrm/eo_parameters.py
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.materials import silica, silicon  # noqa: E402

WIDTH, THICKNESS = 500e-9, 220e-9
WAVELENGTH = 1.55e-6
CELL = 10e-9
N_A = N_D = 5e17            # cm^-3, both sides
X_JUNCTION = 0.0            # junction at the core centre
V_BIAS_SWEEP = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])   # reverse bias, volts
Q_E, KT_Q, EPS0, EPS_SI, N_I = 1.602e-19, 0.02585, 8.854e-12, 11.7, 1.0e10
OUT = os.path.join(ROOT, "reports", "output", "mrm", "eo_parameters.json")


def soref_bennett(dN, dP):
    """Index and absorption change (cm^-1) for carrier *removal* dN, dP (cm^-3)."""
    dn = 8.8e-22 * dN + 8.5e-18 * dP ** 0.8          # positive: carriers removed
    dalpha = -(8.5e-18 * dN + 6.0e-18 * dP)          # negative: absorption removed
    return dn, dalpha


def depletion(V_r):
    v_bi = KT_Q * np.log(N_A * N_D / N_I ** 2)
    NA, ND = N_A * 1e6, N_D * 1e6                    # m^-3
    W = np.sqrt(2 * EPS0 * EPS_SI * (v_bi + V_r) * (NA + ND) / (Q_E * NA * ND))
    return W, W * ND / (NA + ND), W * NA / (NA + ND), v_bi


def main():
    cs = FullEtchStrip(thickness=THICKNESS, core=silicon(out_of_range="raise"), cladding=silica(out_of_range="raise"))
    half = 1.4e-6
    t0 = time.time()
    fde = EmepyFDE(cross_section=cs, parameter_names=("top_width", "curvature"), num_modes=4,
                   wavelength=WAVELENGTH, window=(half, -0.7e-6, 0.7e-6), mesh=int(round(2 * half / CELL)))
    md = fde.solve((WIDTH, 0.0))
    i = next(k for k in range(len(md.neff)) if md.TE_pol[k] > 0.5 and np.real(md.neff[k]) > 1.5)
    neff = float(np.real(md.neff[i]))
    x, y = np.real(md.x), np.real(md.y)
    E = md.E[i]
    intensity = np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2 + np.abs(E[2]) ** 2
    n_si = float(np.real(silicon().index(WAVELENGTH)))
    n_ox = float(np.real(silica().index(WAVELENGTH)))
    core = (np.abs(x[:, None]) <= WIDTH / 2) & (np.abs(y[None, :]) <= THICKNESS / 2)
    eps = np.where(core, n_si ** 2, n_ox ** 2)
    weight = eps * intensity
    total = np.trapezoid(np.trapezoid(weight, y, axis=1), x)
    print(f"TE0 n_eff = {neff:.5f} [{time.time()-t0:.0f} s]; core confinement (eps-weighted) = "
          f"{np.trapezoid(np.trapezoid(weight * core, y, axis=1), x) / total:.3f}")

    # line density along x (core rows only), interpolated to 0.1 nm so a
    # depletion edge inside a cell is not quantised to the 10 nm grid
    rows_core = np.abs(y) <= THICKNESS / 2
    line = np.trapezoid(weight[:, rows_core], y[rows_core], axis=1)
    x_fine = np.arange(-WIDTH / 2, WIDTH / 2 + 1e-13, 1e-10)
    line_fine = np.interp(x_fine, x, line)

    def overlap(x_lo, x_hi):
        sel = (x_fine >= x_lo) & (x_fine < x_hi)
        if sel.sum() < 2:
            return 0.0
        return float(np.trapezoid(line_fine[sel], x_fine[sel]) / total)

    gamma_core = overlap(-WIDTH / 2, WIDTH / 2)
    # static free-carrier absorption of the fully doped core (no depletion):
    _, dalpha_p = soref_bennett(0.0, N_A)
    _, dalpha_n = soref_bennett(N_D, 0.0)
    alpha_fca_full_db_per_cm = -(overlap(-WIDTH / 2, X_JUNCTION) * dalpha_p + overlap(X_JUNCTION, WIDTH / 2) * dalpha_n) * 4.343
    rows = []
    for V in V_BIAS_SWEEP:
        W, Wp, Wn, v_bi = depletion(V)
        g_p = overlap(X_JUNCTION - Wp, X_JUNCTION)
        g_n = overlap(X_JUNCTION, X_JUNCTION + Wn)
        dn_p, da_p = soref_bennett(0.0, N_A)
        dn_n, da_n = soref_bennett(N_D, 0.0)
        dneff = (n_si / neff) * (g_p * dn_p + g_n * dn_n)
        dalpha_db = (g_p * da_p + g_n * da_n) * 4.343        # negative: loss removed by depletion
        c_j = EPS0 * EPS_SI * THICKNESS / W                   # F per metre of junction
        rows.append({"V_reverse": float(V), "W_nm": W * 1e9, "Wp_nm": Wp * 1e9, "Wn_nm": Wn * 1e9,
                     "Gamma_dep": g_p + g_n, "dneff": dneff, "dalpha_db_per_cm": dalpha_db,
                     "Cj_fF_per_um": c_j * 1e15 * 1e-6, "V_bi": v_bi})
    V = np.array([r["V_reverse"] for r in rows]); dn = np.array([r["dneff"] for r in rows])
    da = np.array([r["dalpha_db_per_cm"] for r in rows])
    # slopes at the bias points by central differences
    slope_n = np.gradient(dn, V); slope_a = np.gradient(da, V)
    omega0 = 2 * np.pi * 299792458.0 / WAVELENGTH
    ng = 4.178   # ring_loss.json; re-read below if present
    try:
        with open(os.path.join(ROOT, "reports", "output", "ring", "ring_loss.json")) as handle:
            loss = json.load(handle)
        wl = np.asarray(loss["lambdas_nm"]) * 1e-9
        n_s = np.asarray(loss["index"]["straight"]["neff"])
        ng = float(n_s[1] - wl[1] * (n_s[2] - n_s[0]) / (wl[2] - wl[0]))
    except OSError:
        pass
    for r, sn, sa in zip(rows, slope_n, slope_a):
        r["dneff_dV_per_V"] = float(sn)
        r["dlambda_dV_pm_per_V"] = float(sn * WAVELENGTH / ng * 1e12)
        r["v_to_wr_rad_per_s_per_V"] = float(-omega0 * sn / ng)
        r["dalpha_dV_db_per_cm_per_V"] = float(sa)
        r["VpiL_V_cm"] = float(WAVELENGTH / (2 * sn) * 100) if sn > 0 else None
    out = {"mode": {"neff": neff, "ng": ng, "core_confinement": gamma_core, "cell_nm": CELL * 1e9},
           "junction": {"N_A_cm3": N_A, "N_D_cm3": N_D, "x_junction_nm": X_JUNCTION * 1e9,
                        "V_bi": rows[0]["V_bi"], "alpha_FCA_full_core_db_per_cm": alpha_fca_full_db_per_cm},
           "bias": rows,
           "provenance": "TE0 field from EmepyFDE (DBEME backend) at 10 nm cells; junction abrupt, Soref-Bennett 1987 at 1550 nm"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"doped-core FCA (no depletion): {alpha_fca_full_db_per_cm:.1f} dB/cm; V_bi = {rows[0]['V_bi']:.3f} V")
    print(f"{'V_r':>5} {'W nm':>6} {'Gamma':>7} {'dn_eff':>9} {'dn/dV /V':>10} {'pm/V':>6} {'VpiL Vcm':>9} {'dalpha dB/cm':>12} {'Cj fF/um':>9}")
    for r in rows:
        print(f"{r['V_reverse']:5.1f} {r['W_nm']:6.1f} {r['Gamma_dep']:7.4f} {r['dneff']:9.2e} {r['dneff_dV_per_V']:10.2e} "
              f"{r['dlambda_dV_pm_per_V']:6.1f} {(r['VpiL_V_cm'] or 0):9.2f} {r['dalpha_db_per_cm']:12.2f} {r['Cj_fF_per_um']:9.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
