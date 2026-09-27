"""Ring-waveguide index and loss budget for a 500 x 220 nm Si strip - `tasks/11` §2.2.

Three things, all from direct solves (no dataset; a handful of points):

1. ``n_eff(lambda, R)`` of the bent TE0 mode at 1530 / 1550 / 1570 nm for
   R in {5, 8, 10, 15} um and straight, on the lossless EmepyFDE at 10 nm
   cells - the arc index and its group index for the ring model.
2. Radiation loss from ``PMLModeSolver`` (validated, report 06) at
   R in {2, 3, 5} um; at larger radii it is below anything the solver can
   resolve (report 06 §6: 2e-12 dB/90 deg at R = 5 um already) and is set to
   zero.  PML convergence at R = 3 um: thickness x1.3, stretch 1+3j,
   standoff +0.5 um.
3. Sidewall roughness: Payne-Lacey / EIM for the straight guide with
   sigma = 2 nm, L_c = 50 nm (declared assumptions), scaled to each bend by
   the mode's sidewall field factor.

Output: reports/output/ring/ring_loss.json.
Run:  python examples/run_solver_job.py studies/ring/ring_loss.py
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dbeme.circuit.roughness import roughness_loss_db_per_cm, sidewall_factor  # noqa: E402
from dbeme.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from dbeme.fde.materials import silica, silicon  # noqa: E402
from dbeme.fde.pml import PMLModeSolver, turning_point  # noqa: E402

WIDTH, THICKNESS = 500e-9, 220e-9
LAMBDAS = (1.53e-6, 1.55e-6, 1.57e-6)
RADII_INDEX_UM = (5.0, 8.0, 10.0, 15.0)
CELL_INDEX = 10e-9
CELL_PML = 12e-9
PML_GEOMETRY = {2.0: (-0.9e-6, 3.0e-6, 1.0e-6), 3.0: (-1.0e-6, 3.6e-6, 1.1e-6), 5.0: (-1.0e-6, 4.8e-6, 1.2e-6)}
SIGMA, LC = 2e-9, 50e-9
OUT = os.path.join(ROOT, "reports", "output", "ring", "ring_loss.json")


def section():
    return FullEtchStrip(thickness=THICKNESS, core=silicon(out_of_range="raise"),
                         cladding=silica(out_of_range="raise"))


def te0(md):
    for i in range(len(md.neff)):
        if md.TE_pol[i] > 0.5 and np.real(md.neff[i]) > 1.5:
            return i
    raise RuntimeError("no TE0")


def db_per_cm(neff_imag, wavelength):
    return 4.342944819 * 2 * (2 * np.pi / wavelength) * neff_imag / 100.0


def main():
    cs = section()
    out = {"width_nm": WIDTH * 1e9, "thickness_nm": THICKNESS * 1e9, "lambdas_nm": [w * 1e9 for w in LAMBDAS],
           "index": {}, "radiation": {}, "roughness": {}}

    # 1. index and sidewall factor, lossless solver
    half = 1.4e-6
    for R in list(RADII_INDEX_UM) + [np.inf]:
        key = "straight" if not np.isfinite(R) else f"{R:g}"
        kappa = 0.0 if not np.isfinite(R) else 1.0 / (R * 1e-6)
        rec = {"neff": [], "sidewall_factor": None}
        for wl in LAMBDAS:
            t0 = time.time()
            fde = EmepyFDE(cross_section=cs, parameter_names=("top_width", "curvature"), num_modes=4,
                           wavelength=wl, window=(half, -0.7e-6, 0.7e-6), mesh=int(round(2 * half / CELL_INDEX)))
            md = fde.solve((WIDTH, kappa))
            i = te0(md)
            rec["neff"].append(float(np.real(md.neff[i])))
            if abs(wl - 1.55e-6) < 1e-12:
                F, per = sidewall_factor(md, i, (-WIDTH / 2, WIDTH / 2), (-THICKNESS / 2, THICKNESS / 2))
                rec["sidewall_factor"] = F
                rec["sidewall_factor_inner_outer"] = per
                rec["te_pol"] = float(md.TE_pol[i])
            print(f"  index R={key} lambda={wl*1e9:.0f}: n_eff={rec['neff'][-1]:.6f} [{time.time()-t0:.0f} s]")
        out["index"][key] = rec

    # 2. radiation, PML
    for R_um, (x_min, x_max, pml) in PML_GEOMETRY.items():
        R = R_um * 1e-6
        rec = {"loss_db_per_cm": [], "neff_pml": [], "turning_point_um": None, "convergence": {}}
        for wl in LAMBDAS:
            t0 = time.time()
            straight = PMLModeSolver(cs, wavelength=wl, window=(-1.0e-6, 1.0e-6, -0.7e-6, 0.7e-6),
                                     mesh=int(round(2.0e-6 / CELL_PML)), pml_thickness=0.0, pml_edges=(), num_modes=12)
            n_straight, _ = straight.solve({"top_width": WIDTH, "curvature": 0.0}, 2.45)
            solver = PMLModeSolver(cs, wavelength=wl, window=(x_min, x_max, -0.7e-6, 0.7e-6),
                                   mesh=int(round((x_max - x_min) / CELL_PML)), pml_thickness=pml, num_modes=14)
            value, info = solver.solve({"top_width": WIDTH, "curvature": 1.0 / R}, n_straight.real)
            rec["loss_db_per_cm"].append(db_per_cm(value.imag, wl))
            rec["neff_pml"].append([float(value.real), float(value.imag)])
            rec["turning_point_um"] = turning_point(n_straight.real, 1.444, R) * 1e6
            print(f"  PML R={R_um:g} lambda={wl*1e9:.0f}: n={value.real:.6f}+{value.imag:.2e}j "
                  f"-> {rec['loss_db_per_cm'][-1]:.3e} dB/cm, conf {info['selected_confinement']:.3f} [{time.time()-t0:.0f} s]")
            if R_um == 3.0 and abs(wl - 1.55e-6) < 1e-12:
                base = rec["loss_db_per_cm"][-1]
                for label, kw, win in (("pml_x1.3", {"pml_thickness": pml * 1.3}, (x_min, x_max + 0.3 * pml)),
                                       ("factor_1+3j", {"pml_thickness": pml, "pml_factor": 1 + 3j}, (x_min, x_max)),
                                       ("standoff_+0.5um", {"pml_thickness": pml}, (x_min, x_max + 0.5e-6))):
                    t0 = time.time()
                    s2 = PMLModeSolver(cs, wavelength=wl, window=(win[0], win[1], -0.7e-6, 0.7e-6),
                                       mesh=int(round((win[1] - win[0]) / CELL_PML)), num_modes=14, **kw)
                    v2, _ = s2.solve({"top_width": WIDTH, "curvature": 1.0 / R}, n_straight.real)
                    loss2 = db_per_cm(v2.imag, wl)
                    rec["convergence"][label] = {"loss_db_per_cm": loss2, "ratio": loss2 / base if base else None}
                    print(f"    {label}: {loss2:.3e} dB/cm (x{loss2/base:.3f}) [{time.time()-t0:.0f} s]")
        out["radiation"][f"{R_um:g}"] = rec

    # 3. roughness
    n_si = float(np.real(silicon().index(1.55e-6)))
    n_ox = float(np.real(silica().index(1.55e-6)))
    straight_neff = out["index"]["straight"]["neff"][1]
    loss_eim, det = roughness_loss_db_per_cm(WIDTH, THICKNESS, n_si, n_ox, 1.55e-6, SIGMA, LC)
    loss_full, det2 = roughness_loss_db_per_cm(WIDTH, THICKNESS, n_si, n_ox, 1.55e-6, SIGMA, LC, neff=straight_neff)
    F0 = out["index"]["straight"]["sidewall_factor"]
    out["roughness"] = {"sigma_nm": SIGMA * 1e9, "correlation_length_nm": LC * 1e9,
                        "straight_db_per_cm_eim_index": loss_eim, "straight_db_per_cm_mode_index": loss_full,
                        "details": det, "details_mode_index": det2,
                        "bend_enhancement": {k: v["sidewall_factor"] / F0 for k, v in out["index"].items()
                                             if v["sidewall_factor"] is not None}}
    print(f"  roughness (sigma={SIGMA*1e9:.0f} nm, Lc={LC*1e9:.0f} nm): {loss_eim:.2f} dB/cm (EIM index), "
          f"{loss_full:.2f} dB/cm (mode index); bend enhancement {out['roughness']['bend_enhancement']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
