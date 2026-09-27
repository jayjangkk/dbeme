"""Phase 1.2: the 2-D PML solver against a published SOI bend-loss measurement.

    Y. A. Vlasov and S. J. McNab, "Losses in single-mode silicon-on-insulator
    strip waveguides and bends", Optics Express 12(8), 1622-1631 (2004).
    https://doi.org/10.1364/OPEX.12.001622

445 x 220 nm SOI strip on a 2 um buried oxide, TE, 1500 nm.  Loss per **90
degree bend**, extracted by comparing serpentines of 10 and 20 bends:

    R = 1 um :  0.086 +- 0.005 dB
    R = 2 um :  0.013 +- 0.005 dB
    R = 5 um :  within the +-0.005 dB uncertainty, i.e. negligible

What is and is not comparable
-----------------------------
The authors are explicit that they do **not** separate the mechanisms, and name
three: radiation of the bent mode, mode mismatch at the straight-to-bend
junction, and enhanced sidewall-roughness scattering as the mode is pushed
against the outer wall.

This model computes the **first only**.  So the measurement is an upper bound on
what is being calculated here, and the interesting quantity is not the ratio at
any single radius but how the two curves *diverge*: where they track each other,
radiation dominates; where the measurement flattens above the model, it has
stopped being a radiation problem.

Run:  python examples/study_bend_loss_literature.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402
from dbeme.fde import FullEtchStrip  # noqa: E402
from dbeme.fde.materials import silica, silicon  # noqa: E402
from dbeme.fde.pml import PMLModeSolver, turning_point  # noqa: E402

WAVELENGTH = 1.500e-6
WIDTH, THICKNESS = 0.445e-6, 0.220e-6

#: Target cell size.  The straight guide converges to n_eff = 2.403 by ~13 nm;
#: at 23 nm it reports 1.82, which is not a small error but a different mode.
CELL = 12e-9

#: Table 2 of the paper, dB per 90 degree bend.
MEASURED = {1.0: (0.086, 0.005), 2.0: (0.013, 0.005), 5.0: (0.0, 0.005)}

#: Per radius: inner extent, outer extent, PML thickness.  The outer edge has to
#: clear the turning point and then hold the PML; the inner side never needs
#: more than about a micron, which is why the window is asymmetric.
GEOMETRY = {
    1.0: (-0.8e-6, 2.2e-6, 0.8e-6),
    2.0: (-0.9e-6, 3.0e-6, 1.0e-6),
    3.0: (-1.0e-6, 3.6e-6, 1.1e-6),
    5.0: (-1.0e-6, 4.8e-6, 1.2e-6),
}


def cross_section():
    return FullEtchStrip(
        thickness=THICKNESS,
        core=silicon(out_of_range="raise"),
        cladding=silica(out_of_range="raise"),
    )


def db_per_90(neff_imag, radius):
    k0 = 2 * np.pi / WAVELENGTH
    return 4.342944819 * 2 * k0 * neff_imag * (np.pi * radius / 2)


def main():
    print("Phase 1.2 - 2-D PML vs Vlasov & McNab, Opt. Express 12, 1622 (2004)")
    print("=" * 74)
    print(
        f"{WIDTH*1e9:.0f} x {THICKNESS*1e9:.0f} nm SOI strip, TE, "
        f"{WAVELENGTH*1e9:.0f} nm, SiO2 clad"
    )

    section = cross_section()
    straight_solver = PMLModeSolver(
        section, wavelength=WAVELENGTH,
        window=(-1.0e-6, 1.0e-6, -0.7e-6, 0.7e-6),
        mesh=int(round(2.0e-6 / CELL)),
        pml_thickness=0.0, pml_edges=(), num_modes=12,
    )
    t0 = time.time()
    straight, info = straight_solver.solve(
        {"top_width": WIDTH, "curvature": 0.0}, 2.40
    )
    print(
        f"straight n_eff = {straight.real:.6f}  (confinement "
        f"{info['selected_confinement']:.3f})  [{time.time()-t0:.0f} s]\n"
    )

    print(
        f"{'R (um)':>7} {'u_t (um)':>9} {'ramp':>6} {'grid':>10} "
        f"{'model dB/90':>12} {'measured dB/90':>16} {'ratio':>7} {'s':>5}"
    )
    rows = []
    for radius_um, (x_min, x_max, pml) in GEOMETRY.items():
        radius = radius_um * 1e-6
        t0 = time.time()
        solver = PMLModeSolver(
            section, wavelength=WAVELENGTH,
            window=(x_min, x_max, -0.7e-6, 0.7e-6),
            mesh=int(round((x_max - x_min) / CELL)),
            pml_thickness=pml, num_modes=14,
        )
        value, _ = solver.solve(
            {"top_width": WIDTH, "curvature": 1.0 / radius}, straight.real
        )
        model = db_per_90(value.imag, radius)
        turning = turning_point(straight.real, 1.444, radius)
        measured = MEASURED.get(radius_um)
        if measured and measured[0] > 0:
            shown = f"{measured[0]:.3f} +- {measured[1]:.3f}"
            ratio = f"{model/measured[0]:7.3f}"
        elif measured:
            shown = f"< {measured[1]:.3f}"
            ratio = "  -"
        else:
            shown, ratio = "-", "  -"
        rows.append({
            "radius_um": radius_um,
            "neff": [value.real, value.imag],
            "model_db_per_90": model,
            "measured_db_per_90": measured[0] if measured else None,
            "measured_uncertainty": measured[1] if measured else None,
            "turning_point_um": turning * 1e6,
            "index_ramp": float(np.exp(x_max / radius)),
            "grid": [len(solver._x), len(solver._y)],
        })
        print(
            f"{radius_um:7.1f} {turning*1e6:9.2f} {np.exp(x_max/radius):6.1f} "
            f"{len(solver._x)}x{len(solver._y):<5} {model:12.3e} {shown:>16} "
            f"{ratio} {time.time()-t0:5.0f}"
        )

    one, two = rows[0]["model_db_per_90"], rows[1]["model_db_per_90"]
    print(
        f"\n   radiation falls {one/two:.0f}x from R = 1 to 2 um, while the "
        f"measurement falls only {0.086/0.013:.1f}x."
    )
    print(
        "   So radiation dominates at 1 um and has stopped mattering by 2 um;\n"
        "   beyond that the measured loss is mode mismatch and roughness, which\n"
        "   a PML cannot help with and EME already captures for the mismatch."
    )

    figure(rows, straight)

    payload = {
        "reference": {
            "citation": (
                "Y. A. Vlasov and S. J. McNab, Opt. Express 12(8), 1622 (2004)"
            ),
            "doi": "10.1364/OPEX.12.001622",
            "units": "dB per 90 degree bend",
            "note": (
                "measurement is radiation + junction mismatch + sidewall "
                "roughness; the authors do not separate them"
            ),
        },
        "geometry": {
            "width_nm": WIDTH * 1e9,
            "thickness_nm": THICKNESS * 1e9,
            "wavelength_nm": WAVELENGTH * 1e9,
            "cladding": "SiO2 (the fabricated device may be air clad)",
            "cell_nm": CELL * 1e9,
        },
        "straight_neff": straight.real,
        "rows": rows,
    }
    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "output",
    )
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "pml_literature.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote reports/output/pml_literature.json")

    import shutil

    from _plotting import OUTPUT_DIR

    shutil.copy2(os.path.join(OUTPUT_DIR, "pml_2_literature.png"),
                 os.path.join(report_dir, "pml_2_literature.png"))
    print("wrote reports/output/pml_2_literature.png")
    return payload


def figure(rows, straight):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    radii = [r["radius_um"] for r in rows]
    model = [r["model_db_per_90"] for r in rows]
    ax.semilogy(radii, model, "o-", color="C0", ms=6,
                label="this model: radiation only")

    measured_r = [r["radius_um"] for r in rows if r["measured_db_per_90"]]
    measured_v = [r["measured_db_per_90"] for r in rows if r["measured_db_per_90"]]
    measured_e = [r["measured_uncertainty"] for r in rows if r["measured_db_per_90"]]
    ax.errorbar(measured_r, measured_v, yerr=measured_e, fmt="s", color="C3",
                ms=7, capsize=4, label="Vlasov & McNab 2004 (measured)")
    ax.axhline(0.005, color="C3", ls=":", lw=1.0)
    ax.text(3.1, 0.0058, "measurement floor 0.005 dB", fontsize=7, color="C3")

    ax.set_xlabel("bend radius (um)")
    ax.set_ylabel("loss per 90$\\degree$ bend (dB)")
    ax.set_title(
        "445 x 220 nm SOI strip, TE, 1500 nm\n"
        "the gap is what a PML cannot explain", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(1e-13, 1.0)
    save(fig, "pml_2_literature.png")


if __name__ == "__main__":
    main()
