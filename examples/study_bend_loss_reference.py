"""Phase 1 gate for the PML task: the ruler, and whether the PML matches it.

``tasks/02_pml_backend.md`` Phase 1 says to build the reference *before* the
solver, and sets a stop condition: if the framework cannot reproduce the Airy
result to within ~10 % over three decades, the problem is in the formulation
and a 2-D solver will only hide it.

This script produces the gate table.

    python examples/study_bend_loss_reference.py

Three independent calculations of the same quantity:

* **Airy** - the semi-analytic bent-slab solution, exact for the linearised
  conformal map;
* **WKB** - the tunnelling exponent alone, which fixes the *slope* of the loss
  against radius but not its prefactor;
* **FD + PML** - finite difference on a complex-stretched grid, i.e. the
  machinery a real PML backend would use.

The reference is deliberately **weakly guiding**.  A 220 nm Si slab confines so
hard that its bend loss is below double precision for any radius at which the
linearised conformal map is still valid - the barrier action grows as
3.3e7 * R, so R = 2 um already gives exp(-65).  A weak guide puts the loss in a
measurable range while keeping half_width / radius under 1 %.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402
from em_simulation.reference import BentSlab, attenuation_db_per_cm  # noqa: E402
from em_simulation.reference.fd1d_pml import solve_bent_slab  # noqa: E402

#: Weakly guiding, so the loss is representable where the model is valid.
SLAB = dict(core_index=1.50, clad_index=1.444, half_width=0.75e-6)

RADII = np.array(
    [120e-6, 150e-6, 200e-6, 250e-6, 300e-6, 350e-6, 400e-6, 450e-6, 500e-6]
)

#: The Phase 1 stop condition.
GATE_TOLERANCE_PERCENT = 10.0

#: Phase 3 asks for stability under these; checked here while it is cheap.
PERTURBATIONS = {
    "baseline": {},
    "PML thickness x1.5": {"pml_thickness": 6e-6},
    "stretch factor x2": {"factor": 1 + 4j},
    "standoff x1.5": {"standoff": 2.25e-6},
    "grid step /2": {"dx": 10e-9},
    "inner extent x1.5": {"inner": 6e-6},
    "grading order 3": {"order": 3},
    "PML thickness x0.5": {"pml_thickness": 2e-6},
    "stretch factor /2": {"factor": 1 + 1j},
}

#: Perturbations the task actually names as the stability criterion.  The
#: reduced-parameter cases are reported too, but they are probes of where
#: convergence *ends*, not part of the criterion.
CRITERION_KEYS = ("PML thickness x1.5", "stretch factor x2", "standoff x1.5")

BASE = dict(
    dx=20e-9, inner=4e-6, standoff=1.5e-6, pml_thickness=4e-6,
    factor=1 + 2j, order=2,
)


def main():
    slab = BentSlab(**SLAB)
    print("Phase 1 gate - bend loss reference vs a complex-stretched PML")
    print("=" * 74)
    print(
        f"slab: core {slab.core_index}, clad {slab.clad_index}, "
        f"2a = {2*slab.half_width*1e6:.2f} um, lambda = {slab.wavelength*1e9:.0f} nm"
    )
    print(f"straight n_eff = {slab.straight_neff():.6f}")

    # ---- 1. the reference -------------------------------------------------
    t0 = time.time()
    exact = slab.sweep(RADII)
    print(f"\n1. Airy reference over {len(RADII)} radii  [{time.time()-t0:.2f} s]")

    wkb = np.array([slab.wkb_exponent(n, r) for n, r in zip(exact, RADII)])
    offset = -np.log(exact.imag) - wkb
    slope_measured = np.polyfit(RADII, -np.log(exact.imag), 1)[0]
    slope_wkb = np.polyfit(RADII, wkb, 1)[0]
    print(
        f"   WKB cross-check: d(-ln Im)/dR = {slope_measured:.6g} /m against "
        f"{slope_wkb:.6g} /m, ratio {slope_measured/slope_wkb:.4f}"
    )
    print(
        f"   prefactor offset constant to "
        f"{(offset.max()-offset.min())/offset.mean()*100:.2f} % "
        f"(WKB predicts it is radius independent)"
    )

    # ---- 2. the gate ------------------------------------------------------
    print("\n2. FD + PML against the reference")
    print(
        f"   {'R (um)':>7} {'a/R':>7} {'Airy Im':>12} {'FD Im':>12} "
        f"{'err %':>7} {'dB/cm':>10} {'u_t (um)':>9}"
    )
    computed, errors = [], []
    for radius, reference in zip(RADII, exact):
        value, info = solve_bent_slab(slab, radius, neff_target=reference.real,
                                      **BASE)
        error = abs(value.imag / reference.imag - 1) * 100
        computed.append(value)
        errors.append(error)
        print(
            f"   {radius*1e6:7.0f} {slab.half_width/radius:7.4f} "
            f"{reference.imag:12.4e} {value.imag:12.4e} {error:7.2f} "
            f"{attenuation_db_per_cm(reference.imag, slab.wavelength):10.3e} "
            f"{info['turning_point']*1e6:9.2f}"
        )
    computed = np.array(computed)
    errors = np.array(errors)
    decades = float(np.log10(exact.imag.max() / exact.imag.min()))
    passed = decades >= 3.0 and errors.max() <= GATE_TOLERANCE_PERCENT
    print(
        f"\n   spans {decades:.2f} decades, worst error {errors.max():.2f} % "
        f"(criterion: >= 3 decades within {GATE_TOLERANCE_PERCENT:.0f} %)"
    )
    print(f"   GATE: {'PASS' if passed else 'FAIL'}")

    # ---- 3. PML parameter stability --------------------------------------
    print("\n3. Stability of the PML parameters, at R = 250 um")
    radius = 250e-6
    reference = exact[list(RADII).index(radius)]
    stability = {}
    for name, override in PERTURBATIONS.items():
        keywords = dict(BASE)
        keywords.update(override)
        value, _ = solve_bent_slab(slab, radius, neff_target=reference.real,
                                   **keywords)
        stability[name] = float(value.imag)
        mark = " <- criterion" if name in CRITERION_KEYS else ""
        print(
            f"   {name:22s} Im = {value.imag:.5e}  "
            f"{value.imag/reference.imag - 1:+7.2%}{mark}"
        )

    criterion = np.array([stability[k] for k in CRITERION_KEYS])
    worst_criterion = np.max(
        np.abs(criterion / stability["baseline"] - 1)
    ) * 100
    print(
        f"\n   worst shift under the three named perturbations: "
        f"{worst_criterion:.2f} % (criterion < 5 %)"
    )
    print(
        "   note: the reduced-parameter probes are not part of the criterion. "
        "A 2 um PML\n         and a 1+1j stretch are simply not converged, "
        "which is what they show."
    )

    figure(slab, RADII, exact, computed, errors, stability, reference)

    payload = {
        "slab": {k: float(v) for k, v in SLAB.items()},
        "wavelength_nm": slab.wavelength * 1e9,
        "straight_neff": slab.straight_neff(),
        "radii_um": (RADII * 1e6).tolist(),
        "airy_neff_real": exact.real.tolist(),
        "airy_neff_imag": exact.imag.tolist(),
        "fd_neff_real": computed.real.tolist(),
        "fd_neff_imag": computed.imag.tolist(),
        "error_percent": errors.tolist(),
        "decades": decades,
        "worst_error_percent": float(errors.max()),
        "gate_passed": bool(passed),
        "wkb": {
            "exponent": wkb.tolist(),
            "slope_measured_per_m": float(slope_measured),
            "slope_wkb_per_m": float(slope_wkb),
            "slope_ratio": float(slope_measured / slope_wkb),
        },
        "stability": stability,
        "stability_criterion_worst_percent": float(worst_criterion),
        "pml_settings": {
            k: (str(v) if isinstance(v, complex) else v) for k, v in BASE.items()
        },
    }
    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "output",
    )
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "pml_phase1_gate.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote reports/output/pml_phase1_gate.json")

    import shutil

    from _plotting import OUTPUT_DIR

    for name in ("pml_1_gate.png",):
        shutil.copy2(os.path.join(OUTPUT_DIR, name),
                     os.path.join(report_dir, name))
        print(f"wrote reports/output/{name}")
    return payload


def figure(slab, radii, exact, computed, errors, stability, reference):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.6))
    fig.subplots_adjust(wspace=0.34)

    ax1.semilogy(radii * 1e6, exact.imag, "o-", ms=5, color="C0",
                 label="Airy (reference)")
    ax1.semilogy(radii * 1e6, computed.imag, "x--", ms=7, color="C3",
                 label="FD + PML")
    ax1.set_xlabel("bend radius (um)")
    ax1.set_ylabel(r"$\mathrm{Im}(n_{eff})$")
    ax1.set_title("bend loss over 5.6 decades", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.plot(radii * 1e6, errors, "d-", ms=5, color="C3")
    ax2.axhline(GATE_TOLERANCE_PERCENT, color="C2", ls="--", lw=1.0)
    ax2.text(radii[0] * 1e6, GATE_TOLERANCE_PERCENT * 1.05,
             "gate: 10 %", fontsize=7, color="C2")
    ax2.set_ylim(0, max(GATE_TOLERANCE_PERCENT * 1.25, errors.max() * 1.3))
    ax2.set_xlabel("bend radius (um)")
    ax2.set_ylabel("|FD - Airy| / Airy  (%)")
    ax2.set_title("the gate", fontsize=9)

    names = list(stability)
    shifts = [
        (stability[n] / reference.imag - 1) * 100 for n in names
    ]
    colours = ["C2" if n in CRITERION_KEYS else
               ("C0" if n == "baseline" else "C7") for n in names]
    ax3.barh(range(len(names)), shifts, color=colours)
    ax3.axvline(5, color="C3", ls="--", lw=0.8)
    ax3.axvline(-5, color="C3", ls="--", lw=0.8)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=7)
    ax3.invert_yaxis()
    ax3.set_xlabel("shift vs Airy (%)")
    ax3.set_title("PML parameter stability (green = criterion)", fontsize=9)

    save(fig, "pml_1_gate.png")


if __name__ == "__main__":
    main()
