"""Demo 6 - sidewall angle in the polarization rotator.

Real silicon etches are not vertical.  A 220 nm rib comes out of the tool as a
trapezoid: wider at its base than at the top width you drew, by a few
nanometres per side.  This asks what that does to the bi-level taper rotator of
[report 02](../reports/02_polarization_rotator.md), with the two etch steps at
85 and 87 degrees from horizontal.

The answer is more interesting than a tolerance number, because a trapezoid has
**no horizontal mirror plane**.  That is the same symmetry the partial etch was
introduced to break, so the sidewall angle is a second, independent rotation
mechanism sitting on top of the designed one - not merely a perturbation of it.

Three things are measured:

1. **The mechanism on its own.** A plain full-etch strip, no rib. Its TE1 and
   TM0 branches cross near 650 nm.  With vertical walls they pass straight
   through each other; slant the walls and the same crossing opens into an
   anti-crossing.  Nothing else changes.
2. **The effect on the designed device.** Where the rotator's anti-crossing
   moves and how wide it gets, vertical versus 85/87.
3. **Tolerance.** Conversion efficiency against sidewall angle, which is the
   number a process engineer actually wants.

Run:  python examples/demo_sidewall_angle.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import propagation_axis, save  # noqa: E402
from em_simulation import (  # noqa: E402
    EME,
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
)
from em_simulation.fde import BiLevelStrip, EmepyFDE, FullEtchStrip  # noqa: E402
from em_simulation.matrix_calculation_tool import _redheffer_star_product  # noqa: E402
from em_simulation.validation import (  # noqa: E402
    check_branch_tracking,
    check_power_conservation,
    check_reciprocity,
    check_reflection_symmetry,
    check_slicing_independence,
)

import demo_polarization_rotator as rotator_demo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERTICAL_DATASET = os.path.join(ROOT, "datasets", "Si_bilevel_220_90nm")
ANGLED_DATASET = os.path.join(ROOT, "datasets", "Si_bilevel_220_90nm_sw85_87")

BOTTOM_ANGLE, TOP_ANGLE = 85.0, 87.0
LENGTH = rotator_demo.LENGTH
WINDOW = (2.2e-6, -0.8e-6, 0.8e-6)
MESH = 200

#: Angles to sweep for the tolerance study, on the *full etch*.  The partial
#: etch is held two degrees steeper throughout, as 85/87 implies.
SWEEP_ANGLES = (90.0, 88.0, 86.0, 85.0, 84.0, 82.0)


def angle_pair(bottom):
    """Partial etch runs 2 degrees steeper than the full etch."""
    return bottom, min(90.0, bottom + 2.0)


# ------------------------------------------------------------------- physics


def strip_backend(angle):
    return EmepyFDE(
        cross_section=FullEtchStrip(thickness=220e-9, sidewall_angle=angle),
        num_modes=4,
        wavelength=1.55e-6,
        window=WINDOW,
        mesh=MESH,
    )


def strip_branches(backend, widths):
    """Second and third guided branches of a plain strip, vs width.

    Returns effective indices and TE fractions ordered by ``n_eff``, which is
    what the solver gives - deliberately *not* tracked, because the point is to
    watch the labels swap at a crossing and not swap at an anti-crossing.
    """
    neff = np.full((len(widths), 2), np.nan)
    te = np.full((len(widths), 2), np.nan)
    for k, width in enumerate(widths):
        modes = backend.solve((round(float(width), 12), 0.0))
        values = np.real(modes.neff)
        keep = sorted(
            [i for i in range(len(values)) if values[i] > 1.6],
            key=lambda i: -values[i],
        )
        for slot, index in enumerate(keep[1:3]):
            neff[k, slot] = values[index]
            te[k, slot] = modes.TE_pol[index]
    return neff, te


def rib_anticrossing(bottom, top, samples=25):
    """Closest approach of the rotator's hybridising pair, and where it is."""
    backend = EmepyFDE(
        cross_section=BiLevelStrip(
            thickness=220e-9,
            slab_thickness=90e-9,
            bottom_sidewall_angle=bottom,
            top_sidewall_angle=top,
        ),
        parameter_names=("w_core", "w_slab"),
        num_modes=4,
        wavelength=1.55e-6,
        window=WINDOW,
        mesh=MESH,
    )
    schedule = rotator_demo.width_schedule(LENGTH)
    best = None
    for z in np.linspace(0.02 * LENGTH, 0.6 * LENGTH, samples):
        w_core = float(schedule["w_core"](z))
        w_slab = float(schedule["w_slab"](z))
        modes = backend.solve((round(w_core, 12), round(w_slab, 12)))
        values = np.real(modes.neff)
        keep = [i for i in range(len(values)) if values[i] > 1.55]
        if len(keep) < 2:
            continue
        pair = sorted(keep, key=lambda i: abs(modes.TE_pol[i] - 0.5))[:2]
        imbalance = min(abs(modes.TE_pol[i] - 0.5) for i in pair)
        if best is None or imbalance < best["imbalance"]:
            best = {
                "imbalance": float(imbalance),
                "gap": float(abs(values[pair[0]] - values[pair[1]])),
                "z_um": float(z * 1e6),
                "w_slab_nm": w_slab * 1e9,
                "te": [float(modes.TE_pol[i]) for i in pair],
            }
    return best


def conversion(extractor, resolution=150, length=LENGTH):
    """TM0 -> TE1 conversion for whatever cross section the extractor carries."""
    path = DirectParametricPath(
        extractor,
        rotator_demo.width_schedule(LENGTH),
        total_length=LENGTH,
        resolution=resolution,
    )
    path._verbose = False
    path.calc_output_data()
    n_modes = path.output_data["neff"].shape[1] // 2
    te0, rotating = rotator_demo.identify_branches(path)

    eme = EME(path, force_unitary=True)
    eme.calc_Smatrix()
    smatrix = eme.propagator._find_Smatrix_new_length(length)
    lumped = smatrix[0]
    for j in range(1, len(smatrix)):
        lumped = _redheffer_star_product(lumped, smatrix[j])
    launch = np.zeros(2 * n_modes, dtype=complex)
    launch[rotating] = 1.0
    out = np.abs(lumped @ launch) ** 2

    launch_te0 = np.zeros(2 * n_modes, dtype=complex)
    launch_te0[te0] = 1.0
    out_te0 = np.abs(lumped @ launch_te0) ** 2
    return {
        "tm0_to_te1": float(out[rotating]),
        "te0_through": float(out_te0[te0]),
        "output_te_fraction": float(np.real(path.output_data["TE_pol"])[-1, rotating]),
    }


class _AngleExtractor(DataExtractor):
    """A DataExtractor whose cross section carries a chosen sidewall angle.

    Reuses the vertical dataset's metadata - same stack, wavelength, mode count
    and window - and swaps only the geometry, so the angle is the one thing
    that differs between runs.
    """

    def __init__(self, bottom, top, **kwargs):
        super().__init__(VERTICAL_DATASET, **kwargs)
        self.backend = EmepyFDE(
            cross_section=BiLevelStrip(
                thickness=220e-9,
                slab_thickness=90e-9,
                bottom_sidewall_angle=bottom,
                top_sidewall_angle=top,
            ),
            parameter_names=tuple(self.parameter_names),
            num_modes=self.mode_numbers,
            wavelength=self.wavelength,
            window=(2.2e-6, -0.8e-6, 0.8e-6),
            mesh=220,
        )


# -------------------------------------------------------------------- figures


def figure_mechanism(widths, vertical, angled):
    """A crossing at 90 degrees, an anti-crossing at 85."""
    (n_v, te_v), (n_a, te_a) = vertical, angled

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))
    fig.subplots_adjust(wspace=0.33)

    # Geometry, drawn from the index profile itself.
    ax = axes[0]
    x = np.linspace(-0.6e-6, 0.6e-6, 601)
    y = np.linspace(-0.2e-6, 0.2e-6, 201)
    for angle, colour, style in ((90.0, "C0", "-"), (85.0, "C3", "--")):
        n = FullEtchStrip(thickness=220e-9, sidewall_angle=angle).index(
            x, y, {"top_width": 640e-9, "curvature": 0.0}
        )
        ax.contour(x * 1e9, y * 1e9, n.T, levels=[2.5], colors=colour,
                   linestyles=style, linewidths=1.6)
        ax.plot([], [], colour, ls=style, label=f"{angle:.0f}$\\degree$ sidewall")
    ax.set_xlim(-400, 400)
    ax.set_ylim(-160, 160)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("640 nm strip: the trapezoid has no\nhorizontal mirror plane", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(False)

    for ax, (neff, te), title in (
        (axes[1], vertical, "vertical walls: they cross"),
        (axes[2], angled, f"{BOTTOM_ANGLE:.0f}$\\degree$ walls: they anti-cross"),
    ):
        for slot in (0, 1):
            points = ax.scatter(
                widths * 1e9, neff[:, slot], c=te[:, slot], cmap="coolwarm",
                vmin=0, vmax=1, s=22, zorder=3,
            )
            ax.plot(widths * 1e9, neff[:, slot], color="0.75", lw=0.8, zorder=2)
        gaps = np.abs(neff[:, 0] - neff[:, 1])
        k = int(np.nanargmin(gaps))
        ax.annotate(
            f"$\\Delta n$ = {gaps[k]:.4f}",
            (widths[k] * 1e9, np.nanmean(neff[k])),
            textcoords="offset points", xytext=(10, -16), fontsize=8,
        )
        ax.set_xlabel("strip width (nm)")
        ax.set_ylabel("$n_{eff}$")
        ax.set_title(title, fontsize=9)
    bar = fig.colorbar(points, ax=axes[2])
    bar.set_label("TE fraction", fontsize=8)

    save(fig, "sidewall_1_mechanism.png")


def figure_tolerance(angles, results, reference):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.3)

    conv = np.array([r["tm0_to_te1"] for r in results])
    ax1.plot(angles, 100 * conv, "o-", color="C3", ms=5)
    ax1.axhline(99.0, color="grey", lw=0.7, ls=":")
    ax1.text(angles[0], 99.05, "99 %", fontsize=7, color="grey")
    ax1.set_xlabel("full-etch sidewall angle (degrees from horizontal)")
    ax1.set_ylabel("TM0 -> TE1 conversion (%)")
    ax1.set_title(
        f"tolerance at the published L = {LENGTH*1e6:.0f} um\n"
        "(partial etch held 2$\\degree$ steeper)", fontsize=9)
    ax1.invert_xaxis()

    gaps = np.array([r["gap"] for r in reference])
    slabs = np.array([r["w_slab_nm"] for r in reference])
    # Plotted against the vertical-wall value, so "flat" is unambiguous rather
    # than an artefact of a zoomed axis.
    ax2.plot(angles, gaps / gaps[0], "s-", color="C0", ms=5)
    ax2.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax2.set_ylim(0.0, 2.0)
    ax2.set_xlabel("full-etch sidewall angle (degrees)")
    ax2.set_ylabel("$\\Delta n_{eff}$ / value at 90$\\degree$")
    ax2.invert_xaxis()
    ax2.set_title(
        "the rib's anti-crossing barely notices\n"
        f"($\\Delta n$ = {gaps.min():.3f}-{gaps.max():.3f} throughout)",
        fontsize=9,
    )
    # The crossing position is located on a 25-point scan, so it is quantised;
    # the single apparent step is that quantisation, not a real shift.
    spread = float(np.max(slabs) - np.min(slabs))
    ax2.text(
        0.5, 0.1,
        f"crossing position spans {spread:.0f} nm\n"
        "(located on a 25-point scan, so quantised)",
        transform=ax2.transAxes, ha="center", fontsize=7.5, color="grey",
    )

    save(fig, "sidewall_2_tolerance.png")


def figure_validation(lengths, dataset, direct):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.3))
    fig.subplots_adjust(wspace=0.32)
    ax1.plot(lengths * 1e6, direct, "o-", ms=3, label="direct EME")
    ax1.plot(lengths * 1e6, dataset, "s--", ms=3, label="dataset EME")
    ax1.set_xlabel("taper length (um)")
    ax1.set_ylabel("TM0 -> TE1 conversion")
    ax1.set_title(f"{BOTTOM_ANGLE:.0f}/{TOP_ANGLE:.0f}$\\degree$ walls, both methods",
                  fontsize=9)
    ax1.legend(fontsize=8)
    ax2.semilogy(lengths * 1e6, np.abs(np.asarray(dataset) - np.asarray(direct)),
                 "d-", color="C3")
    ax2.set_xlabel("taper length (um)")
    ax2.set_ylabel("|dataset - direct|")
    ax2.set_title("residual, from the width grid", fontsize=9)
    save(fig, "sidewall_3_validation.png")


# ----------------------------------------------------------------------- main


def main():
    print("Dataset-based EME - sidewall angle in the polarization rotator")
    print("=" * 72)

    # ---- 1. the mechanism, on a plain strip -----------------------------
    print("\n1. does the slant alone turn a crossing into an anti-crossing?")
    widths = np.arange(600, 761, 10) * 1e-9
    t0 = time.time()
    vertical = strip_branches(strip_backend(90.0), widths)
    angled = strip_branches(strip_backend(BOTTOM_ANGLE), widths)
    for tag, (neff, te) in (("vertical", vertical), (f"{BOTTOM_ANGLE:.0f} deg", angled)):
        gaps = np.abs(neff[:, 0] - neff[:, 1])
        k = int(np.nanargmin(gaps))
        print(f"   plain 220 nm strip, {tag:>8} walls: closest approach "
              f"dn = {gaps[k]:.4f} at w = {widths[k]*1e9:.0f} nm, "
              f"TE fractions {te[k, 0]:.3f} / {te[k, 1]:.3f}")
    print(f"   [{time.time()-t0:.0f} s]")
    figure_mechanism(widths, vertical, angled)

    # ---- 2. the rotator's anti-crossing --------------------------------
    print("\n2. the rotator's TM0/TE1 anti-crossing vs sidewall angle")
    reference = []
    for angle in SWEEP_ANGLES:
        bottom, top = angle_pair(angle)
        found = rib_anticrossing(bottom, top)
        reference.append(found)
        print(f"   walls {bottom:4.0f}/{top:4.0f} deg: dn = {found['gap']:.4f} "
              f"at z = {found['z_um']:5.1f} um (w_slab = {found['w_slab_nm']:.0f} nm), "
              f"TE {found['te'][0]:.2f}/{found['te'][1]:.2f}", flush=True)

    # ---- 3. tolerance ---------------------------------------------------
    print(f"\n3. conversion at the published L = {LENGTH*1e6:.0f} um, vs angle")
    results = []
    for angle in SWEEP_ANGLES:
        bottom, top = angle_pair(angle)
        t0 = time.time()
        extractor = _AngleExtractor(bottom, top, cache_size=2048)
        value = conversion(extractor)
        results.append(value)
        print(f"   walls {bottom:4.0f}/{top:4.0f} deg: TM0->TE1 = "
              f"{100*value['tm0_to_te1']:6.2f} %, TE0 through = "
              f"{100*value['te0_through']:6.2f} %, output TE fraction "
              f"{value['output_te_fraction']:.3f}   [{time.time()-t0:.0f} s]", flush=True)
    figure_tolerance(list(SWEEP_ANGLES), results, reference)

    # ---- 4. DBEME vs direct EME at 85/87 -------------------------------
    print(f"\n4. dataset EME vs direct EME at {BOTTOM_ANGLE:.0f}/{TOP_ANGLE:.0f} deg")
    print("-" * 72)
    updater = DataUpdater(ANGLED_DATASET)
    t0 = time.time()
    path = ParametricPath(
        updater, rotator_demo.width_schedule(LENGTH), total_length=LENGTH
    )
    path.calc_output_data()
    build_seconds = time.time() - t0
    te0, rotating = rotator_demo.identify_branches(path)
    n_modes = path.output_data["neff"].shape[1] // 2

    checks = [
        check_reciprocity(path),
        check_reflection_symmetry(path),
        check_power_conservation(path),
        check_branch_tracking(path)[0],
        check_slicing_independence(
            ParametricPath(updater, rotator_demo.width_schedule(LENGTH),
                           total_length=LENGTH, resolution=400, verbose=False),
            ParametricPath(updater, rotator_demo.width_schedule(LENGTH),
                           total_length=LENGTH, resolution=1600, verbose=False),
        ),
    ]
    print("\n   sanity gate")
    for check in checks:
        print("   " + str(check))

    sweep_lengths = np.array([25e-6, 50e-6, 100e-6, 150e-6, 200e-6, 300e-6])
    dataset_curve = []
    eme = EME(path, force_unitary=True)
    eme.calc_Smatrix()
    for L in sweep_lengths:
        smatrix = eme.propagator._find_Smatrix_new_length(L)
        lumped = smatrix[0]
        for j in range(1, len(smatrix)):
            lumped = _redheffer_star_product(lumped, smatrix[j])
        launch = np.zeros(2 * n_modes, dtype=complex)
        launch[rotating] = 1.0
        dataset_curve.append(float((np.abs(lumped @ launch) ** 2)[rotating]))

    extractor = _AngleExtractor(BOTTOM_ANGLE, TOP_ANGLE, cache_size=2048)
    t0 = time.time()
    direct_curve = [conversion(extractor, length=L)["tm0_to_te1"] for L in sweep_lengths]
    direct_seconds = time.time() - t0

    print(f"\n   {'length':>8}  {'dataset':>9}  {'direct':>9}  {'diff':>9}")
    for L, a, b in zip(sweep_lengths, dataset_curve, direct_curve):
        print(f"   {L*1e6:7.0f}u  {a:9.5f}  {b:9.5f}  {abs(a - b):9.2e}")
    max_difference = float(np.max(np.abs(np.array(dataset_curve) - np.array(direct_curve))))
    print(f"   max |difference| = {max_difference:.2e}")
    print(f"   dataset built in {build_seconds:.0f} s, direct sweep {direct_seconds:.0f} s")
    figure_validation(sweep_lengths, dataset_curve, direct_curve)
    updater.save_data()

    # ---- output ---------------------------------------------------------
    payload = {
        "angles": list(SWEEP_ANGLES),
        "angle_pairs": [list(angle_pair(a)) for a in SWEEP_ANGLES],
        "length_um": LENGTH * 1e6,
        "strip_mechanism": {
            "widths_nm": (widths * 1e9).tolist(),
            "vertical_neff": np.nan_to_num(vertical[0], nan=0.0).tolist(),
            "vertical_te": np.nan_to_num(vertical[1], nan=0.0).tolist(),
            "angled_neff": np.nan_to_num(angled[0], nan=0.0).tolist(),
            "angled_te": np.nan_to_num(angled[1], nan=0.0).tolist(),
        },
        "rib_anticrossing": reference,
        "tolerance": results,
        "checks": [
            {"name": c.name, "criterion": c.criterion,
             "measured": None if np.isnan(c.measured) else float(c.measured),
             "passed": bool(c.passed), "note": c.note}
            for c in checks
        ],
        "validation": {
            "lengths_um": (sweep_lengths * 1e6).tolist(),
            "dataset": dataset_curve,
            "direct": direct_curve,
            "max_difference": max_difference,
            "dataset_build_seconds": build_seconds,
            "direct_seconds": direct_seconds,
            "dataset_points": len(updater.neff),
        },
    }
    report_dir = os.path.join(ROOT, "reports", "output")
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "sidewall_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")

    import shutil

    from _plotting import OUTPUT_DIR

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith("sidewall_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote {os.path.relpath(os.path.join(report_dir, name), ROOT)}")

    return payload


if __name__ == "__main__":
    main()
