"""Demo 5 - bi-level taper polarization rotator (TM0 -> TE1).

Stage 1 of the polarization rotator-splitter in

    W. D. Sacher, T. Barwicz, B. J. F. Taylor and J. K. S. Poon,
    "Polarization rotator-splitters in standard active silicon photonics
    platforms", Opt. Express 22(4), 3777 (2014).  doi:10.1364/OE.22.003777

The adiabatic coupler of demo 4 separates TE0 from TE1.  This stage is what
puts the light in TE1 in the first place: it rotates the incoming TM0 into TE1
so that a TE-only splitter can then handle both polarizations.

How it works
------------
A buried strip with symmetric cladding has a horizontal mirror plane, and TE and
TM modes belong to different symmetry classes under it - they cannot couple at
all, and a rotator built on one returns exactly zero conversion.  Etching only
part way through, leaving a slab on the *bottom* of the device layer, destroys
that mirror plane.  TM0 and TE1 are then free to hybridise, and because their
effective indices cross as the slab widens, a slow enough taper carries TM0
continuously into TE1.

The slab is centred on the core, so the *vertical* mirror plane at x = 0
survives.  That is fine: under ``x -> -x`` the odd profile of TE1's ``Ex``
cancels against the sign flip ``Ex`` itself picks up, so TM0 and TE1 sit in the
same class and may couple.  Only the horizontal mirror has to go.

Geometry follows the published device: the slab widens 450 -> 1550 nm and back
to 850 nm while the core goes 450 -> 550 -> 850 nm, over 100 um.  At both ends
slab and core coincide, so the taper starts and finishes as a plain strip - and
its 850 nm output is exactly the coupler's 850 nm input.

Run:  python examples/demo_polarization_rotator.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from _plotting import propagation_axis, save  # noqa: E402
from em_simulation import (  # noqa: E402
    EME,
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
)
from em_simulation.matrix_calculation_tool import _redheffer_star_product  # noqa: E402
from em_simulation.validation import (  # noqa: E402
    check_branch_tracking,
    check_mode_basis_convergence,
    check_power_conservation,
    check_reciprocity,
    check_reflection_symmetry,
    check_slicing_independence,
    format_table,
    transmission,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "datasets", "Si_bilevel_220_90nm")

# Sacher et al. / the published reproduction of it: the bi-level taper polygon
# runs w_1 -> w_pes -> w_3 in the slab and w_1 -> w_2 -> w_3 in the core.
W_IN = 450e-9
W_CORE_MID, W_SLAB_MID = 550e-9, 1550e-9
W_OUT = 850e-9
LENGTH = 100e-6


def width_schedule(length=LENGTH):
    """Piecewise-linear core and slab widths, peaking at the midpoint."""

    def ramp(z, start, mid, end, L):
        z = np.asarray(z, dtype=float)
        first = start + (mid - start) * (2.0 * z / L)
        second = mid + (end - mid) * (2.0 * z / L - 1.0)
        return np.where(z <= 0.5 * L, first, second)

    return {
        "w_core": lambda z, L=length: ramp(z, W_IN, W_CORE_MID, W_OUT, L),
        "w_slab": lambda z, L=length: ramp(z, W_IN, W_SLAB_MID, W_OUT, L),
    }


def build(du, length=LENGTH, **kwargs):
    path = ParametricPath(du, width_schedule(length), total_length=length, **kwargs)
    path.calc_output_data()
    return path


# ------------------------------------------------------------------- analysis


def identify_branches(path):
    """Find the rotating branch and the TE0 branch.

    The rotating branch is identified by what it *does*: it enters TM-polarised
    and leaves TE-polarised.  TE0 is the branch that is TE at both ends and has
    the highest effective index - it takes no part in the rotation.
    """
    te = np.real(path.output_data["TE_pol"])
    neff = np.real(path.output_data["neff"])
    mask = path.output_data["radiation_mode_mask"]
    n_modes = neff.shape[1] // 2

    end_to_end = [i for i in range(n_modes) if not np.any(mask[:, i])]
    rotating = [i for i in end_to_end if te[0, i] < 0.35 and te[-1, i] > 0.65]
    te0 = [i for i in end_to_end if te[0, i] > 0.8 and te[-1, i] > 0.8]

    if not rotating:
        raise RuntimeError(
            "no branch rotates polarization. Input TE fractions "
            f"{np.round(te[0, end_to_end], 3)}, output "
            f"{np.round(te[-1, end_to_end], 3)}. A vertically symmetric cross "
            "section gives exactly zero conversion - check slab_thickness."
        )
    rotating.sort(key=lambda i: -neff[0, i])
    te0.sort(key=lambda i: -neff[0, i])
    return te0[0], rotating[0]


def sweep_length(path, lengths, launch_index, n_modes):
    """Output power per branch vs taper length, reusing the interface matrices."""
    eme = EME(path, force_unitary=True)
    eme.calc_Smatrix()
    out = np.zeros((len(lengths), 2 * n_modes))
    for k, L in enumerate(lengths):
        smatrix = eme.propagator._find_Smatrix_new_length(L)
        lumped = smatrix[0]
        for j in range(1, len(smatrix)):
            lumped = _redheffer_star_product(lumped, smatrix[j])
        launch = np.zeros(2 * n_modes, dtype=complex)
        launch[launch_index] = 1.0
        out[k] = np.abs(lumped @ launch) ** 2
    return out


# -------------------------------------------------------------------- figures


def figure_device(du, path, te0, rotating):
    """Width schedule and the rotating mode at three points along the taper."""
    z = propagation_axis(path.output_data)
    core = np.array([p[0] for p in path.output_data["EME_path"]]) * 1e9
    slab = np.array([p[1] for p in path.output_data["EME_path"]]) * 1e9
    z_ideal = np.linspace(0, LENGTH, 400)
    schedule = width_schedule()

    fig = plt.figure(figsize=(11.6, 3.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1], wspace=0.34)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(z_ideal * 1e6, schedule["w_slab"](z_ideal) * 1e9, "C1-", lw=1.2, label="ideal slab")
    ax.plot(z_ideal * 1e6, schedule["w_core"](z_ideal) * 1e9, "C0-", lw=1.2, label="ideal core")
    ax.step(z, slab, where="post", color="C1", ls="--", lw=0.9, label="grid slab")
    ax.step(z, core, where="post", color="C0", ls="--", lw=0.9, label="grid core")
    ax.set_xlabel("propagation length (um)")
    ax.set_ylabel("width (nm)")
    ax.set_title(f"bi-level taper\n{len(z)} EME sections", fontsize=9)
    ax.legend(fontsize=6.5)

    names = path._tracking_mode_names
    te = np.real(path.output_data["TE_pol"])
    picks = [0, int(np.argmin(np.abs(te[:, rotating] - 0.5))), len(z) - 1]
    labels = ["input (TM0)", "anti-crossing", "output (TE1)"]

    for column, (section, label) in enumerate(zip(picks, labels)):
        point = tuple(path.output_data["EME_path"][section])
        modes = du.solve_point(point)
        n_profile, xg, yg = du.get_index_profile(point)
        solver_index = int(np.flatnonzero(names[section] == rotating)[0])
        intensity = (
            np.abs(modes.E[solver_index, 0]) ** 2 + np.abs(modes.E[solver_index, 1]) ** 2
        )

        ax = fig.add_subplot(grid[0, column + 1])
        ax.pcolormesh(modes.x * 1e6, modes.y * 1e6, intensity.T, cmap="magma", shading="auto")
        ax.contour(xg * 1e6, yg * 1e6, np.real(n_profile).T, levels=[2.5],
                   colors="w", linewidths=0.6)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("x (um)")
        ax.set_title(
            f"{label}\n$n_{{eff}}$={np.real(modes.neff[solver_index]):.4f}, "
            f"TE {te[section, rotating]*100:.0f}%",
            fontsize=8,
        )

    save(fig, "rotator_1_device.png")


def figure_anticrossing(path, te0, rotating):
    """Branches coloured by TE fraction - the rotation in one plot."""
    z = propagation_axis(path.output_data)
    neff = np.real(path.output_data["neff"])
    te = np.real(path.output_data["TE_pol"])
    mask = path.output_data["radiation_mode_mask"]
    n_modes = neff.shape[1] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.3)

    shown = [i for i in range(n_modes) if np.count_nonzero(~mask[:, i]) > len(z) // 2]
    for i in shown:
        values = np.where(mask[:, i], np.nan, neff[:, i])
        colour = np.clip(te[:, i], 0, 1)
        segments = np.stack(
            [np.column_stack([z[:-1], values[:-1]]),
             np.column_stack([z[1:], values[1:]])], axis=1
        )
        lc = LineCollection(segments, cmap="coolwarm", norm=plt.Normalize(0, 1), lw=2.4)
        lc.set_array(0.5 * (colour[:-1] + colour[1:]))
        ax1.add_collection(lc)

    ax1.set_xlim(z[0], z[-1])
    finite = neff[:, shown][np.isfinite(neff[:, shown])]
    ax1.set_ylim(1.45, finite.max() * 1.01)
    ax1.set_xlabel("propagation length (um)")
    ax1.set_ylabel("$n_{eff}$")
    ax1.set_title("supermode branches, coloured by TE fraction", fontsize=9)
    bar = fig.colorbar(lc, ax=ax1)
    bar.set_label("TE polarization fraction", fontsize=8)

    ax2.plot(z, te[:, rotating], "C3-", lw=1.8, label="rotating branch")
    ax2.plot(z, te[:, te0], "C0-", lw=1.4, label="TE0 branch")
    ax2.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("propagation length (um)")
    ax2.set_ylabel("TE polarization fraction")
    ax2.set_title("TM0 becomes TE1; TE0 is untouched", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, "rotator_2_anticrossing.png")


def figure_length_sweep(path, te0, rotating):
    n_modes = path.output_data["neff"].shape[1] // 2
    lengths = np.linspace(5e-6, 300e-6, 200)

    t0 = time.time()
    from_tm0 = sweep_length(path, lengths, rotating, n_modes)
    from_te0 = sweep_length(path, lengths, te0, n_modes)
    elapsed = time.time() - t0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.28)

    ax1.plot(lengths * 1e6, from_tm0[:, rotating], "C3-", lw=1.5, label="TM0 -> TE1 (wanted)")
    ax1.plot(lengths * 1e6, from_te0[:, te0], "C0-", lw=1.5, label="TE0 -> TE0 (pass-through)")
    ax1.set_xlabel("taper length (um)")
    ax1.set_ylabel("output power")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title(f"{len(lengths)} lengths in {elapsed:.1f} s (no mode solves)", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.semilogy(lengths * 1e6, 1 - from_tm0[:, rotating], "C3-", lw=1.5, label="TM0 conversion loss")
    ax2.semilogy(lengths * 1e6, 1 - from_te0[:, te0], "C0-", lw=1.5, label="TE0 insertion loss")
    ax2.axhline(0.01, color="grey", lw=0.7, ls=":")
    ax2.text(lengths[-1] * 1e6, 0.011, "1 %", ha="right", fontsize=7, color="grey")
    ax2.set_xlabel("taper length (um)")
    ax2.set_ylabel("power lost from the tracked branch")
    ax2.set_title("adiabaticity roll-off", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, "rotator_3_length_sweep.png")
    return lengths, from_tm0, from_te0


def figure_validation(lengths, dataset, direct):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.3))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(lengths * 1e6, direct, "o-", ms=3, label="direct EME")
    ax1.plot(lengths * 1e6, dataset, "s--", ms=3, label="dataset EME")
    ax1.set_xlabel("taper length (um)")
    ax1.set_ylabel("TM0 -> TE1 conversion")
    ax1.set_title("same answer from both methods", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.semilogy(lengths * 1e6, np.abs(np.asarray(dataset) - np.asarray(direct)), "d-", color="C3")
    ax2.set_xlabel("taper length (um)")
    ax2.set_ylabel("|dataset - direct|")
    ax2.set_title("residual, from the width grid", fontsize=9)

    save(fig, "rotator_4_validation.png")


# ----------------------------------------------------------------------- main


def main():
    print("Dataset-based EME - bi-level taper polarization rotator")
    print("=" * 70)

    du = DataUpdater(DATASET)
    print(f"dataset opened ({len(du.neff)} points cached)")

    t0 = time.time()
    path = build(du)
    build_time = time.time() - t0
    n_modes = path.output_data["neff"].shape[1] // 2
    print(
        f"path: {len(path.output_data['EME_path'])} sections over "
        f"{LENGTH*1e6:.0f} um; dataset holds {len(du.neff)} points "
        f"[{build_time:.1f} s]"
    )

    te0, rotating = identify_branches(path)
    te = np.real(path.output_data["TE_pol"])
    neff = np.real(path.output_data["neff"])
    print(
        f"\nrotating branch = tracked #{rotating}: "
        f"n_eff {neff[0, rotating]:.4f} -> {neff[-1, rotating]:.4f}, "
        f"TE fraction {te[0, rotating]:.3f} -> {te[-1, rotating]:.3f}"
    )
    print(
        f"TE0 branch      = tracked #{te0}: "
        f"n_eff {neff[0, te0]:.4f} -> {neff[-1, te0]:.4f}, "
        f"TE fraction {te[0, te0]:.3f} -> {te[-1, te0]:.3f}"
    )

    print("\nsanity gate")
    print("-" * 70)
    checks = [
        check_reciprocity(path),
        check_reflection_symmetry(path),
        check_power_conservation(path),
    ]
    tracking_check, _ = check_branch_tracking(path)
    checks.append(tracking_check)
    convergence_check, residuals = check_mode_basis_convergence(
        lambda limit: build(du, limit_mode_number=limit, verbose=False),
        mode_counts=(3, 4, 5, 6),
    )
    checks.append(convergence_check)
    checks.append(
        check_slicing_independence(
            build(du, resolution=400, verbose=False),
            build(du, resolution=1600, verbose=False),
        )
    )
    for check in checks:
        print("  " + str(check))

    at_length = transmission(path, launch_index=rotating)
    te0_through = transmission(path, launch_index=te0)
    print(f"\nTM0 launched, {LENGTH*1e6:.0f} um taper:")
    print(f"  converted to TE1: {at_length[rotating]:.5f}")
    print(f"  TE0 pass-through: {te0_through[te0]:.5f}")

    print("\nfigures:")
    figure_device(du, path, te0, rotating)
    figure_anticrossing(path, te0, rotating)
    lengths, from_tm0, from_te0 = figure_length_sweep(path, te0, rotating)

    thresholds = {}
    for target, name in ((0.99, "99pc"), (0.999, "99.9pc")):
        ok = np.flatnonzero(from_tm0[:, rotating] > target)
        thresholds[name] = float(lengths[ok[0]] * 1e6) if ok.size else None
        if ok.size:
            print(f"TM0 -> TE1 exceeds {target*100:g}% for tapers longer than "
                  f"{lengths[ok[0]]*1e6:.0f} um")

    print("\ndataset EME vs direct EME")
    print("-" * 70)
    de = DataExtractor(DATASET, cache_size=1024)
    n_sections = len(path.output_data["EME_path"])
    t0 = time.time()
    direct_path = DirectParametricPath(
        de, width_schedule(), total_length=LENGTH, resolution=n_sections
    )
    direct_path._verbose = False
    direct_path.calc_output_data()
    direct_time = time.time() - t0
    d_te0, d_rot = identify_branches(direct_path)
    d_modes = direct_path.output_data["neff"].shape[1] // 2

    sweep_lengths = np.array([25e-6, 50e-6, 100e-6, 150e-6, 200e-6, 300e-6])
    dataset_curve = sweep_length(path, sweep_lengths, rotating, n_modes)[:, rotating]
    direct_curve = sweep_length(direct_path, sweep_lengths, d_rot, d_modes)[:, d_rot]

    print(f"{'length':>8}  {'dataset':>9}  {'direct':>9}  {'diff':>9}")
    for L, a, b in zip(sweep_lengths, dataset_curve, direct_curve):
        print(f"{L*1e6:7.0f}u  {a:9.5f}  {b:9.5f}  {abs(a-b):9.2e}")
    print(
        f"max |difference| = {np.max(np.abs(dataset_curve - direct_curve)):.2e}\n"
        f"dataset path built in {build_time:.1f} s (warm), "
        f"direct path in {direct_time:.1f} s"
    )
    figure_validation(sweep_lengths, dataset_curve, direct_curve)

    du.save_data()

    results = {
        "device": {
            "w_core_nm": [W_IN * 1e9, W_CORE_MID * 1e9, W_OUT * 1e9],
            "w_slab_nm": [W_IN * 1e9, W_SLAB_MID * 1e9, W_OUT * 1e9],
            "slab_thickness_nm": du.backend.cross_section.slab_thickness * 1e9,
            "length_um": LENGTH * 1e6,
            "sections": n_sections,
            "dataset_points": len(du.neff),
            "wavelength_nm": du.wavelength * 1e9,
        },
        "branches": {
            "te0_index": int(te0),
            "rotating_index": int(rotating),
            "rotating_neff": [float(neff[0, rotating]), float(neff[-1, rotating])],
            "rotating_te_fraction": [float(te[0, rotating]), float(te[-1, rotating])],
            "te0_neff": [float(neff[0, te0]), float(neff[-1, te0])],
        },
        "checks": [
            {"name": c.name, "criterion": c.criterion,
             "measured": None if np.isnan(c.measured) else float(c.measured),
             "passed": bool(c.passed), "note": c.note}
            for c in checks
        ],
        "mode_convergence": {str(k): float(v) for k, v in residuals.items()},
        "at_length": {
            "length_um": LENGTH * 1e6,
            "tm0_to_te1": float(at_length[rotating]),
            "te0_through": float(te0_through[te0]),
        },
        "thresholds_um": thresholds,
        "validation": {
            "lengths_um": (sweep_lengths * 1e6).tolist(),
            "dataset": dataset_curve.tolist(),
            "direct": direct_curve.tolist(),
            "max_difference": float(np.max(np.abs(dataset_curve - direct_curve))),
            "dataset_seconds": float(build_time),
            "direct_seconds": float(direct_time),
        },
    }

    report_dir = os.path.join(ROOT, "reports", "output")
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "rotator_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")

    import shutil

    from _plotting import OUTPUT_DIR

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith("rotator_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote {os.path.relpath(os.path.join(report_dir, name), ROOT)}")

    return results


if __name__ == "__main__":
    results = main()
    print("\n" + format_table(
        [type("C", (), c)() for c in []] or []
    ) if False else "")
