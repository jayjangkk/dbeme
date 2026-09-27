"""Demo 4 - adiabatic (mode-evolution) coupler.

Reproduces the adiabatic coupler stage of the polarization rotator-splitter in

    W. D. Sacher, T. Barwicz, B. J. F. Taylor and J. K. S. Poon,
    "Polarization rotator-splitters in standard active silicon photonics
    platforms", Opt. Express 22(4), 3777 (2014).  doi:10.1364/OE.22.003777

In that device a bi-level taper turns TM0 into TE1, and this stage then
separates TE0 from TE1 into two physical waveguides.  The paper specifies it
as fully etched 220 nm Si with symmetric SiO2 cladding, a broad guide going
850 -> 650 nm beside a narrow one going 200 -> 500 nm, at a constant 200 nm
gap - which is exactly the ``Si_pair_fulletch_220nm`` dataset here.

How it works
------------
At the input the two guides are strongly detuned: the broad guide carries TE0
and TE1, the 200 nm guide carries nothing.  As the widths sweep past each
other the broad guide's TE1 and the narrow guide's TE0 pass through an
anti-crossing.  Follow that branch slowly enough and the light changes which
waveguide it is in without ever changing supermode - so TE1 ends up in the
narrow guide while TE0, which anti-crosses with nothing, stays put.

The whole thing is one path through a two-parameter dataset, so the length
sweep and the shape studies cost no mode solving once it is built.

Run:  python examples/demo_adiabatic_coupler.py
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
from dbeme import (  # noqa: E402
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
)
from dbeme.matrix_calculation_tool import _redheffer_star_product  # noqa: E402
from dbeme.validation import (  # noqa: E402
    check_branch_tracking,
    check_mode_basis_convergence,
    check_power_conservation,
    check_reciprocity,
    check_reflection_symmetry,
    check_slicing_independence,
    format_table,
    lumped_smatrix,
    transmission,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "datasets", "Si_pair_fulletch_220nm")

# Sacher et al. Fig. 1: broad 850 -> 650 nm, narrow 200 -> 500 nm, 200 nm gap.
W1_IN, W1_OUT = 850e-9, 650e-9
W2_IN, W2_OUT = 200e-9, 500e-9
LENGTH = 100e-6  # the paper gives 475 um for the whole PRS, not this stage


def width_schedule(length=LENGTH):
    """Linear width taper, as parameter functions of propagation length."""
    return {
        "w1": lambda z, L=length: W1_IN + (W1_OUT - W1_IN) * (z / L),
        "w2": lambda z, L=length: W2_IN + (W2_OUT - W2_IN) * (z / L),
    }


def build(du, length=LENGTH, **kwargs):
    path = ParametricPath(du, width_schedule(length), total_length=length, **kwargs)
    path.calc_output_data()
    return path


# ------------------------------------------------------------------- analysis


def branch_localisation(du, path):
    """Broad-guide power fraction of every tracked branch, along the device.

    ``_tracking_mode_names[section, solver_mode]`` is the tracked label the
    mode-ordering step assigned, so inverting it says which solved mode each
    branch corresponds to at each section.

    :returns: ``(n_sections, n_branches)`` array, NaN where a branch is absent.
    """
    cross_section = du.backend.cross_section
    names = path._tracking_mode_names
    n_sections, n_solver = names.shape
    n_branches = int(names.max()) + 1

    fractions = np.full((n_sections, n_branches), np.nan)
    for section, point in enumerate(path.output_data["EME_path"]):
        modes = du.solve_point(tuple(point))
        per_mode = cross_section.power_fractions(modes)
        for solver_index in range(n_solver):
            fractions[section, names[section, solver_index]] = per_mode[solver_index]
    return fractions


def identify_branches(path, fractions):
    """Find the branch that stays broad and the branch that crosses over.

    Identified from the physics rather than by index: of the branches that
    start in the broad guide, the one that *ends* there is TE0, and the one
    that ends in the narrow guide is the crossing TE1.
    """
    neff = np.real(path.output_data["neff"])
    te = path.output_data["TE_pol"]
    n_modes = neff.shape[1] // 2

    candidates = [
        i
        for i in range(n_modes)
        if te[0, i] > 0.5
        and neff[0, i] > 1.5
        and np.isfinite(fractions[0, i])
        and fractions[0, i] > 0.7
    ]
    stay = [i for i in candidates if fractions[-1, i] > 0.5]
    cross = [i for i in candidates if fractions[-1, i] <= 0.5]

    if not stay or not cross:
        raise RuntimeError(
            f"could not identify the coupler branches: candidates={candidates}, "
            f"input fractions={fractions[0, candidates]}, "
            f"output fractions={fractions[-1, candidates]}"
        )
    # Highest neff among each group.
    stay.sort(key=lambda i: -neff[0, i])
    cross.sort(key=lambda i: -neff[0, i])
    return stay[0], cross[0]


def sweep_length(path, lengths, launch_index, n_modes):
    """Output power per branch vs coupler length.

    Stretching the coupler changes only the propagation phases between
    sections - the sequence of cross sections is set by the *normalised*
    width schedule - so the whole sweep reuses the interface matrices.
    """
    from dbeme import EME

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


def figure_device(du, path, stay, cross):
    """Width schedule, and the mode intensities at each end."""
    z = propagation_axis(path.output_data)
    w1 = np.array([p[0] for p in path.output_data["EME_path"]]) * 1e9
    w2 = np.array([p[1] for p in path.output_data["EME_path"]]) * 1e9
    z_ideal = np.linspace(0, LENGTH, 300)
    schedule = width_schedule()

    fig = plt.figure(figsize=(11.0, 5.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.32)

    ax = fig.add_subplot(grid[:, 0])
    ax.plot(z_ideal * 1e6, schedule["w1"](z_ideal) * 1e9, "C0-", lw=1.2, label="ideal $w_1$")
    ax.plot(z_ideal * 1e6, schedule["w2"](z_ideal) * 1e9, "C1-", lw=1.2, label="ideal $w_2$")
    ax.step(z, w1, where="post", color="C0", ls="--", lw=1.0, label="grid $w_1$ (broad)")
    ax.step(z, w2, where="post", color="C1", ls="--", lw=1.0, label="grid $w_2$ (narrow)")
    ax.set_xlabel("propagation length (um)")
    ax.set_ylabel("width (nm)")
    ax.set_title(
        f"Sacher adiabatic coupler stage\n{len(z)} EME sections, 200 nm gap",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    ends = [
        ("input", path.output_data["EME_path"][0]),
        ("output", path.output_data["EME_path"][-1]),
    ]
    names = path._tracking_mode_names
    cs = du.backend.cross_section

    for column, (label, point) in enumerate(ends):
        section = 0 if label == "input" else len(z) - 1
        modes = du.solve_point(tuple(point))
        fractions = cs.power_fractions(modes)
        n_profile, xg, yg = du.get_index_profile(tuple(point))

        for row, branch in enumerate((stay, cross)):
            ax = fig.add_subplot(grid[row, column + 1])
            solver_index = int(np.flatnonzero(names[section] == branch)[0])
            intensity = (
                np.abs(modes.E[solver_index, 0]) ** 2
                + np.abs(modes.E[solver_index, 1]) ** 2
            )
            ax.pcolormesh(
                modes.x * 1e6, modes.y * 1e6, intensity.T, cmap="magma", shading="auto"
            )
            ax.contour(
                xg * 1e6, yg * 1e6, np.real(n_profile).T,
                levels=[2.5], colors="w", linewidths=0.6,
            )
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlim(-1.4, 1.0)
            branch_name = "TE0 branch" if branch == stay else "TE1 branch"
            ax.set_title(
                f"{label}: {branch_name}\n"
                f"$n_{{eff}}$={np.real(modes.neff[solver_index]):.4f}, "
                f"broad {fractions[solver_index]*100:.0f}%",
                fontsize=7.5,
            )
            if row == 1:
                ax.set_xlabel("x (um)")

    save(fig, "coupler_1_device.png")


def figure_anticrossing(path, fractions, stay, cross):
    """The mechanism: branches coloured by which guide they are in."""
    z = propagation_axis(path.output_data)
    neff = np.real(path.output_data["neff"])
    te = path.output_data["TE_pol"]
    mask = path.output_data["radiation_mode_mask"]
    n_modes = neff.shape[1] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.3)

    shown = [
        i for i in range(n_modes)
        if te[0, i] > 0.5 and np.count_nonzero(~mask[:, i]) > len(z) // 2
    ]
    for i in shown:
        values = np.where(mask[:, i], np.nan, neff[:, i])
        colour = np.clip(fractions[:, i], 0, 1)
        segments = np.stack(
            [np.column_stack([z[:-1], values[:-1]]),
             np.column_stack([z[1:], values[1:]])], axis=1
        )
        lc = LineCollection(segments, cmap="coolwarm_r", norm=plt.Normalize(0, 1), lw=2.4)
        lc.set_array(0.5 * (colour[:-1] + colour[1:]))
        ax1.add_collection(lc)
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            tag = "TE0" if i == stay else ("TE1" if i == cross else "")
            if tag:
                ax1.annotate(
                    tag, (z[finite[0]], values[finite[0]]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                )

    ax1.set_xlim(z[0], z[-1])
    finite_all = neff[:, shown][np.isfinite(neff[:, shown])]
    ax1.set_ylim(1.5, finite_all.max() * 1.01)
    ax1.set_xlabel("propagation length (um)")
    ax1.set_ylabel("$n_{eff}$")
    ax1.set_title("supermode branches, coloured by guide", fontsize=9)
    bar = fig.colorbar(lc, ax=ax1)
    bar.set_label("power in broad guide", fontsize=8)

    ax2.plot(z, fractions[:, stay], "C0-", lw=1.6, label="TE0 branch")
    ax2.plot(z, fractions[:, cross], "C3-", lw=1.6, label="TE1 branch")
    ax2.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("propagation length (um)")
    ax2.set_ylabel("power in broad guide")
    ax2.set_title("TE1 changes waveguide; TE0 does not", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, "coupler_2_anticrossing.png")


def figure_length_sweep(path, stay, cross):
    """Adiabaticity: efficiency vs coupler length."""
    n_modes = path.output_data["neff"].shape[1] // 2
    lengths = np.linspace(5e-6, 400e-6, 240)

    t0 = time.time()
    from_te0 = sweep_length(path, lengths, stay, n_modes)
    from_te1 = sweep_length(path, lengths, cross, n_modes)
    elapsed = time.time() - t0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.28)

    ax1.plot(lengths * 1e6, from_te0[:, stay], "C0-", lw=1.4, label="TE0 -> broad (wanted)")
    ax1.plot(lengths * 1e6, from_te0[:, cross], "C0--", lw=1.0, label="TE0 -> narrow (leak)")
    ax1.plot(lengths * 1e6, from_te1[:, cross], "C3-", lw=1.4, label="TE1 -> narrow (wanted)")
    ax1.plot(lengths * 1e6, from_te1[:, stay], "C3--", lw=1.0, label="TE1 -> broad (leak)")
    ax1.set_xlabel("coupler length (um)")
    ax1.set_ylabel("output power")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title(f"{len(lengths)} lengths in {elapsed:.1f} s (no mode solves)", fontsize=9)
    ax1.legend(fontsize=7)

    ax2.semilogy(lengths * 1e6, 1 - from_te0[:, stay], "C0-", lw=1.4, label="TE0 loss from branch")
    ax2.semilogy(lengths * 1e6, 1 - from_te1[:, cross], "C3-", lw=1.4, label="TE1 loss from branch")
    ax2.axhline(0.01, color="grey", lw=0.7, ls=":")
    ax2.text(lengths[-1] * 1e6, 0.011, "1 %", ha="right", fontsize=7, color="grey")
    ax2.set_xlabel("coupler length (um)")
    ax2.set_ylabel("power lost from the tracked branch")
    ax2.set_title("adiabaticity roll-off", fontsize=9)
    ax2.legend(fontsize=7)

    save(fig, "coupler_3_length_sweep.png")
    return lengths, from_te0, from_te1


def figure_validation(lengths, dbeme, direct, cross, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.3))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(lengths * 1e6, direct, "o-", ms=3, label="direct EME")
    ax1.plot(lengths * 1e6, dbeme, "s--", ms=3, label="dataset EME")
    ax1.set_xlabel("coupler length (um)")
    ax1.set_ylabel(label)
    ax1.set_title("same answer from both methods", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.semilogy(lengths * 1e6, np.abs(np.asarray(dbeme) - np.asarray(direct)), "d-", color="C3")
    ax2.set_xlabel("coupler length (um)")
    ax2.set_ylabel("|dataset - direct|")
    ax2.set_title("difference, from the 20 nm width grid", fontsize=9)

    save(fig, "coupler_4_validation.png")


# ----------------------------------------------------------------------- main


def main():
    print("Dataset-based EME - adiabatic coupler (Sacher et al. 2014)")
    print("=" * 68)

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

    fractions = branch_localisation(du, path)
    stay, cross = identify_branches(path, fractions)
    neff = np.real(path.output_data["neff"])
    print(
        f"\nbranches: TE0 = tracked #{stay} "
        f"(n_eff {neff[0, stay]:.4f} -> {neff[-1, stay]:.4f}), "
        f"TE1 = tracked #{cross} "
        f"(n_eff {neff[0, cross]:.4f} -> {neff[-1, cross]:.4f})"
    )
    print(
        f"broad-guide power fraction:  TE0 {fractions[0, stay]:.3f} -> "
        f"{fractions[-1, stay]:.3f}   TE1 {fractions[0, cross]:.3f} -> "
        f"{fractions[-1, cross]:.3f}"
    )

    # ------------------------------------------------------------ sanity gate
    print("\nsanity gate")
    print("-" * 68)
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

    # ------------------------------------------------------------- main result
    at_length = transmission(path, launch_index=cross)
    print(f"\nTE1 launched, {LENGTH*1e6:.0f} um coupler:")
    print(f"  into the narrow guide (TE1 branch): {at_length[cross]:.5f}")
    print(f"  left in the broad guide (TE0 branch): {at_length[stay]:.5f}")

    print("\nfigures:")
    figure_device(du, path, stay, cross)
    figure_anticrossing(path, fractions, stay, cross)
    lengths, from_te0, from_te1 = figure_length_sweep(path, stay, cross)

    for target, name in ((0.99, "99 %"), (0.999, "99.9 %")):
        ok = np.flatnonzero(from_te1[:, cross] > target)
        if ok.size:
            print(
                f"TE1 crossover exceeds {name} for couplers longer than "
                f"{lengths[ok[0]]*1e6:.0f} um"
            )

    # --------------------------------------------------- DBEME vs direct EME
    print("\ndataset EME vs direct EME")
    print("-" * 68)
    de = DataExtractor(DATASET)
    n_sections = len(path.output_data["EME_path"])
    t0 = time.time()
    direct_path = DirectParametricPath(
        de, width_schedule(), total_length=LENGTH, resolution=n_sections
    )
    direct_path._verbose = False
    direct_path.calc_output_data()
    direct_time = time.time() - t0

    direct_fractions = None
    d_neff = np.real(direct_path.output_data["neff"])
    d_modes = d_neff.shape[1] // 2
    # Identify the same two branches on the direct path, by the same rule.
    cs = de.backend.cross_section
    d_names = direct_path._tracking_mode_names
    d_frac = np.full((d_neff.shape[0], d_modes), np.nan)
    for section, point in enumerate(direct_path.output_data["EME_path"]):
        modes = de._solve(tuple(point))
        per_mode = cs.power_fractions(modes)
        for j in range(d_names.shape[1]):
            d_frac[section, d_names[section, j]] = per_mode[j]
    d_stay, d_cross = identify_branches(direct_path, d_frac)
    direct_fractions = d_frac

    sweep_lengths = np.array([25e-6, 50e-6, 100e-6, 200e-6, 300e-6, 400e-6])
    dbeme_curve = sweep_length(path, sweep_lengths, cross, n_modes)[:, cross]
    direct_curve = sweep_length(direct_path, sweep_lengths, d_cross, d_modes)[:, d_cross]

    print(f"{'length':>8}  {'dataset':>9}  {'direct':>9}  {'diff':>9}")
    for L, a, b in zip(sweep_lengths, dbeme_curve, direct_curve):
        print(f"{L*1e6:7.0f}u  {a:9.5f}  {b:9.5f}  {abs(a-b):9.2e}")
    print(
        f"max |difference| = {np.max(np.abs(dbeme_curve - direct_curve)):.2e}\n"
        f"dataset path built in {build_time:.1f} s (warm), "
        f"direct path in {direct_time:.1f} s"
    )
    figure_validation(
        sweep_lengths, dbeme_curve, direct_curve, cross,
        "TE1 -> narrow guide",
    )

    du.save_data()

    results = {
        "device": {
            "w1_nm": [W1_IN * 1e9, W1_OUT * 1e9],
            "w2_nm": [W2_IN * 1e9, W2_OUT * 1e9],
            "gap_nm": du.backend.cross_section.gap * 1e9,
            "length_um": LENGTH * 1e6,
            "sections": len(path.output_data["EME_path"]),
            "dataset_points": len(du.neff),
        },
        "branches": {
            "te0_index": int(stay),
            "te1_index": int(cross),
            "te0_neff": [float(neff[0, stay]), float(neff[-1, stay])],
            "te1_neff": [float(neff[0, cross]), float(neff[-1, cross])],
            "te0_broad_fraction": [float(fractions[0, stay]), float(fractions[-1, stay])],
            "te1_broad_fraction": [float(fractions[0, cross]), float(fractions[-1, cross])],
        },
        "checks": [
            {
                "name": c.name,
                "criterion": c.criterion,
                "measured": None if np.isnan(c.measured) else float(c.measured),
                "passed": bool(c.passed),
                "note": c.note,
            }
            for c in checks
        ],
        "mode_convergence": {str(k): float(v) for k, v in residuals.items()},
        "at_length": {
            "length_um": LENGTH * 1e6,
            "te1_to_narrow": float(at_length[cross]),
            "te1_left_broad": float(at_length[stay]),
        },
        "adiabatic_thresholds_um": {
            name: (float(lengths[np.flatnonzero(from_te1[:, cross] > t)[0]] * 1e6)
                   if np.any(from_te1[:, cross] > t) else None)
            for t, name in ((0.99, "99pc"), (0.999, "99.9pc"))
        },
        "validation": {
            "lengths_um": (sweep_lengths * 1e6).tolist(),
            "dataset": dbeme_curve.tolist(),
            "direct": direct_curve.tolist(),
            "max_difference": float(np.max(np.abs(dbeme_curve - direct_curve))),
            "dataset_seconds": float(build_time),
            "direct_seconds": float(direct_time),
        },
    }
    report_dir = os.path.join(ROOT, "reports", "output")
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "coupler_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")

    # The report lives in reports/ and CLAUDE.md wants its figures beside it,
    # so copy rather than cross-link - a report should be self-contained.
    import shutil

    from _plotting import OUTPUT_DIR

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith("coupler_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote {os.path.relpath(os.path.join(report_dir, name), ROOT)}")

    # Hand the report the numbers it needs.
    return {
        "checks": checks,
        "residuals": residuals,
        "fractions": fractions,
        "direct_fractions": direct_fractions,
        "stay": stay,
        "cross": cross,
        "lengths": lengths,
        "from_te0": from_te0,
        "from_te1": from_te1,
        "sweep_lengths": sweep_lengths,
        "dbeme_curve": dbeme_curve,
        "direct_curve": direct_curve,
        "build_time": build_time,
        "direct_time": direct_time,
    }


if __name__ == "__main__":
    results = main()
    print("\n" + format_table(results["checks"]))
