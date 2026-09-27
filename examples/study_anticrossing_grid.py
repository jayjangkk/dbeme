"""How finely must a dataset be sampled across an anti-crossing?

The adiabatic coupler exposed a limit of the dataset-based method that a taper
or a bend never reaches.  A DBEME dataset models a smooth device as a staircase
of uniform sections joined by abrupt parameter steps, and the step size is the
grid spacing.  That is harmless while the modes of adjacent grid points are
nearly the same modes - and it is fatal where they are not.

At an anti-crossing they are not.  Two branches exchange identity over a narrow
range of the swept parameter, so a step that would be negligible elsewhere
rotates them into each other.  The staircase then scatters power between the
branches at every step, which is precisely the process an adiabatic device is
designed to avoid, and the model destroys the behaviour it was built to
predict.

This script measures the requirement two independent ways:

1. **Directly**, as the off-diagonal overlap between the two crossing branches
   at adjacent grid points, along the device, for several step sizes.  This is
   the mechanism, and it costs a few dozen mode solves.
2. **By its consequence**, as a direct-EME sampling study: the same coupler
   solved with progressively finer width steps until the answer stops moving.
   This is slower but it is the number that matters.

Results are cached to ``output/anticrossing_grid.npz`` so the figure can be
redrawn without repeating the solves.

Run:  python examples/study_anticrossing_grid.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import OUTPUT_DIR, save  # noqa: E402
from dbeme import DataExtractor, DirectParametricPath, EME  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product  # noqa: E402

from demo_adiabatic_coupler import (  # noqa: E402
    DATASET,
    LENGTH,
    W1_IN,
    W1_OUT,
    W2_IN,
    W2_OUT,
    identify_branches,
    width_schedule,
)

CACHE = os.path.join(OUTPUT_DIR, "anticrossing_grid.npz")

#: Width steps to test, in nanometres.
STEPS_NM = (2.5, 5.0, 10.0, 20.0)

#: Section counts for the direct-EME sampling study.
RESOLUTIONS = (27, 60, 120)

#: Lengths at which the sampling study is evaluated.
STUDY_LENGTHS = (100e-6, 200e-6, 400e-6)


def _on_path(fraction):
    """The (w1, w2) point a fraction of the way along the coupler."""
    return (
        W1_IN + (W1_OUT - W1_IN) * fraction,
        W2_IN + (W2_OUT - W2_IN) * fraction,
    )


def measure_supermode_rotation(backend, positions=24):
    """Off-diagonal overlap between the crossing branches, per grid step.

    Walks along the device and, at each position, compares the modes there with
    the modes one step further on.  The largest off-diagonal element among the
    guided TE branches is how much a single step mixes them: 0 means the step
    is invisible to the modes, 1 means it swaps them completely.

    :returns: ``(fractions, w2 values, {step_nm: rotation array})``
    """
    fractions = np.linspace(0.02, 0.95, positions)
    rotations = {step: np.zeros(positions) for step in STEPS_NM}
    w2_values = np.zeros(positions)

    for index, fraction in enumerate(fractions):
        w1, w2 = _on_path(fraction)
        w2_values[index] = w2
        here = backend.solve((round(w1, 12), round(w2, 12)))

        for step in STEPS_NM:
            # Step both widths as the device does, so this is the rotation the
            # real path would see, not an artificial one-parameter move.
            scale = step * 1e-9 / (W2_OUT - W2_IN)
            w1_next = w1 + (W1_OUT - W1_IN) * scale
            w2_next = w2 + step * 1e-9
            there = backend.solve((round(w1_next, 12), round(w2_next, 12)))

            x, y, neff, te, E, H = assemble(
                [here, there], backend.num_modes, prop_axis=2
            )
            O = np.abs(overlap_matrix(E[0], H[1], x, y, prop_axis=2))
            n = backend.num_modes
            guided = [
                i
                for i in range(n)
                if te[0, i] > 0.5 and np.real(neff[0, i]) > 1.6
            ]
            worst = 0.0
            for i in guided:
                for j in guided:
                    if i != j:
                        worst = max(worst, O[i, j])
            rotations[step][index] = worst

    return fractions, w2_values, rotations


def measure_sampling_convergence(extractor):
    """Direct-EME efficiency vs how finely the widths are sampled.

    No dataset and no grid: each section is solved at its exact widths, so the
    only thing changing is how many sections the smooth taper is cut into.

    :returns: ``({resolution: step_nm}, {resolution: [efficiency per length]})``
    """
    steps, curves = {}, {}
    for resolution in RESOLUTIONS:
        t0 = time.time()
        path = DirectParametricPath(
            extractor, width_schedule(), total_length=LENGTH, resolution=resolution
        )
        path._verbose = False
        path.calc_output_data()

        n_modes = path.output_data["neff"].shape[1] // 2
        cross_section = extractor.backend.cross_section
        names = path._tracking_mode_names
        fractions = np.full((path.output_data["neff"].shape[0], n_modes), np.nan)
        for section, point in enumerate(path.output_data["EME_path"]):
            modes = extractor._solve(tuple(point))
            per_mode = cross_section.power_fractions(modes)
            for j in range(names.shape[1]):
                fractions[section, names[section, j]] = per_mode[j]
        _, cross = identify_branches(path, fractions)

        eme = EME(path, force_unitary=True)
        eme.calc_Smatrix()
        values = []
        for length in STUDY_LENGTHS:
            smatrix = eme.propagator._find_Smatrix_new_length(length)
            lumped = smatrix[0]
            for j in range(1, len(smatrix)):
                lumped = _redheffer_star_product(lumped, smatrix[j])
            launch = np.zeros(2 * n_modes, dtype=complex)
            launch[cross] = 1.0
            values.append(float((np.abs(lumped @ launch) ** 2)[cross]))

        steps[resolution] = (W2_OUT - W2_IN) * 1e9 / (resolution - 1)
        curves[resolution] = values
        print(
            f"  {resolution:4d} sections, dw2 = {steps[resolution]:5.2f} nm -> "
            + "  ".join(f"{v:.4f}" for v in values)
            + f"   [{time.time()-t0:.0f} s]",
            flush=True,
        )
    return steps, curves


def figure(fractions, w2_values, rotations, steps, curves):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.5))
    fig.subplots_adjust(wspace=0.3)

    for k, step in enumerate(STEPS_NM):
        ax1.plot(
            w2_values * 1e9, rotations[step], "o-", ms=3, color=f"C{k}",
            label=f"{step:g} nm step",
        )
    ax1.axhline(0.1, color="grey", lw=0.8, ls=":")
    ax1.text(w2_values[0] * 1e9, 0.108, "10 %", fontsize=7, color="grey")
    ax1.set_xlabel("narrow-guide width $w_2$ (nm)")
    ax1.set_ylabel("branch mixing per step  |$O_{ij}$|")
    ax1.set_yscale("log")
    ax1.set_title(
        "one grid step rotates the supermodes\n(peak marks the anti-crossing)",
        fontsize=9,
    )
    ax1.legend(fontsize=7)

    order = sorted(steps, key=lambda r: steps[r])
    step_values = [steps[r] for r in order]
    for k, length in enumerate(STUDY_LENGTHS):
        ax2.plot(
            step_values,
            [curves[r][k] for r in order],
            "s-", ms=4, color=f"C{k}",
            label=f"L = {length*1e6:.0f} um",
        )
    ax2.set_xlabel("width step used to slice the taper (nm)")
    ax2.set_ylabel("TE1 -> narrow guide")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("direct EME: the answer stops moving below ~5 nm", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, "coupler_5_grid_rule.png")


def main():
    print("Anti-crossing grid requirement")
    print("=" * 68)

    extractor = DataExtractor(DATASET, cache_size=2048)

    print("\n1. supermode rotation per grid step")
    t0 = time.time()
    fractions, w2_values, rotations = measure_supermode_rotation(extractor.backend)
    print(f"   measured in {time.time()-t0:.0f} s")
    for step in STEPS_NM:
        print(f"   {step:5g} nm step: worst branch mixing = {rotations[step].max():.3f}")

    print("\n2. direct-EME sampling convergence "
          f"(TE1 -> narrow at L = {'/'.join(f'{v*1e6:.0f}' for v in STUDY_LENGTHS)} um)")
    steps, curves = measure_sampling_convergence(extractor)

    np.savez(
        CACHE,
        fractions=fractions,
        w2_values=w2_values,
        steps_nm=np.array(STEPS_NM),
        rotations=np.array([rotations[s] for s in STEPS_NM]),
        resolutions=np.array(RESOLUTIONS),
        step_sizes=np.array([steps[r] for r in RESOLUTIONS]),
        curves=np.array([curves[r] for r in RESOLUTIONS]),
        study_lengths=np.array(STUDY_LENGTHS),
    )
    print(f"\n   cached to {os.path.relpath(CACHE)}")

    figure(fractions, w2_values, rotations, steps, curves)

    peak = int(np.argmax(rotations[20.0]))
    print(
        f"\nThe anti-crossing sits at w2 = {w2_values[peak]*1e9:.0f} nm. "
        f"There a 20 nm step mixes the branches by {rotations[20.0][peak]:.2f} "
        f"and a 5 nm step by {rotations[5.0][peak]:.2f}."
    )


if __name__ == "__main__":
    main()
