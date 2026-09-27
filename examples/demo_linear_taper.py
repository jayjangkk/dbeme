"""Demo 1 - linear taper in a fully etched Si waveguide.

Widens a 220 nm-thick Si strip from 0.5 um (single-mode) to 1.2 um
(multi-mode) and asks how much power stays in TE0.

What it shows
-------------
1. The cross section the emepy backend builds, and its guided modes.
2. How the taper is snapped onto the dataset's 20 nm width grid.
3. Effective indices and modal power along the taper.
4. Transmission versus taper length - the adiabaticity threshold.

Step 4 is where the dataset pays for itself: every length in the sweep reuses
the same set of cross sections, so the whole sweep costs no mode solves at all.

A caveat visible in that last figure: TE0 -> TE2 crosstalk falls steeply up to
about 15 um and then flattens into a fringed plateau near 1e-4 instead of
continuing to fall.  The likely cause is the 20 nm width grid rather than the
taper: the dataset turns a smooth taper into a staircase, each step scatters a
fixed amount, and stretching the device only changes the phases between those
fixed scattering events.  That has not been proven here - if it is right the
floor should fall as (delta_w)^2 when ``parameters["top_width"]`` is refined.
Until that is measured, read the plateau as an upper bound on the conversion,
not as a physical number.

Run:  python examples/demo_linear_taper.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import (  # noqa: E402
    guided_mode_indices,
    mode_labels,
    propagation_axis,
    save,
    section_amplitudes,
)
from dbeme import EME, DataUpdater, LinearTaper, Runner  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product  # noqa: E402

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "Si_fulletch_220nm",
)

INPUT_WIDTH = 0.5e-6
OUTPUT_WIDTH = 1.2e-6
LENGTH = 10e-6


def figure_cross_section(du):
    """The geometry emepy is handed, and the modes it finds."""
    n, x, y = du.get_index_profile((OUTPUT_WIDTH, 0.0))
    E, H, xf, yf = du.find_mode_fields((OUTPUT_WIDTH, 0.0))
    md = du.solve_point((OUTPUT_WIDTH, 0.0))

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))

    im = axes[0].pcolormesh(x * 1e6, y * 1e6, np.real(n).T, cmap="bone_r", shading="auto")
    axes[0].set_title(f"index profile\n{OUTPUT_WIDTH*1e6:.1f} um x 220 nm Si")
    fig.colorbar(im, ax=axes[0], label="n")

    for k, ax in enumerate(axes[1:]):
        intensity = np.abs(E[k, :, :, 0]) ** 2 + np.abs(E[k, :, :, 1]) ** 2
        ax.pcolormesh(xf * 1e6, yf * 1e6, intensity.T, cmap="magma", shading="auto")
        ax.contour(
            x * 1e6, y * 1e6, np.real(n).T, levels=[2.5], colors="w", linewidths=0.6
        )
        pol = "TE" if md.TE_pol[k] > 0.5 else "TM"
        ax.set_title(
            f"mode {k}  ({pol})\n$n_{{eff}}$ = {np.real(md.neff[k]):.4f}", fontsize=8
        )

    for ax in axes:
        ax.set_xlabel("x (um)")
        ax.set_aspect("equal")
        ax.grid(False)
    axes[0].set_ylabel("y (um)")
    save(fig, "taper_1_cross_section.png")


def figure_discretisation(taper, output_data):
    """Continuous taper vs the dataset grid points it is snapped onto."""
    z = propagation_axis(output_data)
    widths = np.array([p[0] for p in output_data["EME_path"]]) * 1e6

    z_ideal = np.linspace(0, LENGTH, 400)
    w_ideal = np.linspace(INPUT_WIDTH, OUTPUT_WIDTH, 400) * 1e6

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(z_ideal * 1e6, w_ideal, "-", lw=1.2, label="ideal taper")
    ax.step(z, widths, where="post", color="C1", lw=1.0, label="dataset grid (20 nm)")
    ax.plot(z, widths, ".", color="C1", ms=4)
    ax.set_xlabel("propagation length (um)")
    ax.set_ylabel("top width (um)")
    ax.set_title(f"{len(widths)} EME sections over {LENGTH*1e6:.0f} um")
    ax.legend()
    save(fig, "taper_2_discretisation.png")


def figure_propagation(output_data, amplitudes, labels, guided):
    """Effective indices and modal power along the taper."""
    z = propagation_axis(output_data)
    neff = np.real(output_data["neff"])
    power = np.abs(amplitudes) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.1))

    for i in guided:
        ax1.plot(z, np.where(neff[:, i] > 1e-6, neff[:, i], np.nan), label=labels[i])
    ax1.axhline(1.444, ls="--", lw=0.8, color="grey")
    ax1.text(z[-1], 1.452, "SiO$_2$ cladding", ha="right", fontsize=7, color="grey")
    ax1.set_xlabel("propagation length (um)")
    ax1.set_ylabel("$n_{eff}$")
    ax1.set_title("guided modes along the taper")
    ax1.legend(ncol=2, fontsize=7)

    for i in guided:
        ax2.plot(z, power[:, i], label=labels[i])
    ax2.set_xlabel("propagation length (um)")
    ax2.set_ylabel("$|a|^2$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-7, 2)
    ax2.set_title(f"power with TE0 launched ({LENGTH*1e6:.0f} um taper)")
    ax2.legend(ncol=2, fontsize=7)

    save(fig, "taper_3_propagation.png")


def sweep_length(eme, lengths, n_modes):
    """Output power per mode versus taper length.

    Stretching a linear taper changes only the propagation phases between
    sections, not the cross sections themselves - so the whole sweep reuses the
    interface matrices already computed and needs no further mode solving.
    """
    out = np.zeros((len(lengths), 2 * n_modes))
    for k, L in enumerate(lengths):
        smatrix = eme.propagator._find_Smatrix_new_length(L)
        lumped = smatrix[0]
        for j in range(1, len(smatrix)):
            lumped = _redheffer_star_product(lumped, smatrix[j])
        launch = np.zeros(2 * n_modes, dtype=complex)
        launch[0] = 1.0
        out[k] = np.abs(lumped @ launch) ** 2
    return out


def figure_length_sweep(eme, output_data, labels, guided):
    n_modes = output_data["neff"].shape[1] // 2
    # The TE0-TE2 beat period is lambda / delta_neff ~ 2.5 um, so sample the
    # sweep finely enough not to alias the interference fringes.
    lengths = np.linspace(0.2e-6, 40e-6, 500)

    t0 = time.time()
    out = sweep_length(eme, lengths, n_modes)
    elapsed = time.time() - t0

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i in guided:
        if np.max(out[:, i]) < 1e-4:
            continue
        ax.plot(lengths * 1e6, out[:, i], label=labels[i])
    ax.set_xlabel("taper length (um)")
    ax.set_ylabel("output power")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_title(
        f"{INPUT_WIDTH*1e6:.1f} -> {OUTPUT_WIDTH*1e6:.1f} um taper, TE0 launched\n"
        f"{len(lengths)} lengths in {elapsed:.2f} s (no mode solves)"
    )
    ax.legend(ncol=2, fontsize=7)
    save(fig, "taper_4_length_sweep.png")

    return lengths, out


def main():
    print("Dataset-based EME - linear taper demo")
    print("-" * 60)

    t0 = time.time()
    du = DataUpdater(DATASET)
    print(f"dataset opened ({len(du.neff)} points already cached)")

    taper = LinearTaper(
        du, input_width=INPUT_WIDTH, output_width=OUTPUT_WIDTH, length=LENGTH
    )
    output_data = taper.calc_output_data()
    print(
        f"path: {len(output_data['EME_path'])} sections, "
        f"{np.sum(output_data['EME_delta_zs'])*1e6:.2f} um; "
        f"dataset now holds {len(du.neff)} points  [{time.time()-t0:.1f} s]"
    )

    guided = guided_mode_indices(output_data, section=-1)
    labels = mode_labels(output_data, guided, section=-1)
    print("tracked guided modes at the output:", [labels[i] for i in guided])

    t0 = time.time()
    # A truncated mode basis makes each interface matrix slightly non-unitary
    # (~1e-4 of the power per interface, which accumulates over ~40 sections).
    # This solver carries no PML, so it models no radiation loss at all and the
    # structure is lossless by construction - projecting each section's
    # S-matrix onto the nearest unitary matrix removes the numerical drift
    # without discarding physics.  Compare with EME(taper) to see the raw
    # truncation error.
    eme = EME(taper, force_unitary=True)
    eme.calc_Smatrix()
    print(f"EME transfer/scattering matrices in {time.time()-t0:.1f} s")

    runner = Runner(eme)
    n_launch = int(np.count_nonzero(output_data["radiation_mode_mask"][0] == False))
    launch = np.zeros(n_launch, dtype=complex)
    launch[0] = 1.0
    amplitudes = section_amplitudes(runner, runner.propagate(launch))

    n_modes = output_data["neff"].shape[1] // 2
    out = np.abs(amplitudes[-1]) ** 2
    print("\noutput power with TE0 launched:")
    for i in guided:
        print(f"   {labels[i]:>4}  {out[i]:.6f}   ({10*np.log10(max(out[i],1e-12)):+7.2f} dB)")
    print(f"   {'sum':>4}  {out[:n_modes].sum():.6f} forward, "
          f"{out[n_modes:].sum():.2e} reflected")

    print("\nfigures:")
    figure_cross_section(du)
    figure_discretisation(taper, output_data)
    figure_propagation(output_data, amplitudes, labels, guided)
    lengths, sweep = figure_length_sweep(eme, output_data, labels, guided)

    te0 = sweep[:, 0]
    print(f"\nshortest taper in the sweep ({lengths[0]*1e6:.2f} um): TE0 = {te0[0]:.4f}")
    ok = np.where(te0 > 0.999)[0]
    if len(ok):
        print(
            f"TE0 transmission stays above 99.9% from "
            f"{lengths[ok[0]]*1e6:.2f} um onwards"
        )

    du.save_data()


if __name__ == "__main__":
    main()
