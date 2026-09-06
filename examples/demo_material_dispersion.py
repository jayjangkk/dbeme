"""Demo 3 - material dispersion, and what a constant index costs.

The cross sections used to carry bare numbers: ``core_index = 3.4757``,
``clad_index = 1.444``.  Those are right at 1550 nm and wrong everywhere else.
This demo replaces them with dispersion models pulled from
`refractiveindex.info <https://refractiveindex.info>`_ through
`PyOptik <https://github.com/MartinPdeS/PyOptik>`_ and measures the difference.

What it shows
-------------
1. ``n(lambda)`` for Si, SiO2 and Si3N4 across the O to L bands, and how far
   the built-in Sellmeier fallbacks sit from the database entries.
2. The effective index of a 500 x 220 nm Si strip computed properly, against
   the same guide with the index frozen at its 1550 nm value.
3. The group index, which is where a constant index fails hardest - it drops
   the ``lambda dn/dlambda`` term entirely.

The headline is item 3: freezing the index does not just shift ``n_eff``, it
removes most of the material contribution to ``n_g``, so any delay,
free-spectral-range or dispersion number computed that way is wrong by far more
than it looks.

Needs the PyOptik snapshot (``pyoptik setup``); falls back to built-in
Sellmeier formulas with a warning if it is missing.

Run:  python examples/demo_material_dispersion.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402
from em_simulation.fde import EmepyFDE, FullEtchStrip  # noqa: E402
from em_simulation.fde.materials import (  # noqa: E402
    SELLMEIER_COEFFICIENTS,
    ConstantIndex,
    SellmeierMaterial,
    silica,
    silicon,
    silicon_nitride,
)

WIDTH = 500e-9
THICKNESS = 220e-9
REFERENCE_WL = 1.55e-6

# O through L band.
BANDS = {
    "O": (1.260e-6, 1.360e-6),
    "E": (1.360e-6, 1.460e-6),
    "S": (1.460e-6, 1.530e-6),
    "C": (1.530e-6, 1.565e-6),
    "L": (1.565e-6, 1.625e-6),
}
WL_MIN, WL_MAX = 1.26e-6, 1.625e-6


def figure_material_dispersion():
    """n(lambda) from the database, and the error in the offline fallbacks."""
    wl = np.linspace(WL_MIN, WL_MAX, 300)
    entries = [
        ("Si", silicon(), SellmeierMaterial(**SELLMEIER_COEFFICIENTS["Si"])),
        ("SiO2", silica(), SellmeierMaterial(**SELLMEIER_COEFFICIENTS["SiO2"])),
        ("Si3N4", silicon_nitride(), SellmeierMaterial(**SELLMEIER_COEFFICIENTS["Si3N4"])),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.2))
    fig.subplots_adjust(wspace=0.3)

    identical = []
    for k, (label, database, fallback) in enumerate(entries):
        n_db = database.n(wl)
        n_ref = database.n(REFERENCE_WL)

        # Plotted as a deviation from the 1550 nm value: on an absolute axis
        # spanning 1.44 to 3.48 the dispersion is a line width, which is
        # exactly the misreading this demo exists to correct.
        ax1.plot(
            wl * 1e9,
            (n_db - n_ref) * 1e3,
            color=f"C{k}",
            lw=1.4,
            label=f"{label}: n(1550) = {n_ref:.4f}",
        )

        difference = np.abs(n_db - fallback.n(wl))
        if difference.max() == 0:
            identical.append(label)
        else:
            ax2.semilogy(
                wl * 1e9, difference, color=f"C{k}", lw=1.3,
                label=f"{label}: {database.name} vs {fallback.name}",
            )

        print(f"  {label:6s} n(1550) = {n_ref:.5f}, "
              f"varies by {n_db.max()-n_db.min():.4f} over "
              f"{WL_MIN*1e9:.0f}-{WL_MAX*1e9:.0f} nm")

    for lo, _ in BANDS.values():
        ax1.axvline(lo * 1e9, color="grey", lw=0.4, alpha=0.5)
    for name, (lo, hi) in BANDS.items():
        ax1.text((lo + hi) / 2 * 1e9, ax1.get_ylim()[1], name,
                 ha="center", va="bottom", fontsize=7, color="grey")
    ax1.axhline(0, color="grey", lw=0.6)

    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel(r"$n(\lambda) - n(1550\,\mathrm{nm})$  ($\times 10^{-3}$)")
    ax1.set_title("dispersion from refractiveindex.info via PyOptik", fontsize=9)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("|database - built-in fallback|")
    ax2.set_title("cost of running without the database", fontsize=9)
    if identical:
        # The built-in formula for these *is* the database entry's formula, so
        # the fallback loses nothing at all.
        ax2.text(
            0.5, 0.06,
            "identical models, difference is exactly zero:\n" + ", ".join(identical),
            transform=ax2.transAxes, ha="center", va="bottom",
            fontsize=8, color="grey",
        )
    ax2.legend(fontsize=7, loc="upper right")

    save(fig, "materials_1_dispersion.png")


def _solve(wavelength, core, cladding, num_modes=2, mesh=120):
    """n_eff of the fundamental mode of the reference strip."""
    cross_section = FullEtchStrip(
        thickness=THICKNESS, core=core, cladding=cladding,
        reference_wavelength=wavelength,
    )
    backend = EmepyFDE(
        cross_section=cross_section,
        num_modes=num_modes,
        wavelength=wavelength,
        window=(1.6e-6, -0.8e-6, 0.8e-6),
        mesh=mesh,
    )
    return np.real(backend.solve((WIDTH, 0.0)).neff[0])


def figure_effective_index():
    """n_eff and n_g of a real strip, dispersive vs frozen index."""
    # Put the reference wavelength exactly on the grid, so the frozen-index
    # curve can be checked against the dispersive one where they must agree.
    wl = np.unique(
        np.concatenate([np.linspace(WL_MIN, WL_MAX, 13), [REFERENCE_WL]])
    )
    reference = int(np.argmin(np.abs(wl - REFERENCE_WL)))

    core, cladding = silicon(), silica()
    frozen_core = ConstantIndex(core.n(REFERENCE_WL), name="Si frozen at 1550 nm")
    frozen_clad = ConstantIndex(cladding.n(REFERENCE_WL), name="SiO2 frozen at 1550 nm")

    print("\n  solving the 500 x 220 nm strip at 13 wavelengths, twice...")
    dispersive = np.array([_solve(w, core, cladding) for w in wl])
    frozen = np.array([_solve(w, frozen_core, frozen_clad) for w in wl])

    # n_g = n_eff - lambda * d n_eff / d lambda
    def group_index(n_eff):
        slope = np.gradient(n_eff, wl)
        return n_eff - wl * slope

    ng_dispersive = group_index(dispersive)
    ng_frozen = group_index(frozen)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.2))
    fig.subplots_adjust(wspace=0.3)

    ax1.plot(wl * 1e9, dispersive, "o-", ms=3, label="dispersive materials")
    ax1.plot(wl * 1e9, frozen, "s--", ms=3, label="index frozen at 1550 nm")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("$n_{eff}$ (TE0)")
    ax1.set_title(
        f"{WIDTH*1e9:.0f} x {THICKNESS*1e9:.0f} nm Si strip", fontsize=9
    )
    ax1.legend(fontsize=8)

    ax2.plot(wl * 1e9, ng_dispersive, "o-", ms=3, label="dispersive materials")
    ax2.plot(wl * 1e9, ng_frozen, "s--", ms=3, label="index frozen at 1550 nm")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("$n_g$ (TE0)")
    ax2.set_title("group index: waveguide-only vs full", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, "materials_2_effective_index.png")

    agreement = abs(dispersive[reference] - frozen[reference])
    print(f"\n  at {wl[reference]*1e9:.0f} nm:  n_eff = "
          f"{dispersive[reference]:.5f} dispersive, {frozen[reference]:.5f} "
          f"frozen  -> {agreement:.1e} apart, as they must be")
    print(f"  at {wl[0]*1e9:.0f} nm:  n_eff = {dispersive[0]:.5f} dispersive, "
          f"{frozen[0]:.5f} frozen  -> error {abs(dispersive[0]-frozen[0]):.4f}")
    ng_error = abs(ng_dispersive[reference] - ng_frozen[reference])
    print(f"  n_g at {wl[reference]*1e9:.0f} nm: {ng_dispersive[reference]:.4f} "
          f"dispersive, {ng_frozen[reference]:.4f} frozen "
          f"-> {100*ng_error/ng_dispersive[reference]:.1f}% low")
    print("  (n_eff must agree at the reference wavelength; n_g does not, "
          "because\n   freezing the index deletes the material dispersion "
          "term entirely.)")
    return wl, dispersive, frozen, ng_dispersive, ng_frozen


def main():
    print("Material dispersion via PyOptik")
    print("-" * 64)

    print("\nmaterials:")
    figure_material_dispersion()

    figure_effective_index()

    print(
        "\nA constant index is fine at the one wavelength it was taken at, and "
        "wrong across a band.\nThe dataset is single-wavelength today, so this "
        "matters when you rebuild it at\nanother wavelength - and it will "
        "matter directly once a wavelength axis is added."
    )


if __name__ == "__main__":
    main()
