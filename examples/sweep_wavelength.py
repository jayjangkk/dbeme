"""O-band response of the coupler and the rotator, 1260-1360 nm.

Both devices in Sacher et al. were designed for the C band.  This asks what they
do in the O band, which is a different question from "what is their bandwidth" -
the anti-crossings that both rely on sit at particular *widths*, and those
widths move with wavelength.

Quantities, per device:

* **adiabatic coupler** - TE0 insertion loss (it should pass straight through)
  and TE1 crossover into the narrow guide.
* **bi-level taper** - TM0 -> TE1 conversion efficiency, and TE0 insertion loss.

A note on TE1 -> TM0 in the coupler.  There is no such process: the coupler is
two fully etched strips with symmetric cladding, which has a horizontal mirror
plane, so TE and TM are in different symmetry classes and cannot couple at all.
The script measures the TE1 -> TM crosstalk anyway and reports it, because a
number is a better answer than an assertion.  Polarization conversion happens
only in the bi-level taper, where the partial etch removes that mirror plane.

Method
------
One dataset per wavelength - a DBEME dataset is keyed to the mode problem and
the wavelength is part of that key.  For a *single* device at each wavelength
the dataset earns nothing, because there is no reuse across wavelengths, so the
sweep runs on direct EME.  The dataset pays off when you then sweep length or
shape at a fixed wavelength, which ``--validate`` demonstrates by rebuilding one
wavelength both ways and comparing.

Run:  python examples/sweep_wavelength.py [--validate]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402
from dbeme import (  # noqa: E402
    EME,
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
)
from dbeme.matrix_calculation_tool import _redheffer_star_product  # noqa: E402

import demo_adiabatic_coupler as coupler_demo  # noqa: E402
import demo_polarization_rotator as rotator_demo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVELENGTHS_NM = (1260, 1280, 1300, 1310, 1320, 1340, 1360)

#: Published stage lengths (Flexcompute's reproduction of the Sacher device).
COUPLER_LENGTH = 300e-6
ROTATOR_LENGTH = 100e-6


def _dataset(prefix, nm_value):
    return os.path.join(ROOT, "datasets", f"{prefix}_{nm_value}")


def _lumped(geometry, length):
    """Whole-device S-matrix at a given length, unitary-projected."""
    eme = EME(geometry, force_unitary=True)
    eme.calc_Smatrix()
    smatrix = eme.propagator._find_Smatrix_new_length(length)
    lumped = smatrix[0]
    for j in range(1, len(smatrix)):
        lumped = _redheffer_star_product(lumped, smatrix[j])
    return lumped


def _launch(lumped, index, n_modes):
    out = np.zeros(2 * n_modes, dtype=complex)
    out[index] = 1.0
    return np.abs(lumped @ out) ** 2


# ---------------------------------------------------------------- coupler


def coupler_point(nm_value, length=COUPLER_LENGTH):
    """TE0 insertion loss and TE1 crossover at one wavelength."""
    extractor = DataExtractor(_dataset("Si_pair_fulletch_220nm", nm_value),
                             cache_size=2048)
    path = DirectParametricPath(
        extractor,
        coupler_demo.width_schedule(COUPLER_LENGTH),
        total_length=COUPLER_LENGTH,
        # 80 sections over the same width range gives ~3.8 nm width steps,
        # inside the <=5 nm requirement measured in reports/01 section 2.
        resolution=80,
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

    stay, cross = coupler_demo.identify_branches(path, fractions)
    lumped = _lumped(path, length)
    te_pol = np.real(path.output_data["TE_pol"])

    from_te0 = _launch(lumped, stay, n_modes)
    from_te1 = _launch(lumped, cross, n_modes)

    # Everything that ended up TM-polarised, launching TE1.
    tm_out = float(sum(
        from_te1[i] for i in range(n_modes) if te_pol[-1, i] < 0.5
    ))
    return {
        "te0_through": float(from_te0[stay]),
        "te1_crossover": float(from_te1[cross]),
        "te1_to_tm_crosstalk": tm_out,
        "sections": len(path.output_data["EME_path"]),
    }


# ---------------------------------------------------------------- rotator


def rotator_point(nm_value, length=ROTATOR_LENGTH):
    """TM0 -> TE1 conversion and TE0 insertion loss at one wavelength."""
    extractor = DataExtractor(_dataset("Si_bilevel_220_90nm", nm_value),
                              cache_size=2048)
    path = DirectParametricPath(
        extractor,
        rotator_demo.width_schedule(ROTATOR_LENGTH),
        total_length=ROTATOR_LENGTH,
        # ~15 nm slab steps through the anti-crossing; the rotator's branches
        # stay ~0.17 apart in n_eff, so even 20 nm mixes them by only 0.08.
        resolution=150,
    )
    path._verbose = False
    path.calc_output_data()

    n_modes = path.output_data["neff"].shape[1] // 2
    te0, rotating = rotator_demo.identify_branches(path)
    lumped = _lumped(path, length)
    te_pol = np.real(path.output_data["TE_pol"])

    from_tm0 = _launch(lumped, rotating, n_modes)
    from_te0 = _launch(lumped, te0, n_modes)
    return {
        "tm0_to_te1": float(from_tm0[rotating]),
        "te0_through": float(from_te0[te0]),
        "output_te_fraction": float(te_pol[-1, rotating]),
        "sections": len(path.output_data["EME_path"]),
    }


# --------------------------------------------------------------- reporting


def to_db(value):
    """Insertion loss in dB from a transmitted power."""
    return -10.0 * np.log10(np.clip(value, 1e-12, None))


def figure(results):
    wavelengths = np.array(results["wavelength_nm"], dtype=float)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.4))
    fig.subplots_adjust(wspace=0.34)

    coupler = results["coupler"]
    rotator = results["rotator"]

    ax1.plot(wavelengths, to_db(np.array(coupler["te0_through"])), "o-", ms=4,
             color="C0", label="coupler, TE0")
    ax1.plot(wavelengths, to_db(np.array(rotator["te0_through"])), "s-", ms=4,
             color="C2", label="rotator, TE0")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("insertion loss (dB)")
    ax1.set_title("TE0 pass-through loss", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.plot(wavelengths, 100 * np.array(coupler["te1_crossover"]), "o-", ms=4,
             color="C1")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("TE1 into the narrow guide (%)")
    ax2.set_title(f"coupler crossover, L = {COUPLER_LENGTH*1e6:.0f} um", fontsize=9)
    ax2.set_ylim(0, 105)

    ax3.plot(wavelengths, 100 * np.array(rotator["tm0_to_te1"]), "d-", ms=4,
             color="C3")
    ax3.set_xlabel("wavelength (nm)")
    ax3.set_ylabel("TM0 -> TE1 conversion (%)")
    ax3.set_title(f"rotator conversion, L = {ROTATOR_LENGTH*1e6:.0f} um", fontsize=9)
    ax3.set_ylim(0, 105)

    for ax in (ax1, ax2, ax3):
        ax.axvspan(1260, 1360, color="grey", alpha=0.07, lw=0)

    save(fig, "oband_1_sweep.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="also rebuild 1310 nm as a DBEME dataset and compare")
    args = parser.parse_args()

    print("O-band sweep, 1260-1360 nm")
    print("=" * 70)
    print("one dataset per wavelength; the sweep itself runs on direct EME")
    print()

    results = {
        "wavelength_nm": list(WAVELENGTHS_NM),
        "coupler_length_um": COUPLER_LENGTH * 1e6,
        "rotator_length_um": ROTATOR_LENGTH * 1e6,
        "coupler": {k: [] for k in
                    ("te0_through", "te1_crossover", "te1_to_tm_crosstalk", "sections")},
        "rotator": {k: [] for k in
                    ("tm0_to_te1", "te0_through", "output_te_fraction", "sections")},
    }

    print(f"{'lambda':>8} | {'coupler TE0 IL':>14} {'TE1 cross':>10} {'TE1->TM':>9} "
          f"| {'rotator TM0->TE1':>16} {'TE0 IL':>8}")
    print("-" * 78)
    for nm_value in WAVELENGTHS_NM:
        t0 = time.time()
        c = coupler_point(nm_value)
        r = rotator_point(nm_value)
        for key, value in c.items():
            results["coupler"][key].append(value)
        for key, value in r.items():
            results["rotator"][key].append(value)
        print(
            f"{nm_value:6d}nm | {to_db(c['te0_through']):11.3f} dB "
            f"{100*c['te1_crossover']:9.2f}% {c['te1_to_tm_crosstalk']:9.1e} "
            f"| {100*r['tm0_to_te1']:15.2f}% {to_db(r['te0_through']):7.3f} dB"
            f"   [{time.time()-t0:.0f} s]",
            flush=True,
        )

    if args.validate:
        print("\ndataset EME vs direct EME at 1310 nm")
        print("-" * 70)
        results["validation"] = validate_1310()

    report_dir = os.path.join(ROOT, "reports", "output")
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "oband_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")

    figure(results)

    import shutil

    from _plotting import OUTPUT_DIR

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith("oband_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote {os.path.relpath(os.path.join(report_dir, name), ROOT)}")

    return results


def validate_1310():
    """Rebuild 1310 nm as a real DBEME dataset and check it against direct EME.

    This is the part that answers "do I have to build a new library": yes, and
    once built it gives the same answer as direct EME while making every
    further device at that wavelength nearly free.
    """
    dataset = _dataset("Si_bilevel_220_90nm", 1310)
    updater = DataUpdater(dataset)

    t0 = time.time()
    path = ParametricPath(
        updater,
        rotator_demo.width_schedule(ROTATOR_LENGTH),
        total_length=ROTATOR_LENGTH,
        verbose=True,
    )
    path.calc_output_data()
    build_seconds = time.time() - t0

    te0, rotating = rotator_demo.identify_branches(path)
    n_modes = path.output_data["neff"].shape[1] // 2
    lengths = np.array([50e-6, 100e-6, 200e-6, 300e-6])
    dataset_curve = [
        float(_launch(_lumped(path, L), rotating, n_modes)[rotating]) for L in lengths
    ]

    t0 = time.time()
    direct = rotator_point(1310)
    direct_seconds = time.time() - t0

    extractor = DataExtractor(dataset, cache_size=2048)
    direct_path = DirectParametricPath(
        extractor, rotator_demo.width_schedule(ROTATOR_LENGTH),
        total_length=ROTATOR_LENGTH, resolution=150,
    )
    direct_path._verbose = False
    direct_path.calc_output_data()
    d_te0, d_rot = rotator_demo.identify_branches(direct_path)
    d_modes = direct_path.output_data["neff"].shape[1] // 2
    direct_curve = [
        float(_launch(_lumped(direct_path, L), d_rot, d_modes)[d_rot]) for L in lengths
    ]

    print(f"{'length':>8}  {'dataset':>9}  {'direct':>9}  {'diff':>9}")
    for L, a, b in zip(lengths, dataset_curve, direct_curve):
        print(f"{L*1e6:7.0f}u  {a:9.5f}  {b:9.5f}  {abs(a - b):9.2e}")
    print(f"dataset built in {build_seconds:.0f} s; it now holds "
          f"{len(updater.neff)} points and further devices at 1310 nm are free")
    updater.save_data()

    return {
        "lengths_um": (lengths * 1e6).tolist(),
        "dataset": dataset_curve,
        "direct": direct_curve,
        "max_difference": float(np.max(np.abs(
            np.array(dataset_curve) - np.array(direct_curve)))),
        "dataset_build_seconds": build_seconds,
        "direct_seconds": direct_seconds,
        "dataset_points": len(updater.neff),
    }


if __name__ == "__main__":
    main()
