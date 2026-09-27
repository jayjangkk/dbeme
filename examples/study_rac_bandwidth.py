"""Splitting ratio of the RAC coupling region over wavelength *and* length.

`demo_rapid_adiabatic_coupler.py` reports the wavelength sweep at one length,
11.5 um, which is half the published ``L_c = L_g + L_e = 23 um`` on the
assumption that regions II and III split it evenly - the thesis does not give
``L_g`` and ``L_e`` separately.  On the air-clad stack that length leaves the
stage short of balanced, so its absolute splitting ratio is not the quantity to
put next to the measured 50 +- 1.4 %.

The fix costs almost nothing.  A RAC is a *shape*, so rescaling the device in z
reuses the very same cross sections: one path per wavelength, evaluated at
several lengths.  This script builds those paths once and reports the band
flatness at each length, which is the number the published claim can be
compared against.

Run:  python examples/study_rac_bandwidth.py [air|oxide]
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402

CLAD = sys.argv[1] if len(sys.argv) > 1 else "air"
os.environ["RAC_CLAD"] = CLAD

import demo_rapid_adiabatic_coupler as rac  # noqa: E402

#: Lengths to evaluate, in metres.  23 um is the published coupling length for
#: regions II and III together, so region III alone reaching balance near there
#: is the interesting comparison.
LENGTHS = (11.5e-6, 16e-6, 23e-6, 30e-6, 40e-6)

#: The measured band is 50 +- 1.4 % over 145 nm centred near 1550 nm.
TOLERANCE = 1.4

#: Air on a buried oxide box cuts off at 1.444, so a 6-mode solve carries three
#: sub-cutoff modes whose fields are set by the window.  They are harmless in a
#: direct path but they are not physics, and ``splitting`` runs
#: ``force_unitary=True``, which would quietly project the resulting
#: non-unitarity away.  Restrict the basis to the bound modes for the numbers
#: that get compared against measurement.
BOUND_MODES = 3 if CLAD == "air" else None


def _extractor(folder):
    """A DataExtractor, with the mode basis trimmed to bound modes if asked."""
    if BOUND_MODES is None:
        return rac.DataExtractor(folder, cache_size=16384)

    import importlib.util

    from dbeme.fde import EmepyFDE

    spec = importlib.util.spec_from_file_location(
        "dataset_info", os.path.join(folder, "dataset_info.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = module.DatasetInfo()

    backend = EmepyFDE(
        cross_section=info.get_cross_section(),
        parameter_names=list(info.get_parameter_names()),
        num_modes=BOUND_MODES,
        wavelength=info.get_wavelength(),
        window=info._window,
        mesh=200,
    )
    extractor = rac.DataExtractor(folder, backend=backend, cache_size=16384)
    extractor.mode_numbers = BOUND_MODES
    return extractor


def main():
    print(f"RAC coupling region - splitting ratio vs wavelength and length ({CLAD})")
    print("=" * 74)

    # The tilt the demo settled on for this stack.
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "output", f"rac_{CLAD}_results.json",
    )
    with open(results_file, encoding="utf-8") as handle:
        best_tilt = json.load(handle)["tilt_scan"]["best_tilt"]
    basis = BOUND_MODES if BOUND_MODES else "all"
    print(f"using the demo's best constant tilt: {best_tilt:+.2f} degrees, "
          f"{basis} modes\n")

    curves = {"ac": {}, "rac": {}}
    for nm_value in rac.WAVELENGTHS_NM:
        folder = os.path.join(
            rac.ROOT, "datasets", f"Si_rac_region3_220nm{rac.SUFFIX}_{nm_value}"
        )
        extractor = _extractor(folder)
        t0 = time.time()
        for name, tilt in (("ac", 0.0), ("rac", best_tilt)):
            path = rac.build_direct(extractor, rac.schedule(tilt))
            curves[name][nm_value] = [
                rac.splitting(path, length=L) for L in LENGTHS
            ]
        print(f"  {nm_value} nm  [{time.time() - t0:.0f} s]", flush=True)

    print(f"\n{'length':>8}  " + "  ".join(f"{n:>7d}" for n in rac.WAVELENGTHS_NM)
          + f"  {'worst |dev|':>12}  {'in band':>8}")
    summary = []
    for i, L in enumerate(LENGTHS):
        row = [curves["rac"][n][i]["split_percent"] for n in rac.WAVELENGTHS_NM]
        worst = float(np.abs(np.array(row) - 50).max())
        summary.append({"length_um": L * 1e6, "split": row, "worst_deviation": worst})
        print(f"  {L*1e6:6.1f}u  " + "  ".join(f"{v:7.2f}" for v in row)
              + f"  {worst:12.2f}  {'yes' if worst <= TOLERANCE else 'no':>8}")

    best = min(summary, key=lambda s: s["worst_deviation"])
    span = max(rac.WAVELENGTHS_NM) - min(rac.WAVELENGTHS_NM)
    print(
        f"\n  flattest at L = {best['length_um']:.1f} um: "
        f"50 +- {best['worst_deviation']:.2f} % over {span} nm"
    )
    print(f"  measured (full 4-region device): 50 +- 1.4 % over 145 nm")

    figure(curves, summary, best_tilt)

    payload = {
        "cladding": CLAD,
        "mode_numbers": BOUND_MODES,
        "best_tilt": best_tilt,
        "wavelength_nm": list(rac.WAVELENGTHS_NM),
        "lengths_um": [L * 1e6 for L in LENGTHS],
        "ac": {str(n): curves["ac"][n] for n in rac.WAVELENGTHS_NM},
        "rac": {str(n): curves["rac"][n] for n in rac.WAVELENGTHS_NM},
        "summary": summary,
        "flattest": best,
    }
    report_dir = os.path.join(rac.ROOT, "reports", "output")
    out = os.path.join(report_dir, f"rac_{CLAD}_bandwidth.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote reports/output/rac_{CLAD}_bandwidth.json")

    import shutil

    from _plotting import OUTPUT_DIR

    name = f"rac_{CLAD}_6_bandwidth.png"
    shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
    print(f"wrote reports/output/{name}")
    return payload


def figure(curves, summary, best_tilt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.5))
    fig.subplots_adjust(wspace=0.30)

    wavelengths = list(rac.WAVELENGTHS_NM)
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(LENGTHS)))
    for i, L in enumerate(LENGTHS):
        ax1.plot(
            wavelengths,
            [curves["rac"][n][i]["split_percent"] for n in wavelengths],
            "o-", ms=4, color=colours[i], label=f"{L*1e6:.0f} um",
        )
    ax1.plot(
        wavelengths,
        [curves["ac"][n][LENGTHS.index(23e-6)]["split_percent"] for n in wavelengths],
        "s--", ms=4, color="C3", label="AC, 23 um",
    )
    ax1.axhspan(50 - TOLERANCE, 50 + TOLERANCE, color="C2", alpha=0.15, lw=0)
    ax1.axhline(50, color="grey", lw=0.7, ls=":")
    ax1.text(wavelengths[0], 50 + TOLERANCE + 0.4, "measured 50 $\\pm$ 1.4 %",
             fontsize=7, color="C2")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("splitting ratio (%)")
    ax1.set_title(f"region III at {best_tilt:+.1f}$\\degree$, rescaled in z",
                  fontsize=9)
    ax1.legend(fontsize=7, ncol=2)

    ax2.plot([s["length_um"] for s in summary],
             [s["worst_deviation"] for s in summary], "o-", color="C0", ms=5)
    ax2.axhline(TOLERANCE, color="C2", lw=1.0, ls="--")
    ax2.text(summary[0]["length_um"], TOLERANCE * 1.1, "measured tolerance",
             fontsize=7, color="C2")
    ax2.set_xlabel("coupling-region length (um)")
    ax2.set_ylabel("worst |splitting - 50| over the band (%)")
    ax2.set_title("how long region III has to be to hold the band", fontsize=9)

    save(fig, f"rac_{CLAD}_6_bandwidth.png")


if __name__ == "__main__":
    main()
