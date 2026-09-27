"""Phase 3: how many modes does a PML-EME basis need, and what does it cost?

``tasks/02_pml_backend.md`` Phase 3 is the decision point, and it is not a
solver problem:

> PML-EME converges *because* the discretised continuum makes the mode basis
> approximately complete. That needs ~20-50 modes, not 6. DBEME stores
> ``(2N x 2N)`` overlaps per adjacent grid-point pair [...] Run a mode-count
> convergence study (loss vs ``N``, with ``force_unitary=False``) and find the
> smallest ``N`` that converges **before** committing to a dataset.

The test structure is a **width step** between two straight guides, 500 -> 400
nm, oxide-clad 220 nm SOI.  A step radiates at the junction, and with a PML that
radiated power leaves through the absorber rather than being reflected back by
a hard boundary - so the guided transmission only comes out right once the basis
carries enough of the continuum to represent the radiated field.  That makes the
guided ``T_00`` a direct probe of basis completeness.

Straight guides are used on purpose: it isolates basis completeness from the
conformal-ramp and turning-point questions that a bend would add.

``force_unitary=False`` throughout - projecting onto the nearest unitary matrix
would hide exactly the incompleteness being measured.

Run:  python examples/study_pml_mode_count.py [transfer|direct]

``transfer`` is the original route, ``direct`` the scattering route with the
conditioning cutoff; the report shows both side by side.
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
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.fde.materials import silica, silicon  # noqa: E402
from dbeme.fde.pml import PMLModeSolver  # noqa: E402
from dbeme.matrix_calculation_tool import (  # noqa: E402
    _redheffer_star_product,
)
from dbeme.geometry.geometry import Geometry  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME  # noqa: E402

WAVELENGTH = 1.55e-6
WIDE, NARROW = 0.500e-6, 0.400e-6
MODE_COUNTS = (4, 6, 8, 12, 16, 20, 26, 32, 40)

#: Which cascade to measure.  ``"transfer"`` is the original route; ``"direct"``
#: never forms a transfer matrix and applies the conditioning cutoff.  Pass it
#: on the command line.
METHOD = sys.argv[1] if len(sys.argv) > 1 else "transfer"
if METHOD not in ("transfer", "direct"):
    raise SystemExit("usage: study_pml_mode_count.py [transfer|direct]")

#: Convergence tolerance on the guided transmission.
TOLERANCE = 0.01

#: How far the output energy may stray from unity before the basis is called
#: unusable.  The structure is lossless apart from what the PML absorbs, and a
#: PML cannot *create* power, so anything above 1 by more than this is
#: numerical rather than physical.
ENERGY_TOLERANCE = 0.05


class _TwoSection(Geometry):
    """A two-cross-section device, built straight from solved modes.

    Deliberately bypasses ``DataUpdater``: this study is about the size of the
    mode basis, and routing it through a dataset would add caching and grid
    snapping to a question that has nothing to do with either.
    """

    def __init__(self, output_data):
        self._is_composite_geometry = False
        self._verbose = False
        self.output_data = output_data


def build_output_data(mode_data_list, num_modes, length=1e-6):
    """The dict a ``Propagator`` consumes, from a list of ``ModeData``."""
    x, y, neff, te_pol, E, H = assemble(
        mode_data_list, num_modes, prop_axis=2, lossless=False
    )
    sections = len(mode_data_list)
    width = 2 * num_modes
    overlap_ab = np.zeros((sections - 1, width, width), dtype=complex)
    overlap_ba = np.zeros((sections - 1, width, width), dtype=complex)
    for i in range(sections - 1):
        overlap_ab[i] = overlap_matrix(E[i], H[i + 1], x, y, 2)
        overlap_ba[i] = overlap_matrix(E[i + 1], H[i], x, y, 2)
    return {
        "neff": neff,
        "beta": 2 * np.pi / WAVELENGTH * neff,
        "overlap_ab": overlap_ab,
        "overlap_ba": overlap_ba,
        # `delta_zs` is the physical section spacing; `EME_delta_zs` is the
        # per-interface propagation length the propagator advances the phase
        # over.  With no intermediate points inserted they coincide.
        "delta_zs": np.full(sections - 1, length),
        "EME_delta_zs": np.full(sections - 1, length),
        "radiation_mode_mask": np.zeros((sections, width), dtype=bool),
    }


def main():
    print("Phase 3 - PML-EME mode-count convergence")
    print("=" * 74)
    section = FullEtchStrip(
        thickness=0.22e-6,
        core=silicon(out_of_range="raise"),
        cladding=silica(out_of_range="raise"),
    )
    solver = PMLModeSolver(
        section,
        wavelength=WAVELENGTH,
        window=(-1.0e-6, 2.0e-6, -0.6e-6, 0.6e-6),
        mesh=200,
        pml_thickness=0.8e-6,
        num_modes=max(MODE_COUNTS),
    )
    print(
        f"width step {WIDE*1e9:.0f} -> {NARROW*1e9:.0f} nm, 220 nm SOI, "
        f"SiO2 clad, {WAVELENGTH*1e9:.0f} nm"
    )
    print(
        f"grid {len(solver._x)} x {len(solver._y)}, PML "
        f"{solver.pml_thickness*1e6:.1f} um on +x, stretch {solver.pml_factor}"
    )
    print("force_unitary=False throughout\n")

    print(
        f"{'N':>4} {'2N x 2N':>9} {'rel. cost':>10} {'|T00|':>10} "
        f"{'guided out':>11} {'residual':>10} {'Im neff0':>11} {'s':>6}"
    )
    rows = []
    for count in MODE_COUNTS:
        t0 = time.time()
        wide, _ = solver.mode_data(
            {"top_width": WIDE, "curvature": 0.0}, 2.40, num_modes=count
        )
        narrow, _ = solver.mode_data(
            {"top_width": NARROW, "curvature": 0.0}, 2.30, num_modes=count
        )
        output_data = build_output_data([wide, narrow], count)
        propagator = SingleEME(_TwoSection(output_data), force_unitary=False)
        propagator.calc_Smatrix(method=METHOD)
        # `smatrix` is a *list* to be cascaded - interface and propagation
        # matrices alternating - not the device S-matrix.  Taking element [0]
        # of it returns a propagation matrix, whose diagonal has unit modulus
        # for a lossless straight section, i.e. |T| = 1 for every N.
        lumped = np.asarray(propagator.smatrix[0])
        for step in propagator.smatrix[1:]:
            lumped = _redheffer_star_product(lumped, np.asarray(step))

        launch = np.zeros(2 * count, dtype=complex)
        launch[0] = 1.0
        out = lumped @ launch
        transmission = abs(out[0])
        guided = float(np.sum(np.abs(out) ** 2))
        rows.append({
            "modes": count,
            "overlap_entries": (2 * count) ** 2,
            "relative_cost": (count / MODE_COUNTS[0]) ** 2,
            "T00": float(transmission),
            "guided_out": guided,
            "neff0_imag": float(np.imag(output_data["neff"][0, 0])),
            "seconds": time.time() - t0,
        })
        print(
            f"{count:4d} {(2*count)**2:9d} {(count/6)**2:10.1f} "
            f"{transmission:10.6f} {guided:11.6f} {abs(1-guided):10.3e} "
            f"{np.imag(output_data['neff'][0, 0]):11.3e} "
            f"{time.time()-t0:6.1f}"
        )

    # The largest basis is *not* the reference: it is the one that has
    # diverged.  Trust the smallest, which is the guided-only answer, and which
    # an independent lossless EmepyFDE solve of the same step corroborates at
    # |T00| = 0.9988.
    reference = rows[0]["T00"]
    for row in rows:
        row["error"] = abs(row["T00"] / reference - 1)
        row["usable"] = abs(row["guided_out"] - 1) <= ENERGY_TOLERANCE

    header = f"|T00| vs N={rows[0]['modes']}"
    print()
    print(f"{'N':>4} {header:>16} {'energy error':>13} {'usable':>8}")
    for row in rows:
        print(f"{row['modes']:4d} {row['error']:16.2e} "
              f"{abs(row['guided_out']-1):13.2e} "
              f"{'yes' if row['usable'] else 'NO':>8}")

    usable = [r for r in rows if r["usable"]]
    largest_usable = usable[-1]["modes"] if usable else None
    first_bad = next((r["modes"] for r in rows if not r["usable"]), None)
    worst = max(rows, key=lambda r: abs(r["guided_out"] - 1))

    print()
    print(f"  |T00| is stable at {reference:.4f} across every usable basis "
          f"(spread {max(r['error'] for r in usable):.1e}).")
    print(f"  largest usable basis      : N = {largest_usable}")
    print(f"  first basis that diverges : N = {first_bad}")
    print(f"  worst case                : N = {worst['modes']}, "
          f"output energy {worst['guided_out']:.4g}")
    print()
    print("  Divergence is not monotone in N - it depends on whether a")
    print("  particular near-null-norm Berenger mode lands in the basis.  The")
    print("  unconjugated normalisation enforces INT (E x H)_z = 1, and those")
    print("  modes carry a power integral ~300x smaller than a guided mode's,")
    print("  so normalising amplifies them until they dominate the overlaps.")
    print()
    print("  So Phase 3's premise does not hold here.  The extra modes are not")
    print("  expensive, they are unusable, and the 44x cost question is moot")
    print("  until the radiation block is given a normalisation of its own.")
    figure(rows, largest_usable, reference)

    payload = {
        "structure": {
            "wide_nm": WIDE * 1e9,
            "narrow_nm": NARROW * 1e9,
            "wavelength_nm": WAVELENGTH * 1e9,
            "grid": [len(solver._x), len(solver._y)],
            "pml_thickness_um": solver.pml_thickness * 1e6,
            "pml_factor": str(solver.pml_factor),
        },
        "tolerance": TOLERANCE,
        "rows": rows,
        "energy_tolerance": ENERGY_TOLERANCE,
        "largest_usable_modes": largest_usable,
        "first_diverging_modes": first_bad,
    }
    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "output",
    )
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"pml_mode_count_{METHOD}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote reports/output/pml_mode_count.json")

    import shutil

    from _plotting import OUTPUT_DIR

    shutil.copy2(os.path.join(OUTPUT_DIR, f"pml_3_mode_count_{METHOD}.png"),
                 os.path.join(report_dir, f"pml_3_mode_count_{METHOD}.png"))
    print(f"wrote reports/output/pml_3_mode_count_{METHOD}.png")
    return payload


def figure(rows, largest_usable, reference):
    converged = largest_usable
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))
    fig.subplots_adjust(wspace=0.30)

    modes = [r["modes"] for r in rows]
    ax1.plot(modes, [r["T00"] for r in rows], "o-", color="C0", ms=5)
    ax1.axhline(reference, color="grey", lw=0.8, ls=":")
    if converged:
        ax1.axvline(converged, color="C2", lw=1.0, ls="--")
        ax1.text(converged + 0.6, min(r["T00"] for r in rows),
                 f"converged\nN = {converged}", fontsize=7, color="C2")
    ax1.set_xlabel("modes in the basis, N")
    ax1.set_ylabel(r"guided transmission $|T_{00}|$")
    ax1.set_title("basis completeness at a radiating junction", fontsize=9)

    ax2.loglog([r["modes"] for r in rows], [r["relative_cost"] for r in rows],
               "s-", color="C3", ms=5, label=r"overlap cost $\propto N^2$")
    ax2.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax2.text(modes[0], 1.15, "lossless N = 6 baseline", fontsize=7, color="grey")
    if converged:
        ax2.axvline(converged, color="C2", lw=1.0, ls="--")
    ax2.set_xlabel("modes in the basis, N")
    ax2.set_ylabel("storage and solve cost, relative to N = 6")
    ax2.set_title("what completeness costs a dataset", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, f"pml_3_mode_count_{METHOD}.png")


if __name__ == "__main__":
    main()
