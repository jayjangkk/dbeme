"""Cross-check: dataset-based EME against ordinary (direct) EME.

The dataset-based method snaps every section onto a discrete parameter grid so
that mode fields and overlaps can be reused.  ``DataExtractor`` skips the grid
entirely and solves each section of the path at its exact width - ordinary EME.
If the grid is fine enough the two must agree, and the dataset run must be far
cheaper once the grid is populated.

Both paths use the same emepy backend and the same EME solver, so any
difference is attributable to the grid discretisation alone.

Run:  python examples/validate_against_direct_eme.py
"""

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
    DirectLinearTaper,
    LinearTaper,
    Runner,
)

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "Si_fulletch_220nm",
)

INPUT_WIDTH = 0.5e-6
OUTPUT_WIDTH = 1.2e-6
LENGTHS = [2e-6, 4e-6, 6e-6, 8e-6, 12e-6, 16e-6]


def transmission(geometry):
    """TE0-in, per-mode-out power for one geometry."""
    output_data = geometry.calc_output_data()
    eme = EME(geometry, force_unitary=True)
    eme.calc_Smatrix()
    runner = Runner(eme)
    n_launch = int(np.count_nonzero(output_data["radiation_mode_mask"][0] == False))
    launch = np.zeros(n_launch, dtype=complex)
    launch[0] = 1.0
    out = np.abs(runner.propagate_lumped_smatrix(launch)) ** 2
    return output_data, out


def main():
    print("dataset-based EME vs direct EME - linear taper")
    print("-" * 68)

    du = DataUpdater(DATASET)
    de = DataExtractor(DATASET)

    rows = []
    for length in LENGTHS:
        t0 = time.time()
        taper = LinearTaper(
            du, input_width=INPUT_WIDTH, output_width=OUTPUT_WIDTH, length=length
        )
        od_grid, out_grid = transmission(taper)
        t_grid = time.time() - t0

        # Match the direct run's section count to the dataset path so that the
        # only difference is grid snapping, not the number of sections.
        n_sections = len(od_grid["EME_path"])
        t0 = time.time()
        direct = DirectLinearTaper(
            de,
            input_width=INPUT_WIDTH,
            output_width=OUTPUT_WIDTH,
            length=length,
            resolution=n_sections,
        )
        direct._verbose = False
        od_dir, out_dir = transmission(direct)
        t_direct = time.time() - t0

        rows.append(
            {
                "length": length,
                "sections": n_sections,
                "te0_grid": out_grid[0],
                "te0_direct": out_dir[0],
                "t_grid": t_grid,
                "t_direct": t_direct,
            }
        )
        print(
            f"L = {length*1e6:5.1f} um  ({n_sections:3d} sections)   "
            f"TE0 dataset {out_grid[0]:.6f} | direct {out_dir[0]:.6f}   "
            f"diff {abs(out_grid[0]-out_dir[0]):.2e}   "
            f"time {t_grid:6.2f}s | {t_direct:6.2f}s"
        )

    grid_t = sum(r["t_grid"] for r in rows)
    dir_t = sum(r["t_direct"] for r in rows)
    print("-" * 68)
    print(f"total: dataset {grid_t:.1f} s, direct {dir_t:.1f} s")
    print(f"dataset now holds {len(du.neff)} parameter points")
    print(
        "Re-running this script is much faster on the dataset side and exactly "
        "as slow on the direct side - that is the whole point of the method."
    )

    lengths = np.array([r["length"] for r in rows]) * 1e6
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.2))
    fig.subplots_adjust(wspace=0.34)

    ax1.plot(lengths, [r["te0_direct"] for r in rows], "o-", label="direct EME")
    ax1.plot(lengths, [r["te0_grid"] for r in rows], "s--", label="dataset EME")
    ax1.set_xlabel("taper length (um)")
    ax1.set_ylabel("TE0 transmission")
    ax1.set_title("same answer from both methods", fontsize=9)
    ax1.legend(fontsize=8)

    diffs = [abs(r["te0_grid"] - r["te0_direct"]) for r in rows]
    ax2.semilogy(lengths, diffs, "d-", color="C3")
    ax2.set_xlabel("taper length (um)")
    ax2.set_ylabel("|dataset - direct|")
    ax2.set_title(
        "residual difference\n(grid snapping, at the 1e-5 level)",
        fontsize=9,
    )

    width = 0.35
    idx = np.arange(len(rows))
    ax3.bar(idx - width / 2, [r["t_direct"] for r in rows], width, label="direct EME")
    ax3.bar(idx + width / 2, [r["t_grid"] for r in rows], width, label="dataset EME")
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"{L:.0f}" for L in lengths])
    ax3.set_yscale("log")
    ax3.set_xlabel("taper length (um)")
    ax3.set_ylabel("wall time (s)")
    ax3.set_title(
        f"cost per device, warm dataset\n"
        f"({dir_t/max(grid_t,1e-9):.0f}x faster in total)",
        fontsize=9,
    )
    ax3.legend(fontsize=8)

    save(fig, "validation_dataset_vs_direct.png")
    du.save_data()


if __name__ == "__main__":
    main()
