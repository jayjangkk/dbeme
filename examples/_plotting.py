"""Shared plotting helpers for the demo scripts.

Uses the Agg backend so the demos run headless and write PNGs.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
    return path


def guided_mode_indices(output_data, section=0):
    """Indices of tracked modes that are guided at the given section."""
    mask = output_data["radiation_mode_mask"]
    n_modes = mask.shape[1] // 2
    return [i for i in range(n_modes) if not mask[section, i]]


def mode_labels(output_data, indices, section=-1):
    """Name each tracked mode ``TE<k>``/``TM<k>`` from its polarisation and order.

    Tracked-mode indices are assigned in the order modes first appear along the
    path, which is not the same as polarisation order, so the label is derived
    from the TE fraction and the effective-index ranking within a polarisation.
    """
    te_pol = np.real(output_data["TE_pol"][section])
    neff = np.real(output_data["neff"][section])

    te = sorted([i for i in indices if te_pol[i] >= 0.5], key=lambda i: -neff[i])
    tm = sorted([i for i in indices if te_pol[i] < 0.5], key=lambda i: -neff[i])

    labels = {}
    for order, i in enumerate(te):
        labels[i] = f"TE{order}"
    for order, i in enumerate(tm):
        labels[i] = f"TM{order}"
    return labels


def propagation_axis(output_data):
    """Cumulative propagation length in um at every EME section boundary."""
    dz = np.asarray(output_data["EME_delta_zs"], dtype=float)
    return np.concatenate(([0.0], np.cumsum(dz))) * 1e6


def section_amplitudes(runner, sectional_amplitudes):
    """Reduce the per-matrix amplitudes to one row per EME section boundary.

    ``SingleEME`` interleaves phase-propagation and interface matrices, so the
    runner returns ``2 * n_sections - 1`` rows.  Sampling every other row (plus
    the launch row) recovers the amplitudes at the section boundaries, which is
    what the propagation axis indexes.
    """
    sa = np.asarray(sectional_amplitudes)
    return np.insert(sa[1::2], 0, sa[0], axis=0)
