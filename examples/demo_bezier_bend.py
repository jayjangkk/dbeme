"""Demo 2 - Bezier S-bend in a fully etched Si waveguide.

A bend breaks the lateral symmetry of a waveguide, so unlike a symmetric taper
it couples TE0 to TE1 directly.  In a multimode waveguide that coupling is the
dominant impairment, and what drives it is how abruptly the curvature
*changes* - which is why bend shape matters at all.

This demo puts a 1.0 um wide (multimode) Si strip through an S-bend and
compares three centreline shapes at the same footprint:

* **circular arc** - curvature jumps 0 -> +1/R at the input, +1/R -> -1/R at
  the join, and -1/R -> 0 at the output.  Three hard steps.
* **cubic Bezier** - curvature varies smoothly in between, but is still
  non-zero where the bend meets the straight waveguide, so two steps remain.
* **quintic Bezier** - the first three and last three control points are
  collinear, which forces zero curvature at both ends.  No steps at all.

All three are built with straight lead-in and lead-out sections, so the
junctions to the feeding waveguide are part of the simulation rather than
assumed away.  Each shape is a different path through the same dataset, so
after the first run the comparison costs no mode solving.

Run:  python examples/demo_bezier_bend.py
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
from em_simulation import EME, DataUpdater, Runner, SingleCustomBend  # noqa: E402
from em_simulation.geometry.bend_shapes.bezier import BezierCurve  # noqa: E402

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "Si_fulletch_220nm",
)

WIDTH = 1.0e-6  # multimode Si strip: TE0, TE1, TE2 all guided
SPAN = 15e-6  # S-bend footprint along z
OFFSET = 5e-6  # lateral displacement
LEAD = 2e-6  # straight waveguide before and after the bend
RESOLUTION = 600


# --------------------------------------------------------------- centrelines


def bezier_curvature(control_points, resolution=RESOLUTION):
    """Sample a Bezier centreline's arc length and curvature."""
    curve = BezierCurve(control_points, resolution=resolution)
    s, kappa, _ = curve._calc_parameters()
    return np.asarray(s), np.asarray(kappa)


def cubic_s_bend(span, offset):
    """The usual cubic Bezier S-bend.

    The curvature at an endpoint of a Bezier is set by the triangle formed by
    the first three control points.  Here P0-P1 runs along z and P1-P2 runs
    across it, so that triangle has area and the curvature starts finite.
    """
    return [
        (0.0, 0.0),
        (0.5 * span, 0.0),
        (0.5 * span, offset),
        (span, offset),
    ]


def quintic_s_bend(span, offset, s=0.25):
    """Quintic Bezier with collinear end triples, so curvature starts at zero.

    :param s: How far along z each collinear run extends, as a fraction of span.
    """
    return [
        (0.0, 0.0),
        (s * span, 0.0),
        (2 * s * span, 0.0),
        (span - 2 * s * span, offset),
        (span - s * span, offset),
        (span, offset),
    ]


def circular_s_bend(span, offset, resolution=RESOLUTION):
    """Two opposed circular arcs joined at the midpoint.

    Each arc has radius ``R`` and turns through ``theta``.  Matching the
    end-to-end displacement gives

        2 R sin(theta)       = span
        2 R (1 - cos theta)  = offset

    whose ratio is ``tan(theta/2) = offset / span``.

    :returns: ``(arc length samples, curvature, radius)`` in SI units.
    """
    theta = 2.0 * np.arctan(offset / span)
    R = span / (2.0 * np.sin(theta))
    arc_len = R * theta
    s = np.linspace(0, 2 * arc_len, resolution)
    kappa = np.where(s < arc_len, 1.0 / R, -1.0 / R)
    return s, kappa, R


def with_leads(s, kappa, lead=LEAD, eps=1e-9):
    """Bracket a curvature profile with straight sections.

    The junction between a straight waveguide and a bend is where an arc pays
    most of its penalty, so it has to be inside the simulated structure.  The
    ``eps`` offsets make the curvature step abrupt rather than ramped, since
    ``SingleCustomBend`` interpolates linearly between the samples it is given.
    """
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    end = lead + s[-1]
    s_full = np.concatenate(
        ([0.0, lead - eps], lead + s, [end + eps, end + lead])
    )
    k_full = np.concatenate(([0.0, 0.0], kappa, [0.0, 0.0]))
    return s_full, k_full


# --------------------------------------------------------------------- solve


def build(du, s, kappa, width=WIDTH):
    s_full, k_full = with_leads(s, kappa)
    bend = SingleCustomBend(
        du,
        prop_len_list=s_full,
        width_list=np.full_like(s_full, width),
        curvature_list=k_full,
        resolution=RESOLUTION,
        verbose=False,
    )
    return bend, bend.calc_output_data()


def run_eme(geometry, output_data):
    # Lossless by construction (this backend has no PML), so project each
    # section's S-matrix onto the nearest unitary matrix.  That removes the
    # ~1e-4-per-interface drift a truncated mode basis introduces, without
    # discarding any physics the solver actually models.
    eme = EME(geometry, force_unitary=True)
    eme.calc_Smatrix()
    runner = Runner(eme)
    n_launch = int(np.count_nonzero(output_data["radiation_mode_mask"][0] == False))
    launch = np.zeros(n_launch, dtype=complex)
    launch[0] = 1.0
    amplitudes = section_amplitudes(runner, runner.propagate(launch))
    return eme, amplitudes


# ------------------------------------------------------------------- figures


def figure_layout(bend, output_data, s_ideal, k_ideal):
    x, y = bend._calc_xy()
    bdr_x, bdr_y = bend.line_to_bdry_pts(x, y, WIDTH)

    z = propagation_axis(output_data)
    kappa_grid = np.array([p[1] for p in output_data["EME_path"]])
    s_full, k_full = with_leads(s_ideal, k_ideal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.0))

    ax1.fill(bdr_x * 1e6, bdr_y * 1e6, color="C0", alpha=0.35, lw=0)
    ax1.plot(np.asarray(x) * 1e6, np.asarray(y) * 1e6, "k--", lw=0.7, label="centreline")
    ax1.set_xlabel("z (um)")
    ax1.set_ylabel("lateral (um)")
    ax1.set_aspect("equal")
    ax1.set_title(
        f"cubic Bezier S-bend + {LEAD*1e6:.0f} um straight leads\n"
        f"{SPAN*1e6:.0f} x {OFFSET*1e6:.0f} um bend, {WIDTH*1e6:.1f} um wide",
        fontsize=9,
    )
    ax1.legend(fontsize=7)

    ax2.plot(s_full * 1e6, k_full * 1e-3, lw=1.2, label="ideal centreline")
    ax2.step(
        z,
        kappa_grid * 1e-3,
        where="post",
        color="C1",
        lw=1.0,
        label="dataset grid (5000 m$^{-1}$)",
    )
    ax2.plot(z, kappa_grid * 1e-3, ".", color="C1", ms=4)
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xlabel("propagation length (um)")
    ax2.set_ylabel("curvature (10$^3$ m$^{-1}$)")
    ax2.set_title(
        f"tightest radius {1e6/np.max(np.abs(k_ideal)):.1f} um, "
        f"{len(z)} EME sections",
        fontsize=9,
    )
    ax2.legend(fontsize=7)

    save(fig, "bezier_1_layout.png")


def figure_propagation(output_data, amplitudes, labels, guided):
    z = propagation_axis(output_data)
    neff = np.real(output_data["neff"])
    power = np.abs(amplitudes) ** 2
    kappa = np.abs(np.array([p[1] for p in output_data["EME_path"]]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.1))
    fig.subplots_adjust(wspace=0.42)

    # The absolute shift is ~1e-3, invisible on an absolute axis, so plot the
    # shift away from the straight input waveguide instead.
    for i in guided:
        ax1.plot(z, (neff[:, i] - neff[0, i]) * 1e3, label=labels[i])
    ax1.set_xlabel("propagation length (um)")
    ax1.set_ylabel(r"$n_{eff}-n_{eff}(0)$  ($\times 10^{-3}$)")
    ax1.set_title("the bend pushes each mode outwards", fontsize=9)
    ax1.legend(ncol=3, fontsize=7, loc="upper center")

    ax1b = ax1.twinx()
    ax1b.plot(z, kappa * 1e-3, color="grey", lw=0.9, ls=":")
    ax1b.set_ylabel("|curvature| (10$^3$ m$^{-1}$)", color="grey", fontsize=8)
    ax1b.tick_params(axis="y", labelcolor="grey", labelsize=7)
    ax1b.grid(False)

    for i in guided:
        ax2.plot(z, power[:, i], label=labels[i])
    ax2.set_xlabel("propagation length (um)")
    ax2.set_ylabel("$|a|^2$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 2)
    ax2.set_title("power with TE0 launched", fontsize=9)
    ax2.legend(ncol=2, fontsize=7)

    save(fig, "bezier_2_propagation.png")


def figure_shape_comparison(du, shapes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.2))

    for name, entry in shapes.items():
        s_full, k_full = with_leads(entry["s"], entry["kappa"])
        ax1.plot(s_full * 1e6, k_full * 1e-3, lw=1.3, label=name)
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set_xlabel("propagation length (um)")
    ax1.set_ylabel("curvature (10$^3$ m$^{-1}$)")
    ax1.set_title("curvature profile, including the junctions", fontsize=9)
    ax1.legend(fontsize=8)

    names = list(shapes)
    crosstalk = [1.0 - shapes[n]["out"][0] for n in names]

    bars = ax2.bar(range(len(names)), np.array(crosstalk) * 100, color=["C0", "C1", "C2"])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("power leaving TE0 (%)")
    ax2.set_yscale("log")
    ax2.set_title(
        f"TE0 crosstalk, {WIDTH*1e6:.1f} um multimode guide,\n"
        f"same {SPAN*1e6:.0f} x {OFFSET*1e6:.0f} um footprint",
        fontsize=9,
    )
    for b, c in zip(bars, crosstalk):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c*100:.3f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save(fig, "bezier_3_shape_comparison.png")


def main():
    print("Dataset-based EME - Bezier bend demo")
    print("-" * 64)

    du = DataUpdater(DATASET)
    print(f"dataset opened ({len(du.neff)} points already cached)")

    s_arc, k_arc, R = circular_s_bend(SPAN, OFFSET)
    s_cub, k_cub = bezier_curvature(cubic_s_bend(SPAN, OFFSET))
    s_qui, k_qui = bezier_curvature(quintic_s_bend(SPAN, OFFSET))

    definitions = {
        "circular arc": (s_arc, k_arc),
        "cubic Bezier": (s_cub, k_cub),
        "quintic Bezier": (s_qui, k_qui),
    }

    shapes = {}
    for name, (s, kappa) in definitions.items():
        t0 = time.time()
        bend, output_data = build(du, s, kappa)
        _, amplitudes = run_eme(bend, output_data)
        out = np.abs(amplitudes[-1]) ** 2
        shapes[name] = {
            "bend": bend,
            "od": output_data,
            "amps": amplitudes,
            "out": out,
            "s": s,
            "kappa": kappa,
        }
        print(
            f"  {name:>14}: bend length {s[-1]*1e6:5.2f} um, "
            f"tightest R {1e6/np.max(np.abs(kappa)):5.1f} um, "
            f"{len(output_data['EME_path']):3d} EME sections  "
            f"[{time.time()-t0:5.1f} s]"
        )
    print(f"dataset now holds {len(du.neff)} points")

    ref = shapes["cubic Bezier"]
    guided = guided_mode_indices(ref["od"], section=0)
    labels = mode_labels(ref["od"], guided, section=0)
    print("\nguided modes at the input:", [labels[i] for i in guided])

    n_modes = ref["od"]["neff"].shape[1] // 2
    print("\noutput power with TE0 launched (cubic Bezier):")
    for i in guided:
        p = ref["out"][i]
        print(f"   {labels[i]:>4}  {p:.6f}   ({10*np.log10(max(p,1e-12)):+7.2f} dB)")
    print(
        f"   {'sum':>4}  {ref['out'][:n_modes].sum():.6f} forward, "
        f"{ref['out'][n_modes:].sum():.2e} reflected"
    )

    print("\nTE0 crosstalk by bend shape (same footprint, junctions included):")
    for name in shapes:
        c = 1.0 - shapes[name]["out"][0]
        print(f"   {name:>14}  {c*100:8.4f} %   ({10*np.log10(max(c,1e-12)):+7.2f} dB)")

    print("\nfigures:")
    figure_layout(ref["bend"], ref["od"], s_cub, k_cub)
    figure_propagation(ref["od"], ref["amps"], labels, guided)
    figure_shape_comparison(du, shapes)

    du.save_data()


if __name__ == "__main__":
    main()
