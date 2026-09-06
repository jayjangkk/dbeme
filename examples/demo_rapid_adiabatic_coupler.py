"""Demo 7 - rapid adiabatic coupler (RAC), Fargas Cabanillas thesis 3.5.

    J. M. Fargas Cabanillas, "Rapid adiabatic devices enabling integrated
    electronic-photonic quantum systems on chip", PhD thesis, Boston
    University (Popovic group), section 3.5.3 - the 220 nm SOI e-beam
    demonstration.

An adiabatic coupler is broadband and tolerant but long, because staying
adiabatic means keeping the inter-supermode coupling kappa_12 far below the
difference in propagation constants everywhere.  The RAC's insight is that
kappa_12 is not a fixed property of the width schedule:

    kappa_mn = (w/4) (b_m - b_n)^-1 SUM_p (dx_p/dz) INT e*_m . de . e_n |_x_p

Only the *wall slopes* dx_p/dz carry the z dependence, and the four walls
contribute with opposite signs.  Adding a global tilt theta to the whole cross
section adds the same tan(theta) to every wall slope, so there is a theta at
each position for which the sum is exactly zero.  Follow that trajectory
theta_RAC(eta) and the coupling is suppressed all along the device, which lets
it be far shorter than a conventional adiabatic coupler.

This demo models region III of the published 220 nm SOI device - the coupling
region, where the gap is fixed at 100 nm and the widths run from 480/380 nm to a
symmetric 380/380 nm pair.  The tilt enters as ``x_offset``, a rigid lateral
translation of the pair: it does nothing to a single cross section, but between
consecutive sections it displaces the modes relative to each other, which is
precisely the ``dx_p/dz`` term above.

Run:  python examples/demo_rapid_adiabatic_coupler.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import propagation_axis, save  # noqa: E402
from em_simulation import (  # noqa: E402
    EME,
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
)
from em_simulation.fde.assemble import assemble, overlap_matrix  # noqa: E402
from em_simulation.matrix_calculation_tool import _redheffer_star_product  # noqa: E402
from em_simulation.platforms import RAC_W_BOT, RAC_W_EQUAL  # noqa: E402
from em_simulation.validation import (  # noqa: E402
    check_branch_tracking,
    check_power_conservation,
    check_reciprocity,
    check_reflection_symmetry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: ``air`` is the stack of the *fabricated* device - e-beam on 220 nm SOI, left
#: unclad - and is what the published numbers should be compared against.
#: ``oxide`` is the buried stack the rest of this project uses, and is the more
#: relevant one for a foundry platform.  Pass either on the command line.
#: Read from argv only when this file is the program being run - importing it
#: from a test must not pick up the test runner's arguments.
CLAD = os.environ.get("RAC_CLAD") or (
    sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else "air"
)
if CLAD not in ("air", "oxide"):
    raise SystemExit("usage: demo_rapid_adiabatic_coupler.py [air|oxide]")
SUFFIX = "_air" if CLAD == "air" else ""
DATASET = os.path.join(ROOT, "datasets", f"Si_rac_region3_220nm{SUFFIX}")
TAG = f"rac_{CLAD}"

#: Region III alone is half of the published L_c = L_g + L_e = 23 um.
LENGTH = 11.5e-6
W1_IN, W1_OUT = RAC_W_BOT, RAC_W_EQUAL

#: Constant tilts to scan, in degrees.  Negative is the helpful sign.  Air
#: cladding confines the modes far more tightly, which strengthens the coupling
#: and pushes the nulling tilt roughly 1.8x steeper, so the two stacks need
#: different ranges.
TILTS = (
    (0.0, -1.0, -2.0, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0)
    if CLAD == "air"
    else (1.0, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0)
)

#: Range of the coupling map, in degrees, and its number of samples.
TILT_RANGE = (2.0, -9.0, 23) if CLAD == "air" else (2.0, -5.0, 15)

#: The thesis measures 50 +- 1.4 % over 145 nm around 1550 nm.
WAVELENGTHS_NM = (1500, 1530, 1550, 1580, 1610)


def schedule(tilt_degrees, length=LENGTH):
    """Width taper plus a constant lateral tilt.

    The offset is measured from the *middle* of the device rather than its
    input.  Only differences between consecutive sections carry physics - a
    rigid translation of the whole pair changes nothing - and centring halves
    the lateral excursion, which keeps the guides well inside the simulation
    window on the longest devices in the length sweep.
    """
    slope = float(np.tan(np.radians(tilt_degrees)))
    return {
        "w1": lambda z, L=length: W1_IN + (W1_OUT - W1_IN) * (np.asarray(z) / L),
        "x_offset": lambda z, s=slope, L=length: s * (np.asarray(z, dtype=float) - L / 2),
    }


def tilted_schedule(theta_of_eta, length=LENGTH, samples=400):
    """Width taper plus a *position-dependent* tilt theta(eta).

    The offset is the running integral of tan(theta) along the device, since
    theta is a slope rather than a displacement.
    """
    eta = np.linspace(0.0, 1.0, samples)
    slope = np.tan(np.radians(np.asarray(theta_of_eta(eta), dtype=float)))
    offset = np.concatenate(([0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(eta)))) * length
    offset = offset - np.interp(0.5, eta, offset)  # centre it, as in schedule()

    def offset_of_z(z, _eta=eta, _off=offset, L=length):
        return np.interp(np.asarray(z, dtype=float) / L, _eta, _off)

    return {
        "w1": lambda z, L=length: W1_IN + (W1_OUT - W1_IN) * (np.asarray(z) / L),
        "x_offset": offset_of_z,
    }


# ------------------------------------------------------------------- analysis


def splitting(path, length=LENGTH, launch=0):
    """Output guide powers and inter-supermode crosstalk.

    At the symmetric output the two supermodes are the even and odd
    combinations of identical guides, so the guide powers are
    ``|a0 +- a1|^2 / 2``.  Ending in a *pure* supermode therefore splits
    exactly 50/50; every deviation is crosstalk into the other one.
    """
    n_modes = path.output_data["neff"].shape[1] // 2
    eme = EME(path, force_unitary=True)
    eme.calc_Smatrix()
    smatrix = eme.propagator._find_Smatrix_new_length(length)
    lumped = smatrix[0]
    for j in range(1, len(smatrix)):
        lumped = _redheffer_star_product(lumped, smatrix[j])
    amplitudes = np.zeros(2 * n_modes, dtype=complex)
    amplitudes[launch] = 1.0
    out = lumped @ amplitudes
    guide1 = abs(out[0] + out[1]) ** 2 / 2
    guide2 = abs(out[0] - out[1]) ** 2 / 2
    total = guide1 + guide2

    # The thesis extracts transmission and crosstalk as *powers* and forms
    # t+- = (sqrt(a1) +- sqrt(a2))^2 / 2, which discards the relative phase and
    # therefore reports the worst case: the two supermodes assumed to interfere
    # fully constructively at the output.  Keeping the phase (above) is more
    # accurate, but the published 50 +- 1.4 % is in the phase-blind convention,
    # so report both and compare like with like.
    a1, a2 = abs(out[0]) ** 2, abs(out[1]) ** 2
    envelope = (np.sqrt(a1) + np.sqrt(a2)) ** 2 / 2
    return {
        "crosstalk": float(abs(out[1]) ** 2),
        "through": float(abs(out[0]) ** 2),
        "split_percent": float(100 * guide1 / total) if total else float("nan"),
        "split_envelope_percent": (
            float(100 * envelope / (a1 + a2)) if (a1 + a2) else float("nan")
        ),
        "excess_loss_db": float(-10 * np.log10(max(total, 1e-12))),
    }


def build_direct(extractor, functions, length=LENGTH, resolution=70):
    path = DirectParametricPath(
        extractor, functions, total_length=length, resolution=resolution
    )
    path._verbose = False
    path.calc_output_data()
    return path


def coupling_map(backend, w1_values, tilt_values, dz=0.25e-6):
    """Inter-supermode coupling vs position and tilt - the thesis' Fig. 3.27(b).

    At each position, compare the modes there with the modes one short step
    ``dz`` further along a path carrying that tilt.  The off-diagonal overlap
    between the two guided supermodes is the coupling that step produces; the
    tilt at which it passes through zero is the rapid adiabatic condition.

    :returns: ``(map, signed)`` - magnitude and signed coupling,
        shape ``(len(tilt_values), len(w1_values))``.
    """
    magnitude = np.full((len(tilt_values), len(w1_values)), np.nan)
    signed = np.full_like(magnitude, np.nan)
    dw1 = (W1_OUT - W1_IN) / LENGTH * dz

    for j, w1 in enumerate(w1_values):
        here = backend.solve((round(float(w1), 12), 0.0))
        for i, tilt in enumerate(tilt_values):
            shift = float(np.tan(np.radians(tilt)) * dz)
            there = backend.solve(
                (round(float(w1 + dw1), 12), round(shift, 12))
            )
            x, y, neff, te, E, H = assemble([here, there], backend.num_modes, 2)
            O = overlap_matrix(E[0], H[1], x, y, 2)
            magnitude[i, j] = abs(O[0, 1])
            signed[i, j] = np.real(O[0, 1])
    return magnitude, signed


def rac_trajectory(w1_values, tilt_values, signed, magnitude=None):
    """Trace the tilt at which the inter-supermode coupling vanishes.

    ``Re(O_01)`` crosses zero more than once over the first half of region III,
    and only one of those roots is the rapid adiabatic condition.  Picking by
    scan order takes the *positive* branch there, which is the worst tilt in the
    whole scan; picking by ``argmin|O_01|`` is no better, because where the pair
    is strongly asymmetric the coupling is weak at every tilt and the minimum
    lands on the edge of the scanned range.

    So follow the branch instead, which is what the thesis does - "we directly
    solve for the black solid-line theta_RAC(eta) using Gauss-Newton
    continuation".  Seed at the symmetric end, where the pair is closest in
    ``beta``, the coupling is strongest and the root is unique, and walk back
    towards the input taking the root nearest the previous one.

    :returns: ``theta_RAC`` in degrees, ``nan`` where no root exists.
    """
    n = len(w1_values)
    roots_at = []
    for j in range(n):
        column = signed[:, j]
        good = np.flatnonzero(np.isfinite(column))
        if good.size < 2:
            roots_at.append([])
            continue
        values = column[good]
        tilts = np.asarray(tilt_values)[good]
        crossings = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
        roots_at.append([
            float(tilts[k] - values[k] * (tilts[k + 1] - tilts[k])
                  / (values[k + 1] - values[k]))
            for k in crossings
        ])

    theta = np.full(n, np.nan)
    # Seed from the symmetric end (last index) and continue backwards.  If that
    # end has several roots, take the one at the strongest coupling, which is
    # the branch that dominates the device.
    seeded = None
    for j in range(n - 1, -1, -1):
        if not roots_at[j]:
            continue
        if seeded is None:
            if magnitude is None or len(roots_at[j]) == 1:
                theta[j] = roots_at[j][0]
            else:
                anchor = tilt_values[int(np.argmin(magnitude[:, j]))]
                theta[j] = min(roots_at[j], key=lambda r: abs(r - anchor))
            seeded = theta[j]
        else:
            theta[j] = min(roots_at[j], key=lambda r: abs(r - seeded))
            seeded = theta[j]
    return theta


# -------------------------------------------------------------------- figures


def figure_coupling_map(w1_values, tilt_values, magnitude, theta_rac):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))
    fig.subplots_adjust(wspace=0.32)

    eta = (W1_IN - np.asarray(w1_values)) / (W1_IN - W1_OUT)
    mesh = ax1.pcolormesh(
        eta, tilt_values, np.log10(np.maximum(magnitude, 1e-6)),
        cmap="magma", shading="auto",
    )
    ax1.plot(eta, theta_rac, "w-", lw=2.0, label=r"$\theta_{RAC}(\eta)$")
    ax1.set_xlabel(r"position along region III, $\eta$")
    ax1.set_ylabel("tilt (degrees)")
    ax1.set_title("inter-supermode coupling per step", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    bar = fig.colorbar(mesh, ax=ax1)
    bar.set_label(r"$\log_{10}|O_{01}|$", fontsize=8)

    ax2.plot(eta, theta_rac, "o-", color="C3", ms=4)
    ax2.axhline(0.0, color="grey", lw=0.7, ls=":")
    ax2.set_xlabel(r"position along region III, $\eta$")
    ax2.set_ylabel(r"$\theta_{RAC}$ (degrees)")
    ax2.set_title("the rapid adiabatic condition", fontsize=9)

    save(fig, f"{TAG}_1_coupling_map.png")


def figure_tilt_scan(tilts, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.32)

    crosstalk = np.array([r["crosstalk"] for r in results])
    split = np.array([r["split_percent"] for r in results])

    ax1.semilogy(tilts, crosstalk, "o-", color="C3", ms=5)
    k = int(np.argmin(crosstalk))
    ax1.annotate(
        f"best {tilts[k]:.1f}$\\degree$\n{crosstalk[k]:.1e}",
        (tilts[k], crosstalk[k]), textcoords="offset points",
        xytext=(8, 10), fontsize=8,
    )
    ax1.set_xlabel("constant tilt (degrees)")
    ax1.set_ylabel("crosstalk into the other supermode")
    ax1.set_title(f"L = {LENGTH*1e6:.1f} um coupling region", fontsize=9)

    ax2.plot(tilts, split, "s-", color="C0", ms=5)
    ax2.axhspan(49, 51, color="C2", alpha=0.15, lw=0)
    ax2.axhline(50, color="grey", lw=0.7, ls=":")
    ax2.text(tilts[0], 51.1, "50 $\\pm$ 1 %", fontsize=7, color="C2")
    ax2.set_xlabel("constant tilt (degrees)")
    ax2.set_ylabel("splitting ratio (%)")
    ax2.set_title("tilt turns an imperfect AC into a 3 dB coupler", fontsize=9)

    save(fig, f"{TAG}_2_tilt_scan.png")


def figure_length(lengths, ac, rac, stretched):
    """Splitting and crosstalk vs length, for three ways of scaling the design.

    A RAC is a *shape* in ``(w1, x)``: the rapid adiabatic condition fixes the
    lateral excursion per unit width change, not the tilt angle.  Holding theta
    fixed while stretching the device therefore over-tilts the short ones and
    under-tilts the long ones, and the design only sits at its optimum at the
    length it was solved for.  ``stretched`` keeps the outline and rescales z,
    which is the parameterisation the design rule actually specifies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.5))
    fig.subplots_adjust(wspace=0.30)

    series = [
        (ac, "C0", "o-", "conventional AC"),
        (rac, "C1", "^-", r"RAC, $	heta$ fixed"),
        (stretched, "C3", "s-", "RAC, outline fixed"),
    ]
    for data, colour, style, label in series:
        ax1.plot(lengths * 1e6, [r["split_percent"] for r in data], style,
                 ms=4, color=colour, label=label)
        ax2.semilogy(lengths * 1e6, [r["crosstalk"] for r in data], style,
                     ms=4, color=colour, label=label)

    ax1.axhspan(49, 51, color="C2", alpha=0.15, lw=0)
    ax1.axhline(50, color="grey", lw=0.7, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("coupling-region length (um)")
    ax1.set_ylabel("splitting ratio (%)")
    ax1.set_title("splitting vs length", fontsize=9)
    ax1.legend(fontsize=7)

    ax2.axvline(LENGTH * 1e6, color="grey", lw=0.7, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("coupling-region length (um)")
    ax2.set_ylabel("crosstalk into the other supermode")
    ax2.set_title("the tilt buys length, not accuracy at every length",
                  fontsize=9)
    ax2.legend(fontsize=7)

    save(fig, f"{TAG}_3_length.png")


def figure_wavelength(wavelengths, ac, rac):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.4))
    fig.subplots_adjust(wspace=0.32)

    ax1.plot(wavelengths, [r["split_percent"] for r in ac], "o-", ms=4,
             color="C0", label="conventional AC")
    ax1.plot(wavelengths, [r["split_percent"] for r in rac], "s-", ms=4,
             color="C3", label="RAC")
    ax1.axhspan(48.6, 51.4, color="C2", alpha=0.15, lw=0)
    ax1.axhline(50, color="grey", lw=0.7, ls=":")
    ax1.text(wavelengths[0], 51.5, "measured 50 $\\pm$ 1.4 %", fontsize=7, color="C2")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("splitting ratio (%)")
    ax1.set_title(f"L = {LENGTH*1e6:.1f} um, both designs", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.plot(wavelengths, [r["excess_loss_db"] for r in ac], "o-", ms=4,
             color="C0", label="conventional AC")
    ax2.plot(wavelengths, [r["excess_loss_db"] for r in rac], "s-", ms=4,
             color="C3", label="RAC")
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("excess loss (dB)")
    ax2.set_title("mode-conversion loss only (no PML)", fontsize=9)
    ax2.legend(fontsize=8)

    save(fig, f"{TAG}_4_wavelength.png")


def figure_validation(lengths, dataset, direct):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.3))
    fig.subplots_adjust(wspace=0.32)
    ax1.plot(lengths * 1e6, direct, "o-", ms=3, label="direct EME")
    ax1.plot(lengths * 1e6, dataset, "s--", ms=3, label="dataset EME")
    ax1.set_xlabel("coupling-region length (um)")
    ax1.set_ylabel("splitting ratio (%)")
    ax1.set_title("same answer from both methods", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.semilogy(lengths * 1e6,
                 np.abs(np.asarray(dataset) - np.asarray(direct)), "d-", color="C3")
    ax2.set_xlabel("coupling-region length (um)")
    ax2.set_ylabel("|dataset - direct| (% points)")
    ax2.set_title("residual, from the parameter grid", fontsize=9)
    save(fig, f"{TAG}_5_validation.png")


# ----------------------------------------------------------------------- main


def main():
    print("Dataset-based EME - rapid adiabatic coupler (Fargas Cabanillas 3.5.3)")
    print("=" * 74)
    extractor = DataExtractor(DATASET, cache_size=16384)
    backend = extractor.backend
    print(
        f"region III: w1 {W1_IN*1e9:.0f} -> {W1_OUT*1e9:.0f} nm beside a fixed "
        f"{backend.cross_section.w2*1e9:.0f} nm guide, "
        f"{backend.cross_section.gap*1e9:.0f} nm gap, L = {LENGTH*1e6:.1f} um"
    )

    # ---- 1. the rapid adiabatic condition -------------------------------
    print("\n1. mapping the inter-supermode coupling vs position and tilt")
    w1_values = np.linspace(W1_IN, W1_OUT, 13)
    tilt_values = np.linspace(*TILT_RANGE)
    t0 = time.time()
    magnitude, signed = coupling_map(backend, w1_values, tilt_values)
    theta_rac = rac_trajectory(w1_values, tilt_values, signed, magnitude)
    finite = np.isfinite(theta_rac)
    print(
        f"   {int(finite.sum())}/{len(w1_values)} positions have a zero-coupling "
        f"tilt   [{time.time()-t0:.0f} s]"
    )
    if finite.any():
        print(
            f"   theta_RAC spans {np.nanmin(theta_rac):+.2f} to "
            f"{np.nanmax(theta_rac):+.2f} degrees"
        )
    figure_coupling_map(w1_values, tilt_values, magnitude, theta_rac)

    # ---- 2. constant-tilt scan ------------------------------------------
    print("\n2. splitting and crosstalk vs a constant tilt")
    header = f"   {'tilt':>7} {'crosstalk':>11} {'split %':>9} {'excess dB':>10}"
    print(header)
    paths, results = {}, []
    for tilt in TILTS:
        t0 = time.time()
        path = build_direct(extractor, schedule(tilt))
        paths[tilt] = path
        value = splitting(path)
        results.append(value)
        print(
            f"   {tilt:6.2f}d {value['crosstalk']:11.3e} "
            f"{value['split_percent']:9.3f} {value['excess_loss_db']:10.4f}"
            f"   [{time.time()-t0:.0f} s]",
            flush=True,
        )
    best_index = int(np.argmin([r["crosstalk"] for r in results]))
    best_tilt = TILTS[best_index]
    untilted = results[TILTS.index(0.0)]["crosstalk"]
    print(
        f"   best constant tilt: {best_tilt:.2f} degrees "
        f"(crosstalk {results[best_index]['crosstalk']:.2e} vs "
        f"{untilted:.2e} untilted)"
    )
    figure_tilt_scan(list(TILTS), results)

    # ---- 3. theta_RAC(eta) vs the best constant tilt --------------------
    print("\n3. the position-dependent trajectory")
    trajectory = None
    if finite.sum() >= 3:
        eta_nodes = (W1_IN - w1_values[finite]) / (W1_IN - W1_OUT)
        theta_nodes = theta_rac[finite]

        def profile(e, _x=eta_nodes, _y=theta_nodes):
            return np.interp(e, _x, _y)

        rac_path = build_direct(extractor, tilted_schedule(profile))
        trajectory = splitting(rac_path)
        paths["rac"] = rac_path
        print(
            f"   theta_RAC(eta): crosstalk {trajectory['crosstalk']:.3e}, "
            f"split {trajectory['split_percent']:.3f} %"
        )
    else:
        print("   not enough zero crossings to build a trajectory")

    # ---- 4. length sweep -------------------------------------------------
    print("\n4. splitting vs length, conventional AC vs RAC")
    # Capped at 40 um: the angle-fixed curve needs a genuinely new outline at
    # every length, and tan(theta) * L / 2 has to stay well inside the window
    # half-width.  At the air-clad optimum that already reaches 1.6 um at 40 um.
    lengths = np.array([4e-6, 8e-6, 11.5e-6, 16e-6, 23e-6, 40e-6])
    # The untilted device has x_offset == 0 at every section, so its parameter
    # points do not depend on the length and one path serves every point.  A
    # tilted one does: holding *theta* fixed while stretching L means a
    # different lateral trajectory, so each length needs its own path.
    # Untilted, x_offset is identically zero, so one path serves every length.
    ac = [splitting(paths[0.0], length=L) for L in lengths]
    # Outline fixed: the same (w1, x) trajectory, rescaled in z.  This is what
    # the rapid adiabatic condition specifies, and the effective tilt falls as
    # 1/L.  Also one path.
    stretched = [splitting(paths[best_tilt], length=L) for L in lengths]
    # Angle fixed: a genuinely different outline at every length.
    rac = []
    for L in lengths:
        path = paths[best_tilt] if L == LENGTH else build_direct(
            extractor, schedule(best_tilt, length=L), length=L
        )
        rac.append(splitting(path, length=L))
    print(
        f"   {'length':>8} {'AC':>18} {'RAC theta fixed':>18} "
        f"{'RAC outline fixed':>18}"
    )
    print(f"   {'':>8} " + " ".join([f"{'split% xtalk':>18}"] * 3))
    for L, a, r, t in zip(lengths, ac, rac, stretched):
        print(
            f"   {L*1e6:7.1f}u "
            f"{a['split_percent']:8.3f} {a['crosstalk']:9.2e} "
            f"{r['split_percent']:8.3f} {r['crosstalk']:9.2e} "
            f"{t['split_percent']:8.3f} {t['crosstalk']:9.2e}"
        )
    figure_length(lengths, ac, rac, stretched)

    # ---- 5. wavelength ---------------------------------------------------
    print("\n5. splitting vs wavelength")
    wl_ac, wl_rac = [], []
    for nm_value in WAVELENGTHS_NM:
        folder = os.path.join(
            ROOT, "datasets", f"Si_rac_region3_220nm{SUFFIX}_{nm_value}"
        )
        other = DataExtractor(folder, cache_size=16384)
        t0 = time.time()
        a = splitting(build_direct(other, schedule(0.0)))
        r = splitting(build_direct(other, schedule(best_tilt)))
        wl_ac.append(a)
        wl_rac.append(r)
        print(
            f"   {nm_value} nm: AC {a['split_percent']:7.3f} % / "
            f"RAC {r['split_percent']:7.3f} %   "
            f"(crosstalk {a['crosstalk']:.1e} -> {r['crosstalk']:.1e})"
            f"   [{time.time()-t0:.0f} s]",
            flush=True,
        )
    figure_wavelength(list(WAVELENGTHS_NM), wl_ac, wl_rac)

    # ---- 6. DBEME vs direct EME -----------------------------------------
    print("\n6. dataset EME vs direct EME, at the best tilt")
    updater = DataUpdater(DATASET)
    t0 = time.time()
    ds_path = ParametricPath(updater, schedule(best_tilt), total_length=LENGTH)
    ds_path.calc_output_data()
    build_seconds = time.time() - t0

    checks = [
        check_reciprocity(ds_path),
        check_reflection_symmetry(ds_path),
        check_power_conservation(ds_path),
        check_branch_tracking(ds_path)[0],
    ]
    print("\n   sanity gate")
    for check in checks:
        print("   " + str(check))

    sweep = np.array([4e-6, 8e-6, 11.5e-6, 23e-6, 40e-6])
    ds_curve = [splitting(ds_path, length=L)["split_percent"] for L in sweep]
    dir_curve = [splitting(paths[best_tilt], length=L)["split_percent"] for L in sweep]
    print(f"\n   {'length':>8} {'dataset':>9} {'direct':>9} {'diff':>9}")
    for L, a, b in zip(sweep, ds_curve, dir_curve):
        print(f"   {L*1e6:7.1f}u {a:9.4f} {b:9.4f} {abs(a-b):9.2e}")
    max_difference = float(np.max(np.abs(np.array(ds_curve) - np.array(dir_curve))))
    print(f"   max |difference| = {max_difference:.2e} percentage points")
    print(f"   dataset built in {build_seconds:.0f} s, {len(updater.neff)} points")
    figure_validation(sweep, ds_curve, dir_curve)
    updater.save_data()

    payload = {
        "device": {
            "w1_nm": [W1_IN * 1e9, W1_OUT * 1e9],
            "w2_nm": backend.cross_section.w2 * 1e9,
            "gap_nm": backend.cross_section.gap * 1e9,
            "length_um": LENGTH * 1e6,
        },
        "coupling_map": {
            "w1_nm": (w1_values * 1e9).tolist(),
            "tilt_deg": tilt_values.tolist(),
            "theta_rac_deg": [
                None if not np.isfinite(v) else float(v) for v in theta_rac
            ],
        },
        "tilt_scan": {
            "tilts": list(TILTS),
            "results": results,
            "best_tilt": best_tilt,
            "untilted_crosstalk": float(untilted),
        },
        "trajectory": trajectory,
        "length_sweep": {
            "lengths_um": (lengths * 1e6).tolist(),
            "ac": ac,
            "rac_angle_fixed": rac,
            "rac_outline_fixed": stretched,
        },
        "wavelength_sweep": {
            "wavelength_nm": list(WAVELENGTHS_NM),
            "ac": wl_ac,
            "rac": wl_rac,
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
        "validation": {
            "lengths_um": (sweep * 1e6).tolist(),
            "dataset": ds_curve,
            "direct": dir_curve,
            "max_difference": max_difference,
            "dataset_build_seconds": build_seconds,
            "dataset_points": len(updater.neff),
        },
    }
    report_dir = os.path.join(ROOT, "reports", "output")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{TAG}_results.json"), "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote reports/output/{TAG}_results.json")

    import shutil

    from _plotting import OUTPUT_DIR

    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith(TAG) and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote reports/output/{name}")
    return payload


if __name__ == "__main__":
    main()
