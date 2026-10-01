"""Figures for reports/22: every figure of Wan & Wang 2025 with its DBEME counterpart.

    python studies/edge_coupler/figures.py [which ...]

Reads the JSON/NPZ that the other scripts write in ``reports/output/edge``
and the digitised paper curves (``paper_digitised.json``); a panel whose
data do not exist yet is drawn empty with a note, so the script can run at
any stage.  Encoding throughout: colour = polarisation (TE blue, TM
orange), line style = source (paper dashed with open markers, DBEME solid
with filled markers); the two tip heights get separate panels.
"""

import glob
import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device as dv  # noqa: E402

OUT = os.path.join(dv.ROOT, "reports", "output", "edge")
FIG = os.path.join(dv.ROOT, "reports", "output")
PAPER = json.load(open(os.path.join(OUT, "paper_digitised.json")))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
TE_C = "#2a78d6"
TM_C = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
COL = {"TE": TE_C, "TM": TM_C}
BLUES = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=9)


def paper_line(ax, x, y, pol, label=None):
    x, y = np.asarray(x, float), np.array([np.nan if v is None else v for v in y], float)
    ax.plot(x, y, ls=(0, (4, 2.5)), color=COL[pol], lw=1.6, label=label)


def dbeme_line(ax, x, y, pol, label=None, marker="o"):
    ax.plot(x, y, "-", color=COL[pol], lw=2, marker=marker, ms=5, mec=SURFACE, mew=1.0, label=label)


# ------------------------------------------------------------------ Fig. 1(b)


def fig_layout():
    import importlib

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), gridspec_kw={"height_ratios": [1.2, 1]}, sharex=True)
    funcs, total = dv.path_functions()
    z = np.linspace(0, total, 3461)
    wl, wh, g = (np.asarray(funcs[k](z)) for k in ("w_low", "w_high", "gap"))
    ax = axes[0]
    zu = z * 1e6
    for sgn in (1, -1):
        inner = sgn * 0.5 * g * 1e6
        high = sgn * (0.5 * g + wh) * 1e6
        outer = sgn * (0.5 * g + wh + wl) * 1e6
        ax.fill_between(zu, inner, high, color="#e34948", lw=0, label="220 nm Si" if sgn > 0 else None)
        ax.fill_between(zu, high, outer, color="#f2a7a6", lw=0, label="150 nm Si" if sgn > 0 else None)
    zb = np.array(dv.breakpoints()) * 1e6
    for zz in zb[:-1]:
        ax.axvline(zz, color=INK2, lw=0.8, ls=":")
    for name, a, b in zip(("L1", "L2", "L3", "Lt", "Lm", "out"), np.r_[0, zb[:-1]], zb):
        ax.text(0.5 * (a + b), 1.45, name, ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_ylabel("x (µm)")
    ax.set_ylim(-1.3, 1.6)
    ax.set_title("Layout as built from the dataset's snapped path (fibre at z = 0; cf. paper Fig. 1(b), mirrored)")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 0.93))
    ax = axes[1]
    ax.plot(zu, wl * 1e9, color=AQUA, label="w_low (150 nm shelf)")
    ax.plot(zu, wh * 1e9, color=VIOLET, label="w_high (220 nm part)")
    ax.plot(zu, g * 1e9 / 2, color=INK2, label="gap / 2")
    for zz in zb[:-1]:
        ax.axvline(zz, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("z from the facet (µm)")
    ax.set_ylabel("nm")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "edge_1_layout.png")


# ------------------------------------------------------------------ Fig. 2


def fig_modes():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6))
    for row, h in enumerate((220, 150)):
        fn = os.path.join(OUT, f"fields_tip{h}_full.npz")
        if not os.path.exists(fn):
            for ax in axes[row]:
                empty(ax, "no data yet")
            continue
        d = np.load(fn)
        x, y = d["x"] * 1e6, d["y"] * 1e6
        sel = (np.abs(x) < 4.2)
        ysel = (y > -2.4) & (y < 2.0)
        xs, ys = x[sel], y[ysel]               # a refined, non-uniform grid: pcolormesh, not imshow
        n = d["n"][: x.size, : y.size]
        axes[row, 0].pcolormesh(xs, ys, n[np.ix_(sel, ysel)].T, shading="nearest",
                                cmap=LinearSegmentedColormap.from_list("n", ["#fcfcfb", "#b9b8b3", "#e34948"]))
        axes[row, 0].set_title(f"{h} nm tips: index (Si substrate below)")
        for k, pol in enumerate(("TE", "TM")):
            ax = axes[row, k + 1]
            key = f"E_{pol}"
            if key not in d.files:
                empty(ax, "mode not found")
                continue
            e = np.sqrt(np.sum(d[key].astype(float) ** 2, axis=0))
            ax.pcolormesh(xs, ys, (e / e.max())[np.ix_(sel, ysel)].T, shading="nearest",
                          cmap=BLUES, vmin=0, vmax=1)
            ax.set_title(f"{h} nm, {pol} even supermode, |E|")
        for ax in axes[row]:
            ax.grid(False)
            ax.set_ylabel("y (µm)")
    for ax in axes[1]:
        ax.set_xlabel("x (µm)")
    fig.suptitle("Facet modes at 1310 nm (cf. paper Fig. 2): double tip, g = 1.6 µm, Wtip = 130 nm",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "edge_2_modes.png")


# ------------------------------------------------------------------ Fig. 3


def fig_overlap():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    sets = (("wtip", "a", "x_Wtip_nm", "wtip_nm", "Wtip (nm)", "Fig. 3(a): g = 1.6 µm"),
            ("gap", "b", "x_g_um", "g_um", "g, edge to edge (µm)", "Fig. 3(b): Wtip = 130 nm"))
    for col, (part, panel, px, dx, xlabel, title) in enumerate(sets):
        data = load(f"phase1_{part}_full.json")
        pp = PAPER["fig3"][panel]
        for row, h in enumerate((150, 220)):
            ax = axes[row, col]
            for pol in ("TE", "TM"):
                paper_line(ax, pp[px], pp[f"{pol}{h}"], pol, label=f"paper {pol}")
                if data:
                    pts = [p for p in data["points"] if p["height"] == h and p.get(pol)]
                    xs = [p[dx] for p in pts]
                    dbeme_line(ax, xs, [p[pol]["rows"][0]["eq1"] for p in pts], pol, label=f"DBEME {pol}, eq. (1)")
                    ax.plot(xs, [p[pol]["rows"][0]["poynting"] for p in pts], ":", color=COL[pol], lw=1.4,
                            label=f"DBEME {pol}, power coupling")
            ax.set_title(f"{title}, {h} nm tips")
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("overlap with 4 µm MFD beam")
            ax.set_ylim(0, 1)
            if not data:
                empty(ax, "DBEME: not computed yet")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "edge_3_overlap.png")


# ------------------------------------------------------------------ Fig. 4


def fig_propagation(wl=1310, variant="bilayer"):
    fn = os.path.join(OUT, f"fields_{wl}_{variant}.npz")
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.3, 0.9, 1.3, 0.9])
    if not os.path.exists(fn):
        ax = fig.add_subplot(gs[:, :])
        empty(ax, "field planes not computed yet")
        save(fig, "edge_4_propagation.png")
        return
    d = np.load(fn)
    z, x = d["z"] * 1e6, d["x"] * 1e6
    y = d["y"] * 1e6
    sel = np.abs(x) < 4.0
    insets = [k for k in d.files if k.startswith("TE_") or k.startswith("TM_")]
    for r, pol in enumerate(("TE", "TM")):
        ax = fig.add_subplot(gs[2 * r, :])
        top = d[f"top_{pol}"].astype(float)
        zz, xx = np.meshgrid(z, x[sel], indexing="ij")
        ax.pcolormesh(zz, xx, np.sqrt(top[:, sel] / top.max()), cmap=BLUES, shading="gouraud", vmin=0, vmax=1)
        ax.set_ylabel("x (µm)")
        ax.set_title(f"{pol} launch: |E| on the device layer (y = {float(d['y_cut'])*1e9:.0f} nm), "
                     "fibre at z = 0 (cf. paper Fig. 4, mirrored)")
        ax.grid(False)
        for zi in (1.0, 26.0, 60.0, 86.5):
            ax.axvline(zi, color="#e34948", lw=0.8, ls="--")
        ax.set_xlim(z.min(), z.max())
        ks = sorted([k for k in insets if k.startswith(pol + "_")], key=lambda k: float(k.split("_")[1]))
        # the physical interior of the oxide-stack window (PML: 0.5 um on every edge); the
        # continuum part of the launch grows inside the stretched layers, so normalise without them
        ys = (y > -2.105) & (y < 1.9)
        for c, k in enumerate(ks[:4]):
            a = fig.add_subplot(gs[2 * r + 1, c])
            im = d[k].astype(float)[np.ix_(sel, ys)]
            a.pcolormesh(x[sel], y[ys], np.sqrt(im / im.max()).T, shading="nearest", cmap=BLUES)
            a.axhline(-0.11 - 2.0, color=INK2, lw=0.6, ls=":")      # BOX bottom = PML start
            a.set_title(f"z = {float(k.split('_')[1]):.1f} µm", fontsize=9)
            a.grid(False)
            a.tick_params(labelsize=7)
    fig.axes[-1].set_xlabel("x (µm)")
    fig.tight_layout()
    save(fig, "edge_4_propagation.png")


# ------------------------------------------------------------------ Fig. 5 / 8


def device_spectra():
    rows = {}
    for fn in sorted(glob.glob(os.path.join(OUT, "device_*.json"))):
        wl = int(os.path.basename(fn)[7:11])
        rows[wl] = json.load(open(fn))
    return rows


def summary_rows():
    s = load("summary.json")
    return {} if not s else {int(k): v for k, v in s["wavelengths"].items()}


def dbeme_raw_line(ax, x, y, pol, label=None):
    ax.plot(x, y, ":", color=COL[pol], lw=1.3, marker="o", ms=5, mfc=SURFACE, mec=COL[pol], label=label)


def fig_spectra():
    rows = summary_rows()
    p5 = PAPER["fig5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, variant, h in ((axes[0], "bilayer", 150), (axes[1], "conventional", 220)):
        wls = sorted(w for w in rows if variant in rows[w])
        for pol in ("TE", "TM"):
            paper_line(ax, p5["x_nm"], p5[f"{pol}{h}"], pol, label=f"paper 3D-FDTD {pol}")
            if wls:
                dbeme_raw_line(ax, wls, [rows[w][variant][pol]["raw_dB"] for w in wls], pol,
                               label=f"DBEME {pol}, 10 nm lattice as built")
                fin = [rows[w][variant][pol]["final_dB"] for w in wls]
                con = [rows[w][variant][pol]["conservative_dB"] for w in wls]
                dbeme_line(ax, wls, fin, pol, label=f"DBEME {pol}, best estimate")
                for w, a, b in zip(wls, fin, con):
                    if b is not None:
                        ax.plot([w, w], [a, b], color=COL[pol], lw=1.2, alpha=0.55)
                        ax.plot([w], [b], "_", color=COL[pol], ms=10, mew=1.5, alpha=0.8)
        ax.set_title(f"{'Bilayer design, 150 nm tips' if h == 150 else 'Conventional, 220 nm tips'}")
        ax.set_xlabel("wavelength (nm)")
        ax.set_xlim(1255, 1365)
        if not rows:
            empty(ax, "DBEME: no wavelength done yet")
    axes[0].set_ylabel("coupling loss (dB / facet)")
    axes[0].set_ylim(0, 7.0)
    axes[0].legend(loc="upper left", fontsize=7.5, ncol=1)
    fig.suptitle("Simulated coupling loss, 4 µm MFD fibre (cf. paper Fig. 5); bars run from the best estimate "
                 "to the conservative bound", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "edge_5_loss_vs_wavelength.png")
    fig_measured(rows)


def fig_measured(rows):
    p5 = PAPER["fig5"]
    p8 = PAPER["fig8"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    wls = sorted(w for w in rows if "bilayer" in rows[w])
    for pol in ("TE", "TM"):
        x = np.asarray(p8["x_nm"], float)
        lo = np.array([np.nan if v is None else v for v in p8[f"{pol}_meas_min"]], float)
        hi = np.array([np.nan if v is None else v for v in p8[f"{pol}_meas_max"]], float)
        ax.fill_between(x, lo, hi, color=COL[pol], alpha=0.16, lw=0, label=f"measured {pol} (envelope)")
        ax.plot(x, [np.nan if v is None else v for v in p8[f"{pol}_meas_median"]], color=COL[pol], lw=0.9,
                alpha=0.7)
        paper_line(ax, p5["x_nm"], p5[f"{pol}150"], pol, label=f"paper 3D-FDTD {pol}")
        if wls:
            fin = [rows[w]["bilayer"][pol]["final_dB"] for w in wls]
            con = [rows[w]["bilayer"][pol]["conservative_dB"] for w in wls]
            dbeme_line(ax, wls, fin, pol, label=f"DBEME {pol}, best estimate")
            for w, a, b in zip(wls, fin, con):
                if b is not None:
                    ax.plot([w, w], [a, b], color=COL[pol], lw=1.2, alpha=0.55)
                    ax.plot([w], [b], "_", color=COL[pol], ms=10, mew=1.5, alpha=0.8)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("coupling loss (dB / facet)")
    ax.set_ylim(0, 2.8)
    ax.set_title("Bilayer design: DBEME against the measured chip (cf. paper Fig. 8)")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, "edge_6_measured.png")


# ------------------------------------------------------------------ budget


def fig_budget(wl=1310):
    d = load(f"device_{wl}.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    if not d:
        for ax in axes:
            empty(ax, "not computed yet")
        save(fig, "edge_7_budget.png")
        return
    zb = np.array(dv.breakpoints()) * 1e6
    facet = load("phase1_design_full.json")
    eta = {}
    if facet and wl == 1310:
        for p in facet["points"]:
            h = p["height"]
            for pol in ("TE", "TM"):
                r0 = [q for q in p[pol]["rows"] if abs(q["center"][1]) < 1e-12][0]
                eta[(150 if h == 150 else 220, pol)] = r0["poynting"]
    for ax, variant in zip(axes, ("bilayer", "conventional")):
        if variant not in d:
            empty(ax, "not computed yet")
            continue
        h = 150 if variant == "bilayer" else 220
        for pol in ("TE", "TM"):
            r = d[variant][pol]
            z = np.asarray(r["z_um"])
            own = np.asarray(r["guided_TE" if pol == "TE" else "guided_TM"])
            other = np.asarray(r["guided_TM" if pol == "TE" else "guided_TE"])
            # |a|^2 is power only where the mode is confined; near the tip the lossy basis is
            # non-orthogonal, so the first few um are drawn thin and the facet's physical power
            # coupling is marked instead
            near = z < 5.0
            ax.plot(z[near], 10 * np.log10(np.maximum(own[near], 1e-6)), color=COL[pol], lw=0.9, alpha=0.6)
            ax.plot(z[~near], 10 * np.log10(np.maximum(own[~near], 1e-6)), color=COL[pol],
                    label=f"{pol} launch: guided {pol}")
            ax.plot(z, 10 * np.log10(np.maximum(other, 1e-6)), color=COL[pol], lw=1.2, ls=":",
                    label=f"{pol} launch: guided {'TM' if pol == 'TE' else 'TE'}")
            if (h, pol) in eta:
                ax.plot([0.0], [10 * np.log10(eta[(h, pol)])], "D", color=COL[pol], mec=INK, ms=6,
                        label=f"{pol}: facet power coupling (full stack)")
            fs = load(f"fullstack_{wl}.json")
            if fs and variant in fs and "guided_tip" in fs[variant][pol]:
                zt = np.asarray(fs[variant][pol]["guided_tip_z_um"])
                gt = np.asarray(fs[variant][pol]["guided_tip"])
                ax.plot(zt, 10 * np.log10(np.maximum(gt, 1e-6)), color=COL[pol], lw=2.4, ls=(0, (1, 1)),
                        label=f"{pol}: tip on the full stack")
        for zz in zb[:-1]:
            ax.axvline(zz, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{variant}: guided power along z, {wl} nm")
        ax.set_xlabel("z from the facet (µm)")
        ax.set_ylim(-6, 0.3)
    axes[0].set_ylabel("guided power (dB of the fibre power)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "edge_7_budget.png")


def fig_sections(wl=1310):
    d = load(f"device_{wl}.json")
    names = ("L1", "L2", "L3", "Lt", "Lm", "out")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, name in zip(axes.flat, names):
        seg = (d or {}).get("bilayer", {}).get("segments", {}).get(name)
        if not seg:
            empty(ax, "not computed yet")
            ax.set_title(name)
            continue
        L0 = dv.WAN2025_DESIGN["L_out" if name == "out" else name] * 1e6
        for pol in ("TE", "TM"):
            r = seg.get(pol)
            if not r:
                continue
            L = np.asarray(seg["scales"]) * L0
            ax.plot(L, 10 * np.log10(np.maximum(r["scan"], 1e-6)), color=COL[pol], label=pol)
            ax.plot([L0], [10 * np.log10(max(r["T_design"], 1e-6))], "o", color=COL[pol], mec=SURFACE)
        ax.axvline(L0, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"{name} alone")
        ax.set_xlabel(f"{name} length (µm); dotted = Table 1")
        ax.set_ylim(-3, 0.1)
    axes[0, 0].set_ylabel("transmission (dB)")
    axes[1, 0].set_ylabel("transmission (dB)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"Phase 2: each section alone, fundamental in -> fundamental out, length scanned at zero mode solves ({wl} nm)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "edge_9_sections.png")


def fig_neff(wl=1310):
    """Re n_eff of the stored modes along the path, launched branches highlighted."""
    import analyse as an

    du = dv.open_dataset(wl, substrate=False, cache_size=1)
    du._is_testmode = True
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    n_ox = float(du.get_cladding_index())
    for ax, variant in zip(axes, ("bilayer", "conventional")):
        path, _ = dv.build_path(du, variant)
        od = path.calc_output_data()
        cas = an.Cascade(path, od)
        N = cas.N
        z = cas.z[: len(od["EME_path"])] * 1e6
        neff = np.asarray(od["neff"])[:, :N]
        te = np.real(od["TE_pol"][:, :N]).astype(float)
        phys = np.array([dv.physical(od, k) for k in range(len(od["EME_path"]))])
        for j in range(N):
            m = phys[:, j]
            if not m.any():
                continue
            ax.plot(z[m], np.real(neff[m, j]), ".", ms=1.6,
                    color=TE_C if np.median(te[m, j]) >= 0.5 else TM_C, alpha=0.35)
        for pol in ("TE", "TM"):
            b = an.fundamental_branch(od, 0, pol, n_ox, N)
            ax.plot(z, np.real(neff[:, b]), color=COL[pol], lw=2, label=f"launched {pol} branch")
        ax.axhline(n_ox, color=INK2, lw=0.8, ls=":")
        for zz in np.array(dv.breakpoints()[:-1]) * 1e6:
            ax.axvline(zz, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{variant}: n_eff of the physical modes, {wl} nm")
        ax.set_xlabel("z from the facet (µm)")
    axes[0].set_ylabel("Re n_eff")
    axes[0].text(1, n_ox + 0.02, "SiO₂", color=INK2, fontsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "edge_10_neff.png")


def fig_stretch():
    """stretch_scan.py: the join-to-L3 stretch's parts with every dz scaled."""
    sc = load("stretch_scan.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    styles = {"1260": (0, (4, 2.5)), "1310": "-", "1360": (0, (1, 1.5))}
    titles = {"L1_tail": "rest of L1 (13.8 → 25 µm)", "L2": "L2, height converter", "L3": "L3, converging arms"}
    for ax, name in zip(axes, ("L1_tail", "L2", "L3")):
        if not sc:
            empty(ax, "not run yet")
            continue
        for wl, ls in styles.items():
            if wl not in sc:
                continue
            row = sc[wl]["bilayer"][name]
            for pol in ("TE", "TM"):
                ax.plot(row["scales"], row[pol]["loss_dB"], color=COL[pol], ls=ls,
                        lw=2 if wl == "1310" else 1.1, label=f"{pol}, {wl} nm")
        ax.axvspan(1, 3, color=GRID, alpha=0.6, lw=0)
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 3, 8])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("every dz × (1 = Table 1)")
        ax.set_title(titles[name])
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("fundamental → fundamental loss (dB)")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.suptitle("Bilayer: loss of each part with its sections' lengths scaled, zero mode solves "
                 "(shaded: 1-3×, where the length-dependent part is read; beyond ~3× the steps dephase)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "edge_11_stretch.png")


# ------------------------------------------------------------------ Phase 4


def fig_optimisation():
    lo1 = load("opt_lengths_1310.json")
    g1 = load("opt_lengths_global_1310.json")
    gb = load("opt_lengths_global_1260_1310_1360.json")
    fa = load("opt_facet.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    ax = axes[0, 0]
    for g, c, name in ((g1, TE_C, "1310 nm"), (gb, TM_C, "band: 1260/1310/1360 nm")):
        if not g:
            continue
        gens = [h["gen"] for h in g["history"]]
        ax.plot(gens, [h["best_J"] for h in g["history"]], color=c, label=f"CMA-ES best, {name}")
        ax.axhline(g["start"]["J"], color=c, lw=0.9, ls=(0, (4, 2.5)))
        ax.plot([gens[-1]], [g["polished"]["J"]], "o", color=c, mec=SURFACE, ms=7)
    if lo1:
        ax.axhline(lo1["optimum"]["J"], color=AQUA, lw=1.4, label="L-BFGS-B from Table 1, 1310 nm")
    ax.set_xlabel("CMA-ES generation (12 candidates each)")
    ax.set_ylabel("objective: max(TE, TM) + 0.5 PDL (dB)")
    ax.set_title("4.1 lengths: convergence (dashed = Table 1; dot = after polish)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    if lo1:
        cols = (TE_C, TM_C, AQUA, VIOLET, "#eda100", INK2)
        for (k, rows), c in zip(lo1["scans_about_design"].items(), cols):
            L0 = dv.WAN2025_DESIGN[k]
            x = [r["L_um"] * 1e-6 / L0 for r in rows]
            worst = [max(p["TE"], p["TM"]) for r in rows for p in r["per"]]
            ax.plot(x, worst, color=c, label=k)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 3])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("length / Table 1 (one at a time, others at Table 1)")
        ax.set_ylabel("worse of TE/TM loss, 1310 nm (dB)")
        ax.set_title("Length scans: 41 cascades each, zero mode solves")
        ax.legend(fontsize=8, ncol=3)
    else:
        empty(ax, "not run yet")
    ax = axes[1, 0]
    if gb:
        wls = gb["wavelengths_nm"]
        for pol in ("TE", "TM"):
            paper_line(ax, PAPER["fig5"]["x_nm"], PAPER["fig5"][f"{pol}150"], pol, label=f"paper FDTD {pol}")
            dbeme_raw_line(ax, wls, [p[pol] for p in gb["start"]["per"]], pol, label=f"Table 1 lengths {pol}")
            dbeme_line(ax, wls, [p[pol] for p in gb["polished"]["per"]], pol, label=f"band-optimised {pol}")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("coupling loss, 10 nm lattice (dB)")
        ax.set_ylim(0, 2.8)
        ax.set_title("Before / after (10 nm lattice values; the lattice terms are common to both)")
        ax.legend(fontsize=7.5, ncol=3, loc="lower center")
    else:
        empty(ax, "not run yet")
    ax = axes[1, 1]
    if fa:
        h = fa["history"]
        sc = ax.scatter([r["Wtip_nm"] for r in h], [r["g_um"] for r in h], c=[r["worst"] for r in h],
                        cmap=BLUES, s=40, edgecolors=INK2, linewidths=0.4)
        fig.colorbar(sc, ax=ax, label="worse of TE/TM power coupling")
        b = fa.get("best") or max(h, key=lambda r: r["worst"])
        ax.plot(b["Wtip_nm"], b["g_um"], "*", color=TM_C, ms=14, mec=INK, label="best")
        ax.plot(130, 1.6, "s", color=SURFACE, mec=INK, ms=7, label="paper")
        ax.legend(loc="lower right")
        ax.set_xlabel("Wtip (nm)")
        ax.set_ylabel("g (µm)")
        ax.set_title("4.2 CMA-ES on the facet")
    else:
        empty(ax, "4.2 not run yet")
    fig.tight_layout()
    save(fig, "edge_8_optimisation.png")


FIGS = {"layout": fig_layout, "modes": fig_modes, "overlap": fig_overlap, "propagation": fig_propagation,
        "spectra": fig_spectra, "budget": fig_budget, "sections": fig_sections, "neff": fig_neff,
        "optimisation": fig_optimisation, "stretch": fig_stretch}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
