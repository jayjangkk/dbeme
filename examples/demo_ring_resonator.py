"""Demo - all-pass micro-ring resonator from DBEME building blocks (`tasks/11` §3).

The ring is a *circuit* of two DBEME results: the bus-ring point coupler
(`studies/ring/build_coupler.py`, one path per gap over the ring-coupler
dataset, three wavelengths) and the bent strip's index and loss budget
(`studies/ring/ring_loss.py`).  `studies/ring/ring_model.py` turns them into
sax models and closes the loop; this script draws the results:

  1. coupler |kappa|^2 and |t|^2 vs gap, DBEME vs coupled-mode theory from
     the same dataset (§4.1);
  2. through-port spectra over one FSR for every gap;
  3. FSR, loaded / intrinsic Q and extinction vs gap, closed form vs the
     computed spectrum, with the critical-coupling gap marked;
  4. resonance shift per nm of ring width (a sensitivity, not an absolute
     position: `n_eff` is converged to ~2e-3, i.e. ~0.7 nm).

Run:  python examples/demo_ring_resonator.py   (figures -> reports/output/ring/)
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from studies.ring.cmt_check import cmt_kappa2  # noqa: E402
from studies.ring.ring_model import OUT, RingModel, load_coupler_json, load_loss_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "coupler_15[357]0.json")))
    if not paths:
        raise SystemExit("no coupler results; run studies/ring/campaign.sh first")
    table, meta = load_coupler_json(paths)
    loss = load_loss_json()
    R_um = meta["radius_um"]
    gaps = sorted(table)
    print(f"ring R = {R_um} um, gaps {gaps} nm, wavelengths {[round(r[0]*1e9) for r in table[gaps[0]]]} nm")

    # 1. coupler vs gap, with CMT from the same dataset
    k2, k2c, t2, cmt, checks = [], [], [], [], []
    for g in gaps:
        rec = min(table[g], key=lambda r: abs(r[0] - 1.55e-6))
        entry = rec[3]
        k2.append(entry["kappa2"]); k2c.append(entry.get("kappa2_corrected", entry["kappa2"])); t2.append(entry["t2"])
        cmt.append(cmt_kappa2(entry, rec[0])[0])
        checks.append(entry["checks"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(gaps, k2, "o:", label="DBEME |κ|² (raw)")
    ax.semilogy(gaps, k2c, "o-", label="DBEME |κ|² / √(ring survival)")
    ax.semilogy(gaps, cmt, "s--", label="CMT from the same n_eff")
    ax.set_xlabel("bus-ring gap g₀ (nm)"); ax.set_ylabel("cross-coupled power")
    ax.set_title(f"point coupler, R = {R_um:g} µm, 1550 nm", fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, "ring_1_coupler_vs_gap.png")

    # 2-3. rings
    wl = np.linspace(1.5440, 1.5560, 24001)
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for g in gaps:
        rm = RingModel(table[g], loss, R_um)
        T = np.abs(rm.transmission(wl)) ** 2
        ax.plot(wl * 1e3, 10 * np.log10(np.maximum(T, 1e-12)), lw=0.9, label=f"g₀ = {g:.0f} nm")
        cf = rm.closed_form()
        sm = rm.spectrum_metrics(wl)
        rows.append({"g0_nm": g, "kappa2": cf["kappa2"], "r": cf["r"], "a": cf["a"], "n_g": cf["n_g"],
                     "FSR_nm_closed": cf["FSR_m"] * 1e9, "FSR_nm_spectrum": sm.get("FSR_nm"),
                     "Q_loaded_closed": cf["Q_loaded"], "Q_loaded_spectrum": sm.get("Q_loaded"),
                     "Q_intrinsic": cf["Q_intrinsic"], "Q_coupling": cf["Q_coupling"],
                     "extinction_dB_closed": cf["extinction_dB"], "extinction_dB_spectrum": sm.get("extinction_dB"),
                     "r_critical": cf["r_critical"], "loss_db_per_cm": rm.loss_db_per_cm,
                     "arc_length_um": rm.arc_length * 1e6, "coupler_chord_um": rm.chord * 1e6})
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("through (dB)"); ax.set_ylim(-30, 1)
    ax.set_title(f"all-pass ring, R = {R_um:g} µm, 500 × 220 nm, loss {rows[0]['loss_db_per_cm']:.1f} dB/cm", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    save(fig, "ring_2_spectra.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    fig.subplots_adjust(wspace=0.35)
    g = [r["g0_nm"] for r in rows]
    axes[0].semilogy(g, [r["Q_loaded_closed"] for r in rows], "o-", label="Q loaded (closed form)")
    axes[0].semilogy(g, [r["Q_loaded_spectrum"] or np.nan for r in rows], "x", label="Q loaded (spectrum)")
    axes[0].semilogy(g, [r["Q_coupling"] for r in rows], "s--", label="Q coupling")
    axes[0].axhline(rows[0]["Q_intrinsic"], color="k", lw=0.8, ls=":", label="Q intrinsic")
    axes[0].set_xlabel("g₀ (nm)"); axes[0].set_ylabel("Q"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].plot(g, [r["extinction_dB_closed"] for r in rows], "o-", label="closed form")
    axes[1].plot(g, [r["extinction_dB_spectrum"] or np.nan for r in rows], "x", label="spectrum")
    axes[1].set_xlabel("g₀ (nm)"); axes[1].set_ylabel("extinction (dB)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    # critical coupling: where |t| = a
    rr = np.array([r["r"] for r in rows]); aa = rows[0]["a"]
    if np.any(rr < aa) and np.any(rr > aa):
        g_crit = float(np.interp(aa, rr, g))
        axes[1].axvline(g_crit, color="r", lw=0.8, ls="--"); axes[1].text(g_crit, axes[1].get_ylim()[1] * 0.9, f" critical {g_crit:.0f} nm", color="r", fontsize=7)
    else:
        g_crit = None
    axes[2].plot(g, [r["FSR_nm_closed"] for r in rows], "o-", label="λ²/(n_g L)")
    axes[2].plot(g, [r["FSR_nm_spectrum"] or np.nan for r in rows], "x", label="spectrum")
    axes[2].set_xlabel("g₀ (nm)"); axes[2].set_ylabel("FSR (nm)"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    save(fig, "ring_3_metrics_vs_gap.png")

    # 4. resonance shift per nm of width: from the straight strip dataset if present
    sensitivity = None
    strip = os.path.join(ROOT, "datasets", "Si_fulletch_220nm")
    try:
        from dbeme import DataUpdater
        du = DataUpdater(strip)
        n_g = rows[0]["n_g"]
        widths = (480e-9, 500e-9, 520e-9)
        neff = [float(np.real(du.get_neffs([(w, 0.0)])[0][0])) for w in widths]
        dn_dw = (neff[2] - neff[0]) / 40e-9
        sensitivity = {"dneff_dw_per_nm": dn_dw * 1e-9, "dlambda_dw_nm_per_nm": dn_dw * 1e-9 * 1550 / n_g,
                       "widths_nm": [w * 1e9 for w in widths], "neff": neff}
        print(f"  resonance shift {sensitivity['dlambda_dw_nm_per_nm']:.3f} nm per nm of width (n_eff slope {dn_dw*1e-9:.2e}/nm)")
    except Exception as exc:  # the strip dataset may be absent or cold
        print(f"  width sensitivity skipped: {type(exc).__name__}: {exc}")

    summary = {"radius_um": R_um, "rows": rows, "critical_gap_nm": g_crit, "width_sensitivity": sensitivity,
               "checks_per_gap": {str(r["g0_nm"]): c for r, c in zip(rows, checks)}}
    with open(os.path.join(OUT, "ring_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(f"{'g0':>5} {'|k|^2':>8} {'Q_L':>8} {'Q_L spec':>9} {'ER dB':>7} {'ER spec':>8} {'FSR nm':>7}")
    for r in rows:
        print(f"{r['g0_nm']:5.0f} {r['kappa2']:8.5f} {r['Q_loaded_closed']:8.0f} {(r['Q_loaded_spectrum'] or 0):9.0f} "
              f"{r['extinction_dB_closed']:7.2f} {(r['extinction_dB_spectrum'] or 0):8.2f} {r['FSR_nm_closed']:7.3f}")
    print("  wrote ring_summary.json")


if __name__ == "__main__":
    main()
