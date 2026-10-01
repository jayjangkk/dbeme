"""Collect every number the report quotes into reports/output/edge/summary.json."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device as dv  # noqa: E402

OUT = os.path.join(dv.ROOT, "reports", "output", "edge")
paper = json.load(open(os.path.join(OUT, "paper_digitised.json")))


def paper_at(series, wl, key="fig5"):
    x = np.array(paper[key]["x_nm"], float)
    v = np.array([np.nan if a is None else a for a in paper[key][series]], float)
    ok = ~np.isnan(v)
    return float(np.interp(wl, x[ok], v[ok]))


def guided_at(q, z_um):
    z = np.array(q["z_um"])
    g = np.array(q["guided_TE"] if q["pol_key"] == "TE" else q["guided_TM"])
    return float(g[np.argmin(np.abs(z - z_um))])


summary = {"wavelengths": {}, "notes": {
    "raw_dB": "oxide-stack path as built (oxide facet launch): 10 nm width / 20 nm gap lattice, first-order leakage-corrected, fibre at its best vertical offset",
    "full_dB": "the same device with the facet launch and the tip (facet to the 200 nm join) on the full stack, joined to the oxide path by a cross-stack interface; leakage exact to the join, first order after it",
    "tip_lattice_dB": "(tip walked at 10/5/2.5 nm on the oxide _tip dataset, extrapolated to 0) minus (10 nm): the tip's lattice term",
    "in_range_dB": "loss of the launched polarisation's guided power from the tip join (13.8 um) to the end of L3 (68 um) on the dataset path, per wavelength: the converging arms' discretisation (length-independent from 2.3 to 115 um in L3; grows linearly with merged step size; persists with exact geometry values at the same sections)",
    "corners_dB": "at 1310 nm: the main path minus its corner-free twin (the walked corner points of diagonal steps, nearly all in the height converter); confirmed by the exact-value direct EME over L2",
    "adiabatic_dB": "length-dependent part of the join-to-L3 stretch at its Table 1 length, per part (L1 tail, L2, L3; stretch_scan.py: dz scaled at zero mode solves, loss at 1x minus the minimum over 1-3x)",
    "lower_dB": "full_dB + tip_lattice_dB - in_range_dB: the whole join-to-L3 stretch taken as lattice",
    "final_dB": "BEST ESTIMATE: lower_dB + adiabatic_dB - the join-to-L3 stretch's loss is lattice except its length-dependent part",
    "conservative_dB": "UPPER BOUND: full_dB + tip_lattice_dB - corners_dB - only the lattice error measured directly removed",
    "staircase_offset_dB": "kept for the record: main path minus the linear m->0 extrapolation of 1x/2x/3x merged-step staircases; NOT used - the merge factor is not the step size (119/67/51 steps), and for bilayer TE it removes more loss than the stretch contains",
    "uncertainty": "column cap off (the default since 2026-09-30); opting in raises the losses by 0.00-0.12 dB over 1260-1360 nm; first-order leakage after the join; the tip term comes from the oxide tip dataset",
}}
ZJ, Z68 = 13.825, 68.0
_scan_fn = os.path.join(OUT, "stretch_scan.json")
scan = json.load(open(_scan_fn)) if os.path.exists(_scan_fn) else {}


def adiabatic(wl, v, pol):
    """Length-dependent part of the join-to-L3 stretch at its Table 1 length,
    part by part (the L1 tail, L2, L3): loss at 1x minus the minimum over
    1-3x.  Only a fall below the 1x value counts; the rise that follows is the
    staircase's steps dephasing."""
    if str(wl) not in scan or v not in scan[str(wl)]:
        return None
    a = 0.0
    for name in ("L1_tail", "L2", "L3"):
        row = scan[str(wl)][v][name]
        s, L = np.array(row["scales"]), np.array(row[pol]["loss_dB"])
        m = (s >= 1 - 1e-9) & (s <= 3 + 1e-9)
        a += max(0.0, float(L[np.argmin(abs(s - 1))] - L[m].min()))
    return a


_stair_fn = os.path.join(OUT, "staircase_1310.json")
stair = json.load(open(_stair_fn)) if os.path.exists(_stair_fn) else {}
offset = {v: {p: stair[v][p]["L_m1_dB"] - stair[v][p]["L0_linear_dB"] for p in ("TE", "TM")}
          for v in stair if "L0_linear_dB" in stair[v].get("TE", {})}
offset_quad = {v: {p: stair[v][p]["L_m1_dB"] - stair[v][p].get("L0_quadratic_dB", stair[v][p]["L0_linear_dB"])
                   for p in ("TE", "TM")} for v in offset}
corners = {v: {p: max(stair[v][p].get("corners_dB", 0.0), 0.0) for p in ("TE", "TM")} for v in stair}
for fn in sorted(glob.glob(os.path.join(OUT, "device_*.json"))):
    wl = int(os.path.basename(fn)[7:11])
    d = json.load(open(fn))
    _full_fn = os.path.join(OUT, f"fullstack_{wl}.json")
    full = json.load(open(_full_fn)) if os.path.exists(_full_fn) else {}
    row = {}
    for v in ("bilayer", "conventional"):
        if v not in d:
            continue
        r = d[v]
        vr = {"leakage_factor_dB": {p: -10 * np.log10(r["leakage_factor"][p]) for p in ("TE", "TM")},
              "sections": r["sections"], "ports_neff": r["output_neff"]}
        for pol in ("TE", "TM"):
            q = dict(r[pol], pol_key=pol)
            g45, g68, g76 = guided_at(q, 45.0), guided_at(q, 68.0), guided_at(q, 76.0)
            raw = r[pol]["loss_dB_corrected"]
            tip = r["tip"]["smooth"][pol]["loss_dB_corrected"] if r.get("tip") and "smooth" in r["tip"] else raw
            tip10 = r["tip"]["10nm"][pol]["loss_dB_corrected"] if r.get("tip") and "10nm" in r["tip"] else raw
            tip_lat = tip - tip10
            f = full.get(v, {}).get(pol)
            base = f["loss_dB"] if f else raw
            L3 = -10 * np.log10(g68 / g45)
            Lt = -10 * np.log10(g76 / g68)
            in_range = -10 * np.log10(guided_at(q, Z68) / guided_at(q, ZJ))
            off = offset.get(v, {}).get(pol)
            cor = corners.get(v, {}).get(pol)
            adi = adiabatic(wl, v, pol)
            vr[pol] = {
                "raw_dB": raw, "full_dB": None if not f else f["loss_dB"], "tip_smooth_dB": tip,
                "tip_lattice_dB": tip_lat, "L3_dB": L3, "Lt_dB": Lt, "in_range_dB": in_range,
                "adiabatic_dB": adi, "corners_dB": cor, "staircase_offset_dB": off,
                "lower_dB": base + tip_lat - in_range,
                "final_dB": base + tip_lat - in_range + (adi or 0.0),
                "conservative_dB": None if cor is None else base + tip_lat - cor,
                "final_from": "full stack" if f else "oxide stack",
                "full_launch": None if not f else {k: f[k] for k in ("launch_fundamental", "power_coupling_fundamental",
                                                                     "guided_at_join", "best_y_um")},
                "best_y_um": r[pol]["best"]["y_um"],
                "launch_fundamental": r[pol]["launch_top"][0][1],
                "paper_fdtd_dB": paper_at(f"{pol}{150 if v == 'bilayer' else 220}", wl),
                "paper_meas_dB": paper_at(f"{pol}_meas_median", wl, "fig8") if v == "bilayer" else None,
                "tip_lattice_sequence_dB": {k: r["tip"][k][pol]["loss_dB_corrected"] for k in ("10nm", "5nm", "2.5nm")}
                if r.get("tip") and "10nm" in r["tip"] else None,
            }
        row[v] = vr
    summary["wavelengths"][str(wl)] = row

for name in ("staircase_1310", "opt_lengths_1310", "opt_lengths_1260_1310_1360", "opt_lengths_global_1310",
             "opt_lengths_global_1260_1310_1360",
             "opt_facet", "direct_1310_bilayer_exact_0_25", "direct_1310_bilayer_exact_25_45",
             "direct_1310_bilayer_exact_45_68", "direct_1310_bilayer_exact_76_86",
             "direct_1310_bilayer_aligned_0_25", "direct_1310_bilayer_aligned_76_86"):
    p = os.path.join(OUT, f"{name}.json")
    if os.path.exists(p):
        j = json.load(open(p))
        if name.startswith("direct"):
            summary.setdefault("direct", {})[name] = {k: j[k] for k in ("dbeme", "direct", "abs_dT", "sections", "seconds_direct") if k in j} | {
                "formulas": "previous (before the 2026-09-30 corrections)" if "_exact_" in name else "current"}
        elif name.startswith("staircase"):
            summary["staircase_1310"] = {v: {"factors": j[v]["factors"], "TE": j[v]["TE"], "TM": j[v]["TM"], "new_points": j[v]["new_points"], "seconds": j[v]["seconds"]} for v in j}
        elif name == "opt_facet":
            h = j["history"]
            summary["opt_facet"] = {"best": j.get("best"), "evaluations": len({(r["Wtip_nm"], r["g_um"]) for r in h}),
                                    "design": next((r for r in h if r["Wtip_nm"] == 130 and abs(r["g_um"] - 1.6) < 1e-9), None)}
        else:
            summary[name] = {k: j[k] for k in j if k in ("start", "optimum", "cma", "polished", "scipy", "seconds", "evaluations_cma", "wavelengths_nm")}
for wl in (1260, 1310, 1360):
    p = os.path.join(OUT, f"build_{wl}.json")
    if os.path.exists(p):
        summary.setdefault("build", {})[str(wl)] = json.load(open(p))
for name in ("gate_1310", "gate_detail_1310", "window_check", "archive_checks"):
    p = os.path.join(OUT, f"{name}.json")
    if os.path.exists(p):
        summary[name] = json.load(open(p))
for wl in (1260, 1310, 1360):
    p = os.path.join(OUT, f"fullstack_{wl}.json")
    if os.path.exists(p):
        summary.setdefault("fullstack", {})[str(wl)] = {
            v: {k: x for k, x in r.items() if k not in ("TE", "TM")} | {
                pol: {k: x for k, x in r[pol].items() if not k.startswith("guided_tip")} for pol in ("TE", "TM")}
            for v, r in json.load(open(p)).items()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print("wl variant pol: raw(oxide) / full / tip / in-range / corners -> BEST / conservative [from] | paper FDTD / measured")
for wl, row in summary["wavelengths"].items():
    for v, vr in row.items():
        for p in ("TE", "TM"):
            q = vr[p]
            fu = "  -  " if q["full_dB"] is None else f"{q['full_dB']:.2f}"
            cons = "n/a" if q["conservative_dB"] is None else f"{q['conservative_dB']:.2f}"
            m = "" if q["paper_meas_dB"] is None else f" / {q['paper_meas_dB']:.2f}"
            print(f"{wl} {v:12s} {p}: {q['raw_dB']:.2f} / {fu} / {q['tip_lattice_dB']:+.2f} / {q['in_range_dB']:.2f} / "
                  f"{(q['corners_dB'] or 0):.2f} -> {q['final_dB']:.2f} / {cons} [{q['final_from']}] | "
                  f"{q['paper_fdtd_dB']:.2f}{m}")
