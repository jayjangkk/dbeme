"""Print the numeric tables of reports/22 from the JSON outputs (markdown).

    python studies/edge_coupler/summary.py && python studies/edge_coupler/report_tables.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device as dv  # noqa: E402

OUT = os.path.join(dv.ROOT, "reports", "output", "edge")


def load(name):
    p = os.path.join(OUT, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


s = load("summary")
W = s["wavelengths"]


def f2(x):
    return "—" if x is None else f"{x:.2f}"


print("### loss table\n")
print("| dB / facet | oxide launch, 10 nm | full-stack launch, 10 nm | tip | join→L3 | its length-dependent part | **best estimate** | conservative | paper 3D-FDTD | measured (Fig. 8 median) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for wl in ("1310", "1260", "1360"):
    for v in ("bilayer", "conventional"):
        for p in ("TE", "TM"):
            q = W[wl][v][p]
            print(f"| {wl} nm, {v} {p} | {f2(q['raw_dB'])} | {f2(q['full_dB'])} | {q['tip_lattice_dB']:+.2f} | "
                  f"{f2(q['in_range_dB'])} | {f2(q['adiabatic_dB'])} | **{f2(q['final_dB'])}** | {f2(q['conservative_dB'])} | "
                  f"{q['paper_fdtd_dB']:.2f} | {f2(q['paper_meas_dB'])} |")

print("\n### best estimate minus paper, PDL, penalty (conservative bound in brackets)\n")
for wl in ("1260", "1310", "1360"):
    b, c = W[wl]["bilayer"], W[wl]["conventional"]
    fin = {p: b[p]["final_dB"] for p in ("TE", "TM")}
    con = {p: b[p]["conservative_dB"] for p in ("TE", "TM")}
    pap = {p: b[p]["paper_fdtd_dB"] for p in ("TE", "TM")}
    print(f"{wl}: bilayer TE {fin['TE'] - pap['TE']:+.2f} [{con['TE'] - pap['TE']:+.2f}], "
          f"TM {fin['TM'] - pap['TM']:+.2f} [{con['TM'] - pap['TM']:+.2f}]; "
          f"PDL {fin['TM'] - fin['TE']:.2f} [{con['TM'] - con['TE']:.2f}] (paper {pap['TM'] - pap['TE']:.2f}); "
          f"raw PDL {b['TM']['raw_dB'] - b['TE']['raw_dB']:.2f}, full PDL {(b['TM']['full_dB'] or 0) - (b['TE']['full_dB'] or 0):.2f}; "
          f"conv TE {c['TE']['final_dB'] - c['TE']['paper_fdtd_dB']:+.2f}, TM {c['TM']['final_dB'] - c['TM']['paper_fdtd_dB']:+.2f}; "
          f"conv penalty {c['TM']['final_dB'] - c['TE']['final_dB']:.2f} (paper {c['TM']['paper_fdtd_dB'] - c['TE']['paper_fdtd_dB']:.2f})")

st = load("staircase_1310")
if st:
    print("\n### staircase\n")
    print("| 1310 nm, dB | main path | corner-free 1× | 2× | 3× | linear 0× | quadratic 0× | linear-fit residual | corners |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in ("bilayer", "conventional"):
        if v not in st:
            continue
        for p in ("TE", "TM"):
            q = st[v][p]
            L = q["L_dB"]
            print(f"| {v} {p} | {q['L_main_dB']:.3f} | {L[0]:.3f} | {L[1]:.3f} | {L[2]:.3f} | {q['L0_linear_dB']:.3f} | "
                  f"{q['L0_quadratic_dB']:.3f} | {q['linear_fit_residual_dB']:.3f} | {q['corners_dB']:.3f} |")
        print(f"  ({v}: {st[v]['new_points']} new points, {st[v]['seconds']:.0f} s)")

fs = s.get("fullstack", {})
if fs:
    print("\n### full-stack tip\n")
    for wl, rows in fs.items():
        for v, r in rows.items():
            print(f"{wl} {v}: cross T TE {r['cross_T_fundamental_TE']:.4f} TM {r['cross_T_fundamental_TM']:.4f}; "
                  + "; ".join(f"{p}: loss {r[p]['loss_dB']:.3f} launch {r[p]['launch_fundamental']:.3f} "
                              f"eta {r[p]['power_coupling_fundamental']:.3f} join {r[p]['guided_at_join']:.3f} "
                              f"y {r[p]['best_y_um']:+.2f}" for p in ("TE", "TM"))
                  + f"; tail leak {r['leakage_factor_tail']}")

print("\n### direct\n")
import glob  # noqa: E402
for fn in sorted(glob.glob(os.path.join(OUT, "direct_1310_*.json"))):
    d = json.load(open(fn))
    print(os.path.basename(fn), d.get("sections"), {k: d[k] for k in ("dbeme", "direct", "abs_dT") if k in d},
          round(d.get("seconds_direct", 0)))
