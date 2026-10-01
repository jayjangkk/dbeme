"""Is the join-to-L3 stretch's loss lattice or adiabatic?  Zero mode solves.

Every ``dz`` of a stretch is scaled by the same factor (the sections and
their interfaces are untouched), fundamental in -> fundamental out, over the
whole stretch (the 200 nm tip join at 13.825 um to the end of L3) and over its
three parts.  A smooth taper's loss falls with length; a staircase's step
loss does not.  The drop from the Table 1 length to the long-length floor is
the length-dependent - physical - part.

    python studies/edge_coupler/stretch_scan.py 1310 [1260 1360]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402

STRETCHES = {"join_to_L3": (13.825, 68.0), "L1_tail": (13.825, 25.0), "L2": (25.0, 45.0), "L3": (45.0, 68.0)}
SCALES = np.round(np.exp(np.linspace(np.log(0.25), np.log(8.0), 21)), 4)
FLOOR = (4.0, 8.0)   # the long-length floor: mean loss over these scales


def block(cas, ks, s):
    S = None
    for k in ks:
        P = np.zeros((2 * cas.N, 2 * cas.N), complex)
        ph = np.exp(1j * cas.beta[k] * cas.dz[k] * s)
        idx = np.arange(cas.N)
        P[idx, idx] = ph
        P[cas.N + idx, cas.N + idx] = ph
        S = P if S is None else star(S, P)
        S = star(S, cas.mats[2 * k + 1])
    return S


def scan_variant(du, variant):
    path, _ = dv.build_path(du, variant)
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    n_clad = du.get_cladding_index()
    z0 = cas.z[:-1] * 1e6
    out = {}
    for name, (za, zb) in STRETCHES.items():
        ks = np.flatnonzero((z0 >= za - 1e-6) & (z0 < zb - 1e-6))
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        row = {"z_um": [float(z0[k0]), float(cas.z[k1] * 1e6)], "sections": len(ks), "scales": SCALES.tolist()}
        blocks = [block(cas, ks, s) for s in SCALES]
        for pol in ("TE", "TM"):
            i = an.fundamental_branch(od, k0, pol, n_clad, cas.N)
            j = an.fundamental_branch(od, k1, pol, n_clad, cas.N)
            loss = [float(-10 * np.log10(abs(S[j, i]) ** 2)) for S in blocks]
            L1 = float(-10 * np.log10(abs(block(cas, ks, 1.0)[j, i]) ** 2))
            fl = [l for s, l in zip(SCALES, loss) if FLOOR[0] <= s <= FLOOR[1]]
            row[pol] = {"loss_dB": loss, "loss_design_dB": L1, "floor_dB": float(np.mean(fl)),
                        "floor_spread_dB": float(np.ptp(fl)), "adiabatic_dB": L1 - float(np.mean(fl))}
        out[name] = row
    return out


def main():
    wls = [int(a) for a in sys.argv[1:]] or [1310]
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", "stretch_scan.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for wl in wls:
        t0 = time.time()
        du = dv.open_dataset(wl, substrate=False)
        res[str(wl)] = {}
        for v in ("bilayer", "conventional"):
            res[str(wl)][v] = r = scan_variant(du, v)
            for name, row in r.items():
                print(f"{wl} {v:12s} {name:10s} " + "  ".join(
                    f"{p}: design {row[p]['loss_design_dB']:.3f} floor {row[p]['floor_dB']:.3f} "
                    f"(spread {row[p]['floor_spread_dB']:.3f}) adiabatic {row[p]['adiabatic_dB']:+.3f}"
                    for p in ("TE", "TM")), flush=True)
        res[str(wl)]["seconds"] = time.time() - t0
        json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
