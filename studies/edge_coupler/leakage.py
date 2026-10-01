"""Substrate leakage of the weakly bound points, from full-stack solves.

    python examples/run_solver_job.py --cpus 16-23 studies/edge_coupler/leakage.py <nm> [variant] [dz_um] [z_max_um]

The EME basis is the oxide-only stack (a Si substrate inside the window
carries a PML band that no single set clears at both ends of the path).
Where the tip modes are weakly bound, the real device leaks into the Si
substrate 2 um below; and in the oxide-only window the same near-cutoff
tails reach the bottom PML and pick up a spurious loss of their own.  This
script solves the full stack (Si substrate in a thin Si-only PML) at the
path's points up to ``z_max`` every ``dz`` and records the even TE and TM
fundamentals' ``n_eff``; ``analyse.py`` replaces the oxide-only ``Im n`` of
the launched branch with this one:

    P -> P exp(-2 k0 \\int (Im n_full - Im n_ox) dz).

First order in the (1e-4) leakage rate, which does not reshape the mode.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
from dbeme.platforms import wan2025_edge_coupler_dataset_info  # noqa: E402


def path_points(wl_nm, variant):
    """``[(z_start, point)]`` of the snapped path, without solving anything."""
    for attempt in range(20):            # a running build may be rewriting the pickles
        try:
            du = dv.open_dataset(wl_nm, substrate=False, cache_size=1)
            break
        except Exception:
            time.sleep(5)
    path, _ = dv.build_path(du, variant)
    sp, dz = path.calc_simulation_parameters()
    pts, dzs, _, _ = path.interp_multi_adj_pts(sp, dz)
    z = np.concatenate([[0.0], np.cumsum(dzs)])[:-1]
    return list(zip(z, [tuple(map(float, p)) for p in pts])), du


def fundamentals(md, conf):
    out = {}
    for pol in ("TE", "TM"):
        keep = [m for m in range(len(md.neff))
                if ((md.TE_pol[m] >= 0.5) == (pol == "TE")) and conf[m] >= 0.01
                and np.imag(md.neff[m]) < 0.05]
        if not keep:
            out[pol] = None
            continue
        m = max(keep, key=lambda k: np.real(md.neff[k]))
        out[pol] = [float(np.real(md.neff[m])), float(np.imag(md.neff[m])), float(conf[m])]
    return out


def main():
    wl = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else "bilayer"
    step = float(sys.argv[3]) * 1e-6 if len(sys.argv) > 3 else 1.0e-6
    z_max = float(sys.argv[4]) * 1e-6 if len(sys.argv) > 4 else 50e-6
    pts, _ = path_points(wl, variant)
    chosen, last = [], -np.inf
    for z, p in pts:
        if z <= z_max and (z - last >= step - 1e-12) and (not chosen or p != chosen[-1][1]):
            chosen.append((z, p))
            last = z
    be = wan2025_edge_coupler_dataset_info(wl * 1e-9, substrate=True)().get_fde_backend()
    path = os.path.join(dv.ROOT, "reports", "output", "edge", f"leakage_{wl}_{variant}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"wavelength_nm": wl, "variant": variant, "points": []}
    done = {tuple(r["point"]) for r in res["points"]}
    t0 = time.time()
    for z, p in chosen:
        if p in done:
            continue
        params = dict(zip(("w_low", "w_high", "gap"), p), wavelength=wl * 1e-9)
        md, conf = be.mode_data(params, be.target_for(params))
        f = fundamentals(md, conf)
        res["points"].append({"z": float(z), "point": list(p), "full": f})
        json.dump(res, open(path, "w"), indent=1)
        print(f"z={z*1e6:6.2f} {p} TE={f['TE']} TM={f['TM']} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
