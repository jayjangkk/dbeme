"""Early look at a dataset still being built: the path's first ``z_end`` um.

    python studies/edge_coupler/prefix_check.py <nm> <z_end_um> [variant]

Copies the dataset (the build keeps writing it), opens the copy in test
mode - nothing may be solved into it - and cascades the finished prefix of
the path with the real fibre launch: guided power at ``z_end`` and the
fundamental's transmission through the prefix.
"""

import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme import DataUpdater  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch  # noqa: E402


def main():
    wl = int(sys.argv[1])
    z_end = float(sys.argv[2]) * 1e-6
    variant = sys.argv[3] if len(sys.argv) > 3 else "bilayer"
    src = dv.dataset_dir(wl, substrate=False)
    snap = os.path.join(dv.ROOT, "cache", f"edge_snap_{wl}")
    shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(src, snap, ignore=shutil.ignore_patterns("__pycache__"))
    du = DataUpdater(snap, cache_size=1)
    du._is_testmode = True          # refuse any solve into the snapshot
    funcs, _ = dv.path_functions(variant=variant)
    path = dv.EdgeCouplerPath(du, funcs, total_length=z_end, resolution=int(round(z_end / 25e-9)) + 1,
                              max_section_length=2e-6, verbose=False)
    t0 = time.time()
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    N = cas.N
    facet = an.endpoint(du, tuple(od["EME_path"][0]), f"{wl}_{variant}_facet")
    n_ox = float(du.get_cladding_index())
    te = np.real(od["TE_pol"][:, :N]).astype(float)
    phys = np.array([dv.physical(od, k) for k in range(len(od["EME_path"]))])
    out = {"z_end_um": z_end * 1e6, "sections": len(od["EME_path"]), "points": len(du.neff)}
    for pol in ("TE", "TM"):
        best = None
        for yc in an.Y_SCAN:
            L = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                                center=(0.0, yc), pol=pol, n_medium=n_ox)
            amps = cas.march(L["a"])
            pw = np.abs(amps) ** 2
            g = np.sum(pw * phys * ((te >= 0.5) if pol == "TE" else (te < 0.5)), axis=1)
            if best is None or g[-1] > best[1][-1]:
                best = (yc, g, pw)
        yc, g, pw = best
        i = an.fundamental_branch(od, 0, pol, n_ox, N)
        j = an.fundamental_branch(od, len(od["EME_path"]) - 1, pol, n_ox, N)
        S = cas.lumped()
        out[pol] = {"y_um": float(yc * 1e6), "guided_start": float(g[0]), "guided_end": float(g[-1]),
                    "guided_every_10": [float(v) for v in g[::10]],
                    "fundamental_T": float(abs(S[j, i]) ** 2),
                    "neff_in": float(np.real(od["neff"][0, i])), "neff_out": float(np.real(od["neff"][-1, j]))}
        print(pol, json.dumps(out[pol]), flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(dv.ROOT, "reports", "output", "edge",
                                     f"prefix_{wl}_{variant}_{int(z_end*1e6)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
