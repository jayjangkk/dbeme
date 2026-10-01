"""Staircase extrapolation of the whole device: step size -> 0.

    python examples/run_solver_job.py --cpus 0-7 studies/edge_coupler/staircase.py <nm> [variant ...]

Two lattice errors, both first order in the step and both absent from the
smooth device:

* the near-cutoff tip (facet to the 200 nm join): handled by the ``_tip``
  dataset at 10 / 5 / 2.5 nm (``analyse.tip_refinement``);
* everything that moves after it - the arms' outer edges in L1/L2, the
  height converter, and above all the gap in L3 and Lt, where each 20 nm
  step jogs both arms sideways by 10 nm and sheds 0.27 % of TE whatever the
  gap - handled here by merging 2 and 4 consecutive steps of the dataset's
  own staircase (``device.coarse_functions``) between the join and the MMI.

For each merge factor ``m`` the device is cascaded with the finest tip
(2.5 nm) in front; ``-ln P`` is fitted linear in ``m`` and the ``m -> 0``
intercept is the smooth-tail value.  The tip's own extrapolation (2.5 nm ->
0) is added on top.  A few walk corners the coarse paths visit are not in
the dataset and are solved (the dataset gains them).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
import tip  # noqa: E402
from dbeme import DataUpdater  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch  # noqa: E402

FACTORS = (1, 2, 3)


def run(wl_nm, variant, dataset=None):
    """:param dataset: a copy of the wavelength's dataset to add the merged
    steps' links to (two variants must not write one set of pickles at once)."""
    du = DataUpdater(dataset, cache_size=2) if dataset else dv.open_dataset(wl_nm, substrate=False)
    du_tip = DataUpdater(tip.tip_dir(wl_nm), cache_size=1)
    du_tip._is_testmode = True
    zj = tip.join_z(variant)
    # the end of L3: the arms stop converging there.  Lt is left alone - merging
    # its steps would coarsen the arm merge into the MMI, which is physics
    # (TE2 generation), not lattice
    z_mmi = dv.breakpoints()[2]
    out = {"z_join_um": zj * 1e6, "z_to_um": z_mmi * 1e6, "factors": {}}
    before = len(du.neff)
    t0 = time.time()
    n_ox = float(du.get_cladding_index())
    for m in ("main",) + FACTORS:
        if m == "main":
            # the path as the device uses it: 10 nm steps, diagonal steps walked
            # through their corner point (one axis at a time)
            funcs, total = dv.path_functions(variant=variant)
        else:
            funcs, total = dv.coarse_functions(du.parameter_grid, variant, m, zj, z_mmi)
        path = dv.EdgeCouplerPath(du, funcs, total_length=total, resolution=int(round(total / 25e-9)) + 1,
                                  max_section_length=2e-6, verbose=True)
        # the extrapolation family is corner-free: every change, merged or not,
        # is one interface.  Walking would put a corner point (one axis moved,
        # the other not) into every diagonal step - on the (w_low, w_high)
        # axes that is nearly every step of the height converter, where the
        # corner is a narrower arm and n_eff dips - and would walk a merged
        # step back to single steps.  The main path keeps its corners: they
        # are lattice error too, and the offset is taken against it.
        if m != "main":
            path.direct_zrange = (zj, z_mmi)
        od = path.calc_output_data()
        cas = an.Cascade(path, od)
        N = cas.N
        S, info = an.joined_lumped(du, path, od, cas, du_tip, variant, 2.5e-9)
        ports = dv.output_ports(od)
        p0 = tuple(od["EME_path"][0])
        facet = an.endpoint(du, p0, f"{wl_nm}_{variant}_facet")
        output = an.endpoint(du, tuple(od["EME_path"][-1]), f"{wl_nm}_{variant}_output", need_fields=False)
        names = path._tracking_mode_names[-1]
        r = np.empty(N)
        for k in range(N):
            r[int(names[k])] = output["r"][k]
        row = {"sections": len(od["EME_path"]), "steps_in_range": int(np.sum(
            (cas.z[1:-1] >= zj) & (cas.z[1:-1] < z_mmi)))}
        for pol in ("TE", "TM"):
            own = ports["TE0" if pol == "TE" else "TM0"]
            best = -1.0
            for yc in an.Y_SCAN:
                a = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                                    center=(0.0, yc), pol=pol, n_medium=n_ox)["a"]
                fwd = (S @ np.concatenate([a, np.zeros(N, complex)]))[:N]
                best = max(best, float(abs(fwd[own]) ** 2 * r[own]))
            row[pol] = best
        out["factors"][str(m)] = row
        print(wl_nm, variant, m, row, f"{time.time()-t0:.0f}s, new points {len(du.neff) - before}", flush=True)
    db = 10 / np.log(10)
    ms = np.array(FACTORS, float)
    for pol in ("TE", "TM"):
        L = np.array([-np.log(out["factors"][str(m)][pol]) for m in FACTORS])
        main = -np.log(out["factors"]["main"][pol])
        lin2 = np.polyfit(ms[:2], L[:2], 1)             # first order, the two finest
        lin3 = np.polyfit(ms, L, 1)                     # first order, all three
        quad = np.polyfit(ms, L, 2)                     # with a second-order term
        out[pol] = {
            "L_main_dB": float(db * main),
            "L_dB": [float(db * v) for v in L],
            "L_m1_dB": float(db * main),                # what the offset is taken against
            "L0_linear_dB": float(db * lin3[1]),
            "L0_linear_2pt_dB": float(db * lin2[1]),
            "L0_quadratic_dB": float(db * quad[2]),
            "corners_dB": float(db * (main - L[0])),    # main path minus its corner-free twin
            "slope_dB_per_factor": float(db * lin3[0]),
            "linear_fit_residual_dB": float(db * np.max(np.abs(np.polyval(lin3, ms) - L))),
        }
    out["new_points"] = len(du.neff) - before
    out["seconds"] = time.time() - t0
    return out


def main():
    wl = int(sys.argv[1])
    args = sys.argv[2:]
    dataset = None
    if args and args[0].startswith("--dataset="):
        dataset = args[0].split("=", 1)[1]
        args = args[1:]
    variants = args or ["bilayer", "conventional"]
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", f"staircase_{wl}.json")
    for v in variants:
        res = json.load(open(fn)) if os.path.exists(fn) else {}   # re-read: another job may have written
        res[v] = run(wl, v, dataset)
        json.dump(res, open(fn, "w"), indent=1)
        print(json.dumps({k: res[v][k] for k in ("TE", "TM")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
