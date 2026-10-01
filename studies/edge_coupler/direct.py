"""Section 5.8: DBEME against a direct (uncached) EME of the same stretch.

    python examples/run_solver_job.py --cpus 8-15 studies/edge_coupler/direct.py <nm> <z0_um> <z1_um> [variant] [aligned|exact]

Takes the dataset path's sections between ``z0`` and ``z1`` and cascades
them twice with the same interface physics (the scattering route of
``SingleEME``: output-side projection, pseudo-inverse cutoff, the reciprocal
projection of ``SingleEME.INTERFACE_RECIPROCAL`` (on for lossy bases since
2026-10-01), and the column cap only if ``SingleEME.INTERFACE_COLUMN_CAP`` opts
in - off by default since 2026-09-30):

* DBEME: the cached overlaps, tracked and gauge-equalised by the geometry;
* direct: every cross section solved afresh, each pair of neighbours
  assembled and overlapped on the spot, cascaded in the solver order with
  no tracking and no cache.  ``aligned`` solves the dataset's own (snapped)
  points - it tests the cache, the tracking and the bookkeeping; ``exact``
  solves the smooth design values at the same section starts - it adds the
  width-lattice staircase, the discretisation a dataset buys its speed with.

The figure of merit is basis-free: unit power in the start's highest-index
physical TE-like (and TM-like) mode, power arriving in the end's.  The
direct route streams two points at a time (the whole device assembled at
once would take tens of GB).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.fde.assemble import assemble, overlap_matrix  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME  # noqa: E402


def interface(E_a, H_a, E_b, H_b, x, y):
    """One interface scattering matrix, by SingleEME's own formulas and
    whatever its class-level switches are set to."""
    stub = object.__new__(SingleEME)
    stub._lossless = False
    stub.overlap_forward_ab = overlap_matrix(E_a, H_b, x, y)[None]
    stub.overlap_forward_ba = overlap_matrix(E_b, H_a, x, y)[None]
    stub.section_count, stub.mode_count = 2, E_a.shape[0]
    return stub._calc_interface_Smatrix()[0]


def prop(neff, dz, wl):
    N = len(neff)
    P = np.zeros((2 * N, 2 * N), complex)
    ph = np.exp(1j * 2 * np.pi / wl * np.asarray(neff) * dz)
    P[np.arange(N), np.arange(N)] = ph
    P[N + np.arange(N), N + np.arange(N)] = ph
    return P


def fundamental(neff, te, pol, n_clad):
    keep = [m for m in range(len(neff)) if (te[m] >= 0.5) == (pol == "TE")
            and np.imag(neff[m]) < 0.05 and np.real(neff[m]) > n_clad]
    return max(keep, key=lambda m: np.real(neff[m])) if keep else None


def main():
    wl_nm = int(sys.argv[1])
    z0, z1 = float(sys.argv[2]) * 1e-6, float(sys.argv[3]) * 1e-6
    variant = sys.argv[4] if len(sys.argv) > 4 else "bilayer"
    mode = sys.argv[5] if len(sys.argv) > 5 else "aligned"
    wl = wl_nm * 1e-9
    # a snapshot in test mode: the dataset may still be building, and this
    # check must neither solve into it nor race its writes
    import shutil

    from dbeme import DataUpdater

    snap = os.path.join(dv.ROOT, "cache", f"edge_snap_direct_{wl_nm}_{variant}_{mode}_{int(z0*1e6)}")
    shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(dv.dataset_dir(wl_nm, substrate=False), snap, ignore=shutil.ignore_patterns("__pycache__"))
    du = DataUpdater(snap, cache_size=1)
    du._is_testmode = True
    funcs_s, total = dv.path_functions(variant=variant)
    z_path = min(total, z1 + 2e-6)
    path = dv.EdgeCouplerPath(du, funcs_s, total_length=z_path, resolution=int(round(z_path / 25e-9)) + 1,
                              max_section_length=2e-6, verbose=False)
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    zs = cas.z
    ks = [k for k in range(len(cas.dz)) if z0 - 1e-12 <= zs[k] < z1 - 1e-12]
    k_end = ks[-1] + 1
    n_clad = du.get_cladding_index()
    N = cas.N
    P = cas.props()
    S = None
    for k in ks:
        S = P[k] if S is None else star(S, P[k])
        S = star(S, cas.mats[2 * k + 1])
    res = {"wavelength_nm": wl_nm, "z_um": [z0 * 1e6, z1 * 1e6], "variant": variant, "mode": mode,
           "sections": len(ks), "dbeme": {}, "direct": {}}
    for pol in ("TE", "TM"):
        i = fundamental(od["neff"][ks[0], :N], np.real(od["TE_pol"][ks[0], :N]), pol, n_clad)
        j = fundamental(od["neff"][k_end, :N], np.real(od["TE_pol"][k_end, :N]), pol, n_clad)
        res["dbeme"][pol] = float(abs(S[j, i]) ** 2) if i is not None and j is not None else None
    if mode == "aligned":
        pts = [tuple(od["EME_path"][k]) for k in ks + [k_end]]
    else:
        funcs, _ = dv.path_functions(variant=variant, snap=False)
        names = du.parameter_names

        def smooth(z):
            return tuple(float(np.asarray(funcs[nm](np.array([z])))[0]) for nm in names)

        pts = [smooth(zs[k]) for k in ks]
        pts.append(smooth(zs[k_end]) if zs[k_end] < dv.breakpoints()[-1] - 1e-9 else tuple(od["EME_path"][k_end]))
    dzs = [cas.dz[k] for k in ks]
    t0 = time.time()
    be = du.backend

    def solve(p):
        md = be.solve(tuple(p))
        x, y, neff, te, E, H = assemble([md], N, 2, lossless=False)
        return x, y, neff[0, :N], te[0, :N], E[0, :N], H[0, :N]

    prev = solve(pts[0])
    first = prev
    Sd = None
    for n in range(len(dzs)):
        same = pts[n + 1] == pts[n]
        cur = prev if same else solve(pts[n + 1])
        block = prop(prev[2], dzs[n], wl)
        Sd = block if Sd is None else star(Sd, block)
        if not same:
            Sd = star(Sd, interface(prev[4], prev[5], cur[4], cur[5], prev[0], prev[1]))
        prev = cur
        print(f"{n+1}/{len(dzs)} {time.time()-t0:.0f}s", flush=True)
    for pol in ("TE", "TM"):
        i = fundamental(first[2], first[3], pol, n_clad)
        j = fundamental(prev[2], prev[3], pol, n_clad)
        res["direct"][pol] = float(abs(Sd[j, i]) ** 2) if i is not None and j is not None else None
        if res["dbeme"][pol] is not None and res["direct"][pol] is not None:
            res.setdefault("abs_dT", {})[pol] = abs(res["dbeme"][pol] - res["direct"][pol])
    res["seconds_direct"] = time.time() - t0
    res["points_direct"] = [list(p) for p in pts]
    fn = os.path.join(dv.ROOT, "reports", "output", "edge",
                      f"direct_{wl_nm}_{variant}_{mode}_{int(round(z0*1e6))}_{int(round(z1*1e6))}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("dbeme", "direct", "abs_dT", "seconds_direct") if k in res}), flush=True)


if __name__ == "__main__":
    main()
