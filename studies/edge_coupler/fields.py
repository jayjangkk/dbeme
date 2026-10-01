"""The propagating field along the device (the counterpart of the paper's Fig. 4).

    python examples/run_solver_job.py --cpus 0-7 studies/edge_coupler/fields.py <nm> [variant] [n_planes]

The dataset keeps no fields, so the planes are re-solved: at each of
``n_planes`` positions ``z`` the section's point is solved again, its modes
are put in the dataset's basis (``assemble``), mapped to the path's branch
order (``_tracking_mode_names``) with the gauge signs the path applied
(``_equalize_overlap_phase`` flips branch ``j`` of section ``i+1`` by the
sign of the real diagonal of the raw overlap - recomputed here from the
stored overlaps), and summed with the marched amplitudes and the phase
``exp(i beta (z - z_k))``.

Saved: ``reports/output/edge/fields_<nm>_<variant>.npz`` - ``|E|^2`` on the
device layer (``y`` = -35 nm, inside both the 150 and the 220 nm Si) for
every plane and polarisation, and full cross sections at the paper's four
inset planes.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch  # noqa: E402

Y_CUT = -35e-9
INSETS_UM = (1.0, 26.0, 60.0, 86.5)     # paper Fig. 4: X = 77.5, 52.5, 18.5 and the output


branch_maps = an.branch_maps          # one implementation: repeated points keep their gauge


def main():
    wl = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) > 2 else "bilayer"
    n_planes = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    du = dv.open_dataset(wl, substrate=False)
    res, (path, od, cas, facet) = an.analyse_variant(du, wl, variant)
    N = cas.N
    slot, g = branch_maps(path, od, du)
    z_edges = cas.z
    total = z_edges[-1]
    zs = np.unique(np.concatenate([np.linspace(0.0, total, n_planes), np.array(INSETS_UM) * 1e-6]))
    zs = np.clip(zs, 0.0, total - 1e-12)
    ks = np.clip(np.searchsorted(z_edges, zs, side="right") - 1, 0, len(cas.dz) - 1)
    # the output plane belongs to the last point
    ks[np.isclose(zs, total, atol=1e-9)] = len(od["EME_path"]) - 1
    n_ox = float(du.get_cladding_index())
    amps, launches = {}, {}
    for pol in ("TE", "TM"):
        yb = res[pol]["best"]["y_um"] * 1e-6
        L = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                            center=(0.0, yb), pol=pol, n_medium=n_ox)
        amps[pol] = cas.march(L["a"])
        launches[pol] = L
    x = np.real(facet["x"])
    y = np.real(facet["y"])
    jy = int(np.argmin(np.abs(y - Y_CUT)))
    top = {pol: np.zeros((len(zs), x.size), np.float32) for pol in ("TE", "TM")}
    insets = {}
    solved = {}
    t0 = time.time()
    for iz, (z, k) in enumerate(zip(zs, ks)):
        p = tuple(od["EME_path"][k])
        if p not in solved:
            solved.clear()          # keep one point's fields in memory
            _, _, _, _, E, H = dv.point_fields(du, p)
            solved[p] = E
        E = solved[p]
        beta = np.asarray(od["beta"])[k, :N]
        dzk = z - z_edges[k] if k < len(cas.dz) else 0.0
        for pol in ("TE", "TM"):
            c = amps[pol][k] * np.exp(1j * beta * dzk) * g[k]
            field = np.tensordot(c, E[slot[k]], axes=(0, 0))       # (3, nx, ny)
            inten = np.sum(np.abs(field) ** 2, axis=0)
            top[pol][iz] = inten[:, jy]
            if any(abs(z - zi * 1e-6) < 1e-9 for zi in INSETS_UM) or (k == len(od["EME_path"]) - 1 and z > total - 1e-6):
                insets[f"{pol}_{z*1e6:.1f}"] = inten.astype(np.float32)
        print(f"plane {iz+1}/{len(zs)} z={z*1e6:.2f} um  section {k}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(
        os.path.join(dv.ROOT, "reports", "output", "edge", f"fields_{wl}_{variant}.npz"),
        z=zs, x=x, y=y, y_cut=Y_CUT, top_TE=top["TE"], top_TM=top["TM"],
        n=np.real(du.backend.index_profile(dict(zip(du.parameter_names, od["EME_path"][0]),
                                               wavelength=wl * 1e-9))),
        **insets)


if __name__ == "__main__":
    main()
