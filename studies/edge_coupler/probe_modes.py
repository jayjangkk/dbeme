"""Phase 0.3 probe: what a lossy PML basis looks like along the edge coupler.

Solves representative cross sections of the Wan & Wang path on the proposed
grid and prints, per mode: n_eff, core confinement, TE fraction and where
the power sits (substrate, PML, air).  Nothing here is cached; this chooses
the dataset's target, mode count and window before its identity is fixed.

    python studies/edge_coupler/probe_modes.py <tag> [target] [modes] [points...]
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dbeme.fde import BiLevelPair  # noqa: E402
from dbeme.fde.pml import PMLModeSolver  # noqa: E402
from dbeme.fde.materials import air, silica, silicon  # noqa: E402

WL = 1.31e-6
POINTS = {
    "tip150": dict(w_low=0.13e-6, w_high=0.0, gap=1.6e-6),
    "L2mid": dict(w_low=0.15e-6, w_high=0.15e-6, gap=1.6e-6),
    "L3start": dict(w_low=0.0, w_high=0.34e-6, gap=1.6e-6),
    "Ltstart": dict(w_low=0.0, w_high=0.45e-6, gap=0.42e-6),
    "mmi": dict(w_low=0.0, w_high=0.8e-6, gap=0.0),
    "out": dict(w_low=0.0, w_high=0.225e-6, gap=0.0),
    "tip220": dict(w_low=0.0, w_high=0.13e-6, gap=1.6e-6),
}


def grid_kwargs(base=50e-9, fine=10e-9, half_width=5.5e-6, y_min=-2.36e-6, y_max=2.79e-6,
                x_fine=1.2e-6, y_fine=(-0.41e-6, 0.44e-6),
                pml=(("+x", 0.5e-6), ("-x", 0.5e-6), ("+y", 0.5e-6), ("-y", 0.25e-6))):
    mesh = int(round(2 * half_width / base)) + 1
    mesh_y = int(round((y_max - y_min) / base)) + 1
    return dict(
        window=(half_width, y_min, y_max), mesh=mesh, mesh_y=mesh_y,
        pml_thickness=dict(pml), pml_edges=("+x", "-x", "+y", "-y"), colocate=True,
        refine_x=((-x_fine, x_fine, fine),), refine_y=((y_fine[0], y_fine[1], fine),),
    )


def describe(solver, cs, md, conf, params):
    x, y = np.real(md.x), np.real(md.y)
    xi, yi = np.imag(md.x), np.imag(md.y)
    area = solver._cell_area()[: x.size, : y.size]
    in_pml = (xi != 0)[:, None] | (yi != 0)[None, :]
    sub = ((y < cs.y_substrate)[None, :] & ~in_pml) if cs.substrate is not None else np.zeros_like(in_pml)
    top = (y > cs.y_top_oxide)[None, :] & ~in_pml
    rows = []
    for i, n in enumerate(md.neff):
        hx, hy = md.H[i, 0], md.H[i, 1]
        p = (np.abs(hx) ** 2 + np.abs(hy) ** 2) * area
        tot = p.sum()
        ex = md.E[i, 0]
        par = np.sum(ex * ex[::-1, :]) / np.sum(ex * ex) if np.sum(np.abs(ex)) else np.nan
        rows.append(dict(
            i=i, neff_re=float(np.real(n)), neff_im=float(np.imag(n)), conf=float(conf[i]),
            te=float(md.TE_pol[i]), f_sub=float(p[sub].sum() / tot), f_pml=float(p[in_pml].sum() / tot),
            f_air=float(p[top].sum() / tot), parity_ex=float(np.real(par)),
        ))
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "probe"
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 2.3
    modes = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    variant = sys.argv[4] if len(sys.argv) > 4 else "si25"
    names = sys.argv[5:] or list(POINTS)
    if variant == "ox":      # no substrate: the oxide runs into a 0.5 um bottom PML
        cs = BiLevelPair(core=silicon(out_of_range="raise"), cladding=silica(out_of_range="raise"),
                         substrate=None, reference_wavelength=WL)
        kw = grid_kwargs(y_min=-2.81e-6, pml=(("+x", 0.5e-6), ("-x", 0.5e-6), ("+y", 0.5e-6), ("-y", 0.5e-6)))
    else:
        cs = BiLevelPair(core=silicon(out_of_range="raise"), cladding=silica(out_of_range="raise"),
                         reference_wavelength=WL)
        bottom = {"si25": 0.25e-6, "si15": 0.15e-6, "si35": 0.35e-6}[variant]
        kw = grid_kwargs(y_min=-2.11e-6 - bottom,
                         pml=(("+x", 0.5e-6), ("-x", 0.5e-6), ("+y", 0.5e-6), ("-y", bottom)))
    solver = PMLModeSolver(cs, wavelength=WL, num_modes=modes, confinement_threshold=0.01, **kw)
    nx, ny = len(solver.x), len(solver.y)
    print(f"grid {nx} x {ny} = {nx * ny} nodes, target {target}, N {modes}",
          flush=True)
    out = {"target": target, "modes": modes, "nx": nx, "ny": ny, "points": {}}
    for name in names:
        params = dict(POINTS[name], wavelength=WL)
        t0 = time.time()
        md, conf = solver.mode_data(params, target)
        dt = time.time() - t0
        rows = describe(solver, cs, md, conf, params)
        out["points"][name] = {"seconds": dt, "modes": rows}
        print(f"\n== {name}  {dt:.1f} s", flush=True)
        for r in rows:
            print(f"{r['i']:3d} n={r['neff_re']:.5f}{r['neff_im']:+.2e}j conf={r['conf']:.3f} "
                  f"TE={r['te']:.2f} sub={r['f_sub']:.3f} pml={r['f_pml']:.3f} air={r['f_air']:.3f} "
                  f"par={r['parity_ex']:+.3f}", flush=True)
    path = os.path.join(ROOT, "reports", "output", "edge", f"probe_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
