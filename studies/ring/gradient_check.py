"""Phase 5.3 of `tasks/11`: gradients through the cached grid.

The sax models are ``jnp`` functions, so ``jax.grad`` runs through the ring
with respect to any continuous parameter of the models - here the bus-ring
gap, via the coupler interpolated across the solved gaps
(`ring_model.gap_interpolated_coupler`).  The boundary to state: the gradient
exists over the *cached* grid (it is the interpolant's derivative), not
through the mode solver; a gap outside the solved set still costs solves.

Compares ``d(extinction)/d(gap)`` and ``d(kappa^2)/d(gap)`` from ``jax.grad``
with central finite differences of the same interpolant, and with the
finite difference of the *solved* points themselves (the dataset grid).

Run (main venv):  .venv/Scripts/python studies/ring/gradient_check.py
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402

from em_simulation.circuit import all_pass_amplitude, round_trip_phase  # noqa: E402
from studies.ring.ring_model import OUT, bend_index, gap_interpolated_coupler, load_coupler_json, load_loss_json  # noqa: E402


def main():
    paths = sorted(glob.glob(os.path.join(OUT, "coupler_15[357]0.json")))
    table, meta = load_coupler_json(paths)
    loss = load_loss_json()
    R_um = meta["radius_um"]
    coupler = gap_interpolated_coupler(table, wl_m=1.55e-6)
    gaps = np.array(coupler.info["gaps_nm"])
    bend = bend_index(loss, R_um)
    L = 2 * np.pi * R_um * 1e-6
    a = 10 ** (-loss["roughness"]["straight_db_per_cm_eim_index"] * L * 100 / 20)
    ng = float(bend.ng(1.55e-6))

    def kappa2(gap_nm):
        s = coupler(wl=1.55, gap_nm=gap_nm)
        return jnp.abs(s[("o1", "o4")]) ** 2

    def extinction_db(gap_nm):
        s = coupler(wl=1.55, gap_nm=gap_nm)
        r = jnp.abs(s[("o1", "o2")])
        t_res = ((a - r) / (1 - r * a)) ** 2
        t_off = ((a + r) / (1 + r * a)) ** 2
        return 10 * jnp.log10(t_off / t_res)

    rows = []
    for g in np.linspace(gaps[0] + 10, gaps[-1] - 10, 6):
        h = 1.0
        for name, f in (("kappa2", kappa2), ("extinction_dB", extinction_db)):
            grad = float(jax.grad(f)(g))
            fd = float((f(g + h) - f(g - h)) / (2 * h))
            rows.append({"gap_nm": float(g), "quantity": name, "jax_grad_per_nm": grad,
                         "finite_diff_per_nm": fd, "rel_diff": abs(grad - fd) / max(abs(fd), 1e-12)})
    # the solved points' own finite difference (the grid), for scale
    k2 = np.array([float(kappa2(g)) for g in gaps])
    grid_fd = np.gradient(k2, gaps)
    print(f"gaps solved: {gaps.tolist()} nm; kappa^2: {np.round(k2, 5).tolist()}")
    print(f"{'gap':>6} {'quantity':>14} {'jax.grad':>12} {'central FD':>12} {'rel':>8}")
    for r in rows:
        print(f"{r['gap_nm']:6.0f} {r['quantity']:>14} {r['jax_grad_per_nm']:12.4e} {r['finite_diff_per_nm']:12.4e} {r['rel_diff']:8.1e}")
    out = {"radius_um": R_um, "gaps_nm": gaps.tolist(), "kappa2_at_grid": k2.tolist(),
           "grid_finite_difference_per_nm": grid_fd.tolist(), "rows": rows,
           "note": "gradients are of the gap interpolant over the cached grid; the mode solver is not differentiated"}
    with open(os.path.join(OUT, "gradient_check.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print("wrote gradient_check.json")


if __name__ == "__main__":
    main()
