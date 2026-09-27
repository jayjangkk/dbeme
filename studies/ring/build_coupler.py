"""Bus-ring point coupler on the DBEME ring-coupler dataset - `tasks/11` §2.1.

The ring arc is drawn in the straight frame: bus fixed, the ring's inner edge
at ``gap(z) = g0 + R - sqrt(R^2 - (z - z_max)^2)``, ending at the dataset's
largest gap (0.7 um, where the coupling left is ~1e-6).  One path per ``(g0, R)``; every
path over one dataset shares the gap grid points it visits, which is the
method's argument (the second gap costs only the points the first did not
touch).

For each gap the script cascades the path (``force_unitary=False``, interface
projection on the section entered), takes the lumped ``(2N, 2N)`` S-matrix in
tracked-branch order, picks the two TE0 supermodes at the (identical) end
cross sections and rotates them into the bus / ring basis
``psi_bus = (psi_e + s psi_o) / sqrt2`` with the sign ``s`` read off the
fields, so the result is the 4-port coupler in the port convention of
``dbeme.circuit.waveguide.ideal_coupler`` (o1 bus in, o2 bus through,
o3 ring in, o4 ring out).

Sanity checks per gap (`docs/validation_backlog.md` §5): reciprocity ``max|S - S^T|`` on the
guided block, power conservation, reflection, power leaving on any other
guided branch.  Optional: direct (uncached) EME on the same path (§7 A) and
a second slicing (§5.4).

Run (long, through the launcher):
  python examples/run_solver_job.py studies/ring/build_coupler.py --dataset Si_ring_coupler_220nm_1550 --radius-um 10 --gaps 150,200,250,300,350,400 --out reports/output/ring/coupler_1550.json [--direct 200] [--slicing]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dbeme import DataExtractor, DataUpdater, DirectParametricPath, ParametricPath  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME  # noqa: E402
from dbeme.validation import lumped_smatrix  # noqa: E402

SingleEME.INTERFACE_PROJECTION = "output"

W = 500e-9
GAP_END = 0.7e-6   # replaced by the dataset's largest gap in main()


def z_max_for(radius, g0, g_end=None):
    g_end = GAP_END if g_end is None else g_end
    d = g_end - g0
    return float(np.sqrt(max(2.0 * radius * d - d * d, 0.0)))


def gap_function(radius, g0, z_max, g_end=None):
    g_end = GAP_END if g_end is None else g_end

    def gap(z):
        u = np.asarray(z, dtype=float) - z_max
        inside = np.abs(u) <= radius
        g = np.full_like(u, g_end)
        g[inside] = g0 + radius - np.sqrt(radius * radius - u[inside] ** 2)
        return np.minimum(g, g_end)
    return gap


def schedule(radius, g0, width=W):
    z_max = z_max_for(radius, g0)
    return {"w1": width, "w2": width, "gap": gap_function(radius, g0, z_max)}, 2.0 * z_max, z_max


def guided_te_branches(path, n_modes):
    """Tracked branch indices of the guided TE modes at the first section."""
    neff = np.real(path.output_data["neff"])
    te = path.output_data["TE_pol"]
    return [i for i in range(n_modes) if te[0, i] > 0.5 and neff[0, i] > 1.5]


def supermode_sign(du, point, solver_indices, bus_centre_x):
    """``s`` such that ``psi_e + s psi_o`` localises in the bus.

    Uses the pinned fields of the end cross section: ``s = sign(Ex_e Ex_o)``
    at the bus core centre.  Returns ``(s, bus fraction of that combination)``.
    """
    md = du.solve_point(tuple(point))
    ix = int(np.argmin(np.abs(md.x - bus_centre_x)))
    iy = int(np.argmin(np.abs(md.y)))
    ie, io = solver_indices
    ex_e = np.real(md.E[ie, 0, ix, iy])
    ex_o = np.real(md.E[io, 0, ix, iy])
    s = 1.0 if ex_e * ex_o >= 0 else -1.0
    cs = du.backend.cross_section
    cut = cs.gap_centre({"w1": W, "w2": W, "gap": float(point[2])})
    combo = (md.E[ie] + s * md.E[io]) / np.sqrt(2.0)
    intensity = np.abs(combo[0]) ** 2 + np.abs(combo[1]) ** 2
    left = md.x < cut
    total = np.trapezoid(np.trapezoid(intensity, md.y, axis=1), md.x)
    bus = np.trapezoid(np.trapezoid(intensity[left], md.y, axis=1), md.x[left])
    frac_e = cs.power_fractions(md, split_x=cut)[ie]
    frac_o = cs.power_fractions(md, split_x=cut)[io]
    return s, float(bus / total), float(frac_e), float(frac_o)


def four_port(T_sup, s):
    """``U T U^T`` with ``U = [[1, s], [1, -s]] / sqrt2``: rows/cols (bus, ring)."""
    U = np.array([[1.0, s], [1.0, -s]]) / np.sqrt(2.0)
    return U @ T_sup @ U.T


def analyse(du, path, radius, g0, z_max, verbose=True):
    od = path.output_data
    n_modes = od["neff"].shape[1] // 2
    names = path._tracking_mode_names
    branches = guided_te_branches(path, n_modes)
    if len(branches) < 2:
        raise RuntimeError(f"expected 2 guided TE0 branches at the input, got {branches}")
    # the TE0 pair are the two highest-index TE branches (a weakly guided TE1
    # pair at n_eff ~1.5 also passes the TE filter on a 500 nm strip)
    neff_all = np.real(od["neff"])[0, branches]
    top = np.argsort(-neff_all)[:2]
    branches = [int(branches[top[0]]), int(branches[top[1]])]
    neff0 = np.real(od["neff"])[0, branches]
    order = np.argsort(-neff0)          # higher n_eff at the input first
    b_e, b_o = int(branches[order[0]]), int(branches[order[1]])
    first, last = od["EME_path"][0], od["EME_path"][-1]
    solver_e = int(np.flatnonzero(names[0] == b_e)[0])
    solver_o = int(np.flatnonzero(names[0] == b_o)[0])
    cs = du.backend.cross_section
    bus_edges = cs.edges({"w1": W, "w2": W, "gap": float(first[2])})[0]
    bus_centre = 0.5 * (bus_edges[0] + bus_edges[1])
    s, bus_frac, frac_e, frac_o = supermode_sign(du, first, (solver_e, solver_o), bus_centre)

    S = lumped_smatrix(path, force_unitary=False)
    T = S[:n_modes, :n_modes]
    Rf = S[n_modes:, :n_modes]
    T_sup = T[np.ix_([b_e, b_o], [b_e, b_o])]
    S4 = four_port(T_sup, s)
    guided = [i for i in range(n_modes)
              if np.real(od["neff"])[0, i] > du.data_info.get_cladding_index()
              and np.real(od["neff"])[-1, i] > du.data_info.get_cladding_index()]
    Tg = T[np.ix_(guided, guided)]
    other = [i for i in guided if i not in (b_e, b_o)]
    leak_bus = float(np.sum(np.abs(T[other][:, [b_e, b_o]] @ np.array([1.0, s]) / np.sqrt(2)) ** 2)) if other else 0.0

    out = {
        "g0_nm": g0 * 1e9, "radius_um": radius * 1e6, "z_max_um": z_max * 1e6,
        "length_um": 2 * z_max * 1e6, "sections": int(len(od["EME_path"])),
        "distinct_points": int(len(set(map(tuple, od["EME_path"])))),
        "branches": {"even": b_e, "odd": b_o, "neff_even": float(neff0[order[0]]),
                     "neff_odd": float(neff0[order[1]]), "sign": s,
                     "bus_fraction_of_combination": bus_frac,
                     "bus_fraction_even": frac_e, "bus_fraction_odd": frac_o},
        "S4": {"bus_bus": [S4[0, 0].real, S4[0, 0].imag], "bus_ring": [S4[1, 0].real, S4[1, 0].imag],
               "ring_bus": [S4[0, 1].real, S4[0, 1].imag], "ring_ring": [S4[1, 1].real, S4[1, 1].imag]},
        "t2": float(abs(S4[0, 0]) ** 2), "kappa2": float(abs(S4[1, 0]) ** 2),
        "kappa2_from_ring": float(abs(S4[0, 1]) ** 2), "t2_ring": float(abs(S4[1, 1]) ** 2),
        # The ring is the guide that moves; its translation mismatch at every
        # interface (equal to the mode's own overlap with itself shifted by the
        # step, 0.998 at 10 nm) is discarded by the projection and grows
        # linearly with the step count, independent of N (6/12/20: 0.781,
        # 0.781, 0.785 ring-side through power).  Light coupled at the tangent
        # point rides the ring through half the interfaces, so the coupling is
        # corrected by the square root of the ring-side survival; CMT from the
        # same n_eff agrees to 4 % after it (tasks/11 section 2.1).
        "kappa2_corrected": float(abs(S4[1, 0]) ** 2 / np.sqrt(abs(S4[1, 1]) ** 2)),
        "ring_side_survival": float(abs(S4[1, 1]) ** 2 + abs(S4[0, 1]) ** 2),
        "checks": {
            "reciprocity_max_abs_S_minus_ST": float(np.max(np.abs(Tg - Tg.T))),
            "reciprocity_te0_pair": float(np.max(np.abs(T_sup - T_sup.T))),
            "reflection_te0_pair_max_abs2": float(np.max(np.abs(Rf[np.ix_([b_e, b_o], [b_e, b_o])]) ** 2)),
            "power_bus_launch": float(abs(S4[0, 0]) ** 2 + abs(S4[1, 0]) ** 2 + leak_bus),
            "power_conservation_guided_block_max_dev": float(np.max(np.abs(np.sum(np.abs(Tg) ** 2, axis=0) - 1.0))),
            "reflection_max_abs2": float(np.max(np.abs(Rf[np.ix_(guided, guided)]) ** 2)),
            "leak_other_guided_branches_bus_launch": leak_bus,
            "sup_te_polarisation": [float(od["TE_pol"][0, b_e]), float(od["TE_pol"][0, b_o])],
        },
    }
    # even/odd splitting along the path, for the CMT check (§4.1)
    gaps = [float(p[2]) for p in od["EME_path"]]
    ne = np.real(od["neff"])[:, b_e]
    no = np.real(od["neff"])[:, b_o]
    out["path"] = {"gap_nm": [g * 1e9 for g in gaps], "delta_z_um": [float(d) * 1e6 for d in od["EME_delta_zs"]],
                   "neff_even": ne.tolist(), "neff_odd": no.tolist()}
    if verbose:
        print(f"  g0={g0*1e9:.0f} nm R={radius*1e6:.0f} um: {out['sections']} sections, "
              f"|t|^2={out['t2']:.5f} |k|^2={out['kappa2']:.5f} sum={out['checks']['power_bus_launch']:.5f} "
              f"recip(TE0)={out['checks']['reciprocity_te0_pair']:.1e} refl={out['checks']['reflection_max_abs2']:.1e} "
              f"s={s:+.0f} busfrac={bus_frac:.3f}")
    return out


def build_path(du, radius, g0, resolution=4000, max_section_length=10e-6):
    functions, length, z_max = schedule(radius, g0)
    path = ParametricPath(du, functions, total_length=length, resolution=resolution,
                          max_section_length=max_section_length, verbose=False)
    path.calc_output_data()
    return path, z_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--radius-um", type=float, default=10.0)
    ap.add_argument("--gaps", default="150,200,250,300,350,400", help="g0 values in nm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direct", type=float, default=None, help="g0 (nm) to compare with direct EME")
    ap.add_argument("--slicing", action="store_true", help="repeat the first gap at 2x max_section_length halved")
    args = ap.parse_args()

    dataset = os.path.join(ROOT, "datasets", args.dataset)
    radius = args.radius_um * 1e-6
    gaps = [float(g) * 1e-9 for g in args.gaps.split(",")]
    du = DataUpdater(dataset)
    global GAP_END
    GAP_END = float(np.max(du.data_info.get_parameter_grid()["gap"]))
    print(f"{args.dataset}: {len(du.neff)} points cached, N={du.data_info.get_mode_numbers()}, "
          f"lambda={du.data_info.get_wavelength()*1e9:.0f} nm")

    results = {"dataset": args.dataset, "gap_end_nm": GAP_END * 1e9,
               "wavelength_nm": du.data_info.get_wavelength() * 1e9,
               "mode_numbers": int(du.data_info.get_mode_numbers()), "radius_um": args.radius_um,
               "interface_projection": SingleEME.INTERFACE_PROJECTION, "gaps": []}
    for g0 in gaps:
        cached_before = len(du.neff)
        t0 = time.time()
        path, z_max = build_path(du, radius, g0)
        t_cold = time.time() - t0
        t0 = time.time()
        path_warm, _ = build_path(du, radius, g0)
        t_warm = time.time() - t0
        entry = analyse(du, path, radius, g0, z_max)
        entry["timing"] = {"first_build_s": t_cold, "warm_build_s": t_warm,
                           "points_solved": len(du.neff) - cached_before}
        results["gaps"].append(entry)
        print(f"    build {t_cold:.1f} s ({entry['timing']['points_solved']} new points), warm {t_warm:.2f} s")

    if args.slicing:
        g0 = gaps[0]
        path_a, z_max = build_path(du, radius, g0, resolution=4000, max_section_length=10e-6)
        path_b, _ = build_path(du, radius, g0, resolution=8000, max_section_length=0.5e-6)
        a = analyse(du, path_a, radius, g0, z_max, verbose=False)
        b = analyse(du, path_b, radius, g0, z_max, verbose=False)
        results["slicing"] = {"g0_nm": g0 * 1e9, "sections_a": a["sections"], "sections_b": b["sections"],
                              "delta_t2": b["t2"] - a["t2"], "delta_kappa2": b["kappa2"] - a["kappa2"]}
        print(f"  slicing: {a['sections']} vs {b['sections']} sections, d|k|^2 = {results['slicing']['delta_kappa2']:.2e}")

    if args.direct is not None:
        g0 = args.direct * 1e-9
        path, z_max = build_path(du, radius, g0)
        n_sections = len(path.output_data["EME_path"])
        de = DataExtractor(dataset)
        functions, length, _ = schedule(radius, g0)
        t0 = time.time()
        direct = DirectParametricPath(de, functions, total_length=length, resolution=n_sections)
        direct._verbose = False
        direct.calc_output_data()
        t_direct = time.time() - t0

        class _Shim:
            backend = de.backend
            data_info = du.data_info

            @staticmethod
            def solve_point(point):
                return de._solve(tuple(point))
        d = analyse(_Shim(), direct, radius, g0, z_max, verbose=False)
        ref = [e for e in results["gaps"] if abs(e["g0_nm"] - args.direct) < 1e-6][0]
        results["direct"] = {"g0_nm": args.direct, "sections": n_sections, "seconds": t_direct,
                             "dataset_warm_seconds": ref["timing"]["warm_build_s"],
                             "t2_direct": d["t2"], "kappa2_direct": d["kappa2"],
                             "delta_t2": ref["t2"] - d["t2"], "delta_kappa2": ref["kappa2"] - d["kappa2"],
                             "checks_direct": d["checks"]}
        print(f"  direct EME ({n_sections} solves, {t_direct:.0f} s): |k|^2 {d['kappa2']:.5f} vs dataset {ref['kappa2']:.5f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
