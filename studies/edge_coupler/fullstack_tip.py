"""The facet and tip on the full stack, joined to the oxide-stack path.

    python examples/run_solver_job.py --cpus 16-23 studies/edge_coupler/fullstack_tip.py <nm> [variant ...]

Why.  The EME datasets use the oxide-only stack (a Si substrate in the window
carries a PML band that fills any mode set, report 22 section 2.3).  At the
facet that basis is not converged: the near-cutoff tip mode's tail reaches
the bottom PML, and a wider window or a thicker PML moves its eq. (1) overlap
by 0.02-0.09 and its Im n by ~80 % (window_check.json).  On the full stack
the same checks move nothing (<= 7e-4, <= 7 %): the substrate, not the PML,
terminates the tail.

So the launch and the tip - facet to the 200 nm join, where the mode is
confined and the two stacks agree - are solved on the full stack, in a small
dataset whose grid is *identical* to the oxide datasets' (Si PML 0.5 um, so
the window and the stretched layers coincide).  At the join, one cross-stack
interface is built from the full-stack and oxide-stack modes of the same
point on the same grid, and the oxide path continues from there.  Substrate
leakage is then exact up to the join and a first-order correction after it.

Results: reports/output/edge/fullstack_<nm>.json.
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
from dbeme.fde.assemble import assemble  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch, power_coupling  # noqa: E402
from direct import interface  # noqa: E402


def full_dir(wl_nm):
    d = os.path.join(dv.ROOT, "datasets", f"Si_bilevel_pair_220_150nm_full_tip_{int(wl_nm)}")
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, "dataset_info.py")
    if not os.path.exists(fn):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(
                f'"""Dataset: the Wan & Wang edge coupler tip on the full stack, {wl_nm} nm.\n\n'
                "Si substrate under the 2 um BOX, taken into a 0.5 um Si-only PML so the\n"
                "grid and the stretched layers coincide with the oxide-stack datasets';\n"
                "only the facet-to-join stretch of the path is solved here, for a\n"
                "converged fibre launch and exact substrate leakage in the tip\n"
                "(studies/edge_coupler/fullstack_tip.py, reports/22).\n"
                '"""\n\n'
                "from dbeme.platforms import wan2025_edge_coupler_dataset_info\n\n"
                f"WAVELENGTH = {wl_nm * 1e-9!r}\n\n"
                "DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=True, bottom_pml=0.5e-6)\n")
    return d


def assembled(backend, point, N):
    md = backend.solve(tuple(point))
    x, y, neff, te, E, H = assemble([md], N, 2, lossless=False)
    return x, y, neff[0, :N], te[0, :N], E[0, :N], H[0, :N]


def fundamental_near(od, k, pol, n_clad, N, target, margin=0.03):
    """The fundamental at section ``k``: the highest physical mode of the
    polarisation that does not sit above the local lossless fundamental.

    On the full stack a PML-guided band of the Si substrate can put members
    with ``Im n`` < 0.05 and ``Re n`` near 2 into the set; they must not be
    mistaken for the tip mode."""
    neff, te = od["neff"][k, :N], np.real(od["TE_pol"][k, :N])
    keep = [m for m in range(N) if (te[m] >= 0.5) == (pol == "TE") and np.imag(neff[m]) < 0.05
            and n_clad < np.real(neff[m]) <= target + margin]
    return max(keep, key=lambda m: np.real(neff[m])) if keep else None


def branch_fields(E, H, slot, g):
    """Fields re-ordered and re-signed into a path's branch basis."""
    return (g[:, None, None, None] * E[slot]), (g[:, None, None, None] * H[slot])


def leakage_tail(du_ox, cas, wl_nm, variant, z_from):
    """First-order leakage correction for z >= z_from only."""
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", f"leakage_{wl_nm}_{variant}.json")
    if not os.path.exists(fn):
        return {"TE": 1.0, "TM": 1.0}
    _, samples = an.leakage_factor(du_ox, cas, wl_nm, variant)
    k0 = 2 * np.pi / (wl_nm * 1e-9)
    zmid = 0.5 * (cas.z[:-1] + cas.z[1:])
    out = {}
    for pol in ("TE", "TM"):
        s = sorted((samples or {}).get(pol, []))
        if not s:
            out[pol] = 1.0
            continue
        zs = np.array([v[0] for v in s])
        d = np.array([v[1] - v[2] for v in s])
        dim = np.where((zmid >= z_from) & (zmid <= zs.max()), np.interp(zmid, zs, d), 0.0)
        out[pol] = float(np.exp(-2 * k0 * np.sum(dim * cas.dz)))
    return out


def run(wl_nm, variant):
    t0 = time.time()
    du_ox = dv.open_dataset(wl_nm, substrate=False, cache_size=1)
    du_ox._is_testmode = True                       # the main dataset is only read here
    du_f = DataUpdater(full_dir(wl_nm), cache_size=2)
    N = du_ox.mode_numbers
    n_ox = float(du_ox.get_cladding_index())
    zj = tip.join_z(variant)

    # full-stack tip path (facet -> join, 10 nm lattice): solves what it lacks
    fp = tip.fine_path(du_f, variant, 10e-9, zj)
    fp._verbose = False
    od_f = fp.calc_output_data()
    cas_f = an.Cascade(fp, od_f)
    last = len(od_f["EME_path"]) - 1
    p_join = tuple(od_f["EME_path"][last])
    t_tip = time.time() - t0

    # oxide main path, from the join on
    path, _ = dv.build_path(du_ox, variant)
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    k_join = [k for k in range(len(od["EME_path"])) if tuple(od["EME_path"][k]) == p_join][0]

    # the cross-stack interface at the join, in both paths' branch bases
    xf, yf, nf, tef, Ef, Hf = assembled(du_f.backend, p_join, N)
    xo, yo, no, teo, Eo, Ho = assembled(du_ox.backend, p_join, N)
    if not (np.allclose(xf, xo) and np.allclose(yf, yo)):
        raise RuntimeError("the full and oxide grids differ; the cross interface needs one grid")
    slot_f, g_f = an.branch_maps(fp, od_f, du_f)
    slot_m, g_m = an.branch_maps(path, od, du_ox)
    Efb, Hfb = branch_fields(Ef, Hf, slot_f[last], g_f[last])
    Eob, Hob = branch_fields(Eo, Ho, slot_m[k_join], g_m[k_join])
    S_cross = interface(Efb, Hfb, Eob, Hob, xf, yf)

    P = cas.props()
    S_tail = None
    for k in range(k_join, len(cas.dz)):
        S_tail = P[k] if S_tail is None else star(S_tail, P[k])
        S_tail = star(S_tail, cas.mats[2 * k + 1])
    S = star(star(cas_f.lumped(), S_cross), S_tail)

    # launch on the full-stack facet (section 0 of the tip path: solver order, gauge +1)
    p0 = tuple(od_f["EME_path"][0])
    x0, y0, n0, te0, E0, H0 = assembled(du_f.backend, p0, N)
    output = an.endpoint(du_ox, tuple(od["EME_path"][-1]), f"{wl_nm}_{variant}_output", need_fields=False)
    names = path._tracking_mode_names[-1]
    r = np.empty(N)
    for k in range(N):
        r[int(names[k])] = output["r"][k]
    ports = dv.output_ports(od)
    leak = leakage_tail(du_ox, cas, wl_nm, variant, zj)

    res = {"z_join_um": zj * 1e6, "join_point": list(p_join), "k_join": k_join,
           "tip_sections": len(od_f["EME_path"]), "points_full": len(du_f.neff),
           "leakage_factor_tail": leak, "seconds": {"tip_path": t_tip}}
    rule = du_f.backend.target_rule
    t_join = rule(dict(zip(du_f.parameter_names, p_join)))
    t_0 = rule(dict(zip(du_f.parameter_names, od_f["EME_path"][0])))
    for pol in ("TE", "TM"):
        i_f = fundamental_near(od_f, last, pol, n_ox, N, t_join)
        i_o = an.fundamental_branch(od, k_join, pol, n_ox, N)
        res[f"cross_T_fundamental_{pol}"] = float(abs(S_cross[i_o, i_f]) ** 2)
        own = ports["TE0" if pol == "TE" else "TM0"]
        scan = []
        for yc in an.Y_SCAN:
            L = gaussian_launch(x0, y0, E0, H0, 0.5 * dv.MFD, center=(0.0, yc), pol=pol, n_medium=n_ox)
            o = S @ np.concatenate([L["a"], np.zeros(N, complex)])
            scan.append((float(abs(o[own]) ** 2 * r[own]), float(yc), float(np.sum(np.abs(o[N:]) ** 2))))
        best = max(scan)
        centred = min(scan, key=lambda s: abs(s[1]))
        i0 = fundamental_near(od_f, 0, pol, n_ox, N, t_0)
        Lc = gaussian_launch(x0, y0, E0, H0, 0.5 * dv.MFD, center=(0.0, best[1]), pol=pol, n_medium=n_ox)
        amps = cas_f.march(Lc["a"])
        pw = np.abs(amps) ** 2
        te_f = np.real(od_f["TE_pol"][:, :N]).astype(float)
        phys_f = np.array([dv.physical(od_f, k) for k in range(len(od_f["EME_path"]))])
        guided = np.sum(pw * phys_f * ((te_f >= 0.5) if pol == "TE" else (te_f < 0.5)), axis=1)
        res[pol] = {
            "P_raw": best[0], "loss_dB_raw": dv.loss_db(best[0]),
            "loss_dB": dv.loss_db(best[0] * leak[pol]), "best_y_um": best[1] * 1e6,
            "loss_dB_centred": dv.loss_db(centred[0] * leak[pol]), "reflected": best[2],
            "launch_fundamental": float(abs(Lc["a"][i0]) ** 2),
            "power_coupling_fundamental": power_coupling(x0, y0, E0[i0], H0[i0], 0.5 * dv.MFD,
                                                         center=(0.0, best[1]), pol=pol, n_medium=n_ox),
            "guided_at_join": float(guided[-1]),
            "guided_tip_z_um": (cas_f.z[: len(guided)] * 1e6).tolist(), "guided_tip": guided.tolist(),
        }
    res["seconds"]["total"] = time.time() - t0
    return res


def main():
    wl = int(sys.argv[1])
    variants = sys.argv[2:] or ["bilayer", "conventional"]
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", f"fullstack_{wl}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for v in variants:
        out[v] = run(wl, v)
        json.dump(out, open(fn, "w"), indent=1, default=float)
        print(wl, v, {p: (round(out[v][p]["loss_dB"], 3), round(out[v][p]["launch_fundamental"], 3),
                          round(out[v][p]["power_coupling_fundamental"], 3), round(out[v][p]["guided_at_join"], 3))
                      for p in ("TE", "TM")},
              "cross T", {p: round(out[v][f"cross_T_fundamental_{p}"], 4) for p in ("TE", "TM")}, flush=True)


if __name__ == "__main__":
    main()
