"""The sanity gate of reports/22 at 1310 nm, from the warm dataset.

    python examples/run_solver_job.py --cpus 8-15 studies/edge_coupler/gate.py [nm]

Rows (``docs/validation_backlog.md`` numbering):

* 5.1  reciprocity of the physical channel, facet fundamental -> output
       TE0/TM0, lossy criterion ``< 1e-2`` relative;
* passivity: power out of every physical input column of the lumped
       ``|S|^2``, ``< 1.05``; and of the actual fibre launch, ``<= 1``;
* 5.4  slicing: the same device sampled at 50 and 12.5 nm instead of 25 nm
       (walk corners the new sampling needs are solved and counted);
* 5.13a membership: the launched branches' tracked links along the path,
       and at every point the gap between the target rule's fundamental and
       the highest physical mode actually stored;
* F3   parity: power the centred launch puts into x-odd modes;
* F4   completeness: the beam's power in the window, the share the guided
       set takes, and the E-field reconstruction residual.

Results: ``reports/output/edge/gate_<nm>.json``.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import interface_switches  # noqa: E402
from dbeme.fde.assemble import overlap_matrix  # noqa: E402
from dbeme.propagator.fiber import gaussian_fields, gaussian_launch, physical_interior  # noqa: E402


def loss_at_best_y(S, N, facet, ports, r, pol, n_ox):
    own = ports["TE0" if pol == "TE" else "TM0"]
    best = -1.0
    for yc in an.Y_SCAN:
        a = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                            center=(0.0, yc), pol=pol, n_medium=n_ox)["a"]
        out = S @ np.concatenate([a, np.zeros(N, complex)])
        best = max(best, float(abs(out[own]) ** 2 * r[own]))
    return best


def branch_r(path, output, N):
    names = path._tracking_mode_names[-1]
    r = np.empty(N)
    for k in range(N):
        r[int(names[k])] = output["r"][k]
    return r


def variant_gate(du, wl, variant):
    n_ox = float(du.get_cladding_index())
    path, total = dv.build_path(du, variant)
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    N = cas.N
    S = cas.lumped()
    ports = dv.output_ports(od)
    p0, pL = tuple(od["EME_path"][0]), tuple(od["EME_path"][-1])
    facet = an.endpoint(du, p0, f"{wl}_{variant}_facet")
    output = an.endpoint(du, pL, f"{wl}_{variant}_output", need_fields=False)
    r = branch_r(path, output, N)
    out = {"sections": len(od["EME_path"])}

    # 5.1 reciprocity, physical channel - measured without SingleEME's
    # reciprocal projection, which makes a lossy cascade reciprocal by
    # construction (2026-10-01)
    with interface_switches(INTERFACE_RECIPROCAL=False):
        S_basis = an.Cascade(path, od).lumped()
    rec = {}
    for pol in ("TE", "TM"):
        i = an.fundamental_branch(od, 0, pol, n_ox, N)
        j = ports["TE0" if pol == "TE" else "TM0"]
        fwd, bwd = S_basis[j, i], S_basis[N + i, N + j]
        rec[pol] = {"abs": float(abs(fwd - bwd)), "relative": float(abs(fwd - bwd) / abs(fwd)),
                    "T": float(abs(fwd) ** 2)}
    out["reciprocity"] = rec

    # passivity: physical input columns, and the real launch
    phys0 = dv.physical(od, 0)
    col = np.sum(np.abs(S[:, :N]) ** 2, axis=0)
    out["passivity_max_physical_column"] = float(col[phys0].max())
    out["passivity_max_any_column"] = float(col.max())
    launch_tot = {}
    for pol in ("TE", "TM"):
        L = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                            center=(0.0, 0.0), pol=pol, n_medium=n_ox)
        o = S @ np.concatenate([L["a"], np.zeros(N, complex)])
        launch_tot[pol] = {"out_fwd_physical": float(np.sum(np.abs(o[:N]) ** 2 * dv.physical(od, -1))),
                           "reflected": float(np.sum(np.abs(o[N:]) ** 2)),
                           "window_power": L["window_power"]}
    out["launch_power_budget"] = launch_tot

    # 5.13a: links of the launched branches, and fundamental membership
    oab = np.asarray(od["overlap_ab"])
    memb = {}
    for pol in ("TE", "TM"):
        b = an.fundamental_branch(od, 0, pol, n_ox, N)
        links = np.abs(oab[:, b, b])
        k_min = int(np.argmin(links))
        memb[pol] = {"branch": int(b), "min_link": float(links.min()),
                     "at_point": [float(v) for v in od["EME_path"][k_min]],
                     "links_below_0p9": int(np.sum(links < 0.9)),
                     "still_fundamental_at_end": an.fundamental_branch(od, len(od["EME_path"]) - 1, pol, n_ox, N) == b}
    rule = du.backend.target_rule
    gaps = []
    for p in dict.fromkeys(tuple(q) for q in od["EME_path"]):
        target = rule(dict(zip(du.parameter_names, p)))
        n = np.asarray(du.neff[p])[:N]
        te = np.asarray(du.TE_pol[p])[:N]
        phys_te = [m for m in range(N) if te[m] >= 0.5 and n[m].imag < 0.05]
        top = max(np.real(n[phys_te])) if phys_te else np.nan
        gaps.append((float(target - top), [float(v) for v in p]))
    worst = max(gaps, key=lambda g: g[0])
    memb["target_minus_top_TE"] = {"max": worst[0], "at": worst[1],
                                   "median": float(np.median([g[0] for g in gaps]))}
    out["membership"] = memb

    # F3 parity and F4 completeness at the facet (centred beam)
    E = facet["E"].astype(complex)
    H = facet["H"].astype(complex)
    x, y = facet["x"], facet["y"]
    parity = []
    for m in range(N):
        c = 0 if facet["TE_pol"][m] >= 0.5 else 1
        e = E[m, c]
        num = np.sum(e * e[::-1, :])
        den = np.sum(e * e)
        parity.append(float(np.real(num / den)) if abs(den) > 0 else 0.0)
    parity = np.array(parity)
    inside = physical_interior(x, y)
    dx = np.abs(np.diff(np.real(x)))
    dy = np.abs(np.diff(np.real(y)))
    w = np.outer(np.r_[dx, dx[-1]], np.r_[dy, dy[-1]]) * inside
    guided = (np.real(facet["neff"]) > n_ox) & (np.imag(facet["neff"]) < 0.05)
    f34 = {}
    for pol in ("TE", "TM"):
        L = gaussian_launch(x, y, E, H, 0.5 * dv.MFD, center=(0.0, 0.0), pol=pol, n_medium=n_ox)
        a, s = L["a"], L["s"]
        Eg, _ = gaussian_fields(x, y, 0.5 * dv.MFD, pol=pol, n_medium=n_ox)
        rec_E = np.tensordot(s, E, axes=(0, 0))
        c = 0 if pol == "TE" else 1
        resid = np.sum(np.abs(Eg[0, c] - rec_E[c]) ** 2 * w) / np.sum(np.abs(Eg[0, c]) ** 2 * w)
        i = an.fundamental_branch(od, 0, pol, n_ox, N)
        f34[pol] = {
            "odd_power": float(np.sum(np.abs(a[parity < 0]) ** 2)),
            "even_power": float(np.sum(np.abs(a[parity > 0]) ** 2)),
            "fundamental_power": float(abs(a[i]) ** 2),
            "guided_set_power": float(np.sum(np.abs(a[guided]) ** 2)),
            "window_power": L["window_power"],
            "E_reconstruction_residual": float(resid),
        }
    out["facet"] = f34

    # 5.4 slicing: 50 and 12.5 nm sampling instead of 25 nm
    before = len(du.neff)
    base = {pol: loss_at_best_y(S, N, facet, ports, r, pol, n_ox) for pol in ("TE", "TM")}
    sl = {"25nm": {p: dv.loss_db(v) for p, v in base.items()}}
    for pitch in (50e-9, 12.5e-9):
        funcs, tot = dv.path_functions(variant=variant)
        pp = dv.EdgeCouplerPath(du, funcs, total_length=tot, resolution=int(round(tot / pitch)) + 1,
                                max_section_length=2e-6, verbose=False)
        odp = pp.calc_output_data()
        cp = an.Cascade(pp, odp)
        Sp = cp.lumped()
        portsp = dv.output_ports(odp)
        rp = branch_r(pp, output, N)
        v = {pol: loss_at_best_y(Sp, N, facet, portsp, rp, pol, n_ox) for pol in ("TE", "TM")}
        sl[f"{pitch*1e9:g}nm"] = {p: dv.loss_db(q) for p, q in v.items()}
        sl[f"{pitch*1e9:g}nm_absdT"] = {p: abs(v[p] - base[p]) for p in v}
        sl[f"{pitch*1e9:g}nm_sections"] = len(odp["EME_path"])
    sl["points_solved_for_slicing"] = len(du.neff) - before
    out["slicing"] = sl
    return out


def main():
    wl = int(sys.argv[1]) if len(sys.argv) > 1 else 1310
    du = dv.open_dataset(wl, substrate=False)
    t0 = time.time()
    res = {"wavelength_nm": wl}
    for v in ("bilayer", "conventional"):
        res[v] = variant_gate(du, wl, v)
        print(v, json.dumps(res[v], default=float)[:1500], flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(dv.ROOT, "reports", "output", "edge", f"gate_{wl}.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
