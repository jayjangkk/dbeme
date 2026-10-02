"""Archive the side checks reports/22 quotes (warm, read-only).

    python studies/edge_coupler/archive_checks.py

* the interface column cap: capped vs uncapped loss, passivity, and the
  interfaces whose physical columns exceed unit power before capping;
* the facet: which x-odd modes the centred beam excites, and the
  single-mode |a|^2 r of the fundamental;
* the gauge rule: one sign per mode for both overlap sets (``Geometry``
  since tag ``pre_gauge_ab_ba``) against upstream's two-mask rule, kept here
  as ``_two_mask_rule`` (the merge-step reflection and the device loss at
  1260/1310/1360 nm);
* the DBEME side of every direct_*.json recomputed with the current path
  code, cut from the full device path (``direct.py`` cuts it from a path that
  ends 2 um past the stretch).  Every direct run was re-run on the code of
  2026-10-02 (reciprocal projection on), so both sides use the same
  interface formulas.

Writes reports/output/edge/archive_checks.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME  # noqa: E402


def open_ro(wl):
    du = dv.open_dataset(wl, substrate=False, cache_size=1)
    du._is_testmode = True
    return du


def device(du, wl, variant):
    path, _ = dv.build_path(du, variant)
    od = path.calc_output_data()
    cas = an.Cascade(path, od)
    return path, od, cas


def best_loss(S, od, path, du, wl, variant, pol):
    N = S.shape[0] // 2
    facet = an.endpoint(du, tuple(od["EME_path"][0]), f"{wl}_{variant}_facet")
    output = an.endpoint(du, tuple(od["EME_path"][-1]), f"{wl}_{variant}_output", need_fields=False)
    names = path._tracking_mode_names[-1]
    r = np.empty(N)
    for k in range(N):
        r[int(names[k])] = output["r"][k]
    own = dv.output_ports(od)["TE0" if pol == "TE" else "TM0"]
    n_ox = float(du.get_cladding_index())
    best = 0.0
    for yc in an.Y_SCAN:
        a = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                            center=(0.0, yc), pol=pol, n_medium=n_ox)["a"]
        o = S @ np.concatenate([a, np.zeros(N, complex)])
        best = max(best, float(abs(o[own]) ** 2 * r[own]))
    return dv.loss_db(best)


def cap_check(du, wl):
    out = {}
    default_cap = SingleEME.INTERFACE_COLUMN_CAP
    for variant in ("bilayer", "conventional"):
        row = {}
        for cap in ("auto", False):
            SingleEME.INTERFACE_COLUMN_CAP = cap
            path, od, cas = device(du, wl, variant)
            N = cas.N
            S = cas.lumped()
            col = np.sum(np.abs(S[:, :N]) ** 2, axis=0)
            r = {"loss_dB_raw": {p: best_loss(S, od, path, du, wl, variant, p) for p in ("TE", "TM")},
                 "max_physical_column": float(col[dv.physical(od, 0)].max())}
            if cap is False:
                exc = []
                for k in range(len(cas.dz)):
                    M = cas.mats[2 * k + 1]
                    c = np.sum(np.abs(M[:, :N]) ** 2, axis=0)
                    ph = dv.physical(od, k)
                    if ph.any() and c[ph].max() > 1.0:
                        exc.append(float(c[ph].max() - 1.0))
                r["interfaces_with_physical_column_above_1"] = len(exc)
                r["excess_range"] = [min(exc), max(exc)] if exc else None
            row[f"cap={cap}"] = r
        SingleEME.INTERFACE_COLUMN_CAP = default_cap
        row["cap_cost_dB"] = {p: row["cap=auto"]["loss_dB_raw"][p] - row["cap=False"]["loss_dB_raw"][p]
                              for p in ("TE", "TM")}
        out[variant] = row
    return out


def facet_check(du, wl):
    path, od, cas = device(du, wl, "bilayer")
    facet = an.endpoint(du, tuple(od["EME_path"][0]), f"{wl}_bilayer_facet")
    E, H = facet["E"].astype(complex), facet["H"].astype(complex)
    N = E.shape[0]
    par = []
    for m in range(N):
        c = 0 if facet["TE_pol"][m] >= 0.5 else 1
        e = E[m, c]
        par.append(float(np.real(np.sum(e * e[::-1, :]) / np.sum(e * e))))
    par = np.array(par)
    n_ox = float(du.get_cladding_index())
    out = {}
    for pol in ("TE", "TM"):
        L = gaussian_launch(facet["x"], facet["y"], E, H, 0.5 * dv.MFD, center=(0.0, 0.0), pol=pol, n_medium=n_ox)
        a = L["a"]
        odd = [m for m in np.argsort(-np.abs(a) ** 2) if par[m] < -0.9][:5]
        i = an.fundamental_branch(od, 0, pol, n_ox, N)
        out[pol] = {
            "odd_carriers": [{"mode": int(m), "a2": float(abs(a[m]) ** 2), "neff": [float(facet["neff"][m].real),
                              float(facet["neff"][m].imag)], "TE_pol": float(facet["TE_pol"][m]),
                              "parity": float(par[m])} for m in odd],
            "fundamental": {"mode": int(i), "a2": float(abs(a[i]) ** 2), "r": float(facet["r"][i]),
                            "a2_times_r": float(abs(a[i]) ** 2 * facet["r"][i])},
            "sum_a2_all_modes": float(np.sum(np.abs(a) ** 2)),
            "window_power": L["window_power"],
        }
    return out


def _two_mask_rule(self, overlap_ab, overlap_ba, overlap_dict = 0, index_mapping = 0):
    """Upstream's ``Geometry._equalize_overlap_phase`` (tag ``pre_gauge_ab_ba``),
    kept verbatim for the gauge-rule comparison: masks from ``diag(overlap_ab)``
    and, independently, from ``diag(overlap_ba)``.

    Parameters:
        - index_mapping : dictionary mapping from EME_path index to simul_params_index
        - overlap_ab : ndarray overlap along EME_path
        - overlap_dict: dictionary, two keys: "ab" and "ba"
            overlap_dict["ab"]: dict where multi_adj index in simul_params is key and list of overlaps is value
            overlap_dict["ba"]: dict where multi_adj index in simul_params is key and list of overlaps is value
    """
    if overlap_dict == 0:
        overlap_dict = dict()   # dummy overlap_dict
        index_mapping = {i: i for i in range(len(overlap_ab))}  # dummmy index mapping

    section_num, mode_count, _ = overlap_ab.shape
    mode_count = int(mode_count/2)
    backward_mode_mask = np.ones(shape=(mode_count,))
    backward_mode_mask = np.concatenate((backward_mode_mask, -backward_mode_mask))

    i = 0
    for i in range(len(overlap_ab) - 1):
        # Create mask for each diagonal element in the 2D slice overlap[i]
        mask = np.sign(np.diagonal(overlap_ab[i]).real)
        mask *= backward_mode_mask

        # mask needs to be reshaped to broadcast correctly along rows and columns
        mask_row = mask[:, np.newaxis]  # Shape (m, 1) for broadcasting across columns
        mask_col = mask[np.newaxis, :]   # Shape (1, m) is fine for broadcasting across rows

        # Apply the mask across all columns for the i-th layer
        overlap_ab[i] *= mask_col
        if index_mapping[i] in overlap_dict["ab"]:
            overlap_dict["ab"][index_mapping[i]] *= mask_col

        # Apply the mask across all rows for the (i+1)-th layer
        overlap_ab[i + 1] *= mask_row
        if index_mapping[i+1] in overlap_dict["ab"]:
            overlap_dict["ab"][index_mapping[i+1]] *= mask_row

    if i > 0:
        # last overlap matrix row phase
        mask = np.sign(np.diagonal(overlap_ab[i+1]).real)
        mask *= backward_mode_mask
        mask_col = mask[np.newaxis, :]
        overlap_ab[i+1] *= mask_col


    for i in range(len(overlap_ba) - 1):
        # for some reason, overlap_ab and overlap_ba have different sign sometimes
        mask = np.sign(np.diagonal(overlap_ba[i]).real)
        mask *= backward_mode_mask

        mask_row = mask[:, np.newaxis]
        mask_col = mask[np.newaxis, :]

        overlap_ba[i] *= mask_row
        overlap_ba[i + 1] *= mask_col
        if index_mapping[i] in overlap_dict["ba"]:
            overlap_dict["ba"][index_mapping[i]] *= mask_row
        if index_mapping[i+1] in overlap_dict["ba"]:
            overlap_dict["ba"][index_mapping[i+1]] *= mask_col

    if i > 0:
        # last overlap matrix col phase
        mask = np.sign(np.diagonal(overlap_ba[i+1]).real)
        mask *= backward_mode_mask
        mask_row = mask[:, np.newaxis]
        overlap_ba[i+1] *= mask_row

    return overlap_ab, overlap_ba, overlap_dict


def gauge_check(wl):
    du = open_ro(wl)
    out = {}
    for rule in ("single (Geometry)", "two-mask (upstream, pre_gauge_ab_ba)"):
        if rule.startswith("two"):
            dv.EdgeCouplerPath._equalize_overlap_phase = _two_mask_rule
        path, od, cas = device(du, wl, "bilayer")
        N = cas.N
        n_ox = float(du.get_cladding_index())
        pts = [tuple(p) for p in od["EME_path"]]
        ks = [k for k in range(len(pts) - 1) if abs(pts[k][1] - 0.79e-6) < 1e-12
              and abs(pts[k][2] - 0.02e-6) < 1e-12 and pts[k] != pts[k + 1]]
        row = {"loss_dB_raw": {p: best_loss(cas.lumped(), od, path, du, wl, "bilayer", p) for p in ("TE", "TM")}}
        if ks:
            k = ks[0]
            M = cas.mats[2 * k + 1]
            i = an.fundamental_branch(od, k, "TE", n_ox, N)
            row["merge_step_TE0"] = {"transmitted": float(np.sum(np.abs(M[:N, i]) ** 2)),
                                     "reflected": float(np.sum(np.abs(M[N:, i]) ** 2)),
                                     "step": [[float(v) for v in pts[k]], [float(v) for v in pts[k + 1]]]}
        out[rule] = row
    del dv.EdgeCouplerPath._equalize_overlap_phase
    return out


def direct_refresh(du, wl):
    out = {}
    for fn in sorted(glob.glob(os.path.join(dv.ROOT, "reports", "output", "edge", f"direct_{wl}_*.json"))):
        d = json.load(open(fn))
        variant = d["variant"]
        z0, z1 = d["z_um"][0] * 1e-6, d["z_um"][1] * 1e-6
        path, od, cas = device(du, wl, variant)
        N = cas.N
        ks = [k for k in range(len(cas.dz)) if z0 - 1e-12 <= cas.z[k] < z1 - 1e-12]
        k_end = ks[-1] + 1
        P = cas.props()
        S = None
        for k in ks:
            S = P[k] if S is None else star(S, P[k])
            S = star(S, cas.mats[2 * k + 1])
        n_ox = float(du.get_cladding_index())
        row = {"sections": len(ks), "dbeme_stored": d["dbeme"], "direct": d["direct"], "dbeme_current": {},
               "abs_dT_current": {}}
        for pol in ("TE", "TM"):
            i = an.fundamental_branch(od, ks[0], pol, n_ox, N)
            j = an.fundamental_branch(od, k_end, pol, n_ox, N)
            v = float(abs(S[j, i]) ** 2)
            row["dbeme_current"][pol] = v
            if d["direct"].get(pol) is not None:
                row["abs_dT_current"][pol] = abs(v - d["direct"][pol])
        out[os.path.basename(fn)] = row
    return out


def main():
    res = {}
    du = open_ro(1310)
    res["column_cap_1310"] = cap_check(du, 1310)
    print(json.dumps(res["column_cap_1310"], indent=1), flush=True)
    res["facet_1310"] = facet_check(du, 1310)
    res["direct_refresh_1310"] = direct_refresh(du, 1310)
    print(json.dumps(res["direct_refresh_1310"], indent=1), flush=True)
    res["gauge_rule"] = {str(wl): gauge_check(wl) for wl in (1260, 1310, 1360)}
    print(json.dumps(res["gauge_rule"], indent=1), flush=True)
    json.dump(res, open(os.path.join(dv.ROOT, "reports", "output", "edge", "archive_checks.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
