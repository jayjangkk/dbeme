"""Phases 2-3: the edge coupler on a warm dataset - launch, cascade, budget.

    python studies/edge_coupler/analyse.py <nm> [variant ...]

For each variant (``bilayer``, ``conventional``) and polarisation:

* the fibre beam (4 um MFD) is projected onto the facet's modes
  (``dbeme.propagator.fiber.gaussian_launch``), its vertical offset scanned
  and the best one kept, as the measurement aligns the fibre;
* the path's section matrices are cascaded (scattering route, no
  projection) and marched, so the guided power is known after every section;
* the output TE0/TM0 power is corrected for substrate leakage from the
  full-stack solves of ``leakage.py``.

Results: ``reports/output/edge/device_<nm>.json``.  The facet and output
fields cost a mode solve each and are cached under ``cache/edge``.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
from dbeme import EME  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch  # noqa: E402

CACHE = os.path.join(dv.ROOT, "cache", "edge")
SEGMENTS = ("L1", "L2", "L3", "Lt", "Lm", "out")
Y_SCAN = np.round(np.arange(-0.6, 0.601, 0.05), 3) * 1e-6


# ------------------------------------------------------------ cached fields


def endpoint(du, point, tag, need_fields=True):
    """Facet/output fields of a path point in the dataset basis, cached."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"{tag}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        if tuple(np.round(z["point"], 15)) == tuple(np.round(point, 15)):
            return {k: z[k] for k in z.files}
    t0 = time.time()
    x, y, neff, te, E, H = dv.point_fields(du, point)
    r = dv.power_ratio(x, y, E, H)
    out = {"point": np.asarray(point, float), "x": x, "y": y, "neff": neff, "TE_pol": te,
           "r": r, "seconds": np.array(time.time() - t0)}
    if need_fields:
        out.update(E=E.astype(np.complex64), H=H.astype(np.complex64))
    np.savez(fn, **out)
    return out


# ---------------------------------------------------------------- cascade


class Cascade:
    """The path's section matrices, with per-segment length scaling.

    Interfaces depend only on the snapped cross sections; propagation only
    on ``beta`` and ``dz``.  Scaling every ``dz`` of a segment is therefore a
    length change of that segment at zero mode solves - the "free axes" of
    Phase 4 - valid as long as the rebuilt path visits the same points,
    which a path parameterised by relative position within each segment
    does.
    """

    def __init__(self, path, od, design=None):
        self.path, self.od = path, od
        eme = EME(path, force_unitary=False)
        eme.propagator.calc_Smatrix(method="direct")
        self.mats = np.asarray(eme.propagator.smatrix)
        self.N = self.mats.shape[1] // 2
        self.beta = np.asarray(od["beta"])[:, :self.N]
        self.dz = np.asarray(od["EME_delta_zs"], float)
        self.z = np.concatenate([[0.0], np.cumsum(self.dz)])
        zb = dv.breakpoints(design)
        mid = 0.5 * (self.z[:-1] + self.z[1:])
        self.segment = np.searchsorted(np.asarray(zb[:-1]), mid, side="right")
        self.segment = np.minimum(self.segment, len(SEGMENTS) - 1)

    def props(self, scales=None):
        s = np.ones(len(SEGMENTS)) if scales is None else np.asarray(scales, float)
        out = np.zeros((len(self.dz), 2 * self.N, 2 * self.N), complex)
        ph = np.exp(1j * self.beta[: len(self.dz)] * (self.dz * s[self.segment])[:, None])
        idx = np.arange(self.N)
        out[:, idx, idx] = ph
        out[:, self.N + idx, self.N + idx] = ph
        return out

    def lumped(self, scales=None):
        P = self.props(scales)
        S = P[0]
        S = star(S, self.mats[1])
        for k in range(1, len(self.dz)):
            S = star(star(S, P[k]), self.mats[2 * k + 1])
        return S

    def segment_block(self, seg, scale=1.0):
        """Lumped matrix of one segment alone (its sections, their interfaces),
        with that segment's length scaled; and its first/last section index."""
        ks = np.flatnonzero(self.segment == seg)
        s = np.ones(len(SEGMENTS))
        s[seg] = scale
        P = self.props(s)
        S = None
        for k in ks:
            S = P[k] if S is None else star(S, P[k])
            S = star(S, self.mats[2 * k + 1])
        return S, int(ks[0]), int(ks[-1]) + 1

    def march(self, a, scales=None):
        """Forward amplitudes at the start of every section (downstream
        reflections ignored): ``(n_sections, N)``."""
        P = self.props(scales)
        amps = [np.asarray(a, complex)]
        S = None
        v = np.concatenate([a, np.zeros(self.N, complex)])
        for k in range(len(self.dz)):
            S = P[k] if S is None else star(S, P[k])
            S = star(S, self.mats[2 * k + 1])
            amps.append((S @ v)[: self.N])
        return np.asarray(amps)


# ---------------------------------------------------------------- leakage


def leakage_factor(du, cascade, wl_nm, variant):
    """``{pol: exp(-2 k0 int (Im n_full - Im n_ox) dz)}`` and the samples."""
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", f"leakage_{wl_nm}_{variant}.json")
    if not os.path.exists(fn):
        return {"TE": 1.0, "TM": 1.0}, None
    data = json.load(open(fn))
    k0 = 2 * np.pi / (wl_nm * 1e-9)
    samples = {"TE": [], "TM": []}
    for r in data["points"]:
        p = tuple(r["point"])
        if p not in du.neff:
            continue
        n_ox = np.asarray(du.neff[p])[: du.mode_numbers]
        te_ox = np.asarray(du.TE_pol[p])[: du.mode_numbers]
        for pol in ("TE", "TM"):
            full = r["full"].get(pol)
            sel = [m for m in range(len(n_ox)) if (te_ox[m] >= 0.5) == (pol == "TE")
                   and np.imag(n_ox[m]) < 0.05 and np.real(n_ox[m]) > du.get_cladding_index() - 0.01]
            if full is None or not sel:
                continue
            m = max(sel, key=lambda k: np.real(n_ox[k]))
            samples[pol].append((r["z"], full[1], float(np.imag(n_ox[m])), full[0], float(np.real(n_ox[m]))))
    fac = {}
    zmid = 0.5 * (cascade.z[:-1] + cascade.z[1:])
    for pol, s in samples.items():
        if not s:
            fac[pol] = 1.0
            continue
        s = sorted(s)
        zs = np.array([v[0] for v in s])
        d_im = np.array([v[1] - v[2] for v in s])
        zmax = zs.max()
        dim_path = np.where(zmid <= zmax, np.interp(zmid, zs, d_im), 0.0)
        fac[pol] = float(np.exp(-2 * k0 * np.sum(dim_path * cascade.dz)))
    return fac, samples


# ---------------------------------------------------------------- the device


def analyse_variant(du, wl_nm, variant):
    t0 = time.time()
    path, total = dv.build_path(du, variant)
    od = path.calc_output_data()
    t_data = time.time() - t0
    cas = Cascade(path, od)
    S = cas.lumped()
    t_s = time.time() - t0 - t_data
    N = cas.N
    p0, pL = tuple(od["EME_path"][0]), tuple(od["EME_path"][-1])
    facet = endpoint(du, p0, f"{wl_nm}_{variant}_facet")
    output = endpoint(du, pL, f"{wl_nm}_{variant}_output", need_fields=False)
    names = path._tracking_mode_names[-1]
    r_branch = np.empty(N)
    for k in range(N):
        r_branch[int(names[k])] = output["r"][k]
    ports = dv.output_ports(od)
    leak, samples = leakage_factor(du, cas, wl_nm, variant)
    n_ox = float(du.get_cladding_index())
    res = {"variant": variant, "ports": ports, "total_length": total,
           "sections": len(od["EME_path"]), "leakage_factor": leak,
           "leakage_samples": samples,
           "seconds": {"warm_data": t_data, "smatrix": t_s},
           "output_neff": {k: [float(np.real(od["neff"][-1, j])), float(np.imag(od["neff"][-1, j]))]
                           for k, j in ports.items() if j is not None},
           "r_out": {k: float(r_branch[j]) for k, j in ports.items() if j is not None}}
    for pol in ("TE", "TM"):
        scan = []
        for yc in Y_SCAN:
            launch = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                                     center=(0.0, yc), pol=pol, n_medium=n_ox)
            a = launch["a"]
            out = S @ np.concatenate([a, np.zeros(N, complex)])
            fwd = out[:N]
            p = np.abs(fwd) ** 2 * r_branch
            scan.append({"y_um": float(yc * 1e6), "P_TE0": float(p[ports["TE0"]]),
                         "P_TM0": float(p[ports["TM0"]]),
                         "launch_confined": float(np.sum(np.abs(a[:4]) ** 2)),
                         "reflected": float(np.sum(np.abs(out[N:]) ** 2))})
        own = "P_TE0" if pol == "TE" else "P_TM0"
        best = max(scan, key=lambda r: r[own])
        centred = min(scan, key=lambda r: abs(r["y_um"]))
        yb = best["y_um"] * 1e-6
        launch = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                                 center=(0.0, yb), pol=pol, n_medium=n_ox)
        amps = cas.march(launch["a"])
        te_pol = np.real(od["TE_pol"][:, :N]).astype(float)
        phys = np.array([dv.physical(od, k) for k in range(len(od["EME_path"]))])
        pw = np.abs(amps) ** 2
        guided_te = np.sum(pw * phys * (te_pol >= 0.5), axis=1)
        guided_tm = np.sum(pw * phys * (te_pol < 0.5), axis=1)
        lf = leak[pol]
        res[pol] = {
            "best": best, "centred": centred, "scan": scan,
            "P_own": best[own], "P_own_corrected": best[own] * lf,
            "loss_dB": dv.loss_db(best[own]), "loss_dB_corrected": dv.loss_db(best[own] * lf),
            "loss_dB_centred_corrected": dv.loss_db(centred[own] * lf),
            "launch_top": [(int(i), float(abs(launch["a"][i]) ** 2), float(np.real(od["neff"][0, i])),
                            float(te_pol[0, i])) for i in np.argsort(-np.abs(launch["a"]))[:5]],
            "z_um": (cas.z[: len(amps)] * 1e6).tolist(),
            "guided_TE": guided_te.tolist(), "guided_TM": guided_tm.tolist(),
            "branch_power_top": {int(j): (pw[:, j]).tolist() for j in np.argsort(-pw.max(axis=0))[:6]},
        }
    res["segments"] = segment_study(od, cas, du.get_cladding_index())
    res["tip"] = tip_refinement(du, wl_nm, variant, path, od, cas, facet, ports, r_branch, n_ox, leak)
    res["seconds"]["total"] = time.time() - t0
    return res, (path, od, cas, facet)


# ------------------------------------------------------------- tip lattice


def branch_maps(path, od, du):
    """``slot[k][j]`` (solver slot of branch j at section k) and the gauge
    signs ``g[k][j]`` the geometry's phase equalisation applied - recomputed
    from the raw stored overlaps exactly as ``_equalize_overlap_phase`` does."""
    names = np.asarray(path._tracking_mode_names)
    n, N = names.shape
    slot = np.empty_like(names)
    for k in range(n):
        slot[k, names[k].astype(int)] = np.arange(N)
    g = np.ones((n, N))
    pts = [tuple(p) for p in od["EME_path"]]
    for i in range(n - 1):
        if pts[i] == pts[i + 1]:
            g[i + 1] = g[i]
            continue
        O = np.asarray(du.overlap[pts[i]][pts[i + 1]])[:N, :N]
        s = np.sign(np.real(O[slot[i], slot[i + 1]]))
        s[s == 0] = 1.0
        g[i + 1] = g[i] * s
    return slot, g


def joined_lumped(du, path, od, cas, du_tip, variant, lattice):
    """The device with its first stretch walked on ``lattice`` in the tip
    dataset: ``S_tip * J * S_main[k_join:]``, ``J`` the signed permutation
    between the two paths' branch bases at the (identical) join point."""
    import tip

    zj = tip.join_z(variant)
    fp = tip.fine_path(du_tip, variant, lattice, zj)
    fp._verbose = False
    od_f = fp.calc_output_data()
    cas_f = Cascade(fp, od_f)
    last = len(od_f["EME_path"]) - 1
    join_point = tuple(od_f["EME_path"][last])
    ks = [k for k in range(len(od["EME_path"])) if tuple(od["EME_path"][k]) == join_point]
    k_join = ks[0]                     # on the main 10 nm path this starts at z_join exactly
    N = cas.N
    slot_f, g_f = branch_maps(fp, od_f, du_tip)
    slot_m, g_m = branch_maps(path, od, du)
    names_m = np.asarray(path._tracking_mode_names)
    J = np.zeros((N, N))
    for b in range(N):
        bm = int(names_m[k_join][slot_f[last][b]])
        J[bm, b] = g_f[last][b] * g_m[k_join][bm]
    SJ = np.zeros((2 * N, 2 * N))
    SJ[:N, :N] = J
    SJ[N:, N:] = J.T
    P = cas.props()
    S_t = None
    for k in range(k_join, len(cas.dz)):
        S_t = P[k] if S_t is None else star(S_t, P[k])
        S_t = star(S_t, cas.mats[2 * k + 1])
    return star(star(cas_f.lumped(), SJ), S_t), {"k_join": k_join, "z_join_um": zj * 1e6,
                                                  "tip_sections": len(od_f["EME_path"])}


def tip_refinement(du, wl_nm, variant, path, od, cas, facet, ports, r_branch, n_ox, leak):
    """Loss with the tip walked at 10, 5 and 2.5 nm, and the smooth limit."""
    import tip

    tdir = tip.tip_dir(wl_nm)
    if not os.path.exists(os.path.join(tdir, "neff.pkl")):
        return None
    from dbeme import DataUpdater

    du_tip = DataUpdater(tdir, cache_size=1)
    du_tip._is_testmode = True                 # read only: a missing point is an error here
    N = cas.N
    out = {}
    try:
        for lat in tip.LATTICES:
            S, info = joined_lumped(du, path, od, cas, du_tip, variant, lat)
            row = dict(info)
            for pol in ("TE", "TM"):
                own = ports["TE0" if pol == "TE" else "TM0"]
                best = -1.0
                for yc in Y_SCAN:
                    a = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * dv.MFD,
                                        center=(0.0, yc), pol=pol, n_medium=n_ox)["a"]
                    fwd = (S @ np.concatenate([a, np.zeros(N, complex)]))[:N]
                    best = max(best, float(abs(fwd[own]) ** 2 * r_branch[own]))
                row[pol] = {"P": best, "loss_dB_corrected": dv.loss_db(best * leak[pol])}
            out[f"{lat*1e9:g}nm"] = row
    except Exception as exc:                     # the tip dataset may not cover this variant yet
        return {"error": repr(exc)}
    # first-order (in the step) extrapolation from the two finest lattices
    smooth = {}
    for pol in ("TE", "TM"):
        p5, p25 = out["5nm"][pol]["P"], out["2.5nm"][pol]["P"]
        p0 = 2 * p25 - p5
        smooth[pol] = {"P": p0, "loss_dB_corrected": dv.loss_db(p0 * leak[pol]),
                       "richardson_spread_dB": abs(dv.loss_db(p0) - dv.loss_db(p25))}
    out["smooth"] = smooth
    return out


SEGMENT_SCALES = np.round(np.exp(np.linspace(np.log(0.1), np.log(5.0), 31)), 4)


def fundamental_branch(od, k, pol, n_clad, N):
    neff, te = od["neff"][k, :N], np.real(od["TE_pol"][k, :N])
    keep = [m for m in range(N) if (te[m] >= 0.5) == (pol == "TE")
            and np.imag(neff[m]) < 0.05 and np.real(neff[m]) > n_clad]
    return max(keep, key=lambda m: np.real(neff[m])) if keep else None


def segment_study(od, cas, n_clad):
    """Phase 2: each segment alone, fundamental in -> fundamental out, and its
    length scan at zero mode solves (the adiabaticity knee)."""
    out = {}
    N = cas.N
    for seg, name in enumerate(SEGMENTS):
        if not np.any(cas.segment == seg):
            continue
        row = {"scales": SEGMENT_SCALES.tolist()}
        for pol in ("TE", "TM"):
            S, k0, k1 = cas.segment_block(seg)
            i = fundamental_branch(od, k0, pol, n_clad, N)
            j = fundamental_branch(od, k1, pol, n_clad, N)
            if i is None or j is None:
                row[pol] = None
                continue
            scan = []
            for s in SEGMENT_SCALES:
                Ss, _, _ = cas.segment_block(seg, s)
                scan.append(float(abs(Ss[j, i]) ** 2))
            row[pol] = {"T_design": float(abs(S[j, i]) ** 2), "in_branch": i, "out_branch": j,
                        "neff_in": float(np.real(od["neff"][k0, i])), "neff_out": float(np.real(od["neff"][k1, j])),
                        "scan": scan}
        out[name] = row
    return out


def main():
    wl = int(sys.argv[1])
    variants = sys.argv[2:] or ["bilayer", "conventional"]
    du = dv.open_dataset(wl, substrate=False)
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", f"device_{wl}.json")
    allres = json.load(open(fn)) if os.path.exists(fn) else {}
    for v in variants:
        res, _ = analyse_variant(du, wl, v)
        allres[v] = res
        for pol in ("TE", "TM"):
            r = res[pol]
            print(f"{wl} {v} {pol}: loss {r['loss_dB']:.3f} dB, leak-corrected {r['loss_dB_corrected']:.3f} dB "
                  f"(y {r['best']['y_um']:+.2f} um), centred {r['loss_dB_centred_corrected']:.3f} dB", flush=True)
        json.dump(allres, open(fn, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
