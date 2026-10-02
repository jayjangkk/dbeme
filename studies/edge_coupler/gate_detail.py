"""Localise the gate's reciprocity defect and refine F3/F4 (warm, no solves).

    python studies/edge_coupler/gate_detail.py [nm]

The reciprocity rows are computed with ``SingleEME.INTERFACE_RECIPROCAL``
off: the projection (on for lossy bases since 2026-10-01) makes the cascade
reciprocal by construction, and the defect it removes is what is localised.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product as star  # noqa: E402
from dbeme.propagator.fiber import gaussian_fields, gaussian_launch, physical_interior  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import interface_switches  # noqa: E402


def cascade_from(cas, k0):
    P = cas.props()
    S = None
    for k in range(k0, len(cas.dz)):
        S = P[k] if S is None else star(S, P[k])
        S = star(S, cas.mats[2 * k + 1])
    return S


def main():
    wl = int(sys.argv[1]) if len(sys.argv) > 1 else 1310
    du = dv.open_dataset(wl, substrate=False)
    du._is_testmode = True
    n_ox = float(du.get_cladding_index())
    out = {}
    for variant in (sys.argv[2:] or ["bilayer"]):
        res = {}
        for cap in ("auto", False):
            with interface_switches(INTERFACE_COLUMN_CAP=cap, INTERFACE_RECIPROCAL=False):
                path, _ = dv.build_path(du, variant)
                od = path.calc_output_data()
                cas = an.Cascade(path, od)
            N = cas.N
            ports = dv.output_ports(od)
            rows = {}
            for z0 in (0.0, 5.0, 13.8, 25.0, 45.0, 68.0, 76.0):
                k0 = int(np.argmin(np.abs(cas.z[:-1] - z0 * 1e-6)))
                S = cascade_from(cas, k0)
                row = {}
                for pol in ("TE", "TM"):
                    i = an.fundamental_branch(od, k0, pol, n_ox, N)
                    j = ports["TE0" if pol == "TE" else "TM0"]
                    f, b = S[j, i], S[N + i, N + j]
                    row[pol] = float(abs(f - b) / abs(f))
                rows[f"{z0:g}"] = row
            res[f"cap={cap}"] = rows
        out[variant] = {"reciprocity_from_z_um": res}
        # membership: the rule's fundamental (any polarisation) vs the top stored physical mode
        rule = du.backend.target_rule
        gaps = []
        for p in dict.fromkeys(tuple(q) for q in od["EME_path"]):
            n = np.asarray(du.neff[p])[:N]
            phys = [m for m in range(N) if n[m].imag < 0.05]
            gaps.append(float(rule(dict(zip(du.parameter_names, p))) - max(np.real(n[phys]))))
        out[variant]["target_minus_top_any"] = {"max": max(gaps), "min": min(gaps), "median": float(np.median(gaps))}
        # F3/F4 at the facet with strict definitions
        facet = an.endpoint(du, tuple(od["EME_path"][0]), f"{wl}_{variant}_facet")
        E, H, x, y = facet["E"].astype(complex), facet["H"].astype(complex), facet["x"], facet["y"]
        par = []
        for m in range(N):
            c = 0 if facet["TE_pol"][m] >= 0.5 else 1
            e = E[m, c]
            par.append(float(np.real(np.sum(e * e[::-1, :]) / np.sum(e * e))))
        par = np.array(par)
        inside = physical_interior(x, y)
        dx = np.abs(np.diff(np.real(x)))
        dy = np.abs(np.diff(np.real(y)))
        w = np.outer(np.r_[dx, dx[-1]], np.r_[dy, dy[-1]]) * inside
        f34 = {}
        for pol in ("TE", "TM"):
            L = gaussian_launch(x, y, E, H, 0.5 * dv.MFD, center=(0.0, 0.0), pol=pol, n_medium=n_ox)
            a, s = L["a"], L["s"]
            Eg, _ = gaussian_fields(x, y, 0.5 * dv.MFD, pol=pol, n_medium=n_ox)
            c = 0 if pol == "TE" else 1
            i = an.fundamental_branch(od, 0, pol, n_ox, N)
            order = np.argsort(-np.real(facet["neff"]))
            resid = {}
            for label, sel in (("fundamental", [i]),
                               ("Re n > 1.42", [m for m in range(N) if np.real(facet["neff"][m]) > 1.42]),
                               ("Re n > 1.40", [m for m in range(N) if np.real(facet["neff"][m]) > 1.40]),
                               ("all 40", list(range(N)))):
                rec = np.tensordot(s[sel], E[sel], axes=(0, 0))[c]
                resid[label] = float(np.sum(np.abs(Eg[0, c] - rec) ** 2 * w) / np.sum(np.abs(Eg[0, c]) ** 2 * w))
            f34[pol] = {
                "strict_odd_power(p<-0.9)": float(np.sum(np.abs(a[par < -0.9]) ** 2)),
                "mixed_parity_power(|p|<0.9)": float(np.sum(np.abs(a[np.abs(par) < 0.9]) ** 2)),
                "n_mixed_parity_modes": int(np.sum(np.abs(par) < 0.9)),
                "reconstruction_residual": resid,
            }
        out[variant]["facet"] = f34
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(dv.ROOT, "reports", "output", "edge", f"gate_detail_{wl}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
