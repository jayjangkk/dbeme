"""Where the two-mask gauge rule went wrong on the bilayer path (warm, read-only).

    python studies/edge_coupler/gauge_trace.py [nm ...]

Upstream's ``_equalize_overlap_phase`` took one set of signs from
``diag(overlap_ab)`` and another from ``diag(overlap_ba)``.  At every
interface where the two raw diagonals of a branch disagree in sign, that
rule flips the branch's ``overlap_ba`` sign relative to its ``overlap_ab``
sign, and the flip persists until the next disagreement on the same branch.
This script captures the raw (tracked, not yet equalised) overlaps of the
bilayer path and records, per wavelength: the disagreement events, and for
the merge step (gap 20 -> 10 nm at ``w_high`` 790 nm) the branch TE0 couples
to most strongly, its events upstream and the parity they leave.  An odd
parity is what turned that coupling into reflection (reports/22 section 2.7).

Writes reports/output/edge/gauge_trace.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as an  # noqa: E402
import device as dv  # noqa: E402


def raw_overlaps(du, variant):
    path, _ = dv.build_path(du, variant)
    store = {}
    rule = type(path)._equalize_overlap_phase

    def hook(ab, ba, overlap_dict=0, index_mapping=0):
        store["ab"], store["ba"] = np.array(ab, copy=True), np.array(ba, copy=True)
        return rule(path, ab, ba, overlap_dict, index_mapping)

    path._equalize_overlap_phase = hook
    od = path.calc_output_data()
    return path, od, store["ab"], store["ba"]


def trace(wl):
    du = dv.open_dataset(wl, substrate=False)
    du._is_testmode = True
    path, od, ab, ba = raw_overlaps(du, "bilayer")
    N = ab.shape[1] // 2
    n_ox = float(du.get_cladding_index())
    sa = np.sign(np.real(np.diagonal(ab[:, :N, :N], axis1=1, axis2=2)))
    sb = np.sign(np.real(np.diagonal(ba[:, :N, :N], axis1=1, axis2=2)))
    events = np.argwhere(sa * sb < 0)          # (interface, branch)
    pts = [tuple(p) for p in od["EME_path"]]
    ks = [k for k in range(len(pts) - 1) if abs(pts[k][1] - 0.79e-6) < 1e-12
          and abs(pts[k][2] - 0.02e-6) < 1e-12 and pts[k] != pts[k + 1]]
    k = ks[0]
    te0 = an.fundamental_branch(od, k, "TE", n_ox, N)
    cpl = np.abs(ab[k, te0, :N]).copy()
    cpl[te0] = 0.0
    j = int(np.argmax(cpl))
    upstream = [int(i) for i, b in events if b == j and i < k]
    neff = np.asarray(od["neff"])
    return {
        "interfaces": int(len(ab)), "modes": int(N),
        "events": int(len(events)),
        "events_on_physical_branches": int(sum(dv.physical(od, i)[b] or dv.physical(od, i + 1)[b] for i, b in events)),
        "merge_step": {"interface": int(k), "point": [float(v) for v in pts[k]], "next": [float(v) for v in pts[k + 1]],
                       "TE0_branch": int(te0), "partner_branch": j,
                       "abs_O_ab_TE0_partner": float(abs(ab[k, te0, j])),
                       "abs_O_ba_partner_TE0": float(abs(ba[k, j, te0])),
                       "partner_neff_at_merge": [float(neff[k, j].real), float(neff[k, j].imag)],
                       "partner_raw_diagonals_agree_here": bool(sa[k, j] * sb[k, j] > 0),
                       "partner_events_upstream": [{"interface": i, "abs_ab_jj": float(abs(ab[i, j, j])),
                                                    "abs_ba_jj": float(abs(ba[i, j, j])),
                                                    "neff": [float(neff[i, j].real), float(neff[i, j].imag)]}
                                                   for i in upstream],
                       "parity_at_merge": "odd - opposite signs in ab and ba" if len(upstream) % 2
                       else "even - same sign"},
    }


def main():
    wls = [int(a) for a in sys.argv[1:]] or [1260, 1310, 1360]
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", "gauge_trace.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for wl in wls:
        out[str(wl)] = r = trace(wl)
        m = r["merge_step"]
        print(f"{wl}: {r['events']} events ({r['events_on_physical_branches']} touch a physical branch); merge step "
              f"{m['interface']}: TE0 -> branch {m['partner_branch']} |O| {m['abs_O_ab_TE0_partner']:.3f}, "
              f"{len(m['partner_events_upstream'])} upstream events, {m['parity_at_merge']}", flush=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
