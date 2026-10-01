"""Phase 1: facet overlaps of the double tip with a 4 um MFD beam (Fig. 3).

    python examples/run_solver_job.py --cpus 12-15 studies/edge_coupler/phase1.py <part> [stack]

``part``: ``wtip`` (Fig. 3a, Wtip 100-190 nm at g = 1.6 um), ``gap``
(Fig. 3b, g 0.5-1.9 um at Wtip = 130 nm), or ``design`` (the design point
on both stacks, with the fibre-offset scan and the fields for Fig. 2).
``stack``: ``full`` (Si substrate, as the paper's Fig. 2 draws it; default)
or ``ox``.  Each point is one lossy solve with the target at the local
fundamental; for 150 and 220 nm tips, TE and TM, three measures:

* (i) the paper's eq. (1), scalar, on the dominant transverse E;
* (ii) the conjugated Poynting power coupling into the even supermode;
* (iii) the launch the EME uses: ``|t_m|^2`` in the dataset's normalisation.

Results go to ``reports/output/edge/phase1_<part>_<stack>.json``.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
from dbeme.fde.assemble import assemble  # noqa: E402
from dbeme.platforms import wan2025_edge_coupler_dataset_info  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch, paper_overlap, power_coupling  # noqa: E402

WL = 1.31e-6
W0 = 0.5 * dv.MFD


def backend(stack):
    return wan2025_edge_coupler_dataset_info(WL, substrate=(stack == "full"))().get_fde_backend()


def solve(be, point):
    params = dict(point, wavelength=WL)
    target = be.target_for(params)
    md, conf = be.mode_data(params, target)
    N = len(md.neff)
    x, y, neff, te, E, H = assemble([md], N, 2, lossless=False)
    return md, conf, x, y, neff[0, :N], te[0, :N], E[0, :N], H[0, :N], target


def even_mode(md, conf, E, pol):
    """Highest-index mode of the right polarisation and x-even dominant field."""
    c = 0 if pol == "TE" else 1
    best, best_n = None, -np.inf
    for m in range(len(md.neff)):
        te = md.TE_pol[m]
        if (pol == "TE") != (te >= 0.5) or conf[m] < 0.01:
            continue
        e = E[m, c]
        par = np.real(np.sum(e * e[::-1, :]) / np.sum(e * e)) if np.sum(np.abs(e)) else 0
        if par > 0.5 and np.real(md.neff[m]) > best_n:
            best, best_n = m, np.real(md.neff[m])
    return best


def measure(be, point, centers=((0.0, 0.0),)):
    md, conf, x, y, neff, te, E, H, target = solve(be, point)
    n_ox = float(be.cross_section.cladding_index_at(WL))
    out = {"point": point, "target": target,
           "modes": [[float(np.real(n)), float(np.imag(n)), float(t), float(c)]
                     for n, t, c in zip(md.neff, md.TE_pol, conf)]}
    for pol in ("TE", "TM"):
        m = even_mode(md, conf, E, pol)
        if m is None:
            out[pol] = None
            continue
        rows = []
        for c in centers:
            launch = gaussian_launch(x, y, E, H, W0, center=c, pol=pol, n_medium=n_ox)
            rows.append({
                "center": list(c),
                "eq1": paper_overlap(x, y, E[m], W0, center=c, pol=pol),
                "poynting": power_coupling(x, y, E[m], H[m], W0, center=c, pol=pol, n_medium=n_ox),
                "launch": float(abs(launch["a"][m]) ** 2),
                "launch_all_confined": float(np.sum(np.abs(launch["a"][conf >= 0.01]) ** 2)),
            })
        out[pol] = {"mode": int(m), "neff": [float(np.real(md.neff[m])), float(np.imag(md.neff[m]))],
                    "conf": float(conf[m]), "rows": rows}
    return out, (x, y, E, H, md, conf)


def main():
    part = sys.argv[1]
    stack = sys.argv[2] if len(sys.argv) > 2 else "full"
    be = backend(stack)
    res = {"part": part, "stack": stack, "wavelength": WL, "w0": W0, "points": []}
    path = os.path.join(dv.ROOT, "reports", "output", "edge", f"phase1_{part}_{stack}.json")
    t0 = time.time()
    if part == "wtip":
        pts = [(h, w) for w in np.arange(100, 191, 10) for h in (150, 220)]
        for h, w in pts:
            p = {"w_low": w * 1e-9, "w_high": 0.0, "gap": 1.6e-6} if h == 150 else \
                {"w_low": 0.0, "w_high": w * 1e-9, "gap": 1.6e-6}
            r, _ = measure(be, p)
            r.update(height=h, wtip_nm=float(w), g_um=1.6)
            res["points"].append(r)
            print(h, w, {k: r[k]["rows"][0] if r[k] else None for k in ("TE", "TM")}, f"{time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(path, "w"), indent=1)
    elif part == "gap":
        pts = [(h, g) for g in np.round(np.arange(0.5, 1.91, 0.1), 2) for h in (150, 220)]
        for h, g in pts:
            p = {"w_low": 0.13e-6, "w_high": 0.0, "gap": g * 1e-6} if h == 150 else \
                {"w_low": 0.0, "w_high": 0.13e-6, "gap": g * 1e-6}
            r, _ = measure(be, p)
            r.update(height=h, wtip_nm=130.0, g_um=float(g))
            res["points"].append(r)
            print(h, g, {k: r[k]["rows"][0] if r[k] else None for k in ("TE", "TM")}, f"{time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(path, "w"), indent=1)
    elif part == "design":
        centers = [(0.0, yc * 1e-6) for yc in np.round(np.arange(-0.8, 0.81, 0.1), 2)]
        for h in (150, 220):
            p = {"w_low": 0.13e-6, "w_high": 0.0, "gap": 1.6e-6} if h == 150 else \
                {"w_low": 0.0, "w_high": 0.13e-6, "gap": 1.6e-6}
            r, (x, y, E, H, md, conf) = measure(be, p, centers)
            r.update(height=h)
            res["points"].append(r)
            # fields of the two even supermodes for the Fig. 2 counterpart
            np.savez_compressed(
                os.path.join(dv.ROOT, "reports", "output", "edge", f"fields_tip{h}_{stack}.npz"),
                x=np.real(x), y=np.real(y),
                n=np.real(be.index_profile(dict(p, wavelength=WL))),
                **{f"E_{pol}": np.abs(E[r[pol]["mode"]]).astype(np.float32)
                   for pol in ("TE", "TM") if r[pol]})
            print(h, {k: max(rr["eq1"] for rr in r[k]["rows"]) if r[k] else None for k in ("TE", "TM")}, flush=True)
            json.dump(res, open(path, "w"), indent=1)
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
