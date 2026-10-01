"""F5: does the facet answer depend on the window or the PML?

    python examples/run_solver_job.py --cpus 16-23 studies/edge_coupler/window_check.py

The design facet (Wtip 130 nm, g 1.6 um, 150 nm tips) at 1310 nm on the
full stack, solved with the platform's window and PML and with each of
them enlarged in turn: ``n_eff`` of the even TE/TM supermodes and the
paper's eq. (1) overlap.  Also the oxide-only stack (the EME basis), whose
tip modes pick up a spurious loss from the bottom PML that the leakage
correction replaces.  Criterion (tasks/18): overlap moves < 1e-3, Im n
< 10 %.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
import phase1  # noqa: E402
from dbeme.platforms import wan2025_edge_coupler_dataset_info  # noqa: E402
from dbeme.propagator.fiber import paper_overlap  # noqa: E402

WL = 1.31e-6
POINT = {"w_low": 0.13e-6, "w_high": 0.0, "gap": 1.6e-6}
CASES = {
    "full: platform": dict(substrate=True),
    "full: window +-6.5 um": dict(substrate=True, half_width=6.5e-6),
    "full: side/top PML 1.0 um": dict(substrate=True, lateral_pml=1.0e-6, top_pml=1.0e-6),
    "full: Si PML 0.5 um": dict(substrate=True, bottom_pml=0.5e-6),
    "oxide: platform": dict(substrate=False),
    "oxide: window +-6.5 um": dict(substrate=False, half_width=6.5e-6),
    "oxide: PML 1.0 um": dict(substrate=False, lateral_pml=1.0e-6, top_pml=1.0e-6),
}


def main():
    out = {}
    fn = os.path.join(dv.ROOT, "reports", "output", "edge", "window_check.json")
    for name, kw in CASES.items():
        t0 = time.time()
        be = wan2025_edge_coupler_dataset_info(WL, **kw)().get_fde_backend()
        md, conf, x, y, neff, te, E, H, target = phase1.solve(be, POINT)
        row = {"grid": [len(be.x), len(be.y)], "seconds": None}
        for pol in ("TE", "TM"):
            m = phase1.even_mode(md, conf, E, pol)
            row[pol] = None if m is None else {
                "neff": [float(md.neff[m].real), float(md.neff[m].imag)],
                "eq1": paper_overlap(x, y, E[m], 0.5 * dv.MFD, pol=pol),
            }
        row["seconds"] = time.time() - t0
        out[name] = row
        print(name, json.dumps(row), flush=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
