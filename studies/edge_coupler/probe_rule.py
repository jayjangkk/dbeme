"""Phase 0.3 probe, second pass: the dataset backend itself, target rule on.

    python studies/edge_coupler/probe_rule.py <tag> <wavelength_nm> [substrate 0/1] [points...]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbeme.platforms import wan2025_edge_coupler_dataset_info  # noqa: E402
from probe_modes import POINTS, describe  # noqa: E402


def main():
    tag = sys.argv[1]
    wl = float(sys.argv[2]) * 1e-9
    substrate = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    names = sys.argv[4:] or list(POINTS)
    info = wan2025_edge_coupler_dataset_info(wl, substrate=substrate)()
    be = info.get_fde_backend()
    cs = be.cross_section
    print(f"grid {len(be.x)} x {len(be.y)}, N {be.num_modes}, rule: {getattr(be, 'target_rule_name', be.target_neff)}", flush=True)
    out = {"wavelength": wl, "substrate": substrate, "points": {}}
    for name in names:
        params = dict(POINTS[name], wavelength=wl)
        t0 = time.time()
        target = be.target_for(params)
        t1 = time.time()
        md, conf = be.mode_data(params, target)
        dt = time.time() - t1
        rows = describe(be, cs, md, conf, params)
        out["points"][name] = {"target": target, "t_target": t1 - t0, "seconds": dt, "modes": rows}
        print(f"\n== {name}  target {target}  ({t1 - t0:.1f} s)  solve {dt:.1f} s", flush=True)
        for r in rows:
            print(f"{r['i']:3d} n={r['neff_re']:.5f}{r['neff_im']:+.2e}j conf={r['conf']:.3f} "
                  f"TE={r['te']:.2f} sub={r['f_sub']:.3f} pml={r['f_pml']:.3f} air={r['f_air']:.3f} "
                  f"par={r['parity_ex']:+.3f}", flush=True)
    with open(os.path.join(ROOT, "reports", "output", "edge", f"probe_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
