"""End-to-end check of the analysis pipeline on the coarse mini dataset.

    python examples/run_solver_job.py --cpus 28-31 studies/edge_coupler/pipeline_check.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme import DataUpdater  # noqa: E402

du = DataUpdater(os.path.join(dv.ROOT, "cache", "edge_mini_1310"), cache_size=2)
out = {}
for variant in ("bilayer", "conventional"):
    t0 = time.time()
    res, (path, od, cas, facet) = an.analyse_variant(du, "mini1310", variant)
    print(variant, "sections", res["sections"], "abrupt", path.abrupt_interfaces, f"{time.time()-t0:.0f}s")
    for pol in ("TE", "TM"):
        r = res[pol]
        print(f"  {pol}: loss {r['loss_dB']:.3f} dB at y {r['best']['y_um']:+.2f}; centred {r['centred']}")
        print("   launch top", r["launch_top"])
        print("   guided TE along z (every 20):", [round(v, 3) for v in r["guided_TE"][::20]])
        print("   guided TM along z (every 20):", [round(v, 3) for v in r["guided_TM"][::20]])
    print("  ports", res["ports"], "output neff", res["output_neff"], "r_out", res["r_out"])
    # the march's last point must equal the lumped matrix's output
    import numpy as np
    from dbeme.propagator.fiber import gaussian_launch
    L = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 2e-6, pol="TE",
                        n_medium=du.get_cladding_index())
    S = cas.lumped()
    lump = (S @ np.concatenate([L["a"], 0 * L["a"]]))[: cas.N]
    march = cas.march(L["a"])[-1]
    print("  march vs lumped max|d|:", float(np.max(np.abs(lump - march))))
    # length scaling: identity scales reproduce the lumped matrix
    print("  scale=1 vs lumped:", float(np.max(np.abs(cas.lumped(np.ones(6)) - S))))
    out[variant] = res
json.dump(out, open(os.path.join(dv.ROOT, "cache", "edge", "pipeline_check.json"), "w"), indent=1, default=float)
