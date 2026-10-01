"""Exercise the tip join on the coarse mini platform (pipeline check only).

    python examples/run_solver_job.py --cpus 24-31 studies/edge_coupler/tip_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
import tip  # noqa: E402
from dbeme import DataUpdater  # noqa: E402

mini = os.path.join(dv.ROOT, "cache", "edge_mini_1310")
mtip = os.path.join(dv.ROOT, "cache", "edge_mini_tip_1310")
os.makedirs(mtip, exist_ok=True)
with open(os.path.join(mtip, "dataset_info.py"), "w") as f:
    f.write("from dbeme.platforms import wan2025_edge_coupler_dataset_info\n"
            "WAVELENGTH = 1.31e-6\n"
            "DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=False, mode_numbers=16,"
            " base=100e-9, fine=20e-9, y_fine=(-0.41e-6, 0.39e-6), axis_steps={'w': 40e-9, 'gap': 80e-9},"
            " tip_step=2.5e-9)\n")
du = DataUpdater(mini, cache_size=2)
dt = DataUpdater(mtip, cache_size=2)
variant = "bilayer"
zj = tip.join_z(variant)
for lat in tip.LATTICES:
    tip.fine_path(dt, variant, lat, zj).calc_output_data()
path, _ = dv.build_path(du, variant)
od = path.calc_output_data()
cas = an.Cascade(path, od)
# the mini main path walks the join point only if its coarse axis holds it;
# the check here is that the join machinery runs and returns a sane matrix
try:
    for lat in tip.LATTICES:
        S, info = an.joined_lumped(du, path, od, cas, dt, variant, lat)
        N = cas.N
        col = np.sum(np.abs(S[:, :N]) ** 2, axis=0)
        print(lat, info, "max column power", float(col.max()))
except Exception as exc:
    import traceback
    traceback.print_exc()
