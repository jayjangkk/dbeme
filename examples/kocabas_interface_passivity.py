"""Per-interface passivity along a Kocabas sweep path, from the cached overlaps.

    python kocabas_interface_passivity.py <dataset suffix> <gap_nm> [<gap_nm> ...]

For every interface of the path, the largest total power (forward plus
backward) returned for unit power in a physical input mode, and the six worst
interfaces with the two grid points they join.  A clean path sits within a
few percent of 1; a basis-membership break (CLAUDE.md section 5.13a) shows
as one interface far above it, and the cascade's budget then carries a
reflection or a negative deficit out of line with its neighbours.  Runs warm:
every point of the path must already be cached."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import demo_kocabas_converter as demo                              # noqa: E402
from demo_plasmonic_converter import EME, physical                 # noqa: E402
from em_simulation import DataUpdater                              # noqa: E402

suffix, gaps = sys.argv[1], [int(v) for v in sys.argv[2:]]
demo.use_cell(5.0, suffix=suffix, axes="half_slot")
du = DataUpdater(demo.DATASET)
for g in gaps:
    path, _ = demo.device(du, w_gap_nm=g)
    od = path.calc_output_data()
    eme = EME(path, force_unitary=False); eme.calc_Smatrix()
    S = eme.propagator._calc_interface_Smatrix()
    N = od["neff"].shape[1] // 2
    pts = od["EME_path"]
    rows = []
    for k in range(S.shape[0]):
        ph = np.flatnonzero(physical(od, k))
        col = np.abs(S[k]) ** 2
        power = col[:, ph].sum(axis=0)                    # total out (fwd + bwd) per physical input
        refl = col[N:, ph].sum(axis=0)
        j = int(np.argmax(power))
        rows.append((float(power[j]), float(refl[j]), k, int(ph[j])))
    worst = sorted(rows, reverse=True)[:6]
    print(f"=== gap {g} nm: {S.shape[0]} interfaces; max physical-input power {worst[0][0]:.4f}; "
          f"interfaces > 1.001: {sum(1 for r in rows if r[0] > 1.001)}; max reflection {max(r[1] for r in rows):.4f}")
    for power, refl, k, j in worst:
        a, b = pts[k], pts[k + 1]
        print(f"   interface {k:3d}: ({a[0]*1e9:5.1f}, {a[1]*1e9:6.1f}) -> ({b[0]*1e9:5.1f}, {b[1]*1e9:6.1f}) nm | "
              f"input mode {j} n={od['neff'][k, j]:.4f} | power out {power:.4f} reflected {refl:.4f}")
