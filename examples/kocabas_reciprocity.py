"""Where the cascade's reciprocity defect comes from, on a cached Kocabas path.

    python kocabas_reciprocity.py <dataset suffix> [gap_nm]     (report 13 section 11)

In standard form S = [[R1, T12], [T21, R2]] a reciprocal medium gives S = S^T
in the unconjugated-normalised basis.  Per interface the output-side
projection makes T21 = T12^T algebraically, so any defect must come from the
reflection blocks (symmetric only to the order of the unrepresented field in
a truncated basis) and from the pseudo-inverse cutoff, and it accumulates
through the cascade's multiple reflections.  This script measures:

  1. per interface: |T12 - T21^T|, |R1 - R1^T|, |R2 - R2^T| (max entry, and
     on the physical block), and the same after the cutoff is effectively
     removed (rcond 1e-9) and with the input-side projection;
  2. along the cascade: max |S - S^T| of the accumulated S after every stage,
     full and physical block, so that growth can be told from a few bad
     interfaces;
  3. the ranking of interfaces by their own reflection asymmetry.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples"))
import demo_kocabas_converter as demo                                        # noqa: E402
from demo_plasmonic_converter import EME, physical, standard_form            # noqa: E402
from em_simulation import DataUpdater                                        # noqa: E402
from em_simulation import matrix_calculation_tool as mct                     # noqa: E402
from em_simulation.propagator.single_propagator.single_eme import SingleEME  # noqa: E402

suffix = sys.argv[1]
gap = int(sys.argv[2]) if len(sys.argv) > 2 else None
demo.use_cell(5.0, suffix=suffix, axes="half_slot")
du = DataUpdater(demo.DATASET)
path, _ = demo.device(du, w_gap_nm=gap)
od = path.calc_output_data()
N = od["neff"].shape[1] // 2
L = len(od["EME_path"])


def blocks(S):
    """Standard form of one (2N x 2N) scattering matrix: R1, T12, T21, R2."""
    M = standard_form(S)
    return M[:N, :N], M[:N, N:], M[N:, :N], M[N:, N:]


def asym(A, mask=None):
    D = np.abs(A - A.T)
    if mask is not None:
        D = D[np.ix_(mask, mask)]
    return float(D.max()) if D.size else 0.0


def build(rcond, projection):
    SingleEME.INTERFACE_RCOND = rcond
    SingleEME.INTERFACE_PROJECTION = projection
    eme = EME(path, force_unitary=False); eme.calc_Smatrix()
    return eme.propagator._calc_interface_Smatrix(), eme.propagator._calc_propagation_Smatrix()


out = {"dataset": os.path.basename(demo.DATASET), "gap_nm": gap, "sections": L, "variants": {}}
for label, rcond, projection in (("output rcond 1e-2 (default)", 1e-2, "output"),
                                 ("output rcond 1e-9", 1e-9, "output"),
                                 ("input rcond 1e-2", 1e-2, "input")):
    S_int, S_prop = build(rcond, projection)
    per = []
    for k in range(S_int.shape[0]):
        R1, T12, T21, R2 = blocks(S_int[k])
        pa, pb = physical(od, k), physical(od, k + 1)
        per.append(dict(k=k, T=float(np.abs(T12 - T21.T).max()), R1=asym(R1), R2=asym(R2),
                        R1_phys=asym(R1, pa), R2_phys=asym(R2, pb)))
    # cascade: prop_0, int_0, prop_1, int_1, ... as the runner orders them
    S = None; growth = []
    for k in range(S_int.shape[0]):
        for stage, M in (("prop", S_prop[k]), ("int", S_int[k])):
            S = M.copy() if S is None else mct._redheffer_star_product(S, M)
            if stage == "int":
                Ms = standard_form(S)
                phys = np.concatenate([physical(od, 0), physical(od, k + 1)])
                growth.append(dict(k=k, full=asym(Ms), phys=asym(Ms, phys)))
    Ms = standard_form(S)
    i_in = int(np.argmax(np.where(physical(od, 0), od["neff"][0, :N].real, -9)))
    j_out = int(np.argmax(np.where(physical(od, -1), od["neff"][-1, :N].real, -9)))
    channel = float(abs(Ms[i_in, N + j_out] - Ms[N + j_out, i_in]))
    worst = sorted(per, key=lambda r: -max(r["R1"], r["R2"]))[:5]
    out["variants"][label] = dict(channel=channel, total_full=growth[-1]["full"], total_phys=growth[-1]["phys"],
                                  max_T=max(r["T"] for r in per), max_R=max(max(r["R1"], r["R2"]) for r in per),
                                  max_R_phys=max(max(r["R1_phys"], r["R2_phys"]) for r in per),
                                  worst=worst, growth=growth[::max(1, len(growth) // 25)] + [growth[-1]])
    print(f"=== {label}: channel |T12-T21| {channel:.2e}; total |S-S^T| full {growth[-1]['full']:.2e}, physical block {growth[-1]['phys']:.2e}")
    print(f"    per interface: max |T12-T21^T| {out['variants'][label]['max_T']:.1e}; max reflection asymmetry {out['variants'][label]['max_R']:.2e} "
          f"(physical block {out['variants'][label]['max_R_phys']:.2e})")
    print("    worst interfaces by reflection asymmetry:", [(r["k"], f"{max(r['R1'], r['R2']):.1e}") for r in worst])
    g = out["variants"][label]["growth"]
    print("    growth of |S-S^T| (physical block) along the cascade:", " ".join(f"{r['k']}:{r['phys']:.1e}" for r in g[::max(1, len(g)//8)]))
json.dump(out, open(os.path.join(ROOT, "reports", "output", f"kocabas_reciprocity_{suffix}.json"), "w"), indent=1)
print("->", f"reports/output/kocabas_reciprocity_{suffix}.json")
