"""The near-cutoff tip on a finer width lattice, joined to the main path.

    python examples/run_solver_job.py --cpus 0-7 studies/edge_coupler/tip.py build <nm> [variant ...]

At the facet the tip supermode clears the oxide by 0.004 in index, and a
10 nm width step there sheds a few percent of TE (the section-5.8 check on
L1: 0.844 cached against 0.872 direct).  The ``_tip`` dataset carries the
same platform with the width axes refined to 2.5 nm over 90-260 nm, so the
first stretch of the path - from the facet to the point where the arm is
``W_JOIN`` wide - can be walked at 10, 5 and 2.5 nm on one set of solves,
and extrapolated to a smooth taper.

The join is exact: both datasets solve the join point with identical
settings, so their raw modes coincide; the only bookkeeping is each path's
branch order and gauge signs at that point (a signed permutation).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
from dbeme import DataUpdater  # noqa: E402

W_JOIN = 0.20e-6
LATTICES = (10e-9, 5e-9, 2.5e-9)


def tip_dir(wl_nm):
    return os.path.join(dv.ROOT, "datasets", f"Si_bilevel_pair_220_150nm_ox_tip_{int(wl_nm)}")


def ensure_dataset(wl_nm):
    d = tip_dir(wl_nm)
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, "dataset_info.py")
    if not os.path.exists(fn):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(
                f'"""Dataset: the Wan & Wang edge coupler tip on a 2.5 nm width lattice, {wl_nm} nm.\n\n'
                "The main dataset's platform with both width axes refined to 2.5 nm over\n"
                "90-260 nm (tip_step): the first stretch of the path, where the tip mode\n"
                "sits near cut-off, walked on a finer staircase (studies/edge_coupler/tip.py).\n"
                '"""\n\n'
                "from dbeme.platforms import wan2025_edge_coupler_dataset_info\n\n"
                f"WAVELENGTH = {wl_nm * 1e-9!r}\n\n"
                "DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=False, tip_step=2.5e-9)\n")
    return d


def join_z(variant, lattice=dv.CELL):
    """``z`` where the lattice-snapped arm first reaches ``W_JOIN``."""
    funcs, total = dv.path_functions(variant=variant, lattice=lattice)
    z = np.linspace(0.0, total, int(round(total / 25e-9)) + 1)
    key = "w_low" if variant == "bilayer" else "w_high"
    w = np.asarray(funcs[key](z))
    k = int(np.flatnonzero(np.isclose(w, W_JOIN, atol=1e-12))[0])
    return float(z[k])


def fine_functions(variant, lattice, z_join):
    """The path to ``z_join`` on ``lattice``, ending exactly on the join point."""
    fine, _ = dv.path_functions(variant=variant, lattice=lattice)
    main, _ = dv.path_functions(variant=variant)
    out = {}
    for name in ("w_low", "w_high", "gap"):
        def f(z, fi=fine[name], ma=main[name]):
            z = np.asarray(z, dtype=float)
            return np.where(z < z_join - 1e-12, fi(z), ma(np.full_like(z, z_join)))
        out[name] = f
    return out


def fine_path(du, variant, lattice, z_join):
    funcs = fine_functions(variant, lattice, z_join)
    return dv.EdgeCouplerPath(du, funcs, total_length=z_join, resolution=int(round(z_join / 25e-9)) + 1,
                              max_section_length=2e-6, verbose=True)


def build(wl_nm, variants):
    du = DataUpdater(ensure_dataset(wl_nm), cache_size=2)
    for v in variants:
        zj = join_z(v)
        for lat in LATTICES[::-1]:          # the finest first: it holds the others' points
            path = fine_path(du, v, lat, zj)
            path.calc_output_data()
            print(f"{v}: lattice {lat*1e9:.1f} nm to z = {zj*1e6:.3f} um done, {len(du.neff)} points", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]), sys.argv[3:] or ["bilayer", "conventional"])
