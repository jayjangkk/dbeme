"""Where a PML solve of this platform spends its time (a one-off measurement).

    python examples/run_solver_job.py --cpus 24-31 studies/edge_coupler/profile_solve.py
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402
from dbeme.fde.pml import _M_TO_UM  # noqa: E402
from dbeme.platforms import wan2025_edge_coupler_dataset_info  # noqa: E402

be = wan2025_edge_coupler_dataset_info(1.31e-6, substrate=False)().get_fde_backend()
params = {"w_low": 0.15e-6, "w_high": 0.14e-6, "gap": 1.6e-6, "wavelength": 1.31e-6}
from EMpy_gpu.modesolvers.FD import VFDModeSolver  # noqa: E402

cs = be.cross_section


def epsfunc(x_um, y_um):
    n = cs.index(np.real(np.asarray(x_um, dtype=complex)) / _M_TO_UM,
                 np.real(np.asarray(y_um, dtype=complex)) / _M_TO_UM, params)
    return n.astype(complex) ** 2


t = time.time()
target = be.target_for(params)
print("target", target, f"{time.time()-t:.1f}s", flush=True)
t = time.time()
solver = VFDModeSolver(1.31, be.x * _M_TO_UM, be.y * _M_TO_UM, epsfunc, be.boundary)
A = solver.build_matrix().tocsc()
print("build_matrix", f"{time.time()-t:.1f}s", A.shape, A.nnz, flush=True)
k0 = 2 * np.pi / 1.31
sigma = (target * k0) ** 2
M = (A - sigma * sp.identity(A.shape[0], format="csc")).tocsc()
for spec in ("COLAMD", "MMD_AT_PLUS_A", "MMD_ATA"):
    t = time.time()
    lu = spla.splu(M, permc_spec=spec)
    print(f"splu {spec}: {time.time()-t:.1f}s, fill L {lu.L.nnz/1e6:.1f}M U {lu.U.nnz/1e6:.1f}M", flush=True)
t = time.time()
vals, vecs = spla.eigs(A, k=40, which="LM", sigma=sigma, tol=be.accuracy, ncv=160)
print(f"eigs default (ncv 160): {time.time()-t:.1f}s", flush=True)
lu = spla.splu(M, permc_spec="MMD_AT_PLUS_A")
op = spla.LinearOperator(M.shape, matvec=lu.solve, dtype=complex)
for ncv in (160, 100):
    t = time.time()
    v2, _ = spla.eigs(A, k=40, which="LM", sigma=sigma, tol=be.accuracy, ncv=ncv, OPinv=op)
    d = np.max(np.abs(np.sort_complex(np.sqrt(v2) / k0) - np.sort_complex(np.sqrt(vals) / k0)))
    print(f"eigs MMD_AT_PLUS_A OPinv ncv {ncv}: {time.time()-t:.1f}s (+ its splu), max|dn| {d:.2e}", flush=True)
t = time.time()
md, conf = be.mode_data(params, target)
print(f"full mode_data: {time.time()-t:.1f}s", flush=True)
