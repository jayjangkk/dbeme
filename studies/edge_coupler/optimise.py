"""Phase 4: optimisation of the edge coupler.

    python studies/edge_coupler/optimise.py lengths [nm ...]        # 4.1, zero mode solves
    python examples/run_solver_job.py --cpus 8-15 studies/edge_coupler/optimise.py facet   # 4.2

**4.1 - the free axes.**  The six segment lengths move no cross section: on
a warm dataset every evaluation re-cascades the cached interfaces with
rescaled propagation (``analyse.Cascade``), so L-BFGS-B runs on the full
band objective at zero mode solves.  Gradients are central differences in
log-length (each costs two cascades, ~1 s).  Bounds: [0.25, 3] x Table 1.

**4.2 - the facet, by CMA-ES.**  ``Wtip`` and ``g`` set the fibre overlap;
each candidate is one lossy solve of the facet cross section (full stack),
objective the worse of the TE and TM power couplings at the best vertical
fibre offset.  A compact (mu/mu_w, lambda)-CMA-ES (Hansen's ``purecma``
recipe) - no dependency added.

Objective of 4.1, per wavelength: the worse of TE and TM loss (each at its
best fibre offset, substrate-leakage corrected), plus 0.5 x PDL; averaged
over the wavelengths given.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse as an  # noqa: E402
import device as dv  # noqa: E402
from dbeme.platforms import WAN2025_DESIGN  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch, power_coupling  # noqa: E402

OUT = os.path.join(dv.ROOT, "reports", "output", "edge")
LENGTH_KEYS = ("L1", "L2", "L3", "Lt", "Lm", "L_out")
PDL_WEIGHT = 0.5


# ------------------------------------------------------------ 4.1 lengths


class BandModel:
    """Everything a length evaluation needs, precomputed per wavelength."""

    def __init__(self, wl_nm, variant="bilayer", du=None):
        self.wl = wl_nm
        du = du or dv.open_dataset(wl_nm, substrate=False)
        res, (path, od, cas, facet) = an.analyse_variant(du, wl_nm, variant)
        self.cas, self.od = cas, od
        self.N = cas.N
        self.ports = res["ports"]
        names = path._tracking_mode_names[-1]
        out = an.endpoint(du, tuple(od["EME_path"][-1]), f"{wl_nm}_{variant}_output", need_fields=False)
        self.r = np.empty(self.N)
        for k in range(self.N):
            self.r[int(names[k])] = out["r"][k]
        n_ox = float(du.get_cladding_index())
        self.launch = {pol: np.array([gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"],
                                                      0.5 * dv.MFD, center=(0.0, yc), pol=pol,
                                                      n_medium=n_ox)["a"] for yc in an.Y_SCAN])
                       for pol in ("TE", "TM")}
        # leakage integrand per section, so it scales with the segment lengths
        self.k0 = 2 * np.pi / (wl_nm * 1e-9)
        self.dim = {}
        _, samples = an.leakage_factor(du, cas, wl_nm, variant)
        zmid = 0.5 * (cas.z[:-1] + cas.z[1:])
        for pol in ("TE", "TM"):
            s = sorted((samples or {}).get(pol, []))
            if s:
                zs = np.array([v[0] for v in s])
                d = np.array([v[1] - v[2] for v in s])
                self.dim[pol] = np.where(zmid <= zs.max(), np.interp(zmid, zs, d), 0.0)
            else:
                self.dim[pol] = np.zeros_like(zmid)
        self.baseline = res

    def losses(self, scales):
        S = self.cas.lumped(scales)
        N = self.N
        s_sec = np.asarray(scales)[self.cas.segment]
        out = {}
        for pol in ("TE", "TM"):
            own = self.ports["TE0" if pol == "TE" else "TM0"]
            fwd = self.launch[pol] @ S[:N, :N].T                     # (n_y, N)
            p = np.abs(fwd[:, own]) ** 2 * self.r[own]
            leak = np.exp(-2 * self.k0 * np.sum(self.dim[pol] * self.cas.dz * s_sec))
            out[pol] = dv.loss_db(p.max() * leak)
        return out


def objective(models, log_scales):
    scales = np.exp(log_scales)
    per = [m.losses(scales) for m in models]
    j = np.mean([max(p["TE"], p["TM"]) + PDL_WEIGHT * abs(p["TE"] - p["TM"]) for p in per])
    return float(j), per


def optimise_lengths(wls):
    t0 = time.time()
    models = [BandModel(w) for w in wls]
    t_setup = time.time() - t0
    design = np.array([WAN2025_DESIGN[k] for k in LENGTH_KEYS])
    hist = []

    def f(u):
        j, per = objective(models, u)
        hist.append({"scales": np.exp(u).tolist(), "J": j, "per": per})
        return j

    def grad(u, h=0.02):
        g = np.zeros_like(u)
        for i in range(len(u)):
            e = np.zeros_like(u)
            e[i] = h
            g[i] = (objective(models, u + e)[0] - objective(models, u - e)[0]) / (2 * h)
        return g

    bounds = [(np.log(0.25), np.log(3.0))] * len(LENGTH_KEYS)
    u0 = np.zeros(len(LENGTH_KEYS))
    j0, per0 = objective(models, u0)
    r = minimize(f, u0, jac=grad, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 60, "ftol": 1e-6, "gtol": 1e-5})
    j1, per1 = objective(models, r.x)
    lengths = dict(zip(LENGTH_KEYS, (design * np.exp(r.x)).tolist()))
    out = {"wavelengths_nm": list(wls), "objective": "mean over lambda of max(TE, TM loss) + 0.5 PDL, dB",
           "start": {"J": j0, "per": per0, "lengths_um": {k: v * 1e6 for k, v in zip(LENGTH_KEYS, design)}},
           "optimum": {"J": j1, "per": per1, "lengths_um": {k: v * 1e6 for k, v in lengths.items()},
                       "scales": np.exp(r.x).tolist()},
           "scipy": {"success": bool(r.success), "message": str(r.message), "nit": int(r.nit), "nfev": int(r.nfev)},
           "history": hist, "seconds": {"setup_warm": t_setup, "optimise": time.time() - t0 - t_setup},
           "mode_solves": "0 beyond the cached end fields (warm)"}
    # sensitivity: 1-D scans of each length about the optimum
    scans = {}
    for i, k in enumerate(LENGTH_KEYS):
        rows = []
        for s in np.exp(np.linspace(np.log(0.25), np.log(3.0), 41)):
            u = r.x.copy()
            u[i] = np.log(s)
            _, per = objective(models, u)
            rows.append({"L_um": float(design[i] * s * 1e6), "per": per})
        scans[k] = rows
    out["scans_about_optimum"] = scans
    scans0 = {}
    for i, k in enumerate(LENGTH_KEYS):
        rows = []
        for s in np.exp(np.linspace(np.log(0.25), np.log(3.0), 41)):
            u = np.zeros(len(LENGTH_KEYS))
            u[i] = np.log(s)
            _, per = objective(models, u)
            rows.append({"L_um": float(design[i] * s * 1e6), "per": per})
        scans0[k] = rows
    out["scans_about_design"] = scans0
    json.dump(out, open(os.path.join(OUT, f"opt_lengths_{'_'.join(map(str, wls))}.json"), "w"),
              indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("start", "optimum", "scipy", "seconds")}, indent=1, default=float))
    return out


def optimise_lengths_global(wls, popsize=12, generations=40, seed=3):
    """CMA-ES over the six log-lengths (the free axes: zero mode solves), then
    an L-BFGS-B polish from its best - the landscape is multimodal (MMI
    interference; the 1-D scans of 4.1 show several minima in Lt and Lm),
    which a gradient method from Table 1 only explores locally."""
    t0 = time.time()
    models = [BandModel(w) for w in wls]
    t_setup = time.time() - t0
    design = np.array([WAN2025_DESIGN[k] for k in LENGTH_KEYS])
    lo, hi = np.log(0.25), np.log(3.0)
    es = CMAES(x0=np.full(len(LENGTH_KEYS), (0.0 - lo) / (hi - lo)), sigma0=0.2, popsize=popsize, seed=seed)
    hist, best = [], (np.inf, None, None)
    for gen in range(generations):
        us = es.ask()
        fs = []
        for u in us:
            uc = np.clip(u, 0.0, 1.0)
            j, per = objective(models, lo + uc * (hi - lo))
            j += 1.0 * float(np.sum(np.abs(u - uc)))          # keep the mean inside the box
            fs.append(j)
            if j < best[0]:
                best = (j, lo + uc * (hi - lo), per)
        es.tell(us, fs)
        hist.append({"gen": gen, "best_J": float(best[0]), "gen_min": float(min(fs)), "sigma": float(es.sigma)})
        print(hist[-1], flush=True)
    j_cma, u_cma, per_cma = best
    bounds = [(lo, hi)] * len(LENGTH_KEYS)

    def f(u):
        return objective(models, u)[0]

    def grad(u, h=0.02):
        g = np.zeros_like(u)
        for i in range(len(u)):
            e = np.zeros_like(u)
            e[i] = h
            g[i] = (f(u + e) - f(u - e)) / (2 * h)
        return g

    r = minimize(f, u_cma, jac=grad, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
    j1, per1 = objective(models, r.x)
    j0, per0 = objective(models, np.zeros(len(LENGTH_KEYS)))
    out = {"wavelengths_nm": list(wls), "evaluations_cma": popsize * generations,
           "start": {"J": j0, "per": per0},
           "cma": {"J": float(j_cma), "per": per_cma,
                   "lengths_um": dict(zip(LENGTH_KEYS, (design * np.exp(u_cma) * 1e6).tolist()))},
           "polished": {"J": j1, "per": per1, "nit": int(r.nit),
                        "lengths_um": dict(zip(LENGTH_KEYS, (design * np.exp(r.x) * 1e6).tolist()))},
           "history": hist, "seconds": {"setup_warm": t_setup, "optimise": time.time() - t0 - t_setup},
           "mode_solves": 0}
    json.dump(out, open(os.path.join(OUT, f"opt_lengths_global_{'_'.join(map(str, wls))}.json"), "w"),
              indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("start", "cma", "polished", "seconds")}, indent=1, default=float))
    return out


# ------------------------------------------------------------ 4.2 facet


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES, after N. Hansen's ``purecma``."""

    def __init__(self, x0, sigma0, popsize=None, seed=1):
        self.n = n = len(x0)
        self.xmean = np.asarray(x0, float)
        self.sigma = float(sigma0)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.rng = np.random.default_rng(seed)
        self.gen = 0

    def ask(self):
        D2, B = np.linalg.eigh(self.C)
        self.B, self.D = B, np.sqrt(np.maximum(D2, 1e-20))
        z = self.rng.standard_normal((self.lam, self.n))
        return [self.xmean + self.sigma * (B @ (self.D * zi)) for zi in z]

    def tell(self, xs, fs):
        order = np.argsort(fs)
        xs = np.asarray(xs)[order]
        old = self.xmean.copy()
        self.xmean = self.w @ xs[: self.mu]
        y = (self.xmean - old) / self.sigma
        Cinvsqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinvsqrt @ y)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        artmp = (xs[: self.mu] - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (artmp.T * self.w) @ artmp)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1


def facet_objective(be, wtip_nm, g_um, wl=1.31e-6):
    """Worse of TE/TM power coupling (best vertical offset) at one facet, full stack."""
    import phase1  # the solve and even-mode picking live there

    p = {"w_low": wtip_nm * 1e-9, "w_high": 0.0, "gap": g_um * 1e-6}
    md, conf, x, y, neff, te, E, H, target = phase1.solve(be, p)
    n_ox = float(be.cross_section.cladding_index_at(wl))
    best = {}
    for pol in ("TE", "TM"):
        m = phase1.even_mode(md, conf, E, pol)
        if m is None:
            best[pol] = 0.0
            continue
        best[pol] = max(power_coupling(x, y, E[m], H[m], 0.5 * dv.MFD, center=(0.0, yc * 1e-6),
                                       pol=pol, n_medium=n_ox)
                        for yc in np.arange(-0.4, 0.41, 0.1))
    return best


def optimise_facet(budget=36, popsize=6):
    import phase1

    be = phase1.backend("full")
    # search (Wtip [nm], g [um]) in normalised coordinates u in [0, 1]^2 over
    # the fabrication range, clipped at the bounds and snapped to the lattice
    lo, hi = np.array([90.0, 0.6]), np.array([200.0, 2.4])
    scale = hi - lo
    fn = os.path.join(OUT, "opt_facet.json")
    hist, cache, x0, sigma0 = [], {}, np.array([130.0, 1.6]), 0.2
    if os.path.exists(fn):
        # resume: every solved candidate is reused, and the search restarts
        # about the best one so far with a smaller step (the covariance of an
        # interrupted run is not kept)
        old = json.load(open(fn))
        for r in old["history"]:
            cache[(r["Wtip_nm"], r["g_um"])] = {"TE": r["TE"], "TM": r["TM"]}
        hist = old["history"]
        best_old = max(hist, key=lambda r: r["worst"])
        x0, sigma0 = np.array([best_old["Wtip_nm"], best_old["g_um"]]), 0.1
    es = CMAES(x0=(x0 - lo) / scale, sigma0=sigma0, popsize=popsize)
    t0, evals = time.time(), 0
    while evals < budget:
        us = es.ask()
        fs = []
        for u in us:
            q = lo + np.clip(np.asarray(u), 0.0, 1.0) * scale
            wtip = round(float(q[0]) / 5) * 5            # the 5 nm lattice of a real tip
            g = round(round(float(q[1]) / 0.02) * 0.02, 2)   # the gap axis
            key = (wtip, g)
            if key not in cache:
                cache[key] = facet_objective(be, *key)
                evals += 1
            c = cache[key]
            f = -min(c["TE"], c["TM"])
            # a penalty keeps the mean inside the box when a sample was clipped
            f += 0.1 * float(np.sum(np.clip(np.abs(np.asarray(u) - 0.5) - 0.5, 0, None)))
            fs.append(f)
            hist.append({"gen": es.gen, "Wtip_nm": wtip, "g_um": g, "TE": c["TE"], "TM": c["TM"],
                         "worst": min(c["TE"], c["TM"]), "t": time.time() - t0})
            print(hist[-1], flush=True)
        es.tell(us, fs)
        json.dump({"history": hist, "evaluations": evals, "generations": es.gen,
                   "mean_Wtip_g": (lo + np.clip(es.xmean, 0, 1) * scale).tolist(), "sigma": es.sigma},
                  open(fn, "w"), indent=1)
    best = max(hist, key=lambda r: r["worst"])
    json.dump({"history": hist, "evaluations": evals, "generations": es.gen, "best": best,
               "mean_Wtip_g": (lo + np.clip(es.xmean, 0, 1) * scale).tolist(), "sigma": es.sigma,
               "seconds": time.time() - t0},
              open(fn, "w"), indent=1)
    print("best", best)


def main():
    what = sys.argv[1]
    if what == "lengths":
        wls = [int(v) for v in sys.argv[2:]] or [1310]
        optimise_lengths(wls)
    elif what == "lengths_global":
        wls = [int(v) for v in sys.argv[2:]] or [1310]
        optimise_lengths_global(wls)
    elif what == "facet":
        budget = int(sys.argv[2]) if len(sys.argv) > 2 else 36
        optimise_facet(budget)


if __name__ == "__main__":
    main()
