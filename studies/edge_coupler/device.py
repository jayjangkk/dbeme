"""The Wan & Wang bilayer double-tip edge coupler as a DBEME path.

Geometry as ``tasks/18_edge_coupler_oband.md`` fixes it (Table 1 plus the
decisions Q1-Q3): ``z`` runs from the facet (0) to the output guide
(86.5 um); every cross section is a point of the ``(w_low, w_high, gap)``
family of :class:`~dbeme.fde.BiLevelPair`.

* L1 (0-25 um): 150 nm tips, arm width linear from ``Wtip``;
* L2 (25-45 um): the full-height part grows from 0 to the whole arm on the
  inner edge;
* L3 (45-68 um): 220 nm strips, width to ``W0``, gap ``g -> d``;
* Lt (68-76 um): arms widen to ``Wm2 / 2`` while the gap closes to 0;
* Lm (76-78.5 um): the wedge MMI, ``Wm2 -> Wm1``;
* output taper (78.5-86.5 um): ``W1 -> W0``, entered abruptly at the MMI face.

The arm width through L1-L3 is one straight edge ``Wtip -> W0`` (Q1).
"""

import os
import sys
import time
from copy import deepcopy

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dbeme import DataUpdater, ParametricPath  # noqa: E402
from dbeme.fde.assemble import assemble  # noqa: E402
from dbeme.platforms import WAN2025_DESIGN  # noqa: E402
from dbeme.propagator.fiber import gaussian_launch, physical_interior  # noqa: E402
from dbeme.validation import lumped_smatrix  # noqa: E402

UM = 1e-6
NM = 1e-9
CELL = 10 * NM
WAVELENGTHS_NM = (1260, 1280, 1300, 1310, 1320, 1340, 1360)
MFD = 4.0 * UM
PHYSICAL_IMAG_NEFF = 0.05


def dataset_dir(wavelength_nm, substrate=True):
    suffix = "_ox" if not substrate else ""
    return os.path.join(ROOT, "datasets", f"Si_bilevel_pair_220_150nm{suffix}_{int(wavelength_nm)}")


def breakpoints(d=None):
    d = d or WAN2025_DESIGN
    z1 = d["L1"]
    z2 = z1 + d["L2"]
    z3 = z2 + d["L3"]
    z4 = z3 + d["Lt"]
    z5 = z4 + d["Lm"]
    z6 = z5 + d["L_out"]
    return z1, z2, z3, z4, z5, z6


def _snap(v, step=CELL):
    return np.round(np.asarray(v, dtype=float) / step) * step


def coarse_functions(grid, variant, factor, z_from, z_to, design=None, pitch=25 * NM):
    """The path with ``factor`` consecutive steps of each parameter merged
    into one, between ``z_from`` and ``z_to``; unchanged elsewhere.

    For the staircase extrapolation.  A laterally moving arm sheds its
    translation mismatch at every step - 0.27 % of TE per 20 nm gap step in
    L3, independent of the gap - a total linear in the step size that the
    smooth device does not have.  Merging steps of the dataset's own snapped
    staircase (keeping every ``factor``-th change of each parameter, and the
    last one, so each segment still ends on its design value) doubles or
    quadruples the step and leaves the geometry otherwise alone; every value
    is one the fine path already visits.

    :param grid: the dataset's ``parameter_grid``, for the snapping the path
        itself would do.
    """
    fine, total = path_functions(design, variant)
    z = np.linspace(0.0, total, int(round(total / pitch)) + 1)
    held = {}
    for name in ("w_low", "w_high", "gap"):
        ax = np.asarray(grid[name], dtype=float)
        v = ax[np.argmin(np.abs(np.asarray(fine[name](z))[:, None] - ax[None, :]), axis=1)]
        out = v.copy()
        inside = np.flatnonzero((z >= z_from - 1e-12) & (z < z_to - 1e-12))
        if inside.size:
            ch = [i for i in inside[1:] if v[i] != v[i - 1]]
            keep = set(ch[len(ch) - 1::-factor]) if ch else set()
            cur = v[inside[0]]
            for i in inside:
                if i in keep:
                    cur = v[i]
                out[i] = cur
        held[name] = out

    def mk(name):
        vals = held[name]

        def f(zz):
            zz = np.asarray(zz, dtype=float)
            idx = np.clip(np.round(zz / pitch).astype(int), 0, len(vals) - 1)
            return vals[idx]
        return f

    return {n: mk(n) for n in ("w_low", "w_high", "gap")}, total


def path_functions(design=None, variant="bilayer", w12=None, w23=None, snap=True, lattice=CELL):
    """``{name: f(z)}`` for the path, values already on the width lattice.

    :param variant: ``"bilayer"`` (the paper's design) or ``"conventional"``
        (220 nm tips throughout, no height converter - Fig. 3/5 dashed).
    :param w12, w23: arm width at the L1/L2 and L2/L3 boundaries.  ``None``
        keeps the single straight edge ``Wtip -> W0`` (Q1); given, the width
        is piecewise linear through them (Phase 4 frees them).
    :param snap: ``False`` returns the smooth design values, off the width
        lattice - the exact device a direct EME solves (section 5.8).
    :param lattice: The width lattice the values are snapped to (10 nm, the
        cell); the fine tip dataset uses 2.5 nm.
    """
    q = (lambda v: _snap(v, lattice)) if snap else (lambda v: np.asarray(v, dtype=float))
    d = dict(design or WAN2025_DESIGN)
    z1, z2, z3, z4, z5, z6 = breakpoints(d)
    zs = [0.0, z1, z2, z3]
    if w12 is None and w23 is None:
        ws = [d["Wtip"], None, None, d["W0"]]
        ws[1] = d["Wtip"] + (d["W0"] - d["Wtip"]) * z1 / z3
        ws[2] = d["Wtip"] + (d["W0"] - d["Wtip"]) * z2 / z3
    else:
        ws = [d["Wtip"], w12, w23, d["W0"]]

    def arm(z):
        return np.interp(z, zs, ws)

    def lattice_arm(z):
        return q(arm(z))

    def high_fraction(z):
        if variant == "conventional":
            return np.ones_like(z)
        return np.clip((z - z1) / (z2 - z1), 0.0, 1.0)

    def w_high(z):
        z = np.asarray(z, dtype=float)
        tot = lattice_arm(z)
        early = q(tot * high_fraction(z))
        lt = q(np.interp(z, [z3, z4], [d["W0"], 0.5 * d["Wm2"]]))
        mmi = q(np.interp(z, [z4, z5], [0.5 * d["Wm2"], 0.5 * d["Wm1"]]))
        out = np.interp(z, [z5, z6], [0.5 * d["W1"], 0.5 * d["W0"]])
        if snap:
            out = np.round(out / (5 * NM)) * (5 * NM)
        out = np.where(np.abs(z - z5) < 1e-12, 0.5 * d["W1"], out)     # 367.5 nm exactly
        return np.select([z < z3, z < z4, z < z5 - 1e-12], [early, lt, mmi], out)

    def w_low(z):
        z = np.asarray(z, dtype=float)
        tot = lattice_arm(z)
        return np.where(z < z3, np.maximum(tot - w_high(z), 0.0), 0.0)

    def gap(z):
        z = np.asarray(z, dtype=float)
        g = np.interp(z, [z2, z3, z4], [d["g"], d["d"], 0.0])
        return np.where(z < z2, d["g"], g)

    return {"w_low": w_low, "w_high": w_high, "gap": gap}, z6


class EdgeCouplerPath(ParametricPath):
    """A ``ParametricPath`` that walks gradual steps and jumps abrupt ones.

    * A transition that moves any axis by more than ``jump_steps`` grid
      steps between two samples is a physical discontinuity (the MMI's output
      face, 1.5 -> 0.735 um): it becomes one interface, linked directly, and
      is **not** walked through the intermediate widths, which would smear the
      face into a taper over the preceding section.
    * The mutual-neighbour corner points of a diagonal step are not solved:
      the propagator never uses them (``additional_overlap_dict`` is read by
      the geometry only), and on a three-axis path they cost a solve per
      diagonal step.
    """

    jump_steps = 4

    def get_extended_simul_params(self, simulation_params):
        return list(simulation_params), {}

    def _get_additional_overlap_dict(self, simul_params, multi_adj_index):
        return {"ab": {}, "ba": {}}

    def _index(self, j, value):
        grid = np.asarray(self.parameter_grid[self.parameter_names[j]])
        return int(np.argmin(np.abs(grid - value)))

    #: ``(z0, z1)`` in metres inside which every change is one interface
    #: (no walk, no corner point); ``None``: nowhere.  The staircase study sets it.
    direct_zrange = None

    def _abrupt(self, pt1, pt2, z=None):
        if self.direct_zrange is not None and z is not None:
            if self.direct_zrange[0] - 1e-12 <= z < self.direct_zrange[1] - 1e-12:
                return tuple(pt1) != tuple(pt2)
        return any(abs(self._index(j, a) - self._index(j, b)) > self.jump_steps
                   for j, (a, b) in enumerate(zip(pt1, pt2)))

    def interp_multi_adj_pts(self, pts, delta_zs):
        pts = list(deepcopy(pts))
        delta_zs = list(delta_zs)
        delta_z_copy = deepcopy(delta_zs)
        multi_adj_index = []
        added = 0
        index_mapping = {}
        self.abrupt_interfaces = []
        z_end = np.cumsum(delta_z_copy)                  # where each interface sits
        for i, dz in enumerate(delta_z_copy):
            pt1, pt2 = pts[i + added], pts[i + added + 1]
            if self._abrupt(pt1, pt2, z_end[i]):
                self.abrupt_interfaces.append((tuple(pt1), tuple(pt2)))
                index_mapping[i + added] = i
                continue
            walk, current = [], list(pt1)
            for j in range(len(pt1)):
                if current[j] == pt2[j]:
                    continue
                for value in self._grid_steps(j, current[j], pt2[j]):
                    current[j] = value
                    walk.append(tuple(current))
            walk = walk[:-1]
            if walk:
                pts[i + added + 1:i + added + 1] = walk
                delta_zs[i + added:i + added + 1] = list(np.ones(len(walk) + 1) * (dz / (len(walk) + 1)))
                added += len(walk)
                if self.check_if_multi_adj_pt(pt1, pt2):
                    multi_adj_index.append(i)
                for k in range(len(walk)):
                    index_mapping[i + added - len(walk) + k] = i
            index_mapping[i + added] = i
        index_mapping[len(pts) - 1] = len(delta_z_copy)
        return pts, np.array(delta_zs), multi_adj_index, index_mapping


def open_dataset(wavelength_nm, substrate=True, cache_size=2):
    return DataUpdater(dataset_dir(wavelength_nm, substrate), cache_size=cache_size)


def build_path(du, variant="bilayer", design=None, w12=None, w23=None, resolution=None,
               verbose=False):
    funcs, total = path_functions(design, variant, w12, w23)
    if resolution is None:
        resolution = int(round(total / (25 * NM))) + 1          # 25 nm pitch: every breakpoint a sample
    return EdgeCouplerPath(du, funcs, total_length=total, resolution=resolution,
                           max_section_length=2 * UM, verbose=verbose), total


# ------------------------------------------------------------------ fields


def point_fields(du, point):
    """``(x, y, neff, TE_pol, E, H)`` of a path point in the dataset's basis
    (normalised, biorthogonal, forward modes only), by re-solving it."""
    md = du.backend.solve(tuple(point))     # not solve_point: an endpoint is not a dataset update
    N = du.mode_numbers
    x, y, neff, te, E, H = assemble([md], N, du.prop_axis, lossless=du._is_lossless())
    return x, y, neff[0, :N], te[0, :N], E[0, :N], H[0, :N]


def power_ratio(x, y, E, H):
    """``r = P_phys / |P_unconj|`` per mode on the physical interior."""
    inside = physical_interior(x, y)
    dx = np.abs(np.diff(np.real(x)))
    dy = np.abs(np.diff(np.real(y)))
    w = np.outer(np.r_[dx, dx[-1]], np.r_[dy, dy[-1]]) * inside
    wc = np.outer(np.r_[np.diff(x), np.diff(x)[-1]], np.r_[np.diff(y), np.diff(y)[-1]])
    out = []
    for m in range(E.shape[0]):
        phys = 0.5 * np.real(np.sum((E[m, 0] * np.conj(H[m, 1]) - E[m, 1] * np.conj(H[m, 0])) * w))
        unconj = 0.5 * np.sum((E[m, 0] * H[m, 1] - E[m, 1] * H[m, 0]) * wc)
        out.append(phys / abs(unconj) if abs(unconj) else np.nan)
    return np.array(out)


def physical(od, section):
    neff, mask = od["neff"], od["radiation_mode_mask"]
    N = neff.shape[1] // 2
    return (~mask[section, :N]) & (np.imag(neff[section, :N]) < PHYSICAL_IMAG_NEFF)


def output_ports(od):
    """``{"TE0": j, "TM0": j}``: highest-index physical TE-/TM-like branch at the end."""
    neff, te = od["neff"], od["TE_pol"]
    N = neff.shape[1] // 2
    keep = np.flatnonzero(physical(od, -1))
    ports = {}
    for name, sel in (("TE0", np.real(te[-1, keep]) >= 0.5), ("TM0", np.real(te[-1, keep]) < 0.5)):
        cand = keep[sel]
        ports[name] = int(cand[np.argmax(np.real(neff[-1, cand]))]) if cand.size else None
    return ports


def run_device(du, variant="bilayer", pols=("TE", "TM"), centers=None, design=None,
               w12=None, w23=None, facet=None, output=None, verbose=False):
    """Launch the fibre beam and return the power in the output TE0/TM0.

    :param facet, output: cached ``point_fields`` of the end points (they
        cost a mode solve each); ``None`` solves them.
    :returns: dict with the path, S-matrix, launches, output powers, ports.
    """
    t0 = time.time()
    path, total = build_path(du, variant, design, w12, w23, verbose=verbose)
    od = path.calc_output_data()
    t_data = time.time() - t0
    S = lumped_smatrix(path, force_unitary=False, method="direct")
    t_s = time.time() - t0 - t_data
    N = S.shape[0] // 2
    p0, pL = tuple(od["EME_path"][0]), tuple(od["EME_path"][-1])
    if facet is None or tuple(facet["point"]) != p0:
        x, y, n0, te0, E0, H0 = point_fields(du, p0)
        facet = {"point": p0, "x": x, "y": y, "neff": n0, "TE_pol": te0, "E": E0, "H": H0}
    if output is None or tuple(output["point"]) != pL:
        x, y, nL, teL, EL, HL = point_fields(du, pL)
        output = {"point": pL, "neff": nL, "TE_pol": teL, "r": power_ratio(x, y, EL, HL)}
    ports = output_ports(od)
    # output branch j sits at solver slot k with tracking_mode_names[-1, k] == j
    names = path._tracking_mode_names[-1]
    r_branch = np.empty(N)
    for k in range(N):
        r_branch[int(names[k])] = output["r"][k]
    n_ox = float(du.backend.cross_section.cladding_index_at(du.wavelength))
    results = {}
    for pol in pols:
        for c in (centers or [(0.0, 0.0)]):
            launch = gaussian_launch(facet["x"], facet["y"], facet["E"], facet["H"], 0.5 * MFD,
                                     center=c, pol=pol, n_medium=n_ox)
            a = launch["a"]
            out = S @ np.concatenate([a, np.zeros(N, dtype=complex)])
            fwd, bwd = out[:N], out[N:]
            p_fwd = np.abs(fwd) ** 2 * r_branch
            key = f"{pol}@{c[0]*1e6:+.2f},{c[1]*1e6:+.2f}"
            results[key] = {
                "pol": pol, "center": list(c),
                "launch_power_total": float(np.sum(np.abs(a) ** 2)),
                "launch_top": [(int(i), float(abs(a[i]) ** 2)) for i in np.argsort(-np.abs(a))[:4]],
                "window_power": launch["window_power"],
                "P_TE0": float(p_fwd[ports["TE0"]]) if ports["TE0"] is not None else np.nan,
                "P_TM0": float(p_fwd[ports["TM0"]]) if ports["TM0"] is not None else np.nan,
                "reflected": float(np.sum(np.abs(bwd) ** 2)),
                "forward_other_physical": float(np.sum(p_fwd[physical(od, -1)]) -
                                                np.nansum([p_fwd[j] for j in ports.values() if j is not None])),
                "a": a, "fwd": fwd,
            }
    return {"path": path, "od": od, "S": S, "ports": ports, "results": results,
            "facet": facet, "output": output, "total_length": total,
            "seconds": {"data": t_data, "smatrix": t_s, "total": time.time() - t0},
            "r_branch": r_branch}


def loss_db(p):
    return -10.0 * np.log10(max(float(p), 1e-300))
