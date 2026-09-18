"""Kocabas's own measure of the converter: total forward Poynting flux through a
cut past the tip, back-propagated to the tip with the slot mode's loss.

Usage: kocabas_paper_measure.py <dataset suffix> [<out json>]   (a (w_si, half_slot) dataset)

Report 13 section 9.

What the cascade gives is the vector of complex forward amplitudes a_j at the
end of the 200 nm lead-out, in the cascade's basis of the slot cross section:
the dataset's biorthogonal modes, permuted by the mode tracking and given a
+-1 sign gauge by the overlap-phase equalisation, with unit *unconjugated*
power each.  The physical power carried by a superposition is the conjugated
quadratic form
    P = Re sum_jk a_j conj(a_k) W_jk,   W_jk = 1/2 int (E_j x conj(H_k))_z dA
over the physical window (outside the PML, where a Berenger mode's conjugated
power means nothing).  Its diagonal holds the power factors r_j - 1.07 for the
slot mode, up to 10 for a Berenger mode - and its off-diagonal terms are the
interference the modal number ignores.  In the uniform lead-out every mode
propagates as exp(i k0 n_j z), so a_j at any cut z past the tip follows
analytically.  W needs fields, which the dataset does not store, so the end
sections are re-solved here, checked against the stored overlap, and put into
the cascade's basis with the recovered permutation and signs.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import demo_kocabas_converter as demo                       # noqa: E402
from demo_plasmonic_converter import lumped, physical, ports, K0   # noqa: E402
from em_simulation import DataUpdater                        # noqa: E402
from em_simulation.fde.assemble import assemble, overlap_matrix   # noqa: E402

suffix = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "reports", "output", f"kocabas_paper_measure_{suffix}.json")
demo.use_cell(5.0, suffix=suffix, axes="half_slot")
du = DataUpdater(demo.DATASET)
path, length = demo.device(du)
extra = demo.DESIGN["extra"] * 1e-9

# --- every path point must already be cached: this script must not solve
snapped, dz = path.calc_simulation_parameters()
eme_path = path.interp_multi_adj_pts(snapped, dz)[0]
keys = set(du.neff.keys())
missing = [p for p in eme_path if tuple(p) not in keys]
if missing:
    sys.exit(f"{len(missing)} of {len(eme_path)} path points are not cached - refusing to solve here: {missing[:3]}")
print(f"dataset {os.path.basename(demo.DATASET)}: {len(keys)} cached points", flush=True)

t0 = time.time()
res = lumped(path)
S, od = res["S"], res["od"]
N = od["neff"].shape[1] // 2
i_in, j_out = ports(od)
a_end = S[:N, i_in]                                   # forward amplitudes at the lead-out end, cascade basis
n_end = od["neff"][-1, :N]
n_slot = n_end[j_out]
L = len(od["EME_path"])
p_end, p_in = od["EME_path"][-1], od["EME_path"][0]
k = max(i for i in range(L) if tuple(od["EME_path"][i]) != tuple(p_end))      # last distinct point -> interface k
p_prev = od["EME_path"][k]
print(f"cascade {time.time()-t0:.0f} s, {L} sections; input mode {i_in} at {tuple(np.round(np.array(p_in)*1e9,1))}, "
      f"output mode {j_out} at {tuple(np.round(np.array(p_end)*1e9,1))}; slot n = {n_slot:.4f}", flush=True)
modal_end = abs(a_end[j_out]) ** 2
back_extra = np.exp(2 * K0 * n_slot.imag * extra)
print(f"modal: |a_slot|^2 = {modal_end:.4f} at the lead-out end, x{back_extra:.4f} -> {modal_end*back_extra*100:.2f} % at the tip", flush=True)

# --- re-solve the end sections; check against the stored overlap
be = du.backend
t0 = time.time()
d_end = be.solve(tuple(p_end)); d_prev = be.solve(tuple(p_prev)); d_in = be.solve(tuple(p_in))
print(f"re-solved 3 sections in {time.time()-t0:.0f} s (prev = {tuple(np.round(np.array(p_prev)*1e9,1))})", flush=True)
x, y, neff, te, E, H = assemble([d_prev, d_end, d_in], N, lossless=False)
O_fresh = overlap_matrix(E[0], H[1], x, y)[:N, :N]                     # E_prev x H_end, solver order
O_st = np.asarray(du.overlap[tuple(p_prev)][tuple(p_end)])[:N, :N]
dO = np.abs(O_fresh - O_st)
print(f"re-solve check vs stored overlap(prev, end): max |dO| {dO.max():.2e}", flush=True)

# --- the cascade's basis at the end: tracking permutation and equalisation signs
tmn = np.asarray(path._tracking_mode_names)                # tmn[section, solver slot] = tracked index
inv_end = np.argsort(tmn[k + 1][:N]); inv_prev = np.argsort(tmn[k][:N])   # tracked index -> solver slot
n_solver = neff[1][:N]
perm_err = np.abs(n_solver[inv_end] - n_end).max()
O_eq = np.asarray(od["overlap_ab"][k])[:N, :N]                          # equalised, tracked order
O_perm = O_st[np.ix_(inv_prev, inv_end)]                                # stored, tracked order, no signs
defined = np.abs(O_perm) > 1e-12
R = np.where(defined, O_eq / np.where(defined, O_perm, 1.0), 0.0)
ratio_err = np.abs(np.abs(R[defined]) - 1.0).max()
m0 = int(np.argmax(defined.sum(axis=1)))                                # a prev mode overlapping every end mode
signs = np.sign(np.real(R[m0]))
signs[signs == 0] = 1.0
later_flips = [i for i in range(k + 1, L - 1) if (np.real(np.diag(np.asarray(od["overlap_ab"][i])[:N, :N])) < 0).any()]
print(f"basis recovery: permutation check (neff) {perm_err:.1e}; equalisation ratio |R|-1 max {ratio_err:.1e}; "
      f"{int((signs < 0).sum())} sign flips at the end section; later interfaces with negative diagonals: {later_flips}", flush=True)
if perm_err > 1e-6 or ratio_err > 1e-6 or later_flips:
    sys.exit("could not put the re-solved fields into the cascade's basis")
Ec = signs[:, None, None, None] * E[1][inv_end]; Hc = signs[:, None, None, None] * H[1][inv_end]

# --- conjugated power form over the physical window (outside any PML)
xr, yr = np.real(x), np.real(y)
pml = getattr(be, "pml_thickness", 0.0) or 0.0
box = (np.abs(xr)[:, None] <= xr.max() - pml + 1e-12) & (np.abs(yr)[None, :] <= yr.max() - pml + 1e-12)
print(f"physical window |x| <= {(xr.max()-pml)*1e9:.0f} nm, |y| <= {(yr.max()-pml)*1e9:.0f} nm ({box.mean()*100:.0f} % of the grid)", flush=True)
W = overlap_matrix(Ec * box, np.conj(Hc) * box, xr, yr)[:N, :N]
W_in = overlap_matrix(E[2] * box, np.conj(H[2]) * box, xr, yr)[:N, :N]
inv_in = np.argsort(tmn[0][:N])
P_in = float(np.real(W_in[inv_in[i_in], inv_in[i_in]]))
r = np.real(np.diag(W))
phys_end = physical(od, -1)
print(f"power factors r_j: slot {r[j_out]:.3f}; physical {np.round(r[phys_end], 3)}; "
      f"Berenger min/max {r[~phys_end].min():.2f}/{r[~phys_end].max():.2f}; launched power P_in = {P_in:.4f}", flush=True)


def at(z):
    """Forward amplitudes at z past the tip (z >= 0; the S-matrix plane is z = extra)."""
    return a_end * np.exp(1j * K0 * n_end * (z - extra))


def flux(z):
    a = at(z)
    total = float(np.real(a @ W @ np.conj(a)))
    diag = float(np.real(np.sum(np.abs(a) ** 2 * r)))
    slot = float(abs(a[j_out]) ** 2 * r[j_out])
    ber = float(np.real(np.sum(np.abs(a[~phys_end]) ** 2 * r[~phys_end])))
    return dict(z_nm=float(z * 1e9), total=total / P_in, diagonal=diag / P_in, slot=slot / P_in,
                other_physical=(diag - slot - ber) / P_in, berenger=ber / P_in, cross=(total - diag) / P_in,
                back=float(np.exp(2 * K0 * n_slot.imag * z)))


rows = [flux(z) for z in np.arange(0, 2.0001e-6, 50e-9)]
print(f"{'z (nm)':>7} {'total':>7} {'x back':>7} | {'slot':>6} {'other':>6} {'Beren.':>6} {'cross':>7}  (fractions of launched power)")
for row in rows:
    if row["z_nm"] % 200 < 1e-6 or abs(row["z_nm"] - 1100) < 1e-6:
        print(f"{row['z_nm']:7.0f} {row['total']:7.4f} {row['total']*row['back']:7.4f} | {row['slot']:6.4f} {row['other_physical']:6.4f} "
              f"{row['berenger']:6.4f} {row['cross']:+7.4f}")
at_1100 = flux(1100e-9)
print(f"paper's measure (total flux at 1100 nm, back-propagated): {at_1100['total']*at_1100['back']*100:.2f} % "
      f"(diagonal only {at_1100['diagonal']*at_1100['back']*100:.2f} %); modal |a|^2 at the tip {modal_end*back_extra*100:.2f} %, "
      f"as physical power |a|^2 r {modal_end*back_extra*r[j_out]/P_in*100:.2f} %; the paper reports ~{demo.DESIGN['transmission']*100:.0f} %")
json.dump(dict(dataset=os.path.basename(demo.DATASET), sections=L, slot_neff=[float(n_slot.real), float(n_slot.imag)],
               modal_at_tip=float(modal_end * back_extra), modal_physical_at_tip=float(modal_end * back_extra * r[j_out] / P_in),
               P_in=P_in, r_slot=float(r[j_out]), r_berenger=[float(v) for v in r[~phys_end]],
               resolve_check=dict(max_dO=float(dO.max()), perm_err=float(perm_err), ratio_err=float(ratio_err),
                                  sign_flips=int((signs < 0).sum())),
               physical_window_nm=[float((xr.max() - pml) * 1e9), float((yr.max() - pml) * 1e9)],
               curve=rows, paper_measure_1100=float(at_1100["total"] * at_1100["back"]),
               paper_measure_1100_diagonal=float(at_1100["diagonal"] * at_1100["back"])),
          open(out_json, "w"), indent=1)
print("->", out_json)
