"""Demo 4: the Si-wire-to-plasmonic-slot mode converter, as a lateral 2-D model.

Ono et al., *Optica* **3**, 999 (2016) / NTT Technical Review **16**(7) (2018):
a 400 x 200 nm Si wire feeds a gold/air MIM waveguide with a 50 x 20 nm core
through a 600 nm taper; -1 dB simulated with a 20 nm air gap, -1.4 dB
calculated and -1.7 dB measured with the 40 nm gap that was fabricated, and
the slot itself loses about 1 dB/um.

What is modelled here is the **lateral** stepping stone CLAUDE.md prescribes
for this device: the Si width tapers 400 nm -> 0 between two full-height gold
walls that follow the taper at a constant air gap, suspended in air
(``em_simulation.fde.slot_converter``).  The device confines its plasmon
*vertically* (the 20 nm), which one swept width cannot represent; the model's
gap plasmon is the lateral one of the ``2 x gap`` slot the walls leave behind.
Absolute numbers are therefore not the paper's - the report says which
comparisons survive that.

The dataset is a lossy PML basis (``PMLBackend``), so every ``n_eff`` is
complex, the cascade runs on the scattering route (``auto`` -> ``direct``) and
``force_unitary`` stays off - projecting a lossy S-matrix onto the nearest
unitary one would delete the metal loss.

Two things a lossy basis changes about reading an S-matrix:

* modes are normalised with the *unconjugated* power ``1/2 int (E x H)_z``,
  so ``|a_i|^2`` is physical power only up to a per-mode factor
  ``r_i = 1/2 Re int (E_i x H_i*)_z / |1/2 int (E_i x H_i)_z|``.  The headline
  conversion is quoted as ``|S_out,in|^2 r_out / r_in``; the raw number is
  kept beside it;
* reciprocity is ``S = S^T`` in the standard ``[[R1, T12], [T21, R2]]``
  arrangement, which the lumped matrix (ordered ``[b2; b1] = S [a1; a2]``)
  is re-blocked into before the check.

Stages (each cached in ``reports/output/plasmonic_converter.json``):

1. sanity gate: propagation loss of a constant-width walled wire, reciprocity
   and passivity of the 600 nm taper, junction mismatch bare wire -> walled;
2. taper-length sweep on the dataset, both gaps (the literature figure);
3. DBEME vs direct EME with the direct sections at the axis widths, 300 and
   600 nm, plus a 2x coarser direct slicing;
4. ``--cell-check``: the Si-end and slot indices at 8 / 6 / 5 / 4 nm cells.

Run:  python examples/demo_plasmonic_converter.py [--gaps 20 40] [--skip-direct]
                                                  [--cell-check] [--lengths ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402

from _plotting import save  # noqa: E402
from em_simulation import (  # noqa: E402
    EME,
    DataExtractor,
    DataUpdater,
    DirectParametricPath,
    ParametricPath,
    Runner,
)
from em_simulation.fde.materials import air  # noqa: E402
from em_simulation.fde.pml import PMLBackend  # noqa: E402
from em_simulation.fde.slot_converter import PlasmonicSlotConverter  # noqa: E402
from em_simulation.platforms import plasmonic_converter_dataset_info  # noqa: E402
from em_simulation.reference.plasmonic import attenuation_db_per_um  # noqa: E402

TAG = "plasmonic"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {20: "Si_plasmonic_slot_1550", 40: "Si_plasmonic_slot_1550_gap40"}
OUT_JSON = os.path.join(ROOT, "reports", "output", f"{TAG}_converter.json")

WAVELENGTH = 1.55e-6
K0 = 2 * np.pi / WAVELENGTH
W_IN = 400e-9
LENGTHS_NM = (150, 300, 450, 600, 800, 1000, 1500)
DIRECT_NM = (300, 600)
#: A branch losing more than this is not a port: Berenger modes sit at
#: Im ~ 0.8, and a leaky resonance a hair above the air line at Im = 0.10
#: (3.6 dB/um) is not one either.  The slot plasmon is at 0.036.
PHYSICAL_IMAG_NEFF = 0.05

#: Ono et al. - the 3-D device, quoted for the overlay only.
PAPER = {
    20: {"simulated_dB": -1.0, "length_nm": 600, "note": "designed gap"},
    40: {"calculated_dB": -1.4, "measured_dB": -1.7, "length_nm": 600,
         "note": "fabricated gap"},
    "slot_loss_dB_per_um": 1.0,
}


# ----------------------------------------------------------------- devices


def schedule(length):
    """Linear width taper 400 nm -> 0 over ``length``; constant beyond it."""
    return {
        "w_si": lambda z, L=length: W_IN * (
            1.0 - np.clip(np.asarray(z, dtype=float) / L, 0.0, 1.0)
        )
    }


def lumped(geometry):
    """Lumped ``(2N x 2N)`` S-matrix of a geometry, its output data, timing.

    Column ``j`` is the response to unit amplitude in forward mode ``j`` at the
    input: rows ``[:N]`` leave the output port forward, rows ``[N:]`` come
    back out of the input.
    """
    t0 = time.time()
    od = geometry.calc_output_data()
    eme = EME(geometry, force_unitary=False)
    eme.calc_Smatrix()
    runner = Runner(eme)
    mask = od["radiation_mode_mask"]
    launch = np.zeros(int(np.count_nonzero(mask[0] == False)), dtype=complex)
    launch[0] = 1.0
    runner.propagate_lumped_smatrix(launch)
    S = np.asarray(runner.runner._lumped_smatrix)
    return {"S": S, "od": od, "seconds": time.time() - t0,
            "route": eme.propagator.resolve_method()}


def standard_form(S):
    """Re-block ``[b2; b1] = S [a1; a2]`` into ``[[R1, T12], [T21, R2]]``."""
    N = S.shape[0] // 2
    return np.block([[S[N:, :N], S[N:, N:]], [S[:N, :N], S[:N, N:]]])


def physical(od, section):
    """Branches that are physical modes at a section: bound, and not Berenger."""
    neff, mask = od["neff"], od["radiation_mode_mask"]
    N = neff.shape[1] // 2
    return (~mask[section, :N]) & (np.imag(neff[section, :N]) < PHYSICAL_IMAG_NEFF)


def ports(od):
    """Input mode and output port: the highest-index physical branch at either end."""
    neff = od["neff"]

    def best(section):
        keep = np.flatnonzero(physical(od, section))
        if not keep.size:
            raise RuntimeError("no physical mode at section %d" % section)
        return int(keep[np.argmax(np.real(neff[section, keep]))])

    return best(0), best(-1)


def budget(S, od, r_in=1.0, r_out=1.0):
    """Where unit power launched in the input mode ends up."""
    N = S.shape[0] // 2
    i_in, j_out = ports(od)
    fwd = np.abs(S[:N, i_in]) ** 2
    bwd = np.abs(S[N:, i_in]) ** 2
    phys_out, phys_in = physical(od, -1), physical(od, 0)
    converted = float(fwd[j_out])
    converted_phys = converted * r_out / r_in
    return {
        "input_mode": i_in,
        "output_mode": j_out,
        "converted_raw": converted,
        "converted": converted_phys,
        "converted_dB": 10 * np.log10(max(converted_phys, 1e-300)),
        "converted_raw_dB": 10 * np.log10(max(converted, 1e-300)),
        "r_in": float(r_in), "r_out": float(r_out),
        "other_physical_fwd": float(fwd[phys_out].sum() - converted),
        "berenger_fwd": float(fwd[~phys_out].sum()),
        "reflected_physical": float(bwd[phys_in].sum()),
        "reflected_berenger": float(bwd[~phys_in].sum()),
        "deficit": float(1.0 - fwd.sum() - bwd.sum()),
        "forward_out": [float(v) for v in fwd],
    }


def guided_block(S, od):
    """Indices (forward and backward) of physical modes at both ends."""
    N = S.shape[0] // 2
    both = np.flatnonzero(physical(od, 0) & physical(od, -1))
    return np.concatenate([both, both + N])


# ---------------------------------------------------- power normalisation


def power_ratio(backend, w, index=0):
    """``r = P_phys / |P_unconj|`` of one mode - the factor between ``|a|^2``
    and the power it carries.  Normalisation-independent, so it can be taken
    from a raw solve and applied to dataset amplitudes."""
    data, _ = backend.mode_data({"w_si": w}, backend.target_neff)
    E, H = data.E[index], data.H[index]
    x, y = backend.x, backend.y
    dA = np.gradient(x)[:, None] * np.gradient(y)[None, :]
    unconj = 0.5 * np.sum((E[0] * H[1] - E[1] * H[0]) * dA)
    phys = 0.5 * np.real(np.sum((E[0] * np.conj(H[1]) - E[1] * np.conj(H[0])) * np.real(dA)))
    return float(phys / abs(unconj)), complex(data.neff[index])


def end_ratios(payload, gap):
    """``r`` of the input mode at 400 nm and of the slot mode at 0, cached."""
    key = f"power_ratio_gap{gap}"
    if key not in payload:
        backend = plasmonic_converter_dataset_info(WAVELENGTH, gap=gap * 1e-9)().get_fde_backend()
        t0 = time.time()
        r_in, n_in = power_ratio(backend, W_IN)
        r_out, n_out = power_ratio(backend, 0.0)
        payload[key] = {
            "r_in": r_in, "r_out": r_out,
            "neff_in": [n_in.real, n_in.imag], "neff_out": [n_out.real, n_out.imag],
            "seconds": time.time() - t0,
        }
        dump(payload)
        print(f"  power factors: r_in {r_in:.4f} (n = {n_in:.4f}), "
              f"r_out {r_out:.4f} (n = {n_out:.4f})   [{time.time()-t0:.0f} s]")
    rec = payload[key]
    return rec["r_in"], rec["r_out"]


# ------------------------------------------------------------ sanity gate


def sanity_gate(du, taper):
    rows = []
    # (a) a constant-width walled wire loses exactly its propagation loss
    L = 1.0e-6
    res = lumped(ParametricPath(du, {"w_si": W_IN}, total_length=L))
    n0 = res["od"]["neff"][0, 0]
    expected = float(np.exp(-2 * K0 * np.imag(n0) * L))
    measured = float(np.abs(res["S"][0, 0]) ** 2)
    rows.append({
        "check": "constant-width walled wire, 1 um: T = exp(-2 k0 Im(n) L)",
        "criterion": "|T - expected| < 1e-3",
        "expected": expected, "measured": measured,
        "pass": bool(abs(measured - expected) < 1e-3),
        "neff": [float(np.real(n0)), float(np.imag(n0))],
        "loss_dB_per_um": attenuation_db_per_um(n0, WAVELENGTH),
    })
    # (b) reciprocity of the 600 nm taper, standard arrangement
    S, od = taper["S"], taper["od"]
    std = standard_form(S)
    asym = float(np.abs(std - std.T).max())
    block = guided_block(S, od)
    asym_guided = float(np.abs((std - std.T)[np.ix_(block, block)]).max())
    rows.append({
        "check": "reciprocity S = S^T (600 nm taper, force_unitary=False)",
        "criterion": "physical block < 1e-6",
        "measured": asym_guided, "full_matrix": asym,
        "pass": bool(asym_guided < 1e-6),
    })
    # (b') the physical channel itself: forward and backward transmission of
    # the launched mode must agree (T12[0,0] = T21[0,0] for a reciprocal device)
    N = S.shape[0] // 2
    i_in, j_out = ports(od)
    channel = float(abs(S[j_out, i_in] - S[N + i_in, N + j_out]))
    rows.append({
        "check": "reciprocity of the physical channel |T12 - T21| (600 nm taper)",
        "criterion": "< 1e-6",
        "measured": channel, "relative_to_|S|": channel / max(abs(S[j_out, i_in]), 1e-300),
        "pass": bool(channel < 1e-6),
    })
    # (c) passivity: no *physical* input column gains power.  Berenger-mode
    # inputs are excluded on purpose - their unconjugated normalisation puts
    # |a|^2 two to three times off the power they carry (r = 2-3), so a column
    # sum over them is not a power statement.
    column = (np.abs(S) ** 2).sum(axis=0)
    phys_in = np.flatnonzero(physical(od, 0))
    gain = float(column[phys_in].max())
    rows.append({
        "check": "passivity: no physical input column of |S|^2 sums above 1",
        "criterion": "< 1.05 (unconjugated normalisation, see r)",
        "measured": gain, "columns": {int(j): float(column[j]) for j in phys_in},
        "all_columns_max": float(column.max()),
        "pass": bool(gain < 1.05),
    })
    return rows


def junction_mismatch(gap):
    """Bare Si wire -> walled wire at 400 nm: the mode-mismatch coupling.

    The paper's input is a plain wire; the gold begins where the taper does.
    The dataset family has the walls everywhere, so this junction is estimated
    from the two modes on one grid: ``|O_12 O_21| / |P_1 P_2|`` with the
    unconjugated overlap ``O = 1/2 int (E_1 x H_2)_z dA`` on the stretched
    metric.  Reflection and higher modes are neglected - an estimate.
    """
    info = plasmonic_converter_dataset_info(WAVELENGTH, gap=gap * 1e-9)()
    walled = info.get_fde_backend()
    bare = PMLBackend(
        PlasmonicSlotConverter(gap=gap * 1e-9, metal=air(), substrate=air(),
                               reference_wavelength=WAVELENGTH),
        target_neff=2.2, wavelength=WAVELENGTH, window=walled.window,
        mesh=len(walled.x), mesh_y=len(walled.y), pml_thickness=walled.pml_thickness,
        pml_edges=walled.pml_edges, num_modes=6,
    )
    t0 = time.time()
    a, _ = walled.mode_data({"w_si": W_IN}, walled.target_neff)
    b, _ = bare.mode_data({"w_si": W_IN}, 2.2)
    x, y = walled.x, walled.y
    dA = np.gradient(x)[:, None] * np.gradient(y)[None, :]

    def cross(E, H):
        return 0.5 * np.sum((E[0] * H[1] - E[1] * H[0]) * dA)

    P1, P2 = cross(a.E[0], a.H[0]), cross(b.E[0], b.H[0])
    O12, O21 = cross(a.E[0], b.H[0]), cross(b.E[0], a.H[0])
    coupling = float(abs(O12 * O21) / abs(P1 * P2))
    return {
        "walled_neff": [float(np.real(a.neff[0])), float(np.imag(a.neff[0]))],
        "bare_neff": [float(np.real(b.neff[0])), float(np.imag(b.neff[0]))],
        "coupling": coupling, "coupling_dB": 10 * np.log10(coupling),
        "seconds": time.time() - t0,
    }


# ------------------------------------------------------------------ stages


def load():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as handle:
            return json.load(handle)
    return {"paper": PAPER, "wavelength_m": WAVELENGTH}


def dump(payload):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


def path_record(res, r_in, r_out):
    od = res["od"]
    N = od["neff"].shape[1] // 2
    rec = budget(res["S"], od, r_in, r_out)
    rec.update({
        "seconds": res["seconds"], "route": res["route"],
        "sections": len(od["EME_path"]),
        "w_si_nm": [float(p[0] * 1e9) for p in od["EME_path"]],
        "neff_re": np.real(od["neff"][:, :N]).tolist(),
        "neff_im": np.imag(od["neff"][:, :N]).tolist(),
        "physical": np.array([physical(od, s) for s in range(len(od["EME_path"]))]).tolist(),
    })
    if "TE_pol" in od:
        rec["TE_pol"] = np.real(od["TE_pol"][:, :N]).tolist()
    return rec


def sweep(payload, gap, lengths_nm):
    du = DataUpdater(os.path.join(ROOT, "datasets", DATASETS[gap]))
    key = f"sweep_gap{gap}"
    store = payload.setdefault(key, {})
    print(f"\n[{gap} nm gap] taper-length sweep on {DATASETS[gap]}")
    r_in, r_out = end_ratios(payload, gap)
    for L_nm in lengths_nm:
        if str(L_nm) in store:
            print(f"  L = {L_nm:5d} nm  cached  {store[str(L_nm)]['converted_dB']:+.2f} dB")
            continue
        L = L_nm * 1e-9
        res = lumped(ParametricPath(du, schedule(L), total_length=L))
        rec = path_record(res, r_in, r_out)
        store[str(L_nm)] = rec
        dump(payload)
        print(f"  L = {L_nm:5d} nm  ({rec['sections']:3d} sections)  "
              f"converted {rec['converted_dB']:+.2f} dB (raw {rec['converted_raw_dB']:+.2f})  "
              f"other {rec['other_physical_fwd']:.3f}  reflected {rec['reflected_physical']:.4f}  "
              f"deficit {rec['deficit']:.3f}   [{rec['seconds']:.0f} s, {rec['route']}]")
    # warm timing: the 600 nm device again, nothing left to solve
    if "warm_600_seconds" not in store and "600" in store:
        res = lumped(ParametricPath(du, schedule(600e-9), total_length=600e-9))
        store["warm_600_seconds"] = res["seconds"]
        dump(payload)
        print(f"  warm re-run of 600 nm: {res['seconds']:.1f} s "
              f"(cold: {store['600']['seconds']:.0f} s)")
    return du


def gate(payload, du, gap):
    key = f"gate_gap{gap}"
    if key in payload:
        print(f"\n[{gap} nm gap] sanity gate cached")
        return
    print(f"\n[{gap} nm gap] sanity gate")
    taper = lumped(ParametricPath(du, schedule(600e-9), total_length=600e-9))
    rows = sanity_gate(du, taper)
    for row in rows:
        print(f"  {row['check']:64s} {row['measured']:.3e}  "
              f"{'ok' if row['pass'] else 'FAIL'}")
    junction = junction_mismatch(gap)
    print(f"  junction bare wire -> walled wire: {junction['coupling_dB']:+.2f} dB "
          f"({junction['bare_neff'][0]:.4f} -> {junction['walled_neff'][0]:.4f}"
          f"{junction['walled_neff'][1]:+.4f}j)   [{junction['seconds']:.0f} s]")
    payload[key] = {"rows": rows, "junction": junction}
    dump(payload)


def direct(payload, gap, lengths_nm):
    key = f"direct_gap{gap}"
    store = payload.setdefault(key, {})
    sweep_store = payload[f"sweep_gap{gap}"]
    r_in, r_out = end_ratios(payload, gap)
    de = DataExtractor(os.path.join(ROOT, "datasets", DATASETS[gap]))
    # Sections at the axis widths themselves (400, 390, ... 0 nm), so the
    # direct route sees the same grid-aligned cross sections as the dataset;
    # "half" takes every other one.  A width off the axis puts a metal edge
    # inside a cell, and that is a different (worse) discretisation.
    n_axis = len(plasmonic_converter_dataset_info(WAVELENGTH, gap=gap * 1e-9)().parameters["w_si"])
    print(f"\n[{gap} nm gap] DBEME vs direct EME")
    for L_nm in lengths_nm:
        ref = sweep_store[str(L_nm)]
        plans = [("matched", n_axis)]
        if L_nm == 600:
            plans.append(("half", n_axis // 2 + 1))
        for label, sections in plans:
            k = f"{L_nm}_{label}"
            if k in store:
                print(f"  L = {L_nm} nm, {label} slicing cached: max|dT| {store[k]['max_dT']:.2e}")
                continue
            L = L_nm * 1e-9
            geometry = DirectParametricPath(de, schedule(L), total_length=L,
                                            resolution=sections)
            geometry._verbose = False
            res = lumped(geometry)
            rec = path_record(res, r_in, r_out)
            rec["slicing"] = label
            fwd_d, fwd_g = np.array(rec["forward_out"]), np.array(ref["forward_out"])
            n = min(fwd_d.size, fwd_g.size)
            rec["max_dT"] = float(np.abs(fwd_d[:n] - fwd_g[:n]).max())
            rec["d_converted"] = float(rec["converted"] - ref["converted"])
            store[k] = rec
            dump(payload)
            print(f"  L = {L_nm} nm, {label} slicing ({sections} sections): "
                  f"direct {rec['converted_dB']:+.2f} dB vs dataset {ref['converted_dB']:+.2f} dB, "
                  f"max|dT| {rec['max_dT']:.2e}   [{rec['seconds']:.0f} s vs {ref['seconds']:.0f} s cold]")


def cell_check(payload):
    if "cell_check" in payload:
        print("\ncell check cached")
        return
    print("\ncell check (20 nm gap, suspended, grid-aligned): Si end and slot")
    out = {}
    for cell_nm in (8, 6, 5, 4):
        info = plasmonic_converter_dataset_info(WAVELENGTH, gap=20e-9, cell=cell_nm * 1e-9)()
        be = info.get_fde_backend()
        for w_nm in (400, 0):
            t0 = time.time()
            data, conf = be.mode_data({"w_si": w_nm * 1e-9}, be.target_neff)
            n = data.neff[0]
            out[f"cell{cell_nm}_w{w_nm}"] = {
                "neff": [float(np.real(n)), float(np.imag(n))],
                "confinement": float(conf[0]), "TE_pol": float(data.TE_pol[0]),
                "grid": [len(be.x), len(be.y)], "seconds": time.time() - t0,
                "loss_dB_per_um": attenuation_db_per_um(n, WAVELENGTH),
            }
            print(f"  cell {cell_nm} nm, w_si {w_nm:3d} nm: n = {np.real(n):.4f}{np.imag(n):+.4f}j  "
                  f"conf {conf[0]:.2f}  [{time.time()-t0:.0f} s]")
    payload["cell_check"] = out
    dump(payload)


# ----------------------------------------------------------------- figures


def figure_path(payload, gap):
    rec = payload[f"sweep_gap{gap}"]["600"]
    w = np.array(rec["w_si_nm"])
    re, im = np.array(rec["neff_re"]), np.array(rec["neff_im"])
    phys = np.array(rec["physical"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    fig.subplots_adjust(wspace=0.32)
    te = np.array(rec["TE_pol"]) if "TE_pol" in rec else None
    sc = None
    for j in range(re.shape[1]):
        keep = phys[:, j]
        if not keep.any():
            continue
        if te is not None:
            sc = ax1.scatter(w[keep], re[keep, j], c=te[keep, j], cmap="coolwarm",
                             vmin=0, vmax=1, s=10)
        else:
            ax1.plot(w[keep], re[keep, j], ".", ms=4)
        ax2.plot(w[keep], 4.342944819 * 2 * K0 * im[keep, j] * 1e-6, ".", ms=4,
                 label=f"branch {j}")
    if sc is not None:
        fig.colorbar(sc, ax=ax1, label="TE fraction")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.set_xlabel("Si width (nm)")
    ax1.set_ylabel("Re n_eff")
    ax1.set_title(f"physical branches along the {gap} nm-gap taper", fontsize=9)
    ax1.invert_xaxis()
    ax2.set_xlabel("Si width (nm)")
    ax2.set_ylabel("loss (dB/um)")
    ax2.set_title("metal + radiation loss of each branch", fontsize=9)
    ax2.invert_xaxis()
    ax2.legend(fontsize=7)
    save(fig, f"{TAG}_1_neff_path_gap{gap}.png")
    plt.close(fig)


def figure_sweep(payload, gaps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    fig.subplots_adjust(wspace=0.3)
    for gap, color in zip(gaps, ("C0", "C3")):
        store = payload[f"sweep_gap{gap}"]
        Ls = sorted(int(k) for k in store if k.isdigit())
        eff = [store[str(L)]["converted_dB"] for L in Ls]
        ax1.plot(Ls, eff, "o-", color=color, label=f"DBEME, {gap} nm gap (lateral model)")
        ref = PAPER[gap]
        for name, marker in (("simulated_dB", "s"), ("calculated_dB", "^"), ("measured_dB", "*")):
            if name in ref:
                ax1.plot(ref["length_nm"], ref[name], marker, color=color, mfc="none", ms=9,
                         label=f"Ono et al. {name.split('_')[0]}, {gap} nm gap (3-D device)")
        lost = [store[str(L)]["deficit"] for L in Ls]
        refl = [store[str(L)]["reflected_physical"] for L in Ls]
        other = [store[str(L)]["other_physical_fwd"] + store[str(L)]["berenger_fwd"] for L in Ls]
        ax2.plot(Ls, lost, "-", color=color, label=f"absorbed + radiated, {gap} nm")
        ax2.plot(Ls, other, "--", color=color, label=f"other forward branches, {gap} nm")
        ax2.plot(Ls, refl, ":", color=color, label=f"reflected, {gap} nm")
    ax1.set_xlabel("taper length (nm)")
    ax1.set_ylabel("conversion into the slot mode (dB)")
    ax1.set_title("conversion vs taper length (output-side projection)\nno optimum: the 40-step staircase mismatch is length-independent, report 12 section 3", fontsize=8)
    ax1.legend(fontsize=6.5)
    ax2.set_xlabel("taper length (nm)")
    ax2.set_ylabel("fraction of launched |a|^2")
    ax2.set_title("where the rest goes", fontsize=9)
    ax2.legend(fontsize=6.5)
    save(fig, f"{TAG}_2_length_sweep.png")
    plt.close(fig)


def figure_direct(payload, gap):
    key = f"direct_gap{gap}"
    if key not in payload or not payload[key]:
        return
    store, sweep_store = payload[key], payload[f"sweep_gap{gap}"]
    fig, axes = plt.subplots(1, len(store), figsize=(3.4 * len(store), 3.2), squeeze=False)
    for ax, (k, rec) in zip(axes[0], sorted(store.items())):
        L_nm = k.split("_")[0]
        ref = sweep_store[L_nm]
        fwd_d, fwd_g = np.array(rec["forward_out"]), np.array(ref["forward_out"])
        n = min(fwd_d.size, fwd_g.size, 8)
        idx = np.arange(n)
        ax.bar(idx - 0.2, fwd_g[:n], 0.4, label=f"dataset ({ref['sections']} sec.)")
        ax.bar(idx + 0.2, fwd_d[:n], 0.4, label=f"direct ({rec['sections']} sec.)")
        ax.set_yscale("log")
        ax.set_xlabel("output branch")
        ax.set_ylabel("forward |a|^2")
        ax.set_title(f"L = {L_nm} nm, {rec['slicing']} slicing\nmax|dT| = {rec['max_dT']:.1e}", fontsize=8)
        ax.legend(fontsize=6.5)
    save(fig, f"{TAG}_3_direct_vs_dataset_gap{gap}.png")
    plt.close(fig)


# -------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS_NM))
    parser.add_argument("--direct-lengths", type=int, nargs="+", default=list(DIRECT_NM))
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--cell-check", action="store_true")
    args = parser.parse_args()

    print("Demo 4 - Si wire to plasmonic slot, lateral 2-D model (Ono et al. 2016)")
    print("lossy PML basis, scattering route, force_unitary=False")
    print("=" * 78)
    payload = load()
    t_all = time.time()
    for gap in args.gaps:
        du = sweep(payload, gap, args.lengths)
        gate(payload, du, gap)
        if not args.skip_direct:
            direct(payload, gap, args.direct_lengths)
    if args.cell_check:
        cell_check(payload)

    for gap in args.gaps:
        figure_path(payload, gap)
        figure_direct(payload, gap)
    figure_sweep(payload, args.gaps)
    payload["total_seconds_this_run"] = time.time() - t_all
    dump(payload)
    # the report links its figures from reports/output, as the other demos do
    import shutil
    from _plotting import OUTPUT_DIR
    report_dir = os.path.dirname(OUT_JSON)
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith(f"{TAG}_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))
            print(f"wrote reports/output/{name}")
    print(f"\nresults -> {OUT_JSON}   [{time.time()-t_all:.0f} s this run]")


if __name__ == "__main__":
    main()
