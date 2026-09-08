"""Demo 4b: Si wire to plasmonic slot converter, SiO2-embedded (Kocabas 2017).

S. E. Kocabas, *The effect of metal thickness on Si wire to plasmonic slot
waveguide mode conversion*, arXiv:1801.00833 - Table II gives two optimised
converters, everything embedded in SiO2 with the Si and the gold centred on
each other.  Set 2 (250 nm gold): Si 400 x 725 nm -> 250 nm slot, 1700 nm
taper, gap 75 -> 125 nm, transmission ~95 % (COMSOL, 3-D).  That is a
fully specified lateral device with bound modes at both ends - which is why
it replaces the Ono device of report 12 for the sweeps.

The slot tapers independently of the Si, so the dataset has two axes,
``w_si`` and ``gap`` (``datasets/SiO2_kocabas_set2_1550``), and a device is a
path in that plane.  Lossy PML basis, scattering cascade, output-side
projection, ``force_unitary=False`` - as in report 12.

Stages (cached in ``reports/output/kocabas_converter.json``):

1. the design point: transmission into the slot mode, compared with the
   paper's ~95 % after back-propagating the 200 nm slot lead-out to the tip
   as the paper does;
2. transmission vs the Si-gold gap at the taper start (``--gaps``), the slot
   end fixed at 250 nm;
3. transmission vs taper length (``--lengths``) at the design gap;
4. sanity gate: constant-slot propagation loss, reciprocity, passivity;
5. ``--direct``: the design path re-solved section by section without the
   cache (same grid points - a finer or shifted slicing would put a metal
   edge inside a cell, report 12 section 7.3).

Run:  python examples/demo_kocabas_converter.py [--gaps 25 50 75 100 125 150]
                                                [--lengths 500 ... 3000] [--direct]
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

from _plotting import OUTPUT_DIR, save  # noqa: E402
from demo_plasmonic_converter import (  # noqa: E402
    PHYSICAL_IMAG_NEFF, budget, guided_block, lumped, physical, ports, standard_form,
)
from em_simulation import DataExtractor, DataUpdater, DirectParametricPath, ParametricPath  # noqa: E402
from em_simulation.platforms import KOCABAS_SETS, kocabas_converter_dataset_info, kocabas_path  # noqa: E402

TAG = "kocabas"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Grid pitch in nm.  It sets the dataset *and* both parameter axes: ``w_si``
#: steps ``2 cell`` and ``gap`` steps ``cell``, so that the Si edge and the
#: metal's inner edge land on cell boundaries at every grid point.  Refining
#: an axis therefore means refining the cell - see report 13 section 7.
CELL_NM = 5.0
SUFFIX = ""
DATASET = os.path.join(ROOT, "datasets", "SiO2_kocabas_set2_1550")
OUT_JSON = os.path.join(ROOT, "reports", "output", f"{TAG}_converter.json")


def use_cell(cell_nm):
    """Point the demo at the dataset for this pitch (5.0 or 2.5 nm)."""
    global CELL_NM, SUFFIX, DATASET, OUT_JSON
    CELL_NM = float(cell_nm)
    SUFFIX = "" if abs(CELL_NM - 5.0) < 1e-9 else "_c25"
    DATASET = os.path.join(ROOT, "datasets", "SiO2_kocabas_set2_1550" + SUFFIX)
    OUT_JSON = os.path.join(ROOT, "reports", "output",
                            f"{TAG}_converter{SUFFIX}.json")
SET = 2
WAVELENGTH = 1.55e-6
K0 = 2 * np.pi / WAVELENGTH
DESIGN = KOCABAS_SETS[SET]
GAPS_NM = (25, 50, 75, 100, 125, 150)
LENGTHS_NM = (500, 800, 1100, 1400, 1700, 2100, 2500, 3000)


# ------------------------------------------------------------------ pieces


def load():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as handle:
            return json.load(handle)
    return {"paper": DESIGN, "wavelength_m": WAVELENGTH, "set": SET}


def dump(payload):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


def device(du, w_gap_nm=None, l_taper_nm=None):
    funcs, length = kocabas_path(
        SET,
        w_gap=None if w_gap_nm is None else w_gap_nm * 1e-9,
        l_taper=None if l_taper_nm is None else l_taper_nm * 1e-9,
    )
    return ParametricPath(du, funcs, total_length=length), length


def record(res, extra_m):
    """Budget plus the paper's convention: back-propagate the slot lead-out."""
    od = res["od"]
    N = od["neff"].shape[1] // 2
    rec = budget(res["S"], od)
    n_out = od["neff"][-1, rec["output_mode"]]
    rec["slot_neff"] = [float(np.real(n_out)), float(np.imag(n_out))]
    rec["slot_Lp_nm"] = float(WAVELENGTH / (4 * np.pi * max(np.imag(n_out), 1e-12)) * 1e9)
    back = float(np.exp(2 * K0 * np.imag(n_out) * extra_m))
    rec["converted_at_tip"] = rec["converted"] * back
    rec["converted_at_tip_dB"] = 10 * np.log10(max(rec["converted_at_tip"], 1e-300))
    rec.update({
        "seconds": res["seconds"], "route": res["route"],
        "sections": len(od["EME_path"]),
        "w_si_nm": [float(p[0] * 1e9) for p in od["EME_path"]],
        "gap_nm": [float(p[1] * 1e9) for p in od["EME_path"]],
        "neff_re": np.real(od["neff"][:, :N]).tolist(),
        "neff_im": np.imag(od["neff"][:, :N]).tolist(),
        "physical": np.array([physical(od, s) for s in range(len(od["EME_path"]))]).tolist(),
    })
    if "TE_pol" in od:
        rec["TE_pol"] = np.real(od["TE_pol"][:, :N]).tolist()
    return rec


def line(label, rec):
    print(f"  {label:34s} slot {rec['converted_dB']:+.2f} dB at the lead-out end, "
          f"{rec['converted_at_tip_dB']:+.2f} dB ({rec['converted_at_tip']*100:.1f} %) at the tip | "
          f"reflected {rec['reflected_physical']:.4f}  other {rec['other_physical_fwd']:.4f}  "
          f"deficit {rec['deficit']:.3f}  [{rec['seconds']:.0f} s, {rec['sections']} sections]")


# ------------------------------------------------------------------ stages


def design(payload, du):
    if "design" in payload:
        print("design point cached"); return
    print("\n[design point] Set 2 as published")
    path, length = device(du)
    res = lumped(path)
    rec = record(res, DESIGN["extra"] * 1e-9)
    rec["paper_transmission"] = DESIGN["transmission"]
    payload["design"] = rec
    dump(payload)
    line("Set 2, gap 75, L 1700", rec)
    print(f"  slot mode {rec['slot_neff'][0]:.4f}{rec['slot_neff'][1]:+.4f}j, Lp {rec['slot_Lp_nm']:.0f} nm; "
          f"paper ~{DESIGN['transmission']*100:.0f} %")


def gap_sweep(payload, du, gaps_nm):
    store = payload.setdefault("gap_sweep", {})
    print("\n[gap sweep] Si-gold gap at the taper start, slot end fixed at 250 nm, L = 1700 nm")
    for g in gaps_nm:
        if str(g) in store:
            print(f"  gap {g} nm cached: {store[str(g)]['converted_at_tip_dB']:+.2f} dB"); continue
        path, length = device(du, w_gap_nm=g)
        rec = record(lumped(path), DESIGN["extra"] * 1e-9)
        store[str(g)] = rec
        dump(payload)
        line(f"gap {g} nm", rec)


def length_sweep(payload, du, lengths_nm):
    store = payload.setdefault("length_sweep", {})
    print("\n[length sweep] taper length at gap 75 -> 125 nm")
    for L in lengths_nm:
        if str(L) in store:
            print(f"  L {L} nm cached: {store[str(L)]['converted_at_tip_dB']:+.2f} dB"); continue
        path, length = device(du, l_taper_nm=L)
        rec = record(lumped(path), DESIGN["extra"] * 1e-9)
        store[str(L)] = rec
        dump(payload)
        line(f"L {L} nm", rec)


def gate(payload, du):
    if "gate" in payload:
        print("gate cached"); return
    print("\n[sanity gate]")
    rows = []
    # constant slot: propagation loss carried through the cascade
    L = 1.0e-6
    res = lumped(ParametricPath(du, {"w_si": 0.0, "gap": DESIGN["w_slot"] * 0.5e-9}, total_length=L))
    n0 = res["od"]["neff"][0, 0]
    expected = float(np.exp(-2 * K0 * np.imag(n0) * L)); measured = float(np.abs(res["S"][0, 0]) ** 2)
    rows.append({"check": "constant 250 nm slot, 1 um: T = exp(-2 k0 Im n L)", "criterion": "|T - expected| < 1e-3",
                 "expected": expected, "measured": measured, "pass": bool(abs(measured - expected) < 1e-3),
                 "neff": [float(np.real(n0)), float(np.imag(n0))]})
    path, length = device(du)
    res = lumped(path); S, od = res["S"], res["od"]; N = S.shape[0] // 2
    std = standard_form(S); block = guided_block(S, od)
    i_in, j_out = ports(od)
    channel = float(abs(S[j_out, i_in] - S[N + i_in, N + j_out]))
    block_asym = float(np.abs((std - std.T)[np.ix_(block, block)]).max()) if block.size else float("nan")
    rows.append({"check": "reciprocity of the physical channel |T12 - T21| (design path)", "criterion": "< 1e-6",
                 "measured": channel, "relative": channel / max(abs(S[j_out, i_in]), 1e-300),
                 "block": block_asym, "pass": bool(channel < 1e-6)})
    column = (np.abs(S) ** 2).sum(axis=0); phys_in = np.flatnonzero(physical(od, 0))
    rows.append({"check": "passivity: physical input columns of |S|^2", "criterion": "< 1.05",
                 "measured": float(column[phys_in].max()), "columns": {int(j): float(column[j]) for j in phys_in},
                 "pass": bool(column[phys_in].max() < 1.05)})
    for r in rows:
        print(f"  {r['check']:62s} {r['measured']:.3e}  {'ok' if r['pass'] else 'FAIL'}")
    payload["gate"] = rows
    dump(payload)


class _FixedPointPath(DirectParametricPath):
    """A direct (uncached) run over an explicit list of cross sections.

    ``DirectParametricPath`` samples the parameter functions at uniform ``z``.
    On a single-axis linear taper that happens to land on the dataset's own
    axis values; on a two-axis one it does not.  Here ``w_si`` steps 10 nm and
    ``gap`` 5 nm, the taper has 200 nm of lead-in in front of it, and uniform
    sampling puts every metal edge somewhere inside a cell - a different, and
    per report 12 section 7.3 a worse, discretisation.  This subclass takes the
    dataset path's points and its section lengths, so the only difference left
    between the two routes is where the modes came from.
    """

    def __init__(self, data_extractor, parameter_functions, points, delta_zs, **kwargs):
        delta_zs = np.asarray(delta_zs, dtype=float)
        super().__init__(data_extractor, parameter_functions,
                         total_length=float(delta_zs.sum()),
                         resolution=len(points), **kwargs)
        self._points = [tuple(float(v) for v in p) for p in points]
        self._delta_zs = delta_zs

    def calc_simulation_parameters(self):
        return list(self._points), self._delta_zs


def direct(payload):
    """Both directions of the section-placement question.

    ``aligned`` re-solves the dataset's own cross sections: that is the
    section 5.8 check, does the cache reproduce a fresh solve.  ``offgrid``
    samples the continuous device at uniform ``z``, which is what an EME with
    no dataset would naturally do, and so measures what snapping a continuous
    taper onto a 10 nm / 5 nm grid costs.
    """
    store = payload.setdefault("direct", {})
    if "converted" in store:                      # pre-2026-09-08 flat record
        payload["direct"] = store = {"offgrid": store}
        dump(payload)
    de = DataExtractor(DATASET)
    ref = payload["design"]
    funcs, length = kocabas_path(SET)
    print("\n[direct EME] the design path solved section by section, no cache")

    if "offgrid" not in store:
        geometry = DirectParametricPath(de, funcs, total_length=length,
                                        resolution=len(ref["w_si_nm"]))
        geometry._verbose = False
        store["offgrid"] = record(lumped(geometry), DESIGN["extra"] * 1e-9)
        dump(payload)

    if "aligned" not in store:
        du = DataUpdater(DATASET)
        od = ParametricPath(du, funcs, total_length=length).calc_output_data()
        geometry = _FixedPointPath(de, funcs, od["EME_path"], od["EME_delta_zs"])
        geometry._verbose = False
        store["aligned"] = record(lumped(geometry), DESIGN["extra"] * 1e-9)
        dump(payload)

    fwd_g = np.array(ref["forward_out"])
    for label in ("aligned", "offgrid"):
        rec = store[label]
        fwd_d = np.array(rec["forward_out"])
        m = min(fwd_d.size, fwd_g.size)
        rec["max_dT"] = float(np.abs(fwd_d[:m] - fwd_g[:m]).max())
        line(f"direct, {label}, {rec['sections']} sections", rec)
        print(f"    max|dT| vs dataset {rec['max_dT']:.2e}; "
              f"{rec['seconds']:.0f} s against {ref['seconds']:.0f} s cold")
    dump(payload)


# ----------------------------------------------------------------- figures


def figures(payload):
    d = payload.get("design")
    if d:
        w = np.array(d["w_si_nm"]); re_ = np.array(d["neff_re"]); im_ = np.array(d["neff_im"]); ph = np.array(d["physical"])
        z = np.arange(len(w))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4)); fig.subplots_adjust(wspace=0.32)
        for j in range(re_.shape[1]):
            keep = ph[:, j]
            if keep.any():
                ax1.plot(z[keep], re_[keep, j], ".", ms=4, label=f"branch {j}")
                ax2.plot(z[keep], 4.342944819 * 2 * K0 * im_[keep, j] * 1e-6, ".", ms=4)
        ax1.axhline(np.sqrt(2.0852), color="k", lw=0.6, ls=":"); ax1.set_xlabel("section (lead-in, taper, lead-out)"); ax1.set_ylabel("Re n_eff")
        ax1.set_title("physical branches along the Set 2 path", fontsize=9); ax1.legend(fontsize=7)
        ax2.set_xlabel("section"); ax2.set_ylabel("loss (dB/um)"); ax2.set_title("metal + radiation loss", fontsize=9)
        save(fig, f"{TAG}_1_neff_path{SUFFIX}.png"); plt.close(fig)
    gs, ls = payload.get("gap_sweep", {}), payload.get("length_sweep", {})
    if gs or ls:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4)); fig.subplots_adjust(wspace=0.3)
        if gs:
            g = sorted(int(k) for k in gs); ax1.plot(g, [gs[str(k)]["converted_at_tip_dB"] for k in g], "o-", label="DBEME, at the tip")
            ax1.plot(g, [gs[str(k)]["converted_dB"] for k in g], "s--", label="DBEME, after 200 nm of slot")
            ax1.axhline(10 * np.log10(DESIGN["transmission"]), color="C3", ls=":", label="Kocabas Set 2, ~95 % (gap 75)")
            ax1.set_xlabel("Si-gold gap at the taper start (nm)"); ax1.set_ylabel("power in the slot mode (dB)"); ax1.legend(fontsize=7)
            ax1.set_title("vs gap, L = 1700 nm", fontsize=9)
        if ls:
            L = sorted(int(k) for k in ls); ax2.plot(L, [ls[str(k)]["converted_at_tip_dB"] for k in L], "o-", label="DBEME, at the tip")
            ax2.plot(L, [ls[str(k)]["converted_dB"] for k in L], "s--", label="after 200 nm of slot")
            ax2.axhline(10 * np.log10(DESIGN["transmission"]), color="C3", ls=":", label="Kocabas Set 2, ~95 % (L 1700)")
            ax2.set_xlabel("taper length (nm)"); ax2.set_ylabel("power in the slot mode (dB)"); ax2.legend(fontsize=7)
            ax2.set_title("vs taper length, gap 75 -> 125 nm", fontsize=9)
        save(fig, f"{TAG}_2_sweeps{SUFFIX}.png"); plt.close(fig)
    import shutil
    report_dir = os.path.dirname(OUT_JSON)
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if name.startswith(f"{TAG}_") and name.endswith(".png"):
            shutil.copy2(os.path.join(OUTPUT_DIR, name), os.path.join(report_dir, name))


# -------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", type=int, nargs="*", default=list(GAPS_NM))
    parser.add_argument("--lengths", type=int, nargs="*", default=list(LENGTHS_NM))
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--cell", type=float, default=5.0, choices=[5.0, 2.5],
                        help="grid pitch in nm; 2.5 halves both parameter axes")
    args = parser.parse_args()
    use_cell(args.cell)
    print("Demo 4b - Kocabas Set 2 converter, SiO2-embedded, lossy PML basis")
    print(f"{CELL_NM:g} nm cell -> w_si axis {2*CELL_NM:g} nm, gap axis "
          f"{CELL_NM:g} nm; dataset {os.path.basename(DATASET)}")
    payload = load(); t0 = time.time()
    du = DataUpdater(DATASET)
    design(payload, du)
    gate(payload, du)
    length_sweep(payload, du, args.lengths)
    gap_sweep(payload, du, args.gaps)
    if args.direct:
        direct(payload)
    figures(payload)
    payload["total_seconds_this_run"] = time.time() - t0
    dump(payload)
    print(f"\nresults -> {OUT_JSON}   [{time.time()-t0:.0f} s this run]")


if __name__ == "__main__":
    main()
