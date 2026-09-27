"""Coupled-mode theory from the same dataset vs the DBEME cascade - `tasks/11` §4.1.

For two identical guides the point coupler's cross-coupling is

    kappa_c^2 = sin^2( sum_k pi (n_even,k - n_odd,k) dz_k / lambda )

with the supermode splitting read from the very ``n_eff`` the cascade used
(`build_coupler.py` stores it per section).  No overlaps, no cascade: this
tests the EME cascade against the mode solver's own supermode splitting.
Agreement is expected to 5 % while ``kappa_c^2 < 0.3`` (weak coupling); the
divergence beyond is CMT's, not DBEME's.

Run:  python studies/ring/cmt_check.py reports/output/ring/coupler_1550.json
"""

import json
import sys

import numpy as np


def cmt_kappa2(entry, wavelength_m):
    path = entry["path"]
    dn = np.asarray(path["neff_even"]) - np.asarray(path["neff_odd"])
    dz = np.asarray(path["delta_z_um"]) * 1e-6
    n = min(len(dn), len(dz))
    phase = np.pi * np.sum(dn[:n] * dz[:n]) / wavelength_m
    return float(np.sin(phase) ** 2), float(phase)


def main(path):
    with open(path) as handle:
        data = json.load(handle)
    wl = data["wavelength_nm"] * 1e-9
    rows = []
    print(f"{path}: R = {data['radius_um']} um, lambda = {data['wavelength_nm']:.0f} nm")
    print(f"{'g0 (nm)':>8} {'DBEME k^2':>11} {'corrected':>10} {'CMT k^2':>10} {'raw/CMT':>8} {'corr/CMT':>9} {'CMT phase':>10}")
    for entry in data["gaps"]:
        k2, phase = cmt_kappa2(entry, wl)
        kc = entry.get("kappa2_corrected", entry["kappa2"])
        ratio = entry["kappa2"] / k2 if k2 > 0 else float("nan")
        ratio_c = kc / k2 if k2 > 0 else float("nan")
        rows.append({"g0_nm": entry["g0_nm"], "kappa2_dbeme": entry["kappa2"], "kappa2_corrected": kc,
                     "kappa2_cmt": k2, "ratio_raw": ratio, "ratio_corrected": ratio_c, "cmt_phase_rad": phase})
        print(f"{entry['g0_nm']:8.0f} {entry['kappa2']:11.5f} {kc:10.5f} {k2:10.5f} {ratio:8.3f} {ratio_c:9.3f} {phase:10.4f}")
    out = path.replace(".json", "_cmt.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
