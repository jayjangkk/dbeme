"""Report 13 design point under the 2026-09-30/10-01 fixes, warm and read-only.

    python examples/kocabas_gauge_check.py [dataset ...]

Configurations of the cascade, each with the interface column cap on and
off (off is the default since 2026-09-30; it was on for lossy bases):

* ``upstream``   - two-mask sign gauge and upstream's reflection block (what
  report 13 was written with);
* ``one gauge``  - one sign per mode in both overlap sets, upstream's
  reflection block;
* ``continuity reflection`` - one gauge and the reflection block from field
  continuity, no reciprocal projection (the default of 2026-09-30);
* ``current``    - the same with the reciprocal projection
  (``SingleEME.INTERFACE_RECIPROCAL``, the default since 2026-10-01);
* ``current + self-overlap estimate`` - the same with
  ``SingleEME.INTERFACE_SELF_OVERLAP = "estimate"`` (opt-in);
* ``..., reflections zeroed`` - the same transmission blocks with both
  reflection blocks set to zero: the transmission chain alone.

Each row records the tip, the deficit, the lumped passivity (largest
physical-input column of ``|S|^2``, all outputs), the same over physical
outputs only, both again for the physical inputs at the far end, and the
launched channel's reciprocity.

README, "What changed relative to upstream": "One sign per mode in both
overlap sets", "Interface reflection block", "Interface column cap is
opt-in" and "Reciprocal interface matrix".  The upstream pieces are
frozen copies: the two-mask rule from ``studies/edge_coupler/archive_checks.py``
and the reflection assembly below.  Datasets are opened in test mode; a
dataset without pickles is skipped (``DataUpdater`` would create empty ones).

Writes reports/output/kocabas_gauge_check.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "studies", "edge_coupler"))

from dbeme.data_updater.data_updater import DataUpdater  # noqa: E402
from dbeme.geometry.geometry import Geometry  # noqa: E402
from dbeme.propagator.single_propagator.single_eme import SingleEME, cap_columns  # noqa: E402
from dbeme.validation import lumped_smatrix  # noqa: E402
import demo_kocabas_converter as dk  # noqa: E402
from archive_checks import _two_mask_rule  # noqa: E402

OUT = os.path.join(ROOT, "reports", "output", "kocabas_gauge_check.json")
DATASETS = ["SiO2_kocabas_set2_1550", "SiO2_kocabas_set2_1550_hs_c20", "SiO2_kocabas_set2_1550_hs_c20_N40",
            "SiO2_kocabas_set2_1550_c25", "SiO2_kocabas_set2_1550_r100-300_2.5+y95-135_2.5_c20",
            "SiO2_kocabas_set2_1550_fem_c20_hs1_w2.5"]


def upstream_interface_smatrix(self):
    """The direct-route interface matrix before 2026-09-30: reflection
    ``1/2 (O_ab^T - O_ba) T`` and ``-R21`` in the (1,2) block."""
    rcond = self.INTERFACE_RCOND
    ab, ba = self.overlap_forward_ab, self.overlap_forward_ba
    T12 = self._calc_transmission_matrix(ab, ba, rcond)
    T21 = self._calc_transmission_matrix(ba, ab, rcond)
    R12 = 0.5 * (np.transpose(ab, (0, 2, 1)) - ba) @ T12
    R21 = 0.5 * (np.transpose(ba, (0, 2, 1)) - ab) @ T21
    N = self.mode_count
    S = np.zeros((len(ab), 2 * N, 2 * N), dtype=complex)
    S[:, :N, :N], S[:, :N, N:], S[:, N:, :N], S[:, N:, N:] = T12, -R21, R12, T21
    return cap_columns(S) if self._column_cap_enabled() else S


def transmission_only_smatrix(self):
    """The current interface matrix with both reflection blocks zeroed: what
    the transmission chain alone does to passivity."""
    T12, T21, _, _ = self._calc_interface_blocks(self.INTERFACE_RCOND)
    N = self.mode_count
    S = np.zeros((len(T12), 2 * N, 2 * N), dtype=complex)
    S[:, :N, :N], S[:, N:, N:] = T12, T21
    return cap_columns(S) if self._column_cap_enabled() else S


ONE_GAUGE = Geometry._equalize_overlap_phase
CURRENT = SingleEME._calc_interface_Smatrix
ESTIMATE = {"INTERFACE_SELF_OVERLAP": "estimate"}
# label: (sign gauge, interface matrix, SingleEME switches)
CONFIGS = {
    "upstream": (_two_mask_rule, upstream_interface_smatrix, {}),
    "one gauge": (ONE_GAUGE, upstream_interface_smatrix, {}),
    "continuity reflection": (ONE_GAUGE, CURRENT, {"INTERFACE_RECIPROCAL": False}),
    "current": (ONE_GAUGE, CURRENT, {}),
    "current + self-overlap estimate": (ONE_GAUGE, CURRENT, ESTIMATE),
    "current, reflections zeroed": (ONE_GAUGE, transmission_only_smatrix, {}),
    "current + self-overlap estimate, reflections zeroed": (ONE_GAUGE, transmission_only_smatrix, ESTIMATE),
}
SWITCHES = ("INTERFACE_COLUMN_CAP", "INTERFACE_RECIPROCAL", "INTERFACE_SELF_OVERLAP")


def design_point(du):
    path = dk.device(du)[0]
    od = path.calc_output_data()
    S = np.asarray(lumped_smatrix(path, force_unitary=False))
    rec = dk.record({"S": S, "od": od, "seconds": 0, "route": "direct"}, dk.DESIGN["extra"] * 1e-9)
    column = np.sum(np.abs(S) ** 2, axis=0)
    phys_in = np.flatnonzero(dk.physical(od, 0))
    N = S.shape[0] // 2
    # physical outputs: forward at the last section, backward at the first;
    # far-end inputs: backward physical modes entering at the last section
    phys_out = np.concatenate([dk.physical(od, len(od["EME_path"]) - 1), dk.physical(od, 0)])
    phys_far = N + np.flatnonzero(dk.physical(od, len(od["EME_path"]) - 1))
    i, j = rec["input_mode"], rec["output_mode"]
    return {"sections": len(od["EME_path"]), "at_tip": float(rec["converted_at_tip"]),
            "deficit": float(rec["deficit"]), "reflected_physical": float(rec["reflected_physical"]),
            "passivity": float(column[phys_in].max()),
            "passivity_physical_outputs": float((np.abs(S[phys_out][:, phys_in]) ** 2).sum(axis=0).max()),
            "passivity_far_end": float(column[phys_far].max()),
            "passivity_far_end_physical_outputs": float((np.abs(S[phys_out][:, phys_far]) ** 2).sum(axis=0).max()),
            "reciprocity": float(abs(S[j, i] - S[N + i, N + j]) / abs(S[j, i]))}


def main():
    names = sys.argv[1:] or DATASETS
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rule0, smat0 = Geometry._equalize_overlap_phase, SingleEME._calc_interface_Smatrix
    switches0 = {k: getattr(SingleEME, k) for k in SWITCHES}
    try:
        for name in names:
            d = os.path.join(ROOT, "datasets", name)
            if not os.path.exists(os.path.join(d, "overlap.pkl")):
                print(f"{name}: no pickles, skipped")
                continue
            du = DataUpdater(d, cache_size=1)
            du._is_testmode = True
            suffix = name[len("SiO2_kocabas_set2_1550"):].lstrip("_")
            if suffix:
                dk.use_cell(5.0, suffix=suffix, axes="half_slot" if "half_slot" in du.parameter_names else "gap")
            row = {}
            for label, (rule, smat, switches) in CONFIGS.items():
                Geometry._equalize_overlap_phase, SingleEME._calc_interface_Smatrix = rule, smat
                for cap in ("auto", False):
                    for k in SWITCHES:
                        setattr(SingleEME, k, switches0[k])
                    for k, v in switches.items():
                        setattr(SingleEME, k, v)
                    SingleEME.INTERFACE_COLUMN_CAP = cap
                    row[f"{label}, cap {'on' if cap else 'off'}"] = r = design_point(du)
                    print(f"{name[22:] or '(base)':32s} {label:50s} cap {'on ' if cap else 'off'}: tip {100 * r['at_tip']:7.3f} %  "
                          f"deficit {r['deficit']:+.4f}  passivity {r['passivity']:.3f} "
                          f"(physical outputs {r['passivity_physical_outputs']:.3f}; far end {r['passivity_far_end']:.3f} / "
                          f"{r['passivity_far_end_physical_outputs']:.3f})  reciprocity {r['reciprocity']:.1e}",
                          flush=True)
            out[name] = row
            json.dump(out, open(OUT, "w"), indent=1)
    finally:
        Geometry._equalize_overlap_phase, SingleEME._calc_interface_Smatrix = rule0, smat0
        for k in SWITCHES:
            setattr(SingleEME, k, switches0[k])


if __name__ == "__main__":
    main()
