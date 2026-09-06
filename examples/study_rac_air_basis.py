"""Why the air-clad RAC *dataset* fails its sanity gate, and what fixes it.

The air-clad direct-EME results are sound - reciprocity 2.5e-4, reflection
symmetry 4.3e-3, and a 6-mode and a 3-mode basis agree on the splitting ratio
to 0.6 points.  The cached dataset built from the same cross sections is not:
power conservation comes out at 5e2 instead of 1e-1.

The difference is what the two paths do with the mode basis.  Air on a buried
oxide box cuts off at ``n_sub = 1.444``, so a 6-mode solve returns 3 bound modes
and 3 modes below cutoff whose fields are set by the window rather than by the
waveguide.  Two of those sit at ``n_eff = 1.19`` and ``1.17`` - a near-degenerate
pair.  ``_pin_gauge`` fixes one global phase *per mode*, which is enough for a
non-degenerate mode (the gauge group is just ``{+-1}``) but cannot pin a
degenerate subspace, where the freedom is a continuous complex rotation.  A
direct path escapes because it solves each point once per process and keeps one
gauge throughout; a dataset persists overlaps and must have every stored overlap
touching a grid point agree about that subspace - which is exactly the open item
in ``CLAUDE.md`` section 5.6.

So the fix is not to pin harder, it is to stop storing modes that are not
resolved in the first place.  This script rebuilds the air dataset with a
bound-mode-only basis and re-runs the gate.

Run:  python examples/study_rac_air_basis.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["RAC_CLAD"] = "air"

import demo_rapid_adiabatic_coupler as rac  # noqa: E402
from em_simulation import DataExtractor, DataUpdater, ParametricPath  # noqa: E402
from em_simulation.fde import EmepyFDE  # noqa: E402
from em_simulation.validation import (  # noqa: E402
    check_branch_tracking,
    check_power_conservation,
    check_reciprocity,
    check_reflection_symmetry,
    format_table,
    guided_throughout,
)

#: Bound modes only.  At 1550 nm the air-clad pair supports three above the
#: 1.444 box cutoff; everything below it is a window artefact.
BOUND_MODES = 3

DATASET = os.path.join(rac.ROOT, "datasets", "Si_rac_region3_220nm_air_bound")
LENGTHS = (4e-6, 8e-6, 11.5e-6, 23e-6, 40e-6)


def _dataset_info(folder):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dataset_info", os.path.join(folder, "dataset_info.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DatasetInfo()


def main():
    print("Air-clad RAC: a bound-mode-only basis")
    print("=" * 74)

    with open(
        os.path.join(rac.ROOT, "reports", "output", "rac_air_results.json"),
        encoding="utf-8",
    ) as handle:
        published = json.load(handle)
    best_tilt = published["tilt_scan"]["best_tilt"]
    six_mode = published["checks"]
    print(f"tilt {best_tilt:+.2f} degrees, L = {rac.LENGTH*1e6:.1f} um\n")

    print("the 6-mode dataset, as built by the demo:")
    for check in six_mode:
        mark = "pass" if check["passed"] else "FAIL"
        print(f"   {check['name']:36s} {check['measured']:10.3e}  {mark}")

    info = _dataset_info(DATASET)
    print(f"\nrebuilding with {info.get_mode_numbers()} modes "
          f"(cutoff n = {info.get_cladding_index():.4f})")

    updater = DataUpdater(DATASET)
    t0 = time.time()
    ds_path = ParametricPath(updater, rac.schedule(best_tilt),
                             total_length=rac.LENGTH)
    ds_path.calc_output_data()
    build_seconds = time.time() - t0
    updater.save_data()

    neff = np.real(ds_path.output_data["neff"])
    n = neff.shape[1] // 2
    print(f"  built in {build_seconds:.0f} s, {len(updater.neff)} points")
    print(f"  neff in : {np.round(neff[0, :n], 4)}")
    print(f"  neff out: {np.round(neff[-1, :n], 4)}")
    print(f"  guided end to end: {guided_throughout(ds_path)}")

    checks = [
        check_reciprocity(ds_path),
        check_reflection_symmetry(ds_path),
        check_power_conservation(ds_path),
        check_branch_tracking(ds_path)[0],
    ]
    print("\n" + format_table(checks))

    backend = EmepyFDE(
        cross_section=info.get_cross_section(),
        parameter_names=list(info.get_parameter_names()),
        num_modes=BOUND_MODES,
        wavelength=info.get_wavelength(),
        window=info._window,
        mesh=200,
    )
    extractor = DataExtractor(DATASET, backend=backend, cache_size=16384)
    extractor.mode_numbers = BOUND_MODES
    t0 = time.time()
    direct_path = rac.build_direct(extractor, rac.schedule(best_tilt))
    direct_seconds = time.time() - t0
    print(f"\ndirect path in {direct_seconds:.0f} s")

    print(f"\n{'length':>8} {'dataset':>10} {'direct':>10} {'diff':>10}")
    dataset_curve, direct_curve = [], []
    for L in LENGTHS:
        a = rac.splitting(ds_path, length=L)["split_percent"]
        b = rac.splitting(direct_path, length=L)["split_percent"]
        dataset_curve.append(a)
        direct_curve.append(b)
        print(f"  {L*1e6:6.1f}u {a:10.4f} {b:10.4f} {abs(a-b):10.2e}")
    worst = float(np.max(np.abs(np.array(dataset_curve) - np.array(direct_curve))))
    print(f"  max |difference| = {worst:.2e} percentage points")

    payload = {
        "mode_numbers": BOUND_MODES,
        "cutoff_index": info.get_cladding_index(),
        "best_tilt": best_tilt,
        "six_mode_checks": six_mode,
        "bound_mode_checks": [
            {
                "name": c.name,
                "criterion": c.criterion,
                "measured": None if np.isnan(c.measured) else float(c.measured),
                "passed": bool(c.passed),
                "note": c.note,
            }
            for c in checks
        ],
        "lengths_um": [L * 1e6 for L in LENGTHS],
        "dataset": dataset_curve,
        "direct": direct_curve,
        "max_difference": worst,
        "dataset_build_seconds": build_seconds,
        "dataset_points": len(updater.neff),
        "direct_seconds": direct_seconds,
    }
    out = os.path.join(rac.ROOT, "reports", "output", "rac_air_basis.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote reports/output/rac_air_basis.json")
    return payload


if __name__ == "__main__":
    main()
