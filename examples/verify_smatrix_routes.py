"""Stage 4 gate: the two cascade routes on every published device.

Stage 1 proved ``SingleEME``'s ``direct`` scattering route matches the original
``transfer`` route on synthetic data and one real taper.  This checks the
*actual* device paths behind reports 01-05, on their datasets, with the same
``force_unitary`` setting each demo used - and it grades what a report can
actually quote.

What is graded and what is not
------------------------------
* **headline** (< 1e-6): the guided transmission or conversion a report quotes.
  The transfer route stores its matrices as complex64, so the floor is ~1.2e-7
  per section and accumulates to ~2.7e-6 over the RAC's 246 section matrices.
* **guided block** (< 5e-5): every entry among the modes that are guided end to
  end.  One interface on the README taper carries a radiation mode a hair above
  cutoff (condition number 3e7); there the transfer route's T->S conversion
  applies a second magnitude screen that rewrites the blocks, and the guided
  difference it leaves is 2.3e-5, localised to that interface.
* **radiation block**: reported, not graded.  Without a PML those modes are
  window artefacts whose overlaps carry no physics (``validation.py``), and the
  two routes handle their junk differently by design.

A cache that predates the current ``BASIS_CONVENTION`` is refused by the
dataset identity check.  That is correct, and it is not this script's job to
regenerate an 80-minute dataset: such a device is skipped *and counted*.

Run:  python examples/verify_smatrix_routes.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbeme import EME, DataUpdater, LinearTaper, ParametricPath  # noqa: E402
from dbeme.data_updater.dataset_identity import DatasetIdentityError  # noqa: E402
from dbeme.matrix_calculation_tool import _redheffer_star_product  # noqa: E402
from dbeme.validation import guided_throughout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADLINE = 1e-6
GUIDED = 5e-5


class _Skip(Exception):
    pass


def _open(name):
    try:
        return DataUpdater(os.path.join(ROOT, "datasets", name))
    except DatasetIdentityError as error:
        raise _Skip(f"{name}: cache predates the current basis convention") from error


def lumped(section_matrices):
    out = np.asarray(section_matrices[0])
    for step in section_matrices[1:]:
        out = _redheffer_star_product(out, np.asarray(step))
    return out


def both_routes(geometry, force_unitary, new_length=None):
    """Device S-matrix by each route, from one geometry."""
    results = {}
    for method in ("transfer", "direct"):
        propagator = EME(geometry, force_unitary=force_unitary).propagator
        if new_length is None:
            propagator.calc_Smatrix(method=method)
            sections = propagator.smatrix
        else:
            sections = propagator._find_Smatrix_new_length(new_length, method=method)
        results[method] = lumped(sections)
    return results["transfer"], results["direct"]


def report(name, transfer, direct, guided, headline=None):
    """Grade the guided block and the headline; report the full-matrix diff."""
    modes = transfer.shape[0] // 2
    both = list(guided) + [g + modes for g in guided]
    block = np.ix_(both, both)
    guided_diff = float(np.abs((transfer - direct)[block]).max())
    full_diff = float(np.abs(transfer - direct).max())
    ok = guided_diff < GUIDED
    line = (f"  {name:34s} guided |dS| {guided_diff:8.2e}  "
            f"(full {full_diff:8.2e})  {'ok' if ok else 'FAIL'}")
    if headline is not None:
        label, index = headline
        t, d = abs(transfer[index]), abs(direct[index])
        head_ok = abs(t - d) < HEADLINE
        ok = ok and head_ok
        line += f"   {label} {t:.9f} vs {d:.9f} {'ok' if head_ok else 'FAIL'}"
    print(line)
    return ok


def check(name, geometry, headline, lengths=(), force_unitary=True):
    """Both routes at the demo's own length, then at rescaled lengths."""
    guided = guided_throughout(geometry)
    results = []
    tr, di = both_routes(geometry, force_unitary)
    results.append(report(name, tr, di, guided, headline))
    for length in lengths:
        tr, di = both_routes(geometry, force_unitary, new_length=length)
        results.append(report(f"  rescaled to {length*1e6:.1f} um", tr, di, guided, headline))
    return results


def main():
    print("Stage 4 gate - transfer vs direct on the published device paths")
    print(f"graded: headline < {HEADLINE:.0e}, guided block < {GUIDED:.0e}; "
          "radiation block reported only")
    print("=" * 78)
    passed, skipped, devices = [], [], {}
    TE0 = ("TE0", (0, 0))
    S00 = ("S00", (0, 0))

    # ---- README: linear taper, then the quintic Bezier S-bend ---------------
    try:
        t0 = time.time()
        du = _open("Si_fulletch_220nm")
        taper = LinearTaper(du, input_width=0.5e-6, output_width=1.2e-6, length=10e-6)
        taper.calc_output_data()
        devices["taper"] = taper
        passed += check("linear taper 0.5->1.2 um, 10 um", taper, TE0, (4.9e-6, 20e-6))
        print(f"    [{time.time()-t0:.1f} s]")

        t0 = time.time()
        import demo_bezier_bend as bz
        s, kappa = bz.bezier_curvature(bz.quintic_s_bend(bz.SPAN, bz.OFFSET))
        bend, _ = bz.build(du, s, kappa)
        devices["bezier"] = bend
        passed += check("quintic Bezier S-bend, 1.0 um guide", bend, TE0)
        print(f"    [{time.time()-t0:.1f} s]")
    except _Skip as why:
        skipped.append(str(why))
        print(f"  SKIPPED  {why}")

    # ---- 01: adiabatic coupler ----------------------------------------------
    try:
        t0 = time.time()
        import demo_adiabatic_coupler as ac
        coupler = ac.build(_open("Si_pair_fulletch_220nm"))
        coupler.calc_output_data()
        devices["coupler"] = coupler
        passed += check("adiabatic coupler, 100 um", coupler, S00, (300e-6,))
        print(f"    [{time.time()-t0:.1f} s]")
    except _Skip as why:
        skipped.append(str(why))
        print(f"  SKIPPED  {why}")

    # ---- 02: polarization rotator -------------------------------------------
    try:
        t0 = time.time()
        import demo_polarization_rotator as pr
        rotator = pr.build(_open("Si_bilevel_220_90nm"))
        rotator.calc_output_data()
        devices["rotator"] = rotator
        passed += check("bi-level rotator, 100 um", rotator, S00, (50e-6,))
        print(f"    [{time.time()-t0:.1f} s]")
    except _Skip as why:
        skipped.append(str(why))
        print(f"  SKIPPED  {why}")

    # ---- 05: rapid adiabatic coupler, oxide ---------------------------------
    try:
        t0 = time.time()
        os.environ["RAC_CLAD"] = "oxide"
        import demo_rapid_adiabatic_coupler as rac
        path = ParametricPath(_open("Si_rac_region3_220nm"), rac.schedule(-2.5),
                              total_length=rac.LENGTH)
        path.calc_output_data()
        devices["RAC"] = path
        passed += check("RAC region III, -2.5 deg, 11.5 um", path, S00, (4e-6, 23e-6, 40e-6))
        print(f"    [{time.time()-t0:.1f} s]")
    except _Skip as why:
        skipped.append(str(why))
        print(f"  SKIPPED  {why}")

    # ---- the sanity gate, with the projection off ----------------------------
    print("\nwith force_unitary=False (what the sanity gate measures):")
    for name, geometry in devices.items():
        resolved = EME(geometry, force_unitary=False).propagator.resolve_method()
        tr, di = both_routes(geometry, False)
        passed.append(report(f"{name}  [auto -> {resolved}]", tr, di,
                             guided_throughout(geometry)))

    print()
    for why in skipped:
        print(f"  not checked: {why}")
    if not passed:
        print("NOTHING CHECKED - every cache was stale")
        return 2
    if all(passed):
        print(f"ALL {len(passed)} CHECKS AGREE ({len(devices)} devices, "
              f"{len(skipped)} skipped)")
        return 0
    print(f"{sum(not p for p in passed)} of {len(passed)} checks DISAGREE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
