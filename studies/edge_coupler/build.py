"""Cold-build the edge-coupler path points of one wavelength's dataset.

    python examples/run_solver_job.py --cpus 0-3 studies/edge_coupler/build.py 1310 [variants...]

Solves every cross section the bilayer (and, by default, the conventional
220 nm) path visits and links it to its path neighbours; the pickles are
saved after every point, so a killed job resumes where it stopped.  A log
line per variant goes to ``reports/output/edge/build_<nm>.json``.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as dv  # noqa: E402


def main():
    wl = int(sys.argv[1])
    variants = sys.argv[2:] or ["bilayer", "conventional"]
    du = dv.open_dataset(wl, substrate=False)
    log_path = os.path.join(dv.ROOT, "reports", "output", "edge", f"build_{wl}.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    for variant in variants:
        before = len(du.neff)
        t0 = time.time()
        path, _ = dv.build_path(du, variant, verbose=True)
        od = path.calc_output_data()
        dt = time.time() - t0
        solved = len(du.neff) - before
        entry = {"variant": variant, "seconds": dt, "points_solved": solved,
                 "eme_sections": len(od["EME_path"]), "points_total": len(du.neff),
                 "per_point_s": dt / solved if solved else None,
                 "abrupt": [[list(map(float, a)), list(map(float, b))] for a, b in path.abrupt_interfaces]}
        print(json.dumps(entry), flush=True)
        log.append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
