"""Run a study script and count the solves it makes (task 16, Phase 0 gate).

    python scripts/warm_check.py studies/tapeout/run_sparams_sin.py --no-check

Wraps the two places work is done and prints the counts when the script
ends:

* **path solves** - ``DirectGeometry.calc_output_data``, which every path
  cache calls only on a miss.  On a warm cache this must be 0.
* **mode solves** - ``solve`` of every FDE backend (``EmepyFDE``,
  ``PMLBackend``, ``FemwellBackend`` when importable).  Several studies solve
  the two or three end cross sections of a device on every run to identify
  its ports; those are not cached by design, so this count is reported, not
  required to be 0.

The script runs as ``__main__`` with ``sys.argv`` set, from the repo root.
"""

import collections
import os
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
COUNTS = collections.Counter()


def _count(cls, name, key):
    original = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        COUNTS[key] += 1
        return original(self, *args, **kwargs)

    setattr(cls, name, wrapped)


def install():
    from em_simulation.geometry.direct_geometry import DirectGeometry
    from em_simulation.fde.emepy_fde import EmepyFDE
    from em_simulation.fde.pml import PMLBackend

    _count(DirectGeometry, "calc_output_data", "path solves")
    _count(EmepyFDE, "solve", "mode solves")
    _count(PMLBackend, "solve", "mode solves")
    try:
        from em_simulation.fde.femwell_fde import FemwellBackend
    except ImportError:
        return
    _count(FemwellBackend, "solve", "mode solves")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    install()
    sys.argv = sys.argv[1:]
    os.chdir(ROOT)
    start = time.time()
    status = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as stop:
        if isinstance(stop.code, str):
            print(stop.code)
        status = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    finally:
        print(f"\nwarm_check {script}: path solves {COUNTS['path solves']}, "
              f"mode solves {COUNTS['mode solves']}, {time.time() - start:.0f} s, "
              f"exit {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
