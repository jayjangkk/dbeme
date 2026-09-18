"""Run a long solver script as a job that is not parked on the E-cores.

    python run_solver_job.py [--cpus 0-15] <script.py> [script args...]

Before the script runs, the process switches Windows 11 power throttling
(EcoQoS) off for itself, raises its priority class to ABOVE_NORMAL, pins
itself to the given logical CPUs - by default 0-15, the eight P-cores of the
i9-13900HX this project runs on - and caps the BLAS thread pools at that
many threads so they do not oversubscribe the pinned cores.

Why (2026-09-18): a detached FEM run (WMI-created ``cmd -> python``, the way
long runs survive session boundaries here) solved cross sections at ~420 s
each while the identical solve in a shell-spawned process took 60 s.  The
run's main thread was busy only 45 % of the wall time and the process was in
the same session with the same priority, affinity and environment as the
fast ones: the scheduler had placed its single-threaded eigensolve on the
E-cores at reduced clocks.  Pinned to the P-cores the rate returned to ~65 s
per point.  The earlier FEM taper run of report 13 (82 756 s for 112 points,
739 s each against 161 s of measured solve time) had suffered the same.
Multi-threaded finite-difference runs spread over all cores and did not show
it; a single-threaded sparse factorisation is what gets parked.

On another machine give ``--cpus`` its P-cores, or ``--cpus all`` to only
apply the throttling and priority settings.
"""
import argparse
import ctypes
import os
import runpy
import sys


def _cpu_list(spec, count):
    if spec == "all":
        return list(range(count))
    cpus = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return [c for c in cpus if c < count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpus", default="0-15", help="logical CPUs to pin to, e.g. 0-15 or 0,2,4 or all (default 0-15)")
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()

    cpus = _cpu_list(opts.cpus, os.cpu_count() or 1)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(len(cpus)))

    if sys.platform == "win32":
        import ctypes.wintypes as w

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p       # a 64-bit pseudo-handle; the default int return truncates it
        k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = k32.GetCurrentProcess()
        state = PowerThrottlingState(1, 0x1, 0)               # control EXECUTION_SPEED, state 0: throttling off
        ok_throttle = bool(k32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state)))
        ok_priority = bool(k32.SetPriorityClass(handle, 0x8000))   # ABOVE_NORMAL_PRIORITY_CLASS
        print(f"power throttling off: {ok_throttle}; priority above normal: {ok_priority}", flush=True)

    import psutil

    psutil.Process().cpu_affinity(cpus)
    print(f"affinity: {psutil.Process().cpu_affinity()}; BLAS threads {os.environ['OPENBLAS_NUM_THREADS']}", flush=True)

    script = os.path.abspath(opts.script)
    os.chdir(os.path.dirname(script))
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + opts.args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
