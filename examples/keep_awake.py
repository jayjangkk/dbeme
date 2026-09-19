"""Keep the system from sleeping while another process runs.

    python keep_awake.py <pid>

Holds Windows' ES_SYSTEM_REQUIRED execution-state request (the one a media
player holds during playback: the machine stays awake, the display may still
turn off) until the given process exits, then releases it and exits.  For a
solver job that was launched without ``run_solver_job.py``, which now makes
the same request itself.  Why: the 1 nm-wall gap sweep of report 13 lost 22 h
to the laptop entering modern standby with the job frozen inside it.
"""
import ctypes
import sys
import time

import psutil

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

pid = int(sys.argv[1])
k32 = ctypes.windll.kernel32
k32.SetThreadExecutionState.restype = ctypes.c_uint32
prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(f"keeping the system awake for pid {pid} (previous state {prev:#x})", flush=True)
try:
    while psutil.pid_exists(pid):
        time.sleep(30)
finally:
    k32.SetThreadExecutionState(ES_CONTINUOUS)
    print("released", flush=True)
