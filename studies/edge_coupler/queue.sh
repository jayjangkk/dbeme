#!/usr/bin/env bash
# Run a sequence of solver jobs on one CPU block, one after another, after an
# optional process (matched by command-line substring) has finished.
#   bash studies/edge_coupler/queue.sh <cpus> <wait-for-substring|-> "<script args>" ...
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CPUS="$1"; shift
WAIT="$1"; shift
if [ "$WAIT" != "-" ]; then
  while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*$WAIT*' }) { exit 0 } else { exit 1 }"; do
    sleep 30
  done
fi
for job in "$@"; do
  set -- $job
  script="$1"; shift
  tag="$(echo "$script $*" | tr ' /' '__')"
  echo "[$(date +%H:%M:%S)] start $script $*" >> "$ROOT/reports/output/edge/queue_${CPUS}.log"
  "$ROOT/.venv/Scripts/python" "$ROOT/examples/run_solver_job.py" --cpus "$CPUS" "$ROOT/studies/edge_coupler/$script" "$@" \
      > "$ROOT/reports/output/edge/job_${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $script $* (exit $?)" >> "$ROOT/reports/output/edge/queue_${CPUS}.log"
done
