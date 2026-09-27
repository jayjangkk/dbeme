#!/bin/sh
# Mode-count convergence of the bus-ring coupler (tasks/11 section 2): the
# same short path (g0 = 400 nm, R = 10 um) at N = 6, 12, 20, 30 modes.
cd /c/Users/Jay/Projects/DBEME
OUT=/c/Users/Jay/Projects/DBEME/reports/output/ring
for N in 6 12 20 30; do
  echo "== N=$N"
  .venv/Scripts/python examples/run_solver_job.py studies/ring/build_coupler.py --dataset Si_ring_coupler_220nm_1550_N$N --radius-um 10 --gaps 400 --out $OUT/conv_N$N.json 2>&1 | grep -v -i warning
done
echo CONVERGENCE DONE
