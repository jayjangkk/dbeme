#!/bin/sh
# Task 11 Phase 2 coupler campaign. Usage: sh studies/ring/campaign.sh <N modes>
# Every run goes through the launcher (E-core rule). Datasets
# Si_ring_coupler_220nm_<lambda> must have been written with mode_numbers=N.
N=${1:?mode count}
cd /c/Users/Jay/Projects/DBEME
PY=.venv/Scripts/python
OUT=/c/Users/Jay/Projects/DBEME/reports/output/ring
run() { echo "== $*"; $PY examples/run_solver_job.py studies/ring/build_coupler.py "$@" 2>&1 | grep -v -i warning; }
run --dataset Si_ring_coupler_220nm_1550 --radius-um 10 --gaps 150,200,250,300,350,400 --out $OUT/coupler_1550.json --slicing --direct 300
run --dataset Si_ring_coupler_220nm_1530 --radius-um 10 --gaps 150,200,250,300,350,400 --out $OUT/coupler_1530.json
run --dataset Si_ring_coupler_220nm_1570 --radius-um 10 --gaps 150,200,250,300,350,400 --out $OUT/coupler_1570.json
run --dataset Si_ring_coupler_220nm_1550 --radius-um 5 --gaps 200,300 --out $OUT/coupler_1550_R5.json
run --dataset Si_ring_coupler_220nm_1550_c20 --radius-um 10 --gaps 200 --out $OUT/coupler_1550_c20.json
echo CAMPAIGN DONE
