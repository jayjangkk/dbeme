#!/usr/bin/env bash
# Set up the virtualenv for DBEME.  Run from the project root.
set -e
PY=${PYTHON:-python}
$PY -m venv .venv
VENV_PY=".venv/Scripts/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/bin/python"

$VENV_PY -m pip install --upgrade pip
$VENV_PY -m pip install numpy scipy matplotlib tqdm pyyaml shapely pytest
$VENV_PY -m pip install ElectroMagneticPythonGpu
# --no-deps: emepy pins simphony 0.6 / tidy3d-beta for parts we do not use.
$VENV_PY -m pip install emepy --no-deps

# Material data. Optional: materials.py falls back to built-in Sellmeier
# formulas if PyOptik or its snapshot is missing.
$VENV_PY -m pip install PyOptik
echo "Downloading the refractiveindex.info snapshot (one-off, ~100 MB)..."
$VENV_PY -c "from PyOptik import download_snapshot; download_snapshot()"   || echo "  snapshot download failed - the built-in Sellmeier fallbacks will be used"

$VENV_PY -m pytest tests -q
