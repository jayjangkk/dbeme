"""Dataset: Wan & Wang 2025 bilayer double-tip edge coupler at 1300 nm.

SOI 220 nm, partially etched to 150 nm, oxide-only substrate (the 2 um BOX
runs into an oxide PML), 2 um top oxide, air above; lossy PML basis with a
point-by-point shift-invert target at the local fundamental.  The Si
substrate is not in this basis: inside the window it carries a PML-guided
band (Re n 1.9-3.2) that no single set can clear at both ends of the path,
so substrate leakage is added perturbatively from full-stack solves of the
weakly bound points (tasks/18, Phase 0.3; reports/22).

The platform - stack, cross section, grid, target rule - is defined once in
dbeme/platforms.py (wan2025_edge_coupler_dataset_info); this file pins the
wavelength.
"""

from dbeme.platforms import wan2025_edge_coupler_dataset_info

WAVELENGTH = 1.3e-06

DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=False)
