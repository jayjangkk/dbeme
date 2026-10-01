"""Dataset: the Wan & Wang edge coupler tip on the full stack, 1310 nm.

Si substrate under the 2 um BOX, taken into a 0.5 um Si-only PML so the
grid and the stretched layers coincide with the oxide-stack datasets';
only the facet-to-join stretch of the path is solved here, for a
converged fibre launch and exact substrate leakage in the tip
(studies/edge_coupler/fullstack_tip.py, reports/22).
"""

from dbeme.platforms import wan2025_edge_coupler_dataset_info

WAVELENGTH = 1.3100000000000002e-06

DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=True, bottom_pml=0.5e-6)
