"""Dataset: the Wan & Wang edge coupler tip on a 2.5 nm width lattice, 1310 nm.

The main dataset's platform with both width axes refined to 2.5 nm over
90-260 nm (tip_step): the first stretch of the path, where the tip mode
sits near cut-off, walked on a finer staircase (studies/edge_coupler/tip.py).
"""

from dbeme.platforms import wan2025_edge_coupler_dataset_info

WAVELENGTH = 1.3100000000000002e-06

DatasetInfo = wan2025_edge_coupler_dataset_info(WAVELENGTH, substrate=False, tip_step=2.5e-9)
