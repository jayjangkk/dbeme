"""Bus-ring point coupler, 220 nm Si, 1550 nm, 20 nm cells with 20 nm gap steps.

The coarse-grid variant for the gap-axis / cell convergence row of tasks/11
section 2 (the axis is tied to the cell, see ring_coupler_dataset_info).
"""

from em_simulation.platforms import ring_coupler_dataset_info

WAVELENGTH = 1.55e-6
DatasetInfo = ring_coupler_dataset_info(WAVELENGTH, cell=20e-9, fine_step=20e-9, coarse_step=40e-9)
