"""Bus-ring point coupler, 220 nm Si, 1550 nm, 20 modes - the mode-count convergence set (tasks/11 section 2)."""

from dbeme.platforms import ring_coupler_dataset_info

WAVELENGTH = 1.55e-6
DatasetInfo = ring_coupler_dataset_info(WAVELENGTH, mode_numbers=20)
