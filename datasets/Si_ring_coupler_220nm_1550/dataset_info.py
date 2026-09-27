"""Bus-ring point coupler, 220 nm Si, 1550 nm.

One dataset per wavelength (see Si_pair_fulletch_220nm_1310); the platform
is em_simulation.platforms.ring_coupler_dataset_info - tasks/11 section 2.1.
"""

from em_simulation.platforms import ring_coupler_dataset_info

WAVELENGTH = 1.55e-6
DatasetInfo = ring_coupler_dataset_info(WAVELENGTH)
