"""Dataset: RAC coupling region (Fargas Cabanillas 3.5.3) at 1530 nm.

One wavelength point for the splitting-ratio sweep, on the buried
SiO2-clad stack this project uses elsewhere.

The platform lives in ``em_simulation/platforms.py``; a dataset file is a
wavelength and a cladding.
"""

from em_simulation.platforms import rac_dataset_info

WAVELENGTH = 1.5300000000000002e-06

DatasetInfo = rac_dataset_info(WAVELENGTH)
