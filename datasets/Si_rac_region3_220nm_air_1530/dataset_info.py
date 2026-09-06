"""Dataset: RAC coupling region (Fargas Cabanillas 3.5.3), air clad at 1530 nm.

Air clad on a SiO2 box - the stack of the *fabricated* device, which
was e-beam written on 220 nm SOI and left unclad.  Confinement is much
stronger than in the buried case, so this is the dataset the published
numbers should be compared against.

The platform lives in ``em_simulation/platforms.py``; a dataset file is a
wavelength and a cladding.
"""

from em_simulation.platforms import rac_dataset_info

WAVELENGTH = 1.5300000000000002e-06

DatasetInfo = rac_dataset_info(WAVELENGTH, clad="air")
