"""Dataset: Sacher bi-level taper (220 nm Si core on a 90 nm slab) at 1310 nm.

O-band point for the wavelength sweep in ``examples/sweep_wavelength.py``.

The platform - stack, cross section, grid and window - is defined once in
``em_simulation/platforms.py``; this file pins the one thing that makes a
dataset a dataset, its wavelength. A DBEME dataset is keyed to the mode problem
and the wavelength is part of that key, so a wavelength sweep means one dataset
per wavelength rather than a wavelength axis: overlaps are only ever used
between adjacent points on a propagation path, light never propagates from one
wavelength to another, and a lambda axis would therefore solve two extra
neighbours per point that nothing can use.
"""

from em_simulation.platforms import sacher_bilevel_dataset_info

WAVELENGTH = 1.3100000000000002e-06

DatasetInfo = sacher_bilevel_dataset_info(WAVELENGTH)
