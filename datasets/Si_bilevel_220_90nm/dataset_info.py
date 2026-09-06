"""Dataset: Sacher bi-level taper (220 nm Si core on a 90 nm slab) at 1550 nm.

The TM0 -> TE1 polarization rotator stage, at the paper's design
wavelength.

The platform - stack, cross section, grid and window - is defined once in
``em_simulation/platforms.py``; this file pins the one thing that makes a
dataset a dataset, its wavelength. A DBEME dataset is keyed to the mode problem
and the wavelength is part of that key, so a wavelength sweep means one dataset
per wavelength rather than a wavelength axis: overlaps are only ever used
between adjacent points on a propagation path, and light never propagates from
one wavelength to another, so a lambda axis would solve two extra neighbours per
point that nothing can use.
"""

from em_simulation.platforms import sacher_bilevel_dataset_info

WAVELENGTH = 1.55e-06

DatasetInfo = sacher_bilevel_dataset_info(WAVELENGTH)
