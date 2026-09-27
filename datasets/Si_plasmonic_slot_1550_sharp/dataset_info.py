"""Dataset: the 20 nm-gap plasmonic converter with *sharp* gold corners.

The shipped ``Si_plasmonic_slot_1550`` rounds the plates' corners (20 nm); this
is the same family with ``corner_radius = 0`` - the basis report 12 sections
1-7 were measured on, kept for the sharp-vs-rounded comparison of section 8.
"""

from dbeme.platforms import plasmonic_converter_dataset_info

WAVELENGTH = 1.55e-6
GAP = 20e-9

DatasetInfo = plasmonic_converter_dataset_info(WAVELENGTH, gap=GAP, corner_radius=0.0)
