"""Dataset: the 20 nm-gap plasmonic converter at a 4 nm cell.

Convergence check for ``Si_plasmonic_slot_1550`` (5 nm).  Same family and
window; a finer grid, and therefore - the axis is snapped to ``2 * cell`` so
that every edge sits on a grid point - an 8 nm width axis.  A solve here costs
~80 s against ~40 s; build it only for a full-sweep convergence check.
"""

from em_simulation.platforms import plasmonic_converter_dataset_info

WAVELENGTH = 1.55e-6
GAP = 20e-9

DatasetInfo = plasmonic_converter_dataset_info(WAVELENGTH, gap=GAP, cell=4e-9)
