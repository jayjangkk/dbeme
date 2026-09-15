"""Dataset: Kocabas Set 2, 5 nm grid, gold corners rounded 20 nm, (w_si, half_slot) axes.

The like-for-like baseline of the wall-refined dataset
``SiO2_kocabas_set2_1550_r100-300_1+y95-135_1_c20``: the same rounded
geometry (radius 20 nm, ENZ-safe fill floor 0.06 - report 13 section 7) and
the same parameterisation, with the wall stepping 5 nm.  A sharp metal wedge
does not converge as its corner is resolved (the 200 nm TE branch moved 0.05
from 5 nm to 1 nm cells); a 20 nm arc moves 0.012, so it is the geometry on
which a wall refinement can be compared at all.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    corner_radius=20e-9, fill_floor=0.06, axes="half_slot",
)
