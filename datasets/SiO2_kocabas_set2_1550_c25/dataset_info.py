"""Dataset: Kocabas Set 2 at a 2.5 nm cell - the fine-step convergence run.

Same family and window as ``SiO2_kocabas_set2_1550`` (5 nm); half the pitch,
so the width axis runs in 5 nm steps and the gap axis in 2.5 nm steps and the
40-step staircase of the design path becomes an 80-step one.  A solve costs
about four times as much; build it only for the convergence row of report 13.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(set_number=2, wavelength=WAVELENGTH, cell=2.5e-9)
