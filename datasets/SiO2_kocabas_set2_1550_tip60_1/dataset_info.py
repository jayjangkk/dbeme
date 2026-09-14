"""Dataset: Kocabas Set 2 on the 5 nm grid, refined to 1 nm for |x| < 60 nm.

Same family, window and base pitch as ``SiO2_kocabas_set2_1550``; the strip
around the Si tip is cut into 1 nm cells, so the width axis steps 2 nm up to
120 nm and 10 nm beyond, and the second axis is the metal's inner edge
(``half_slot``, on the 5 nm grid) rather than the gap.  Two thirds of the
design path's staircase loss sat in the last 100 nm of Si width (report 13
section 3); this grid buys 5x finer steps there for a fifth more unknowns.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9, tip_refine=(60e-9, 1e-9)
)
