"""Dataset: Kocabas Set 2 on the 5 nm grid, refined to 1 nm for 100 < |x| < 300 nm.

The strip the gold wall moves through: along the design path the metal's
inner edge runs from 275 nm to 125 nm, and the gap plasmon's ~23 nm skin
depth reaches 25 nm into the gold beyond it.  Measured on clean axes, the
design path's single-mode staircase is the wall's 5 nm jumps once the mode is
slot-like - 6.1 % of 7.2 % - and refining the walls to 1 nm in x alone moves
the slot mode more than half of the way from the 5 nm value to the 2.5 nm one
(1.4498 -> 1.4751 -> 1.4941; report 13 section 7).  So this grid steps the
wall 1 nm at a time (``half_slot`` axis at 1 nm inside the strip), keeps the
Si width axis at 10 nm, and costs +68 % unknowns against 4.9x per solve for
the uniform 2.5 nm grid.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9, refine=((100e-9, 300e-9, 1e-9),)
)
