"""Dataset: Kocabas Set 2, rounded corners, 5 nm grid, (w_si, half_slot) axes, 40 modes.

The basis-size test on the rounded baseline ``SiO2_kocabas_set2_1550_hs_c20``
(82.4 % at the tip with 20 modes).  At the 200 nm-width design point the two
sets share the same 7 physical modes to four digits; every one of the extra
20 is a Berenger mode of the discretised continuum (Im n 0.07 -> 1.08), which
is precisely the field the 20-mode set could only absorb.  One solve costs
155 s against 116 s at 20 modes - the factorisation dominates, not the
eigensolve - so the design path is ~4 h.  See report 13 section 7.
"""

from dbeme.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    corner_radius=20e-9, fill_floor=0.06, axes="half_slot", mode_numbers=40,
)
