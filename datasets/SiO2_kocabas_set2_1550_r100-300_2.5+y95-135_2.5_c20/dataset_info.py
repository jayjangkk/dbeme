"""Dataset: Kocabas Set 2, rounded gold corners, the wall strip and the corner rows at 2.5 nm.

The same experiment as ``SiO2_kocabas_set2_1550_r100-300_1+y95-135_1_c20`` at
half the refinement: the wall strip (100 < |x| < 300 nm) and the rounded
corners' rows (95 < |y| < 135 nm) cut to 2.5 nm, so the corner cells stay
isotropic, the wall steps 2.5 nm through 86 positions, and the grid is
551 x 363 against 471 x 347 (+22 % unknowns) instead of +98 %; about 101
path points against 197.  Corners
rounded 20 nm with the ENZ-safe fill floor 0.06.  Compare with
``SiO2_kocabas_set2_1550_hs_c20`` (82.4 % at the tip on the plain 5 nm grid):
the wall's staircase should halve and the rounded modes move part of the way
toward their 1 nm values (report 13 section 7).
"""

from dbeme.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    refine=((100e-9, 300e-9, 2.5e-9),), refine_y=((95e-9, 135e-9, 2.5e-9),),
    corner_radius=20e-9, fill_floor=0.06,
)
