"""Dataset: Kocabas Set 2, rounded gold corners, the wall strip and the corner rows at 1 nm.

The strip the gold wall moves through (100 < |x| < 300 nm; the wall runs
275 -> 125 nm along the design path, plus the ~23 nm skin depth) at 1 nm in
``x``, and the rows holding the rounded corners (95 < |y| < 135 nm) at 1 nm
in ``y``, so the corner cells are isotropic - 1 x 5 nm cells at a corner
returned an E field 37x too large and a TE fraction of 0.09 for a TE mode.
Corners rounded 20 nm with the ENZ-safe fill floor 0.06 (report 13 section 7).
791 x 411 against 471 x 347, +98 % unknowns; the wall steps 1 nm through 191
positions; the Si axis stays at 10 nm.  Compare with
``SiO2_kocabas_set2_1550_hs_c20``, the same geometry and axes on the plain
5 nm grid.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    refine=((100e-9, 300e-9, 1e-9),), refine_y=((95e-9, 135e-9, 1e-9),),
    corner_radius=20e-9, fill_floor=0.06,
)
