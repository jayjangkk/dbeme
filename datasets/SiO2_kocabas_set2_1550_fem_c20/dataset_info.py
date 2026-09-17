"""Dataset: Kocabas Set 2 on the finite-element backend, corners rounded 20 nm.

The same device and ``(w_si, half_slot)`` axes as the rounded finite-difference
baseline ``SiO2_kocabas_set2_1550_hs_c20`` (82.4 % at the tip), solved on a
boundary-conforming gmsh mesh instead of a 5 nm grid: the gold corners are
true arcs, so the gap plasmon converges with element size where the FD grid
could not (report 13 sections 7-8).  The 5 nm ``cell`` here is only the pitch
of the grid the fields are evaluated on for the overlaps; both parameter axes
step 5 nm because nothing has to align with a cell.  Elements 6 nm on the
gold, 15 nm on the silicon; a lossy ring (eps'' 0.5 over 0.3 um) stands in for
the PML.
"""

from em_simulation.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    corner_radius=20e-9, axes="half_slot", solver="femwell",
)
