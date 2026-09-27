"""Dataset: Kocabas Set 2 on the finite-element backend, corners rounded 20 nm, the wall stepping 1 nm and the Si width 2.5 nm.

The z-discretisation extrapolation of report 13 section 11: the same platform
as ``SiO2_kocabas_set2_1550_fem_c20_hs1`` (85.6 % at the tip, wall steps
1 nm, Si steps 5 nm) with the parameter-axis steps - which are the steps of
the wall position and the Si width along the taper, one cross section per
grid point - set to the wall stepping 1 nm and the Si width 2.5 nm.  The cross-sectional mesh is unchanged and
already converged; only the number of distinct cross sections along the
taper grows.  Mesh convention 2.
"""

from dbeme.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(
    set_number=2, wavelength=WAVELENGTH, cell=5e-9,
    corner_radius=20e-9, axes="half_slot", solver="femwell",
    axis_steps={"half_slot": 1e-9, "w_si": 2.5e-9},
)
