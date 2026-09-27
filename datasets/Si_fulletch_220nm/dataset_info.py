"""Dataset definition: fully etched 220 nm Si strip waveguide, SiO2 clad.

This file plays the role that ``wg_crosssection.lms`` + ``dataset_info.py``
played in the Lumerical version: it declares both the parameter grid the
dataset is sampled on and the cross section the mode solver should build.

Platform
--------
======================  =========================
core                    Si (Li 1980, 293 K), n = 3.4757 @ 1550 nm
core thickness          220 nm (full etch, no slab)
cladding / BOX          SiO2 (Malitson 1965), n = 1.4440 @ 1550 nm
wavelength              1550 nm
======================  =========================

Parameter grid
--------------
``top_width``   0.40 - 1.50 um in 20 nm steps (56 points)
``curvature``   -200000 to +200000 1/m in 5000 1/m steps (81 points):
                straight, down to a 5 um bend radius either way

The grid step sets the accuracy of the method: a device is snapped onto these
points, so 20 nm in width and 5000 1/m in curvature is the resolution at which
geometry changes are resolved.  Refine either axis if a device needs it - only
the points a device actually visits are ever computed.
"""

import numpy as np

from dbeme.fde import EmepyFDE, FullEtchStrip
from dbeme.fde.materials import silica, silicon

WAVELENGTH = 1.55e-6
THICKNESS = 220e-9

#: Index models rather than numbers, so the stack stays correct if this
#: dataset is ever rebuilt at another wavelength.  Both come from
#: refractiveindex.info through PyOptik and fall back to the built-in
#: Sellmeier formulas when that database is not installed.
#:
#:   Si   - Li 1980 at 293 K, valid 1.2-14 um, n = 3.4757 at 1550 nm
#:   SiO2 - Malitson 1965, valid 0.21-6.7 um, n = 1.4440 at 1550 nm
#:
#: Across the O to L bands Si moves by 0.040 in index and SiO2 by 0.005, so a
#: constant index would make any bandwidth result meaningless.  `out_of_range`
#: is set to raise: silently extrapolating a fitted dispersion formula past its
#: data is exactly how a wavelength sweep goes quietly wrong.
CORE_MATERIAL = silicon(out_of_range="raise")
CLAD_MATERIAL = silica(out_of_range="raise")

#: Modes solved per cross section (forward only; the dataset stores 2x this).
MODE_NUMBERS = 6

#: Transverse mesh: points across the full solve window in x.  The y count is
#: scaled to keep cells square, giving 20 nm cells - fine enough that the
#: guided-mode effective indices are converged to ~0.002.
MESH = 160


class DatasetInfo:
    def __init__(self):
        self.description = {
            "Description": "overlap, neff and TE_pol for a full-etched Si strip waveguide",
            "material": f"{CORE_MATERIAL.name} core / {CLAD_MATERIAL.name} cladding",
            "structure": "Fully etched strip waveguide, 220 nm thick",
            "solver": "emepy MSEMpy (EMpy vectorial finite difference)",
            "mesh": f"{MESH} points across the window",
        }
        self.file_structure = {
            "dataset_info.py": "this file - parameter grid and cross section",
            "overlap.pkl": "pickled dict: point -> {adjacent point -> (2N,2N) overlap}",
            "neff.pkl": "pickled dict: point -> (2N,) complex effective indices",
            "TE_pol.pkl": "pickled dict: point -> (2N,) TE polarization fraction",
        }
        # Cross-section axes of the mode solver; propagation is along the
        # remaining axis (z).
        self.FDE_crosssection = {
            "crosssection_x": "x",
            "crosssection_y": "y",
        }
        self.mode_numbers = MODE_NUMBERS
        self.wavelength = WAVELENGTH
        self.cladding_index = float(CLAD_MATERIAL.n(WAVELENGTH))
        self.parameter_names = ["top_width", "curvature"]
        self.parameter_types = {
            "top_width": "Length",
            "curvature": "Number",
        }
        self.parameters = {
            # 0.40 - 1.50 um in 20 nm steps
            "top_width": np.round(np.linspace(0.4e-6, 1.5e-6, 56), 9),
            # +/- 200000 1/m in 5000 1/m steps (R = 5 um at the tightest).
            # The upper limit is set by the solve window, not by the physics:
            # the conformal bend transform multiplies the index by
            # exp(kappa * x), so at the window edge the cladding ramps up to
            # n_clad * exp(kappa * half_width).  Once that reaches the guided
            # mode index the solver starts returning modes pinned to the window
            # wall instead.  With n_clad = 1.444 and half_width = 1.6 um,
            # staying below n = 2.1 (the weakest well-guided mode of interest
            # here) means kappa < ln(2.1/1.444)/1.6e-6 = 2.3e5 1/m.  EmepyFDE
            # warns if a solve crosses that line.
            # The axis is symmetric about zero on purpose: a left bend and a
            # right bend are mirror images, and folding them onto |kappa| would
            # make the two halves of an S-bend look identical and erase the
            # mode conversion at their join.  See dbeme/geometry/
            # curvature.py.  A wide axis costs nothing - only the points a
            # device actually visits are ever solved.
            "curvature": np.round(np.linspace(-200000, 200000, 81), 0),
        }

        # Solve window: fixed for every parameter point, since the overlap
        # integrals between different points need one common grid.
        # Wide enough that the guided modes have decayed at the walls and the
        # effective indices no longer move with the window (checked to <1e-3
        # against a 8 x 4 um window), and no wider - solve cost scales with area.
        self._window = (
            1.6e-6,  # half-width in x
            -0.8e-6,  # y min
            +0.8e-6,  # y max
        )

        self._backend = None

    # ------------------------------------------------------------- solver

    def get_cross_section(self):
        return FullEtchStrip(
            thickness=THICKNESS,
            core=CORE_MATERIAL,
            cladding=CLAD_MATERIAL,
            reference_wavelength=WAVELENGTH,
        )

    def get_fde_backend(self):
        """Mode solver for this dataset (cached, so the grid is stable)."""
        if self._backend is None:
            self._backend = EmepyFDE(
                cross_section=self.get_cross_section(),
                parameter_names=self.parameter_names,
                num_modes=self.mode_numbers,
                wavelength=self.wavelength,
                window=self._window,
                mesh=MESH,
            )
        return self._backend

    # --------------------------------------------------------- accessors

    def get_description(self):
        return self.description

    def get_file_structure(self):
        return self.file_structure

    def get_parameter_names(self):
        return self.parameter_names

    def get_parameter_grid(self):
        return self.parameters

    def get_parameter_types(self):
        return self.parameter_types

    def get_mode_numbers(self):
        return self.mode_numbers

    def get_crosssection_x(self):
        return self.FDE_crosssection["crosssection_x"]

    def get_crosssection_y(self):
        return self.FDE_crosssection["crosssection_y"]

    def get_wavelength(self):
        return self.wavelength

    def get_cladding_index(self):
        return self.cladding_index

    def _is_variable_FDE(self):
        return False
