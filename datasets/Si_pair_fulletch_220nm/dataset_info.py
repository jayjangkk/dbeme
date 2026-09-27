"""Dataset definition: two fully etched 220 nm Si strips side by side.

The coupled-pair counterpart of ``Si_fulletch_220nm``.  Serves any device built
from two parallel strips in one device layer - adiabatic couplers, asymmetric
directional couplers, mode (de)multiplexers.

Platform
--------
======================  ==========================================
cores                   two Si strips (Li 1980, 293 K), n = 3.4757
core thickness          220 nm, fully etched
cladding / BOX          SiO2 (Malitson 1965), n = 1.4440
gap                     200 nm, edge to edge (fixed, not swept)
wavelength              1550 nm
======================  ==========================================

This is the stack of the adiabatic coupler stage in Sacher et al.,
*Opt. Express* **22**, 3777 (2014): "fully-etched Si waveguides with symmetric
SiO2 cladding", 220 nm device layer, 200 nm gap.

Parameter grid
--------------
Both axes are **non-uniform**: 20 nm steps in general, 5 nm across the
anti-crossing region (``w1`` 700-800 nm, ``w2`` 280-440 nm).

That refinement is not cosmetic.  Away from the anti-crossing the supermodes of
adjacent grid points are essentially the same modes - the overlap matrix
between them is the identity to 1e-3.  At the crossing they are not: a single
20 nm step rotates the two crossing branches into each other by |overlap| =
0.54, i.e. it scatters 29 % of the power at one interface.  A staircase that
coarse is a different device from the smooth taper it is meant to represent,
and it destroys exactly the adiabatic behaviour the coupler relies on.

The requirement was measured directly (see ``reports/01_adiabatic_coupler.md``):
a direct-EME sampling study gives a stable answer for width steps of 5 nm and
below, and a badly wrong one at 11.5 nm.  5 nm is used here.

The rule generalises: sample densely wherever ``|dn_eff/dp|`` is large, which
is wherever two branches approach each other.  A wide axis costs nothing until
a device visits it, so refining a region is cheap; getting it wrong is not.

The gap is deliberately **not** a grid axis.  Every axis adds two neighbour
solves per path point, and the devices this dataset was built for hold the gap
constant; a gap sweep is done by building one dataset per gap, which is both
cheaper and what the fingerprint guard expects.  ``CoupledStrips`` accepts
``gap`` as a swept parameter if a future device needs it.

Cost
----
Only the grid points a device actually visits are ever solved (~2 s each at
this mesh), so the nominal 16 x 19 rectangle is not the cost - the path
through it is.
"""

import numpy as np

from dbeme.fde import EmepyFDE
from dbeme.fde.cross_section import CoupledStrips
from dbeme.fde.materials import silica, silicon

WAVELENGTH = 1.55e-6
THICKNESS = 220e-9
GAP = 200e-9

CORE_MATERIAL = silicon(out_of_range="raise")
CLAD_MATERIAL = silica(out_of_range="raise")

#: Modes solved per cross section (forward only; the dataset stores 2x this).
#: Six covers the three guided TE branches that matter here - the broad guide's
#: TE0 and TE1 and the narrow guide's TE0 - plus the TM modes and one spare,
#: which is what the EME basis needs to stay near-unitary across the
#: anti-crossing.
MODE_NUMBERS = 6

#: Points across the window in x; y is scaled to keep 20 nm cells.
MESH = 220


def nm(start, stop, step):
    """Grid values in metres, from a range given in nanometres."""
    return np.arange(start, stop, step) * 1e-9


def _axis(*segments):
    """Join grid segments into one sorted, de-duplicated axis."""
    return np.round(np.unique(np.concatenate(segments)), 12)


class DatasetInfo:
    def __init__(self):
        self.description = {
            "Description": "overlap, neff and TE_pol for a coupled pair of Si strips",
            "material": f"{CORE_MATERIAL.name} cores / {CLAD_MATERIAL.name} cladding",
            "structure": (
                f"Two fully etched strips, {THICKNESS*1e9:.0f} nm thick, "
                f"{GAP*1e9:.0f} nm gap"
            ),
            "solver": "emepy MSEMpy (EMpy vectorial finite difference)",
            "mesh": f"{MESH} points across the window",
        }
        self.file_structure = {
            "dataset_info.py": "this file - parameter grid and cross section",
            "overlap.pkl": "pickled dict: point -> {adjacent point -> (2N,2N) overlap}",
            "neff.pkl": "pickled dict: point -> (2N,) complex effective indices",
            "TE_pol.pkl": "pickled dict: point -> (2N,) TE polarization fraction",
        }
        self.FDE_crosssection = {"crosssection_x": "x", "crosssection_y": "y"}

        self.mode_numbers = MODE_NUMBERS
        self.wavelength = WAVELENGTH
        self.cladding_index = float(CLAD_MATERIAL.n(WAVELENGTH))
        self.parameter_names = ["w1", "w2"]
        self.parameter_types = {"w1": "Length", "w2": "Length"}
        # Both axes carry the device endpoints (650/850 nm and 200/500 nm)
        # exactly: a device whose ends fall between grid points starts and
        # finishes at the wrong width, which is a silent geometry error.
        self.parameters = {
            "w1": _axis(nm(610, 700, 20), nm(700, 805, 5), nm(810, 915, 20)),
            "w2": _axis(nm(180, 280, 20), nm(280, 445, 5), nm(460, 545, 20)),
        }

        # Fixed window, wide enough for the widest pair plus a decay margin.
        half_width = 0.5 * (0.91e-6 + GAP + 0.54e-6) + 1.3e-6
        self._window = (half_width, -0.8e-6, 0.8e-6)
        self._backend = None

    # ------------------------------------------------------------- solver

    def get_cross_section(self):
        return CoupledStrips(
            thickness=THICKNESS,
            gap=GAP,
            core=CORE_MATERIAL,
            cladding=CLAD_MATERIAL,
            swept_parameters=("w1", "w2"),
            reference_wavelength=WAVELENGTH,
        )

    def get_fde_backend(self):
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
