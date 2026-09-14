"""Sacher et al. 2014 Si3N4-on-SOI converter stack - `tasks/10` Phase 0f.

W. D. Sacher et al., Opt. Express 22(9), 11167 (2014).  No pickles: the
study runs on ``DirectParametricPath``; this file carries the wavelength,
mode count, cladding index and axes that ``DataExtractor`` reads.

wavelength   1550 nm
stack        150 nm Si (thinned from 220 nm), 50 nm planarised SiO2 nominal
             (~10 nm as built), 400 nm LPCVD Si3N4; vertical walls [assumed]
indices      Si Palik (table reaches 1.6 um); Si3N4 1.996 and SiO2 1.444
             [assumed - the platform's O-band tables do not reach 1550 nm]
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from studies.sn2sipr.cross_section import PARAMETERS, sacher2014_section  # noqa: E402

WAVELENGTH = 1.550e-6
MODE_NUMBERS = 10


class DatasetInfo:
    def __init__(self):
        cs = sacher2014_section(wavelength=WAVELENGTH)
        self.description = {
            "Description": "Sacher 2014 Si3N4-on-SOI TM0-TE1 converter stack",
            "material": f"Si3N4 {cs.sin_index_at(WAVELENGTH):.4f} / "
                        f"Si {cs.core_index_at(WAVELENGTH):.4f} / "
                        f"SiO2 {cs.cladding_index_at(WAVELENGTH):.4f}",
        }
        self.file_structure = {"dataset_info.py": "metadata"}
        self.FDE_crosssection = {"crosssection_x": "x", "crosssection_y": "y"}
        self.mode_numbers = MODE_NUMBERS
        self.wavelength = WAVELENGTH
        self.cladding_index = cs.cladding_index_at(WAVELENGTH)
        self.parameter_names = list(PARAMETERS)
        self.parameter_types = {name: "Length" for name in PARAMETERS}
        self.parameters = {name: np.array([0.0]) for name in PARAMETERS}

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
