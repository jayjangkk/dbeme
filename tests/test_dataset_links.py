"""Dataset updates solve only what a path consumes.

On a d-axis grid every visited point has up to 2d grid neighbours; the EME
cascade needs overlaps only between consecutive path points.  Before this,
``DataUpdater`` linked every grid neighbour of every visited point - three
solves per path point on the two-axis Kocabas dataset, two of them wasted.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_simulation.data_updater.data_updater import DataUpdater  # noqa: E402
from em_simulation.fde.base import ModeData  # noqa: E402
from em_simulation.geometry.geometry import Geometry  # noqa: E402

INFO = """
import numpy as np


class DatasetInfo:
    def __init__(self):
        self.mode_numbers = 2
        self.wavelength = 1.55e-6
        self.cladding_index = 1.444
        self.parameter_names = ["w", "g"]
        self.parameter_types = {"w": "Length", "g": "Length"}
        self.parameters = {"w": np.array([1.0, 2.0, 3.0]), "g": np.array([10.0, 20.0, 30.0])}
        self.FDE_crosssection = {"crosssection_x": "x", "crosssection_y": "y"}
        self.description = {}
        self.file_structure = {}

    def get_parameter_grid(self):
        return self.parameters

    def get_parameter_names(self):
        return self.parameter_names

    def get_parameter_types(self):
        return self.parameter_types

    def get_mode_numbers(self):
        return self.mode_numbers

    def get_wavelength(self):
        return self.wavelength

    def get_cladding_index(self):
        return self.cladding_index

    def get_fde_backend(self):
        raise RuntimeError("the test supplies its own backend")

    def get_crosssection_x(self):
        return "x"

    def get_crosssection_y(self):
        return "y"

    def _is_variable_FDE(self):
        return False
"""


class CountingBackend:
    """Two orthogonal 'modes' on a tiny grid; counts how often it is asked."""

    lossless = True
    parameter_names = ("w", "g")

    def __init__(self):
        self.calls = []
        self.window = (1.0, -1.0, 1.0)
        self.x = np.linspace(-1, 1, 12)
        self.y = np.linspace(-1, 1, 10)
        self.cross_section = None

    def solve_grid(self):
        return self.x, self.y

    def solve(self, point):
        self.calls.append(tuple(point))
        nx, ny = self.x.size, self.y.size
        E = np.zeros((2, 3, nx, ny), dtype=complex)
        H = np.zeros((2, 3, nx, ny), dtype=complex)
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        E[0, 0] = np.exp(-X ** 2 - Y ** 2)
        H[0, 1] = E[0, 0]
        E[1, 0] = X * np.exp(-X ** 2 - Y ** 2)
        H[1, 1] = E[1, 0]
        return ModeData(x=self.x, y=self.y, E=E, H=H,
                        neff=np.array([2.0, 1.8]) + 0.01 * float(point[0]), TE_pol=np.array([1.0, 1.0]))


@pytest.fixture
def updater(tmp_path):
    (tmp_path / "dataset_info.py").write_text(INFO, encoding="utf-8")
    return DataUpdater(str(tmp_path), backend=CountingBackend())


def test_explicit_neighbours_restrict_the_solves(updater):
    centre = (2.0, 20.0)
    updater.calc_data_point_modified(centre, neighbours=[(1.0, 20.0)])
    assert sorted(set(updater.backend.calls)) == [(1.0, 20.0), (2.0, 20.0)]
    assert set(updater.overlap[centre]) == {(1.0, 20.0)}
    assert set(updater.overlap[(1.0, 20.0)]) == {centre}
    # the other three grid neighbours were neither solved nor linked
    assert (2.0, 10.0) not in updater.neff and (3.0, 20.0) not in updater.neff


def test_without_neighbours_every_grid_neighbour_is_linked(updater):
    centre = (2.0, 20.0)
    updater.calc_data_point_modified(centre)
    assert set(updater.overlap[centre]) == {(1.0, 20.0), (3.0, 20.0), (2.0, 10.0), (2.0, 30.0)}
    assert len(set(updater.backend.calls)) == 5


def test_path_links_are_the_consecutive_pairs():
    path = [(1.0, 10.0), (2.0, 10.0), (2.0, 20.0), (3.0, 20.0)]
    links = Geometry._path_links(path)
    assert links[(1.0, 10.0)] == {(2.0, 10.0)}
    assert links[(2.0, 10.0)] == {(1.0, 10.0), (2.0, 20.0)}
    assert links[(2.0, 20.0)] == {(2.0, 10.0), (3.0, 20.0)}
    assert links[(3.0, 20.0)] == {(2.0, 20.0)}
    # a diagonal step's mutual-neighbour detour is linked on both sides
    links = Geometry._path_links(path, simul_params=[(1.0, 10.0), (2.0, 20.0)],
                                 additional_param_dict={(1.0, 10.0): [(1.0, 20.0)]})
    assert (1.0, 20.0) in links[(1.0, 10.0)] and (2.0, 20.0) in links[(1.0, 20.0)]


def test_a_repeated_point_links_nothing_to_itself():
    links = Geometry._path_links([(1.0, 10.0), (1.0, 10.0), (2.0, 10.0)])
    assert (1.0, 10.0) not in links[(1.0, 10.0)]
