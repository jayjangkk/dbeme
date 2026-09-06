"""Direct (uncached) mode extraction along an EME path.

Where :class:`~em_simulation.data_updater.data_updater.DataUpdater` snaps every
section onto a dataset grid and reuses stored results, ``DataExtractor`` solves
each section of the path as it comes.  That is ordinary EME: slower, but it
needs no dataset and it is the natural reference to validate the dataset-based
result against.

The upstream implementation drove Lumerical MODE through ``lumapi``; this one
calls an :class:`~em_simulation.fde.base.FDEBackend` (emepy by default).  The
returned arrays and their conventions are unchanged.
"""

import importlib.util
import os

import numpy as np

from ..fde.assemble import assemble, overlap_matrix, prop_axis_index


class DataExtractor:
    """Solve every section of a path directly, with no dataset in between.

    :param data_directory: Directory holding ``dataset_info.py``.  Only the
        metadata (mode count, wavelength, cross section) is used - no pickles
        are read or written.
    :type data_directory: str
    :param backend: Mode solver.  Defaults to ``DatasetInfo.get_fde_backend()``.
    :param is_testmode: Skip solver construction, for tests.
    :type is_testmode: bool
    :param cache_size: Number of solved cross sections kept in memory.  Paths
        commonly revisit the same parameter point (a straight run, or a bend of
        constant radius), so this pays for itself.
    :type cache_size: int
    """

    def __init__(self, data_directory, backend=None, is_testmode=False, cache_size=256):
        self.data_directory = data_directory
        self.is_testmode = is_testmode

        dataset_info_path = os.path.join(data_directory, "dataset_info.py")
        spec = importlib.util.spec_from_file_location("dataset_info", dataset_info_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.data_info = module.DatasetInfo()

        self.parameter_names = self.get_parameter_names()
        self.parameter_types = self.get_parameter_types()
        self.mode_numbers = self.get_mode_numbers()
        self.crosssection_x = self.get_crosssection_x()
        self.crosssection_y = self.get_crosssection_y()
        self.wavelength = self.get_wavelength()
        self.prop_axis = prop_axis_index(self.crosssection_x, self.crosssection_y)

        self.backend = None
        if not is_testmode:
            self.backend = (
                backend if backend is not None else self.data_info.get_fde_backend()
            )

        self._cache = {}
        self._cache_size = int(cache_size)
        self._pending_points = None
        self._pending_modes = None

    # ------------------------------------------------ dataset metadata

    def get_parameter_names(self):
        return self.data_info.get_parameter_names()

    def get_parameter_types(self):
        return self.data_info.get_parameter_types()

    def get_mode_numbers(self):
        return self.data_info.get_mode_numbers()

    def get_crosssection_x(self):
        return self.data_info.get_crosssection_x()

    def get_crosssection_y(self):
        return self.data_info.get_crosssection_y()

    def get_wavelength(self):
        return self.data_info.get_wavelength()

    def get_cladding_index(self):
        return self.data_info.get_cladding_index()

    def _is_lossless(self):
        """Whether the backend's modes are lossless by construction.

        Defaults to True when there is no backend (test mode), which keeps the
        historical behaviour for every real-index dataset.
        """
        return bool(getattr(self.backend, "lossless", True))

    # ------------------------------------------------------- solve path

    def set_fde_sweep(self, parameter_points):
        """Record the path to solve.  Kept for API compatibility."""
        self._pending_points = [tuple(p) for p in parameter_points]
        self._pending_modes = None

    def run_fde_sweep(self):
        """Solve every section of the recorded path."""
        if self._pending_points is None:
            raise RuntimeError("call set_fde_sweep() before run_fde_sweep()")
        self._pending_modes = [self._solve(p) for p in self._pending_points]

    def post_process_sweep_data(self, parameter_points):
        """Normalise the fields and build the interface overlap matrices.

        :param parameter_points: The path, ordered from input to output port.
        :returns: ``(neffs, TE_pols, overlap_ab, overlap_ba)``.  ``neffs`` and
            ``TE_pols`` have shape ``(n_sections, 2N)``; the overlaps have
            shape ``(n_sections - 1, 2N, 2N)``.
        """
        points = [tuple(p) for p in parameter_points]
        if self._pending_modes is not None and self._pending_points == points:
            mode_data = self._pending_modes
        else:
            mode_data = [self._solve(p) for p in points]

        x, y, neffs, TE_pols, E, H = assemble(
            mode_data, self.mode_numbers, self.prop_axis,
            lossless=self._is_lossless(),
        )

        n_sec = len(points)
        shape = (n_sec - 1, 2 * self.mode_numbers, 2 * self.mode_numbers)
        overlap_ab = np.zeros(shape, dtype=np.complex64)
        overlap_ba = np.zeros(shape, dtype=np.complex64)
        for i in range(n_sec - 1):
            overlap_ab[i] = overlap_matrix(E[i], H[i + 1], x, y, self.prop_axis)
            overlap_ba[i] = overlap_matrix(E[i + 1], H[i], x, y, self.prop_axis)

        self._pending_modes = None
        return neffs, TE_pols, overlap_ab, overlap_ba

    def find_mode_fields(self, parameter_point):
        """Raw modes of one cross section, shaped ``(modes, nx, ny, 3)``."""
        md = self._solve(tuple(parameter_point))
        return (
            np.transpose(md.E, (0, 2, 3, 1)),
            np.transpose(md.H, (0, 2, 3, 1)),
            md.x,
            md.y,
        )

    # -------------------------------------------------------- internals

    def _solve(self, point):
        if self.is_testmode:
            raise RuntimeError("DataExtractor is in test mode; no solver attached")
        cached = self._cache.get(point)
        if cached is not None:
            return cached
        modes = self.backend.solve(point)
        if len(self._cache) < self._cache_size:
            self._cache[point] = modes
        return modes

    def clear_mode_cache(self):
        self._cache.clear()
