"""A device described by arbitrary parameter functions of propagation length.

``SingleWaveguide`` hard-codes the parameter set of a one-core dataset
(``top_width``, ``curvature``, optionally ``rotation_angle``).  A coupled pair
sweeps ``w1``, ``w2`` and possibly ``gap``, and the next cross section will
sweep something else again, so the snapping logic is generalised here: give it
one function of ``z`` per dataset parameter and it produces the EME path.

The algorithm is the one in ``SingleWaveguide.calc_simulation_parameters``:

1. sample every parameter function along ``z``;
2. snap each sample onto its dataset axis;
3. keep the propagation lengths at which *any* parameter changes, since that
   is where an EME interface belongs;
4. drop consecutive duplicates, then close the path at the true device end -
   otherwise a constant-parameter run at the output is silently dropped;
5. subdivide any run longer than ``max_section_length`` so that a long
   uniform stretch does not become a single enormous section.
"""

import numpy as np

from .direct_geometry import DirectGeometry
from .geometry import Geometry


class ParametricPath(Geometry):
    """Geometry from a mapping of parameter name to ``f(z)``.

    :param dataset: The :class:`~dbeme.data_updater.data_updater.DataUpdater`.
    :param parameter_functions: ``{name: callable}``, one per dataset parameter,
        each taking propagation length in metres and returning that parameter's
        value.  A constant is accepted in place of a callable.
    :param total_length: Device length in metres.
    :param resolution: Number of samples used to find the snap points.
    :param max_section_length: Longest EME section before it is subdivided, in
        metres.  Subdividing a long uniform run costs nothing in mode solving -
        the sections are identical - but keeps the phase per section moderate.
    :param limit_mode_number: Passed to :class:`Geometry`.
    :param verbose: Print dataset-update progress.
    """

    def __init__(
        self,
        dataset,
        parameter_functions,
        total_length,
        resolution=800,
        max_section_length=10e-6,
        limit_mode_number=0,
        verbose=True,
    ):
        super().__init__(dataset, limit_mode_number=limit_mode_number, verbose=verbose)
        self._total_length = float(total_length)
        self._resolution = int(resolution)
        self._max_section_length = float(max_section_length)

        missing = set(self.parameter_names) - set(parameter_functions)
        if missing:
            raise ValueError(
                f"no function given for dataset parameter(s) {sorted(missing)}; "
                f"this dataset sweeps {list(self.parameter_names)}"
            )
        self._functions = {
            name: self._as_callable(parameter_functions[name])
            for name in self.parameter_names
        }

    @staticmethod
    def _as_callable(value):
        if callable(value):
            return value
        return lambda z, _v=float(value): np.full_like(np.asarray(z, dtype=float), _v)

    def parameter_value(self, name, z):
        """The ideal (unsnapped) value of ``name`` at propagation length ``z``."""
        return np.asarray(self._functions[name](np.asarray(z, dtype=float)), dtype=float)

    def _snap(self, name, values):
        axis = np.asarray(self.parameter_grid[name], dtype=float)
        values = np.atleast_1d(np.asarray(values, dtype=float))
        indices = np.argmin(np.abs(values[:, None] - axis[None, :]), axis=1)
        return axis[indices]

    def calc_simulation_parameters(self):
        """Snap the parameter functions onto the dataset grid.

        :returns: ``(simulation_parameters, delta_zs)`` - a list of parameter
            tuples ordered as ``parameter_names``, and the length of each gap
            between them.
        """
        z = np.linspace(0.0, self._total_length, self._resolution)
        snapped = {name: self._snap(name, self.parameter_value(name, z))
                   for name in self.parameter_names}

        # An interface belongs wherever any parameter steps to a new grid value.
        changed = np.zeros(z.size, dtype=bool)
        changed[0] = True
        for values in snapped.values():
            changed[1:] |= values[1:] != values[:-1]
        indices = np.flatnonzero(changed)

        lengths = list(z[indices])
        points = [
            tuple(snapped[name][i] for name in self.parameter_names) for i in indices
        ]

        # Close the path: `changed` only marks transitions, so a run of constant
        # parameters at the output would otherwise be dropped and the device
        # would come out short.
        if self._total_length - lengths[-1] > 1e-12:
            lengths.append(self._total_length)
            points.append(points[-1])

        lengths, points = self._subdivide(lengths, points)

        if len(points) == 1:
            points = points * 2
            lengths = [0.0, self._total_length]

        delta_zs = np.diff(np.asarray(lengths, dtype=float))
        return points, delta_zs

    def _subdivide(self, lengths, points):
        """Split any section longer than ``max_section_length``."""
        out_lengths, out_points = [lengths[0]], [points[0]]
        for start, stop, point in zip(lengths[:-1], lengths[1:], points[1:]):
            span = stop - start
            if span > self._max_section_length:
                pieces = int(np.ceil(span / self._max_section_length))
                for k in range(1, pieces):
                    out_lengths.append(start + span * k / pieces)
                    # The parameters do not change across a subdivided run, so
                    # the inserted sections repeat the *preceding* point.
                    out_points.append(out_points[-1])
            out_lengths.append(stop)
            out_points.append(point)
        return out_lengths, out_points

    # Concrete devices are described entirely by their parameter functions, so
    # these carry no extra information.
    def calc_total_length(self):
        return self._total_length


class DirectParametricPath(DirectGeometry):
    """The same device as :class:`ParametricPath`, solved without a grid.

    Samples the parameter functions at uniform propagation length and solves
    each section at its *exact* parameter values.  That is ordinary EME: no
    dataset, no snapping, and no reuse between runs.  It is the reference the
    dataset-based result is checked against, so the only difference between the
    two is the grid.

    :param data_extractor: A
        :class:`~dbeme.data_extractor.data_extractor.DataExtractor`.
    :param parameter_functions: ``{name: callable}``, as :class:`ParametricPath`.
    :param total_length: Device length in metres.
    :param resolution: Number of sections.  Match it to the dataset path's
        section count to compare like with like.
    """

    def __init__(
        self,
        data_extractor,
        parameter_functions,
        total_length,
        resolution=40,
        limit_mode_number=0,
        use_existing_data=False,
    ):
        super().__init__(
            data_extractor,
            limit_mode_number=limit_mode_number,
            use_existing_data=use_existing_data,
        )
        self._total_length = float(total_length)
        self._resolution = int(resolution)

        missing = set(self.parameter_names) - set(parameter_functions)
        if missing:
            raise ValueError(
                f"no function given for dataset parameter(s) {sorted(missing)}"
            )
        self._functions = {
            name: ParametricPath._as_callable(parameter_functions[name])
            for name in self.parameter_names
        }

    def calc_simulation_parameters(self):
        z = np.linspace(0.0, self._total_length, self._resolution)
        columns = [
            np.atleast_1d(np.asarray(self._functions[name](z), dtype=float))
            for name in self.parameter_names
        ]
        points = [tuple(float(c[i]) for c in columns) for i in range(len(z))]
        return points, np.diff(z)

    def calc_total_length(self):
        return self._total_length
