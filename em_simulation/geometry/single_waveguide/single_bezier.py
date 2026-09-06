import numpy as np
from scipy.interpolate import interp1d

from ...data_updater.data_updater import DataUpdater
from ..bend_shapes.bezier import BezierCurve
from .single_waveguide import SingleWaveguide


class SingleBezierCurve(SingleWaveguide, BezierCurve):
    """A single waveguide whose centreline is a Bezier curve.

    The curve is sampled, its local curvature and tangent angle are extracted,
    and those become the parameter functions the dataset is indexed with.  A
    constant-width Bezier bend is therefore just a path through the
    ``curvature`` axis of the dataset.
    """

    def __init__(
        self,
        dataset: DataUpdater,
        top_width,
        control_points,
        resolution=1000,
        limit_mode_number=0,
    ):
        """Create a single waveguide segment defined by a Bezier centerline.

        :param dataset: Precomputed modal dataset supplying overlap and effective
            index information.
        :type dataset: DataUpdater
        :param top_width: Waveguide top width in meters.
        :type top_width: float
        :param control_points: Ordered Bezier control points describing the bend
            geometry, as ``[(x0, y0), (x1, y1), ...]`` in meters.
        :type control_points: array-like
        :param resolution: Number of discretization samples along the curve.
        :type resolution: int
        :param limit_mode_number: Maximum number of modes to include when
            querying the dataset (``0`` uses all available modes).
        :type limit_mode_number: int
        """
        SingleWaveguide.__init__(self, dataset, limit_mode_number=limit_mode_number)
        BezierCurve.__init__(self, control_points, resolution=resolution)
        self._top_width = top_width
        self._control_points = control_points
        self._resolution = resolution

        # `_calc_parameters` drops the final sample, where the curvature of a
        # Bezier with coincident end control points is ill-conditioned.  The
        # usable length is therefore the arc length of the samples that remain,
        # which is what the parameter interpolants are defined over.
        self.prop_length, self.curvatures, self.prop_angles = (
            BezierCurve._calc_parameters(self)
        )
        self._total_length = float(self.prop_length[-1])

    def _parametrized_function(self):
        top_widths = self._top_width * np.ones(len(self.prop_length))

        curvature_function = interp1d(self.prop_length, self.curvatures)
        prop_angle_function = interp1d(self.prop_length, self.prop_angles)
        top_width_function = interp1d(self.prop_length, top_widths)

        return top_width_function, curvature_function, prop_angle_function

    def calc_total_length(self):
        return self._total_length

    def _calc_xy(self):
        x, y = BezierCurve._calc_xy(self)
        return np.asarray(x), np.asarray(y)
