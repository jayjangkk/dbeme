"""Dataset: Sacher bi-level taper with slanted sidewalls, at 1550 nm.

Same stack and grid as ``Si_bilevel_220_90nm``, with the etched sidewalls at
their fabricated angles instead of vertical: 85 degrees on the full etch
that defines the slab and 87 degrees on the partial etch that defines the
core above it. Both are measured from horizontal, so each layer is wider at its
base than at the quoted top width.

A slanted wall is not only a fabrication tolerance. A trapezoid has no
horizontal mirror plane, so it breaks the very symmetry the partial etch was
introduced to break - see ``reports/04_sidewall_angle.md``.
"""

from dbeme.platforms import sacher_bilevel_dataset_info

WAVELENGTH = 1.55e-06
BOTTOM_SIDEWALL_ANGLE = 85.0
TOP_SIDEWALL_ANGLE = 87.0

DatasetInfo = sacher_bilevel_dataset_info(
    WAVELENGTH,
    bottom_sidewall_angle=BOTTOM_SIDEWALL_ANGLE,
    top_sidewall_angle=TOP_SIDEWALL_ANGLE,
)
