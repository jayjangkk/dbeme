"""Dataset: air-clad RAC coupling region at 1550 nm, bound modes only.

Air on a buried oxide box cuts off at n = 1.444, so a 6-mode solve returns
three bound modes and three sub-cutoff modes whose fields are set by the
simulation window.  Two of those sit at n_eff = 1.19 and 1.17 - near-degenerate,
which ``_pin_gauge`` cannot pin (see ``CLAUDE.md`` 5.6), so their stored
overlaps disagree about the subspace rotation and the cached S-matrix is
unusable.  Restricting the basis to the modes that are actually resolved is the
fix; ``examples/study_rac_air_basis.py`` measures the difference.
"""

from dbeme.platforms import rac_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = rac_dataset_info(WAVELENGTH, mode_numbers=3, clad="air")
