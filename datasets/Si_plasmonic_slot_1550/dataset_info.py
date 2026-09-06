"""Dataset: Si-wire-to-plasmonic-slot converter (Ono et al. 2016), air gap 20 nm.

The designed device: the simulated optimum (-1 dB at 600 nm).

Lateral 2-D model, suspended in air, gold plates the full Si height
(see ``PlasmonicSlotConverter``).  Lossy basis on a PML grid - ``PMLBackend`` - so this dataset carries complex
effective indices and is cascaded on the scattering route (``auto`` resolves
to ``direct`` because the backend declares itself lossy).
"""

from em_simulation.platforms import plasmonic_converter_dataset_info

WAVELENGTH = 1.55e-6
GAP = 20e-9

DatasetInfo = plasmonic_converter_dataset_info(WAVELENGTH, gap=GAP)
