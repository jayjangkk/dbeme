"""Dataset: Kocabas Set 2 converter (arXiv:1801.00833, Table II), 1550 nm.

Si wire 400 x 725 nm to a 250 nm plasmonic slot in 250 nm gold, everything
embedded in SiO2 and vertically centred.  Two axes, ``w_si`` and ``gap``,
because the slot tapers independently of the Si (gap 75 -> 125 nm along the
paper's design).  Lossy PML basis; the paper's material constants.
"""

from dbeme.platforms import kocabas_converter_dataset_info

WAVELENGTH = 1.55e-6

DatasetInfo = kocabas_converter_dataset_info(set_number=2, wavelength=WAVELENGTH)
