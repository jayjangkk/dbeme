"""Independent references to validate the solver against.

Nothing in here imports the EME machinery.  These modules exist to answer
"is the number the solver produced correct?", so a shared bug would defeat the
purpose - see ``tasks/02_pml_backend.md`` Phase 1.
"""

from .bent_slab import (
    BentSlab,
    attenuation_db_per_cm,
    bend_loss_db_per_90deg,
    imag_neff_from_db_per_cm,
)

__all__ = [
    "BentSlab",
    "attenuation_db_per_cm",
    "bend_loss_db_per_90deg",
    "imag_neff_from_db_per_cm",
]
