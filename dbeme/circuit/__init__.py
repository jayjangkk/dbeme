"""Circuit layer: DBEME building blocks composed as S-parameter circuits.

DBEME solves cross sections and cascades them along ``z``; it cannot close a
loop.  A ring, a Mach-Zehnder, a coupled-resonator filter are *circuits* of
DBEME building blocks, and this package is where those blocks become circuit
models.  It is a consumer of S-matrices downstream of ``FDEBackend`` - it is
not a backend, and nothing here touches a mode solver (`tasks/11` §Why).

Two engines share the models built here:

* **sax** (frequency domain, JAX, installed) - the interface.  Every model is
  a sax model: a keyword-only function of the wavelength ``wl`` **in microns**
  returning an ``SDict`` keyed ``(port_in, port_out)``, reciprocal, with
  ``jnp`` values so gradients flow.
* **circulax** (time domain, harmonic balance, nonlinear; Phase 5) - reuses
  sax models directly.

Units, stated once: everything *entering* a factory from DBEME is SI (metres,
1/m); everything a sax model *receives* is in sax's own units (``wl`` in
microns, loss in dB/cm), because a sax netlist passes one ``wl`` to every
instance and the built-in models expect microns.  Sign convention is the
project's ``exp(-i omega t)``, ``exp(+i beta z)``; sax's ``straight`` uses the
same (``exp(+i 2 pi n_eff L / wl)``), verified in
``tests/test_circuit_sax_adapter.py``.

JAX runs in float32 unless told otherwise; a 50 rad phase held to 1e-9 rad
needs float64, so importing this package enables x64 process-wide.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .phase_fit import OpticalLengthFit, PchipJax, PolynomialFit, unwrap_against, wrap  # noqa: E402
from .ring import (  # noqa: E402
    add_drop,
    add_drop_amplitude,
    all_pass,
    all_pass_amplitude,
    ring_metrics,
    round_trip_phase,
    single_pass_amplitude,
)
from .sax_model import sax_model_from_samples  # noqa: E402
from .waveguide import (  # noqa: E402
    IndexModel,
    arc_model,
    ideal_coupler,
    loss_db_per_cm_from_imag_neff,
    straight_model,
)

__all__ = [
    "IndexModel",
    "OpticalLengthFit",
    "PchipJax",
    "PolynomialFit",
    "add_drop",
    "add_drop_amplitude",
    "all_pass",
    "all_pass_amplitude",
    "arc_model",
    "ideal_coupler",
    "loss_db_per_cm_from_imag_neff",
    "ring_metrics",
    "round_trip_phase",
    "sax_model_from_samples",
    "single_pass_amplitude",
    "straight_model",
    "unwrap_against",
    "wrap",
]
