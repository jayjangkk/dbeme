# Defining your own platform

*Moved verbatim from `README.md` by task 16 step 1.5b (2026-09-27); the README is now a landing page that links here.*

A dataset directory needs exactly one file, `dataset_info.py`, declaring the
parameter grid and the cross section. See
`datasets/Si_fulletch_220nm/dataset_info.py`; the essential part is:

```python
import numpy as np
from dbeme.fde import EmepyFDE, FullEtchStrip, silica, silicon


class DatasetInfo:
    def __init__(self):
        self.parameter_names = ["top_width", "curvature"]
        self.parameters = {
            "top_width": np.round(np.linspace(0.4e-6, 1.5e-6, 56), 9),   # 20 nm steps
            "curvature": np.round(np.linspace(-2e5, 2e5, 81), 0),        # 1/m
        }
        self.mode_numbers = 6          # forward modes; the dataset stores 2x
        self.wavelength = 1.55e-6
        self.cladding_index = float(silica().n(1.55e-6))

    def get_fde_backend(self):
        return EmepyFDE(
            cross_section=FullEtchStrip(
                thickness=220e-9,
                core=silicon(out_of_range="raise"),
                cladding=silica(out_of_range="raise"),
            ),
            num_modes=6, wavelength=1.55e-6,
            window=(1.6e-6, -0.8e-6, 0.8e-6), mesh=160,
        )
```

For a different geometry — rib, ridge, nitride, angled sidewalls — subclass
`CrossSection` and implement `index(x, y, params)`. That single method is the
whole geometry definition.

### Materials

A layer takes a `Material`, a plain number, or a `"shelf/book/page"` identifier
from [refractiveindex.info](https://refractiveindex.info):

```python
from dbeme.fde.materials import PyOptikMaterial, silicon, silica

FullEtchStrip(core=silicon(), cladding=silica())          # database, with fallback
FullEtchStrip(core="main/Si/Salzberg", cladding=1.444)    # explicit page + a number
FullEtchStrip(core_index=3.4757, clad_index=1.444)        # the pre-materials form
```

`silicon()`, `silica()` and `silicon_nitride()` take the database entry when
PyOptik and its snapshot are installed and fall back to a built-in Sellmeier
formula (with a warning) when they are not. For SiO2 and Si3N4 the fallback
*is* the same published formula, so nothing is lost; for Si the fallback is
Salzberg 1957 against the database's Li 1980, which differ by 2e-3 in index.

Pass `out_of_range="raise"` for anything that sweeps wavelength. The default
warns and extrapolates, which is how a bandwidth plot quietly ends up outside
the range its dispersion formula was ever fitted to.

`materials.py` carries the index as complex `n + ik`. The EME layer does not:
its backward-mode basis is built by conjugation and `Im(n_eff) < 0` is clipped
to zero, both of which assume a real index. A material with real absorption —
a metal — therefore raises a warning and only its real part is used.

**Changing a material invalidates the cache.** The dataset records its
materials in `fingerprint.json` and refuses to open against a different stack;
delete the `.pkl` files to regenerate.

**Bends** are handled by the conformal transformation: a bend of radius
`R = 1/κ` is solved as a straight guide with `n_eq(x, y) = n(x, y)·exp(κx)`.
Keep the `curvature` axis symmetric about zero unless every device you will
simulate bends only one way.

**The grid must be fixed.** Overlap integrals between two parameter points are
only meaningful if both are sampled on the same `(x, y)` mesh, so the solve
window is chosen once for the widest cross section in the sweep and never
changes. `EmepyFDE` raises if the grid ever moves.
