# DBEME — dataset-based eigenmode expansion, on an open-source mode solver

Dataset-based EME for integrated photonics, with the commercial Lumerical MODE
FDE solver replaced by the open-source **emepy** finite-difference mode solver.
Everything runs on a plain Python install — no licence, no Ansys.

Built from two upstream projects:

| | |
|---|---|
| [thdwotjd/dataset-based-eme](https://github.com/thdwotjd/dataset-based-eme) | the DBEME method and solver (MIT). Song & Sohn, *Opt. Express* **33**, 46815 (2025), [doi:10.1364/OE.567425](https://doi.org/10.1364/OE.567425) |
| [BYUCamachoLab/emepy](https://github.com/BYUCamachoLab/emepy) | the open-source mode solver used in place of Lumerical FDE (MIT) |

Both are vendored unmodified under `ref_dbeme/` and `ref_emepy/` for reference.

---

## The idea in one paragraph

Ordinary EME slices a device along z and calls a mode solver on every slice.
Most of that work is repeated: a taper, a bend and a directional coupler all ask
for the modes of "a 700 nm wide waveguide" over and over, and sweeping a
parameter re-solves everything from scratch. DBEME instead pre-computes, on a
**discrete grid of cross-section parameters** (here: top width and curvature),
the effective indices and the mode-to-mode overlap integrals between
*neighbouring* grid points, and caches them. A device is then just a **path
through that grid**: the EME transfer matrices are assembled from table
look-ups, and no mode solving happens at all. Sweeping geometry becomes
essentially free, which is what makes optimisation over multimode devices
practical.

```
      cross section + parameter grid          a device
      (dataset_info.py)                       (Geometry)
              |                                   |
              v                                   v
   emepy FDE  ->  neff, TE_pol, overlaps  ->  path through the grid
              (cached in .pkl, computed once)     |
                                                  v
                                    interface + phase matrices  ->  S-matrix
```

---

## What changed relative to upstream

The physics, the dataset format, the geometry classes and the EME solver are
upstream's. The solver boundary and the Python-3.13 compatibility work are new.

**New — `em_simulation/fde/`**

| file | role |
|---|---|
| `base.py` | `FDEBackend` / `ModeData`: the one interface `DataUpdater` needs from a mode solver |
| `emepy_fde.py` | `EmepyFDE`, wrapping `emepy.fd.MSEMpy` on a grid held fixed across all parameter points |
| `cross_section.py` | `FullEtchStrip` (one core), `CoupledStrips` (two cores + gap), `BiLevelStrip` (rib, for polarization rotation) — replacing the `.lms` model file |
| `materials.py` | `n(λ)` from [refractiveindex.info](https://refractiveindex.info) via [PyOptik](https://github.com/MartinPdeS/PyOptik), with built-in Sellmeier fallbacks |
| `assemble.py` | field normalisation, the backward-mode basis, and the overlap matrices |
| `_compat.py` | shims that let emepy/EMpy import and run on NumPy 2 / SciPy 1.18 / Python 3.13 |
| `pml.py` | `PMLModeSolver` / `PMLBackend`: EMpy's vectorial solver on a complex-stretched grid, shift-invert mode selection and a core-confinement filter, so `n_eff` is complex and radiation and metal loss exist (reports 06, 12) |
| `slot_converter.py` | `PlasmonicSlotConverter`: a Si core between two gold walls with a gap, complex permittivity averaged in ε, rounded corners, a `core_mask` for the confinement filter; the gap can be a second path parameter (`sweep_gap`) for slots that taper independently of the Si |

**New — `em_simulation/reference/`**

Closed-form rulers the solvers are checked against, none of which import a
solver: the bent slab's exact Airy solution and a 1-D finite-difference PML
(`bent_slab.py`, `fd1d_pml.py`), and the single-interface SPP and the
symmetric MIM gap plasmon (`plasmonic.py`).

**New — `em_simulation/geometry/parametric_path.py`**

`ParametricPath` / `DirectParametricPath`: a device given as one function of
propagation length per dataset parameter, snapped onto the grid (or not, for the
direct reference). `SingleWaveguide` hard-codes the parameter set of a one-core
dataset; this takes any of them, which is what the coupled-pair devices need.

**New — `em_simulation/platforms.py`**

Each published stack written down once, so a dataset file is just a wavelength.
A DBEME dataset is keyed to the mode problem and λ is part of that key, so a
wavelength sweep is one dataset per wavelength — *not* a λ grid axis, which
would compute neighbour overlaps between wavelengths that nothing can use.

**New — `em_simulation/validation.py`**

The Tier-1 invariant checks packaged so a report can open with a table of what
was measured: reciprocity, reflection symmetry, power conservation with the
unitary projection off, mode-basis convergence, slicing independence and branch
tracking.

**New — `em_simulation/data_updater/dataset_identity.py`**

Records what a cached dataset was built for — index models, wavelength, mesh,
window, mode count, parameter grid — and refuses to serve it for anything else.
Swapping `main/Si/Li-293K` for `main/Si/Salzberg` moves the core index by
2e-3, a hundred times the accuracy the method otherwise delivers, and without
this the pickles would keep loading and the answers would keep looking
reasonable.

**Rewritten**

* `data_updater/data_updater.py` — drives an `FDEBackend` instead of `lumapi`.
  Same public API, same pickle format. Adds a mode cache (each grid point is
  adjacent to several others and would otherwise be solved repeatedly) and a
  non-interactive `populate_dataframe(..., confirm=False)`.
* `data_extractor/data_extractor.py` — the direct, uncached EME path, likewise
  moved off `lumapi`.

**Fixed**

* `matrix_calculation_tool.py` — Ray no longer starts at import time (it was
  packaging and uploading the virtualenv on every import). The `*_ray` helpers
  now fall through to the equivalent serial NumPy code by default; set
  `DBEME_USE_RAY=1` to distribute. For typical mode counts the serial path is
  faster.
* `geometry/bend_shapes/bezier.py` — `np.math.factorial` (removed in NumPy 2),
  and an arc length computed with `cumulative_trapezoid` over the segment
  lengths rather than a running sum.
* `geometry/single_waveguide/single_bezier.py` — unbound `BezierCurve` method
  calls, and the missing concrete `calc_total_length` / `_calc_xy` overrides
  that made the class impossible to instantiate.
* `geometry/single_waveguide/single_waveguide.py` — the path stopped at the
  last parameter *change*, dropping any constant-parameter run at the end and
  making the device short by up to one grid step.
* **Curvature sign** (`geometry/curvature.py`, new) — both the grid and the
  direct path took `|κ|` unconditionally, on the grounds that a left bend is
  the mirror of a right bend. That is exact for a bend that never reverses, but
  an S-bend does: folding put its two halves on the *same* grid point, so the
  join looked seamless and its mode conversion disappeared (a circular-arc
  S-bend came out at exactly 0 % crosstalk). Folding now happens only when the
  dataset's `curvature` axis has nothing negative to fold onto; a symmetric
  axis — what `datasets/Si_fulletch_220nm` now uses — keeps the sign and gets
  the join right.
* **Mode gauge** (`fde/emepy_fde.py::_pin_gauge`) — an eigenvector is defined
  only up to a complex scale and ARPACK starts from a random vector, so the
  *same* cross section solved twice came back with some modes sign-flipped
  (measured: overlaps differing by 2.0 between consecutive identical solves).
  Normalising to unit power does not remove that freedom, because
  `(E, H) → (−E, −H)` leaves `(E × H)_z` unchanged. Since the dataset persists
  overlap matrices but not fields, an overlap computed in one session and one
  computed in another were in inconsistent gauges for the same point. The
  global phase is now pinned deterministically before a mode is returned;
  repeated solves reproduce their overlaps to ~2e-8, i.e. to eigensolver
  convergence. Regression tests in `tests/test_mode_gauge.py`.
* **PML stretch sign** (`fde/pml.py::stretched_grid`) — EMpy's `stretchmesh`
  ends with `abs(Im)` on the stretched coordinate. The stretch is antisymmetric
  by construction and must be, because the operator consumes `diff(x)`, whose
  imaginary part has to carry one sign in both layers; the `abs` turned the
  `-x` and `-y` layers into **gain**. A guided mode barely notices (it is
  evanescent there), so a `+x`-only validation cannot see it; with `+x` and
  `-x` together the spectrum went PT-symmetric (`Im n_eff` of every radiating
  mode exactly 0 instead of doubling), and with a third edge the gain/loss
  corners hosted near-real Berenger modes that shift-invert returned instead of
  the physical mode. The stretch is now applied here, without the `abs`; the
  vendored routine is untouched. Report 06 §10, `tests/test_pml_solver.py`.
* **Radiation mask on a lossy basis** (`geometry/geometry.py`) — a mode was
  "radiation" if `Re n_eff` fell below the cladding *or* `|Im n_eff|` exceeded
  100 dB/cm. The second rule is a lossless-model heuristic and it condemns every
  plasmonic mode (a Si wire 20 nm from gold sits at 0.2 dB/µm, a gap plasmon at
  1–2 dB/µm); it now applies only when the dataset says it is lossless.
* **Interface projection side** (`propagator/single_propagator/single_eme.py`)
  — upstream imposes tangential continuity against the modes of the section
  being *left*, `T12 = 2 inv(O_ab + O_baᵀ)`, whose single-mode limit is
  `1/|O|²`: whatever part of the mismatch field the truncated basis cannot
  represent comes back as gain. On six-mode Si tapers that is 2.5e-4 per
  interface and `force_unitary` hid it; on the plasmonic taper of report 12
  it compounded to ×1.44 over forty interfaces. The default is now the
  section being *entered* (`INTERFACE_PROJECTION = "output"`,
  `2 O_abᵀ inv(O_abᵀ + O_ba) O_ba`, limit `|O|²`); `"input"` restores
  upstream's form. Report 06 §11.
* **Multi-step parameter jumps** (`geometry/geometry.py`) — the dataset only
  stores overlaps between *adjacent* grid points, and `interp_multi_adj_pts`
  assumed a path never moves further than that in one parameter. Continuous
  geometries never do; discontinuous ones do, and a two-arc S-bend flipping
  from +1/R to −1/R raised a `KeyError`. The path now walks the grid one step
  at a time, and the expanded path is built *before* the dataset is populated
  so the inserted points get solved.

---

## Install

Python 3.11+ (developed and tested on 3.13, Windows).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy matplotlib tqdm pyyaml shapely pytest
.venv/Scripts/python -m pip install ElectroMagneticPythonGpu
.venv/Scripts/python -m pip install emepy --no-deps
.venv/Scripts/python -m pip install PyOptik
.venv/Scripts/python -c "from PyOptik import download_snapshot; download_snapshot()"
```

or just run `./install.sh`.

`emepy` is installed with `--no-deps` on purpose: it pins `simphony` 0.6 and
`tidy3d-beta`, which are only needed for its own EME engine and its circuit
export. We use its mode solver alone, and `em_simulation/fde/_compat.py` loads
just that part of the package. Ray is optional (`pip install ray`).

`PyOptik` supplies the material data. The last line downloads the
refractiveindex.info snapshot once (~100 MB into
`%APPDATA%/PyOptik`); `pyoptik setup` does the same thing. Both PyOptik and
the snapshot are **optional** — without them the built-in Sellmeier formulas
in `materials.py` take over, with a warning, and everything still runs.

Verify:

```bash
.venv/Scripts/python -m pytest tests -q
```

---

## Quick start

```python
from em_simulation import DataUpdater, LinearTaper, EME, Runner
import numpy as np

du = DataUpdater("datasets/Si_fulletch_220nm")

taper = LinearTaper(du, input_width=0.5e-6, output_width=1.2e-6, length=10e-6)
eme = EME(taper, force_unitary=True)
eme.calc_Smatrix()

runner = Runner(eme)

# The launch vector spans the modes guided at the input; modes below the
# cladding index are excluded, so ask the geometry how many there are.
n_launch = (taper.output_data["radiation_mode_mask"][0] == False).sum()
launch = np.zeros(n_launch, dtype=complex)
launch[0] = 1.0                                  # TE0

out = np.abs(runner.propagate_lumped_smatrix(launch)) ** 2
n_modes = taper.output_data["neff"].shape[1] // 2
print(out[:n_modes])                             # power per forward mode
```

The first run for a given region of parameter space solves the cross sections
it needs and writes them to `datasets/…/{neff,overlap,TE_pol}.pkl`. Every later
run that stays inside that region does no mode solving at all.

### Demos

```bash
cd examples
python demo_linear_taper.py
python demo_bezier_bend.py
python demo_material_dispersion.py
python demo_adiabatic_coupler.py       # -> reports/01
python demo_polarization_rotator.py    # -> reports/02
python sweep_wavelength.py             # -> reports/03  (O band, ~45 min)
python demo_sidewall_angle.py          # -> reports/04  (~80 min)
python demo_rapid_adiabatic_coupler.py # -> reports/05  (~90 min)
python demo_plasmonic_converter.py     # -> reports/12  (lossy PML dataset, ~2 h cold)
python demo_kocabas_converter.py       # -> reports/13  (SiO2-embedded Si-to-slot converter, two-axis lossy dataset, hours cold)
python study_rac_bandwidth.py air      # RAC splitting vs wavelength and length
python study_rac_air_basis.py          # the air-clad bound-mode basis
python study_bend_loss_reference.py    # -> reports/06  (PML phase 1 gate, ~2 s)
python study_anticrossing_grid.py
python validate_against_direct_eme.py
```

Figures land in `examples/output/`.

### What the demos produce

**Taper** — 0.5 to 1.2 µm, 220 nm full-etch Si, TE0 launched. A symmetric taper
only couples TE0 to TE2 (TE1 is forbidden by symmetry), and the coupling is
weak: TE0 transmission passes 99.9 % by 4.9 µm of taper length, and at 10 µm
the residual TE2 crosstalk is −35 dB. The 500-length sweep behind that number
runs in about 5 s with no mode solving at all.

**Bend** — S-bend, 15 × 5 µm footprint, 1.0 µm wide multimode guide with 2 µm
straight leads at each end, TE0 launched. A bend *does* couple TE0 to TE1, and
the coupling is driven by how abruptly the curvature changes:

| centreline | curvature steps | power leaving TE0 |
|---|---|---|
| circular arc | 3 (in, join, out) | **2.91 %** (−15.4 dB) |
| cubic Bézier | 2 (in, out) | **0.76 %** (−21.2 dB) |
| quintic Bézier | none | **0.057 %** (−32.5 dB) |

Same footprint, same dataset, 50× less crosstalk. The whole comparison is three
different paths through one set of cached cross sections.

**Polarization rotator-splitter** — both stages of the Sacher et al. PRS
([Opt. Express 22, 3777](https://doi.org/10.1364/OE.22.003777)), each on its
published geometry, with three write-ups in [reports/](reports/):

| stage | result | published length | 99.9 % threshold here |
|---|---|---|---|
| [bi-level taper](reports/02_polarization_rotator.md) | TM0 → TE1 at **99.6 %**, TE fraction 0.065 → 0.965 | 100 µm | 76 µm |
| [adiabatic coupler](reports/01_adiabatic_coupler.md) | TE0 stays **98.9 %** broad, TE1 exits **98.2 %** narrow | 300 µm | 304 µm |

Both published lengths sit just above the computed adiabaticity thresholds,
which is a stronger check than either device alone. The
[O-band sweep](reports/03_oband_wavelength_sweep.md) then finds the rotator
transfers to 1260–1360 nm unchanged (≥99.9 %) while the coupler collapses to
34 %, because its anti-crossing is evanescent coupling across a *fixed* 200 nm
gap and tighter confinement at short λ shrinks it 3×.

**Rapid adiabatic coupler** — the coupling region of the first experimental RAC
(Fargas Cabanillas thesis §3.5.3, 220 nm SOI e-beam), in
[reports/05](reports/05_rapid_adiabatic_coupler.md). A RAC keeps the width
schedule of an ordinary adiabatic coupler and changes only its *transverse
position evolution*: a global tilt adds the same `tan θ` to all four wall
slopes, and because the four walls contribute to κ₁₂ with opposite signs there
is a tilt at each position that nulls the sum.

| 11.5 µm coupling region, 1550 nm | crosstalk | splitting |
|---|---|---|
| untilted adiabatic coupler | 2.17e-2 | 48.12 % |
| best constant tilt, −2.50° | 4.30e-4 (51×) | 51.62 % |
| **θ_RAC(η) trajectory** | **2.22e-4 (98×)** | **49.37 %** |

Same cross sections in all three rows — only the lateral offset schedule
differs, carried as one extra dataset axis. Against the thesis: a 3.13× length
advantage over a conventional AC (3.65× published) and 50 ± 1.62 % splitting
over 110 nm (50 ± 1.4 % over 145 nm measured, for the full four-region device).
Two lessons came out of it — a dataset axis must be commensurate with the
*slope* of the device trajectory, not merely fine, and the zero-coupling branch
must be traced by continuation rather than by scan order, which is worth 29× on
its own.

**Sidewall angle** — the same rotator with the etches at 85° and 87° instead of
vertical ([reports/04](reports/04_sidewall_angle.md)). Conversion moves from
99.99 % to 99.93 %, and stays above 99.90 % down to 82°, so for this device the
etch angle is not a design variable. The reason is worth knowing: a trapezoid
has no horizontal mirror plane, so a slant is *itself* a rotation mechanism —
on a plain strip it turns the TE1/TM0 crossing into an anti-crossing with
Δn_eff = 0.021 — but the partial etch already opens 0.174, eight times more, so
the rib saturates the symmetry breaking and inherits the tolerance.

**Materials** — Si moves by 0.037 in index between 1260 and 1625 nm, SiO2 by
0.004, Si3N4 by 0.011. Freezing the index at its 1550 nm value costs 0.034 in
n_eff by 1260 nm for a 500 × 220 nm strip, and — worse — it deletes the
`λ·dn/dλ` term from the group index, which comes out 3 % low *at the reference
wavelength itself*, where n_eff is exact by construction. Anything involving
delay, FSR or dispersion needs the real index model.

---

**Plasmonic mode converter** (`demo_plasmonic_converter.py` → `reports/12`) —
the first device on the lossy PML backend. The lateral model (Si width
400 nm → 0 between 400 nm gold walls at a 20 nm air gap, suspended) has a
bound slot plasmon at 1.145 + 0.036j — 1.27 dB/µm, the paper's "about
1 dB/µm" — and its 41-interface cascade carries propagation loss exactly and
is passive, but only after the interface projection was moved to the output
side (below): upstream's form returned the mismatch field the basis cannot
hold as *gain*, 0.2–1.9 % per 10 nm step and ×1.44 over the taper,
independently of mode count, of the PML and of basis selection. The passive
cascade gives −1.61 dB at 600 nm on co-located E/H fields (−1.85 with sharp
corners on the same fields), of which ≈ −1.4 dB is the mismatch loss of a
40-step staircase and the rest metal loss, monotonic in length with no optimum: the grid-snapped path visits the same 41 cross sections at every
length, and the shed field that would have to interfere destructively for a
taper to be adiabatic is in no basis of this kind. So the paper's −1 dB and
its 600 nm optimum are not reproduced, and the report says why they cannot
be. Five things were found on the way and are now tests: the `-x`/`-y` PML
layers were gain, the radiation mask rejected every lossy mode, a metal edge
inside a cell aliases `n_eff` by 0.1, a lateral slot binds only between tall
walls, and the projection side of the interface algebra. The shipped platform
rounds the gold corners (20 nm; `Si_plasmonic_slot_1550_sharp` keeps the sharp
variant): a sharp metal wedge carries a singular field that no grid resolves,
and rounding it moves the slot plasmon from 1.145 + 0.036j to 1.351 + 0.025j -
better bound, 0.87 instead of 1.27 dB/um - while the Si-end mode, which lives
on the flat wall face, barely moves, and the taper's per-step mismatch does not
move at all: the corners were the slot's problem, the flat-wall gap field is the
taper's (report 12 section 8).

---

**Piecewise-refined solve grids** (`PMLModeSolver(refine_x=, refine_y=)`) —
the uniform grid can be cut into finer cells inside chosen regions, with the
PML and the outer cells untouched; the finite-difference operator already took
per-cell spacings, and the three places that inferred cell geometry from the
centres alone (the cross-section fill fractions, the rounded-corner fill, the
power sums behind confinement and `TE_pol`) now reconstruct exact cell edges
and weight by area. A refinement enters the dataset identity. Strips of the
Kocabaş cross section (`--refine LO HI CELL`, axes `(w_si, half_slot)`):
the Si tip at 1 nm (`--tip 60 1`, +20 % unknowns) bought 0.7 points, because
on axes where the two edges move separately the staircase turns out to be
the gold wall's 5 nm jumps, not the silicon's steps; the wall strip at 1 nm
(`--refine 100 300 1`, +68 %) is the one that addresses it.

**Si-wire-to-slot converter, silica-embedded** (`demo_kocabas_converter.py` →
`reports/13`) — the same physics on a device whose every dimension is
published (Kocabaş, arXiv:1801.00833, Table II Set 2: Si 400 × 725 nm, gold
250 nm, slot 250 nm, 1700 nm taper). Because both ends are embedded in SiO₂
they are both bound, which the NTT device's air core is not, and the dataset
is two-axis: `(w_si, gap)`, 471 × 347 at a 5 nm cell, 20 modes about a
shift-invert target of 2.3. The cascade gives **72.3 % into the slot mode**
where the paper reports ~95 % of total power; the deficit is 0.22 dB of
staircase mismatch, 0.19 dB of the supermode's own metal loss, and amplitude
scattered into the Berenger set. Two results matter more than the number.
**The converter is adiabatic here**: transmission rises monotonically with
taper length (65.4 → 73.9 % from 500 to 3000 nm) and the scattered amplitude
falls as roughly 1/L — the behaviour report 12's device could not show,
because there a 10 nm edge step sheds a field no shift-invert basis holds.
And **basis membership is a correctness condition**: with the target at 1.9
the wire's TE fundamental left the 16 nearest eigenvalues part-way along the
path, the tracker linked a cutoff branch instead, and the cascade read −20 dB
without failing. The target must sit among the physical branches, not below
them (CLAUDE.md §5.13a).

## Two cascade routes

`SingleEME` can build its section scattering matrices two ways, selected by
`SingleEME.SMATRIX_METHOD` or per call with `calc_Smatrix(method=...)`:

| route | what it does | when |
|---|---|---|
| `transfer` | assemble a transfer matrix per interface, convert to S at the end — the original code path | lossless datasets: every published number was produced this way and stays bit-for-bit reproducible |
| `direct` | assemble the scattering matrix straight from the overlaps; never forms a transfer matrix | lossy or PML bases, where a transfer matrix's backward block grows as `exp(+k₀n″Δz)` and cascading amplifies it without bound |
| `auto` *(default)* | `transfer` if the backend declares its modes lossless, `direct` otherwise | — |

The two are algebraically identical (`S = [[T12, −R21], [R12, T21]]` is what
converting the transfer matrix yields, with the `inv(T21)` round trip
cancelling) and agree to ~1e-7 — the transfer route's complex64 storage —
everywhere except at an interface whose mode-matching matrix is near singular,
where they differ by ~2e-5 in the guided block and ~1e-3 in the radiation
block. Neither number there is physics, which is why lossless data keeps its
published route. `examples/verify_smatrix_routes.py` is the gate;
[reports/06](reports/06_pml_phase1_gate.md) has the measurements.

## Defining your own platform

A dataset directory needs exactly one file, `dataset_info.py`, declaring the
parameter grid and the cross section. See
`datasets/Si_fulletch_220nm/dataset_info.py`; the essential part is:

```python
import numpy as np
from em_simulation.fde import EmepyFDE, FullEtchStrip, silica, silicon


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
from em_simulation.fde.materials import PyOptikMaterial, silicon, silica

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

---

## Accuracy and limitations

Read this before trusting a number.

**Validated.** The backend reproduces emepy's own rectangle solver to 1e-3, and
`tests/` checks the invariants the method rests on: the self-overlap matrix is
the identity to ~2 %, adjacent grid points overlap > 0.9, the backward basis
mirrors the forward one, and the grid is shared. A constant-width "taper" comes
out at transmission 1.000000 with zero reflection.

`validate_against_direct_eme.py` runs the same tapers through both the dataset
and the direct (ungridded) path, which share the emepy backend and the EME
solver, so the only difference is grid snapping:

| taper length | dataset | direct | difference |
|---|---|---|---|
| 2 µm | 0.993724 | 0.993703 | 2.1e-5 |
| 6 µm | 0.999130 | 0.999153 | 2.3e-5 |
| 16 µm | 0.999859 | 0.999867 | 8.2e-6 |

They agree to ~2e-5 across the whole range, including the short tapers where
the geometry changes fastest. On a warm dataset the six-point sweep took 0.2 s
against 82 s for the direct path, a 460× speed-up.

(Before the mode-gauge fix above, this table showed a 1e-2 discrepancy at 2 µm,
which was easy to misread as a grid-discretisation error. It was the gauge bug.
Any cross-method agreement claim is only as good as the gauge test that backs
it.)

**Truncated mode basis.** With `N` modes the interface matrices are slightly
non-unitary; measured here at ~1.3e-4 of the power per interface, which over
~40 sections accumulates to ~0.5 %. Because this solver models no radiation
loss (see below), the structure is lossless by construction, so the demos pass
`force_unitary=True`, which projects each section's S-matrix onto the nearest
unitary matrix and removes the drift. Do **not** use `force_passive=True`
alone — on its own it over-attenuates badly (0.63 instead of 1.00 on the taper
above).

**Radiation and metal loss exist only on the PML backend.** `MSEMpy` in
emepy 1.2.2 has no working PML, so on the default `EmepyFDE` backend effective
indices come out real and bend radiation loss is not captured: a tight bend
shows mode conversion into higher-order *guided* modes correctly but reports
zero bending loss. `PMLBackend` (`fde/pml.py`, report 06) lifts that for the
datasets that opt into it - complex `n_eff`, the scattering cascade, 16-40
modes rather than 6 - at a solve cost of 25-90 s per cross section instead of
~1 s. Modes below the cladding index are flagged by `radiation_mode_mask` and
excluded from the launch basis, which is a cutoff criterion, not a loss model;
on a lossless dataset the mask also rejects anything above 100 dB/cm as
numerical, a rule that is switched off for a lossy basis.

**Bend curvature is bounded by the solve window.** The conformal map multiplies
the index by `exp(κx)`, so at the window edge the cladding ramps up to
`n_clad · exp(κ · half_width)`. Once that reaches the guided-mode index the
solver returns modes pinned to the window wall instead of real ones. With
`n_clad = 1.444` and a 1.6 µm half-window that puts the ceiling near
`κ = 2.3e5` 1/m (R ≈ 4 µm), which is why the shipped grid stops at 2e5.
`EmepyFDE` warns if a solve crosses the line.

**An anti-crossing needs a much finer grid.** Where two branches exchange
identity, the supermodes turn over within a narrow range of the swept parameter,
and a step that is invisible elsewhere becomes destructive. Measured on the
adiabatic coupler: a 20 nm step scatters 29 % of the power between the two
crossing branches *at a single interface*, against ~0.1 % away from the
crossing. A dataset built at 20 nm there does not give a slightly worse answer —
it reports 0.44 where the truth is 0.99, inverting the design trend. Sample at
≲5 nm across the crossing and 20 nm elsewhere; a non-uniform axis costs nothing
because only visited points are solved. See
[reports/01_adiabatic_coupler.md](reports/01_adiabatic_coupler.md) §2.

**Reciprocity has a layout gotcha.** `_convert_3Dmatrix` produces
`[[T_forward, R_right], [R_left, T_backward]]`, *not* the textbook
`[[S11, S12], [S21, S22]]`. So `S == S.T` is not the reciprocity test and fails
at 1e-2 on a perfectly good taper. The real conditions are
`T_forward == T_backward.T` and each reflection block symmetric; those hold to
2e-5 and 1e-12. `em_simulation.validation.check_reciprocity` implements the
correct one.

**The grid step is probably a floor — not yet proven.** Snapping onto a 20 nm
width grid turns a smooth taper into a staircase, and each step scatters. Past
the adiabatic threshold the residual mode conversion stops falling with device
length and settles on a fringed plateau near 1e-4
(`taper_4_length_sweep.png`). The natural explanation is the width staircase,
in which case the floor should scale as `(Δw)²` — each step is a small
perturbation, so scattered *amplitude* goes as `Δw` and power as `Δw²`. **That
scaling has not been measured here.** Until it is, treat the plateau as an
upper bound on the physical conversion, not as a physical result: rebuild the
`top_width` axis at 10 nm and 5 nm and check the floor drops 4× and 16×. If it
does not, the plateau is numerical.

**Mesh.** The default 20 nm transverse cells put the guided-mode effective
indices within ~0.002 of the converged value. `MESH` in `dataset_info.py`
trades that against solve time (~1.3 s per cross section at the default).

**Absolute effective indices.** A 500 × 220 nm SiO2-buried Si strip comes out at
n_eff(TE0) = 2.449, n_eff(TM0) = 1.788 at 1550 nm. If you are comparing against
air-clad numbers from the literature, note this is the symmetric buried case.

**One wavelength per dataset.** Material dispersion is handled — the index is
evaluated at whatever wavelength a cross section is solved at — but the
parameter grid has no wavelength axis, so a dataset is built at one wavelength
and a bandwidth sweep means one dataset per wavelength. The plumbing for an
axis is in place: `EmepyFDE` passes `wavelength` to the cross section
alongside the geometry parameters, so adding it to the grid needs no change in
`materials.py` or `cross_section.py`.

---

## Layout

```
em_simulation/          ported DBEME core
  fde/                  NEW - the emepy mode-solver backend + materials.py,
                        pml.py (lossy PML backend), slot_converter.py
  reference/            NEW - closed-form rulers: Airy bent slab, 1-D PML,
                        SPP and MIM gap plasmon
  data_updater/         dataset build + look-up (was lumapi)
                        + dataset_identity.py, the cache guard
  data_extractor/       direct, uncached EME path (was lumapi)
  geometry/             tapers, bends, Bezier, Euler, composite
  propagator/           EME transfer and scattering matrices
  runner/               launch, propagate, sweep
datasets/
  Si_fulletch_220nm/    220 nm full-etch Si strip, SiO2 clad
  Si_pair_*, Si_bilevel_*, Si_rac_*   coupled pairs, rib, RAC (reports 01-05)
  Si_plasmonic_slot_1550[_sharp|_gap40|_c4]  lossy PML basis: Si wire in a gold slot (report 12)
  SiO2_kocabas_set2_1550               two-axis (w_si, gap) lossy basis: Kocabas's embedded converter (report 13)
backups/                timestamped snapshots of em_simulation/ (backup.py,
                        restore.py) taken before each change to the algebra
examples/               demo scripts and studies, figures in output/
reports/                device write-ups, with their figures in output/
tests/                  physics checks: backend, gauge, grid handling, materials
ref_dbeme/, ref_emepy/  upstream sources, unmodified
```

## Licence

MIT, matching both upstream projects. If you use the method, cite Song & Sohn,
*Opt. Express* **33**, 46815–46827 (2025).
