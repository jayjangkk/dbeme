# dbeme

**Dataset-based eigenmode expansion for integrated photonics: cached mode
datasets, lossy/PML and FEM backends, and a differentiable circuit layer** — a
port and extension of
[thdwotjd/dataset-based-eme](https://github.com/thdwotjd/dataset-based-eme)
(Song & Sohn, *Opt. Express* **33**, 46815 (2025)), with the commercial
Lumerical FDE solver replaced by the open-source
[emepy](https://github.com/BYUCamachoLab/emepy) finite-difference solver, so
everything runs on a plain Python install.

A cross section is solved once; every device that walks through the same grid
points afterwards is free. On the two-axis Kocabaş converter dataset
(471 × 347 grid, 20 modes) the 52-point design path took **15 034 s** cold, and
each of the eight taper lengths swept after it cost **0 s**
([reports/13](reports/13_kocabas_converter.md) §4).

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

## What's in the box

| layer | what | where |
|---|---|---|
| mode solvers | three backends behind one `FDEBackend` contract, chosen per dataset: `EmepyFDE` (lossless finite difference, ~1 s per cross section), `PMLBackend` (complex-stretched PML: complex `n_eff`, radiation and metal loss), `FemwellBackend` (boundary-conforming FEM for curved metal, optional `[fem]` extra) | `dbeme/fde/` |
| datasets | a parameter grid of cross sections with cached `n_eff`, polarisation and neighbour-only overlaps; an identity guard refuses a cache built for another stack, mesh or λ | `dbeme/data_updater/`, `datasets/` |
| devices | tapers, Bézier and Euler bends, S-bends, parametric paths through any grid, Hungarian mode tracking through anti-crossings | `dbeme/geometry/` |
| cascade | transfer and scattering-matrix routes, interface projection, Redheffer products | `dbeme/propagator/` |
| materials | `n(λ)` from refractiveindex.info via PyOptik, Sellmeier fallbacks, tabulated owner data | `dbeme/fde/materials.py` |
| circuit | sax models of DBEME blocks, phase fits that survive coarse λ sampling, analytic rings, a circulax delay-line extension | `dbeme/circuit/` |
| inverse design | a JAX cascade whose S-matrix is exactly differentiable in every section length, usable inside a sax circuit loss | `dbeme/circuit/cascade_jnp.py` |
| analytic references | Airy bent slab, 1-D FD PML, SPP and MIM gap plasmon, closed-form rings | `dbeme/reference/`, `dbeme/circuit/ring.py` |
| validation | reciprocity, symmetry, power, basis convergence, slicing independence, tracking checks | `dbeme/validation.py`, `tests/` |

Shipped datasets are literature stacks: 220 nm SOI strips, pairs, bi-level
ribs, RAC regions and ring couplers, the Sacher 2014 SiN/Si PRS stack, and the
plasmonic slot and Kocabaş converters.

## Validation at a glance

| check | result | source |
|---|---|---|
| constant-width guide (every interface against itself) | T = 1.0000 ± 2e-4 after biorthogonalisation (1.59 before) | [docs/validation_backlog.md](docs/validation_backlog.md) §5.6 |
| reciprocity, lossless basis | `T_forward = T_backwardᵀ` to 2e-5, reflection blocks symmetric to 1e-12 | [docs/accuracy_and_limitations.md](docs/accuracy_and_limitations.md) |
| reciprocity, lossy truncated basis | 1e-3 on the launched channel, and why that is not a conjugation error | [reports/13](reports/13_kocabas_converter.md) §11 |
| PML against the exact Airy bent slab | worst case 2.03 % over 5.6 decades of `Im n_eff` | [reports/06](reports/06_pml_phase1_gate.md) |
| bend loss against Vlasov & McNab 2004 (measured) | within 19 % at R = 1 µm, below the measurement as it must be | [reports/06](reports/06_pml_phase1_gate.md) §6 |
| backend against emepy's own rectangle solver | 1e-3 | [docs/accuracy_and_limitations.md](docs/accuracy_and_limitations.md) |
| dataset against direct (ungridded) EME | tapers agree to ~2e-5; warm sweep 460× faster | [docs/accuracy_and_limitations.md](docs/accuracy_and_limitations.md) |
| Kocabaş Si-to-slot converter against the paper | 87.0 % modal, 89–91 % on the paper's own measure, against its ~95 % | [reports/13](reports/13_kocabas_converter.md) §9–§12 |

## What it is not

* Not FDTD: a modal method in the frequency domain; nothing outside the mode basis exists.
* On the default `EmepyFDE` backend, not a loss model: `n_eff` is real and bend radiation is zero - use `PMLBackend`.
* Not a wavelength grid: one dataset per wavelength, and a λ sweep is a set of datasets.
* Not converged by default: a metal corner, an anti-crossing or a staircase floor each need the checks in `docs/validation_backlog.md`.
* Not a PDK: no foundry stack ships here; platforms are yours to define (`docs/defining_a_platform.md`).

## Install

Python 3.11+, one virtual environment (developed on 3.13, Windows).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install emepy --no-deps
.venv/Scripts/python -m pip install -e ".[fem,circuit,gds,test]"
.venv/Scripts/python -c "from PyOptik import download_snapshot; download_snapshot()"
.venv/Scripts/python -m pytest -q
```

`emepy` goes in first with `--no-deps`: it pins `simphony` 0.6 and
`tidy3d-beta` for its own EME engine and circuit export, which this package
does not use; `dbeme/fde/_compat.py` loads only its mode solver. The extras are
optional - `fem` (femwell, GPL-3.0), `circuit` (jax, sax), `gds`
(gdsfactory). The PyOptik snapshot (~100 MB, once) is optional too: without it
the built-in Sellmeier formulas take over with a warning. The circulax time-
domain tests (`tests_circuit/`) need their own environment with circulax's
jax and sax pins.

The upstream sources are git submodules under `ref_dbeme/`, `ref_emepy/` and
`ref_pyoptik/` (`git submodule update --init`), for reference only - nothing
imports them.

## Quick start

```python
from dbeme import DataUpdater, LinearTaper, EME, Runner
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

More: [docs/demos.md](docs/demos.md) lists every demo, its runtime and what it
shows.

## Concepts

* **One dataset per cross-section family, never per device.** The grid is keyed
  to the mode problem (stack, λ, mesh, window, mode count); the longitudinal
  path is free. How to lay out axes that stay affordable in several
  dimensions: [docs/dataset_doctrine.md](docs/dataset_doctrine.md).
* **What has been checked, and what must be before a number is reported** -
  the reciprocity, convergence, slicing and gauge checks with their pass
  criteria: [docs/validation_backlog.md](docs/validation_backlog.md). Its
  limits in plain terms: [docs/accuracy_and_limitations.md](docs/accuracy_and_limitations.md).
* **Two cascade routes** - transfer matrices for lossless bases, scattering
  matrices for lossy ones: [docs/cascade_routes.md](docs/cascade_routes.md).
* **Circuits** - rings, Mach-Zehnders and filters as sax circuits of DBEME
  blocks, and the time domain through circulax: [docs/circuit_layer.md](docs/circuit_layer.md).
* **Your own platform** - a dataset is one `dataset_info.py`; a new geometry is
  one `index(x, y, params)` method: [docs/defining_a_platform.md](docs/defining_a_platform.md).
* **How demos and reports are written** - the comparison protocol and the
  report template: [docs/demo_plan.md](docs/demo_plan.md); literature:
  [docs/references.md](docs/references.md).

## Reports

| # | device | what it shows |
|---|---|---|
| [01](reports/01_adiabatic_coupler.md) | adiabatic TE0/TE1 coupler (Sacher 2014) | the anti-crossing needs a ≲5 nm grid; published length sits just above the computed threshold |
| [02](reports/02_polarization_rotator.md) | bi-level taper rotator (Sacher 2014) | TM0 → TE1 at 99.6 % |
| [03](reports/03_oband_wavelength_sweep.md) | the same PRS over the O band | the rotator holds, the coupler collapses to 34 % |
| [04](reports/04_sidewall_angle.md) | the rotator with slanted sidewalls | etch angle is not a design variable for this device |
| [05](reports/05_rapid_adiabatic_coupler.md) | rapid adiabatic coupler (Fargas Cabanillas) | the θ_RAC trajectory cuts crosstalk 98× at the same length |
| [06](reports/06_pml_phase1_gate.md) | PML backend gate | Airy bent slab, Vlasov & McNab, mode count, the two cascade routes |
| [12](reports/12_plasmonic_converter.md) | lateral plasmonic converter (NTT) | a passive lossy cascade; why the paper's optimum is not reproducible in 2-D |
| [13](reports/13_kocabas_converter.md) | Si-to-slot converter, silica-embedded (Kocabaş) | converged to 87 % on the FEM backend; the first adiabatic plasmonic device here |
| [16](reports/16_ring_resonator.md) | ring resonator, bus-ring coupler | DBEME point coupler inside a sax ring circuit |
| [16_1](reports/16_1_MRM.md) | electro-optic microring modulator | the circuit layer with a modulated ring |

Some reports cite companion work on a private platform (reports 07-11, 14, 15,
17-20); those references are kept for the record and are listed in
`scripts/check_links_companion.txt`.

---

## What changed relative to upstream

The physics, the dataset format, the geometry classes and the EME solver are
upstream's. The solver boundary and the Python-3.13 compatibility work are new.

**New — `dbeme/fde/`**

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

**New — `dbeme/reference/`**

Closed-form rulers the solvers are checked against, none of which import a
solver: the bent slab's exact Airy solution and a 1-D finite-difference PML
(`bent_slab.py`, `fd1d_pml.py`), and the single-interface SPP and the
symmetric MIM gap plasmon (`plasmonic.py`).

**New — `dbeme/geometry/parametric_path.py`**

`ParametricPath` / `DirectParametricPath`: a device given as one function of
propagation length per dataset parameter, snapped onto the grid (or not, for the
direct reference). `SingleWaveguide` hard-codes the parameter set of a one-core
dataset; this takes any of them, which is what the coupled-pair devices need.

**New — `dbeme/platforms.py`**

Each published stack written down once, so a dataset file is just a wavelength.
A DBEME dataset is keyed to the mode problem and λ is part of that key, so a
wavelength sweep is one dataset per wavelength — *not* a λ grid axis, which
would compute neighbour overlaps between wavelengths that nothing can use.

**New — `dbeme/validation.py`**

The Tier-1 invariant checks packaged so a report can open with a table of what
was measured: reciprocity, reflection symmetry, power conservation with the
unitary projection off, mode-basis convergence, slicing independence and branch
tracking.

**New — `dbeme/data_updater/dataset_identity.py`**

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

## Citing

If you use the method, cite Song & Sohn; if you use the emepy solver path,
cite Hammond et al.; `CITATION.cff` describes this repository.

* J. Song and Y.-I. Sohn, *Ultra-fast and accurate multimode waveguide design
  based on a dataset-based eigenmode expansion method*, Opt. Express **33**(22),
  46815–46827 (2025), [doi:10.1364/OE.567425](https://doi.org/10.1364/OE.567425).
* I. M. Hammond, A. M. Hammond and R. M. Camacho, *Deep learning-enhanced,
  open-source eigenmode expansion*, Opt. Lett. **47**(6), 1383–1386 (2022),
  [doi:10.1364/OL.443664](https://doi.org/10.1364/OL.443664).

## Licence

MIT (`LICENSE`), matching upstream dataset-based-eme, emepy and PyOptik; the
derived parts of `dbeme/` keep upstream's copyright notice. Dependencies keep
their own licences: EMpy is BSD, sax and circulax Apache-2.0, and femwell
GPL-3.0 - which is why the FEM backend is the optional `[fem]` extra and
nothing else imports it.
