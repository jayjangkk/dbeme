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
| [22](reports/22_edge_coupler_oband.md) | O-band bilayer double-tip edge coupler (Wan & Wang 2025) | fibre launch on a lossy basis, full-stack facet joined to an oxide-stack path; Fig. 3 reproduced; TE within 0.1 dB of the paper's 3D-FDTD, TM 0.5-0.6 dB high (PDL gate fails) once the width lattice is removed; Phase 4 length optimisation at zero mode solves, facet CMA-ES |

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
| `cross_section.py` | `FullEtchStrip` (one core), `CoupledStrips` (two cores + gap), `BiLevelStrip` (rib, for polarization rotation), `BiLevelPair` (two mirrored arms at two heights over a BOX/substrate stack, merging into one strip at `gap = 0` — the double-tip edge coupler, report 22) — replacing the `.lms` model file |
| `materials.py` | `n(λ)` from [refractiveindex.info](https://refractiveindex.info) via [PyOptik](https://github.com/MartinPdeS/PyOptik), with built-in Sellmeier fallbacks |
| `assemble.py` | field normalisation, the backward-mode basis, and the overlap matrices |
| `_compat.py` | shims that let emepy/EMpy import and run on NumPy 2 / SciPy 1.18 / Python 3.13 |
| `pml.py` | `PMLModeSolver` / `PMLBackend`: EMpy's vectorial solver on a complex-stretched grid, shift-invert mode selection and a core-confinement filter, so `n_eff` is complex and radiation and metal loss exist (reports 06, 12). A PML depth per edge (`pml_thickness={edge: depth}`); `TargetRulePMLBackend`, a shift-invert target chosen point by point, for a path whose guided index spans the PML's Berenger band (report 22 §2.3); the shift-invert factorised with a minimum-degree ordering on `Aᵀ + A` (fill 9.8 M → 5.7 M, `eigs` 102 → 63 s on 146 k unknowns, eigenvalues unchanged to 1e-13) |
| `../propagator/fiber.py` | a Gaussian fibre beam projected onto a cross section's modes in the dataset's unconjugated, biorthogonal basis: the launch vector of an edge coupler (Fresnel-exact per-mode transmission), the paper-style scalar overlap and the conjugated power coupling (report 22 §2.6) |
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
* **One sign per mode in both overlap sets**
  (`geometry/geometry.py::_equalize_overlap_phase`, and the same method in
  `geometry/direct_geometry.py`).
  * **The bug.** The equalisation gives each mode a ±1 sign per section.
    Upstream derived it twice: once from the diagonal of `overlap_ab`, and
    again, independently, from the diagonal of `overlap_ba`.
  * **Why that is wrong.** A sign belongs to the mode's field, and the
    interface formulas combine `O_ab` with `O_ba`, so both sets must carry
    the same sign.
  * **When it goes wrong.** In a lossless basis `O_ba ≈ O_abᵀ`, and the two
    passes agree. On a PML basis a radiating branch can have diagonals of
    opposite sign in the two sets, typically where the tracking alternates
    between a continuum and a Berenger mode, or where a branch enters as a
    bound mode at |diag| ~ 1e-7. From that interface on, the branch carried
    opposite signs in `O_ab` and `O_ba`, and its couplings to the other modes
    moved between `T` and `R`. Only the parity of these events counts.
  * **Results it changed:**
    * **Report 22.** At 1260 and 1360 nm the branch that becomes TE2 had
      five such events upstream, and it couples to TE0 at |O| ≈ 0.24 where
      the arms merge.
      * Under upstream's reflection block with the cap, where this was first
        found, 7.0 % / 5.8 % of TE0 came out as reflection there. The
        band-edge TE loss read 3.79 / 3.81 dB instead of 2.15 / 2.00 dB.
      * On the final code the two-mask rule makes that interface non-passive
        for TE0: 1.073 transmitted and 7.3 % reflected at 1260 nm. The
        band-edge TE loss (raw) reads 1.81 / 1.33 dB instead of
        2.07 / 1.89 dB.
      * At 1310 nm the count was four, so the result came out right.
      * The study had worked around this with an override, which is now
        removed.
    * **Report 13 §12.** The 312-section cascade
      (`SiO2_kocabas_set2_1550_fem_c20_hs1_w2.5`) returned 87.3 %, a deficit
      of −0.19 and a lumped passivity of 6.9 without the interface column
      cap. That was the case that motivated `INTERFACE_COLUMN_CAP`. With one
      gauge and upstream's reflection block it is passive uncapped (0.96,
      85.5 %). With the corrected reflection block it is not passive; see
      "Interface reflection block" below.
  * **The fix.** The masks now come from `overlap_ab` alone and are applied to
    `overlap_ba` with rows and columns swapped (`ba[i]` holds section `i+1`
    in its rows).
  * **What does not change.** On paths of three or more interfaces with no
    exactly-zero diagonal, `overlap_ab` is bit-identical, and so is
    `overlap_ba` wherever the two diagonals never disagree. That covers the
    lossless datasets checked and every warm path of reports 12 and 13 on
    their headline datasets.
  * **Also fixed in the same method:**
    * a zero diagonal no longer deletes its mode for the rest of the path
      (`np.sign(0)` was used as a mask); the mode keeps the previous
      section's sign;
    * the last section of a one- or two-interface path is now equalised;
    * the method runs without an `overlap_dict`.
  * **Evidence:** `tests/test_overlap_gauge.py`. The previous rule is kept
    verbatim in `studies/edge_coupler/archive_checks.py::_two_mask_rule`.
* **Interface reflection block**
  (`propagator/single_propagator/single_eme.py::_calc_reflection_matrix`,
  `_calc_interface_Smatrix`, and the T→S conversions in
  `matrix_calculation_tool.py`).
  * **The bug.** Upstream computed `R12 = ½(O_abᵀ − O_ba)·T12`, a product of
    two section-(k+1)-by-k matrices. That is not a reflection block of either
    section. It is not gauge-covariant, and for one mode it is `−r12`. The
    T→S conversion put a minus on its (1,2) block, making it neither the
    textbook conversion nor its own inverse.
  * **What that did.** Together the two gave a lossless Fresnel step
    `[[t, r21], [−r12, t]]`, with |SᴴS − I| = 2t|r|. They also gave the
    wrong sign on every round trip, so a Fabry–Perot slab cascaded to more
    than unit transmission in the complete-basis model of the tests. `MultiPropagator` had the right
    reflection formula but the same conversion.
  * **The fix.** Tangential continuity projected on the modes of the section
    being left gives `R12 = ½(O_baᵀ − O_ab)·T12`, used with either
    projection's `T12`. The conversion is `n12 = +m12·inv(m22)`, and the
    interface matrix is `[[T12, R21], [R12, T21]]`.
  * **Why not the other projection's reflection.** Projecting on the modes of
    the section entered gives `R12 = inv(O_abᵀ + O_ba)(O_ba − O_abᵀ)`. It is
    equal in a complete basis, but on a truncated PML basis it has no factor
    of `T` to damp the modes the basis cannot represent. Its reflection
    columns reached 3.1× (`R12`) and 4.1× (`R21`) unit power uncapped
    (report 13's 40-mode path).
  * **Checked against exact references** (`tests/test_interface_reflection.py`):
    a Fresnel step; a complete-basis slab junction, lossless and lossy; a
    Fabry–Perot cascade against the global field solution; gauge covariance;
    and the conversion, both ways.
  * **What moved:**
    * **The lossless route gate** (`examples/verify_smatrix_routes.py`:
      the README linear taper, the Bezier S-bend and report 05's RAC; the
      datasets of reports 01–04 are skipped, because their caches predate
      the basis convention):
      * headline transmissions move by ≤ 2.2e-4;
      * the two routes agree to 1.2e-3 on the full matrix, against 2.7e-3
        before;
      * the median per-interface |SᴴS − I| on the linear taper is 2.5e-3,
        against 9.2e-3 before.
    * **Report 12:** −1.61 → −1.69 dB at 600 nm. The length sweep is no
      longer monotonic: −1.75 / −1.66 / −1.66 / −1.69 dB at 150 / 300 /
      450 / 600 nm. This comes from coherent round trips over the 40-step
      staircase; with the reflection blocks zeroed the sweep is monotonic.
    * **Report 13:** design point 72.0 → 72.2 % under the cap (72.5 %
      at the new default, cap off). The variants move −0.15 to +0.4 pt
      (`examples/kocabas_gauge_check.py`).
    * **Report 22, under the cap:** the losses move ≤ 0.008 dB, and the TE
      reciprocity defect falls from 2.6–4.1 % to 0.2–0.5 %. Uncapped it is
      0.03–0.05 %.
  * **One result became non-passive.** Report 13's 312-section cascade
    went from passive (0.96 uncapped, with one gauge) to a lumped passivity
    of 28 (capped) / 38 (uncapped).
    * The continuum's transmission is not passive: with every reflection
      block zeroed, the uncapped transmission chain alone compounds to 38,
      forward and into non-physical outputs.
    * The column cap holds that chain to 0.93. With the corrected
      reflections the capped path still reaches 28: under the cap the gain
      comes back through the reflection blocks.
    * Upstream's reflection block is not power-conserving (|SᴴS − I| =
      2t|r|), and it had been damping the gain.
    * Both gains turned out to be the self-overlap floor, not truncation of
      the mode set ("Reciprocal interface matrix" below).
      * The default reciprocal projection removes the reflection-borne gain.
        With the cap opted in the path is then passive (0.93, 85.7 %).
      * Uncapped at the default the tip is 85.6 % and the physical outputs
        are passive (0.851), but the transmission floor remains: 38.9 over
        all outputs.
      * The opt-in self-overlap estimate removes it too (0.934, 85.6 %).
  * **Evidence:** the previous assembly is kept in
    `examples/kocabas_gauge_check.py::upstream_interface_smatrix`. The
    previous conversion had `n12 = −m12·inv(m22)`.
* **Interface column cap is opt-in**
  (`SingleEME.INTERFACE_COLUMN_CAP`, default `False`; `"auto"` or `True`
  opts in).
  * **Why it was on.** It was introduced on 2026-09-23 against a lumped gain
    of 6.9 that came with the two-mask gauge. The per-interface excess it
    caps is the self-overlap floor of "Reciprocal interface matrix" below:
    on the corrected cascade, the 312-section path's transmission chain
    alone reaches 38 uncapped and 0.93 capped.
  * **What it does now.**
    * Without the reciprocal projection it does not keep that path passive
      once reflections are included (28). With it, the capped path is
      passive (0.93, 85.7 %).
    * Every other lossy path re-run on the final code is passive uncapped
      (at most 0.91): the report 12 demo, the report 13 demo and the design
      points of `kocabas_gauge_check.py`, and the six report-22 device
      paths.
    * The 232- and 382-section paths of the same family (report 13 §12) and
      the FEM 112-section row were not re-run, so their passivity is
      unknown.
    * Where the cap acted, it cost 0.3–2.4 pt on report 13's design point and
      variant datasets, and moved the demo sweeps by −0.3 to +0.3 pt.
    * On report 22 it cost 0.00–0.12 dB, and it degraded reciprocity (at
      1310 nm 0.41 % against 0.025 %).
  * **Evidence:** `tests/test_interface_column_cap.py`;
    `examples/kocabas_gauge_check.py` runs every configuration with the cap
    on and off.
* **Reciprocal interface matrix; self-overlap correction (opt-in)**
  (`propagator/single_propagator/single_eme.py`:
  `SingleEME.INTERFACE_RECIPROCAL`, default `"auto"`, i.e. lossy bases
  only; `SingleEME.INTERFACE_SELF_OVERLAP`, default `False`, `"estimate"`
  opts in).
  * **The defect.** Biorthogonalisation (`fde/assemble.py`) makes only the
    symmetric part of each section's unconjugated self-overlap
    `M = ½∫e_i × h_j` equal to `I`.
    * On the FEM basis the antisymmetric part `A` survives: a median column
      norm of 0.05 for continuum modes and 0.006 for physical ones.
      The origin is not settled. The overlaps are taken over the inner
      window, which leaves out the absorber ring where continuum modes still
      carry field. On saved fields the window-boundary term of the Lorentz
      identity tracks `A`, while the quadrature weights move it by only 6e-4.
      So the truncated overlap domain, not quadrature, is the likely main
      source.
    * The interface formulas assume `M = I`. With the output-side projection
      (the lossy default), an interface between a section and itself then
      transmits `I − A²` and reflects `−A(I − A²)`. The input side transmits
      `I` and reflects `−A`.
    * The reflection is antisymmetric, in phase with the transmission, and
      the same at a 0.25 nm step as at 10 nm. The transmission gains
      `2 Re(AᵀA)_jj` per continuum column (`2|A e_j|²` for real `A`).
  * **What it did.** On report 13's 311-interface FEM path (5.3 nm
    sections, continuum damped only 0.39 % per section) both compounded.
    * 36 of the 38 units of lumped gain came back through the antisymmetric
      reflections.
    * With the reflections zeroed, the transmission floor alone reached
      38.5, all of it in forward continuum.
    * The tip read 92.5 %, 7 points above every configuration without that
      gain.
  * **The default fix: reciprocal projection.**
    * Lorentz reciprocity in the unconjugated normalisation makes the
      channel-ordered interface matrix `[[R12, T21], [T12, R21]]` symmetric
      (Svendsen et al. 2013, `docs/references.md`).
      * The uncorrected transmission blocks satisfy `T21 = T12ᵀ` on any
        basis.
      * The reflection blocks are symmetric only on a complete basis with
        `M = I`. Truncation also breaks it, and so, on FD-PML bases, do
        degenerate continuum pairs.
    * Each lossy interface matrix is now projected onto its reciprocal part:
      `R ← ½(R + Rᵀ)` for both reflection blocks, `T12 ← ½(T12 + T21ᵀ)`.
    * This removes the zero-step reflection exactly.
      * At a finite step `δ` it differs from the self-overlap-corrected
        reflection by `O(A·δ)`: a relative `O(A)` error on that step's
        reflection, since the corrected reflection is itself not symmetric
        at that order.
      * It changes nothing on a complete basis with `M = I`, and it is
        gauge-covariant.
    * Lossless bases keep the validated route bit for bit.
  * **The opt-in fix: self-overlap estimate.** `"estimate"` puts
    `M = I + A` into the projected continuity equations, with `A` read off
    the antisymmetric part of the uncorrected reflection blocks. This is a
    Galerkin projection with the modes' Gram matrix, as in CAMFR's
    non-orthogonal interface. A zero step is then the identity to `O(A³)`.
    With the default projection on, the corrected blocks are projected
    afterwards; they break `T21 = T12ᵀ` by up to 5.4e-2, and projecting them
    moves the tip by 0.01 points.
    * **Against exact `M`**, from fresh solves of sections 200–311 of the
      312-section path:
      * the estimate matches `A` to 2–3 %;
      * the corrected equations bring that segment's transmission chain from
        43.4 to 1.004;
      * the full cascade agrees with the exact-`M` hybrid to 0.02–0.03
        points (85.59 % against 85.61–85.62 %);
      * in the offline replica the path is passive in the conjugated
        Poynting metric too (gain 0.98).
    * **Why opt-in.** The estimate holds only where the reflection asymmetry
      is the self-overlap error. On report 22's FD-PML bases it reaches norm
      6.6, at degenerate continuum pairs. Above
      `SELF_OVERLAP_ESTIMATE_LIMIT` (0.3) it raises instead of guessing.
      * The limit guards only the zero-step series.
      * At a real step on a truncated basis, truncation asymmetry enters the
        estimate as well (27 % off at one FD section).
      * Against exact `M` it is checked only on the FEM path.
  * **Considered and not adopted:** a passivity clip in the conjugated
    Poynting metric of the modes.
    * It needs that Gram matrix at every point, which the datasets do not
      store.
    * It is not sufficient: propagation itself is not contractive in that
      metric. With every interface of the 250–311 segment clipped, the
      segment still gains 1.09.
    * It costs the launched channel 1.4e-4 per interface.
    * It would distort the passive FD paths: on the two FD interfaces
      checked it lowers the launched `|T11|²`.
      * Their flux gain is not caused by `A` (exact `M` changes it by less
        than 0.006).
      * Whether it is truncation or an artefact of that metric, which gains
        1.07–1.18 per section in pure propagation, was not settled.
    * An exact treatment for future datasets is to store each point's `M`.
      It is already formed in `biorthogonalise`, so it is free at build
      time.
  * **Checked** (`tests/test_reciprocal_interface.py`, 42 cases; mutants of
    the new code are killed):
    * zero step with `M = I + A`;
    * a 2 nm step of the complete lossy slab with a perturbed overlap
      quadrature. On the output side, production misses the exact interface
      by 8e-2, the projection by 5e-2 and the estimate by 3e-3;
    * the corrected equations fed the true `M` reproduce it to 1e-12;
    * complete-basis exactness on both routes and both projections;
    * gauge covariance;
    * a reciprocal truncated cascade;
    * lossless bases unchanged;
    * the transfer route still without the cutoff;
    * the cap applied after the projection;
    * the regime limit on either section.
    * The full suite has 473 passed.
  * **What moved** (warm re-runs, cap off, against the 2026-09-30 code):
    * **Report 13, 312-section path:** tip 92.48 → 85.59 %.
      * Lumped passivity: 38.1 → 38.9 over all outputs, 9.72 → 0.851 over
        physical outputs.
      * With the estimate: 85.59 % and passivity 0.934.
      * This agrees with the reflections-zeroed 85.4 % and with upstream's
        block under one gauge, 85.5 % (`examples/kocabas_gauge_check.py`).
    * **Report 13, design point and variants:** 72.53 → 72.42 %; the variants
      move by −0.34 to +0.10 points. With the estimate the design point and
      variants are −0.19 to +0.03 points from the 2026-09-30 values.
    * **Report 12:** −1.69 dB at 600 nm, unchanged (−1.690 → −1.687).
      * The short tapers gain 0.01 dB: −1.74 / −1.65 dB at 150 / 300 nm.
      * Gate passivity 0.690 → 0.691.
      * The estimate moves these by 0.0003 dB or less.
    * **Report 22** (regenerated on this code, 2026-10-02): best estimates
      move by at most +0.009 dB (TM, 1310 nm).
      * TE 0.92 / 0.86 / 0.94, TM 1.56 / 1.59 / 1.88, PDL 0.64 / 0.73 /
        0.94 dB.
      * The 1310 nm TM excess over the paper is now +0.4999 dB. That passes
        the task's 0.5 dB criterion by 8e-5 dB, well inside the ±0.01 dB
        digitisation error, so the call is not resolved.
    * **Reciprocity:** an uncapped lossy cascade is now reciprocal to
      round-off by construction (1e-14 or less).
      * The reciprocity rows of the Kocabas, plasmonic and edge-coupler gates,
        and `examples/kocabas_reciprocity.py`, are therefore computed with
        the projection off (`single_eme.interface_switches`). They still
        measure the basis: 1.2e-3 on report 13's design point, and 5.6e-2 /
        1.0e-3 on report 12, as before.
      * The projection also hides a sign-gauge disagreement between the two
        overlap sets, which would otherwise show as reflection.
  * **Not covered:**
    * `MultiPropagator`'s own interface formulas (geometry intersections);
    * the 232- and 382-section datasets and the FEM 112-section row, which
      are not in this checkout;
    * `tests_circuit`, whose venv is not in this checkout.
* **The `pre_*` tags.** `pre_gauge_ab_ba`, `pre_reflection_formula` and
  `pre_column_cap_default` all mark commit 03e9e19, the last commit before
  these three fixes. That commit also predates the uncommitted task-18
  solver work, so the tags are markers, not separate restore points.
  `pre_passivity` is a restore point: a snapshot commit (5043bce, on no
  branch) of the whole working tree except `datasets/`, taken just before the
  reciprocal projection.

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
