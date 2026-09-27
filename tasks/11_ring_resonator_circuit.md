# Task 11 — micro-ring resonator and the circuit layer (sax, circulax)

**Prompt to start with:**

> Read `CLAUDE.md`, then `tasks/11_ring_resonator_circuit.md`, and do Phase 1.
> Stop at the Phase 1 gate and report the acceptance table before going further.

Conventions, units and the dataset doctrine are in `CLAUDE.md`; §5.x refers to
its validation backlog. Read `reports/06_pml_phase1_gate.md` §6 and §8 and the
memory notes "S-param phase traps" and "PML: slicing before physics" before
touching phase or loss.

## Why

Jae asked whether an MRR (FSR, Q, resonance wavelength) can be done with
DBEME, and whether a circuit simulator should be added. Decisions taken:

1. **Frequency domain is exact for FSR, Q and resonance position.** They are
   linear, steady-state properties; the ring transfer function is closed-form
   in the coupler amplitudes, round-trip phase and round-trip loss (Yariv,
   *Electron. Lett.* 36, 321 (2000); Bogaerts et al., *Laser Photon. Rev.* 6,
   47 (2012)). Time domain is needed for transients, modulation and nonlinear
   or thermal dynamics, not for the static spectrum.
2. **DBEME supplies building blocks, not the circuit.** A circuit solver is a
   consumer of S-matrices downstream of `FDEBackend`; it must not be a backend.
   New code lives in `em_simulation/circuit/`.
3. **sax is the interface, circulax is the time-domain and nonlinear engine.**
   sax 0.14.7, jax 0.11.1 and klujax are already in `.venv`. circulax
   (github.com/gdsfactory/circulax: JAX, transient / DC / AC / harmonic
   balance, "existing SAX models plug in directly") is **not** installed and
   its API is known only from its README. One adapter, written once, serves
   both.
4. **Loss budget, honestly.** Report 06 §6 showed that for a 220 nm Si strip
   pure radiation bend loss is negligible past R ≈ 2 µm and measured bend
   loss is junction mismatch plus sidewall roughness. Report 06 §8 recommends
   no lossy DBEME dataset yet (mode count). So: coupler on the **lossless**
   pair dataset (ratios among guided modes, which it gets right); ring loss
   from a **direct `PMLModeSolver` solve** per (R, w, λ), the pattern of
   `studies/sirac/run_bend_loss.py`; roughness from a statistical model with
   σ and L_c as declared inputs, using a mode-field sidewall factor computed
   here. Report 06 line 354: "Roughness scattering needs a statistical model
   this code does not have." This task adds that model.

## What exists (reuse, do not rebuild)

| need | where |
|---|---|
| ring guide `n_eff(w, κ)` | `datasets/Si_fulletch_220nm` — signed κ axis; transform at `em_simulation/fde/cross_section.py:340`; ramp guard `emepy_fde._check_conformal_ramp`; valid R > 4.3 µm (§5.10) |
| two-core coupler | `CoupledStrips` (`cross_section.py:385-580`), `swept_parameters=("w1","w2","gap")` supported at L416; `datasets/Si_pair_fulletch_220nm_{1260..1360}` have gap fixed at 200 nm |
| wavelength | one dataset per λ via `em_simulation/platforms.py` factories; driver `examples/sweep_wavelength.py`; Sellmeier Si/SiO₂ in `em_simulation/fde/materials.py` |
| n_g, delay, phase | `examples/demo_material_dispersion.py::group_index`; `studies/tapeout/sparams_phase.py::dominant_delay`; convention `exp(-iωt)`, `exp(+iβz)`, `τ = +dφ/dω` (`reports/output/tapeout/sisnprs_smatrix.json` → `phase_convention`) |
| bend loss | `em_simulation/fde/pml.py::PMLModeSolver`, `turning_point`; `studies/sirac/run_bend_loss.py`; `examples/study_bend_loss_literature.py` |
| S-matrix export | `studies/tapeout/run_sparams.py` (CSV/JSON), `run_smatrix_sisnprs.py` (VPI `SmatrixMeasuredOpt` .dat) |
| per-point scalar stored in a dataset | `TE_pol.pkl` — the pattern for the new sidewall factor |

Not built: `curvature` on the pair dataset (pulley coupler). Out of scope
here; the point coupler below does not need it.

## Phase 1 — `em_simulation/circuit/`: DBEME → sax adapter, and the ring closed form

### 1.1 `em_simulation/circuit/sax_model.py`

`sax_model_from_samples(wavelengths, S, ports, modes) -> callable` returning
a sax model `f(wl=...) -> SDict` keyed `("o1@TE0", "o2@TE0")` etc. Multimode
ports use sax's `port@mode` naming.

Interpolation across the solved λ set is the crux: DBEME gives S at 5–7
wavelengths, a ring needs pm resolution. Rules:
* magnitude: interpolate `|S|` (cubic or PCHIP);
* phase: **never** interpolate wrapped phase, and never linear `n_eff(λ)`
  (memory "S-param phase traps": it staircases the delay). Unwrap against a
  continuous reference (`phase_fit.unwrap_against`, the `sparams_phase.py`
  idea), convert to the optical length `Λ = φλ/2π` and fit **Λ** with a
  polynomial in `(λ − λ0)/λ0` of degree ≤ 3 (its coefficients are
  `n_eff·L`, `n_g`, dispersion; exact whenever `n_eff` is polynomial in λ —
  a polynomial in `1/λ` of the phase itself, as first written here, spends
  a degree on the trivial carrier). Keep the residual as a check.
  *Found in Phase 1:* the reference must get the **group** index right, to
  `λ²/(2 L Δλ)` (0.4 for 300 µm at 10 nm steps), because the unwrap acts on
  the residual's step between samples; an `n_eff` offset only adds `2πm`,
  invisible to every S-parameter and delay. A ring's coupler is a few µm
  long and needs no reference; the ring arc's phase comes from `n_eff(λ)`
  directly (1.2), never from unwrapping;
* sign convention: sax's `straight` uses `exp(+i·2π·n_eff·L/λ)`; confirm
  against the project's `exp(+iβz)` on a bare waveguide before any ring
  result. Wrong sign flips the resonance side of every asymmetry.

`SDict` values must be `jnp` arrays so gradients flow (this is what circulax
needs later). Reciprocity is enforced by construction (`S = Sᵀ`).

### 1.2 `em_simulation/circuit/waveguide.py`

`arc_model(neff_of_wl, ng, length, alpha_db_per_cm)` and
`straight_model(...)`: sax models from a fitted `n_eff(λ)` and a loss. The
ring arc is `2πR` at the ring's centre radius with `n_eff(w, κ=1/R, λ)`.

### 1.3 `em_simulation/circuit/ring.py` — the analytical solution, no sax

`all_pass(r, a, phi)`, `add_drop(r1, r2, a, phi)`, and
`ring_metrics(...) -> FSR, FWHM, Q_loaded, Q_intrinsic, Q_coupling,
extinction, finesse, critical-coupling condition`. ~60 lines. This is the
reference that Phase 3's sax netlist must reproduce, and the first thing
every ring number is compared with.

The equations, as in Bogaerts et al. 2012 (doi:10.1002/lpor.201100017,
§2) and Yariv 2000 (doi:10.1049/el:20000340). `r` is the coupler's
self-coupling amplitude, `κ` its cross-coupling (`r² + κ² = 1` lossless),
`a` the single-pass amplitude (`a² = exp(−α L)`, `α` power loss per
length), `φ = β L = 2π n_eff L / λ`, `L = 2πR`.

```
all-pass, intensity:
  T_pass = (a² − 2 r a cos φ + r²) / (1 − 2 r a cos φ + r² a²)
add-drop, intensity:
  T_pass = (r₂² a² − 2 r₁ r₂ a cos φ + r₁²) / (1 − 2 r₁ r₂ a cos φ + (r₁ r₂ a)²)
  T_drop = (1 − r₁²)(1 − r₂²) a        / (1 − 2 r₁ r₂ a cos φ + (r₁ r₂ a)²)
resonance:      φ = 2πm  ⇔  n_eff L = m λ
FSR:            Δλ = λ² / (n_g L)
FWHM all-pass:  δλ = (1 − r a) λ² / (π n_g L √(r a))
FWHM add-drop:  δλ = (1 − r₁ r₂ a) λ² / (π n_g L √(r₁ r₂ a))
Q = λ/δλ ;  finesse = FSR/FWHM = π √(r a) / (1 − r a)
Q_intrinsic:    a alone (r → 1):   π n_g L √a / (λ (1 − a))
Q_coupling:     r alone (a → 1):   π n_g L √r / (λ (1 − r))  ≈ 2π n_g L / (λ κ²) for small κ
1/Q_loaded = 1/Q_intrinsic + 1/Q_coupling  (to first order)
critical coupling: all-pass r = a ; add-drop r₁ = r₂ a
```

Amplitude form for the phase check (project convention `exp(+iβz)`):
`E_pass/E_in = (r − a e^{iφ}) / (1 − r a e^{iφ})` up to a common phase.
Sign conventions differ between papers; derive it once in the project
convention, verify `|·|²` reproduces `T_pass` above, and keep that
derivation in the docstring.

Tests in `tests/test_ring_closed_form.py`: every identity above against a
brute-force evaluation of the intensity formulas on a fine `φ` grid
(peak spacing, half-max width, peak depth), lossless unitarity, critical
coupling `T_pass = 0` at resonance, and the Q-decomposition to 1 %.

### Phase 1 gate

| check | criterion |
|---|---|
| bare waveguide: adapter vs sax `straight` | phase agrees to 1e-9 rad over the band (sign confirmed) |
| lossless all-pass `|T| = 1` | max deviation < 1e-12 over λ |
| add-drop power conservation (lossless) | `|T|²+|D|² = 1` to 1e-12 |
| FSR identity `λ²/(n_g L)` from `ring_metrics` | < 0.5 % vs peak spacing |
| phase fit on the SIPR export (`reports/output/tapeout/sipr_sparams.csv`) | fitted group delay matches the export's `tau_through_ps` to 1 % (degree 2). *The 5 mrad residual first written here is unreachable on that file and measured the wrong thing:* the export's phase and its own delay column disagree by up to 1 rad when the delay is integrated over λ, and no polynomial below degree 7 gets under 0.15 rad — the 10 nm reconstruction of a 300 µm device is noise-limited at ~0.3 rad per sample. Residual asserted only against a π-flip (< 1 rad) |

Tests: `tests/test_circuit_sax_adapter.py`, `tests/test_ring_closed_form.py`.

### Phase 1 — done 2026-09-20 (measured)

`em_simulation/circuit/{__init__,phase_fit,sax_model,waveguide,ring}.py`;
27 tests. The package enables JAX float64 on import (nothing else in the
repo imports JAX). Units: SI into the factories, sax's microns and dB/cm at
the model call.

| check | criterion | measured | pass |
|---|---|---|---|
| adapter vs sax `straight`, 5 µm, 5 samples → 1001 λ | phase 1e-9 rad | 2.4e-14 rad, `‖S‖−1` 3e-16 | ✓ (sign convention confirmed: both `exp(+i2πnL/λ)`) |
| lossless all-pass `‖T‖ = 1` | 1e-12 | 0 (exact) | ✓ |
| add-drop `‖T‖²+‖D‖² = 1` lossless | 1e-12 | 1.6e-15 | ✓ |
| FSR identity vs dense scan, R = 10 µm, first-order dispersive index | 0.5 % | 9.084 vs 9.104 nm, −0.22 % (the identity is first order in dispersion) | ✓ |
| FWHM, Q, resonance depth, Q-decomposition vs scan | 1 % | all < 1 % (`test_metrics_against_dense_scan`) | ✓ |
| SIPR export phase fit, degree 2 | delay to 1 % | residual 0.52 rad (export noise, see row above), `τ` to 0.96 % | ✓ |
| **Phase 3's check, done early with the ideal coupler:** sax all-pass and add-drop ring vs `ring.py` | 1e-6 | `‖T‖` 4e-15, phase 1e-15 rad; add-drop drop/through both < 1e-14 | ✓ sax closes the loop correctly, port naming and sign right |
| long-device unwrap with a reference (300 µm, 10 nm steps) | continuous to 1e-9 | 1e-9 up to `2πm`; a constant-`n_eff` reference fails as predicted | ✓ |

Amplitude form derived in the project convention (docstring of `ring.py`):
`E_pass/E_in = (r − a e^{iφ})/(1 − r a e^{iφ})`,
`E_drop/E_in = −κ₁κ₂√a e^{iφ/2}/(1 − r₁r₂ a e^{iφ})`, with the coupler
`[[r, iκ],[iκ, r]]` and ports o1 bus in, o2 bus through, o3 ring in, o4 ring
out (`waveguide.ideal_coupler`). The Phase 2/3 DBEME coupler must be exported
with that port meaning.

## Phase 2 — building blocks for a 220 nm Si ring at 1550

### 2.1 Coupler dataset with a gap axis

`Si_pair_fulletch_220nm_gap_1550` (and 1530, 1570 for n_g; add more only if
the phase-fit residual demands): `swept_parameters=("w1","w2","gap")`,
`w1 = w2 = 500 nm` fixed range (single point each is fine — the point coupler
keeps widths constant), gap axis non-uniform: 10 nm from 100–400 nm, 50 nm to
1.0 µm, then 250 nm to 2.5 µm. Rationale in CLAUDE.md §3.3. Modes: TE0 + TE1
per guide (N = 4 forward) so the coupler's higher-order excitation is seen;
`force_unitary=False`, `INTERFACE_PROJECTION = "output"`.

Path: straight bus, ring arc approximated in the straight frame,
`gap(z) = g0 + R − sqrt(R² − z²)` for `|z| ≤ z_max` with `z_max` set where
the coupling has fallen 40 dB below its peak (check, do not assume 2 µm).
State the limitation: the ring mode's outward shift under curvature is
neglected. Estimate its size from `Si_fulletch_220nm` (mode centroid shift at
κ = 1/R) and report it; if > 20 nm at R = 5 µm, restrict claims to R ≥ 10 µm
until the pulley (curved-pair) dataset exists.

Output: `t(λ), κ_c(λ)` per gap, plus the TE1 leakage. DBEME vs direct EME per
§7 protocol A on one gap, cold and warm timings, since the gap sweep is the
speed-up claim.

### 2.1 — findings while building (2026-09-20)

* **The bus must not move.** `CoupledStrips` centres the pair on the
  mid-gap, so a gap step would translate the bus by half a step and scatter
  the bus mode at every interface. `platforms.BusRingStrips` keeps the bus
  at (−1.0, −0.5) µm and moves only the ring (`ring_coupler_dataset_info`).
  A two-axis `(gap, x_offset)` route was rejected: `interp_multi_adj_pts`
  inserts an intermediate point for a diagonal move, which brings the bus
  shift back.
* **The gap axis is tied to the cell.** With 20 nm cells and 10 nm gap steps
  the ring's edges sat mid-cell on alternate points and its index aliased by
  1e-3 — larger than the coupling splitting — so the tracked branches
  zig-zagged. 10 nm cells with 10 nm steps: smooth, monotone even/odd
  branches, splitting 2e-5 at 1 µm to 4e-3 at 400 nm. Cost ~35 s per point
  (464 × 160 cells, N = 6).
* **A moving guide needs a large basis.** With N = 6, every interface lost
  1.4 % of the *ring-side* power and put 7 % amplitude into the weakly
  guided TE1 pair, for a 50 nm ring step: the displaced field is not in
  the basis, so it is discarded (interface projection "output") and the
  loss grows *linearly* with the step count — ring-launched power 0.57
  after 24 interfaces while bus-launched power was 0.9999. It is not the
  §5.3 Δ² floor: that needs the scattered field to stay in the basis and
  cancel between steps, which is the PML-EME mode-count argument (§5.9
  item 5, 20–40 modes). Hence `studies/ring/mode_convergence.sh`: N = 6,
  12, 20, 30 on one short path, judged on ring-launch power conservation
  and `κ²`. The gap axis was shortened to 0.7 µm (splitting 1.5e-4 there,
  residual coupling ~1e-6) with 20 nm coarse steps, so the ring walks
  fewer, smaller steps.
* **Mode-count convergence, measured (g0 = 400 nm, R = 10 µm, 16 points,
  20 nm coarse steps):**

  | N | bus `‖t‖²` | bus `κ²` | ring-side through | s/point |
  |---|---|---|---|---|
  | 6 | 0.99958 | 0.00038 | 0.781 | 26 |
  | 12 | 0.99958 | 0.00039 | 0.781 | 60 |
  | 20 | 0.99960 | 0.00039 | 0.785 | 104 |
  | 30 | 11247 | 4622 | — | 180 (transfer route breaks down: singular interfaces) |

  The basis size does **not** recover the ring side. Single-step paths on
  the cached dataset (400→390→400 etc.) show the per-interface loss equals
  the ring mode's own overlap with itself shifted by the step: mismatch
  2.0e-3 at 10 nm, 7.7e-3 at 20 nm, 2.9e-2 at 40 nm, 1.1e-1 at 80 nm
  (nearly ∝ step, not step²: the strip mode is far sharper than a
  Gaussian). That residual is a localised, high-spatial-frequency field a
  dozen box modes cannot hold, so no affordable N cancels it, and the
  loss adds up linearly along the path. **Decision:** N = 6; the bus-side
  `t`, `κ` (converged, power-conserving) are used; the ring-side through
  amplitude is imposed by the symmetry of identical guides,
  `S(o3,o4) = S(o1,o2)`; and `κ²` is corrected by the square root of the
  ring-side survival because light coupled at the tangent point rides the
  ring through half the interfaces: `κ²_corr = κ² / √(‖t'‖²)` — 0.00044,
  against CMT's 0.00046 from the same `n_eff` (4 %; uncorrected 0.00039,
  17 %). The straight-frame moving-guide EME is therefore honest for the
  bus side only; the ring side is the price of the frame, stated in the
  report. A frame with the ring fixed (conformal, ring straight) would
  move the bus instead — same price, other guide.
* The near-degenerate TE0 pair at large gaps comes back as a *real*
  rotation of even/odd (bus fractions 0.36/0.64 at 950 nm), which is
  harmless: fields are real to 1e-16, unconjugated and conjugated powers
  agree to 1e-5. §5.6's complex-mixture failure did not occur here.

### 2.2 Ring loss

* **Radiation**: `PMLModeSolver` on `FullEtchStrip`, w = 500 nm, R ∈ {3, 5,
  8, 10, 15} µm, λ ∈ band; `α = 2 k0 Im(n_eff)`; PML convergence sweep as
  report 06 §Phase 3 (thickness, stretch, standoff, < 5 %). Expect negligible
  above 5 µm; report it anyway.
* **Roughness**: `em_simulation/circuit/roughness.py` implementing Payne &
  Lacey, *Opt. Quantum Electron.* 26, 977 (1994) with the sidewall field
  factor from the mode (`∫|E|² along the two sidewall lines / ∫|E|² dA`).
  Store that factor per grid point as `sidewall_factor.pkl` beside
  `TE_pol.pkl` (gauge-invariant ratio of quadratics; same plumbing; goes into
  `dataset_identity` so old caches are not silently missing it). Inputs
  `σ = 2 nm, L_c = 50 nm` as the default, both **reported as assumptions**.
  Cross-check the straight-guide number against a literature 220 nm strip
  loss (2–3 dB/cm at 500 nm width) to within a factor of 2 — this model is
  order-of-magnitude, say so.
* Junction mismatch at the coupler ends is already inside the coupler S.

Deliverable: a loss budget table (radiation / roughness / total) per R and λ,
and `Q_intrinsic(R)`.

### Phase 2 gate

| check | criterion |
|---|---|
| §5.1 reciprocity on the coupler | `max|S − Sᵀ| < 1e-6` |
| §5.2 basis convergence, N = 4, 6, 8 | residual `1 − ΣT` falls with N |
| §5.4 slicing independence on `gap(z)` | `max|ΔT| < 1e-4` for Δz halved |
| gap-axis convergence | `κ_c` at 10 nm vs 5 nm gap step < 1 % |
| DBEME vs direct EME | `max|ΔT| < 1e-3`, timings recorded |
| PML convergence | loss holds to 5 % under the three perturbations |

### Phase 2 — done 2026-09-20 (measured; full tables in `reports/15` §1)

Datasets `Si_ring_coupler_220nm_{1530,1550,1570}` (N = 6, 10 nm cells) plus
`_N6/_N12/_N20/_N30` and `_c20`; `reports/output/ring/coupler_*.json`,
`ring_loss.json`. Gate: reciprocity (TE0 pair) ≤ 2.5e-6 — marginal against
1e-6; N convergence ✓ on the bus side, ✗ on the ring side (frame, §2.1
notes); slicing Δκ² 2.9e-6 ✓; **gap axis / cell 10 vs 20 nm: +14 % in κ²,
and CMT on each dataset's own `n_eff` shows the same +14 %, so it is the
mode solve at 20 nm cells, not the step** — ✗ against 1 %, quoted as the
coupling's uncertainty; DBEME vs direct EME Δκ² 6e-5 ✓ (531 s vs 0.006 s);
PML convergence 1 % ✓. Loss budget: radiation 0.58 / 7e-4 / 1e-9 dB/cm at
R = 2 / 3 / 5 µm; roughness 4.4 dB/cm (EIM form; 7.8 with the mode index);
`n_g` 4.178. The R = 5 µm rings cost 0 new solves after the R = 10 µm gaps.

### Phase 3 — done 2026-09-20

`studies/ring/ring_model.py` builds the 4-port sax coupler per gap from the
three wavelengths (phase reference: the straight guide's optical length over
the chord), the arc from `ring_loss.json`, and the all-pass netlist;
`examples/demo_ring_resonator.py` draws it. **Coupler construction rule
(after two failures):** `t` from DBEME, `κ = i √(1 − ‖t‖²) t/‖t‖`,
`S(ring→ring) = t` — the DBEME cross-term *phase* was 8–13° off quadrature
and the ring transmitted 1.66 before this. sax vs closed form on the DBEME
coupler: Q within 1–5 % at every gap (the residual is `‖t(λ)‖` moving
between 1550 nm and the dip). Results: FSR 9.151 nm, Q_i 1.66e5, critical
gap 247 nm, 27 dB extinction at 250 nm, 0.57 nm resonance shift per nm of
width. Report written: `reports/15_ring_resonator.md`.

### Phase 4 — internal rows done, Lumerical and literature rows left blank for Jae

CMT ✓ (5 % at R = 10 µm; 14 % at R = 5 µm / 200 nm), direct EME ✓, slicing ✓,
gradient check ✓ (`gradient_check.json`, ≤ 3e-4 on κ²). The Lumerical table
(`reports/15` §5.2) and the literature table (§5.3) carry the DBEME column
and empty reference columns; scripts for Lumerical belong under
`studies/ring/lumerical/`.

## Phase 3 — the ring in sax, cross-checked against the closed form

Netlist: `coupler (2.1) → arc (1.2, with 2.2 loss) → [second coupler] → arc`.
sax composes the loop. Requirement: sax spectrum equals `ring.py` closed form
to 1e-6 in `|T|` and 1e-6 rad in phase across a full FSR. That agreement is
the proof that the adapter, the port naming and the phase sign are right;
without it no spectrum is reported.

Then `examples/demo_ring_resonator.py`: through/drop vs λ, FSR, loaded and
intrinsic Q, extinction vs gap with the critical-coupling gap marked,
resonance shift vs width (sensitivity, not absolute position), timings.
The report (`reports/15_ring_resonator.md`, template CLAUDE.md §8) is
written **after the Phase 4 gate**, because §7 protocol C forbids a report
without its literature table.

**What the report may claim.** FSR, extinction vs gap, Q_loaded vs coupling,
critical-coupling gap, resonance *shifts*: comparable to literature.
**Absolute resonance wavelength: no.** `n_eff` is converged to ~2e-3 (§5.7),
i.e. `Δλ ≈ λ·Δn/n_g ≈ 0.7 nm` at 1550, a large fraction of a 10 nm FSR.
State it. Intrinsic Q: radiation part physical, roughness part model-bound.

## Phase 4 — is the ring right? Independent references

Three tiers, from exact to empirical. Every row of the final table carries
its discrepancy **and a physical cause** (CLAUDE.md §7 protocol B). The
question each tier answers is different; do not let a pass in one stand in
for another.

### 4.1 Internal references (same physics, different route)

* **Closed-form identities** — Phase 1 gate. Tests the algebra, nothing else.
* **Coupler by coupled-mode theory from the same dataset.** For identical
  guides the point coupler's cross-coupling is
  `κ_c² = sin²( ∫ π·Δn(z)/λ dz )`, `Δn = n_even − n_odd` read from
  `neff.pkl` at `gap(z)`. No cascade, no overlaps: it tests the EME cascade
  against the mode solver's own supermode splitting. Criterion: agree to
  5 % in `κ_c²` for `κ_c² < 0.3`. Divergence above that is expected (CMT is
  weak-coupling); explain it, do not tune it away.
* **Direct EME** — §7 protocol A on one gap (Phase 2 gate).

### 4.2 Lumerical, same geometry (installed: v231 at
`C:/Program Files/Lumerical/v231`; FDTD, MODE `fd-engine`, `eme-engine`,
INTERCONNECT `qinterconnect`; API `api/python/lumapi.py`)

Scripts under `studies/ring/lumerical/`, run through `lumapi`
(`sys.path.append(r"C:/Program Files/Lumerical/v231/api/python")`); check
the licence is reachable before planning a batch. Save only the extracted
numbers (JSON) and the script, not `.fsp`/`.lms` project files. Record
provenance as `studies/tapeout/provenance.py` does.

| ref | what it checks | criterion |
|---|---|---|
| **MODE FDE**, 500 × 220 strip, straight and bent (MODE's bent-waveguide solver) at R ∈ {5, 10} µm | `n_eff`, `n_g`, bend `n_eff` shift, bend loss vs `PMLModeSolver` | `Δn_eff < 2e-3`, `Δn_g < 1 %`, bend loss within a factor 2 where > 0.01 dB/90° |
| **FDTD 3-D**, bus + ring arc, R = 10 µm, gap ∈ {150, 200, 250, 300} nm, 1500–1600 nm | `κ_c²(λ)`, `t²(λ)`, TE1 leakage. The **main coupler check**. FDTD keeps the ring mode's outward shift that 2.1 neglects, so the residual here *is* that approximation | `|Δκ_c²| < 0.01` absolute or 5 % relative, whichever is larger |
| **Lumerical EME** of the same coupler | third EME, independent mode solver; separates cascade error from solver error | `|Δκ_c²| < 0.01` |
| **INTERCONNECT** ring from the DBEME-exported S (VPI `.dat` writer exists; add a Lumerical `.dat` writer if INTERCONNECT will not read it) | sax composition vs a commercial circuit solver; also its transient mode is a second time-domain reference for Phase 5 | `|T|` to 1e-4, resonance positions to 1 pm |
| **FDTD 3-D full ring**, all-pass, R = 5 µm, one gap near critical coupling, 1540–1560 nm (≈ 12 × 12 × 1.5 µm³, ~2–4 h at 20 nm mesh; run it once, detached, `run_solver_job.py` rule) | the **end-to-end** reference: FSR, Q_loaded, extinction, resonance position, with the same lossless-material, smooth-sidewall physics as the model. Apples to apples | FSR 1 %, Q_loaded 10 %, extinction 1 dB, resonance position 0.5 nm (mesh-limited on both sides — state the FDTD mesh convergence too) |

The full-ring FDTD row is the one that validates the *assembled* result. A
failure there with 4.1 and the coupler rows passing points at the arc model
(1.2) or the phase fit (1.1), not at the coupler.

### 4.3 Literature (fabricated devices: roughness and CD bias present)

Nothing in `references/` covers rings; obtain these (verified 2026-09-20).

* **Bogaerts et al., "Silicon microring resonators", Laser Photon. Rev.
  6(1), 47–73 (2012), doi:10.1002/lpor.201100017.** The formula reference
  (§1.3 above is taken from it) and typical 220 nm numbers (`n_g`, Q and
  loss relations, measured all-pass and add-drop spectra). Use for:
  closed-form equations, FSR sanity, Q ↔ loss conversions.
* **A. Yariv, Electron. Lett. 36(4), 321–322 (2000),
  doi:10.1049/el:20000340.** The universal all-pass relation and critical
  coupling; the amplitude form in §1.3 derives from it.
* **Chrostowski & Hochberg, *Silicon Photonics Design*, CUP 2015,
  ISBN 978-1-107-08545-9, ch. 4** (table of contents confirms: directional
  couplers with coupler-gap dependence and FDTD modelling, ring resonators).
  Closest platform match: 220 nm SOI, 500 nm TE, oxide clad, 1550.
  **Digitise the coupling-vs-gap curve** and overlay DBEME and the 4.2 FDTD
  on it. Confirm the figure number, radius and mesh in the copy used before
  quoting; the companion measurement data is the SiEPIC/edX course
  (github.com/SiEPIC).
* **Dumon et al., IEEE Photon. Technol. Lett. 16, 1328–1330 (2004),
  ieeexplore.ieee.org/document/1291500.** 220 nm SOI, 248 nm DUV, wires up to
  600 nm wide, 1 µm BOX; 2.4 dB/cm propagation loss; ring and racetrack
  resonators with Q > 3000 (a racetrack at Q = 3086). Take the radius,
  width and gap from the paper for each device compared; do not assume
  500 nm. Compare FSR (n_g check) and the *coupling-limited* Q; its 1 µm
  BOX also leaks to the substrate, which our 220 nm strip model does not
  include — say so in the cause column.
* **Circuit-level cross-checks already in `.venv`:** simphony 0.7.3 has an
  add-drop filter tutorial
  (simphonyphotonics.readthedocs.io/en/latest/tutorials/filters.html); sax's
  docs (flaport.github.io/sax) show composite ring circuits. Either is a
  second, independent frequency-domain composition of the same models,
  cheaper than INTERCONNECT; agreement to 1e-6 is expected, not 1e-4.

**How to compare Q.** Compare the **coupling-limited** Q, over-coupled
regime: `Q_c = 2π n_g L / (λ κ_c²)` depends only on what DBEME computes and
is where the literature Q is set by the gap, not by fab loss. Intrinsic Q is
compared as a *range* with σ and L_c stated. The critical-coupling gap
depends on both, so report it as a function of the assumed loss, never as a
single number.

Discrepancy table rows and their usual causes: `n_eff` mesh (0.7 nm in
position), missing roughness (Q_int, extinction), width and CD bias (450 vs
500 nm, ±10 nm fab), cladding (air vs oxide in some papers), the neglected
ring-mode shift in the coupler (2.1), and the paper's own fit method.

### Phase 4 gate

| check | criterion |
|---|---|
| 4.1 CMT vs DBEME coupler | 5 % for `κ_c² < 0.3` |
| 4.2 MODE `n_eff`, `n_g` | 2e-3, 1 % |
| 4.2 FDTD coupler `κ_c²` vs gap | 0.01 abs / 5 % rel |
| 4.2 INTERCONNECT vs sax | `|T|` 1e-4, 1 pm |
| 4.2 FDTD full ring | FSR 1 %, Q 10 %, ER 1 dB, position 0.5 nm |
| 4.3 Chrostowski coupling vs gap overlay | discrepancy per gap with cause |
| 4.3 Dumon FSR and coupling-limited Q at the paper's geometry | FSR 2 %; Q within a factor 2 with the loss and BOX-leak causes attributed |
| simphony or sax-docs ring vs `ring.py` | 1e-6 |

A failure in the FDTD coupler or full-ring row blocks the report.

## Phase 5 — circulax: time domain on the same models

1. Install circulax into `.venv` pinned to a tag (if it drags an incompatible
   jax, use a separate `.venv-circuit` like `.venv-gds`; record it in
   `install.sh`). Read its docs; the README is the only source so far.
   Confirm: how a sax model is wrapped, port/mode naming, the optical
   carrier convention (baseband envelope vs real field), and how wavelength
   enters a transient run.
2. Transient ring build-up and ring-down with the Phase 3 models. Gate:
   ring-down time `τ = Q_loaded·λ/(2πc)` from the transient must match Phase
   3's `Q_loaded` to 2 %. That closes the loop between the two domains.
3. Differentiability check: `d(extinction)/d(gap)` via `jax.grad` through the
   gap-interpolated coupler model vs finite difference on the dataset grid.
   Note the boundary: gradients exist over the cached grid (interpolant),
   not through the mode solver. A new geometry axis value still costs solves,
   though neighbours already in the cache make finite differences cheap.
4. Record in `README.md` (*Circuit layer*) what the layer is for beyond the
   ring: modulator-loaded rings, thermal tuning, laser + modulator + ring
   link budgets, and gradient-based design of a CPO channel. Those are
   follow-on tasks, not this one.

### Phase 5 — status 2026-09-20 (measured)

* circulax 0.2.3 installed in `.venv-circuit` (jax 0.9.2, sax 0.18.2; its
  pins `jax < 0.10`, `sax ≥ 0.15.15` exclude the main venv's jax 0.11 /
  sax 0.14.7). `studies/ring/circulax_ringdown.py`.
* **Integration works.** The Phase 1 sax models compile unchanged through
  `circulax.sax_component`; with an `OpticalSource` at the input and a
  `Resistor(R = 1)` (z0 = 1, matched) at the output, the steady-state
  solve batched over wavelength reproduces the sax spectrum of the
  ideal-coupler ring to **8e-10 in |T| and 1.3e-9 rad** over 2201
  wavelengths across an FSR. Two conventions to know: testbench parameters
  go under `"settings"`, and a complex circuit's solution comes back
  real-unrolled `[re | im]` of length `2·sys_size`.
* **The ring-down gate (5.2) — met with a plugin, no fork (2026-09-20,
  later the same day).** An S-matrix component is memoryless, so a ring of
  them has no round-trip time; the earlier transient failure
  (`AttributeError ... 'subs'`) was *mine* — `wl` passed through the
  diffrax kwargs instead of `params=`. `em_simulation/circuit/circulax_ext.py`
  adds `OpticalDelayLine` on circulax's own `@component` API: the arc as a
  chain of N first-order sections with the exact mean delay `n_g L/c` and
  the carrier's `a e^{iφ}`, matched wave stamps at both ports, and a DC
  solution equal to the sax arc; plus `OpticalGate`, a CW source with
  on/off. Measured (`tests_circuit/test_delay_line.py`, `.venv-circuit`
  only): mean delay exact to 1 % at N = 8 and 32; delay-line ring steady
  state = sax ring to 1e-6; **ring-down τ = 18.749 / 18.736 ps (N = 8 / 16)
  vs Q λ/(2πc) = 18.721 ps — 0.15 % / 0.08 %**; build-up with 2τ_E to 5 %.
  Why so few sections suffice: what the ring-down sees is the chain's phase
  at the linewidth, `(τ/τ_E)²/N`, not its step rise. Decision on forking
  circulax (plan of 2026-09-20): plugin against the pinned release; fork
  only for a change inside circulax, as a GitHub fork with an editable
  install and an upstream PR, never a vendored copy.
* 5.3 gradient check: `studies/ring/gradient_check.py` (main venv, jax
  through the gap-interpolated coupler), run once the coupler campaign has
  produced the gap table.
* 5.4 README section: pending with the report.

## Out of scope, named so nobody drifts into it

Pulley / curved-pair coupler (needs `curvature` on `CoupledStrips`; separate
task). A lossy DBEME dataset (report 06 §8). Nonlinear or thermal models in
circulax beyond the ring-down check. Absolute resonance wavelength claims.

## Verification summary

```
.venv/Scripts/python -m pytest tests -q          # plus the two new test files
python examples/demo_ring_resonator.py           # figures -> reports/output/
python examples/validate_against_direct_eme.py   # coupler, one gap
```

Long dataset builds through `examples/run_solver_job.py` (E-core rule).
