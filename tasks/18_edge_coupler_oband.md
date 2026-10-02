# Task 18 — O-band bilayer double-tip edge coupler (Wan & Wang 2025)

**Prompt to start with:**

> Read `CLAUDE.md`, `docs/validation_backlog.md` and this file, then do
> Phase 0 and Phase 1 only. Stop at the Phase 1 gate and report its table.
> The report is `reports/22_edge_coupler_oband.md`.

**Reference.** Y. Wan, J. Wang, *O-band low loss and polarization
insensitivity bilayer and double-tip edge coupler*, **Adv. Photonics Nexus
4(2), 026004 (2025)** — https://m.researching.cn/articles/OJ1ad18a083e231c67/
(DOI to confirm from the PDF; expected 10.1117/1.APN.4.2.026004).

---

## Why

This is the first **fibre-to-chip** device in the repo, and the first one
whose figure of merit is an *absolute* loss in dB. Every earlier demo scored
ratios between guided modes. Here the headline number is how much of a
Gaussian beam ends up in the output waveguide. That needs three things DBEME
does not have yet:

1. **A fibre launch.** A Gaussian beam projected onto the facet's modal
   basis, used as the input vector for the S-matrix. Nothing in
   `dbeme/propagator` does this today.
2. **A cross section with two mirrored arms, each at two heights**
   (150 nm and 220 nm in the same 220 nm layer, the 220 nm part on one edge).
   `BiLevelStrip` is one core with a centred slab. `CoupledStrips` is two
   cores at one height.
3. **The real stack, with loss.** At the tip the mode is ~4 µm across and
   sits over a 2 µm BOX on a Si substrate. Substrate leakage and unguided
   fibre light are part of the answer, which means the PML backend
   (`dbeme/fde/pml.py`) or FEM (`dbeme/fde/femwell_fde.py`), not the lossless
   `EmepyFDE`.

What DBEME buys here is the method's usual argument. The paper tuned `L1`,
`L2`, `L3` and `Lt` with 3D-FDTD. Once the path's cross sections are cached,
every length sweep costs zero mode solves.

---

## The device, from the paper

**Stack.** SOI with 220 nm Si and a 2 µm BOX on a Si substrate, and 2 µm of
PECVD SiO₂ on top (air above that). Two etches: 220 nm through (the outline)
and 70 nm partial (leaves 150 nm Si). Sidewalls are not stated; take them as
vertical. Material indices are not stated; use `platforms._materials` (Si from
Li 1980, SiO₂ from Malitson, `out_of_range="raise"`).

**Layout, fibre side first** (`z = 0` at the facet; Fig. 1(b)):

| section | length | cross section | what changes |
|---|---|---|---|
| L1, double tip | 25 µm | two 150 nm strips | width from `Wtip` (0.13 µm) at the facet (see Q1) |
| L2, height converter | 20 µm | two L-shaped arms: 150 nm base plus a 220 nm part on the **inner** edge (Fig. 1(c)) | the 220 nm part widens from 0 to the full arm width |
| L3 | 23 µm | two 220 nm strips | width linear to `W0` = 0.45 µm, gap linear from `g` to `d` = 0.415 µm |
| Lt, MMI access | 8 µm | two 220 nm strips merging | arms widen, gap closes to 0 at the MMI face (see Q3) |
| Lm, wedge MMI | 2.5 µm | one 220 nm strip | width `Wm2` = 1.6 µm (arm side) to `Wm1` = 1.5 µm (output side) |
| output taper | 8 µm (Q3) | one 220 nm strip | `W1` = 0.735 µm to `W0` = 0.45 µm |

Table 1, verbatim (µm): `W0` 0.45, `W1` 0.735, `Wm1` 1.5, `Wm2` 1.6,
`Wtip` 0.13, `d` 0.415, `g` 1.6, `Lm` 2.5, `Lt` 8, `L1` 25, `L2` 20, `L3` 23.
The stated total is 86.5 µm. The listed lengths add up to 78.5 µm.

**Fibre.** Lensed fibre with a 4 µm MFD, i.e. a Gaussian with
`w0 = 2 µm`. The polarisation is TE (E along x) or TM (E along y).

**Paper's overlap**, eq. (1): `|∫E1 E2 dA|² / (∫|E1|² dA ∫|E2|² dA)`, with
`E1` the fibre field and `E2` the waveguide field. This is scalar and has no
Poynting weighting.

### Numbers to hit

All values below were read by eye, ±0.02 / ±0.1 dB. Digitise them properly in
Phase 1.

| source | quantity | 150 nm (this design) | 220 nm (conventional) |
|---|---|---|---|
| Fig. 3(a), `Wtip` = 130 nm | overlap TE / TM | 0.855 / 0.86 | 0.83 / 0.43 |
| Fig. 3(b), `g` = 1.6 µm | overlap TE / TM | 0.855 / 0.855 | 0.80 / 0.44 |
| Fig. 5, 3D-FDTD, 1260–1340 nm | loss TE | 0.75–1.05 dB, ~12 nm ripple (text: < 1.09 dB) | 0.9–1.6 dB |
| | loss TM | 1.05–1.17 dB, flat (text: < 1.24 dB) | 4.8 → 3.0 dB, monotonic |
| Fig. 8 / text, measured | loss TE / TM at 1310 nm | 1.18 / 1.46 dB/facet | — |
| | over 1260–1360 nm | TE < 1.52, TM < 2 dB/facet; PDL 0.48 dB (Table 2) | — |

Fig. 3 sweeps `Wtip` over 100–195 nm and `g` over 0.5–1.9 µm. Its y-axis says
"%" but the values plotted are fractions.

**Paper inconsistency.** The text says red is TE and blue is TM. The Fig. 3
and Fig. 5 legends say blue is TE and red is TM. The legends fit the physics
(the 220 nm TM curve is the poor one), so use the legends.

The paper blames the TM penalty on the L2 height transition, where "the
optical field dissipates into the cladding" (Fig. 4(b)). Phase 3 should either
reproduce that or show it is wrong.

---

## Geometry decisions (Jae, 2026-09-29)

The paper leaves these open. They are this project's readings, confirmed by
Jae, and the report must state them as such.

* **Q1. Arm width through L1–L3.** Table 1 gives the width only at the facet
  (`Wtip`) and at the end of L3 (`W0`). It runs as one straight edge from
  0.13 to 0.45 µm over L1+L2+L3 (68 µm), which gives 0.248 µm at the L1/L2
  boundary and 0.342 µm at L2/L3.
* **Q2. `g`.** It is the edge-to-edge gap, the same convention as `d` and as
  `CoupledStrips.gap`. It is held constant through L1 and L2 and only changes
  in L3.
* **Q3. Output side.** Over `Lt`, each arm widens from 0.45 to 0.80 µm
  (= `Wm2`/2) while the gap closes from 0.415 to 0, so the V-notch ends at the
  MMI face. The output taper `W1 → W0` is 8 µm long (86.5 − 78.5).
* **Q4. Stack and loss.** The full stack (Si substrate under the 2 µm BOX,
  2 µm top oxide, air above) on a lossy basis. No lossless pre-pass.

The resulting path, `z` from the facet, in µm:

| z | `w` (per arm) | `w_hi` | `gap` |
|---|---|---|---|
| 0 → 25 (L1) | 0.130 → 0.248 | 0 | 1.6 |
| 25 → 45 (L2) | 0.248 → 0.342 | 0 → `w` | 1.6 |
| 45 → 68 (L3) | 0.342 → 0.450 | `w` | 1.6 → 0.415 |
| 68 → 76 (`Lt`) | 0.450 → 0.800 | `w` | 0.415 → 0 |
| 76 → 78.5 (`Lm`) | 0.800 → 0.750 | `w` | 0 |
| 78.5 → 86.5 (output taper) | 0.3675 → 0.225 | `w` | 0 |

`w` jumps from 0.75 to 0.3675 at z = 78.5, the MMI's output face (1.5 µm to
0.735 µm).

---

## Phase 0 — cross section, platform, backend choice

**0.1 `BiLevelPair(CrossSection)`** in `dbeme/fde/cross_section.py`. Two arms
mirrored about `x = 0`. Each arm has a 150 nm base of width `w`, plus the top
70 nm over a width `w_hi ≤ w` aligned to the arm's **inner** edge. `gap` is
edge-to-edge between the inner edges.

```python
parameter_names = ("w", "w_hi", "gap")
```

The whole path is one family:

* L1: `w_hi = 0`
* L2: `0 < w_hi < w`
* L3 and `Lt`: `w_hi = w`
* MMI and output: `gap = 0`, and `w` is half the strip width

Keeping one family means one grid and one window, so every junction on the
path is an ordinary neighbour overlap. There is no cross-dataset interface.
Implement both `index()` (sub-pixel fill, for the FD backends) and
`polygons()` / `background_epsilon()` (for FEM).

Gap → 0 is a physical merge here, not the silent failure `tasks/01` §1.1
guards against. Allow `gap = 0` exactly. Near 0, sample the gap axis finely
(the notch tip) and tie the axis to the cell (`tasks/11` §2.1).

**0.2 Platform factory** `wan2025_edge_coupler_dataset_info(wavelength, ...)`
in `dbeme/platforms.py`. Datasets `datasets/Si_bilevel_pair_220_150nm_<λ>`.
The window must hold the fibre beam: 1e-3 of a `w0 = 2 µm` Gaussian's power
lies beyond r = 3.7 µm. So the window needs ≳ ±4.5 µm lateral plus PML, the
substrate in the bottom, and air in the top. Refine the grid over the cores
(≤ 10 nm; the 70 nm step needs ≥ 7 cells vertically) and coarsen it in the
cladding (`PMLModeSolver(refine_x=, refine_y=)`, or the FEM mesh).

**0.3 Backend choice, measured.** At the facet cross section, for both
polarisations, measure against cell size, window margin and PML: the even-TE
and even-TM supermode `n_eff` (real and imaginary parts), and the eq. (1)
overlap with the fibre. Pick FD-PML or FEM on cost at a converged
`Im(n_eff)`, and record per-solve times (cold). If a solve takes more than
~60 s, runs go through `examples/run_solver_job.py`.

**0.4 Tests** (`tests/test_bilevel_pair.py`):

* `w_hi = w` reproduces `CoupledStrips` at 220 nm.
* `w_hi = 0` reproduces a 150 nm `CoupledStrips`.
* `gap = 0`, `w_hi = w` reproduces a `FullEtchStrip` of width `2w`.
* The mirror symmetry of `index()` holds.
* `polygons()` agrees with `index()` in painted area.

## Phase 1 — facet overlaps (FDE only, no propagation)

Reproduce Fig. 3(a) and 3(b): the overlap against `Wtip` and against `g`, at
150 nm and 220 nm, TE and TM, at 1310 nm. Compute it two ways:

* **(i)** the paper's eq. (1) against the dominant transverse E component of
  the even supermode;
* **(ii)** the Poynting-form power coupling into the even supermode
  (unconjugated/biorthogonal on a lossy basis, `docs/validation_backlog.md`
  §5.16).

The two differ, and the difference is part of the result.

Put the fibre launch here as a function, not in a script:
`dbeme/propagator/fiber.py::gaussian_launch(mode_data, w0, center, pol, n_medium)`
returns the modal coefficient vector.

### Phase 1 gate

| # | check | criterion |
|---|---|---|
| F1 | Fig. 3 overlay, 150 nm | (i) within ±0.05 of the digitised curves; TE/TM gap ≤ 5 points across 100–150 nm, as the paper claims |
| F2 | Fig. 3 overlay, 220 nm | TM ≥ 30 points below TE at `Wtip` = 130 nm |
| F3 | parity | centred launch: power into x-odd supermodes < 1e-4 |
| F4 | launch completeness | Σ\|c_m\|² over the basis vs 1 − (Gaussian power outside the window); report what the guided modes hold and what the PML/box modes hold |
| F5 | window/PML convergence (§5.7) | eq. (1) overlap and `Im(n_eff)` of the tip mode move < 1e-3 / < 10 % |

## Phase 2 — sections alone

Sweep each section's length on warm points, both polarisations, 1310 nm:

* L1 alone: loss from the tip supermode to the L1/L2 cross section.
* L2 alone: this is where the paper puts the TM loss. Find where adiabaticity
  knees in. Is 20 µm past it for both polarisations?
* L3 plus `Lt` plus MMI plus output taper: the even supermode into output
  TE0/TM0. Near-equal for both polarisations, as the paper claims?

Sanity-gate rows as in `docs/demo_plan.md` §8:

* §5.1 reciprocity (lossy criterion);
* §5.4 slicing;
* §5.8 DBEME vs direct EME on one section;
* §5.13a basis membership along the path (the launched branch present at
  every point; the Kocabaş −20 dB lesson).

## Phase 3 — the device, and the 220 nm reference

The full 86.5 µm path, with the Gaussian launch and output TE0/TM0 power.
Report the loss in dB/facet over 1260–1360 nm, using one dataset per
wavelength at 1260/1280/1300/1310/1320/1340/1360 (the `Si_bilevel_220_90nm_<λ>`
pattern, `reports/03`). Run 1260/1310/1360 first.

The conventional structure is the same path with `w_hi = w` throughout: no
L2, and 220 nm tips. It lives in the same dataset family and needs no new
code. Overlay both on Fig. 5 and on the measured Fig. 8.

Then the method's demonstration, all warm: loss against `L1`, `L2` and `L3`
at zero mode solves, with cold and warm timings (`CLAUDE.md` §1).

Also look at the Fig. 5 TE ripple (~12 nm period at 1310 nm). If DBEME shows a
ripple, find which pair of modes beats: `Δn_g · L = λ²/Δλ ≈ 137 µm`.

### Phase 3 gate

| check | criterion |
|---|---|
| 150 nm design at 1310 nm vs Fig. 5 | TE and TM each within 0.5 dB |
| PDL over 1260–1360 nm | < 0.5 dB, as the paper reports |
| 220 nm reference | TM penalty ≥ 1.5 dB over TE, falling with λ |
| §5.8 DBEME vs direct EME, full device, 1310 nm | `max\|ΔT\| < 1e-3` |

## Phase 4 — optimisation (Jae, 2026-09-29)

Last step, after Phase 3. Start from the paper's design point and objective:

* **Objective:** worst-case loss over TE and TM, averaged over the O band,
  plus a PDL penalty. Report it at 1310 nm alone and over the band.

**4.1 Lengths only: L-BFGS-B on the free axes.** `L1`, `L2`, `L3`, `Lt` and
`Lm` do not move any cross section. On a warm cache every evaluation is
`change_strucutre_length` on cached S-matrices, with zero mode solves. The
gradient comes from finite differences, which cost nothing here, or from
the analytic `d|S|²/dL` of the phase terms. Bounds: each length within
[0.25, 3] × its Table 1 value.

**4.2 Geometry: CMA-ES** over `Wtip`, `g` and the width at the L1/L2 and
L2/L3 boundaries (which frees Q1), with the lengths re-optimised by 4.1
inside each evaluation. Every new geometry solves new cross sections, so:

* snap candidates to the dataset axes;
* cap the population;
* log the cold solves per generation.

The cache means neighbouring candidates share most of their points. Stop on
budget: say how many generations, and at what solve cost.

**Deliverables:**

* the optimised design (a table next to Table 1);
* the loss spectra before and after, for both polarisations;
* convergence histories;
* the cost of each evaluation, warm and cold.

State plainly what the optimum trades away (e.g. tolerance to `Wtip`).

## Visual comparison with the paper (Jae, 2026-09-29)

Every paper figure gets a DBEME counterpart, overlaid on the digitised paper
curves (`reports/output/edge/paper_digitised.json`) where there is data:

* Fig. 1(b), layout: the path as drawn from the dataset's snapped points (top
  view and `w`, `w_hi`, `gap` against `z`).
* Fig. 2, mode fields: TE and TM at the facet, 150 nm and 220 nm.
* Fig. 3(a)(b), overlaps: DBEME eq. (1) and power coupling on the paper's
  curves.
* Fig. 4, field propagation: top-view |E| along the device for TE and TM, with
  cross-section insets at the paper's four planes.
* Fig. 5, simulated loss: DBEME on the paper's 3D-FDTD, for 150 nm and 220 nm.
* Fig. 8, measured loss: DBEME on the measured band (median and envelope).
* Beyond the paper: `n_eff` along the path, loss per section, and the Phase 4
  before/after.

## Status 2026-09-30 — done; `reports/22_edge_coupler_oband.md`

All phases run at 1260 / 1310 / 1360 nm. The other four wavelengths were
not built (compute). The numbers are after four corrections to the cascade:
the three this task led to (see *Found on the way*) and the reciprocal
interface projection that followed. The report was regenerated on the 1.0.0
solver on 2026-10-02.

- **Phase 0.** Done.
  - Built `BiLevelPair` with axes `(w_low, w_high, gap)`, the fibre launch and
    the platform.
  - The Si substrate cannot sit in the path's EME window: its PML-guided band
    fills any mode set. So the path datasets use the oxide-only stack.
  - The oxide stack is not converged at the facet (F5 fails there). The
    launch and the tip up to the 200 nm join are therefore solved on the full
    stack, on an identical grid, and joined by a cross-stack interface
    (T ≥ 0.996).
  - Also added: a point-by-point shift-invert target
    (`TargetRulePMLBackend`), per-edge PML depth, and an MMD ordering that
    makes each solve 1.5× faster.
- **Phase 1.** Fig. 3 is reproduced: series mean errors are within 0.05,
  with two 150 nm TE points out at 180–190 nm. F2 passes. F3 fails by the
  letter: 0.6–1.1 % of the power goes into decaying odd continuum members.
- **Phase 2.** The join-to-L3 loss is mostly the lattice.
  - L3 is flat from 0.25× to 8× its length (within ±0.04 dB for TE at
    1260 nm).
  - The stretch is linear in merged step size.
  - It persists with exact values.
  - Only a small part falls with length: TM 0.01 / 0.04 / 0.12 dB, TE
    ≤ 0.02 dB, read from a zero-solve scan (`stretch_scan.py`).
  - The walked corners of the height converter cost 0.27 dB (TE); the
    `(w_low, w_high)` axis choice causes this.
  - The paper's L2 TM attribution is not reproduced at 1260–1310 nm, and
    only partly at 1360 nm (0.09 dB).
- **Phase 3.** Best estimate against the paper's FDTD:
  - TE is within +0.03 … +0.09 dB.
  - TM is +0.47 … +0.64 dB, mostly the Q3 wedge-MMI TM loss. So the PDL
    gate fails (0.64–0.94 dB); about half of the excess at 1310 nm is the
    facet's reversed TE/TM order.
  - The 1310 nm TM best estimate is +0.4999 dB. It meets the 0.5 dB
    criterion by 8e-5 dB, a hundredth of the digitisation error, so the call
    is not resolved; the conservative bound does not meet it.
  - The conservative bound is TE 1.68–1.75 dB and TM 2.11–2.35 dB.
  - The conventional tips' TM penalty is 4.0 / 3.3 / 2.7 dB, falling with λ.
  - §5.8 aligned matches a from-scratch EME to ≤ 2.2e-6 on 60 of 280
    sections (≤ 3.6e-5 when cut from the device's own path); the full device
    was not run.
  - §5.4 slicing fails by the letter (2.4e-3 against 1e-4).
  - Reciprocity is 2.5e-4, measured with the reciprocal projection off
    (the column cap is off; with it, 4.1e-3).
- **Phase 4.** Lengths optimised at zero mode solves.
  - The 1310 nm objective falls from 2.06 to 1.74 dB and the band objective
    from 2.28 to 2.03 dB (CMA-ES plus polish).
  - L-BFGS-B alone stalls at 1310 nm and converges 0.02 dB short on the
    band.
  - The seeded CMA-ES landed on a different 1310 nm optimum after the
    reciprocal projection (43/19/42 → 31/31/57 µm for L1/L2/L3, 1.755 →
    1.739 dB), although the projection moved the Table 1 objective by only
    0.001 dB: the surface is flat-bottomed.
  - The staircase moves by as much under a length change, so the gains are
    indicative.
  - A facet CMA-ES (41 solves, 38 archived) found `Wtip` 95 nm, `g` 1.28 µm,
    taking the worse polarisation from 0.831 to 0.872. The device loss of
    that design was not computed.
- **Found on the way.** Three corrections to the cascade (README →
  "What changed relative to upstream" → Fixed):
  - **One sign per mode in both overlap sets.**
    `Geometry._equalize_overlap_phase` masked `overlap_ab` and `overlap_ba`
    separately. On a lossy basis a branch can collect an odd number of sign
    disagreements between the two sets, which mis-signed its couplings at the
    merge step at 1260 and 1360 nm. The base class and `DirectGeometry` now
    take one gauge (tag `pre_gauge_ab_ba`), and the study's override is gone.
  - **The interface reflection block.** Upstream's `R12` mixed the indices
    of the two sections, and the T→S conversion had a sign error. Fixed
    against exact references (tag `pre_reflection_formula`). Here it moved
    the losses by ≤ 0.008 dB and cut the reciprocity defect tenfold.
  - **The interface column cap is opt-in.** It had been introduced against
    a gain that was the gauge bug, and it cost this device 0.00–0.12 dB
    (tag `pre_column_cap_default`).
  - **Later: the reciprocal interface projection** (from report 13's
    passivity work, tag `pre_passivity`). Here it moved the best estimates
    by ≤ 0.009 dB. It also removes the two-mask rule's reflection signature
    (merge-step reflection 0.073 → 1.3e-4) but not its spurious transmission
    (1.073).

## Limits to state

* The comparable reference is the paper's **3D-FDTD** (Fig. 5), not the
  measurement. Fig. 8 adds facet quality, alignment, Fresnel reflection
  (~0.15 dB for air/SiO₂) and fabrication.
* The launch is a Gaussian in the facet plane, in SiO₂, with no facet
  reflection. The Fresnel term is reported separately.
* EME is unidirectional in its launch. A reflection between the facet and
  the MMI shows up in the S-matrix but not in the launch.
* Vertical sidewalls; the paper does not give the angles.
* Q1–Q3 are this project's readings, not the paper's numbers. Table 1 does
  not fix the width profile through L1–L3 or the output taper's length. If
  Phase 3 misses Fig. 5, check Q1 first: the L2 width sets the TM loss the
  paper attributes to that section.
