# 12 — Plasmonic mode converter (lateral 2-D model)

> **Revised (2026-09-08).** Sections 1–8 were first computed on fields with a
> half-cell E/H misregistration in `PMLModeSolver` (report 07 §23, found by
> the SIRAC session): E was zero-padded from the cell centres onto H's node
> grid, which makes every stored overlap non-symmetric — the `O_ab = 1.0001`
> against `O_ba = 0.9980` of §7.5 is that offset. The platforms now solve with
> `colocate=True`, both caches have been rebuilt, and **every number in this
> report is on co-located fields**. §10 says what the offset was and was not
> responsible for, and what changed.

> **Outcome in one paragraph.** The first device on the lossy PML backend
> runs end to end and is passive, but only after the interface algebra was
> changed: upstream projects tangential continuity on the modes of the section
> being *left*, which for a field the truncated basis cannot represent returns
> the missing part as **gain** — 0.2–1.9 % per 10 nm step, ×1.44 over the
> taper, independent of mode count and of the PML. Projected on the section
> being *entered* the same dataset gives a bounded cascade. With the gold
> corners rounded (20 nm, §8) the converter reads **−1.61 dB at 600 nm**, of
> which about −1.4 dB is the mismatch loss of a 40-step staircase and the rest
> metal loss; the slot itself loses **0.87 dB/µm** (sharp corners: 1.27;
> paper: "about 1 dB/µm"). The paper's −1 dB and its 600 nm optimum are not
> reproduced, and §3 and §5 say why a fixed-grid staircase cannot show them.
> Reciprocity of the physical channel holds to 1e-3, not 1e-6 (§1).

**Device.** Ono, Taniyama, Kuramochi, Nozaki, Notomi — *Toward Application of
Plasmonic Waveguides to Optical Devices*, NTT Technical Review **16**(7), 2018
(the device: Ono *et al.*, Optica **3**, 999, 2016). A 400 × 200 nm Si wire
feeds a gold/air metal–insulator–metal (MIM) waveguide with a 50 × 20 nm air
core through a 600 nm taper. The paper's numbers at 1.55 µm: coupling
**−1 dB simulated** with the designed 20 nm air gap, **−1.4 dB calculated and
−1.7 dB measured** with the 40 nm gap that was fabricated, and a slot loss of
**about 1 dB/µm** in the deep-subwavelength regime.

**What this report models.** The 2-D *lateral* stepping stone CLAUDE.md
prescribes for this device (§7, demo 4): the Si width tapers 400 nm → 0
between two gold walls that follow the taper at a constant air gap, and the
walls' `2 × gap` slot is the output waveguide. The device confines its plasmon
**vertically** — the 20 nm is between gold above and below — and one swept
width cannot represent that; the model's plasmon is the lateral gap plasmon of
the slot the walls leave behind. §5 says which comparisons survive it.

**Method.** DBEME on a **lossy PML basis** (`PMLBackend`, report 06): complex
`n_eff`, the scattering-matrix cascade (`SMATRIX_METHOD = "auto"` → `direct`
because the backend declares itself lossy), `force_unitary=False` throughout
— projecting a lossy S-matrix onto the nearest unitary one would delete the
metal loss it is supposed to report — and the interface continuity projected
on the output side (`INTERFACE_PROJECTION = "auto"` → `"output"` for a lossy
basis; §7.5, report 06 §11).

---

## 1. Sanity gate

Shipped configuration (§2: rounded corners), 20 nm gap. Every number is in
`reports/output/plasmonic_converter.json` (keys without suffix), the
sharp-corner run of §8 under the `_sharp` keys.

| check (§5.x) | criterion | measured | pass |
|---|---|---|---|
| constant-width walled wire, 1 µm: `T = exp(−2 k₀ Im n L)` — propagation loss carried through the cascade | `|T − expected| < 1e-3` | 0.98061 measured and expected (`n = 2.3538 + 0.0024j`, 0.085 dB/µm) | ✓ |
| passivity: launched (physical) column of `|S|²` sums below 1 | `< 1.05` | **0.704** at 600 nm (co-located; 0.741 on the offset fields, 1.009 at 150 nm there). Was **1.462** with the input-side projection (§7.5) | ✓ |
| interface power change per 10 nm step, `1 − Σ_j|S_j0|²` (fig. 4) | reported | −0.13 % at 400 → 390 nm falling to −1.7 % at 10 → 0 nm — every interface loses. With the input-side form every interface gained the same amount | ✓ |
| 5.1 reciprocity of the physical channel, `|T12 − T21|` for the launched mode, 600 nm | `< 1e-6` | **1.0e-3** on co-located fields (was 2.1e-2 on the offset fields: the half-cell E/H offset was most of it, §10); per interface `T21 = T12ᵀ` to 6e-7 | ✗ residual truncation |
| 5.8 DBEME vs direct EME, 600 nm, direct sections at the 41 axis widths | `max|ΔT| < 1e-3` | −1.61 dB vs −1.61 dB, `max|ΔT|` = **8.8e-4** per branch on co-located fields; 5044 s direct against < 0.1 s warm | ✓ (§4) |
| 5.4 slicing: direct at 21 sections (20 nm steps) vs 41 | `max|ΔT| < 1e-4` | **−2.88 dB vs −1.61 dB**, `max|ΔT|` = 0.18 — the staircase mismatch doubles when the step doubles | ✗ by construction (§3) |
| 5.2 mode-count independence of the interface error, 400 → 390 / 100 → 90 nm (sharp corners, input form) | should fall with N | N = 16: 0.21 % / 0.65 %; N = 24: 0.20 % / 1.35 %; N = 32: 0.27 % / — | ✗ does not fall — §7.5 |
| same junctions with the PML off (PEC box, real box modes) | | 0.24 % / 1.35 % | same — not the Berenger modes |
| same junctions with a second shift-invert target at 1.05, non-guided modes kept by lowest loss | | 0.195 % / 1.35 %; launched mode couples to `Σ_j|O_0j|² = 1e-5` | same — not basis selection |
| power-normalisation factors `r = P_phys/|P_unconj|` of the end modes | reported | `r_in` 1.0000, `r_out` 0.9968 | — |
| 5.7 cell convergence (sharp corners), Si end / slot, 8 / 6 / 5 / 4 nm | reported, not graded | Si end 2.4033 / 2.4548 / **2.3459** / 2.3416; slot 1.3402 / 1.4007 / **1.1447** / 1.1795. Only 5 and 4 nm put the 20 nm gap on cell boundaries: between those the Si-end mode moves 0.2 %, the corner-concentrated slot mode 3 %. The unaligned 8 / 6 / 4 nm series had read 2.602 / 2.471 / 2.400 — partial metal cells, not the pitch (§7.3) | — |
| grid alignment (§7.3): `n_eff` smooth in width, every straight edge on a cell boundary | | 400 / 390 / 380 nm: 2.3538 / 2.3364 / 2.3132 | ✓ |
| 5.6 gauge, 5.13 tracking, PML low-side sign (report 06 §10), rounding geometry | tests | `tests/test_plasmonic_slot.py`, `test_pml_solver.py`, `test_lossy_assumptions.py`, `test_smatrix_direct.py` | ✓ |

Two things a lossy basis changes about *reading* an S-matrix, both applied:

* modes are normalised with the **unconjugated** power `½∫(E × H)_z`, so
  `|a_i|²` is physical power only up to `r_i`. For the two end modes `r` is
  1.0000 and 0.9968. Berenger modes have `r` = 2–3, which is why column sums
  over *their* inputs are not a passivity statement and are not graded.
* reciprocity is `S = Sᵀ` in the standard two-port arrangement. The lumped
  matrix is ordered `[b₂; b₁] = S [a₁; a₂]`; testing it directly reads 1.1
  and means nothing.

---

## 2. Device and dataset

### The model, and every departure from the paper

| | paper (3-D device) | this model (lateral) | why |
|---|---|---|---|
| Si core | 400 × 200 nm | 400 × **220** nm | every dataset in this project is 220 nm |
| taper | 600 nm, lateral **and vertical** | width only, 400 nm → 0 | one swept axis; the vertical taper needs a second geometric axis |
| MIM core | 50 × 20 nm, gold above and below | the `2 × gap` slot between the walls: **40 nm wide × 400 nm tall** | vertical confinement is not representable |
| air gap Si–metal | 20 nm designed / 40 nm fabricated | 20 nm; the 40 nm sweep was not run — its 80 nm slot is only marginally bound (`n = 1.0069`, sharp) | §7.4 |
| gold | thickness not stated | walls **400 nm tall, centred on the Si, corners rounded 20 nm** | wall *height* is what binds a lateral slot mode (§7.4); a sharp metal wedge carries a singular field no grid resolves (§8) |
| stack | Si on SiO₂, air above | **suspended in air** | over oxide the lateral slot's mode sits below the substrate index and leaks — §7.4 |
| input | plain Si wire, gold begins at the taper | walled wire (gold alongside the 400 nm section) | the dataset family has the walls everywhere; the plain → walled junction is estimated separately: **−0.48 dB** (mode-mismatch overlap, 2.0716 → 2.3538 + 0.0024j) |

### Dataset `Si_plasmonic_slot_1550` (and `_sharp`)

* cross section `PlasmonicSlotConverter`, axis `w_si` = 0 … 400 nm in
  **10 nm** steps (41 points); gold Johnson & Christy (n = 0.524 + 10.74j at
  1550 nm, ε = −115 + 11j); permittivity averaged in **ε**, not n; corners
  rounded by 20 nm with the arcs sub-sampled 8 × 8 per cell (`_sharp`:
  radius 0).
* grid: **5 nm** cell, 369 × 301 points (window ±0.92 × ±0.75 µm), snapped so
  that every straight edge falls on a cell boundary at every axis point
  (§7.3); PML 0.25 µm on all four edges, stretch 1 + 2j; 16 modes about a
  shift-invert target of 2.0; the PML recipe and the corner radius are part
  of the dataset identity.
* cost: 60–110 s per point; the cold build of the axis took **75 min**
  (4504 s, 41 points with their neighbour overlaps; 73 min for the sharp
  variant). Every device on the warm dataset then costs **< 0.1 s** — the
  7-length sweep below, and the re-evaluation of a whole sweep after the
  interface algebra changed, cost nothing. Direct EME costs 41 solves *per
  device*.

### The two end modes (aligned 5 nm grid, rounded corners)

| end | mode | `n_eff` | loss | confinement |
|---|---|---|---|---|
| Si, 400 nm | lateral hybrid plasmonic (TE-like) | 2.3538 + 0.0024j | 0.085 dB/µm | 0.94 |
| slot, 0 nm | gap plasmon of the 40 × 400 nm slot | 1.3513 + 0.0246j | **0.87 dB/µm** | 0.50 |

The slot loss is directly comparable with the paper's "about 1 dB/µm". The
Si-end mode is not the bare wire's TE0 (2.0716 suspended): with gold 20 nm
from each sidewall it is a hybrid plasmonic mode with its field pulled into
the gaps — which is also what the real device's input becomes once the gold
begins. Along the taper this is the only physical branch; the other fifteen
are Berenger modes.

![branches](output/plasmonic_1_neff_path_gap20.png)

---

## 3. DBEME result

![sweep](output/plasmonic_2_length_sweep.png)

| L (nm) | sections | converted `|S_out,in|² r_out/r_in` | dB | reflected (physical) | absorbed + radiated |
|---|---|---|---|---|---|
| 150 | 42 | 0.733 | −1.35 | 0.0398 | 0.203 |
| 300 | 42 | 0.706 | −1.51 | 0.0139 | 0.265 |
| 450 | 42 | 0.697 | −1.57 | 0.0082 | 0.286 |
| 600 | 42 | **0.690** | **−1.61** | 0.0076 | 0.296 |
| 800 | 42 | 0.678 | −1.69 | 0.0038 | 0.313 |
| 1000 | 42 | 0.668 | −1.75 | 0.0038 | 0.323 |
| 1500 | 42 | 0.642 | −1.92 | 0.0018 | 0.352 |

(Co-located fields, §10. On the offset fields the same sweep read −0.94 →
−1.83 dB with 6.6 % reflected at 150 nm.)

Monotonic in length, with no optimum. That is what the model must say, and
it is the report's most useful negative result about DBEME on this device:

* the grid-snapped path visits the **same 41 cross sections at every
  length**. The mismatch loss of the 40 steps is therefore length-independent
  — from the stored overlaps the single-mode staircase product is
  `A = Π|t_m|² = 0.723` (−1.41 dB); the cascade at 150 nm, where metal loss is
  negligible and the steps are 3.7 nm apart, sits at 0.733 with 4.0 % coming
  back as reflection — the one place the coherent multi-step physics is
  visible;
* what grows with length is the metal loss along the taper (0.085 →
  0.87 dB/µm across the width range): 0.989 at 150 nm, 0.956 at 600 nm, 0.893
  at 1500 nm as a propagation factor. `A × propagation` is 0.691 at 600 nm
  against the cascade's 0.690 — on co-located fields the multi-mode cascade
  and the single-branch staircase agree to three digits, because the
  spurious couplings to the Berenger branches that the offset fields carried
  (§10) are gone;
* the physics that makes a real taper *better* when longer — the
  destructive interference of the field shed at successive steps, which is
  what "adiabatic" means — needs that shed field to be in the basis. Here
  `Σ_j|O_0j|²` is a tenth of `1 − |t|²` at every step (§7.5): the basis holds
  one physical branch and Berenger modes, and the near field a 10 nm shift of
  a metal edge displaces is in neither. The output-side projection loses it,
  correctly, instead of returning it as gain; it cannot make it interfere.

So the −1.61 dB decomposes as **≈ −1.4 dB of staircase mismatch** (a
discretisation artefact that scales with the width step — per-step loss ∝
Δw², steps ∝ 1/Δw, so the total ∝ Δw — confirmed by the 21-section direct run
of §4) **and ≈ −0.2 dB of metal loss**; the plain-wire junction adds −0.48 dB
in front of it. The lateral model's converged answer at 600 nm, extrapolating
the staircase to Δw → 0, is therefore of order −0.7 dB — metal loss plus
junction — *if* the taper is adiabatic, which this basis cannot test.

### Where the power goes, interface by interface

![excess](output/plasmonic_4_interface_excess.png)

The direct route stores an interface and a propagation matrix per section.
Every propagation matrix removes the branch's metal loss (red, 0.02 → 0.25 %
per section); with the output-side projection every interface matrix
removes its unrepresented mismatch (blue, 0.18 → 1.7 %). The right panel
follows the launched branch's own `|T00|²` and, separately, the sum over all
2N outputs; on co-located fields the two coincide along the whole cascade
(0.692 and 0.704 at the end). On the offset fields the total had carried
Berenger amplitudes that dropped out in a 4 % step near 195 nm — the
couplings §10 describes. With
upstream's input-side projection the blue curve had the same magnitude and
the opposite sign, and the sharp-corner cascade ended at 1.439 instead of
0.721 (§8).

---

## 4. DBEME vs direct EME

![direct](output/plasmonic_3_direct_vs_dataset_gap20.png)

| 600 nm taper, rounded corners, co-located fields | sections | converted | `max|ΔT|` per branch | time |
|---|---|---|---|---|
| dataset (grid-snapped path) | 42 | −1.61 dB | — | **< 0.1 s** warm (92 min cold, once) |
| direct, sections at the 41 axis widths | 41 | −1.61 dB | **8.8e-4** | 5044 s |
| direct, every other axis width | 21 | −2.88 dB | 0.18 | 451 s (widths already cached in-process) |

On the offset fields the same rows read −1.44 / −1.43 dB at 9.7e-4 and
−2.84 dB at 21 sections; the sharp-corner run of §8, −1.65 / −1.64 dB at
1.3e-3 and −3.23 dB.

At matched slicing the residual is section placement: the dataset path puts
its interfaces where the width crosses a grid mid-point, the direct path at
uniform `z`, so the first and last sections differ by half a step. The direct
sections are placed *at* the axis widths deliberately; any other width puts a
metal edge inside a cell (§7.3) and compares a different discretisation, not
a different method. The half-slicing row is not an agreement check: with
20 nm steps the mismatch loss doubles, the ∝ Δw behaviour of §3.

Timing is the method's claim: the 7-length sweep of §3 cost one cold build
and then nothing measurable per device; the same sweep by direct EME would
cost seven times 2670 s. Re-evaluating an entire sweep after the interface
algebra changed (§7.5) cost 0 s of mode solving.

---

## 5. DBEME vs literature

### What is comparable, and what is not

| quantity | comparable? | why |
|---|---|---|
| slot propagation loss | **yes** | a property of the plasmon itself: **0.87 dB/µm** here (1.27 with sharp corners), "about 1 dB/µm" in the paper |
| plain-wire → walled-wire junction | as an order of magnitude | −0.48 dB here is mode mismatch only; in the device it is part of the −1 dB |
| conversion efficiency at 600 nm | **no** | −1.61 dB is ≈ −1.4 dB of staircase artefact; and lateral vs vertical confinement, suspended vs on oxide, 220 vs 200 nm Si, no vertical taper |
| optimum taper length | **no** | the model has none, for the reason in §3 |
| 20 nm vs 40 nm gap | **no** | in the device the gap is a taper parameter and the MIM core stays 50 × 20 nm; here the gap sets the output slot, and the 80 nm slot barely binds |

### Discrepancy table

| quantity | paper | this model | cause |
|---|---|---|---|
| slot loss | ~1 dB/µm (50 × 20 nm vertical core) | 0.87 dB/µm (40 × 400 nm lateral slot, rounded); 1.27 sharp | different slot geometry; both deep-subwavelength gap plasmons in gold; sharp corners add corner absorption and leakage (§8) |
| coupling at 600 nm, 20 nm gap | −1 dB | −1.61 dB cascade; ≈ −0.7 dB with the staircase extrapolated away (metal loss plus junction) | staircase mismatch of a 10 nm-step basis (§3); lateral model; junction counted separately |
| coupling at 600 nm, 40 nm gap | −1.4 / −1.7 dB | not computed | the lateral 80 nm slot is not a bound port |
| optimum length ≈ 600 nm | yes | none | shed field not in the basis (§3, §7.5) |

---

## 6. Conclusions and limits

**Established.**
1. The lossy chain works end to end: `PMLBackend` behind `DataUpdater`,
   complex `n_eff` in the pickles, the PML recipe and geometry in the dataset
   identity, Hungarian tracking through 41 interfaces, the scattering route
   chosen by `auto`, propagation loss carried exactly, and a passive cascade
   once the interface is projected on the output side. Warm devices cost
   < 0.1 s, and a change of interface algebra re-evaluates a whole sweep for
   free.
2. The lateral slot plasmon of a 40 nm gold slot is at 1.351 + 0.025j —
   0.87 dB/µm, the paper's regime — once the walls are tall enough, the
   structure is suspended (§7.4) and the corners are rounded (§8).
3. The interface projection side is a real choice with a real consequence
   in a truncated basis (§7.5, report 06 §11): upstream's form is unbounded
   there. `INTERFACE_PROJECTION = "auto"` takes the bounded form for a lossy
   basis and leaves the lossless datasets, whose transfer route needs the
   other form invertible, exactly as validated.
4. DBEME's fixed grid cannot show an adiabatic optimum for a device whose
   step-to-step mismatch field is outside its basis: the mismatch loss is
   length-independent and only the metal loss varies. This is CLAUDE.md
   §5.9 item 5 and §7 demo-4 blocker 4 measured, not argued.
5. Rounding the metal corners changes the slot mode (binding and loss) and
   not the taper's step mismatch: the singular corner field was the slot's
   problem, the flat-wall gap field is the taper's (§8).

**Not established.** The paper's efficiency or optimum length, or any gap
sensitivity. Reciprocity of the physical channel is 1e-3, not 1e-6.

**What would improve it, in order of cost.**
1. *Finer aligned steps*: the staircase loss is ∝ Δw; 5 nm steps halve it
   but need a 2.5 nm cell for alignment (4× the cost per solve).
2. *A variational interface*: least-squares matching of `E_t` and `H_t` over
   the aperture rather than projection onto a truncated basis, which would
   also bound the reflection blocks — a Stage 6 of the S-matrix work.
3. *A basis that holds the near field*: not more shift-invert modes (N = 24,
   32, a second target — all tried, none helped) but edge-adapted functions
   or a 3-D reference. FDTD of the same lateral structure is what the report
   protocol asks for, and nothing here replaces it.

**Limits carried regardless.** Between the two exact grids (5 and 4 nm) the
Si-end index moves 0.2 % and the corner-concentrated slot index 3 % (sharp
corners; §1); the model is lateral; the 40 nm gap has no bound lateral
counterpart.

---

## 7. Five things found on the way

None was visible on the lossless datasets or on the `+x`-only PML validation;
each is now a regression test.

### 7.1 The `-x` and `-y` PML layers were gain (report 06 §10)

EMpy's `stretchmesh` ends with `abs(Im)` on the stretched coordinate. The
stretch is antisymmetric by construction and must be, because the operator
consumes `diff(x)`; the `abs` made the low-side layers amplify. With `+x` and
`-x` together every radiating mode's `Im n_eff` came out exactly 0
(PT-symmetric spectrum), and with a third edge the gain/loss corners hosted
near-real Berenger modes that shift-invert returned instead of the physical
mode. `stretched_grid` now applies the stretch itself. `tests/test_pml_solver.py`.

### 7.2 The radiation mask condemned every plasmonic mode

`radiation_mode_mask` rejected any mode above 100 dB/cm — a lossless-model
heuristic. A Si wire 20 nm from gold sits at 0.09 dB/µm and the slot at
~1 dB/µm, so the launch basis was empty. The rule now applies only when the
dataset says it is lossless. `tests/test_lossy_assumptions.py`.

### 7.3 Sub-cell aliasing of a metal edge

With the metal edge free to fall anywhere inside a cell, the ε-averaged edge
cell swings between air and gold as the width changes, and the hybrid mode's
index with it: at a 6 nm cell, 400 / 395 / 375 nm read 2.471 / 2.371 / 2.429
— a 0.1 jump per 5 nm where the physical slope is 0.002 per nm. Every
interface then carried a spurious geometric step. The cure is alignment: a
pitch exactly equal to the cell with the origin on a node, and an axis in
steps of `2 × cell`, so every straight edge sits on a cell boundary at every
axis point and every cross section is discretised identically; the rounded
arcs are sub-sampled and translate by whole cells along the axis, so they are
discretised identically too.
`tests/test_plasmonic_slot.py::test_platform_grid_is_aligned_with_the_geometry`.

The alignment also removed a comfortable illusion: with partial-metal cells
the 40 nm lateral slot had read 1.10–1.34 depending on where the edge fell;
discretised exactly, between 220 nm walls, it is **0.869** — below the air
line. Which is how §7.4 was found.

### 7.4 What binds a lateral slot mode is the wall height

| walls (sharp corners) | slot 20 nm | slot 40 nm | slot 80 nm |
|---|---|---|---|
| 220 nm (Si height) | 1.037 + 0.084j | 0.869 (leaky) | — |
| 400 nm, centred | — | **1.145 + 0.036j** | 1.007 + 0.027j (marginal) |

A 40 nm MIM slab plasmon sits at 1.47, but a slot 220 nm tall and open above
and below is a slab of index 1.47 and thickness λ/7 in air — `V ≈ 1`, barely
bound. The device does not have this problem because it confines vertically.
The model therefore uses 400 nm walls centred on the Si, and a window tall
enough to keep the slot mode's tail 0.3 µm from the PML. Over oxide the slot
mode is below the substrate index in every configuration tried (1.18 + 0.058j
at best) and leaks.

### 7.5 Which side an interface is projected on

The taper's first run ended at ×1.44 with every interface *adding* power. It
was not the mode count (16, 24, 32 gave 0.21, 0.20, 0.27 % at the first
step), not the PML (a PEC box gave 0.24 %), not the pseudo-inverse
(`rcond=None`, identical digits), not basis selection (a second target near
the cladding line, 0.195 %). The single-mode algebra on the same overlaps
gave `|t|² = 0.998`; the cascade gave `1.002 = 1/0.998`, and that identity
is the mechanism. Tangential continuity is imposed weakly, by projecting the
field equations on a mode set, and upstream projects on the section being
*left*: `T12 = 2 inv(O_ab + O_baᵀ)`, single-mode limit `1/|O|²` — the part
of the mismatch the basis cannot hold comes back as gain. Projected on the
section being *entered*, `T12 = 2 O_abᵀ inv(O_abᵀ + O_ba) O_ba`, limit
`|O|²`, it is lost. On six-mode Si tapers the two differ by 2.5e-4 per
interface, under the truncation floor every earlier report carries and under
the `force_unitary=True` they all ran with; here they are 1.44 against 0.72
for the same forty interfaces and the same dataset. `SingleEME.
INTERFACE_PROJECTION = "auto"`; report 06 §11; `tests/test_smatrix_direct.py`.

Why the lossless datasets keep upstream's form: the transfer route inverts
`T21`, and the output-side `T` inherits the near-zero overlap of a radiation
mode that barely exists on one side of an interface. On the README taper,
whose one near-cutoff radiation mode already gives an interface condition
number of 3e7, the Stage 4 gate read 9e4 with the output form forced. A lossy
basis runs the direct route, where no inverse of `T` is ever formed.

---

## 8. Sharp against rounded gold corners

Sections 1–7 were first measured with sharp plates, which then read
−1.65 dB (−1.85 dB once the fields were co-located, §10). Jae's reading of
that number was that a sharp plasmonic edge carries a very dense, singular
field that the grid cannot resolve and that radiates and absorbs, and that the
demonstration should round it. The platform now does (radius 20 nm, four
cells; arcs sub-sampled 8 × 8 so the fill fraction never lands on the
ε-near-zero mix), and the sharp basis is kept as `Si_plasmonic_slot_1550_sharp`.
Same grid, same axis, same algebra:

| | sharp corners, r = 0 | rounded, r = 20 nm |
|---|---|---|
| slot mode at 0 nm | 1.1447 + 0.0362j, **1.27 dB/µm** | **1.3513 + 0.0246j, 0.87 dB/µm** |
| Si-end mode at 400 nm | 2.3459 + 0.0026j, 0.093 dB/µm | 2.3538 + 0.0024j, 0.085 dB/µm |
| single-mode step `|t|²`, first / last interface | 0.99808 / 0.98228 | 0.99819 / 0.98237 |
| staircase product `A = Π|t_m|²` | 0.7014 (−1.54 dB) | 0.7233 (−1.41 dB) |
| propagation factor, 600 nm | 0.9454 | 0.9557 |
| converted, 600 nm | 0.654 (**−1.85 dB**) | 0.690 (**−1.61 dB**) |
| sweep 150 → 1500 nm | −1.66 → −2.18 dB | −1.35 → −1.92 dB |
| reflected into physical modes at 150 nm | 0.06 % | 4.0 % |
| launched column total, 600 nm | 0.667 | 0.704 |
| physical-channel reciprocity | 7.0e-4 | 1.0e-3 |
| direct EME, 600 nm, 41 / 21 sections | −1.84 / −3.28 dB, `max|ΔT|` 1.2e-3 | −1.61 / −2.88 dB, `max|ΔT|` 8.8e-4 |

Both columns are on co-located fields (§10); the datasets differ only in
`corner_radius`.

**The reading.** The corner singularity was the **slot mode's** problem, and
rounding fixes it: the slot plasmon binds much better (0.145 → 0.351 above the
air line) and its loss falls by a third — the missing third was absorption and
leakage at four sharp wedges, where the mesh cannot resolve the field that a
90° metal corner concentrates. It was **not the taper's** problem: the
per-step mismatch `1 − |t|²` is the same to three digits in both bases,
because the field a 10 nm step displaces lives in the 20 nm gap along the
*flat* wall face, 90 nm from the nearest corner. The 0.24 dB the device gains
at 600 nm splits as 0.13 dB of staircase (`A`) and 0.10 dB of metal loss along
the taper — that is, rounding buys a better *output mode*, not a better
transition. What remains between −1.61 dB and the paper's −1 dB is the
staircase discretisation, ∝ Δw, and the lateral geometry — not the corners.

The one row that moves the other way is the reflection at the shortest taper:
4.0 % rounded against 0.06 % sharp at 150 nm. Per-step reflection is tiny in
both bases (`max |r|²` = 5.9e-5 sharp), so this is the coherent sum of forty
of them at 3.75 nm spacing, where every step reflects nearly in phase. It is
an artefact of the staircase's periodicity, it falls to 0.8 % / 0.05 % by
600 nm, and it should not be read as a property of either geometry.

## 9. The device as published has no bound lateral output on oxide

Jae's request for the paper's own curves - coupling vs air gap at 600 nm and
vs taper length at 40 nm - forced the question §2 had deferred: what does
the 50 × 20 nm core look like, and does a lateral model of it have a port at
all. The NTT review (`references/ntttechnical.pdf`) settles the geometry:
gold with **air** as the insulator, on an SOI substrate, and a converter
"tapered in both the lateral and vertical directions", coupling to an MIM
"that differs greatly in height via the air gap". The core is an air slot
between 20 nm-thick gold films on the buried oxide, air above, 50 nm wide.

Solved as a 2-D cross section that core has no bound mode, and it is not a
resolution or a metal-thickness effect:

| air slot on SiO₂, air above, `w_si = 0` | best quasi-mode | loss |
|---|---|---|
| gold 20 nm, slot 50 nm, 5 nm cell | 1.34 + 0.15j | 5.2 dB/µm |
| gold 20 nm, slot 50 nm, **2.5 nm cell** | 1.37 + 0.11j | 3.9 dB/µm |
| gold 20 nm, slot 20 nm, 2.5 nm cell | 1.37 + 0.11j | 3.7 dB/µm |
| gold 50 / 100 nm, slot 50 nm | 1.35 / 1.36 + 0.13 / 0.12j | 4.4 / 4.1 dB/µm |
| Si-loaded: 50 or 100 nm Si strip between the films | 1.35 / 1.36 + 0.14 / 0.13j | ~4.8 dB/µm |

Jae's Lumerical FDE on the 20 × 20 nm core found only `0.05 + 1.17i` - an
evanescent, below-cutoff solution, not a mode - so two solvers agree. The
picture that explains all of it is elementary: a lateral slot mode is the
infinite-plate MIM mode confined vertically by the film height, i.e. a slab
of index `n_MIM` and thickness `h_Au` in the surrounding dielectric. For an
**air**-filled 50 nm gap `n_MIM` is 1.39, already below the oxide's 1.444;
20 nm of height then leaves nothing to bind. And a Si strip narrower than
~200 nm on oxide is below the substrate index by itself, so the narrow half
of a lateral-only taper has no bound mode whatever the metal does. The
published device escapes this with its vertical taper, which one swept width
cannot represent.

The same estimate says when a thin-film slot *does* bind, and Kocabas's
converters (arXiv:1801.00833) are the test: they are fully embedded in SiO₂,
so the slot is silica-filled (`n_MIM` = 2.28 for 30 nm) and the environment
symmetric. Against his structure the solver reads:

| SiO₂-embedded slot, his gold | solver | slab estimate | paper |
|---|---|---|---|
| 250 nm gold, 30 nm slot | 1.835 + 0.024j, confinement 0.96 | 1.74 | → 2.28 as the height grows |
| 100 nm gold, 30 nm slot | 1.453 + 0.052j | 1.52 | — |
| 30 nm gold, 30 nm slot | 1.467 + 0.038j, `L_p` 3.2 µm | 1.455 | `L_p` 6.1 µm |

Bound in every row, on the slab-estimate curve, with the barely bound 30 nm
case within a factor two of the paper's propagation length (a 1 µm
evanescent tail on a 0.6 µm window). That geometry - specified to the
nanometre in the paper's Table II, bound at both ends, lateral by
construction, with a 95 % transmission to hit - is where the sweeps go:
`examples/demo_kocabas_converter.py`, dataset `SiO2_kocabas_set2_1550`,
report 13.

## 10. The half-cell E/H offset, and what it was responsible for

Found by the SIRAC session (report 07 §23): `PMLModeSolver` zero-padded the
cell-centred E components onto H's node grid, a half-cell misregistration in
both axes; `colocate=True` interpolates E onto the nodes as `MSEMpy` does,
both plasmonic platforms now set it, and it is part of the dataset identity.
Every number in this report before this section was first measured on the
offset fields; both platforms have been rebuilt and remeasured.

What the offset was **not**: the launched channel's step mismatch. On the
400 → 390 nm junction the co-located overlaps of the launched mode are the
offset ones to four digits (`O_ab` 1.00119 vs 1.00121, `O_ba` 0.99701), the
single-mode `|t|²` per step and the staircase product `A` (0.7233 vs 0.724)
are unchanged, and so is the projection-side result of §7.5: the input-side
column power is 1.0018 and the output-side 0.9982 on co-located fields.

What it **was**: the spurious couplings. The launched mode's overlap with
the other basis members drops from `Σ_j|O_0j|² ≈ 2e-4` per step to ~1e-5;
the reflection at 150 nm halves (6.6 → 4.0 %), the physical-channel
reciprocity residual falls twentyfold (2.1e-2 → 1.0e-3), and the cascade
lands on the single-branch staircase to three digits (0.690 against
`A × propagation` = 0.691) where the offset fields had it 4 % above. The
headline moves from −1.44 to **−1.61 dB** at 600 nm: the offset had been
handing power back into the launched channel through couplings that are not
there.

The sharp platform behaves the same way, which is the useful control: its
staircase product is unmoved (0.702 → 0.7014), its reciprocity residual falls
from 1.9e-2 to 7.0e-4, and its cascade drops from −1.65 to −1.85 dB. The
sharp basis keeps more genuine off-diagonal coupling than the rounded one
(`Σ_j|O_0j|²` summed over the taper, 3.4e-3 against 1.2e-3), and correspondingly
its cascade sits 1.4 % *below* its own single-branch estimate where the
rounded lands on it to 0.1 % — the singular corner field really does scatter,
it is simply not what limits this taper.

---

### Reproducing

```bash
cd examples
python demo_plasmonic_converter.py --gaps 20 --direct-lengths 600                    # rounded, ~2.5 h cold
python demo_plasmonic_converter.py --gaps 20 --direct-lengths 600 --variant sharp --cell-check
```

Results are cached stage by stage in `reports/output/plasmonic_converter.json`
(sharp-corner keys carry the `_sharp` suffix); a second run is warm. The
interface profile and the staircase estimate are scratch analyses whose
outputs are `plasmonic_interface_excess[_sharp].json`,
`plasmonic_single_mode_estimate[_sharp].json` and
`plasmonic_4_interface_excess[_sharp].png`.
