# Dataset design doctrine

*Moved verbatim from `CLAUDE.md` by task 16 step 0.7 (2026-09-27). Section numbers are unchanged, so a citation of the section number in older text still finds it here.*

## 3. Dataset design doctrine

**One dataset per cross-section family, never per device.**

A DBEME dataset is keyed to `{material stack, λ, mesh + window, mode count,
cross-section topology, swept parameters}`. It is *not* keyed to the device.
The longitudinal path — linear taper vs. Bezier vs. adiabatic-optimised — is
free; that is the entire economic argument for the method. Anything that
changes the mode problem (etch depth, cladding, λ, window) forces a new
dataset.

| dataset | topology | axes | serves | status |
|---|---|---|---|---|
| `Si_fulletch_220nm` | 1 strip core | `top_width`, `curvature` | tapers, Bezier/Euler bends, S-bends, ring waveguide | **shipped** |
| `Si_fulletch_220nm_pair` | 2 strip cores | `w1`, `w2`, `gap`, `curvature` | adiabatic coupler, RAC, ADC, curvy/pulley ring coupler | **needed — demos 1, 3** |
| `Si_bilevel_<t1>_<t2>` | 2 cores, vertically asymmetric | `w1`, `w2`, `gap`, `slab`/etch | polarization rotator-splitter | **needed — demo 2** |
| `Si_plasmonic_slot_1550[_sharp]` | Si wire between gold walls, suspended, lossy PML basis | `w_si` | lateral stand-in for the NTT converter | **shipped** — `reports/12` |
| `SiO2_kocabas_set2_1550` | Si wire + gold slot, SiO₂-embedded, lossy PML basis | `w_si`, `gap` | Kocabaş converter (arXiv:1801.00833), gap and length sweeps | **shipped** — `reports/13` |
| `SiO2_kocabas_set2_1550_fem_c20_hs1` | same, rounded corners, FEM mesh, wall axis at 1 nm | `w_si` (5 nm), `half_slot` (1 nm) | the converged Kocabaş numbers; gap sweep | **shipped** — `reports/13` §10 |
| `Si_ring_coupler_220nm_<1530|1550|1570>[_N6..N30|_c20]` | straight bus + moving ring core, bus fixed (`platforms.BusRingStrips`) | `w1`, `w2`, `gap` (10 nm cell, axis tied to the cell) | bus-ring point coupler of the all-pass ring | **shipped** - `reports/16`; ring side of the coupler is imposed by symmetry, see tasks/11 section 2.1 |
| `Si_bilevel_pair_220_150nm_ox[_tip]_<1260|1310|1360>`, `..._full_tip_<λ>` | two mirrored arms at 150/220 nm merging into one strip (`BiLevelPair`), oxide-only stack (`_full_tip`: with the Si substrate), lossy PML basis, target point by point | `w_low` 10 nm; `w_high` 10 nm with 5 nm over 220-370 nm and 367.5 nm (both 2.5 nm over 90-260 nm in `_tip`); `gap` 20 nm (10 nm below 200 nm) and 415 nm | the Wan & Wang double-tip edge coupler: tips, height converter, pair taper, MMI and output on one grid | **shipped** - `reports/22`; the 10 nm lattice costs ~1.1 dB (TE) from the tip join to the end of the converging pair: per step, not per length. The `(w_low, w_high)` axes put a corner point at every diagonal step of the height converter (0.27 dB TE). The facet needs the full stack: `..._full_tip_<λ>`, same grid |

*What a two-axis dataset costs, measured.* `SiO2_kocabas_set2_1550` is
471 × 347 at a 5 nm cell with 20 modes: **~10 min per cross section**, so the
52-point design path took 15 034 s cold. Every one of the eight taper lengths
in report 13 §4 then cost **0 s** — a different longitudinal path over the
same grid points solves nothing new, which is the method's entire argument.
A *new* `w_gap`, by contrast, walks a different part of the gap axis and pays
for the points it has not seen (10 119 s at 25 nm, 5 318 s at 125 nm). Plan
sweeps so the expensive axis is the one held fixed.

*And watch the overlap, not only the solve.* Until 2026-09-08
`overlap_matrix` formed the full `(2N, 2N, 3, nx, ny)` cross product before
summing it away — 11.7 GiB and 23.6 s per call at 471 × 347 with 20 modes,
four calls per point, about 40 % of a build. At 941 × 693 it wanted 46.5 GiB
and the run died. It is now a contraction (`tests/test_overlap_contraction.py`).
The general point: cost here scales as `N_modes² × grid`, so a finer grid and
a larger basis multiply, and anything that materialises that product is a
wall rather than a slowdown.

Naming: `<material/stack>_<geometry>_<λ in nm>`, e.g. `Si_fulletch_220nm_1310`.

### Cost control for the multi-dimensional datasets

Naive grids explode: a 4-D `(w1, w2, gap, κ)` axis at 50 points each is 6.25M
cross sections. Three mitigations, apply all of them:

1. **Neighbour-only overlaps.** Only adjacent grid points are ever needed
   (`interp_multi_adj_pts` already inserts intermediate points), so cost is
   `~d · N^d · M²`, not `N^{2d}`. *Verified and tightened 2026-09-07:* the
   updater never built a pair table, but it linked every visited point to
   **all** its grid neighbours — up to `2d + 1` solves per path point, three
   on the two-axis Kocabaş dataset with two of them for points no device
   visits. A geometry now passes its path links (`Geometry._path_links`,
   `DataUpdater.calc_data_point_modified(..., neighbours)`), so a path costs
   one solve per distinct point; `populate_dataframe` keeps the full
   neighbourhood. `tests/test_dataset_links.py`.
2. **Reparameterise to cut dimensions.** For a coupled pair the supermode
   physics depends on `Δw / κ_coupling`, so sweep `(w̄, Δw)` at a small number
   of coarse `gap` values instead of a dense 3-D `(w1, w2, gap)`.
3. **Non-uniform axes.** Sample densely only where `|dn_eff/dp|` is large —
   i.e. near the anti-crossing, which is exactly where overlaps depart from
   the identity. Elsewhere the axis can be 5–10× coarser. Lazy evaluation
   means a wide axis costs nothing until a device visits it.

   *On a metal-edge platform the axis is tied to the cell* (an edge inside
   a cell aliases `n_eff`, report 12 §7.3), so a finer axis where it matters
   needs a finer **grid** where it matters. `PMLModeSolver(refine_x=,
   refine_y=)` cuts the uniform grid into finer cells inside given regions
   (`pml.refined_axis`: region bounds on base nodes, integer ratio, two base
   cells clear of each end so the PML and the cross section's edge
   reconstruction stay on uniform cells). Everything downstream already took
   per-cell spacings except three things that inferred cell geometry from
   the centres — `_fill_fraction`, `_rounded_rect_fill`, and the unweighted
   power sums behind confinement and `TE_pol` — which now use exact cell
   edges and per-node areas (`tests/test_nonuniform_grid.py`). A refinement
   is part of the dataset identity. On a SiN strip, refining `x` and `y`
   over the core reproduces the uniformly fine `n_eff` to 0.3 % of the
   coarse-to-fine gap at half the unknowns. On the Kocabaş converter
   (`kocabas_converter_dataset_info(refine=((lo, hi, cell_fine), ...))`,
   strips in |x|, axes `(w_si, half_slot)` so that both edges are grid
   values in their own right) the lesson was *where*: the Si tip at 1 nm
   (2 nm width steps, +20 %) bought 0.7 points because the silicon's steps
   were never the staircase — the gold wall's 5 nm jumps are, 6.1 % of the
   7.2 % single-mode mismatch on clean axes, and the wall's skin depth is
   also what the 5 nm grid under-resolves (report 13 §7). Measure which edge
   carries the loss on axes where the edges move separately *before*
   choosing the strip; an attribution on `(w_si, gap)` axes counts wall
   motion as Si motion.

   *Metal corners, three rules learned the hard way (report 13 §7).* A
   refined strip must keep the cells at any metal corner **isotropic** —
   1 × 5 nm cells at a gold wedge returned |E_y| 37× the coarse value and a
   TE fraction of 0.09 for a TE mode while H and `n_eff` looked fine, so
   pair an `x` strip with a `refine_y` strip over the corner rows. A
   **sharp** metal wedge does not converge as it is resolved (the 200 nm
   Kocabaş branch moved 0.05 from 5 nm to 1 nm cells); the 5 nm grid's
   answer is the grid's regularisation, and only a rounded corner is a
   geometry a refinement can converge on (0.012 for a 20 nm arc). And a
   rounded metal arc on this silica platform needs
   `PlasmonicSlotConverter(fill_floor=0.06)`: the 8 × 8 sub-sampled fill
   lands on the ε-near-zero mix (1/64 = 0.0156 against 0.0162), which
   produced |E| = 7.8 against a bulk of 0.01. Off by default; in the
   fingerprint when on.
