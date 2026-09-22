# CLAUDE.md — working agreement for the DBEME repo

Read `README.md` first. It describes *what the code is and what was changed*.
This file describes *how to work in it*: conventions, the dataset design
doctrine, what must still be verified before any number is reported, and the
demo/report plan.

**Owner:** Jaehyuck Jang (PIC / optical interconnect, CPO). Answers should be
concise and technical: conclusion first, then mechanism, then PIC-level
implication. Reference lists only when they add something.

---

## 0. Ground rules

* `ref_dbeme/` and `ref_emepy/` are **vendored upstream, never edited**. If a
  fix belongs upstream, patch the copy under `em_simulation/` and note the
  divergence in `README.md` → *What changed relative to upstream*.
* The **dataset pickles are a cache, not source**. `dataset_info.py` is the
  source of truth. If a grid axis, mesh, window, mode count, wavelength or
  material index changes, the pickles are invalid — delete and regenerate.
  Never hand-edit a `.pkl`.
* **The solve grid is sacred.** Overlap integrals are only meaningful between
  cross sections sampled on the same `(x, y)` mesh. Any change to `window` or
  `MESH` invalidates every overlap in that dataset. `EmepyFDE` raises if the
  grid moves — do not work around that check, regenerate instead.
* New mode solvers go behind `em_simulation/fde/base.py::FDEBackend`. That ABC
  is the whole point of the port; do not let solver specifics leak into
  `data_updater` or the geometry classes.
* Debugging style: identify the root cause, then propose the **minimal patch**.
  Do not rewrite a module unless asked.

## 1. Commands

```bash
.venv/Scripts/python -m pytest tests -q            # invariants — must stay green
cd examples
python demo_linear_taper.py                        # figures -> examples/output/
python demo_bezier_bend.py
python validate_against_direct_eme.py              # DBEME vs uncached EME
```

Cold-cache runs solve cross sections (~1.3 s each at `MESH=160`); warm runs do
no mode solving. Always report both timings when claiming a speed-up.

**Long runs go through `examples/run_solver_job.py <script> [args]`.** This
machine is a hybrid i9-13900HX, and a detached single-threaded solver job
(the FEM eigensolve) gets scheduled onto its E-cores at ~7× the time — a
FEM point took 420 s in the run against 60 s in a shell-spawned process,
with the same session, priority, affinity and environment. The launcher
pins the process to the P-cores (logical 0–15), switches power throttling
off, raises the priority class and caps BLAS at 16 threads. If a run's
per-point time is far above the standalone solve, compare the main thread's
CPU time with the wall time before suspecting the code.

## 2. Units and conventions

| quantity | unit in code | note |
|---|---|---|
| length, width, gap, thickness | **metres** | `0.5e-6`, not `0.5` |
| curvature `κ` | **1/m**, signed | `κ = 1/R`; sign matters, see §5.9 |
| wavelength | metres | dataset is single-λ today |
| `neff` array | `(2N,)` complex | forward modes first, then backward |
| overlap | `(2N, 2N)` | keyed `point -> {adjacent point -> matrix}` |

Plot conventions for reports: transmission in dB and linear, phase unwrapped,
group delay in fs, lengths in µm, `n_g` annotated.

---

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
| `Si_ring_coupler_220nm_<1530|1550|1570>[_N6..N30|_c20]` | straight bus + moving ring core, bus fixed (`platforms.BusRingStrips`) | `w1`, `w2`, `gap` (10 nm cell, axis tied to the cell) | bus-ring point coupler of the all-pass ring | **shipped** - `reports/15`; ring side of the coupler is imposed by symmetry, see tasks/11 section 2.1 |

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

---

## 4. Where the current simulator stands

Validated (see `tests/`, and README → *Accuracy and limitations*):

* backend reproduces emepy's rectangle solver to 1e-3
* self-overlap ≈ identity to ~2 %; adjacent-point overlap > 0.9
* constant-width "taper" → T = 1.000000, zero reflection
* truncation non-unitarity ~1.3e-4 per interface, ~0.5 % over ~40 sections

Not yet established — see §5. **No demo report may be written until that
demo's prerequisite checks pass.**

---

## 5. Simulator validation backlog

Ordered. Each item states a **pass criterion**. Add each as a test under
`tests/` where it can be automated.

### Tier 1 — correctness invariants (do these before *any* new device)

**5.1 Reciprocity.** For a lossless reciprocal structure, `S = Sᵀ` in a
power-normalised basis. This is *independent* of unitarity and is the check
that catches a wrong conjugation convention in the overlap integral — which
unitarity alone will not reveal.
*Pass:* `max|S − Sᵀ| < 1e-6` on the linear taper and on a Bezier bend, with
`force_unitary=False`.
*On a lossy, truncated basis the number is 1e-3, and that is not a failure.*
Per interface the transmission blocks are symmetric by construction, but the
reflection block `R12 = ½(O_abᵀ − O_ba) T12` is symmetric only to the extent
that the two projections of the interface agree, and for the discretised
continuum they differ by ~0.2 (a Berenger mode of one section is represented
on the other side by a different set); the physical block stays reciprocal to
1e-3 per interface and the cascade sums to ~1e-3 on the launched channel
(report 13 §11, `examples/kocabas_reciprocity.py`). Not a gauge, not the
cutoff, not the projection side, and it does not fall with basis size. For a
lossy basis the pass criterion is `< 1e-2` relative to the transmission; a
value of order 1 is the conjugation error this check exists for.

**5.2 Mode-basis convergence with `force_unitary=False`.** The demos currently
run `force_unitary=True`, which projects each section onto the nearest unitary
matrix and therefore **hides** truncation error rather than measuring it. Run
the taper at `mode_numbers = 4, 6, 8, 10, 12` with the projection off.
*Pass:* total power → 1 monotonically, and the residual `1 − ΣT` falls with
`N`. If it plateaus or rises, there is a bug, not a truncation floor.
*Which side the interface is projected on matters once the basis is
incomplete.* Upstream imposes continuity against the modes of the section
being *left* (`T12 = 2 inv(O_ab + O_baᵀ)`), whose single-mode limit is
`1/|O|²` — the unrepresented mismatch comes back as **gain**. The default is
now the section being *entered* (`SingleEME.INTERFACE_PROJECTION = "output"`,
`T12 = 2 O_abᵀ inv(O_abᵀ + O_ba) O_ba`, limit `|O|²`: what the basis cannot
hold is lost). On a six-mode Si taper the two differ by 2.5e-4 per interface;
on the plasmonic taper they are ×1.44 against 0.72 over forty interfaces
(`reports/12`, report 06 §11). Both routes share the method, so Stage 4 holds
either way.

*Policy:* `force_unitary=True` is only defensible while the model is lossless
by construction. It must be **off** for anything reporting bend loss,
radiation, or metal loss. Never use `force_passive=True` alone (README
documents it over-attenuating to 0.63 on a unit-transmission taper).

**5.3 Grid-step scaling of the conversion floor.** The ~1e-4 plateau in
`taper_4_length_sweep.png` is claimed to be the 20 nm width-staircase floor.
Prove it: rebuild the `top_width` axis at 20 / 10 / 5 nm.
*Pass:* the floor scales ≈ `(Δw)²` (each step is a small perturbation, so
scattered amplitude ∝ Δw, power ∝ Δw²). If it scales as `Δw¹` or not at all,
the floor is numerical, not physical, and every adiabatic-limit number in the
reports is meaningless.

**5.4 Slicing independence.** The same physical device sliced with different
`delta_z` must give the same S-matrix (up to the grid snap).
*Pass:* `max|ΔT| < 1e-4` between two slicings differing by 2×.

**5.5 Round-trip / reversal.** (a) Taper 0.5→1.2 µm then 1.2→0.5 µm returns
to TE0 with conversion ≈ the incoherent sum of the two stages. (b) Running a
device backwards gives the transposed S.
*Pass:* both to 1e-4.

**5.6 Mode gauge — already handled; the residual risk is degeneracy.**
*Audited, and an earlier note in this file was wrong: it is fixed.*
`emepy_fde._pin_gauge` pins each mode's global phase deterministically inside
`solve()`, before `assemble` ever sees the fields. The mechanism it protects
against is worth understanding, because it is the reason the dataset is
allowed to persist overlaps without persisting fields:

* EMpy calls ARPACK (`scipy.sparse.linalg.eigs`, `EMpy_gpu/modesolvers/FD.py`
  L994) with **no `v0`**, so the starting vector is random and the same cross
  section can return a mode with the opposite sign on two runs.
* Power normalisation cannot remove that: `P = ½∫(E×H)_z` is **quadratic** in
  the eigenvector, so `P(−ψ) = P(ψ)` and dividing by `√P` rescales without
  ever flipping. The conjugated form is quadratic too, so it is equally blind.
* Precisely: because the normalisation is *unconjugated*, `ψ → e^{iθ}ψ` sends
  `P → e^{2iθ}P`, so fixing `P = +1` already kills every phase except
  `θ = 0, π`. The gauge group collapses `ℂ* → ℤ₂ = {±1}` — which is why the
  residual freedom is a sign and not a general phase, and why it shows up as
  the `±` branch of the complex `sqrt` in `normalize_field`.
* The overlap is **linear** in each mode, so a flip of mode `i` at point `a`
  flips row `i` of `O(a,b)`.
* Within one consistently built chain it cancels — a gauge `G_k = diag(±1)`
  enters as `O(k−1,k) G_k` and `G_k O(k,k+1)`, and `G_k² = I`. The
  cancellation needs *the same* `G_k` on both sides, i.e. every overlap
  touching point `k` must come from one field array. Since only overlaps are
  persisted, a later re-solve of `P` in a fresh gauge would leave `O(A,P)`
  (pickle) and `O(P,B)` (fresh) inconsistent. Hence `_pin_gauge`.

*Which stored files carry gauge:* `neff.pkl` (eigenvalue) and `TE_pol.pkl`
(ratio of quadratics) are gauge-**invariant** and structurally immune.
`overlap.pkl` is gauge-**covariant**, `O(a,b) → G_a O(a,b) G_b`, and the
covariance cancels in the cascade only if every stored overlap touching a
given grid point used the same gauge for that point.

*Regression guard:* `tests/test_gauge_reproducibility.py`.

**The near-degenerate subspace — FIXED 2026-09-03.** `_pin_gauge` fixes one
global phase **per mode** and cannot fix a **degenerate or near-degenerate
subspace**: the freedom left after normalisation is the group preserving the
bilinear form, `Mᵀ M = I`, i.e. complex orthogonal `O(m, ℂ)`. For a
non-degenerate mode `m = 1` and that is just `{±1}`, which `_pin_gauge` fully
fixes; for an `m`-fold degeneracy it is *continuous*. ARPACK then returns an
arbitrary rotation within the subspace.

This was predicted to bite at "the coupler anti-crossing", and it did. On the
coupled Si pair, two modes at `Δn = 0.0101` had a **self**-overlap
`|⟨E₄,H₅⟩| = 0.0745`, so `O_aa ≠ I` and a **constant-width** guide — where the
answer must be `T = 1` exactly — transmitted **1.59**.

*Fixed in* `em_simulation/fde/assemble.py::biorthogonalise`: Löwdin symmetric
orthogonalisation in the **unconjugated** metric (§5.16), applied per cross
section inside `assemble`, so every backend and dataset gets it. `A = S^{-1/2}`
with `S = ½(M + Mᵀ)` — the symmetric part, because that is what the interface
formula `O + Oᵀ` consumes. Löwdin because it is the *minimal* change, so modes
keep their character and their `n_eff` labels stay meaningful.

Two details that matter: `_inverse_sqrt` takes the **real** branch when `S` is
real, because a complex mixing matrix would break the reality structure that
makes the two backward-basis constructions coincide (§5.16a); and a
non-positive eigenvalue of `S` means two solver outputs span the same field, so
no biorthogonal basis exists and it raises rather than papering over.

*Measured:* constant-width `T` 1.59 → 1.0000 ± 2e-4; per-interface
`|SᴴS − I|` 1.5e-1 → 2.1e-3; on the SIRAC taper at N = 6, max 6.4e-1 → 9.6e-3.
*Regression:* `tests/test_mode_basis.py`. *Dataset impact:* stored overlaps
change meaning, so `dataset_identity.BASIS_CONVENTION = 2` invalidates every
pre-fix cache — see `reports/07_sirac_optimization.md` §8.3, §8.6.

**Also open for a lossy backend.** `_pin_gauge` rotates so the transverse
field is real at its peak, then fixes the sign from the first raster point
above half-max. For a strongly complex (leaky, metallic) mode the field is not
globally real after that rotation, so the half-max test is taken on a real
part that may be near zero and the sign becomes fragile. Revisit with §5.16a.

**5.7 Mesh and window convergence of the *overlap*, not just `n_eff`.**
`n_eff` is converged to ~0.002 at `MESH=160`, but overlap integrals converge
more slowly than eigenvalues, and the overlap is what the method rests on.
*Pass:* `max|ΔO| < 1e-3` between `MESH=160` and `MESH=240`, and between the
shipped window and a 2× wider one. Check at the *narrowest* width (0.4 µm),
where the mode is weakest and window-sensitive — the TM mode there is barely
guided.

**5.8 DBEME vs. direct EME, extended.** `validate_against_direct_eme.py`
currently covers the taper. Extend to: a Bezier bend, an S-bend (the case the
curvature-sign fix was written for), and a multimode launch (TE1 in).
*Pass:* `max|ΔT| < 1e-3` per port; record cold/warm timing for the speed-up
claim.

### Tier 2 — physics gaps that block specific demos

**5.9 No PML → no radiation loss.** `MSEMpy` (emepy 1.2.2) has no working PML,
so `n_eff` is real and bend/substrate/metal loss is not captured. Consequences
that must be stated in every report:
* absolute insertion loss **cannot** be compared against literature;
* only ratios among guided modes (splitting ratio, conversion efficiency,
  crosstalk) are comparable;
* a tight bend shows correct guided-mode conversion but **zero** bend loss.
*Audit of how dead the PML actually is (2026-08-31).* The README's claim is
correct at the emepy level but understates what is available underneath:

* **Dead as shipped.** `MSEMpy(PML=...)` is documented "Only works for Tidy3D,
  not EMpy" (`emepy/fd.py:103`); the MSEMpy branch is commented out
  (`fd.py:208`) with `nlayers` hardcoded to `[1,0,1,0]`; the 1-D solver that
  did carry a stretchmesh PML recipe is commented out wholesale
  (`fd.py:396-402`), including the intended `factor = 1 + 2j`.
* **But the substrate is present.** `EMpy_gpu.modesolvers.FD.stretchmesh`
  implements complex coordinate stretching and says so in its docstring
  (`xx = x.astype(complex)`). `VFDModeSolver` builds its operator purely from
  `dx = diff(x)` / `dy = diff(y)` via the n/s/e/w spacings, and contains **no**
  real-casts anywhere in the class body (L279–1040) — so complex coordinates
  flow straight through to a complex operator and complex `n_eff`. `MSEMpy`
  accepts explicit `x=`/`y=`, which `_build_solver` already passes.

So wiring PML in is small (~100 lines plus an `MSEMpy` subclass to bypass the
hardcoded trim). **The wiring is not the work.** What follows it is:

1. spurious/PML mode filtering — the spectrum fills with Bérenger modes and
   `sort(key=-Re(neff))` will happily return them; needs a core-confinement
   filter and a shift-invert target;
2. the overlap integral must use the **stretched metric** (complex `dx, dy`) or
   biorthogonality breaks and `O(a,a) = I` stops holding;
3. the backward-mode construction (§5.16a) and the `Im(n_eff)` clipping
   (§5.16b) must both be fixed first;
4. `_pin_gauge` gets fragile for strongly complex modes (§5.6);
5. **mode count.** PML-EME converges *because* the discretised continuum makes
   the basis approximately complete — that needs ~20–50 modes, not 6. DBEME
   stores `(2N × 2N)` overlaps per adjacent pair, so N = 6 → 40 is a **~44×**
   blow-up in overlap storage and solve time. This is the DBEME-specific cost
   and it is the one to think hardest about before committing;
6. PML parameter convergence — reported loss must be swept against PML
   thickness, stretch factor and standoff distance before it means anything.

*Acceptance test when this is attempted:* the **bent slab with the exact Airy
solution** — 1-D, semi-analytic, cheap, and it probes the radiating tail
directly, which is precisely what a PML must get right. Then a published SOI
bend-loss-vs-radius curve, then FDTD.

*Alternative — built 2026-09-17:* `em_simulation/fde/femwell_fde.py::FemwellBackend`
(femwell 0.1.12 / scikit-fem / gmsh), a boundary-conforming FEM mesh behind
the same `FDEBackend` contract. Geometry comes from `CrossSection.polygons()`
(shapely, per material region; implemented for `FullEtchStrip` and
`PlasmonicSlotConverter`, whose rounded corners are true arcs); radiation is
absorbed in a lossy outer ring rather than a stretched-coordinate PML; fields
are point-evaluated onto the dataset's uniform grid (every point located
once per solve by a trapezoid map, two probe matrices applied to all modes,
points on element edges averaged over the sharing elements — skfem's own
`probes` is quadratic per batch and cost more than the eigensolve), which
is the one new term in the error budget and what
`tests/test_femwell_backend.py` pins against the FD backend on a dielectric
strip (`|O₀₀| > 0.99` between the two solvers' fields). Built because the FD
grid cannot converge a rounded metal corner at any affordable cell (report
13 §7–§8); on the Kocabaş platform `kocabas_converter_dataset_info(solver="femwell")`,
with both parameter axes at the cell since nothing has to align. Note this
**coexists** with `EmepyFDE` rather than replacing it — `get_fde_backend()` is per dataset, so lossless datasets keep
the fast uniform-grid EMpy path. Backends cannot be mixed *within* one dataset,
because the overlaps require one shared grid and femwell's unstructured mesh
must be interpolated onto it (that interpolation error then enters every
overlap).

*Status: built (2026-09-02 → 09-06), Route A.* `em_simulation/fde/pml.py` —
`PMLModeSolver` / `PMLBackend` on EMpy's vectorial solver with a complex
stretched grid, shift-invert selection and a core-confinement filter; validated
against the Airy bent slab (3 % over seven decades) and Vlasov & McNab 2004
(`reports/06`). Items 1–4 above are done; item 5 (mode count) is answered by
the scattering-route cascade (`SMATRIX_METHOD = "auto"`, report 06 §7–8):
usable at N = 4…40 on the direct route. Item 6 remains per device.
**One more thing had to be got right:** EMpy's `stretchmesh` takes `abs(Im)`
of the stretched coordinate, which makes the `-x`/`-y` layers *gain*; only a
`+x`-only validation cannot see it. `stretched_grid` applies the stretch
itself (report 06 §10). Lossless datasets keep the fast `EmepyFDE` path —
`get_fde_backend()` is per dataset.

**5.10 Curvature limit from the conformal transform.** `n_eq = n·exp(κx)`
means the cladding at the window edge ramps to `n_clad·exp(κ·half_width)`.
With `n_clad = 1.444`, `half_width = 1.6 µm`, modes above ~2.1 require
`κ < 2.3e5 1/m` (R > 4.3 µm). A ring coupler at R = 5–10 µm sits right at that
edge. *Check:* confirm `EmepyFDE` actually warns, and consider shifting the
window centre outward with κ so the mode stays centred.

**5.11 Vertical symmetry forbids polarization rotation.** `FullEtchStrip`
defaults to `substrate_index = clad_index`, i.e. a symmetric buried guide with
a horizontal mirror plane. Such a cross section has **exactly zero** TE↔TM
coupling by symmetry — a polarization rotator built on it will return 0 %
conversion and the result will look like a bug. Demo 2 needs a genuinely
asymmetric stack: partial etch (rib/bi-level), or asymmetric cladding, or an
angled sidewall.

**5.12 Single-core cross sections only.** `FullEtchStrip` has one core;
`CompositeGeometry` / `MultiEME` cascade sections **along z**, they do not
place waveguides side by side. Demos 1–3 all need a new `CrossSection`
subclass with two cores and a `gap` parameter, plus a `Geometry` subclass that
emits paths in that higher-dimensional space. This is the single largest piece
of work in the plan — do it once, correctly, and all three coupler demos fall
out of it.

**5.13 Mode tracking through anti-crossings — FIXED 2026-09-03.** FDE returns
modes sorted by `n_eff`, so labels swap at an anti-crossing — precisely the
operating point of an adiabatic coupler, RAC and PSR. Tracking must be by
**maximum overlap with the previous grid point** (Hungarian assignment), not by
`n_eff` order.

The code did maximum overlap but **not the assignment**: `Geometry` and
`DirectGeometry` each held an identical per-column `argmax` with a 0.5
threshold. Nothing enforced a bijection, so at an anti-crossing two branches
could claim the same predecessor while another went unclaimed; and a branch
below the threshold was declared a *new* mode, overflowing the `num_modes`
slots and leaving one at its initialised **zero**. A mode with `n_eff = 0` has
no field behind it, so its overlap rows are junk and the interface matrix goes
singular.

*Fixed in* `em_simulation/geometry/mode_tracking.py::hungarian_mode_links`
(`scipy.optimize.linear_sum_assignment`); both classes delegate to it. With a
fixed `N`-mode basis on both sides of every interface the map *must* be a
permutation, so there is no case where greedy is right and this is not. The
threshold is dropped deliberately — a low best-overlap means tracking is
*uncertain*, not that a mode appeared; `link_quality` reports that instead.

*Regression:* `tests/test_mode_basis.py`. Found by
`reports/07_sirac_optimization.md` §8.2.

**5.13a Shift-invert target on a lossy dataset — put it among the physical
branches.** `PMLBackend` returns the `N` eigenvalues nearest to `target²`.
The Berenger band of a PML basis sits at `Re n ≈ 1.2–1.45`, `Im ≈ 0.2–0.35`,
i.e. `|n²| ≈ 1.5–2`, and there are dozens of them. A target *below* the
physical branches (1.9 on the Kocabaş converter, whose wire mode is 2.44)
makes the Berenger modes nearer than the fundamental, which then drops out of
the set at some widths; the tracker links what is left, and the launched
power rode a higher-order branch into cutoff — 99 % lost. Place the target so
that every physical branch of the path beats the Berenger band in `|n² −
target²|` (2.3 there: wire 1.2, slot 2.7, Berenger ≈ 3.2), and check the
stored, *unordered* sets along the path for missing members before trusting
a cascade (`reports/13` §1). *The window's bottom edge bites too:* the
`N`-th continuum member can change between two adjacent points (best
overlap 0.30), the Hungarian bijection then links it to a stranger, and the
interface matrix gains a near-null direction — 3.2× the power of a physical
input at one interface of the 1 nm-wall FEM gap sweep, which the cascade
reported as 3.6 % reflection and a negative deficit. The scattering route's
`SingleEME.INTERFACE_RCOND` (1e-2 since 2026-09-20) discards such
directions; a path whose budget shows reflection or a negative deficit out
of line with its neighbours should be checked interface by interface
(`max` over physical inputs of the column power of each interface
S-matrix) before its number is used.

**5.13b E and H on one grid.** `compute_other_fields` returns E on the
cell centres and H on the nodes; `PMLModeSolver` zero-padded E to the node
shape, a half-cell misregistration that makes every unconjugated overlap
non-symmetric and every interface matrix gain or lose `|anti(M)|²`
(constant-width guide 1.07; report 07 §23, the SIRAC session's finding).
`PMLModeSolver(colocate=True)` interpolates E onto the nodes as `MSEMpy`
does; it is off by default in the solver and **on in both plasmonic
platforms**, and it enters the dataset identity. Any PML dataset built
before 2026-09-07 carries the defect.

**5.14 Single wavelength.** The dataset is fixed at 1550 nm. Demos 2 and 3
both make **bandwidth** claims (Fargas: 50±1 % over 200 nm, S+C+L). Add a λ
axis or, simpler, one dataset per λ plus a sweep helper. Jae's own work is
O-band, so also build a 1310 nm dataset.

**5.15 Material dispersion.** `CORE_INDEX = 3.4757` is a constant. Over a
200 nm span, Si and SiO2 dispersion is not negligible and neither is waveguide
dispersion. Any bandwidth plot made with a constant index is wrong.
*Fix:* Sellmeier `n_Si(λ)` (Li 1980) and `n_SiO2(λ)` (Malitson 1965) in the
cross section before the first bandwidth report.

**5.16 Overlap convention — audited, already correct.**
`assemble.overlap_matrix` computes `O_ij = ½∫(E_a,i × H_b,j)_z dA` with **no
conjugation**, which is the correct reciprocal (unconjugated) form and stays
valid for lossy, non-Hermitian problems. Nothing to fix here. Two *other*
lossless assumptions sit right next to it, and both do block demo 4:

**5.16a Backward-mode construction assumes a lossless medium.** `assemble`
builds the backward basis as `E₋ = conj(E₊)`, `H₋ = −conj(H₊)`,
`n_eff → −n_eff`. That is time reversal, and it is only valid for a real
index — conjugating a lossy mode turns loss into **gain**. For metal the
backward partner of a reciprocal medium is `(E_t, −E_z, −H_t, H_z)` with
`β → −β` and **no** conjugation. *Fixed (Phase 0):* `assemble(...,
lossless=)` builds the reciprocal partner for a lossy backend and the
conjugate one otherwise; the two coincide for real fields
(`tests/test_lossy_assumptions.py`).

**5.16b `correct_gain_modified` clips `Im(n_eff) < 0` to zero.** Sensible as a
round-off guard in a lossless model; fatal in a lossy one, where it can
silently delete all the loss depending on the solver's sign convention.
*Fixed (Phase 0):* gated on `FDEBackend.lossless`; the project convention is
`e^{iβz}`, so an absorbing mode has `Im(n_eff) > 0` and the PML solver folds
the branch that way. The same lossless-only gating now also applies to the
100 dB/cm rule of the radiation-mode mask (`geometry.radiation_mode_mask`),
which would otherwise reject every plasmonic mode.

### Tier 3 — documentation and hygiene

**5.17** `README.md` contradicts itself: *Defining your own platform* says "the
dataset stores non-negative curvature only", while *What changed* and the
shipped `dataset_info.py` use a symmetric ±2e5 axis (correctly — folding
breaks S-bends). Fix the README paragraph.

**5.18** Record the shipped baseline in a test so regressions are caught:
500 × 220 nm buried Si strip → `n_eff(TE0) = 2.449`, `n_eff(TM0) = 1.788`
at 1550 nm.

---

## 6. Capability matrix vs. the demo list

| demo | 2 cores | asym. stack | λ sweep | complex modes / loss | verdict |
|---|---|---|---|---|---|
| 1. adiabatic coupler | ✗ needed | – | optional | – | after §5.12 |
| 2. polarization rotator | ✗ needed | ✗ needed (§5.11) | ✗ needed | – | after §5.11 + §5.12 + §5.14 |
| 3. rapid adiabatic coupler | ✗ needed | – | ✗ needed (§5.14, §5.15) | – | after §5.12 + §5.14 |
| 4. plasmonic converter | metal | – | – | ✓ `PMLBackend` (§5.9, §5.16a/b done) | **2-D lateral model shipped** — `reports/12`; the 3-D taper still needs a second geometric axis |
| 4b. Si-to-slot converter, SiO₂-embedded | metal, 2 axes | – | – | ✓ `PMLBackend`, `FemwellBackend` | **shipped, converged** — `reports/13`; 72.3 % modal with sharp corners; with 20 nm-rounded corners 82.4 % (FD 5 nm) / 80.7 % (FEM, 5 nm wall) / **85.6 % (FEM, 1 nm wall)**, and **91.6 % under Kocabaş's own measure** (total flux at his cut, §9–§10) vs his ~95 % — the rest is a percent of Si staircase and the sharp-rectangle geometry. Length is the design parameter, the starting gap is not (flat within 1.6 points over 25–150 nm, §4). The first plasmonic device here that is adiabatic (T rises with `L`) |

Build order: **5.12 (coupled pair) → demo 1 → 5.14/5.15 (λ + dispersion) →
demo 3 → 5.11 (bi-level) → demo 2 → new backend → demo 4.**

---

## 7. Demo and report plan

Each demo gets `examples/demo_<name>.py` and a report at
`reports/<nn>_<name>.md`, with figures in `reports/output/`.

### Common comparison protocol

**A. DBEME vs. direct EME.** Same geometry, same mode count, same slicing,
`force_unitary=False` on both.
Report: overlaid transmission curves; `max|ΔT|` per port; cold-cache and
warm-cache wall time, and the warm-cache speed-up on a length/shape sweep
(that ratio is the actual claim of the method).

**B. DBEME vs. literature.** State up front which quantities are comparable.
With no radiation loss in the model (§5.9), **absolute insertion loss is not
comparable**. Compare instead:
* splitting ratio / power imbalance vs. length or wavelength
* mode-conversion efficiency and crosstalk between guided modes
* the *shape* of the adiabaticity roll-off, and the length at which it knees

Digitise the literature figure, overlay it, and give an explicit discrepancy
table with a physical cause for each row (mesh, grid step, missing radiation,
index/stack mismatch, fabrication in the paper's case).

**C. Sanity gate.** Every report opens with a short table of the §5 checks
that were run and their measured values. A report without that table is not
finished.

### Demo 1 — Adiabatic coupler

**⚠ Reference needs confirming.** The URL given for demo 1 is the same as for
demo 2 (`oe-22-4-3777`), which is the Sacher PSR paper. Two readings:
(a) it was a copy-paste slip and a different adiabatic-coupler paper was
intended; (b) it means the **asymmetric adiabatic coupler stage** inside that
PSR, which splits TE0/TE1 into two guides. Assume (b) and build the TE1/TE0
adiabatic splitter unless told otherwise — but ask.

Physics targets: TE0 stays in the wide guide, TE1 crosses to the narrow one;
adiabaticity set by `|⟨ψ₂|∂ψ₁/∂z⟩| ≪ |β₁ − β₂|` (Sun/Liu/Yariv criterion).
Deliverables: crossover vs. length; the anti-crossing `n_eff` diagram with
tracked branches (also serves as the §5.13 evidence); sensitivity to `gap`.

### Demo 2 — Polarization rotator (bi-level taper PSR)

Reference: W. D. Sacher, T. Barwicz, B. J. F. Taylor, J. K. S. Poon,
*Polarization rotator-splitters in standard active silicon photonics
platforms*, **Opt. Express 22(4), 3777 (2014)**, doi:10.1364/OE.22.003777.
(Confirm the exact stack from the paper before building the cross section.)

Mechanism: the bi-level (partially etched) taper breaks vertical symmetry, so
TM0 evolves adiabatically into TE1; a following asymmetric coupler splits TE1
off. **Prerequisite §5.11** — a vertically symmetric cross section gives
identically zero rotation. Also needs TM in the mode basis and enough modes to
resolve the TM0/TE1 anti-crossing.
Deliverables: TM0→TE1 conversion vs. taper length; PER; conversion vs. λ over
C-band; and the `n_eff` anti-crossing with TE fraction (`TE_pol`) colour-coded
— that plot is the whole mechanism in one figure.

### Demo 3 — Rapid adiabatic coupler

Reference: J. M. Fargas Cabanillas, PhD thesis, Boston University (Popović
group) — local copy at `references/FargasCabanillas_bu_0017E_17256.pdf`.
Related: *Rapid Adiabatic 3 dB Coupler with 50±1 % Splitting Over 200 nm
including S, C and L Bands in 45 nm CMOS Platform*, FiO 2021, FTu6B.2.

Note the platform is GF 45 nm monolithic electronic-photonic SOI, **not** a
220 nm strip — thickness, cladding and etch differ. Either rebuild the dataset
for that stack or state clearly that the comparison is qualitative (shape and
bandwidth trend, not absolute numbers).
**Prerequisites §5.12, §5.14, §5.15** — the headline claim is a 200 nm
bandwidth, so a constant refractive index invalidates the comparison.
Deliverables: splitting ratio vs. λ over 200 nm with the ±1 % band drawn;
splitting vs. device length; imbalance vs. gap and width tolerance.

### Demo 4 — Plasmonic mode converter

Reference: M. Ono, H. Taniyama, E. Kuramochi, K. Nozaki, M. Notomi, *Toward
Application of Plasmonic Waveguides to Optical Devices*, NTT Technical Review
**16**(7), 2018. Device: Si wire 400 × 200 nm → Au/air MIM 50 × 20 nm, 600 nm
3-D taper, 40 nm air gap, ≈ −1.7 dB measured at 1.55 µm.

**Status (2026-09-06): the 2-D lateral stepping stone is done** —
`examples/demo_plasmonic_converter.py`, `reports/12_plasmonic_converter.md`,
datasets `Si_plasmonic_slot_1550[_gap40|_c4]`. Of the four blockers below,
1–3 are closed by the PML backend and Phase 0; 4 is what the report measures:
the shed field of a 10 nm edge step is in no shift-invert basis, so the
staircase mismatch is length-independent and the model has no adiabatic
optimum — and with upstream's interface projection that unrepresented field
came back as gain (§5.2, report 06 §11). Gold corners are rounded (20 nm)
on the shipped platform — Jae's point that a sharp plasmonic edge carries a
singular, unresolvable field; the sharp variant is `Si_plasmonic_slot_1550_sharp`
(report 12 §8).
What the lateral model *cannot* do, and why its numbers are not the paper's:
the device binds its plasmon vertically (the 20 nm), so the lateral slot must be
suspended in air and walled by full-height gold to have a bound output mode at
all; over oxide the 40 nm lateral slot's mode sits at 1.18 + 0.058j, below the
substrate, and leaks. The grid is snapped so every metal edge sits on a cell boundary
(a metal edge inside a cell aliases `n_eff` by 0.1); between the 5 and 4 nm
exact grids the Si-end index moves 0.2 % and the slot's 3 % (report 12 §1).

The blockers as originally written:
1. Au needs complex ε; the current backend returns real `n_eff` only (§5.9).
2. SPP and leaky modes need PML; without it the basis is wrong, not just
   truncated.
3. The overlap form is already correct (§5.16), but the **backward-mode
   construction and the `Im(n_eff)` clipping are not** (§5.16a, §5.16b) — both
   assume a real index and must be fixed first.
4. A 600 nm taper between a 400 × 200 nm and a 50 × 20 nm core is far from
   adiabatic and deeply sub-wavelength — a truncated guided-mode basis will
   not converge. EME here needs a large radiation-mode basis, and FDTD is the
   honest reference.

Recommended scope: implement a `FemwellFDE` (or Lumerical) backend, do the
**2-D lateral** converter first as a tractable stepping stone, validate against
FDTD rather than against the paper's measured −1.7 dB, and treat the full 3-D
taper as a stretch goal. The 3-D taper also needs a second geometric axis
(height as well as width), so the cross section becomes `(w, h, gap)`.

---

## 8. Report template

```markdown
# <nn> — <device>

## 1. Sanity gate
| check (§5.x) | criterion | measured | pass |

## 2. Device and dataset
geometry, parameter path, dataset used, grid steps, mode count, λ

## 3. DBEME result
figures + the physical reading of them

## 4. DBEME vs. direct EME
overlay, max|ΔT|, cold/warm timing, speed-up on a sweep

## 5. DBEME vs. literature
what is comparable and what is not; overlay; discrepancy table with causes

## 6. Conclusions and limits
what this result does and does not support
```

---

## 9. References

Method and solver

* J. Song and B.-H. Sohn, *Opt. Express* **33**, 46815–46827 (2025),
  doi:10.1364/OE.567425 — the DBEME method; cite this if the method is used.
* D. Gallagher and T. Felici, *Eigenmode expansion methods for simulation of
  optical propagation in photonics*, Proc. SPIE **4987** (2003) — EME
  formulation, interface matrices.
* P. Bienstman and R. Baets, *Opt. Quantum Electron.* **33**, 327 (2001) —
  PML, complex/leaky modes, why the unconjugated overlap.
* A. W. Snyder and J. D. Love, *Optical Waveguide Theory*, ch. 31 —
  orthogonality in lossy / non-Hermitian waveguides.
* M. Heiblum and J. H. Harris, *IEEE J. Quantum Electron.* **11**, 75 (1975) —
  conformal transformation for bend modes.

Devices

* X. Sun, H.-C. Liu, A. Yariv, *Opt. Lett.* **34**, 280 (2009) — adiabaticity
  criterion and the shortest adiabatic mode transformer. The design rule for
  demos 1 and 3.
* W. D. Sacher *et al.*, *Opt. Express* **22**, 3777 (2014) — bi-level taper
  PSR. Demos 1 (splitter stage) and 2.
* J. M. Fargas Cabanillas, PhD thesis, Boston University — RAC; local copy in
  `references/`. Plus FiO 2021 FTu6B.2 and CLEO 2021 SW4E.2.
* M. Ono *et al.*, *NTT Technical Review* **16**(7) (2018) — plasmonic MIM
  mode converter. Demo 4.

Material data

* H. H. Li, *J. Phys. Chem. Ref. Data* **9**, 561 (1980) — Si dispersion.
* I. H. Malitson, *J. Opt. Soc. Am.* **55**, 1205 (1965) — SiO2 dispersion.
