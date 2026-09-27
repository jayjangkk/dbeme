# Simulator validation: state and backlog

*Moved verbatim from `CLAUDE.md` by task 16 step 0.7 (2026-09-27). Section numbers are unchanged, so a citation of the section number in older text still finds it here.*

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
*On a lossy basis the interface columns are capped at unit power instead*
(`SingleEME.INTERFACE_COLUMN_CAP = "auto"`, 2026-09-23): the projection is
not passive for the discretised continuum (5–16 % gain on a continuum
input, 83 % of all columns slightly above 1), and a cascade of a few hundred
interfaces compounds it until the guided channel itself reports more power
than launched (6.9 for unit input at 311 interfaces; 0.91 at 231). The cap
is the weakest passivity statement in the basis's own measure; it moves the
guided channel by 0.3 points and makes long cascades usable (report 13
§12). A singular-value clip is *not* equivalent — the 2-norm is not power
in an unconjugated-normalised basis — and cut the guided channel in half.

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
