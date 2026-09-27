# Task 02 — a PML backend: complex modes and radiation loss

**Prompt to start with:**

> Read `CLAUDE.md`, then `tasks/02_pml_backend.md`. Do Phase 0 and Phase 1 only.
> Phase 1 builds the *reference*, not the solver — do not touch any backend
> until the Airy validator reproduces analytic bend loss. Report the
> convergence table at the Phase 1 gate.

Conventions are in `docs/validation_backlog.md`; §5.x refers to its validation backlog.
Prerequisite: none. This is independent of `tasks/01_coupled_pair.md` and can
run in either order — but 01 is the higher priority, because no demo's primary
figure of merit needs radiation loss.

---

## Why, and why not yet

Today `n_eff` is real. That means **zero** bend loss, zero substrate leakage,
zero metal loss. Consequences already documented in §5.9: absolute insertion
loss can never be compared against literature, and only ratios among guided
modes are meaningful.

Where it actually bites:

| device | needs loss? |
|---|---|
| taper, Bezier bend, adiabatic coupler, RAC | **no** — conversion and splitting ratios are guided-mode ratios |
| pulley coupler | **partly** — `κ` and phase matching are fine; the narrow outer bus is the leakiest guide in the structure and its radiation is often what kills the design |
| tight ring bends, Q budget | **yes** |
| plasmonic converter (demo 4) | **yes, and this alone is not sufficient** — see §5.16a/b |

So this task is worth doing, and worth doing *after* task 01.

---

## Audit: how dead is the PML, really

Done 2026-08-31. The README is right that emepy has no working PML, but that
understates what sits underneath.

**Dead as shipped:**

* `MSEMpy(PML=...)` is documented *"Only works for Tidy3D, not EMpy"* —
  `emepy/fd.py:103`.
* The MSEMpy PML branch is commented out (`fd.py:208`), `nlayers` hardcoded to
  `[1,0,1,0]`.
* The 1-D solver that carried a real stretchmesh PML recipe is commented out
  wholesale (`fd.py:396-402`) — including the intended `factor = 1 + 2j`.

**But the substrate is fully present:**

* `EMpy_gpu.modesolvers.FD.stretchmesh` implements complex coordinate
  stretching and says so in its docstring; it opens with
  `xx = x.astype(complex)`.
* `VFDModeSolver` builds its operator purely from `dx = diff(x)` /
  `dy = diff(y)` via the n/s/e/w cell spacings, and contains **no** real-casts
  anywhere in the class body (L279–1040). Complex coordinates therefore flow
  straight through to a complex-symmetric operator and complex `n_eff`.
* `MSEMpy` accepts explicit `x=` / `y=`, which `EmepyFDE._build_solver`
  already passes.

**Conclusion: the wiring is ~100 lines. The wiring is not the work.**

---

## Status (2026-09-07): closed

Phases 0–3 done (report 06); first real use done on the plasmonic converter
(report 12) with a lossy dataset, DBEME vs direct EME and DBEME vs literature.
What the backend is: `PMLModeSolver` / `PMLBackend` (`em_simulation/fde/pml.py`),
complex `n_eff`, PML recipe in the dataset identity, scattering cascade by
`SMATRIX_METHOD = "auto"`, output-side interface projection by
`INTERFACE_PROJECTION = "auto"`. What it is not: a basis that holds the near
field a metal edge sheds (report 12 §3, §7.5) — a fixed-grid staircase cannot
show an adiabatic optimum for a plasmonic taper, and the fixes are listed in
order of cost in report 12 §6. The pulley bus of Phase 4 remains open.

## Phase 0 — remove the lossless assumptions (safe to do now)

Four places hard-code "the medium is real". All are no-ops on a real-index
dataset, so this phase is safe and independently testable *before* any complex
mode exists.

**0.1 Backward-mode construction (§5.16a).** `fde/assemble.py` builds
`E₋ = conj(E₊)`, `H₋ = −conj(H₊)`. That is time reversal, valid only for a
real index — conjugating a lossy mode turns loss into **gain**. For a
reciprocal lossy medium the backward partner is

```
(Ex, Ey, −Ez,  −Hx, −Hy, Hz)   with   β → −β,   no conjugation
```

Switch on a `lossless` flag carried by the backend, not on `Im(neff)` at
runtime.

**0.2 `correct_gain_modified` (§5.16b).** It clips `Im(n_eff) < 0` to zero.
Correct as a round-off guard in a lossless model; it silently deletes *all*
loss in a lossy one, depending on the solver's sign convention. Gate it behind
the same flag, and **first verify which sign the backend uses for a lossy
mode** — get this backwards and everything amplifies.

**0.3 Overlap metric.** `overlap_calculation_tool.compute_differences` is
called as `np.asarray(x, dtype=float)` in `assemble.overlap_matrix`. Under
complex coordinate stretching the integration measure is complex, and the
overlaps must use the **stretched metric** or biorthogonality breaks and
`O(a,a) = I` stops holding. Make the spacing path complex-safe. This is the
change most likely to be forgotten and hardest to debug afterwards, because
the symptom is a mildly non-unitary S-matrix that looks like truncation error.

**0.4 `force_unitary` guard.** It projects each section onto the nearest
unitary matrix — legitimate only while the structure is lossless by
construction. Raise if `force_unitary=True` and any `Im(n_eff) ≠ 0`. Without
this guard the first lossy run will silently report zero loss and look fine.

**Phase 0 gate:** every existing test in `tests/` still green, unchanged.
These are no-ops on `Si_fulletch_220nm`. If anything moves, the change is wrong.

---

## Phase 1 — build the ruler before the thing being measured

**Do not write a backend in this phase.** Bend loss out of a mode solver is
notoriously sensitive to PML thickness, stretch factor and standoff distance;
without an independent reference there is no way to tell a converged answer
from a confident wrong one.

**1.1 The bent slab with the exact Airy solution.** 1-D, semi-analytic, cheap,
and it probes the radiating tail directly — which is precisely what a PML has
to get right. Standalone module, no DBEME dependency:

* conformal map the 1-D slab bend, solve exactly in terms of Airy functions
* output complex `n_eff(κ)` over ≥ 3 decades of `Im(n_eff)`
* this becomes the fixture other phases assert against

**1.2 A published SOI bend-loss curve.** Digitise loss vs. radius for a
220 nm × 500 nm SiO2-clad Si strip, TE0, 1550 nm, from a named paper — cite it
in the test docstring. Do not assert numbers from memory; the spread across
papers is large and fab-dependent, so treat this as an order-of-magnitude band,
not a tolerance.

**1.3 Optional, cheapest reference of all.** There is an `.ansys` directory in
the user's home. If that is a live Lumerical FDE seat, resurrecting upstream's
Lumerical backend purely as a *validation reference* is the fastest route to a
trustworthy loss baseline. Not a production path — ask before assuming the
licence exists.

**Phase 1 gate:** report a table of analytic vs. computed `Im(n_eff)` across
the radius sweep. **Stop condition:** if the framework cannot reproduce the
Airy result to within ~10 % over three decades, do not proceed to Phase 2 —
the problem is in the formulation, and a 2-D solver will only hide it.

---

## Phase 2 — the backend itself

> **Status:** done — `em_simulation/fde/pml.py` (`PMLModeSolver`, `PMLBackend`), report 06 §5–6, §10.

Two routes. They **coexist** with `EmepyFDE` rather than replacing it:
`get_fde_backend()` is per dataset, so lossless datasets keep the fast uniform
grid. Backends must **never** be mixed *within* one dataset — the overlaps
require one shared grid.

| | Route A — EMpy + complex stretch | Route B — `FemwellFDE` |
|---|---|---|
| grid | **fixed uniform** — exactly what overlaps need, no interpolation error | unstructured FEM mesh → must be interpolated onto the shared grid; that error enters *every* overlap |
| PML | we own it | maintained upstream |
| bends | conformal map already in `cross_section.py` | native |
| ecosystem | none | same as gdsfactory, already in use |
| risk | spurious-mode filtering, stretched metric, all validation | interpolation error budget, extra dependency |

Recommendation: **Route A for production** (the fixed grid is worth a lot here),
**Route B as the independent second opinion** on the loss numbers. Doing only
one leaves no cross-validation without a Lumerical seat.

### Route A sketch

1. `EmepyPMLFDE(EmepyFDE)` — build `x, y` via `stretchmesh(x, y, nlayers, factor)`
   with `factor = 1 + 2j` as the starting point; pass them to `MSEMpy`.
2. Subclass `MSEMpy` to bypass the hardcoded `nlayers = [1,0,1,0]` trim in
   `get_mode` (`fd.py:208`), which will otherwise trim the wrong layers.
3. Replace `which='LR'` mode selection with **shift-invert around a target
   `n_eff`** — with PML the spectrum fills with Bérenger modes and
   `sort(key=-Re(neff))` will happily return them.
4. **Confinement filter**: keep modes whose `∫|E|²` fraction inside the core
   region exceeds a threshold. This is the classic time sink — budget for it.
5. `_check_grid` must be satisfied with `nlayers` **fixed across the dataset**;
   the returned grid size depends on it.
6. `_pin_gauge` gets fragile for strongly complex modes (§5.6): it rotates so
   the transverse field is real at its peak, then takes a sign from the first
   raster point above half-max — on a real part that may be near zero. Revisit.

---

## Phase 3 — the DBEME-specific cost, and the decision point

> **Status:** done — answered by the scattering cascade rather than by a mode count: report 06 §7–8, §11; `SMATRIX_METHOD = "auto"`, `INTERFACE_PROJECTION = "auto"`.

This is the part to think hardest about, and it is not a solver problem.

PML-EME converges *because* the discretised continuum makes the mode basis
approximately complete. That needs **~20–50 modes, not 6**. DBEME stores
`(2N × 2N)` overlaps per adjacent grid-point pair, so:

| `mode_numbers` | overlap entries per pair | relative |
|---|---|---|
| 6 | 144 | 1× |
| 20 | 1600 | 11× |
| 40 | 6400 | **44×** |

in both storage and solve time, on top of the PML grid being larger than the
physical one. Run a **mode-count convergence study** (loss vs. `N`, with
`force_unitary=False`) and find the smallest `N` that converges before
committing to a dataset. If `N` lands near 40, the honest conclusion may be
that a lossy DBEME dataset is not economic for 3-D/4-D parameter spaces and
should be built only for narrow, targeted sweeps.

**Report this number before generating any lossy dataset.**

Also required at this phase: PML parameter convergence. Every reported loss
must be stable to < 5 % under PML thickness ×1.5, stretch factor ×2, and
standoff ×1.5. A number that moves under those is not a result.

---

## Phase 4 — first real use

The pulley bus. It is at a *larger* radius than the ring and *narrower* (for
phase matching), so it is the leakiest guide in the structure and its
radiation is the thing a lossless model most conspicuously misses. Deliverable:
bus radiation loss vs. `(w_bus, R)`, overlaid on the lossless `κ` map from
`tasks/01_coupled_pair.md` Phase 3, so the design trade-off is visible in one
figure.

Demo 4 (plasmonic) needs this **plus** everything in §5.16a/b **plus** a mode
basis large enough for a deeply sub-wavelength 600 nm taper. Treat it as a
separate decision after Phase 4, not as a continuation.

> **Status (2026-09-06).** The order was inverted on request: demo 4's 2-D
> lateral stepping stone was the first real use (`reports/12`,
> `examples/demo_plasmonic_converter.py`, datasets `Si_plasmonic_slot_1550*`).
> It exercised three PML edges at once and found that EMpy's `stretchmesh`
> makes the low-side layers gain (report 06 §10) — the pulley bus, which
> radiates outward only, would never have hit that. The pulley deliverable
> above is still open.
