# Task 01 — two-core cross section and the coupled-pair dataset

**Prompt to start with:**

> Read `CLAUDE.md`, then `tasks/01_coupled_pair.md`, and do Phase 1. Stop at the
> Phase 1 gate and report the acceptance-criteria table before going further.

Conventions, units and the dataset doctrine are in `CLAUDE.md` — this file does
not repeat them. Sections referenced as §5.x are that file's validation backlog.

---

## Why

`FullEtchStrip` has one core. `CompositeGeometry` / `MultiEME` cascade sections
**along z**; they do not place waveguides side by side. So nothing in the repo
can currently represent two coupled guides.

This one piece unblocks four devices: the pulley coupler, the adiabatic
coupler (demo 1), the RAC (demo 3), and any ADC. Build it once, carefully.

**Radiation loss is explicitly out of scope.** All four devices' primary
figures of merit — coupling coefficient, phase matching, splitting ratio,
crosstalk — are ratios among guided modes, which the lossless model gets
right. Do not wait for a PML backend, and do not report an insertion loss.

---

## Phase 1 — cross section and a straight dataset

### 1.1 `CoupledStrips(CrossSection)` in `dbeme/fde/cross_section.py`

```python
parameter_names = ("w1", "w2", "gap", "curvature")
```

Two rectangles of the same `thickness`, on the same y-layer, separated by
`gap`. Reuse the existing `_fill_fraction` for sub-pixel edges — the 20 nm
width grid depends on it.

**Origin convention — decide this once and document it in the docstring.**
Place the pair symmetrically about `x = 0`, i.e. **mid-gap at the origin**:

```
   core 1                    core 2
 [-(gap/2 + w1), -gap/2]   [+gap/2, +gap/2 + w2]
                    ^
                  x = 0
```

Not ring-centre, not core-1-centre. The reason is the conformal bend map: the
ramp `n_eq = n·exp(κx)` grows from the origin, so centring on the mid-gap
roughly **halves** `|x|` at the window edge and buys back a factor ~2 in
usable curvature (see §5.10 and 1.4 below). The propagation length for a bent
pair is then the arc length **at the mid-gap radius** — state that in the
docstring, because every subsequent phase-matching calculation depends on it.

`default_window(max_params)`: `half_width = gap_max/2 + max(w1,w2) + clearance`,
clearance ≈ 1.2 µm. Keep the existing y extent.

**Minimum gap constraint.** Two cores separated by fewer than ~2 mesh cells
will merge under sub-pixel averaging into one wide core, silently. At the
default 20 nm cells, `gap_min = 100 nm` is safe and `50 nm` is not. Raise a
`ValueError` in `index()` if `gap < 2 * cell_size`, rather than returning a
plausible-looking wrong index profile.

### 1.2 Dataset `datasets/Si_fulletch_220nm_pair/dataset_info.py`

Start **2-D and straight** — do not build the curvature axis yet:

| axis | range | step | note |
|---|---|---|---|
| `w2` | 0.30 – 0.80 µm | 20 nm | `w1` **fixed** at 0.50 µm for now |
| `gap` | 0.10 – 0.50 µm | 20 nm | below 0.10 µm see 1.1 |
| `curvature` | `[0.0]` | — | single point, straight only |

`mode_numbers`: **8**, not 6. Two guides need at least TE0±, TE1±, TM0± in the
basis and 6 will not leave margin. Confirm by inspecting `TE_pol`.

Recompute and document the conformal ramp limit for the new (wider) window in
the `parameters` comment, the way the single-guide dataset does.

### 1.3 Acceptance criteria — Phase 1 gate

Report these as a table. **Do not start Phase 2 until they pass.**

| # | check | criterion |
|---|---|---|
| A1 | **Symmetric pair** `w1 = w2 = 0.5 µm`, gap 0.2 µm | two lowest modes are the even/odd supermode pair, both `TE_pol ≈ 1`, `Δn_eff > 0` |
| A2 | **Coupling length** from A1 | `L_c = λ / (2 Δn_eff)`; cross-check against an independent 2×2 coupled-mode estimate. Order of magnitude for 220 nm Si at 200 nm gap is tens of µm — a result in mm or in nm means the cross section is wrong |
| A3 | **Decoupling limit** gap → 1.5 µm | `Δn_eff → 0`, and each supermode `n_eff` converges to the **single-guide** value from `datasets/Si_fulletch_220nm` at the same width, to < 1e-3 |
| A4 | **Full power transfer** | launch even+odd (= one guide) on a straight pair of length `2 L_c`: power must oscillate fully between guides and return; total power conserved to < 1e-3 with `force_unitary=False` |
| A5 | **Anti-crossing** sweep `w2` through `w1` at fixed gap | tracked `n_eff` branches must not cross; minimum splitting = `2κ` |

A3 is the strongest check available — it is the only one that validates the new
cross section against an *independently built* dataset. Prioritise it.

A5 is where it is most likely to break: see §5.13. If the tracking sorts by
`n_eff` it will swap labels at the avoided crossing. Replace with Hungarian
assignment on `|overlap|` against the previous grid point.

### 1.4 Cost guard — verify before any 3-D grid

Instrument `DataUpdater` to count `overlap_matrix` calls for one device path
and confirm the count scales as *(path length) × (neighbours)*, **not** as
`N²` over the grid. If upstream ever builds a full pair table, fix that first —
a 3-D or 4-D axis is not survivable otherwise (see `docs/dataset_doctrine.md` §3).

---

## Phase 2 — geometry classes and the coupled taper

New package `dbeme/geometry/coupled_waveguide/`, following the
`single_waveguide/` pattern: subclass `Geometry`, implement
`calc_simulation_parameters()` returning `(simul_params, delta_zs)`.

| class | path through the grid | serves |
|---|---|---|
| `CoupledStraight` | one point, held for length `L` | ADC, pulley arc |
| `CoupledLinearTaper` | `w2` varies linearly | adiabatic coupler, RAC |
| `CoupledCustomTaper` | caller-supplied `w2(z)` | shape optimisation |

Acceptance: §5.1 (reciprocity), §5.4 (slicing independence), and §5.8 (DBEME
vs. `validate_against_direct_eme.py`) on a coupled linear taper.

---

## Phase 3 — curvature axis and the pulley arc

Add `curvature` as a real axis and a `PulleyArc` geometry: constant
`(w1, w2, gap, κ)` held over arc length `θ · R_ref`, with `R_ref` the **mid-gap**
radius per 1.1.

The pulley is the cheapest device in the whole plan — the arc is a *single*
grid point, so the wrap angle `θ` is a free parameter and sweeping it costs
zero mode solves. Use it as the first real device rather than the adiabatic
coupler.

Phase-matching target for the design sweep:

```
n_eff,bus · R_bus = n_eff,ring · R_ring      (angular, not linear, β)
```

`R_bus > R_ring`, so the bus must be **narrower**. Deliverables: `κ` vs.
`(w_bus, gap)`, coupling vs. `θ`, and the CIFS estimate from the arc supermode
`n_eff` against the isolated ring `n_eff`.

---

## Gotchas

* **Window widens → curvature headroom shrinks.** Recompute the ramp limit and
  update `_check_conformal_ramp`'s warning text for the new `half_width`.
* **Near-degenerate supermodes at large gap.** At gap ≥ 1 µm the even/odd pair
  approaches degeneracy, which is the `O(m, ℂ)` gauge problem in §5.6 that
  `_pin_gauge` cannot fix. Test explicitly at gap = 1.5 µm; if A3 is noisy,
  that is the cause, not the cross section.
* **Do not enable symmetry boundary conditions.** They halve the solve, but
  `w1 ≠ w2` is the whole point of an adiabatic coupler and the cross section is
  then asymmetric. The dataset must not assume symmetry anywhere.
* **`w1` fixed in Phase 1 is a deliberate simplification.** When you later free
  it, prefer reparameterising to `(w̄, Δw)` at a few coarse `gap` values rather
  than a dense 3-D `(w1, w2, gap)` — see `docs/dataset_doctrine.md` §3.
