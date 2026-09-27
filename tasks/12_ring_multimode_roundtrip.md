# Task 12 — the multimode ring as a round-trip operator; inner-path design

**Prompt to start with:**

> Read `CLAUDE.md`, `tasks/11_ring_resonator_circuit.md` (its Phase 1–3
> status), then this file. Do Phase A and stop at its gate.

Origin: an assessment Jae obtained from Claude in chat on 2026-09-20 of
*whether DBEME is powerful for micro-rings*. It is reproduced below with its
claims checked against the repo as it stands after Task 11 Phases 1–2, and
turned into a plan. The parts of the assessment that were already out of
date when it was written are marked; the idea at its centre — the ring as
an eigenproblem of a **multimode round-trip operator** built from DBEME's
full `(2N, 2N)` S-matrix — is new to this repo and is what this task builds.

---

## 1. The assessment, annotated

> **Core answer.** Yes, and it can be powerful — but for a *specific* class
> of ring problems, not for ring optimization in general. The dividing line
> is loss: DBEME has no radiation or sidewall-scattering loss, and Q_int is
> loss. Until the PML backend (task 02) or an empirical α(w, R) table
> exists, DBEME cannot optimize the one thing most ring work is about.

*Status 2026-09-20 — partly outdated.* The PML backend exists and is
validated (`em_simulation/fde/pml.py`, `reports/06`, task 02 done): bend
radiation loss per `(w, R, λ)` is a direct solve, and `studies/ring/ring_loss.py`
computes it for the 500 × 220 nm ring at R = 2, 3, 5 µm. Sidewall scattering
is now a model in `em_simulation/circuit/roughness.py` (Payne–Lacey via
the effective-index method, σ and L_c declared, scaled to each bend by the
mode's sidewall field factor). What remains true: a **lossy DBEME dataset**
(PML basis inside the cascade) is not built — report 06 §8, the mode count —
so loss enters the ring as a **per-mode scalar on the arc**, exactly the
"external scalar" the assessment proposes, but computed rather than measured.
The width/radius trade-off of Q_int is therefore already answerable by a
small table of direct solves; what is not answerable is loss *inside* a
non-uniform inner path (a width-modulated ring), because that path's S-matrix
comes from the lossless basis.

> **Where it is powerful — the multimode round-trip formulation.** Treat the
> ring as a closed path through the dataset: coupler region (coupled-pair
> dataset, curvature axis) + inner-waveguide schedule (width(z), offset(z) →
> curvature(z)) in the single-guide bent dataset. DBEME returns the full
> 2N×2N S-matrix of that path, forward *and* backward modes. The ring then
> becomes an eigenproblem of the round-trip operator:
>
> M(λ) = S_inner(λ) · S_coupler,through(λ), resonances where det[I − M(λ)] = 0.
>
> For a single mode this collapses to the usual a·t·e^{iφ} = 1. For a
> multimode ring it directly gives: mode-hybridized resonances, TE0↔TE1
> conversion per round trip (|M₁₀|), split resonances from backreflection
> (the backward block of S), and mode-selective loaded Q. Those are exactly
> what t-CMT can't represent and what 3D FDTD of a Q=10⁴ ring costs hours
> per iteration to see. The inverse-design FoM for the inner waveguide is
> then clean: maximize |M₀₀|, suppress |M₁₀| and the reflection block over
> the band. That is a real niche: wide low-loss rings that must stay
> single-mode, adiabatic width narrowing at the coupler, angular
> Bragg/width-modulated rings for FSR-free or mode-selective operation,
> backscatter-immune designs.

*Agreed, and not yet built.* Task 11 composes a **single-mode** ring in sax
from a 4-port coupler and an analytic arc. The circuit layer already
carries multimode ports (`"o1@TE0"`, `"o1@TE1"`, …,
`em_simulation/circuit/sax_model.py`), so a multimode ring is a netlist away;
the eigen-formulation below is the cheaper and more informative route for
resonances and Q, and the two must agree (that is the gate). One caveat on
the operator as written: with backreflection the round trip couples forward
and backward waves, so `M` is the `2N × 2N` transfer operator of the closed
loop, not the forward block alone — §3.1 states the exact form.

> **What it cannot do (yet).**
>
> | target | status (as written) |
> |---|---|
> | Q_int, bend-loss vs. width trade-off | ❌ needs PML or measured α(w,R) |
> | stochastic sidewall backscatter | ❌ not in EME; Payne–Lacey estimate is a cheap add-on |
> | κ(λ) of a pulley coupler with two different radii | ⚠️ needs coupled-pair dataset with curvature axis |
> | gradients w.r.t. width/gap/offset | ⚠️ CMA-ES on the path, autograd on arc lengths |

*Status 2026-09-20:*

| target | status now |
|---|---|
| Q_int, bend loss vs width | **available** for uniform rings: PML per `(w, R, λ)` (`ring_loss.py`); for a non-uniform inner path still a per-section scalar from the same table |
| sidewall scattering loss | **model built** (`circuit/roughness.py`); *backscatter* (the coherent part that splits resonances) is not — see Phase C |
| pulley coupler κ(λ) | **not built**; Task 11 built the *point* coupler in the straight frame with the bus fixed (`platforms.BusRingStrips`, gap axis tied to a 10 nm cell). A pulley needs `curvature` on the pair; the assessment's `(w_bus, w_ring, gap, R)` with gap and w_bus fixed per family is the right cut |
| gradients | the sax models are `jnp` functions, so `jax.grad` runs through the *cached* grid (Task 11 Phase 5.3); geometry beyond the grid still costs solves; CMA-ES on the path as in `studies/sirac/optimize.py` |
| circuit layer | **built** (Task 11 Phase 1: sax adapter, closed-form ring, tests); circulax installed in `.venv-circuit` (jax 0.9.2, sax 0.18.2 — incompatible with the main venv's pins, hence separate) |

> **Honest assessment of "powerful".** For an MRM, the inner-waveguide
> shaping buys you: junction-overlap vs. loss (needs loss model → not yet),
> single-mode wide rings (yes, now), and backscatter/split-resonance
> suppression (yes, now). The coupler is where DBEME already pays off
> unambiguously. So: powerful for mode-purity and coupler design today;
> becomes a full ring optimizer only after PML.

*Updated reading.* Loss for a uniform ring: yes now. Loss along a shaped
inner path: only as a per-section scalar (no lossy dataset). Mode purity,
coupler, split resonances from *deterministic* reflection (junctions,
width steps): yes now via the operator. Stochastic backscatter: needs a
model (Phase C). "Full ring optimizer" therefore means: uniform-width ring
fully; shaped ring with loss as a scalar per section.

> **Build order:** (1) coupled-pair + curvature dataset → pulley κ(λ); (2)
> round-trip operator M(λ) + circuit layer (circulax/sax, JAX); (3) loss: PML
> or empirical α; (4) inner-path inverse design.

*Reordered by what exists:* (2) first — the operator is pure algebra on
S-matrices Task 11 already produces; (3) is done for uniform guides; (1) is
a dataset build with a known recipe; (4) last.

---

## 2. What exists (reuse)

| need | where |
|---|---|
| full `(2N, 2N)` S of a path, tracked order, `[b_out; b_in] = S [a_in; a_out]` | `em_simulation/validation.py::lumped_smatrix`; index convention `examples/demo_plasmonic_converter.py:115-118` |
| 4-port coupler in bus/ring basis, sign from fields | `studies/ring/build_coupler.py::analyse`, `four_port` |
| bent-guide `n_eff(λ, R)`, PML loss, roughness | `reports/output/ring/ring_loss.json`, `studies/ring/ring_loss.py` |
| sax models with multimode ports; phase-fitted λ interpolation | `em_simulation/circuit/sax_model.py`, `waveguide.py` |
| closed-form ring and metrics | `em_simulation/circuit/ring.py` |
| width-schedule optimisation on a cached path | `studies/sirac/optimize.py` (CMA-ES), report 07 |
| single-guide bent dataset (`top_width`, signed `curvature`) | `datasets/Si_fulletch_220nm` |

## 3. Phase A — the round-trip operator on Task 11's building blocks

### 3.1 `em_simulation/circuit/roundtrip.py`

For a loop made of segments with S-matrices `S_k` (each `2N × 2N` in the
port convention above), form each segment's **transfer matrix**
`T_k = tm(S_k)` mapping `[a_in; b_in]` at its input to `[b_out; a_out]` at
its output (standard S→T conversion; singular when a segment is fully
reflective, guard with `rcond`). The closed loop's round-trip operator is
`M(λ) = T_K ⋯ T_2 T_1` (all `2N × 2N`, forward and backward). A field
`v` on the loop is a steady state iff `M v = v`:

```
resonance:  det[ I − M(λ) ] = 0
```

With only forward modes (no reflection) `M` block-diagonalises and the
forward block is the assessment's `S_inner · S_coupler,through`; with one
mode it is `a t e^{iφ} = 1`. Implement:

* `transfer_from_smatrix(S, N, rcond=1e-10)`;
* `roundtrip_operator(segments, wl)` — segments are sax models or
  `(wavelengths, S)` samples (via `sax_model_from_samples`), evaluated at
  `wl`;
* `resonances(M_of_wl, band, n_guess)` — complex-wavelength roots of
  `det[I − M]` by Newton from the minima of `|det|` on a real-λ scan; each
  root gives `λ_res = Re`, `Q = Re / (2 Im)`, and the eigenvector `v` of
  `M(λ_res)` with eigenvalue 1 is the resonant **mode mixture**: its weights
  on TE0 / TE1 / backward give hybridisation and standing-wave (split)
  character directly;
* `roundtrip_metrics(M)` on real λ: `|M_00|`, `|M_10|`, reflection-block
  norm — the inverse-design FoM.

The coupler segment: the through-block of `build_coupler.py`'s 4-port S
restricted to the ring ports (ring in → ring out) is the single-mode case;
for the multimode operator take the tracked `(2N, 2N)` lumped S and keep the
ring supermode components (the bus/ring rotation of §2.1 in Task 11 applies
per TE order).

### 3.2 Gate

| check | criterion |
|---|---|
| single-mode ring: roots of `det[I − M]` vs `ring.py` resonances and Q | λ 1e-12 m, Q 1e-6 relative (same algebra, two routes) |
| lossless multimode loop with a unitary coupler: `M` unitary | `‖M†M − I‖ < 1e-10` |
| sax multimode ring netlist (ports `o@TE0`, `o@TE1`) vs operator resonances | peak positions 1 pm, Q 1 % |
| reflection: insert a synthetic 1 % reflector; the operator shows the doublet, sax shows the same doublet | splitting agrees to 1 % |

## 4. Phase B — a real multimode ring

A 1.0 µm-wide, 220 nm ring (TE0, TE1 guided) at R = 10 µm with the
point coupler: needs the coupler dataset at `w2 = 1.0 µm` (bus 500 nm) —
`ring_coupler_dataset_info(widths=…)` with `w1 ≠ w2` (the `BusRingStrips`
frame already allows it) — and the inner ring as a uniform bent guide from
`Si_fulletch_220nm` at `top_width = 1.0 µm`. Deliverables: TE0 and TE1
resonance families, their Q, TE0→TE1 per round trip `|M₁₀|`, and the
coupler's mode selectivity vs gap (the design knob of Hosseini et al. 2010).
Loss: per-mode scalars from `ring_loss.py` extended to 1.0 µm width.

## 5. Phase C — inner-path shaping

A width schedule `w(θ)` on the ring (e.g. 1.0 µm in the arcs narrowing to
0.5 µm at the coupler) as a `ParametricPath` on `Si_fulletch_220nm`
(`top_width`, `curvature = 1/R` constant), giving `S_inner`; the FoM from
§3.1 (`|M_00|` high, `|M_10|` and reflection low over the band); optimise
with CMA-ES as `studies/sirac/optimize.py` does — every evaluation is a
re-cascade over cached points. Loss along the path: per-section scalar from
the `ring_loss.py` table interpolated in width (state it). Stochastic
backscatter (Little et al., *Opt. Lett.* 22, 4 (1997) coupled-mode model of
contra-directional scattering; the Payne–Lacey spectrum gives the
backscatter strength from the same σ, L_c) enters `M` as a per-section
forward↔backward coupling — an ensemble over realisations gives the
split-resonance statistics. Gate: a width-modulated ring known to split
resonances (angular Bragg) reproduces the splitting from the deterministic
reflection block.

## 6. Phase D — pulley coupler

`CoupledStrips` with `curvature` on the grid `(w1, w2, gap, κ)`, `w1` and
`gap` fixed per family (the assessment's cut), mid-gap origin per
`tasks/01_coupled_pair.md` §1.1, and the two guides at different radii
represented by the common conformal `exp(κx)` with their offsets. The
pulley's κ(λ) then replaces the point coupler in Phases A–C. Cost: one more
axis, ~`N_κ` × the point-coupler dataset; keep `N_κ` ≤ 4.

## References

- W. Bogaerts et al., *Laser Photon. Rev.* 6, 47 (2012) — ring compact model, coupling regimes.
- E. S. Hosseini et al., *Opt. Express* 18, 2127 (2010) — pulley coupler, mode-selective coupling.
- F. P. Payne and J. P. R. Lacey, *Opt. Quantum Electron.* 26, 977 (1994) — sidewall-scattering estimate from the mode field.
- B. E. Little, J.-P. Laine, S. T. Chu, *Opt. Lett.* 22, 4 (1997) — surface-roughness-induced contra-directional coupling in rings.
