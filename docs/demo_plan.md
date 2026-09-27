# Capability matrix, demo and report plan, report template

*Moved verbatim from `CLAUDE.md` by task 16 step 0.7 (2026-09-27). Section numbers are unchanged, so a citation of the section number in older text still finds it here.*

## 6. Capability matrix vs. the demo list

| demo | 2 cores | asym. stack | λ sweep | complex modes / loss | verdict |
|---|---|---|---|---|---|
| 1. adiabatic coupler | ✗ needed | – | optional | – | after §5.12 |
| 2. polarization rotator | ✗ needed | ✗ needed (§5.11) | ✗ needed | – | after §5.11 + §5.12 + §5.14 |
| 3. rapid adiabatic coupler | ✗ needed | – | ✗ needed (§5.14, §5.15) | – | after §5.12 + §5.14 |
| 4. plasmonic converter | metal | – | – | ✓ `PMLBackend` (§5.9, §5.16a/b done) | **2-D lateral model shipped** — `reports/12`; the 3-D taper still needs a second geometric axis |
| 4b. Si-to-slot converter, SiO₂-embedded | metal, 2 axes | – | – | ✓ `PMLBackend`, `FemwellBackend` | **shipped, converged** — `reports/13`; 72.3 % modal with sharp corners; with 20 nm-rounded corners 82.4 % (FD 5 nm) / 80.7 % (FEM, 5 nm wall) / 85.3 % (FEM, 1 nm wall) / **87.0 % (FEM, 0.5 nm wall)**, and **89–91 % under Kocabaş's own measure** (total flux at his cut, §9–§12) vs his ~95 %; the z-step terms floor at ~1 % each (§12) — the rest is a percent of Si staircase and the sharp-rectangle geometry. Length is the design parameter, the starting gap is not (flat within 1.6 points over 25–150 nm, §4). The first plasmonic device here that is adiabatic (T rises with `L`) |

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
