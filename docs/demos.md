# Demos

*Moved verbatim from `README.md` by task 16 step 1.5b (2026-09-27); the README is now a landing page that links here.*

```bash
cd examples
python demo_linear_taper.py
python demo_bezier_bend.py
python demo_material_dispersion.py
python demo_adiabatic_coupler.py       # -> reports/01
python demo_polarization_rotator.py    # -> reports/02
python sweep_wavelength.py             # -> reports/03  (O band, ~45 min)
python demo_sidewall_angle.py          # -> reports/04  (~80 min)
python demo_rapid_adiabatic_coupler.py # -> reports/05  (~90 min)
python demo_plasmonic_converter.py     # -> reports/12  (lossy PML dataset, ~2 h cold)
python demo_kocabas_converter.py       # -> reports/13  (SiO2-embedded Si-to-slot converter, two-axis lossy dataset, hours cold)
python study_rac_bandwidth.py air      # RAC splitting vs wavelength and length
python study_rac_air_basis.py          # the air-clad bound-mode basis
python study_bend_loss_reference.py    # -> reports/06  (PML phase 1 gate, ~2 s)
python study_anticrossing_grid.py
python validate_against_direct_eme.py
```

Figures land in `examples/output/`.

### What the demos produce

**Taper** — 0.5 to 1.2 µm, 220 nm full-etch Si, TE0 launched. A symmetric taper
only couples TE0 to TE2 (TE1 is forbidden by symmetry), and the coupling is
weak: TE0 transmission passes 99.9 % by 4.9 µm of taper length, and at 10 µm
the residual TE2 crosstalk is −35 dB. The 500-length sweep behind that number
runs in about 5 s with no mode solving at all.

**Bend** — S-bend, 15 × 5 µm footprint, 1.0 µm wide multimode guide with 2 µm
straight leads at each end, TE0 launched. A bend *does* couple TE0 to TE1, and
the coupling is driven by how abruptly the curvature changes:

| centreline | curvature steps | power leaving TE0 |
|---|---|---|
| circular arc | 3 (in, join, out) | **2.91 %** (−15.4 dB) |
| cubic Bézier | 2 (in, out) | **0.76 %** (−21.2 dB) |
| quintic Bézier | none | **0.057 %** (−32.5 dB) |

Same footprint, same dataset, 50× less crosstalk. The whole comparison is three
different paths through one set of cached cross sections.

**Polarization rotator-splitter** — both stages of the Sacher et al. PRS
([Opt. Express 22, 3777](https://doi.org/10.1364/OE.22.003777)), each on its
published geometry, with three write-ups in [reports/](../reports/):

| stage | result | published length | 99.9 % threshold here |
|---|---|---|---|
| [bi-level taper](../reports/02_polarization_rotator.md) | TM0 → TE1 at **99.6 %**, TE fraction 0.065 → 0.965 | 100 µm | 76 µm |
| [adiabatic coupler](../reports/01_adiabatic_coupler.md) | TE0 stays **98.9 %** broad, TE1 exits **98.2 %** narrow | 300 µm | 304 µm |

Both published lengths sit just above the computed adiabaticity thresholds,
which is a stronger check than either device alone. The
[O-band sweep](../reports/03_oband_wavelength_sweep.md) then finds the rotator
transfers to 1260–1360 nm unchanged (≥99.9 %) while the coupler collapses to
34 %, because its anti-crossing is evanescent coupling across a *fixed* 200 nm
gap and tighter confinement at short λ shrinks it 3×.

**Rapid adiabatic coupler** — the coupling region of the first experimental RAC
(Fargas Cabanillas thesis §3.5.3, 220 nm SOI e-beam), in
[reports/05](../reports/05_rapid_adiabatic_coupler.md). A RAC keeps the width
schedule of an ordinary adiabatic coupler and changes only its *transverse
position evolution*: a global tilt adds the same `tan θ` to all four wall
slopes, and because the four walls contribute to κ₁₂ with opposite signs there
is a tilt at each position that nulls the sum.

| 11.5 µm coupling region, 1550 nm | crosstalk | splitting |
|---|---|---|
| untilted adiabatic coupler | 2.17e-2 | 48.12 % |
| best constant tilt, −2.50° | 4.30e-4 (51×) | 51.62 % |
| **θ_RAC(η) trajectory** | **2.22e-4 (98×)** | **49.37 %** |

Same cross sections in all three rows — only the lateral offset schedule
differs, carried as one extra dataset axis. Against the thesis: a 3.13× length
advantage over a conventional AC (3.65× published) and 50 ± 1.62 % splitting
over 110 nm (50 ± 1.4 % over 145 nm measured, for the full four-region device).
Two lessons came out of it — a dataset axis must be commensurate with the
*slope* of the device trajectory, not merely fine, and the zero-coupling branch
must be traced by continuation rather than by scan order, which is worth 29× on
its own.

**Sidewall angle** — the same rotator with the etches at 85° and 87° instead of
vertical ([reports/04](../reports/04_sidewall_angle.md)). Conversion moves from
99.99 % to 99.93 %, and stays above 99.90 % down to 82°, so for this device the
etch angle is not a design variable. The reason is worth knowing: a trapezoid
has no horizontal mirror plane, so a slant is *itself* a rotation mechanism —
on a plain strip it turns the TE1/TM0 crossing into an anti-crossing with
Δn_eff = 0.021 — but the partial etch already opens 0.174, eight times more, so
the rib saturates the symmetry breaking and inherits the tolerance.

**Materials** — Si moves by 0.037 in index between 1260 and 1625 nm, SiO2 by
0.004, Si3N4 by 0.011. Freezing the index at its 1550 nm value costs 0.034 in
n_eff by 1260 nm for a 500 × 220 nm strip, and — worse — it deletes the
`λ·dn/dλ` term from the group index, which comes out 3 % low *at the reference
wavelength itself*, where n_eff is exact by construction. Anything involving
delay, FSR or dispersion needs the real index model.

---

**Plasmonic mode converter** (`demo_plasmonic_converter.py` → `reports/12`) —
the first device on the lossy PML backend. The lateral model (Si width
400 nm → 0 between 400 nm gold walls at a 20 nm air gap, suspended) has a
bound slot plasmon at 1.145 + 0.036j — 1.27 dB/µm, the paper's "about
1 dB/µm" — and its 41-interface cascade carries propagation loss exactly and
is passive, but only after the interface projection was moved to the output
side (below): upstream's form returned the mismatch field the basis cannot
hold as *gain*, 0.2–1.9 % per 10 nm step and ×1.44 over the taper,
independently of mode count, of the PML and of basis selection. The passive
cascade gives −1.61 dB at 600 nm on co-located E/H fields (−1.85 with sharp
corners on the same fields), of which ≈ −1.4 dB is the mismatch loss of a
40-step staircase and the rest metal loss, monotonic in length with no optimum: the grid-snapped path visits the same 41 cross sections at every
length, and the shed field that would have to interfere destructively for a
taper to be adiabatic is in no basis of this kind. So the paper's −1 dB and
its 600 nm optimum are not reproduced, and the report says why they cannot
be. Five things were found on the way and are now tests: the `-x`/`-y` PML
layers were gain, the radiation mask rejected every lossy mode, a metal edge
inside a cell aliases `n_eff` by 0.1, a lateral slot binds only between tall
walls, and the projection side of the interface algebra. The shipped platform
rounds the gold corners (20 nm; `Si_plasmonic_slot_1550_sharp` keeps the sharp
variant): a sharp metal wedge carries a singular field that no grid resolves,
and rounding it moves the slot plasmon from 1.145 + 0.036j to 1.351 + 0.025j -
better bound, 0.87 instead of 1.27 dB/um - while the Si-end mode, which lives
on the flat wall face, barely moves, and the taper's per-step mismatch does not
move at all: the corners were the slot's problem, the flat-wall gap field is the
taper's (report 12 section 8).

---

**Piecewise-refined solve grids** (`PMLModeSolver(refine_x=, refine_y=)`) —
the uniform grid can be cut into finer cells inside chosen regions, with the
PML and the outer cells untouched; the finite-difference operator already took
per-cell spacings, and the three places that inferred cell geometry from the
centres alone (the cross-section fill fractions, the rounded-corner fill, the
power sums behind confinement and `TE_pol`) now reconstruct exact cell edges
and weight by area. A refinement enters the dataset identity. Strips of the
Kocabaş cross section (`--refine LO HI CELL`, axes `(w_si, half_slot)`):
the Si tip at 1 nm (`--tip 60 1`, +20 % unknowns) bought 0.7 points, because
on axes where the two edges move separately the staircase turns out to be
the gold wall's 5 nm jumps, not the silicon's steps. Refining the wall then
exposed three things (report 13 §7): 1 × 5 nm cells at a metal corner corrupt
the reconstructed E while H and `n_eff` look fine, so corner rows need a
`--refine-y` strip too; a sharp metal wedge does not converge as it is
resolved, so only a rounded corner is a geometry a refinement can converge
on; and a rounded gold arc in silica needs `fill_floor=0.06`, because the
sub-sampled fill lands on the ε-near-zero mix. The like-for-like pair on the
rounded geometry (`--suffix hs_c20` against
`--suffix r100-300_1+y95-135_1_c20`, both `--axes half_slot`) is the
experiment.

**A finite-element backend** (`dbeme/fde/femwell_fde.py`) — a
boundary-conforming mesh behind the same backend contract, for the case the
finite-difference grid cannot converge: a rounded metal corner is a staircase
at any affordable cell, and the gap plasmon moved 0.03–0.05 in `n_eff`
between 5, 2.5 and 1 nm cells. Cross sections describe themselves as shapely
polygons (`CrossSection.polygons()`), femwell/gmsh meshes them with the arcs
as curves, a lossy outer ring stands in for the PML, and the fields are
point-evaluated onto the dataset's uniform grid so that nothing downstream
changes. Validated against the FD backend on a dielectric strip, where the
two solvers' fields overlap as the same modes to better than 0.99
(`tests/test_femwell_backend.py`); `kocabas_converter_dataset_info(solver="femwell")`
selects it (report 13 §8).

**Si-wire-to-slot converter, silica-embedded** (`demo_kocabas_converter.py` →
`reports/13`) — the same physics on a device whose every dimension is
published (Kocabaş, arXiv:1801.00833, Table II Set 2: Si 400 × 725 nm, gold
250 nm, slot 250 nm, 1700 nm taper). Because both ends are embedded in SiO₂
they are both bound, which the NTT device's air core is not, and the dataset
is two-axis: `(w_si, gap)`, 471 × 347 at a 5 nm cell, 20 modes about a
shift-invert target of 2.3. The cascade gives **72.3 % into the slot mode**
with the paper's sharp gold corners, and **82.4 %** once they are rounded by
20 nm on the same grid (report 13 §7: the sharp wedge's singular field,
regularised by a 5 nm grid, was most of what looked like a basis floor) —
and that number is converged: 40 modes give 82.9 %, the finite-element mesh
80.7 % with the wall still stepping 5 nm, 85.3 % at 1 nm and **87.0 %** at
0.5 nm, where the z-step terms floor (§8, §10, §12) — and under the paper's
own measure, total forward flux 1100 nm past the tip back-propagated,
computed from the same cascade (§9, §12), **89–91 %**, where the paper reports
~95 %; on that platform the
gap sweep is flat (84.6–86.2 % over 25–150 nm) and only the length matters; the deficit is 0.22 dB of
staircase mismatch, 0.19 dB of the supermode's own metal loss, and amplitude
scattered into the Berenger set. Two results matter more than the number.
**The converter is adiabatic here**: transmission rises monotonically with
taper length (65.4 → 73.9 % from 500 to 3000 nm) and the scattered amplitude
falls as roughly 1/L — the behaviour report 12's device could not show,
because there a 10 nm edge step sheds a field no shift-invert basis holds.
And **basis membership is a correctness condition**: with the target at 1.9
the wire's TE fundamental left the 16 nearest eigenvalues part-way along the
path, the tracker linked a cutoff branch instead, and the cascade read −20 dB
without failing. The target must sit among the physical branches, not below
them (docs/validation_backlog.md §5.13a).
