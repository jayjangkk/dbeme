# Accuracy and limitations

*Moved verbatim from `README.md` by task 16 step 1.5b (2026-09-27); the README is now a landing page that links here.*

Read this before trusting a number.

**Validated.** The backend reproduces emepy's own rectangle solver to 1e-3, and
`tests/` checks the invariants the method rests on: the self-overlap matrix is
the identity to ~2 %, adjacent grid points overlap > 0.9, the backward basis
mirrors the forward one, and the grid is shared. A constant-width "taper" comes
out at transmission 1.000000 with zero reflection.

`validate_against_direct_eme.py` runs the same tapers through both the dataset
and the direct (ungridded) path, which share the emepy backend and the EME
solver, so the only difference is grid snapping:

| taper length | dataset | direct | difference |
|---|---|---|---|
| 2 µm | 0.993724 | 0.993703 | 2.1e-5 |
| 6 µm | 0.999130 | 0.999153 | 2.3e-5 |
| 16 µm | 0.999859 | 0.999867 | 8.2e-6 |

They agree to ~2e-5 across the whole range, including the short tapers where
the geometry changes fastest. On a warm dataset the six-point sweep took 0.2 s
against 82 s for the direct path, a 460× speed-up.

(Before the mode-gauge fix above, this table showed a 1e-2 discrepancy at 2 µm,
which was easy to misread as a grid-discretisation error. It was the gauge bug.
Any cross-method agreement claim is only as good as the gauge test that backs
it.)

**Truncated mode basis.** With `N` modes the interface matrices are slightly
non-unitary; measured here at ~1.3e-4 of the power per interface, which over
~40 sections accumulates to ~0.5 %. Because this solver models no radiation
loss (see below), the structure is lossless by construction, so the demos pass
`force_unitary=True`, which projects each section's S-matrix onto the nearest
unitary matrix and removes the drift. Do **not** use `force_passive=True`
alone — on its own it over-attenuates badly (0.63 instead of 1.00 on the taper
above).

**Radiation and metal loss exist only on the PML backend.** `MSEMpy` in
emepy 1.2.2 has no working PML, so on the default `EmepyFDE` backend effective
indices come out real and bend radiation loss is not captured: a tight bend
shows mode conversion into higher-order *guided* modes correctly but reports
zero bending loss. `PMLBackend` (`fde/pml.py`, report 06) lifts that for the
datasets that opt into it - complex `n_eff`, the scattering cascade, 16-40
modes rather than 6 - at a solve cost of 25-90 s per cross section instead of
~1 s. Modes below the cladding index are flagged by `radiation_mode_mask` and
excluded from the launch basis, which is a cutoff criterion, not a loss model;
on a lossless dataset the mask also rejects anything above 100 dB/cm as
numerical, a rule that is switched off for a lossy basis.

**Bend curvature is bounded by the solve window.** The conformal map multiplies
the index by `exp(κx)`, so at the window edge the cladding ramps up to
`n_clad · exp(κ · half_width)`. Once that reaches the guided-mode index the
solver returns modes pinned to the window wall instead of real ones. With
`n_clad = 1.444` and a 1.6 µm half-window that puts the ceiling near
`κ = 2.3e5` 1/m (R ≈ 4 µm), which is why the shipped grid stops at 2e5.
`EmepyFDE` warns if a solve crosses the line.

**An anti-crossing needs a much finer grid.** Where two branches exchange
identity, the supermodes turn over within a narrow range of the swept parameter,
and a step that is invisible elsewhere becomes destructive. Measured on the
adiabatic coupler: a 20 nm step scatters 29 % of the power between the two
crossing branches *at a single interface*, against ~0.1 % away from the
crossing. A dataset built at 20 nm there does not give a slightly worse answer —
it reports 0.44 where the truth is 0.99, inverting the design trend. Sample at
≲5 nm across the crossing and 20 nm elsewhere; a non-uniform axis costs nothing
because only visited points are solved. See
[reports/01_adiabatic_coupler.md](../reports/01_adiabatic_coupler.md) §2.

**Reciprocity has a layout gotcha.** `_convert_3Dmatrix` produces
`[[T_forward, R_right], [R_left, T_backward]]`, *not* the textbook
`[[S11, S12], [S21, S22]]`. So `S == S.T` is not the reciprocity test and fails
at 1e-2 on a perfectly good taper. The real conditions are
`T_forward == T_backward.T` and each reflection block symmetric; those hold to
2e-5 and 1e-12. `dbeme.validation.check_reciprocity` implements the
correct one.

**The grid step is probably a floor — not yet proven.** Snapping onto a 20 nm
width grid turns a smooth taper into a staircase, and each step scatters. Past
the adiabatic threshold the residual mode conversion stops falling with device
length and settles on a fringed plateau near 1e-4
(`taper_4_length_sweep.png`). The natural explanation is the width staircase,
in which case the floor should scale as `(Δw)²` — each step is a small
perturbation, so scattered *amplitude* goes as `Δw` and power as `Δw²`. **That
scaling has not been measured here.** Until it is, treat the plateau as an
upper bound on the physical conversion, not as a physical result: rebuild the
`top_width` axis at 10 nm and 5 nm and check the floor drops 4× and 16×. If it
does not, the plateau is numerical.

**Mesh.** The default 20 nm transverse cells put the guided-mode effective
indices within ~0.002 of the converged value. `MESH` in `dataset_info.py`
trades that against solve time (~1.3 s per cross section at the default).

**Absolute effective indices.** A 500 × 220 nm SiO2-buried Si strip comes out at
n_eff(TE0) = 2.449, n_eff(TM0) = 1.788 at 1550 nm. If you are comparing against
air-clad numbers from the literature, note this is the symmetric buried case.

**One wavelength per dataset.** Material dispersion is handled — the index is
evaluated at whatever wavelength a cross section is solved at — but the
parameter grid has no wavelength axis, so a dataset is built at one wavelength
and a bandwidth sweep means one dataset per wavelength. The plumbing for an
axis is in place: `EmepyFDE` passes `wavelength` to the cross section
alongside the geometry parameters, so adding it to the grid needs no change in
`materials.py` or `cross_section.py`.
