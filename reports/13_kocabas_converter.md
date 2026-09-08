# 13 — Si wire to plasmonic slot converter, SiO₂-embedded (Kocabaş 2017)

**Why this device.** Report 12 ended with a negative result about the NTT
converter: its air-core MIM on oxide has no bound lateral mode (§9 there),
so the coupling-versus-gap and coupling-versus-length curves Jae asked for
cannot be computed on that geometry with one swept width. S. E. Kocabaş,
*The effect of metal thickness on Si wire to plasmonic slot waveguide mode
conversion*, arXiv:1801.00833 (2017), is the same device class — a laterally
tapered Si wire feeding a gold slot through an air-gap-like clearance — with
every dimension published (Table II), everything embedded in SiO₂ so that
both ends are bound, and numbers to hit: ~95 % transmission for the 250 nm
gold design (Set 2), ~88 % for 30 nm gold (Set 1), slot propagation lengths
versus gold thickness (Fig. 4), and transmission versus gold thickness
(Fig. 7). It is therefore the better test of the PML backend and the
converter demonstration this work is for.

**The device (Set 2).** Si wire 400 × 725 nm; gold 250 nm thick, centred on
the Si; slot 250 nm; taper 1700 nm after a 200 nm lead-in, with the Si going
400 → 0 nm and the slot going 550 → 250 nm independently, so the Si–gold
clearance runs 75 → 125 nm; 200 nm of slot after the tip. Material constants
as in the paper's Table I (Si 12.085, SiO₂ 2.0852, Au −126.80 + 5.37i in
this project's convention). COMSOL 3-D with PML in the paper; DBEME on a
two-axis `(w_si, gap)` dataset here.

**Method.** `PMLBackend` dataset `SiO2_kocabas_set2_1550` (5 nm cell, grid
snapped so every straight edge sits on a cell boundary at every grid point;
PML 0.25 µm on four edges; 20 modes about a shift-invert target of 2.3 —
§1 records why not 16 about 1.9), scattering cascade,
output-side interface projection, `force_unitary=False`. The transmission is
the power in the slot mode at the end of the 200 nm lead-out, and — the
paper's convention — back-propagated to the tip with the slot mode's own
loss. The launched mode is the wire's TE-like **fundamental** (E along x,
the slot's polarisation), 2.873 on this grid; a 725 nm-tall wire also carries
a TM-like fundamental at 3.002 and first-vertical-order TE/TM modes at
2.443 / 2.451, which a shift-invert target of 1.9 returned *instead of* the
fundamental (§1).

---

## 1. Sanity gate

Rows from `reports/output/kocabas_converter.json` (co-located fields,
target 2.3, 20 modes) and from the window study (`kocabas_window.json`):

| check (§5.x) | criterion | measured | pass |
|---|---|---|---|
| constant 250 nm slot, 1 µm: `T = exp(−2 k₀ Im n L)` | `|T − expected| < 1e-3` | 0.93244 vs 0.93244 | ✓ |
| 5.1 reciprocity of the physical channel, design path | `< 1e-6` | 1.1e-3 | ✗ — the same magnitude as report 12 §10 on co-located fields (1.0e-3); the residual of the unconjugated overlap on a 5 nm staircase, not a convention error (§5.1 there) |
| passivity: physical input columns of `|S|²` | `< 1.05` | 0.81 | ✓ |
| 5.8 DBEME vs direct EME, sections at the dataset's own points | `max|ΔT| < 1e-3` | pending (§5) | |
| 5.4 slicing: the same device with sections at uniform `z` instead | reported | 84.1 % against 72.3 %, `max|ΔT|` = 1.2e-1 — off-grid sections put every metal edge inside a cell *and* let the gap move 1.25 nm at a time instead of 5 (§5) | — |
| 5.7 window: the weakly bound slot modes with 0.5 / 1.5 / 2.5 µm margins | reported | 250 nm slot: 1.4498 + 0.0086j (`L_p` 14.3 µm) / 1.4475 + 0.0060j (20.7) / 1.4476 + 0.0055j (22.3); 220 nm: 14.2 → 19.3 µm; **30 nm: identical** (1.8349 + 0.0243j) | — |
| 5.13a basis membership along the path: is the launched branch in the stored set at every width? | every physical branch present | target 1.9, 16 modes: the TE-like fundamental (2.87 at 400 nm) **absent** at 300 nm; the tracker linked the first-vertical-order branch into cutoff and the cascade read **−20 dB**. Target 2.3, 20 modes: fundamental present at 400 / 300 / 260 / 160 / 60 / 0 nm (2.873 / 2.541 / 2.326 / 1.800 / 1.534 / 1.450) | ✓ after the change |
| 5.6 gauge, 5.13 tracking, PML sign, grid alignment, gap as a path parameter | tests | `tests/test_plasmonic_slot.py` | ✓ |

The window row is the one real limit of the platform grid: a mode bound by
0.006 above silica has a 2 µm tail, four times the 0.5 µm margin the dataset
is built on, and the PML takes about a third of its apparent loss. The
index moves by 0.002, the mode shape not measurably (confinement 0.475 vs
0.486). The converter numbers below therefore carry the slot mode with a
loss ~50 % too high; the 200 nm lead-out back-propagation that this affects
is 1.4 % against 0.9 %. A dataset with 1.5 µm margins would cost 4× per
solve and is the first thing to buy if the slot loss itself is the quantity
of interest.

## 2. Modes

### The wire, the middle and the slot (5 nm cell, 0.5 µm margins, target 2.3)

| cross section | branch | `n_eff` | confinement |
|---|---|---|---|
| wire 400 × 725 nm, gap 75 nm | TM-like fundamental | 3.002 + 0.000j | 0.99 |
| | **TE-like fundamental (launched)** | **2.873 + 0.000j** | 0.98 |
| | first vertical order, TM / TE | 2.451 / 2.443 | 0.98 / 0.91 |
| | second order, TE / TM | 1.841 / 1.783 | 0.89 / 0.86 |
| | slot-like | 1.451 + 0.030j | 0.22 |
| Si 260 nm, gap 95 nm | TM / TE fundamental | 2.748 / 2.326 | 0.98 / 0.96 |
| Si 160 nm, gap 105 nm | TM / TE fundamental | 2.318 / 1.800 | 0.95 / 0.85 |
| Si 90 nm, gap 115 nm | TM / TE fundamental | 1.751 / 1.592 | 0.78 / 0.71 |
| slot 250 nm, gold 250 nm | slot mode (port) | **1.4498 + 0.0086j** (`L_p` 14.3 µm) | 0.49 |

![branches](output/kocabas_1_neff_path.png)

Left: the physical branches the tracker follows along the path, from the
400 nm wire to the 250 nm slot. Right: their loss. The launched branch
(orange) starts at 2.873 with essentially no loss and ends at 1.450 with
0.3 dB/µm; the branches that reach a dead end part-way (red, purple, brown)
are guided modes going into cutoff as the Si narrows, where their loss rises
and they leave the physical set.

The TE-like fundamental descends continuously — 2.873, 2.541 (300 nm),
2.326, 2.159 (230), 1.800 (160), 1.693 (130), 1.642 (110), 1.592 (90) — and
becomes the slot mode: the wire mode and the slot mode are one supermode,
which is what an adiabatic converter needs. The TM-like fundamental stays
above it all the way and ends just below the silica line.

The output mode is bound by only 0.006 above the silica line: the
silica-filled 250 nm gap's MIM index is barely above 1.444 to begin with, and
250 nm of height leaves a slab of `V` ≈ 0.14. Its evanescent tail is about
2 µm, four times the window margin, so the value above carries PML
absorption; §1 quantifies that with a wide-window solve.

### Slot propagation length against the paper's Fig. 4 (gold 250 nm)

| `w_slot` | `n_eff` | `L_p` | confinement | MIM slab limit (infinite height, his gold) |
|---|---|---|---|---|
| 30 nm | 1.8349 + 0.0242j | 5.1 µm | 0.48 | 2.281 + 0.0153j, 8.1 µm |
| 50 nm | 1.6224 + 0.0175j | 7.1 µm | 0.58 | 1.988 + 0.0104j, 11.9 µm |
| 100 nm | 1.4921 + 0.0113j | 10.9 µm | 0.54 | — |
| 220 nm | 1.4517 + 0.0087j | 14.2 µm | 0.51 | — |

Same ordering and the same monotonic trend as the paper's Fig. 4 (`L_p`
grows with the slot width as the field leaves the metal), on the 0.5 µm-margin
platform window. The wider slots are bound by a few thousandths above the
silica index and their tails reach the PML; the wide-window solve of §1 says
how much of the loss is the window's.

## 3. The design point

**72.3 % (−1.41 dB) into the slot mode at the tip**; 71.3 % (−1.47 dB) at
the end of the 200 nm lead-out. Reflection 1.3e-4 into the physical modes
and 1.7e-3 into the Berenger set; nothing in the other physical forward
branch (the TM-like family is decoupled by the `y` mirror symmetry, as in
the paper's PMC plane); 0.094 of the launched `|a|²` leaves the lead-out as
non-physical forward amplitude, the scattered field the PML modes
represent. The paper's Set 2 number is ~95 %.

**What the paper's number is.** Kocabaş integrates the *total* Poynting
flux through a cut 1100 nm after the tip and back-propagates it to the tip
with the slot mode's own `L_p` (his §II and Fig. 7); the modal power,
obtained with the unconjugated inner product this project also uses, lies
below the total — for Set 1 he reports the total at "slightly less than
88 %" and the bound-mode fit under it. So 95 % is an upper bound on the
modal conversion, and the quantity computed here is the modal power. The
comparison is 72 % against something between ~90 and 95 %.

**Where the 28 % goes** (the design path matrix by matrix, same construction
as report 12 §3):

![interface profile](output/kocabas_3_interface_profile.png)

| factor | product along the path | reading |
|---|---|---|
| interface matrices (unrepresented mismatch) | 0.951 (−0.22 dB) | 51 steps of 10 nm in `w_si` and 5 nm in the gap; 0.1–0.9 % each, largest at the narrow end |
| propagation matrices | 0.850 (−0.71 dB) | the supermode's own metal loss integrates to 0.957 over the taper and 0.986 over lead-in and lead-out; the other ~11 % is the decay of amplitude the interfaces scattered into the Berenger set — the model's radiation |
| end of the cascade | 0.808 total, 0.713 in the slot branch | the difference is non-physical forward amplitude |

Two of the three loss channels are discretisation, not device: a smooth
1700 nm taper between these two modes has no 10 nm staircase to scatter
from, and a 20-mode basis cannot let scattered field re-couple further
along, as a 3-D FEM with PML does implicitly. The candidates for the gap to
the paper, in the order they should be tested:

1. **The parameter axes, and the gap axis first.** §5 runs the same device
   with its sections placed continuously instead of on the grid, and it
   reaches 84.1 %. Some of that is a sub-cell artefact and some is real, but
   it bounds the axis-snapping cost at about 12 points — half the deficit.
   The gap axis is the suspect: it steps 5 nm, so the gold wall stands still
   for several sections and then jumps, and per-step scattering is convex in
   the step. Refining it to 2.5 nm adds points to the *same* grid and is
   cheap.
2. **The cell.** A 2.5 nm cell halves every geometric step too, but on this
   471 × 347 grid it costs ~4× per solve and ~100 h for one path.
3. **The basis size**, 20 modes against the 40–50 a PML-EME normally wants.
4. **The window**, which inflates the slot-like branch's `Im n_eff` by about
   a third over the last 300 nm (§1) — worth about 1 % of power, not 20.

## 4. Transmission against the Si–gold gap and against taper length

![sweeps](output/kocabas_2_sweeps.png)

### Taper length (gap 75 nm, slot 250 nm, warm cache — every length is a new path over the same grid points)

| `L_taper` | slot mode at the tip | scattered (Berenger) amplitude at the lead-out | reflected |
|---|---|---|---|
| 500 nm | 65.4 % (−1.85 dB) | 0.193 | 6.7e-3 |
| 800 nm | 67.2 % (−1.72 dB) | 0.156 | 1.5e-3 |
| 1100 nm | 69.3 % (−1.59 dB) | 0.129 | 2.5e-4 |
| 1400 nm | 71.0 % (−1.49 dB) | 0.109 | 2.0e-4 |
| **1700 nm (paper)** | **72.3 % (−1.41 dB)** | 0.094 | 1.3e-4 |
| 2100 nm | 73.2 % (−1.35 dB) | | 1e-4 |
| 2500 nm | 73.6 % (−1.33 dB) | | 1e-4 |
| 3000 nm | 73.9 % (−1.31 dB) | | 1e-4 |

The curve rises monotonically and flattens beyond ~2 µm, with the
scattered amplitude falling roughly as `1/L` and the reflection as its
square — the adiabatic behaviour report 12's device could not show, because
there the 40-step staircase was the whole loss. The paper's 1700 nm sits at
the knee. The plateau near 74–75 % is not the device's limit: the
length-independent part is the staircase and basis loss of §3, and the
supermode's metal loss grows with `L` (0.957 at 1700 nm) and takes over
beyond it.

### Si–gold gap at the taper start (`w_gap`; slot end fixed at 250 nm, `L` = 1700 nm)

`w_gap` sets the Si–gold clearance where the taper *starts*; the clearance at
the tip is fixed by the slot at 125 nm. So the parameter really chooses how
the gold wall moves relative to the Si edge, and 125 nm is the special case
where it does not move at all.

| `w_gap` | clearance along the taper | path points | slot mode at the tip |
|---|---|---|---|
| 25 nm | 25 → 125 nm | 62 | 68.5 % (−1.65 dB) |
| **75 nm (paper)** | 75 → 125 nm | 52 | **72.3 % (−1.41 dB)** |
| 125 nm | 125 nm, constant | 42 | 73.4 % (−1.34 dB) |

Monotonic, and **the model cannot separate the physics from its own
discretisation here**: the dataset's gap axis has a 5 nm step, so a path that
opens the clearance by 100 nm visits exactly 20 grid points more than one that
holds it constant, and 42 + (125 − `w_gap`)/5 reproduces the third column
exactly. The 4.9 points of transmission between the ends of the sweep divided
by those 20 extra sections is 0.25 % each — the same size as the per-interface
mismatch of §3. A wall that stays put costs nothing to step past; that is true
of the physics *and* of the staircase, and this dataset cannot say in what
proportion.

What can be said: nothing in the range 25–125 nm reaches the paper's
efficiency, the ordering does not contradict it, and the paper's own choice of
75 nm is not reproduced as an optimum — this model has no interior optimum in
`w_gap`, exactly as report 12's device had none in length. Testing it properly
needs the gap axis refined to 2.5 nm and the comparison made at fixed section
count, which is the same convergence question as §3(i).

## 5. DBEME vs direct EME

Two direct runs, because on a two-axis dataset *where the sections are placed*
turns out to matter more than whether the modes came from a cache.

| route | sections | at the tip | `max|ΔT|` vs the dataset | cost |
|---|---|---|---|---|
| dataset (grid-snapped path) | 52 | 72.3 % (−1.41 dB) | — | **0 s** warm, 15 034 s cold once |
| direct, sections at the dataset's own points | 52 | pending | pending | ~9 900 s |
| direct, sections at uniform `z` (off grid) | 52 | **84.1 % (−0.75 dB)** | **1.2e-1** | 9 879 s |

**The off-grid row is not a failure of the cache; it is a different device.**
`DirectParametricPath` samples the parameter functions at uniform `z`, which
on report 12's single-axis linear taper lands exactly on the axis widths and
here does not: the taper carries 200 nm of lead-in, `w_si` steps 10 nm and
`gap` steps 5 nm, so uniform sampling gives widths like 398.6 and 388.9 nm and
gaps moving 1.25 nm at a time. Every metal edge then sits inside a cell, which
report 12 §7.3 measured as an `n_eff` alias of up to 0.1, and every interface
sees a smaller perturbation than the dataset's, whose gap can only move in
5 nm jumps. Both differences push the same way, and together they are worth
12 points of transmission. The demo now runs the aligned placement as well
(`_FixedPointPath`); the row above is the §5.8 check proper, and the
difference between the two direct rows is what snapping a continuous taper
onto this grid costs.

That number matters for §3: **about half the deficit against the paper may be
the gap axis, not the method.** A continuous-section EME of the same device
loses 16 %, the grid-snapped one 28 %. Refining the gap axis to 2.5 nm is
therefore the first thing to buy, ahead of the 2.5 nm *cell* — the axis is
cheap (more points on the same grid), the cell is not (4× per solve).

**The warm-cache claim**, which the two sweeps already support: the design
point cost 15 034 s cold; every one of the eight taper lengths in §4 then cost
**0 s**, because a different longitudinal path over the same `(w_si, gap)`
grid points solves nothing new. A direct EME pays ~9 900 s for each of them.

## 6. Conclusions and limits

**What this report establishes.**

1. The lossy PML backend runs a two-axis dataset end to end on a device with
   a published geometry, and lands at **72 % modal conversion** where the
   paper reports ~95 % total power. The gap is understood in kind if not in
   full: about 5 % is the interface staircase, about 11 % is amplitude
   scattered into the Berenger set and then absorbed, and the paper's number
   is a *total*-power figure that includes the scattered field this model
   discards (his §II, and the Set 1 example where the bound-mode fit sits
   below the total).
2. **The converter is adiabatic in this model and the report-12 device was
   not.** Transmission rises monotonically with taper length and flattens
   near 2 µm, and the scattered amplitude falls as roughly `1/L`. That is the
   qualitative behaviour a mode converter must have, and it is the first time
   this simulator has shown it on a plasmonic device — report 12's air-slot
   taper had a length-*independent* mismatch because a 10 nm edge step sheds a
   field no shift-invert basis holds. The difference is the device: here both
   ends are bound and silica-embedded, the supermode is continuous from
   `n_eff` 2.87 to 1.45, and the wall moves 5 nm at a time against a 250 nm
   slot.
3. **Basis membership is a first-class correctness condition for a
   shift-invert lossy solver.** With the target at 1.9 the wire's TE
   fundamental left the 16 nearest eigenvalues part-way along the path, the
   tracker linked a first-vertical-order branch instead, and the cascade read
   −20 dB — a plausible-looking wrong answer, not a crash. CLAUDE.md §5.13a
   now states the rule; §1 records the evidence.

**Limits.**

* The slot mode is bound by 0.006 above silica and its tail is four times the
  window margin, so its loss carries about a third of PML absorption (§1).
  Everything back-propagated over the 200 nm lead-out inherits that, which is
  1.4 % against a true 0.9 %.
* 20 modes, not the 40–50 a PML-EME basis usually wants; the Berenger set is
  the only representation of radiation here and it cannot re-couple.
* The gap axis is coarse against the physics it carries: 5 nm steps of the
  gold wall on a device whose whole taper moves that wall 50 nm. §5 bounds
  what that costs at ~12 points of transmission, which is the single largest
  identified term between this model and the paper, and the cheapest to
  remove.
* One wavelength, one metal thickness. The paper's Fig. 7 sweep over
  `h_Au` = 30–250 nm is a dataset per thickness, since `h_Au` changes the
  cross-section topology and therefore the mode problem.
* Modal power, not total power: comparable to the paper's cyan curve, not its
  blue one.

---

### Reproducing

```bash
cd examples
python demo_kocabas_converter.py --gaps 25 75 125 --direct
```

Stages are cached in `reports/output/kocabas_converter.json`, so a second run
is warm. The interface profile of §3 is a scratch analysis over the same
dataset; its numbers are in `reports/output/kocabas_interface_profile.json`.
The mode listings of §2 come from `kocabas_ends.py` and the window study from
`kocabas_window.json`.
