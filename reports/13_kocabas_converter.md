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
PML 0.25 µm on four edges; 16 modes about 1.9), scattering cascade,
output-side interface projection, `force_unitary=False`. The transmission is
the power in the slot mode at the end of the 200 nm lead-out, and — the
paper's convention — back-propagated to the tip with the slot mode's own
loss. The launched mode is the wire's TE-like branch (E along x, the slot's
polarisation): the 725 nm-tall wire's TM-like branch sits 0.008 above it.

---

## 1. Sanity gate

Rows from `reports/output/kocabas_converter.json` (TBD until the run
completes) and from the window study (`kocabas_window.json`):

| check (§5.x) | criterion | measured | pass |
|---|---|---|---|
| constant 250 nm slot, 1 µm: `T = exp(−2 k₀ Im n L)` | `|T − expected| < 1e-3` | TBD | |
| 5.1 reciprocity of the physical channel, design path | `< 1e-6` | TBD | |
| passivity: physical input columns of `|S|²` | `< 1.05` | TBD | |
| 5.8 DBEME vs direct EME on the design path | `max|ΔT| < 1e-3` | TBD | |
| 5.7 window: the weakly bound slot modes with 0.5 / 1.5 / 2.5 µm margins | reported | 250 nm slot: 1.4498 + 0.0086j (`L_p` 14.3 µm) / 1.4475 + 0.0060j (20.7) / 1.4476 + 0.0055j (22.3); 220 nm: 14.2 → 19.3 µm; **30 nm: identical** (1.8349 + 0.0243j) | — |
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

### The two ends and the middle (5 nm cell, 0.5 µm margins)

| cross section | branch | `n_eff` | `L_p` | confinement |
|---|---|---|---|---|
| wire 400 × 725 nm, gap 75 nm | TE-like (launched) | 2.4427 + 0.0002j | 0.63 mm | 0.91 |
| | TM-like | 2.4505 + 0.0002j | 0.75 mm | 0.98 |
| | next TE-like | 1.8408 + 0.0019j | 64 µm | 0.89 |
| mid-taper, Si 200 nm, gap 100 nm | TE-like | 1.9864 + 0.0027j | 46 µm | 0.92 |
| | TM-like | 1.9327 + 0.0003j | 363 µm | 0.91 |
| | slot-like | 1.4485 + 0.0250j | 4.9 µm | 0.45 |
| slot 250 nm, gold 250 nm | slot mode (port) | **1.4498 + 0.0086j** | **14.3 µm** | 0.49 |

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

TBD — transmission at the lead-out end and at the tip, against ~95 %.

## 4. Transmission against the Si–gold gap and against taper length

TBD — `kocabas_2_sweeps.png`.

## 5. DBEME vs direct EME

TBD.

## 6. Conclusions and limits

TBD.

---

### Reproducing

```bash
cd examples
python demo_kocabas_converter.py --gaps 25 50 75 100 125 150 --direct
```
