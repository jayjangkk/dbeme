# Task 13 — depletion-mode ring modulator: circulax's example fed by DBEME

**Prompt to start with:**

> Read `CLAUDE.md`, `tasks/11` (Phases 1–5 status) and this file; the
> report is `reports/16_1_MRM.md`.

## Why

Jae asked for an electro-optic demo following circulax's `ring_modulator`
example (https://gdsfactory.github.io/circulax/examples/ring_modulator/)
"with DBEME". The example's ring is a temporal coupled-mode (t-CMT) model
with an electrical port and typed-in parameters. What DBEME contributes is
the *provenance* of every one of those parameters:

| t-CMT parameter | from |
|---|---|
| `ng`, `L`, roughness loss | `reports/output/ring/ring_loss.json` (Task 11) |
| `gamma` (through amplitude) | the point coupler at the chosen gap, `coupler_1550.json` bus side (Task 11) |
| `dn_eff/dV` → `v_to_wr`, `d alpha/dV` → `alpha1`, doped-core FCA → `alpha0`, `C_j(V)` | `studies/mrm/eo_parameters.py`: the TE0 field of the strip over a lateral pn junction (abrupt, declared doping), Soref–Bennett 1987 |
| `R_s` | assumed (50 Ω) |

The junction is a model; the DBEME content is the field overlap. Doping,
junction position and `R_s` are declared inputs, stated in the report.

## What exists

`dbeme/circuit/circulax_ext.py` (delay line, gated source),
`studies/mrm/eo_parameters.py`, `studies/mrm/circulax_mrm.py`
(`RingModulatorCMT` with the `v_e` port in the example's form, `BiasedAC`,
`make_nrz`), `tests_circuit/` (runs in `.venv-circuit`).

## Analyses (as the example, all in `.venv-circuit`)

1. optical step on/off → photon lifetime vs `tau` from the same `gamma`, `a`;
2. small-signal EO response 0.5–80 GHz: harmonic balance, transient sweep,
   analytic RC × second-order optical form;
3. NRZ eye at symbol rates set by the lifetime (`T_bit = 3 tau`, `1.5 tau`).

## Gate

| check | criterion |
|---|---|
| `dn_eff/dV` at 1 V reverse, 5e17/5e17 lateral junction | 10–50 pm/V (literature range for this doping); V_π·L 0.5–3 V·cm |
| photon lifetime from the optical step vs t-CMT `tau` | 2 % |
| EO response: transient vs harmonic balance vs analytic | 0.5 dB to the −3 dB point |
| eye at `T_bit = 3 tau` open, at `1.5 tau` visibly closing | qualitative |

## Status 2026-09-20 — done; `reports/16_1_MRM.md`

Gate measured: 28.4 pm/V, V_π·L 1.01 V·cm at 1 V ✓; lifetime 2.7 %
(marginal; energy decays with `τ/2`, the t-CMT `τ` being the amplitude
lifetime); transient vs HB ≤ 0.26 dB ✓, vs analytic ≤ 0.62 dB (marginal);
eyes 0.44 / 0.22 at 10.6 / 21.2 GBaud ✓. −3 dB at 9.6 GHz, RC pole 221 GHz:
photon-lifetime limited. Two lessons: the bias-induced resonance shift
must enter once (the ring applies `v_to_wr V` itself; folding it into
`f_resonance` too gave 1.7 linewidths of detuning), and the depletion
overlap must be integrated on a sub-cell grid or `dn/dV` is quantised.
Next design point: R = 5 µm (its coupler exists) and a 50 nm gap.

## Limits to state

No thermal, no self-heating, no carrier transit (the junction is quasi-
static), no series-resistance model (assumed), no absolute resonance
wavelength (Task 11), and the coupler's ring side is imposed by symmetry
(Task 11 §2.1). The ring is the Task 11 500 × 220 nm, R = 10 µm design:
its lifetime is tens of ps, so it is a few-GHz modulator; a faster one
needs a smaller, lossier, more strongly coupled ring — the numbers show
which lever (coupling or loss) buys bandwidth at which cost in extinction.
