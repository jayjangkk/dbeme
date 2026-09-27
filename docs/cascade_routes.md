# Two cascade routes

*Moved verbatim from `README.md` by task 16 step 1.5b (2026-09-27); the README is now a landing page that links here.*

`SingleEME` can build its section scattering matrices two ways, selected by
`SingleEME.SMATRIX_METHOD` or per call with `calc_Smatrix(method=...)`:

| route | what it does | when |
|---|---|---|
| `transfer` | assemble a transfer matrix per interface, convert to S at the end — the original code path | lossless datasets: every published number was produced this way and stays bit-for-bit reproducible |
| `direct` | assemble the scattering matrix straight from the overlaps; never forms a transfer matrix | lossy or PML bases, where a transfer matrix's backward block grows as `exp(+k₀n″Δz)` and cascading amplifies it without bound |
| `auto` *(default)* | `transfer` if the backend declares its modes lossless, `direct` otherwise | — |

The two are algebraically identical (`S = [[T12, −R21], [R12, T21]]` is what
converting the transfer matrix yields, with the `inv(T21)` round trip
cancelling) and agree to ~1e-7 — the transfer route's complex64 storage —
everywhere except at an interface whose mode-matching matrix is near singular,
where they differ by ~2e-5 in the guided block and ~1e-3 in the radiation
block. Neither number there is physics, which is why lossless data keeps its
published route. `examples/verify_smatrix_routes.py` is the gate;
[reports/06](../reports/06_pml_phase1_gate.md) has the measurements.
