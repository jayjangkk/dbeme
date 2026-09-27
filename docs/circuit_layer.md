# Circuit layer

*Moved verbatim from `README.md` by task 16 step 1.5b (2026-09-27); the README is now a landing page that links here.*

DBEME cascades cross sections along `z`; it cannot close a loop. A ring, a
Mach-Zehnder, a coupled-resonator filter are *circuits* of DBEME building
blocks, and `dbeme/circuit/` is where those blocks become circuit
models (task 11). It is a consumer of S-matrices downstream of
`FDEBackend`, not a backend.

* **sax is the interface.** Every model is a sax model: a keyword-only
  function of the wavelength `wl` **in microns** returning a reciprocal
  `SDict` with `jnp` values, so gradients flow. `sax_model_from_samples`
  turns an S-matrix sampled at a few solved wavelengths into one:
  magnitude by a monotone cubic, phase through a polynomial fit of its
  *optical length* after unwrapping against a reference whose group index
  is right (`phase_fit.py`; a linear index between samples staircases the
  delay, and a plain unwrap fails past ~30 um at 10 nm steps).
  `waveguide.py` has straight, arc and ideal-coupler models; `ring.py` the
  analytical all-pass / add-drop ring (Bogaerts 2012, Yariv 2000) that
  every ring number is checked against; `roughness.py` a Payne-Lacey
  sidewall-scattering estimate. Importing the package enables JAX float64.
* **circulax** (`.venv-circuit`, pinned apart because it needs jax < 0.10
  and sax >= 0.15) compiles the same sax models unchanged; its steady-state
  solve batched over wavelength reproduces the sax spectrum of a ring to
  1e-9. S-matrix components are memoryless, so for the time domain
  `circuit/circulax_ext.py` adds an envelope delay line on circulax's own
  component API (a plugin, not a fork): the ring-down of a ring then
  matches `Q lambda / (2 pi c)` to 0.1 % (`tests_circuit/`, run in
  `.venv-circuit`). The layer is there for what comes after the ring:
  modulator-loaded rings, thermal tuning, laser + modulator + ring link
  budgets, gradient-based design of a CPO channel.
* **The ring** (`examples/demo_ring_resonator.py`, `reports/16`): the
  bus-ring point coupler is one DBEME path per gap over a dataset whose
  ring core moves while the bus stays fixed (`platforms.BusRingStrips`);
  the arc is the bent strip's `n_eff(lambda, 1/R)` with a loss budget from
  the PML solver (radiation) and the roughness model. Two lessons from
  building it are in `tasks/11` section 2.1: tie a moving edge's axis to
  the cell, and expect a translating guide to shed its mismatch at every
  interface regardless of mode count - the bus side is exact, the ring
  side is imposed by symmetry.
* **The workflow for circuit design** (`reports/17`, `tasks/14`): blocks
  as sax models of geometry and wavelength built from a dataset's own
  quantities, a differentiable circuit with a band-wise loss, Adam over
  lengths, then the composed blocks verified as DBEME cascades on cached
  points and the design handed to circulax. A 4-channel cascaded
  Mach-Zehnder demux (circulax's example 03) reaches 90 % contrast with
  the coupler's real dispersion in the loop; `reports/16_1` adds the
  electro-optic ring modulator.
