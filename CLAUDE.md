# CLAUDE.md — working agreement for the DBEME repo

Read `README.md` first. It describes *what the code is and what was changed*.
This file describes *how to work in it*: ground rules, commands, units. The
dataset doctrine, the validation backlog, the demo/report plan and the
references are in `docs/` - see *Read next* at the end.

**Owner:** Jaehyuck Jang (PIC / optical interconnect, CPO). Answers should be
concise and technical: conclusion first, then mechanism, then PIC-level
implication. Reference lists only when they add something.

---

## 0. Ground rules

* `ref_dbeme/` and `ref_emepy/` are **vendored upstream, never edited**. If a
  fix belongs upstream, patch the copy under `dbeme/` and note the
  divergence in `README.md` → *What changed relative to upstream*.
* The **dataset pickles are a cache, not source**. `dataset_info.py` is the
  source of truth. If a grid axis, mesh, window, mode count, wavelength or
  material index changes, the pickles are invalid — delete and regenerate.
  Never hand-edit a `.pkl`.
* **The solve grid is sacred.** Overlap integrals are only meaningful between
  cross sections sampled on the same `(x, y)` mesh. Any change to `window` or
  `MESH` invalidates every overlap in that dataset. `EmepyFDE` raises if the
  grid moves — do not work around that check, regenerate instead.
* New mode solvers go behind `dbeme/fde/base.py::FDEBackend`. That ABC
  is the whole point of the port; do not let solver specifics leak into
  `data_updater` or the geometry classes.
* Debugging style: identify the root cause, then propose the **minimal patch**.
  Do not rewrite a module unless asked.
* **Nothing platform-specific in this repository.** A foundry stack -
  thicknesses, indices, sidewall angles, GDS layers - and the devices built on
  it belong in a separate, private repository that uses this one as a
  submodule; so does any test that needs such a stack.
* Before changing `dbeme/`, commit or `git tag pre_<label>`, so the previous
  solver can always be restored.

## 1. Commands

```bash
.venv/Scripts/python -m pytest -q                  # tests/ + studies/*/tests — must stay green
.venv/Scripts/python -m pytest tests -q            # solver invariants only (public)
.venv-circuit/Scripts/python -m pytest tests_circuit -q   # circulax layer, its own venv
.venv/Scripts/python scripts/check_links.py        # every path a document cites exists
cd examples
python demo_linear_taper.py                        # figures -> examples/output/
python demo_bezier_bend.py
python validate_against_direct_eme.py              # DBEME vs uncached EME
```

Cold-cache runs solve cross sections (~1.3 s each at `MESH=160`); warm runs do
no mode solving. Always report both timings when claiming a speed-up.

**Long runs go through `examples/run_solver_job.py <script> [args]`.** This
machine is a hybrid i9-13900HX, and a detached single-threaded solver job
(the FEM eigensolve) gets scheduled onto its E-cores at ~7× the time — a
FEM point took 420 s in the run against 60 s in a shell-spawned process,
with the same session, priority, affinity and environment. The launcher
pins the process to the P-cores (logical 0–15), switches power throttling
off, raises the priority class and caps BLAS at 16 threads. If a run's
per-point time is far above the standalone solve, compare the main thread's
CPU time with the wall time before suspecting the code.

Solved paths that studies cache live under `cache/` (ignored by git).
`scripts/warm_check.py <script> [args]` runs a script and counts its path and
mode solves - a warm rerun must show zero path solves.

## 2. Units and conventions

| quantity | unit in code | note |
|---|---|---|
| length, width, gap, thickness | **metres** | `0.5e-6`, not `0.5` |
| curvature `κ` | **1/m**, signed | `κ = 1/R`; sign matters, see `docs/validation_backlog.md` §5.9 |
| wavelength | metres | dataset is single-λ today |
| `neff` array | `(2N,)` complex | forward modes first, then backward |
| overlap | `(2N, 2N)` | keyed `point -> {adjacent point -> matrix}` |

Plot conventions for reports: transmission in dB and linear, phase unwrapped,
group delay in fs, lengths in µm, `n_g` annotated.

---

## Read next

Sections 3-9 of this file moved to `docs/` on 2026-09-27 with their numbers
unchanged, so "§5.13a" or "CLAUDE.md §5.2" in older text means the file below.

| § | file | read it when |
|---|---|---|
| 3 | `docs/dataset_doctrine.md` | building or extending a dataset: one per cross-section family, cost control, grids on metal edges |
| 4-5 | `docs/validation_backlog.md` | before reporting any number: what is validated, and the §5.x checks with pass criteria (`force_unitary`, lossy-basis caveats, gauge, tracking) |
| 6-8 | `docs/demo_plan.md` | writing a demo or report: capability matrix, comparison protocol, report template |
| 9 | `docs/references.md` | citing: method, device and material references |

A device task is `tasks/<nn>_<name>.md` and its write-up `reports/<nn>_<name>.md`;
the numbers of the two series do not match - the task file names its report.
