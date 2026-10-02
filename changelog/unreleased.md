# dbeme — unreleased

Previous: v1.0.0 · Bump: *(decide at release, `docs/git_workflow.md` §1)*

Entries for changes made since v1.0.0. Template and rules in
[`README.md`](README.md); add the entry in the same commit as the change.

## Results changed

## Added

## Changed

## Fixed

## Examples, reports, datasets

* **Report 22 regenerated on the 1.0.0 solver.** The reciprocal interface
  projection was not on when the report was first written. Every
  cascade-derived number, table and figure is now regenerated, including the
  six §5.8 direct-EME runs and the Fig. 4 planes; no dataset changed.
  *Bump: PATCH* (no validated number moves beyond its tolerance).
  * **Best estimates** move by ≤ 0.009 dB: TE 0.92 / 0.86 / 0.94, TM
    1.56 / 1.59 / 1.88, PDL 0.64 / 0.73 / 0.94 dB at 1260 / 1310 / 1360 nm.
  * **The 1310 nm TM excess** over the paper is +0.4999 dB against the
    task's 0.5 dB criterion (was +0.49).
  * **Phase 4:** the seeded length search lands elsewhere on a flat surface
    (1310 nm 1.755 → 1.739 dB).
  * **§2.7, the two-mask rule:** the projection removes the gauge error's
    reflection (0.073 → 1.3e-4) but not its spurious 1.073 transmission.
* **`studies/edge_coupler/gate_detail.py`** localises the reciprocity defect
  with the projection off; with it on, the cascade is reciprocal by
  construction.
* **`summary.py` and `archive_checks.py`** drop their "previous formulas"
  markers: every direct run is now on the current code.
