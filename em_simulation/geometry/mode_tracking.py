"""Link the modes of one section to the next - docs/validation_backlog.md §5.13.

FDE returns modes sorted by ``n_eff``, so labels swap at an anti-crossing.
Tracking therefore has to follow **maximum overlap with the previous grid
point**, and §5.13 names the algorithm: Hungarian assignment.

The earlier implementation (identical in ``Geometry`` and ``DirectGeometry``)
did maximum overlap but not the *assignment*: it took ``argmax`` down each
column independently and accepted the link only above a 0.5 threshold.  Two
failures follow, and an anti-crossing triggers both:

* **Nothing enforced a bijection.**  Where two branches hybridise - which is
  what an anti-crossing *is* - both have comparable overlap with both
  predecessors, and independent argmaxes can hand them the same one.  One
  predecessor is then claimed twice and another never claimed.
* **The threshold had no fallback.**  A branch whose best overlap was 0.49 was
  declared a *new* mode, growing the name space past ``num_modes``;
  ``_reorder_data`` has only ``num_modes`` slots, so the unclaimed ones kept
  their initialised **zero**.

The observed symptom was ``n_eff = 0`` for two branches at a coupled taper's
output - a mode with no field behind it, whose overlap rows are junk, making
the interface matrix singular and the cascade diverge
(``reports/07_sirac_optimization.md`` §8.2).

With a fixed ``N``-mode basis on both sides of every interface the map **must**
be a permutation, so there is no case in which the greedy answer is right and
the assignment is not; where greedy already gives a permutation the two agree.
The threshold is dropped deliberately: a low best-overlap means the tracking is
*uncertain*, not that a mode appeared from nowhere.  :func:`link_quality`
exposes that uncertainty instead of silently acting on it.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_mode_links(overlap_matrices):
    """``mode_links[i, j]`` - which mode of section ``i`` branch ``j`` continues.

    :param overlap_matrices: ``(num_sections - 1, 2N, 2N)``; only the forward
        block is used, and only its magnitude - the gauge sign (§5.6) says
        nothing about *which* mode a branch continues into.
    :returns: ``(mode_links, quality)``, the second being the assigned overlap
        magnitude for each link, so a caller can report how well tracked the
        branches are.
    """
    overlaps = np.asarray(overlap_matrices)
    num_sections = overlaps.shape[0] + 1
    num_modes = int(overlaps.shape[1] / 2)
    strength = np.abs(overlaps[:, :num_modes, :num_modes])

    links = np.empty((num_sections - 1, num_modes), dtype=int)
    quality = np.empty((num_sections - 1, num_modes))
    for i in range(num_sections - 1):
        # Rows index section i, columns section i+1 (overlap_ab is
        # <E_i, H_{i+1}>), so maximise the total assigned overlap.
        rows, cols = linear_sum_assignment(-strength[i])
        links[i, cols] = rows
        quality[i, cols] = strength[i][rows, cols]
    return links, quality


def link_quality(quality):
    """Summarise how confidently the branches were tracked."""
    worst = np.min(quality, axis=1)
    return {
        "min_overlap": float(worst.min()),
        "worst_section": int(np.argmin(worst)),
        "mean_min_overlap": float(worst.mean()),
        "sections_below_half": int((worst < 0.5).sum()),
    }
