"""Continuous phase and group delay from a cascade sampled on a coarse grid.

The S-parameters of a few-hundred-micron device, taken every 10 nm, cannot be
differenced: ``arg(S)`` advances ~40 rad per step and wraps six or seven times
in between.  A modal reconstruction of the delay leaves the phase aliased;
this module recovers the phase itself.

**The idea.**  The light rides one tracked branch through an adiabatic device,
so the propagation phase ``phi_ref = sum_k (2 pi / lambda) n_eff[k, b_k] dz_k``
carries all the fast wavelength dependence, and ``arg(S) - phi_ref`` is the
scattering phase alone - milliradians per 10 nm.  That residual unwraps without
ambiguity, and ``phi_ref + unwrap(residual)`` equals ``arg(S)`` at every sample
while being continuous between them.

**Three traps it has to step around, all measured on a polarization
rotator-splitter cascade (2026-09-14):**

* *Power-weighting the branches is wrong for phase.*  Averaging ``n_eff``
  over every branch in proportion to its power lets truncated power parked
  on high-index branches of another core drag that average, and a
  dense-wavelength cascade shows it off by up to 0.45 rad per 10 nm.  The branch
  carrying the most power, ``b_k = argmax``, agrees with the dense cascade to
  <= 9e-3 rad.
* *Branch slots are not modes.*  The backend sorts, and between wavelengths
  high-index and cladding-level branches change slots (65 % of one device's
  slots move somewhere in the band).  ``np.gradient`` along the wavelength axis of a
  raw ``n_eff`` stack differences different modes; the largest step falls from
  0.78 to 0.02 once branches are matched first.
* *A mode's sign is not continuous in wavelength.*  ``_pin_gauge`` fixes each
  mode's sign from the first raster point above half its maximum, and for a
  mode with two near-equal lobes - a coupler's entry TE1 - which lobe that is
  can change between two wavelengths.  One coupler's TE1 -> cross amplitude
  went from -0.588 + 0.750j to +0.582 - 0.756j over 25 nm: same magnitude,
  propagation phase unchanged modulo 2 pi, sign flipped.  Nothing physical does
  that on a route at |S| 0.95, and an unwrap that believes it adds pi.  So flips
  are removed before unwrapping (:func:`reconstruct`), and independently
  recovered from the interface matrices themselves (:func:`gauge_chain`), which
  is also what lets a dense-wavelength cascade interpolate between two solves
  without its amplitude collapsing through zero.

No study imports here, so tests can use it without dragging the solver, or a
study's process-wide ``INTERFACE_PROJECTION``, into the suite.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

C_LIGHT = 299792458.0

#: Weight of the ``TE_pol`` mismatch against the ``n_eff`` mismatch when matching
#: branches.  ``n_eff`` moves ~3e-3 per 10 nm of dispersion; the polarisation
#: term only has to break near-ties, not override the index.
TE_WEIGHT = 0.05


def wrap(angle):
    """Into ``(-pi, pi]``."""
    return np.angle(np.exp(1j * np.asarray(angle, dtype=float)))


def match(n_a, te_a, n_b, te_b, te_weight=TE_WEIGHT):
    """``q`` such that branch ``q[j]`` at *b* is the same mode as branch ``j`` at *a*."""
    n_a, te_a = np.asarray(n_a, dtype=float), np.asarray(te_a, dtype=float)
    n_b, te_b = np.asarray(n_b, dtype=float), np.asarray(te_b, dtype=float)
    cost = (np.abs(n_a[:, None] - n_b[None, :])
            + te_weight * np.abs(te_a[:, None] - te_b[None, :]))
    rows, cols = linear_sum_assignment(cost)
    q = np.empty(len(n_a), dtype=int)
    q[rows] = cols
    return q


def aligned_gradient(neff_stack, te_stack, lams):
    """``d(n_eff)/d(lambda)`` per ``(lambda, section, branch)``, on physical modes.

    Every wavelength is matched to its neighbour section by section, outward
    from the middle of the grid; the matched ``n_eff`` are differenced (second
    order at the edges); the result is returned in each wavelength's own slot
    order, so ``dn[i, k, b]`` belongs to branch ``b`` *as stored at wavelength i*.

    :returns: ``(dn, stats)`` - ``stats`` says how much reordering there was.
    """
    stack = np.asarray(neff_stack, dtype=float)
    te = np.asarray(te_stack, dtype=float)
    count, points, branches = stack.shape
    ref = count // 2
    perm = np.zeros((count, points, branches), dtype=int)
    perm[ref] = np.arange(branches)
    for direction in (+1, -1):
        prev = ref
        for i in range(ref + direction, count if direction > 0 else -1,
                       direction):
            for k in range(points):
                slots = perm[prev, k]
                perm[i, k] = match(stack[prev, k, slots], te[prev, k, slots],
                                   stack[i, k], te[i, k])
            prev = i
    aligned = np.take_along_axis(stack, perm, axis=2)
    d_aligned = np.gradient(aligned, np.asarray(lams, dtype=float), axis=0,
                            edge_order=2 if count > 2 else 1)
    dn = np.empty_like(d_aligned)
    np.put_along_axis(dn, perm, d_aligned, axis=2)
    stats = {"slots_reordered_fraction":
             float(np.mean(perm != np.arange(branches)[None, None, :])),
             "max_step_aligned": float(np.abs(np.diff(aligned, axis=0)).max())
             if count > 1 else 0.0,
             "max_step_unaligned": float(np.abs(np.diff(stack, axis=0)).max())
             if count > 1 else 0.0}
    return dn, stats


def dominant_delay(neff, dz, weights, lam, dn):
    """``(phase, tau, mean n_g, branch per section)`` along the dominant branch.

    ``weights`` are forward powers per branch after each cascade step - two
    per section (propagation,
    then interface), so every other one is the state inside section ``k``.  On an
    adiabatic route the dominant branch is the launched one throughout; where
    the tracker relabels the light's mode, or light is carried across a crossing
    diabatically, it follows the light.
    """
    dz = np.asarray(dz, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) == 2 * len(dz):
        w = w[0::2]
    neff = np.real(np.asarray(neff))
    n = min(len(dz), len(w), neff.shape[0])
    modes = w.shape[1]
    ref = np.argmax(w[:n, :modes], axis=1)
    idx = np.arange(n)
    n_ref = neff[idx, ref]
    ng_ref = n_ref - lam * np.asarray(dn)[idx, ref]
    phase = float(np.sum(2.0 * np.pi / lam * n_ref * dz[:n]))
    tau = float(np.sum(ng_ref * dz[:n]) / C_LIGHT)
    return phase, tau, float(np.average(ng_ref, weights=dz[:n])), ref


def reconstruct(nm, s_port, phi_ref):
    """``phi_ref + unwrap(arg(S) - phi_ref)``, gauge flips removed, and its delay.

    A step of the wrapped residual beyond pi/2 is read as a sign flip of a port
    mode, not as physics: the residual is the scattering phase, it moves by
    milliradians per sample on these routes, and a real pi change would need
    ``|S|`` to pass through zero between two samples where it is ~0.95.  Flips are
    undone by a per-sample sign, and ``S_aligned = sign * S`` is the amplitude
    whose phase the result is.

    *Safe* means every remaining step is below pi/4 - with pi-periodic flips
    removed, that is what leaves the unwrap no choice to make.
    ``equals_arg_S_to_rad`` is the largest difference between the result and
    ``arg(S_aligned)`` modulo 2 pi, zero to rounding by construction.

    :returns: dict with ``phase_rad``, ``residual_rad``, ``gauge_signs``,
        ``gauge_flips``, ``S_aligned``, ``max_residual_step_rad``, ``safe``,
        ``equals_arg_S_to_rad`` and ``tau_from_phase_ps``.
    """
    s_port = np.asarray(s_port, dtype=complex)
    phi_ref = np.asarray(phi_ref, dtype=float)
    raw = np.angle(s_port * np.exp(-1j * phi_ref))
    flip = np.abs(wrap(np.diff(raw))) > 0.5 * np.pi
    signs = np.concatenate([[1.0], np.cumprod(np.where(flip, -1.0, 1.0))])
    aligned = s_port * signs
    residual = np.angle(aligned * np.exp(-1j * phi_ref))
    steps = np.abs(wrap(np.diff(residual)))
    unwrapped = np.unwrap(residual)
    phase = phi_ref + unwrapped
    omega = 2.0 * np.pi * C_LIGHT / (np.asarray(nm, dtype=float) * 1e-9)
    tau = np.gradient(phase, omega, edge_order=2 if len(phase) > 2 else 1)
    unit = aligned / np.where(np.abs(aligned) > 0, np.abs(aligned), 1.0)
    check = float(np.max(np.abs(np.angle(np.exp(1j * phase) / unit))))
    return {"phase_rad": phase, "residual_rad": unwrapped,
            "gauge_signs": signs, "gauge_flips": int(np.count_nonzero(flip)),
            "S_aligned": aligned,
            "max_residual_step_rad": float(steps.max()) if len(steps) else 0.0,
            "safe": bool(steps.max() < 0.25 * np.pi) if len(steps) else True,
            "equals_arg_S_to_rad": check,
            "tau_from_phase_ps": tau * 1e12}


def reorder_interface(matrix, q_left, q_right):
    """Re-index an interface S-matrix into another wavelength's slot order.

    The propagator's convention puts the transmissions on the diagonal blocks:
    columns ``[:N]`` are the left section's forward inputs and rows ``[N:]`` its
    backward outputs; rows ``[:N]`` are the right section's forward outputs and
    columns ``[N:]`` its backward inputs.  ``q_left[j]`` / ``q_right[j]`` give,
    for slot ``j`` of the target order, the slot to read in ``matrix``.
    """
    matrix = np.asarray(matrix)
    n = len(q_left)
    rows = np.concatenate([np.asarray(q_right), n + np.asarray(q_left)])
    cols = np.concatenate([np.asarray(q_left), n + np.asarray(q_right)])
    return matrix[np.ix_(rows, cols)]


def sign_interface(matrix, v_left, v_right):
    """Apply per-mode gauge signs to an interface S-matrix.

    A mode whose field changes sign changes the sign of every element it indexes,
    forward and backward alike (the backward partner is built from the same
    field), so rows ``[:N]`` / columns ``[N:]`` take the right section's signs and
    rows ``[N:]`` / columns ``[:N]`` the left section's.
    """
    v_left = np.asarray(v_left, dtype=float)
    v_right = np.asarray(v_right, dtype=float)
    rows = np.concatenate([v_right, v_left])
    cols = np.concatenate([v_left, v_right])
    return np.asarray(matrix) * rows[:, None] * cols[None, :]


def gauge_chain(interfaces_a, interfaces_b, chain, floor=0.2):
    """Gauge signs of wavelength *b* relative to *a*, along the light's path.

    ``interfaces_b`` must already be in *a*'s slot order (:func:`reorder_interface`)
    and ``chain[k]`` is the branch the light is on at point ``k`` (one entry per
    point, so ``len(interfaces) + 1``).  The transmitted element along the path,
    ``T[k][chain[k+1], chain[k]]``, scales as ``s_k s_{k+1}`` with the signs of
    the two modes it joins; comparing its sign at the two wavelengths therefore
    gives ``s_{k+1}`` from ``s_k``, starting from ``s_0 = +1``.

    Where the element is weaker than ``floor`` at either wavelength its sign is
    not trusted and the sign carries over; the count is returned so that is
    visible.

    :returns: ``(v, port_sign, untrusted)`` - ``v[k]`` is the sign vector for
        point ``k`` (+1 except on the chain branch), ``port_sign`` is
        ``s_0 * s_last``, the sign by which *b*'s port-to-port amplitude along
        the chain must be multiplied to be continuous with *a*'s.
    """
    count = len(interfaces_a)
    modes = np.asarray(interfaces_a[0]).shape[0] // 2
    chain = np.asarray(chain, dtype=int)
    if len(chain) != count + 1:
        raise ValueError(f"chain has {len(chain)} entries for {count} "
                         f"interfaces; it needs one per point ({count + 1})")
    s = np.ones(count + 1)
    untrusted = 0
    for k in range(count):
        ta = np.asarray(interfaces_a[k])[chain[k + 1], chain[k]]
        tb = np.asarray(interfaces_b[k])[chain[k + 1], chain[k]]
        if abs(ta) < floor or abs(tb) < floor:
            s[k + 1] = s[k]
            untrusted += 1
            continue
        same = np.sign(np.real(ta * np.conj(tb)))
        s[k + 1] = s[k] * (same if same != 0 else 1.0)
    v = np.ones((count + 1, modes))
    v[np.arange(count + 1), chain] = s
    return v, float(s[0] * s[-1]), untrusted


# ------------------------------------------------ frozen-interface cascade

def star(first, second):
    """Redheffer star product, ``first`` then ``second``, batched over leading axes.

    The block convention and algebra of
    `em_simulation.matrix_calculation_tool._redheffer_star_product` - rows
    ``[:N]`` forward outputs, columns ``[:N]`` forward inputs - with ``@`` and
    ``inv`` broadcasting, so a whole frequency grid cascades in one pass.
    """
    a = np.asarray(first, dtype=complex)
    b = np.asarray(second, dtype=complex)
    n = a.shape[-1] // 2
    a11, a12, a21, a22 = a[..., :n, :n], a[..., :n, n:], a[..., n:, :n], a[..., n:, n:]
    b11, b12, b21, b22 = b[..., :n, :n], b[..., :n, n:], b[..., n:, :n], b[..., n:, n:]
    eye = np.eye(n, dtype=complex)
    left = b11 @ np.linalg.inv(eye - a12 @ b21)
    right = a22 @ np.linalg.inv(eye - b21 @ a12)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=complex)
    out[..., :n, :n] = left @ a11
    out[..., :n, n:] = left @ a12 @ b22 + b12
    out[..., n:, :n] = a21 + right @ b21 @ a11
    out[..., n:, n:] = right @ b22
    return out


def propagate_into(interface, phase):
    """A section's propagation, then its interface, as one S-matrix.

    Propagation ``diag(phase, phase)`` reflects nothing, so the star product
    collapses to a scaling: the forward-input columns and the backward-output
    rows each take the section's phase once.  ``phase`` is ``(..., N)``, its
    leading axes a frequency grid.
    """
    phase = np.asarray(phase, dtype=complex)
    n = phase.shape[-1]
    out = np.broadcast_to(np.asarray(interface, dtype=complex),
                          phase.shape[:-1] + (2 * n, 2 * n)).copy()
    out[..., :, :n] *= phase[..., None, :]
    out[..., n:, :] *= phase[..., :, None]
    return out


def cascade(interfaces, phases):
    """Lumped S of ``prop_0, interface_0, prop_1, interface_1, ...``.

    The propagator's order (``_find_Smatrix_new_length``): section ``k``
    propagates on point ``k``'s modes, then meets interface ``k``.

    :param interfaces: ``(K, 2N, 2N)``.
    :param phases: ``(..., K, N)``, ``exp(i beta_k dz_k)``.
    """
    phases = np.asarray(phases, dtype=complex)
    lumped = None
    for k in range(len(interfaces)):
        step = propagate_into(interfaces[k], phases[..., k, :])
        lumped = step if lumped is None else star(lumped, step)
    return lumped


def smoothstep(t):
    """``3 t^2 - 2 t^3`` on ``[0, 1]`` - a blend weight flat at both ends."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def port_signs(reference, other):
    """Row and column signs that make ``other`` continue ``reference``.

    Two cascades of one device at the same frequencies, built on the interfaces
    of two different solved wavelengths.  Interior mode gauges cancel inside
    each, so what can differ is a sign per port mode - a whole row or column -
    on top of the small genuine change between the two snapshots.  Every sign
    pattern is scored by ``sum Re(reference * conj(sign * other))`` and the best
    kept; the first column is held at +1, since flipping everything together
    changes nothing.

    :param reference: ``(..., rows, columns)``.
    :param other: the same shape.
    :returns: ``(row_signs, column_signs, score)``.
    """
    ref = np.asarray(reference, dtype=complex)
    oth = np.asarray(other, dtype=complex)
    rows, cols = ref.shape[-2:]
    corr = np.real(ref * np.conj(oth))
    if ref.ndim > 2:
        corr = corr.sum(axis=tuple(range(ref.ndim - 2)))
    best = None
    for code in range(2 ** (rows + cols - 1)):
        bits = [1.0 - 2.0 * ((code >> b) & 1) for b in range(rows + cols - 1)]
        r = np.array(bits[:rows])
        c = np.array([1.0] + bits[rows:])
        score = float(np.sum(corr * r[:, None] * c[None, :]))
        if best is None or score > best[2]:
            best = (r, c, score)
    return best
