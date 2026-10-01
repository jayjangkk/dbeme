import contextlib

import numpy as np
from copy import deepcopy

from ... import matrix_calculation_tool as mct
from ...geometry.geometry import Geometry
from ..propagator import Propagator


def cap_columns(smatrix):
    """Scale every column of ``smatrix`` (``(..., m, m)``) whose power
    ``sum_i |S_ij|^2`` exceeds 1 down to unit power; columns at or below 1 are
    returned unchanged.  See :attr:`SingleEME.INTERFACE_COLUMN_CAP`."""
    power = np.sum(np.abs(smatrix) ** 2, axis=-2, keepdims=True)
    return smatrix / np.sqrt(np.maximum(power, 1.0))


@contextlib.contextmanager
def interface_switches(**switches):
    """Set :class:`SingleEME`'s class-level interface switches for the body of
    a ``with`` block and restore them afterwards - e.g.
    ``interface_switches(INTERFACE_RECIPROCAL=False)`` around a reciprocity
    check, which on a lossy basis measures nothing with the projection on."""
    saved = {name: getattr(SingleEME, name) for name in switches}
    try:
        for name, value in switches.items():
            setattr(SingleEME, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(SingleEME, name, value)


class SingleEME(Propagator):
    """
    The simulation algorithm basically follows the thesis "P. Bienstman, “Rigorous and efficient modelling of wavelenght scale photonic components / Peter Bienstman.,” 2001."
    To see the detail of the mathematical backgorund and terms, see chater2 of the thesis. 
    """
    def __init__(self, geometry:Geometry, force_passive = False, force_unitary = False):
        super().__init__(geometry, force_passive=force_passive, force_unitary=force_unitary)


        self.overlap_forward_ab = self.output_data["overlap_ab"][:,:self.mode_count, :self.mode_count]
        self.overlap_forward_ba = self.output_data["overlap_ba"][:,:self.mode_count, :self.mode_count]
        self.beta_forward = self.output_data["beta"][:,:self.mode_count]

        self._interface_Tmatrix = None  # ndarray with shape (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count)

        # status
        self._is_interface_Tmatrix_calcualted = False


    #: How the section scattering matrices are built.
    #:
    #: ``"transfer"`` is the original route: assemble a transfer matrix per
    #: interface, then convert to scattering form.  ``"direct"`` assembles the
    #: scattering matrix straight from the overlaps and never forms a transfer
    #: matrix at all.  ``"auto"`` - the default - picks ``transfer`` when the
    #: backend declares its modes lossless and ``direct`` otherwise, and an
    #: explicit ``method=`` always wins.
    #:
    #: Why not simply switch to ``direct``.  The two routes are algebraically
    #: identical (see :meth:`_calc_interface_Smatrix`) but not numerically so
    #: everywhere: at an interface whose mode-matching matrix is near singular
    #: - a radiation mode sitting a hair above cutoff, condition number 3e7 on
    #: the README taper - the transfer route's T->S conversion applies a second
    #: magnitude screen that rewrites every block, and the two then differ by
    #: ~1e-3 in the radiation block and ~2e-5 in the guided one.  Neither value
    #: there is physics.  Keeping lossless datasets on the route they were
    #: published with makes those numbers bit-for-bit reproducible; lossy and
    #: PML bases, which the transfer route cannot carry at all, get the stable
    #: one.
    #:
    #: They stop being numerically identical exactly where it matters.  A
    #: transfer matrix propagates the backward block as ``exp(-i beta dz)``,
    #: which for a lossy or evanescent mode is ``exp(+k0 n'' dz)`` and *grows*;
    #: cascading then amplifies it without bound.  A plasmonic mode
    #: (``n'' ~ 0.08``) reaches e^2 over a 600 nm taper, and the evanescent
    #: modes a sub-wavelength converter needs (``n'' ~ 1-2``) reach e^48.  The
    #: scattering form has no growing exponential anywhere.
    SMATRIX_METHOD = "auto"

    #: Singular-value cutoff for the mode-matching inverse, used by the
    #: **scattering route only**.
    #:
    #: A lossless guided basis never reaches it: measured condition numbers of
    #: ``O_ab + O_ba^T`` run 1.2 to 35, i.e. a smallest-to-largest ratio of
    #: 0.029, thirty times above this cutoff, so nothing is discarded and the
    #: result is the exact inverse.  A PML basis does reach it, and discarding
    #: those directions takes the measured output energy from 8.8 to 1.03 at
    #: N = 20 and from 2.8e3 to 1.4 at N = 40, with the guided transmission
    #: recovering from 0.956 to 0.997.
    #:
    #: It is deliberately *not* applied on the transfer route.  There the
    #: result is inverted again to build the transfer matrix, and truncating
    #: first makes that inverse rank deficient - measured energy went the wrong
    #: way, 2.8e3 to 2.7e8.  Never enable this without the scattering route.
    #:
    #: 1e-3 -> 1e-2 on 2026-09-20.  A shift-invert window of ``N`` modes has
    #: an edge, and at a step where the ``N``-th member changes - one point's
    #: last continuum mode has no partner at the next (best overlap 0.30) -
    #: the forced bijection leaves ``O_ab^T + O_ba`` with one singular value
    #: at 1.1e-3 of the largest, just above the old cutoff; the output-side
    #: projection then returned 3.2x the power of a physical input at that
    #: one interface (Kocabas, 1 nm-wall FEM dataset, the 50 nm gap: 3.6 %
    #: reflected, deficit -1.4 %).  Any cutoff from 2e-3 to 3e-2 repairs it
    #: (85.5 %, worst interface 1.012) and moves the clean paths by nothing
    #: to four digits; 1e-2 is an order of magnitude clear of the break and
    #: still three times below the lossless basis's floor of 0.029.
    INTERFACE_RCOND = 1e-2

    #: Which section's modes the interface continuity equations are projected
    #: on - see :meth:`_calc_transmission_matrix`.  ``"auto"``: ``"input"``
    #: (upstream's form) for a lossless basis, ``"output"`` for a lossy one.
    #: The transfer route inverts ``T21``, and the output-side ``T`` inherits
    #: the near-zero overlaps of radiation modes that barely exist on one side
    #: of an interface - on the README taper that inverse then diverges - so
    #: the lossless datasets, which run the transfer route, keep the form that
    #: was validated with it; a lossy basis runs the direct route, where no
    #: inverse of ``T`` is ever formed and the bounded form is safe.
    INTERFACE_PROJECTION = "auto"

    #: Cap every input column of an interface scattering matrix at unit power
    #: (``sum_i |S_ij|^2 <= 1``; columns above are scaled down, none up).
    #: ``"auto"``: on for a lossy basis, off for a lossless one; ``True`` /
    #: ``False``: always / never.  Default ``False`` - see the last paragraph.
    #:
    #: Why (2026-09-23, report 13 section 12).  On a lossy truncated basis
    #: the mode-matching projection is not passive for the discretised
    #: continuum: a Berenger mode of one section is represented on the other
    #: side by a different set, the two overlaps of the interface differ by
    #: ~0.2, and the transmission column of such an input carries 5-16 % more
    #: power than it receives (83 % of all columns exceed 1 slightly; the
    #: physical columns by at most 0.8 %).  A cascade compounds that: power
    #: that has leaked into the continuum is re-amplified at every later
    #: interface, and past a few hundred interfaces the physical channel
    #: itself reports more power than launched (a 311-interface Kocabas path
    #: gave 6.9 for unit input, a 231-interface one 0.91).  It is not a
    #: resonance (round-trip spectral radius < 0.3), not the pseudo-inverse
    #: cutoff (no effect from 1e-2 to 0.2) and a singular-value clip is wrong
    #: here (the 2-norm is not power in an unconjugated-normalised basis; it
    #: cut the guided channel from 85.6 to 54 %).  Capping the columns is the
    #: weakest statement of passivity in the basis's own measure - no input
    #: yields more than it carries - and it moves the guided channel by
    #: 0.3 points on the 231-interface path while making the 311-interface
    #: one sane (87.3 % with deficit -0.19 -> 85.9 % with +0.09).
    #:
    #: Off by default since 2026-09-30 (``"auto"`` or ``True`` opts in).  The
    #: 6.9 above came with the two-mask sign gauge of
    #: ``Geometry._equalize_overlap_phase``: with one gauge and upstream's
    #: reflection block that path is passive uncapped (0.96).  The continuum
    #: gain described above is real, though: with the reflection block from
    #: field continuity (:meth:`_calc_reflection_matrix`) the path's
    #: transmission chain alone compounds to 38 (reflections zeroed), which the
    #: cap holds to 0.93 - but with the reflections the capped path still
    #: reaches 28, so the cap does not keep it passive.  Every other lossy path
    #: re-run (reports 12, 13 and 22) is passive uncapped (at most 0.91).
    #: Where it acted the cap cost up to 2.4 points (report 13 design point and
    #: variants) and 0.12 dB (report 22, 1260-1360 nm), and it degraded
    #: reciprocity there (at 1310 nm 0.41 % against 0.025 %).  README, "What
    #: changed relative to upstream"; ``examples/kocabas_gauge_check.py``.
    #: (2026-10-01: the 38 and the 28 both come from the self-overlap floor
    #: described under :attr:`INTERFACE_RECIPROCAL` and
    #: :attr:`INTERFACE_SELF_OVERLAP`, not from the truncation of the mode
    #: set; with the reciprocal projection the capped path is passive, 0.93.)
    INTERFACE_COLUMN_CAP = False

    #: Project every interface matrix on its reciprocal part:
    #: ``R12 <- (R12 + R12^T)/2``, ``R21`` likewise, ``T12 <- (T12 + T21^T)/2``
    #: and ``T21 <- T12^T``.  ``"auto"`` (default): on for a lossy basis, off
    #: for a lossless one; ``True`` / ``False``: always / never.
    #:
    #: Why (2026-10-01).  Lorentz reciprocity in the unconjugated
    #: normalisation makes the channel-ordered interface matrix
    #: ``[[R12, T21], [T12, R21]]`` symmetric.  The uncorrected transmission
    #: blocks satisfy ``T21 = T12^T`` by construction, on any basis; the
    #: reflection blocks are symmetric only on a complete basis whose
    #: self-overlap ``M = 1/2 int e_i x h_j`` is ``I``.  Biorthogonalisation
    #: forces only the symmetric part of ``M`` to ``I``.  On the FEM basis the
    #: antisymmetric part ``A`` keeps a continuum column norm of ~0.05 - the
    #: overlaps are taken over the inner window, which leaves out the absorber
    #: ring where continuum modes still carry field, and the window-boundary
    #: term of the Lorentz identity tracks ``A`` on saved fields - and with the
    #: output-side projection (the lossy default) an interface between a
    #: section and itself reflects ``R = -A (I - A^2)``: antisymmetric, in
    #: phase with the transmission, and the same at a 0.25 nm step as at
    #: 10 nm.  On report 13's 311-interface FEM path those reflections build
    #: 36 of the 38 units of lumped gain and +7 points at the tip.  The
    #: symmetric part removes that zero-step reflection exactly; at a finite
    #: step it differs from the self-overlap-corrected reflection by
    #: ``O(A dz)``, a relative ``O(A)`` error on that step's reflection.  On
    #: FD-PML bases truncation and degenerate continuum pairs make ``R``
    #: non-symmetric as well, and the projection removes that too.  Off on
    #: lossless bases, which keep the validated transfer route bit for bit
    #: (its T->S screen breaks reciprocity again, and under ``force_unitary``
    #: a below-cutoff channel of the README taper flips).  It makes a lossy
    #: cascade reciprocal by construction, so a reciprocity check there
    #: measures the basis only with this switched off
    #: (:func:`interface_switches`).
    INTERFACE_RECIPROCAL = "auto"

    #: Correct the projected continuity equations for each section's
    #: self-overlap ``M = I + A`` (see :attr:`INTERFACE_RECIPROCAL`), with
    #: ``A`` estimated from the antisymmetric part of the uncorrected
    #: reflection blocks - :meth:`_self_overlap_corrected_blocks`.  ``False``
    #: (default) or ``"estimate"``.
    #:
    #: Why.  The output-side projection with ``M = I`` also leaves a
    #: transmission floor: a zero step transmits ``I - A^2``, a gain of
    #: ``2 Re(A^T A)_jj`` per continuum column (``2 |A e_j|^2`` for real
    #: ``A``; 0.3-0.4 % median, 3.8 % for the window-edge modes), the same at
    #: every step size, against 0.39 % damping per 5.3 nm section of the
    #: weakly damped FEM continuum.  Over report 13's 311 interfaces it
    #: compounds to 38.5 in forward continuum with the reflections zeroed
    #: (38.9 under the reciprocal projection, which leaves it in place).
    #: With the exact ``M`` of sections 200-311 (fresh solves) the equations
    #: remove it (that segment's transmission chain 43.4 -> 1.004); the
    #: estimate matches that ``A`` to 2-3 % and the cascade to 0.02-0.03
    #: points (85.59 % against 85.61-85.62 %, passivity 0.934).  With
    #: :attr:`INTERFACE_RECIPROCAL` on, the corrected blocks - which break
    #: ``T21 = T12^T`` by up to 5.4e-2 there - are projected afterwards (the
    #: tip moves 0.01 points).  Opt-in, because the estimate holds only while
    #: the reflection asymmetry is the self-overlap error: on report 22's
    #: FD-PML bases it reaches norm 6.6 (degenerate continuum pairs) and the
    #: correction destabilises the continuum.  Above
    #: :attr:`SELF_OVERLAP_ESTIMATE_LIMIT` it raises instead of guessing.
    INTERFACE_SELF_OVERLAP = False

    #: Largest spectral norm of an estimated ``A`` that ``"estimate"``
    #: accepts.  It guards the zero-step series only (the estimate is
    #: ``A - A^3 + ...``, about 10 % off at 0.3).  At a real step on a truncated
    #: basis, reflection asymmetry from truncation enters the estimate too, and
    #: that error does not shrink with ``A`` (FD hs_c20 section 56: 27 % off at
    #: a column median of 0.014), so below the limit the estimate is unverified
    #: wherever that asymmetry competes with ``A``.  Against exact ``M`` it is
    #: checked only on the 312-section FEM path (largest norm 0.19); report
    #: 13's 40-mode FD path reaches 0.26 and passes without such a check.
    SELF_OVERLAP_ESTIMATE_LIMIT = 0.3

    #region main functions
    def resolve_method(self, method=None):
        """Turn ``None``/``"auto"`` into the route that will actually run."""
        method = (method or self.SMATRIX_METHOD).lower()
        if method == "auto":
            method = "transfer" if getattr(self, "_lossless", True) else "direct"
        if method not in ("transfer", "direct"):
            raise ValueError("method must be 'transfer', 'direct' or 'auto'")
        return method

    def calc_Smatrix(self, method=None):
        """Section scattering matrices, by either route.

        :param method: ``"transfer"``, ``"direct"`` or ``"auto"``; defaults to
            :attr:`SMATRIX_METHOD`.
        """
        method = self.resolve_method(method)
        if method == "direct":
            return self._calc_Smatrix_direct()

        if not self._is_tmatrix_calculated:
            self.calc_Tmatrix()
        # smatrix = mct._convert_3Dmatrix(self.tmatrix)
        smatrix = mct._convert_3Dmatrix_ray(self.tmatrix)

        if self._force_unitary:
            # smatrix = mct._find_nearest_unitary_3D(smatrix)
            smatrix = mct._find_nearest_unitary_3D_ray2(smatrix)
        if self._force_passive:
            smatrix = mct._make_passive_3D(smatrix)
        self.smatrix = smatrix
        self._is_smatrix_calculated = True

    def calc_Tmatrix(self):
        interfaces = self._calc_interface_Tmatrix()
        phase_propagations = self._calc_phase_propagation_Tmatrix()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
            lengths_per_matrix[2*i] = delta_zs[i]

        self.tmatrix = deepcopy(total_matrices)
        self._lengths_per_matrix = lengths_per_matrix
        self._is_tmatrix_calculated = True
        return total_matrices
    
    def _calc_Smatrix_direct(self):
        """Scattering matrices assembled without ever forming a transfer matrix.

        Same interleaving as :meth:`calc_Tmatrix` - propagation through a
        section, then the interface at its end - so the result cascades with
        :func:`_redheffer_star_product` in the same order as the converted
        route, and ``_lengths_per_matrix`` keeps its meaning.
        """
        interfaces = self._calc_interface_Smatrix()
        propagations = self._calc_propagation_Smatrix()
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])

        total = np.zeros(
            (2 * self.section_count - 2, 2 * self.mode_count, 2 * self.mode_count),
            dtype=complex,
        )
        lengths_per_matrix = np.zeros(2 * self.section_count - 2, dtype=float)
        for i in range(self.section_count - 1):
            total[2 * i] = propagations[i]
            total[2 * i + 1] = interfaces[i]
            lengths_per_matrix[2 * i] = delta_zs[i]

        if self._force_unitary:
            total = mct._find_nearest_unitary_3D_ray2(total)
        if self._force_passive:
            total = mct._make_passive_3D(total)

        self.smatrix = total
        self._lengths_per_matrix = lengths_per_matrix
        self._is_smatrix_calculated = True
        return total

    def _calc_interface_Smatrix(self):
        """The interface scattering matrix, straight from the overlaps.

        Converting the transfer matrix built by :meth:`_calc_interface_Tmatrix`
        gives, block by block::

            n22 = inv(m22) = inv(inv(T21))                       = T21
            n12 = m12 inv(m22) = R21 inv(T21) T21                = R21
            n21 = -inv(m22) m21 = -T21 (-inv(T21) R12)           = R12
            n11 = m11 - m12 inv(m22) m21
                = (T12 - R21 inv(T21) R12) + R21 inv(T21) R12    = T12

        so the scattering matrix is just ``[[T12, R21], [R12, T21]]`` and the
        whole ``inv(T21)`` round trip cancels.  (Upstream's conversion put a
        minus on ``n12``, and this matrix read ``[[T12, -R21], [R12, T21]]``;
        with upstream's reflection formula that made a lossless Fresnel step
        ``[[t, r21], [-r12, t]]`` - see :meth:`_calc_reflection_matrix`.)
        That inversion is the only
        reason the transfer route needs ``T21`` to be invertible, and it is
        what fails once a near-null-norm PML mode enters the basis: the
        interface matrix reaches a condition number of 5.6e4 and the cascade
        reports output energy of 3070 against a physical maximum of 1.

        The blocks are those of :meth:`_calc_interface_blocks`, after the
        reciprocal projection and the optional self-overlap correction.
        """
        T12, T21, R12, R21 = self._calc_interface_blocks(self.INTERFACE_RCOND)

        interface_Smatrix = np.zeros(
            (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count),
            dtype=complex,
        )
        interface_Smatrix[:, :self.mode_count, :self.mode_count] = T12
        interface_Smatrix[:, :self.mode_count, self.mode_count:] = R21
        interface_Smatrix[:, self.mode_count:, :self.mode_count] = R12
        interface_Smatrix[:, self.mode_count:, self.mode_count:] = T21
        if self._column_cap_enabled():
            interface_Smatrix = cap_columns(interface_Smatrix)
        return interface_Smatrix

    def _column_cap_enabled(self):
        mode = self.INTERFACE_COLUMN_CAP
        if mode == "auto":
            return not getattr(self, "_lossless", True)
        if mode in (True, False):
            return bool(mode)
        raise ValueError(f"INTERFACE_COLUMN_CAP must be 'auto', True or False, got {mode!r}")

    def _reciprocal_enabled(self):
        mode = self.INTERFACE_RECIPROCAL
        if mode == "auto":
            return not getattr(self, "_lossless", True)
        if mode in (True, False):
            return bool(mode)
        raise ValueError(f"INTERFACE_RECIPROCAL must be 'auto', True or False, got {mode!r}")

    def _calc_interface_blocks(self, rcond=None):
        """``T12, T21, R12, R21`` of every interface, with
        :attr:`INTERFACE_SELF_OVERLAP` and :attr:`INTERFACE_RECIPROCAL`
        applied.  Both routes build their interface matrices from these."""
        ab, ba = self.overlap_forward_ab, self.overlap_forward_ba
        T12 = self._calc_transmission_matrix(ab, ba, rcond)
        T21 = self._calc_transmission_matrix(ba, ab, rcond)
        R12 = self._calc_reflection_matrix(ab, ba, T12)
        R21 = self._calc_reflection_matrix(ba, ab, T21)

        mode = self.INTERFACE_SELF_OVERLAP
        if mode == "estimate":
            T12, T21, R12, R21 = self._self_overlap_corrected_blocks(R12, R21, rcond)
        elif mode is not False:
            raise ValueError(f"INTERFACE_SELF_OVERLAP must be False or 'estimate', got {mode!r}")

        if self._reciprocal_enabled():
            T12 = 0.5 * (T12 + np.swapaxes(T21, -1, -2))
            T21 = np.swapaxes(T12, -1, -2)
            R12 = 0.5 * (R12 + np.swapaxes(R12, -1, -2))
            R21 = 0.5 * (R21 + np.swapaxes(R21, -1, -2))
        return T12, T21, R12, R21

    def _self_overlap_corrected_blocks(self, R12, R21, rcond=None):
        """The interface blocks with each section's self-overlap in the
        projected continuity equations (a Galerkin projection with the
        modes' Gram matrix, as CAMFR's non-orthogonal interface does).

        With ``M_x = 1/2 int e_x x h_x`` (``a`` the section left, ``b`` the
        one entered), continuity projected on the modes of ``b`` reads
        ``O_ab^T (I + R) = M_b^T T`` and ``O_ba (I - R) = M_b T``: the
        uncorrected problem with ``O_ab -> O_ab inv(M_b)`` and
        ``O_ba -> inv(M_b) O_ba``.  Projected on ``a`` it reads
        ``M_a^T (I + R) = O_ba^T T`` and ``M_a (I - R) = O_ab T``: the
        uncorrected problem with ``O_ab -> inv(M_a) O_ab`` and
        ``O_ba -> O_ba inv(M_a)``.  ``T`` is solved on the side
        :attr:`INTERFACE_PROJECTION` picks and ``R`` on ``a``, as before, so
        ``M = I`` gives the uncorrected blocks and a zero step ``(I, 0)``.

        ``M`` is not stored in the datasets.  Its symmetric part is ``I``
        (biorthogonalisation), and its antisymmetric part ``A`` is what makes
        the uncorrected ``R12`` non-symmetric - a zero step gives
        ``R12 = -A (I - A^2)`` on the output side, ``-A`` on the input side -
        so ``A_a = -(R12 - R12^T)/2`` and ``A_b = -(R21 - R21^T)/2``, good to
        ``O(A^3)`` at a zero step.  On a complete basis with ``M = I``,
        ``R12`` is symmetric, ``A = 0`` and nothing changes.
        """
        def antisymmetric(R):
            R = np.asarray(R, dtype=complex)
            return -0.5 * (R - np.swapaxes(R, -1, -2))

        A_a, A_b = antisymmetric(R12), antisymmetric(R21)
        if A_a.size:
            norms = np.maximum(np.linalg.norm(A_a, ord=2, axis=(-2, -1)),
                               np.linalg.norm(A_b, ord=2, axis=(-2, -1)))
            worst = int(np.argmax(norms))
            if norms[worst] > self.SELF_OVERLAP_ESTIMATE_LIMIT:
                raise ValueError(
                    f"INTERFACE_SELF_OVERLAP='estimate': the estimated self-overlap error reaches "
                    f"norm {norms[worst]:.3g} at interface {worst} (limit "
                    f"{self.SELF_OVERLAP_ESTIMATE_LIMIT}); the reflection asymmetry there is not a "
                    f"small self-overlap error, so the estimate does not hold")
        eye = np.eye(A_a.shape[-1])
        M_a, M_b = eye + A_a, eye + A_b

        def on_left(o_ab, o_ba, M):       # continuity projected on the section left
            return np.linalg.solve(M, o_ab), np.swapaxes(np.linalg.solve(np.swapaxes(M, -1, -2),
                                                                         np.swapaxes(o_ba, -1, -2)), -1, -2)

        def on_right(o_ab, o_ba, M):      # ... on the section entered
            return (np.swapaxes(np.linalg.solve(np.swapaxes(M, -1, -2), np.swapaxes(o_ab, -1, -2)), -1, -2),
                    np.linalg.solve(M, o_ba))

        projection = self.INTERFACE_PROJECTION
        if projection == "auto":
            projection = "input" if getattr(self, "_lossless", True) else "output"
        ab = np.asarray(self.overlap_forward_ab, dtype=complex)
        ba = np.asarray(self.overlap_forward_ba, dtype=complex)
        for_T12 = on_left(ab, ba, M_a) if projection == "input" else on_right(ab, ba, M_b)
        for_T21 = on_left(ba, ab, M_b) if projection == "input" else on_right(ba, ab, M_a)
        T12 = self._calc_transmission_matrix(*for_T12, rcond)
        T21 = self._calc_transmission_matrix(*for_T21, rcond)
        R12 = self._calc_reflection_matrix(*on_left(ab, ba, M_a), T12)
        R21 = self._calc_reflection_matrix(*on_left(ba, ab, M_b), T21)
        return T12, T21, R12, R21

    def _calc_propagation_Smatrix(self, length_ratio=1.0):
        """Propagation in scattering form: the *same* block twice.

        The transfer form carries ``exp(+i beta dz)`` forward and
        ``exp(-i beta dz)`` backward, and the second of those grows
        exponentially for any mode with loss or decay.  In scattering form both
        waves travel forward along their own direction, so both blocks are the
        decaying ``exp(+i beta dz)`` and nothing can grow.

        Converting the transfer version confirms it:
        ``n22 = inv(exp(-i beta dz)) = exp(+i beta dz)``.
        """
        diagonal = np.eye(self.mode_count, dtype=complex)
        i, j, _ = np.meshgrid(
            np.arange(0, self.section_count - 1),
            np.arange(0, self.mode_count),
            np.arange(0, self.mode_count),
            indexing="ij",
        )
        steps = self.output_data["EME_delta_zs"][i] * length_ratio
        forward = np.exp(1j * self.beta_forward[i, j] * steps) * diagonal

        result = np.zeros(
            (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count),
            dtype=complex,
        )
        result[:, :self.mode_count, :self.mode_count] = forward
        result[:, self.mode_count:, self.mode_count:] = forward
        return result

    def change_strucutre_length(self, new_length):
        # The transfer matrix is only meaningful on the transfer route; on the
        # scattering route it is never used and, for a lossy basis, it is the
        # thing that overflows.
        if self.resolve_method() == "transfer":
            self.tmatrix = self._find_Tmatrix_new_length(new_length)
        self.smatrix = self._find_Smatrix_new_length(new_length)

        lengths_per_matrix = np.zeros(2*self.section_count-2, dtype = float)

        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        initial_length = np.sum(delta_zs)
        length_ratio = new_length/initial_length
        for i in range(self.section_count - 1):
            lengths_per_matrix[2*i] = delta_zs[i]

        self._lengths_per_matrix = lengths_per_matrix*length_ratio


        print("Total Length is changed to ", str(new_length * 1e6), "um")
    
    #endregion main functions



    #region functions used in calc_Tmatrix
    def _calc_interface_Tmatrix(self):
        T12, T21, R12, R21 = self._calc_interface_blocks()

        inverse_T21 = mct._inverse_3D_matrix_ray(T21)
        m11 = T12 - R21 @ inverse_T21 @ R12
        m12 = R21 @ inverse_T21
        m21 = (-1) * inverse_T21 @ R12
        m22 = inverse_T21

        interface_Tmatrix = np.zeros(shape = (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count), dtype = complex)
        interface_Tmatrix[:,:self.mode_count, :self.mode_count] = m11
        interface_Tmatrix[:,:self.mode_count, self.mode_count:] = m12
        interface_Tmatrix[:,self.mode_count:, :self.mode_count] = m21
        interface_Tmatrix[:,self.mode_count:, self.mode_count:] = m22

        self._interface_Tmatrix = interface_Tmatrix
        self._is_interface_Tmatrix_calcualted = True

        return interface_Tmatrix
    
    def _calc_phase_propagation_Tmatrix(self):
        diagonal_mask = np.eye(self.mode_count, dtype = np.complex64)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.beta_forward[i, j]*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result
    
    def _calc_transmission_matrix(self, overlap_ab, overlap_ba, rcond=None):
        """Forward transmission block ``T12`` of every interface.

        ``overlap_ab[k, i, j] = 1/2 int (E_a,i x H_b,j)_z`` at interface ``k``;
        ``overlap_ba`` has the roles of the two sections swapped.

        Tangential-field continuity is imposed weakly, by projecting the two
        field equations onto a set of modes, and which set is a choice:

        * ``INTERFACE_PROJECTION = "input"`` - the modes of the section being
          left: ``T12 = 2 inv(O_ab + O_ba^T)``.  Upstream's form.
        * ``"output"`` - the modes of the section being entered:
          ``T12 = 2 O_ab^T inv(O_ab^T + O_ba) O_ba``.
        * ``"auto"`` (default) - ``"input"`` for a lossless basis, ``"output"``
          for a lossy one; see :attr:`INTERFACE_PROJECTION` for why.

        In a complete basis the two coincide.  In a truncated one they differ
        by the part of the field the basis cannot represent, and with opposite
        signs: for a single mode the input form gives ``1/|O|^2`` (a *gain* of
        exactly the unrepresented mismatch) and the output form ``|O|^2``
        (the unrepresented part is lost, as the radiation it stands for would
        be).  On the lossless Si taper with six modes the difference is
        2.5e-4 per interface either way.  On the plasmonic taper of report 12,
        where a 10 nm shift of a metal edge displaces a near field no mode set
        holds, the input form compounded to x1.44 over forty interfaces; the
        output form is bounded.  Reciprocity is identical for both.
        """
        overlap_tolerance = 0.5  # this value should be adjusted if it does not works.
        oba_t = np.transpose(overlap_ba, (0, 2, 1))
        projection = self.INTERFACE_PROJECTION
        if projection == "auto":
            projection = "input" if getattr(self, "_lossless", True) else "output"
        if projection == "input":
            return 2 * mct._inverse_3D_matrix_ray(
                overlap_ab + oba_t, tolerance=overlap_tolerance, rcond=rcond
            )
        if projection != "output":
            raise ValueError(
                f"INTERFACE_PROJECTION must be 'auto', 'input' or 'output', got "
                f"{self.INTERFACE_PROJECTION!r}"
            )
        oab_t = np.transpose(overlap_ab, (0, 2, 1))
        inner = mct._inverse_3D_matrix_ray(
            oab_t + overlap_ba, tolerance=overlap_tolerance, rcond=rcond
        )
        return 2 * (oab_t @ inner @ overlap_ba)

    def _calc_reflection_matrix(self, overlap_ab, overlap_ba, transmission_matrix):
        """Reflection block ``R12`` of every interface: the backward modes of the
        section being left, excited by a forward wave in it.  Called with the
        two overlap sets swapped (and ``T21``) it gives ``R21``.

        From tangential continuity, ``E^a (I + R) = E^b T`` and
        ``H^a (I - R) = H^b T``, projected on the modes of ``a``:
        ``I + R = O_ba^T T`` and ``I - R = O_ab T``, so

            ``R12 = 1/2 (O_ba^T - O_ab) T12``,

        ``a``-by-``a``, moving as ``D_a R12 D_a`` under a sign gauge.  It is used
        with either projection's ``T12``.  Projected on the modes of ``b``
        instead, the same equations give ``R12 = inv(O_ab^T + O_ba)
        (O_ba - O_ab^T)``; in a complete basis the two coincide
        (``tests/test_interface_reflection.py``), but on a truncated PML basis
        that form has no factor of ``T`` to damp the modes the basis cannot
        represent and returned reflection columns up to 3.1x (``R12``) and
        4.1x (``R21``) unit power uncapped (Kocabas 40-mode path), against
        0.03 here.

        Upstream had ``1/2 (O_ab^T - O_ba) T12``: a product of two
        ``b``-by-``a`` matrices, which is not a reflection block of either
        section, is not gauge-covariant, and for one mode is ``-r12``.
        """
        oba_t = np.transpose(overlap_ba, (0, 2, 1))
        return 0.5 * (oba_t - overlap_ab) @ transmission_matrix
    
    #endregion functions used in calc_Tmatrix

    #region functions for change length 
    def _find_Smatrix_new_length(self, new_length, method=None):
        """Section scattering matrices with the device rescaled in z.

        Every length sweep in ``examples/`` goes through here, so it needs the
        same choice of route as :meth:`calc_Smatrix` - otherwise switching the
        default would leave length sweeps on the unstable path.
        """
        method = self.resolve_method(method)

        if method == "direct":
            initial_length = np.sum(deepcopy(self.output_data["EME_delta_zs"]))
            interfaces = self._calc_interface_Smatrix()
            propagations = self._calc_propagation_Smatrix(
                length_ratio=new_length / initial_length
            )
            smatrix = np.zeros(
                (2 * self.section_count - 2, 2 * self.mode_count, 2 * self.mode_count),
                dtype=complex,
            )
            for i in range(self.section_count - 1):
                smatrix[2 * i] = propagations[i]
                smatrix[2 * i + 1] = interfaces[i]
        else:
            tmatrix = self._find_Tmatrix_new_length(new_length)
            smatrix = mct._convert_3Dmatrix(tmatrix)

        if self._force_unitary:
            smatrix = mct._find_nearest_unitary_3D(smatrix)
        if self._force_passive:
            smatrix = mct._make_passive_3D(smatrix)

        return smatrix

    def _find_Tmatrix_new_length(self, new_length):
        if not self._is_interface_Tmatrix_calcualted:
            self._calc_interface_Tmatrix()
        
        interfaces = deepcopy(self._interface_Tmatrix)
        phase_propagations = self._calc_phase_propagation_Tmatrix_new_length(new_length)

        total_matrices = np.zeros((2*self.section_count - 2, 2*self.mode_count, 2*self.mode_count), dtype = np.complex64)
        for i in range(self.section_count - 1):
            total_matrices[2*i] = phase_propagations[i]
            total_matrices[2*i + 1] = interfaces[i]
        
        return total_matrices

    def _calc_phase_propagation_Tmatrix_new_length(self, new_length):
        initial_length = np.sum(deepcopy(self.output_data["EME_delta_zs"]))
        length_ratio = new_length/initial_length

        diagonal_mask = np.eye(self.mode_count, dtype = complex)
        i, j, _ = np.meshgrid(np.arange(0, self.section_count-1),\
                              np.arange(0, self.mode_count),\
                              np.arange(0, self.mode_count),\
                              indexing = 'ij')

        forward_matrix = np.exp(1j*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask
        backward_matrix = np.exp((-1j)*self.output_data["beta"][i, j]*length_ratio*self.output_data["EME_delta_zs"][i]) * diagonal_mask

        result = np.zeros(shape = (self.section_count-1, 2*self.mode_count, 2*self.mode_count), dtype = complex)
        result[:,:self.mode_count, :self.mode_count] = forward_matrix
        result[:,self.mode_count:, self.mode_count:] = backward_matrix

        return result
    
    #endregion functions for change length