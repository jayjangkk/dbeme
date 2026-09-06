import numpy as np
from copy import deepcopy

from ... import matrix_calculation_tool as mct
from ...geometry.geometry import Geometry
from ..propagator import Propagator


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
    INTERFACE_RCOND = 1e-3

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
            n12 = -m12 inv(m22) = -R21 inv(T21) T21              = -R21
            n21 = -inv(m22) m21 = -T21 (-inv(T21) R12)           = +R12
            n11 = m11 - m12 inv(m22) m21
                = (T12 - R21 inv(T21) R12) + R21 inv(T21) R12    = T12

        so the scattering matrix is just ``[[T12, -R21], [R12, T21]]`` and the
        whole ``inv(T21)`` round trip cancels.  That inversion is the only
        reason the transfer route needs ``T21`` to be invertible, and it is
        what fails once a near-null-norm PML mode enters the basis: the
        interface matrix reaches a condition number of 5.6e4 and the cascade
        reports output energy of 3070 against a physical maximum of 1.
        """
        rcond = self.INTERFACE_RCOND
        T12 = self._calc_transmission_matrix(self.overlap_forward_ab, self.overlap_forward_ba, rcond)
        T21 = self._calc_transmission_matrix(self.overlap_forward_ba, self.overlap_forward_ab, rcond)
        R12 = self._calc_reflection_matrix(self.overlap_forward_ab, self.overlap_forward_ba, T12)
        R21 = self._calc_reflection_matrix(self.overlap_forward_ba, self.overlap_forward_ab, T21)

        interface_Smatrix = np.zeros(
            (self.section_count - 1, 2 * self.mode_count, 2 * self.mode_count),
            dtype=complex,
        )
        interface_Smatrix[:, :self.mode_count, :self.mode_count] = T12
        interface_Smatrix[:, :self.mode_count, self.mode_count:] = -R21
        interface_Smatrix[:, self.mode_count:, :self.mode_count] = R12
        interface_Smatrix[:, self.mode_count:, self.mode_count:] = T21
        return interface_Smatrix

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
        T12 = self._calc_transmission_matrix(self.overlap_forward_ab, self.overlap_forward_ba)
        T21 = self._calc_transmission_matrix(self.overlap_forward_ba, self.overlap_forward_ab)
        R12 = self._calc_reflection_matrix(self.overlap_forward_ab, self.overlap_forward_ba, T12)
        R21 = self._calc_reflection_matrix(self.overlap_forward_ba, self.overlap_forward_ab, T21)

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
        result = 0.5 * (np.transpose(overlap_ab, (0,2,1)) - overlap_ba) @ transmission_matrix
        return result
    
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