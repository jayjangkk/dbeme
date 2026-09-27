r"""Backend-agnostic interface for the eigenmode (FDE) solver.

The dataset-based EME method needs exactly one thing from a mode solver: for a
point in waveguide-parameter space, the first ``num_modes`` guided modes
sampled **on a grid that is identical for every parameter point**.  The shared
grid is what makes the overlap integrals

.. math::  \langle E_a, H_b \rangle = \tfrac12 \int (E_a \times H_b)_z \, dA

between two different parameter points well defined.

``DataUpdater`` talks to a backend only through :class:`FDEBackend`, so the
Lumerical MODE solver used by the original implementation can be swapped for
any other solver (here: emepy) without touching the EME machinery.
"""

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class ModeData:
    """Modes of one parameter point, sampled on the backend's common grid.

    :param x: Transverse coordinate 1 (metres), shape ``(nx,)``.
    :param y: Transverse coordinate 2 (metres), shape ``(ny,)``.
    :param E: Electric field, shape ``(num_modes, 3, nx, ny)``.  Component axis
        is ordered ``(x, y, z)`` with ``z`` the propagation direction.
    :param H: Magnetic field, same shape and ordering as ``E``.
    :param neff: Complex effective indices, shape ``(num_modes,)``.
    :param TE_pol: TE polarisation fraction in ``[0, 1]``, shape ``(num_modes,)``.
    """

    x: np.ndarray
    y: np.ndarray
    E: np.ndarray
    H: np.ndarray
    neff: np.ndarray
    TE_pol: np.ndarray

    def __post_init__(self):
        n_modes = len(self.neff)
        expected = (n_modes, 3, len(self.x), len(self.y))
        if self.E.shape != expected or self.H.shape != expected:
            raise ValueError(
                f"field shape {self.E.shape}/{self.H.shape} does not match "
                f"expected {expected}"
            )


class FDEBackend(metaclass=abc.ABCMeta):
    """Mode solver plugged into :class:`~em_simulation.data_updater.data_updater.DataUpdater`."""

    #: Names of the geometry parameters this backend accepts, in the order used
    #: by the dataset's parameter tuples.  Set by the concrete backend.
    parameter_names = ()

    @property
    def lossless(self) -> bool:
        """Whether this backend's modes are lossless *by construction*.

        Three things downstream are only valid for a real index, and all three
        are switched on this flag rather than on ``Im(n_eff)`` at runtime -
        because a lossy mode whose loss has already been clipped to zero looks
        exactly like a lossless one:

        * the backward basis is built by conjugation, which is time reversal
          and turns loss into **gain** for a complex mode (5.16a);
        * ``correct_gain_modified`` clips ``Im(n_eff) < 0``, a sensible
          round-off guard in a lossless model and a silent deletion of all the
          loss in a lossy one (5.16b);
        * ``force_unitary`` projects each section onto the nearest unitary
          matrix, which is only defensible while nothing absorbs.

        A backend with PML or an absorbing material must report ``False``.
        """
        return True

    @property
    @abc.abstractmethod
    def num_modes(self) -> int:
        """Number of forward modes solved per parameter point."""

    @property
    @abc.abstractmethod
    def wavelength(self) -> float:
        """Free-space wavelength in metres."""

    @abc.abstractmethod
    def grid(self):
        """Return the common ``(x, y)`` sampling grid in metres."""

    @abc.abstractmethod
    def solve(self, parameter_point) -> ModeData:
        """Solve one parameter point.

        :param parameter_point: Tuple of parameter values ordered as
            :attr:`parameter_names`.
        :returns: The modes of that cross section.
        :rtype: ModeData
        """
