r"""Analytic plasmonic references: the ruler for the metal work.

Two closed-form (or one-line transcendental) problems that a lossy mode solver
must reproduce before any plasmonic device is trusted:

* the **single-interface surface plasmon polariton** between a metal of
  permittivity :math:`\varepsilon_m` and a dielectric :math:`\varepsilon_d`,

  .. math::  n_{spp} = \sqrt{\frac{\varepsilon_m \varepsilon_d}
                                  {\varepsilon_m + \varepsilon_d}}

* the **symmetric metal-insulator-metal slab** (gap plasmon), whose
  fundamental TM mode - :math:`H_y` even across the gap, so :math:`E_x` even,
  no cutoff for any gap - satisfies

  .. math::  \varepsilon_m k_d \tanh(k_d g/2) + \varepsilon_d k_m = 0,
             \qquad k_i = \sqrt{\beta^2 - k_0^2 \varepsilon_i}

  and tends to :math:`n_{spp}` as the gap opens.  The other branch,
  :math:`\coth`, is the antisymmetric mode and has a cutoff.

Nothing here imports the solver.  Conventions follow the rest of the project:
fields go as :math:`e^{i\beta z}`, so a lossy mode has :math:`\mathrm{Im}(n_{eff}) > 0`
and an absorbing metal has :math:`\mathrm{Im}(\varepsilon_m) > 0` (Johnson &
Christy gold at 1550 nm: :math:`\varepsilon \approx -115 + 11i`).

Reference: S. A. Maier, *Plasmonics: Fundamentals and Applications* (Springer,
2007), sections 2.2-2.3.
"""

import numpy as np

__all__ = [
    "spp_index",
    "MIMSlab",
    "propagation_length",
    "attenuation_db_per_um",
]


def _lossy_branch(value):
    """Pick the square-root branch that propagates forward and decays."""
    value = complex(value)
    if value.real < 0:
        value = -value
    if value.imag < 0 and abs(value.imag) > 1e-14 * abs(value):
        value = value.conjugate()
    return value


def spp_index(eps_metal, eps_dielectric):
    """Effective index of the single-interface SPP."""
    eps_m, eps_d = complex(eps_metal), complex(eps_dielectric)
    return _lossy_branch(np.sqrt(eps_m * eps_d / (eps_m + eps_d)))


def propagation_length(neff, wavelength):
    """Distance over which the *power* falls to 1/e, metres."""
    k0 = 2 * np.pi / wavelength
    return 1.0 / (2 * k0 * float(np.imag(neff)))


def attenuation_db_per_um(neff, wavelength):
    k0 = 2 * np.pi / wavelength
    return 4.342944819 * 2 * k0 * float(np.imag(neff)) * 1e-6


class MIMSlab:
    """Symmetric metal / dielectric / metal slab, TM modes.

    :param wavelength: Metres.
    :param eps_metal: Complex permittivity of the metal, ``Im > 0`` absorbing.
    :param eps_dielectric: Permittivity of the gap material (1.0 for air).
    """

    def __init__(self, wavelength, eps_metal, eps_dielectric=1.0):
        self.wavelength = float(wavelength)
        self.eps_metal = complex(eps_metal)
        self.eps_dielectric = complex(eps_dielectric)

    @property
    def k0(self):
        return 2 * np.pi / self.wavelength

    def _decay(self, beta, eps):
        """Transverse decay constant with ``Re >= 0`` (fields decay away)."""
        k = np.sqrt(beta**2 - self.k0**2 * eps)
        return -k if k.real < 0 else k

    def dispersion(self, beta, gap, branch="fundamental"):
        """Zero at a mode.  ``"fundamental"`` is the even-``E_x`` gap plasmon."""
        kd = self._decay(beta, self.eps_dielectric)
        km = self._decay(beta, self.eps_metal)
        half = kd * gap / 2
        if branch == "fundamental":
            return self.eps_metal * kd * np.tanh(half) + self.eps_dielectric * km
        if branch == "antisymmetric":
            return self.eps_metal * kd / np.tanh(half) + self.eps_dielectric * km
        raise ValueError("branch must be 'fundamental' or 'antisymmetric'")

    def solve(self, gap, neff_guess=None, branch="fundamental", tol=1e-13, max_iter=200):
        """Complex ``n_eff`` of the gap plasmon at one gap width.

        Seeded from the single-interface SPP, which is the wide-gap limit; a
        sweep should pass the previous answer as ``neff_guess``.
        """
        if neff_guess is None:
            neff_guess = spp_index(self.eps_metal, self.eps_dielectric) * (1 + 1e-3)
        b0 = self.k0 * complex(neff_guess)
        b1 = b0 * (1 + 1e-6) + 1e-6 * self.k0
        f0 = self.dispersion(b0, gap, branch)
        f1 = self.dispersion(b1, gap, branch)
        for _ in range(max_iter):
            if abs(f1 - f0) < 1e-300:
                break
            step = f1 * (b1 - b0) / (f1 - f0)
            b0, f0 = b1, f1
            b1 = b1 - step
            f1 = self.dispersion(b1, gap, branch)
            if abs(step) < tol * abs(b1):
                break
        else:
            raise RuntimeError(f"no convergence at gap {gap:.3e} m")
        return _lossy_branch(b1 / self.k0)

    def sweep(self, gaps, branch="fundamental"):
        """Wide gap first, then continuation inward."""
        gaps = np.asarray(gaps, dtype=float)
        order = np.argsort(-gaps)
        out = np.empty(len(gaps), dtype=complex)
        guess = None
        for i in order:
            out[i] = self.solve(gaps[i], neff_guess=guess, branch=branch)
            guess = out[i]
        return out
