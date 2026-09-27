"""The analytical micro-ring resonator - the reference every ring number meets first.

All-pass and add-drop transfer functions in the form of Bogaerts et al.,
Laser Photon. Rev. 6, 47 (2012) §2 and Yariv, Electron. Lett. 36, 321 (2000).
No sax, no JAX: numpy only, so it can never share a bug with the circuit
composition it is there to check (`tasks/11` §1.3, §4.1).

Symbols: ``r`` the coupler's self-coupling amplitude, ``kappa`` its cross
coupling (``r**2 + kappa**2 = 1`` for a lossless coupler), ``a`` the
single-pass amplitude (``a**2 = exp(-alpha L)``), ``phi = beta L`` the
round-trip phase, ``L = 2 pi R``.

Amplitude forms are derived in the project convention ``exp(+i beta z)``
with the lossless symmetric coupler ``[[r, i kappa], [i kappa, r]]``:

    ring output   E_r_out = i kappa E_in + r E_r_in
    round trip    E_r_in  = a e^{i phi} E_r_out
    =>  E_pass / E_in = (r - a e^{i phi}) / (1 - r a e^{i phi})

and for the add-drop with the second coupler half way round,

    E_pass / E_in = (r1 - r2 a e^{i phi}) / (1 - r1 r2 a e^{i phi})
    E_drop / E_in = -kappa1 kappa2 sqrt(a) e^{i phi / 2} / (1 - r1 r2 a e^{i phi})

whose moduli squared are the intensity formulas below.  The intensity
functions are written out separately, not as ``abs(amplitude)**2``, so the
tests can hold the two against each other.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def single_pass_amplitude(loss_db_per_cm, length_m):
    """``a`` for a power loss in dB/cm over ``length_m``."""
    return 10.0 ** (-np.asarray(loss_db_per_cm, dtype=float) * length_m * 100.0 / 20.0)


def round_trip_phase(neff, length_m, wavelength_m):
    """``phi = 2 pi n_eff L / lambda``."""
    return TWO_PI * np.asarray(neff, dtype=float) * length_m / np.asarray(wavelength_m, dtype=float)


def all_pass_amplitude(r, a, phi):
    e = a * np.exp(1j * np.asarray(phi, dtype=float))
    return (r - e) / (1.0 - r * e)


def all_pass(r, a, phi):
    """Through-port intensity of the all-pass ring."""
    c = np.cos(np.asarray(phi, dtype=float))
    return (a * a - 2.0 * r * a * c + r * r) / (1.0 - 2.0 * r * a * c + (r * a) ** 2)


def add_drop_amplitude(r1, r2, a, phi):
    """``(E_pass, E_drop) / E_in`` for the add-drop ring, add port dark."""
    phi = np.asarray(phi, dtype=float)
    e = a * np.exp(1j * phi)
    den = 1.0 - r1 * r2 * e
    k1 = np.sqrt(1.0 - r1 * r1)
    k2 = np.sqrt(1.0 - r2 * r2)
    t_pass = (r1 - r2 * e) / den
    t_drop = -k1 * k2 * np.sqrt(a) * np.exp(0.5j * phi) / den
    return t_pass, t_drop


def add_drop(r1, r2, a, phi):
    """``(T_pass, T_drop)`` intensities of the add-drop ring."""
    c = np.cos(np.asarray(phi, dtype=float))
    den = 1.0 - 2.0 * r1 * r2 * a * c + (r1 * r2 * a) ** 2
    t_pass = ((r2 * a) ** 2 - 2.0 * r1 * r2 * a * c + r1 * r1) / den
    t_drop = (1.0 - r1 * r1) * (1.0 - r2 * r2) * a / den
    return t_pass, t_drop


def ring_metrics(r, a, ng, length_m, wavelength_m, r2=None):
    """Closed-form figures of merit at ``wavelength_m``.

    ``r2=None`` is the all-pass ring; otherwise ``r`` is the bus coupler and
    ``r2`` the drop coupler.  Returns a dict with FSR and FWHM in metres,
    the loaded, intrinsic and coupling Q, finesse, the on-resonance
    through-port transmission and extinction ratio, the drop-port peak for
    the add-drop, and the critical-coupling ``r``.
    """
    lam = float(wavelength_m)
    L = float(length_m)
    rr = r if r2 is None else r * r2
    ra = rr * a
    fsr = lam * lam / (ng * L)
    fwhm = (1.0 - ra) * lam * lam / (np.pi * ng * L * np.sqrt(ra))
    out = {
        "FSR_m": fsr,
        "FWHM_m": fwhm,
        "Q_loaded": lam / fwhm,
        "finesse": np.pi * np.sqrt(ra) / (1.0 - ra),
        "Q_intrinsic": np.pi * ng * L * np.sqrt(a) / (lam * (1.0 - a)) if a < 1.0 else np.inf,
        "Q_coupling": np.pi * ng * L * np.sqrt(rr) / (lam * (1.0 - rr)) if rr < 1.0 else np.inf,
    }
    if r2 is None:
        t_res = all_pass(r, a, 0.0)
        t_off = all_pass(r, a, np.pi)
        out["r_critical"] = a
    else:
        t_res, d_res = add_drop(r, r2, a, 0.0)
        t_off, _ = add_drop(r, r2, a, np.pi)
        out["T_drop_peak"] = float(d_res)
        out["r_critical"] = r2 * a
    out["T_pass_resonance"] = float(t_res)
    out["T_pass_off"] = float(t_off)
    out["extinction_dB"] = float(10.0 * np.log10(t_off / t_res)) if t_res > 0 else np.inf
    return out
