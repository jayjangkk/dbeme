"""The ring as a sax circuit from the Phase 2 results - `tasks/11` §3.

Reads ``reports/output/ring/coupler_<lambda>.json`` (one per wavelength,
`build_coupler.py`) and ``ring_loss.json`` (`ring_loss.py`) and builds:

* the 4-port coupler sax model per bus-ring gap ``g0``, interpolated across
  the solved wavelengths by `dbeme.circuit.sax_model` (magnitude
  PCHIP, phase through the optical length, unwrapped against the straight
  guide's propagation phase over the coupler chord);
* the ring arc: ``2 pi R`` minus the chord the coupler region already
  covers, with the bent guide's ``n_eff(lambda, R)`` and the loss budget
  (PML radiation + Payne-Lacey roughness x bend factor) as dB/cm;
* the all-pass ring netlist, its spectrum, and the figures of merit both
  from the spectrum and from the closed form (`circuit.ring.ring_metrics`).

Everything a sax model receives is in microns; everything read from the
JSON is SI (`dbeme.circuit` docstring).
"""

import json
import os

import numpy as np
import jax.numpy as jnp
import sax

from dbeme.circuit import (
    IndexModel,
    PchipJax,
    all_pass_amplitude,
    arc_model,
    ring_metrics,
    round_trip_phase,
    sax_model_from_samples,
    straight_model,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "output", "ring")
PORTS = ("o1", "o2", "o3", "o4")  # bus in, bus through, ring in, ring out
C_LIGHT = 299792458.0


def load_coupler_json(paths):
    """``{g0_nm: [(wavelength_m, S4 2x2, z_max_m), ...]}`` sorted by wavelength."""
    table = {}
    meta = {}
    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        wl = data["wavelength_nm"] * 1e-9
        meta.setdefault("radius_um", data["radius_um"])
        for entry in data["gaps"]:
            S4 = np.array([[complex(*entry["S4"]["bus_bus"]), complex(*entry["S4"]["ring_bus"])],
                           [complex(*entry["S4"]["bus_ring"]), complex(*entry["S4"]["ring_ring"])]])
            table.setdefault(round(entry["g0_nm"]), []).append((wl, S4, entry["z_max_um"] * 1e-6, entry))
    for g0 in table:
        table[g0].sort(key=lambda item: item[0])
    return table, meta


def load_loss_json(path=os.path.join(OUT, "ring_loss.json")):
    with open(path) as handle:
        return json.load(handle)


def straight_index(loss):
    wl = np.asarray(loss["lambdas_nm"]) * 1e-9
    return IndexModel.from_samples(wl, loss["index"]["straight"]["neff"], degree=2)


def bend_index(loss, radius_um):
    wl = np.asarray(loss["lambdas_nm"]) * 1e-9
    key = f"{radius_um:g}"
    if key not in loss["index"]:
        raise KeyError(f"no bent index for R = {radius_um} um; have {list(loss['index'])}")
    return IndexModel.from_samples(wl, loss["index"][key]["neff"], degree=2)


def arc_loss_db_per_cm(loss, radius_um, wavelength_m=1.55e-6, roughness=True, radiation=True):
    """Loss budget at one wavelength: radiation (PML, interpolated in lambda,
    zero where the table has no entry because it is < 1e-9 dB/cm) plus the
    straight-guide roughness estimate times the bend's sidewall factor."""
    total = 0.0
    parts = {}
    key = f"{radius_um:g}"
    if radiation and key in loss["radiation"]:
        wl = np.asarray(loss["lambdas_nm"]) * 1e-9
        parts["radiation"] = float(np.interp(wavelength_m, wl, loss["radiation"][key]["loss_db_per_cm"]))
        total += parts["radiation"]
    else:
        parts["radiation"] = 0.0
    if roughness:
        base = loss["roughness"]["straight_db_per_cm_eim_index"]
        factor = loss["roughness"]["bend_enhancement"].get(key, 1.0)
        parts["roughness"] = base * factor
        total += parts["roughness"]
    parts["total"] = total
    return total, parts


def coupler_sax_model(records, straight, degree=2):
    """4-port sax model of one gap from its per-wavelength ``S4`` samples.

    ``records``: ``[(wavelength_m, S4, z_max_m, entry), ...]``.  The
    unwrapping reference is the straight guide's propagation phase over the
    coupler chord ``2 z_max``: its group index is right to 1e-3, far inside
    the ``lambda^2 / (2 L d_lambda)`` window (`phase_fit.unwrap_against`).
    """
    wls = np.array([r[0] for r in records])
    length = 2.0 * records[0][2]
    S = np.zeros((len(records), 4, 4), dtype=complex)
    for m, (_, S4, _, _) in enumerate(records):
        # Bus-side entries are what DBEME computes well (the bus never moves:
        # power conserved to 1e-5, converged in N).  The ring-side through
        # amplitude is *not* taken from the cascade: the moving ring loses its
        # translation mismatch at every interface (tasks/11 section 2.1), so
        # for identical guides it is imposed by symmetry, S(o3,o4) = S(o1,o2),
        # and the cross coupling is corrected for the half-path ring survival.
        # The cross term's *phase* also passed through the ring-side staircase
        # and came out 8-13 deg away from the quadrature a symmetric lossless
        # coupler requires (the sax ring then showed T = 1.66 at g0 = 150 nm).
        # Bus-side power conservation gives |kappa|^2 = 1 - |t|^2 to 1e-4 of
        # the survival-corrected value, so the coupler is built from the
        # converged bus through amplitude alone: exactly unitary.
        t = S4[0, 0]
        k = 1j * np.sqrt(max(1.0 - abs(t) ** 2, 0.0)) * t / abs(t)
        S[m, 0, 1] = S[m, 1, 0] = t          # bus in -> bus through (DBEME)
        S[m, 0, 3] = S[m, 3, 0] = k          # bus in -> ring out (quadrature, |k|^2 = 1 - |t|^2)
        S[m, 2, 1] = S[m, 1, 2] = k          # ring in -> bus through (reciprocity)
        S[m, 2, 3] = S[m, 3, 2] = t          # ring in -> ring out (symmetry of identical guides)

    def reference(wl_m):
        return np.full((4, 4), 2 * np.pi * float(np.asarray(straight.neff(wl_m))) * length / float(wl_m))

    model = sax_model_from_samples(wls, S, PORTS, phase_reference=reference if len(records) > 1 else None,
                                   degree=degree)
    model.info["coupler_chord_m"] = length
    return model


def ring_netlist():
    return {
        "instances": {"cp": {"component": "coupler"}, "arc": {"component": "arc"},
                      "chord": {"component": "chord"}},
        "connections": {"cp,o4": "arc,o1", "arc,o2": "chord,o1", "chord,o2": "cp,o3"},
        "ports": {"in": "cp,o1", "out": "cp,o2"},
    }


class RingModel:
    """All-pass ring from one gap's coupler records and the loss table."""

    def __init__(self, records, loss, radius_um, roughness=True, radiation=True):
        self.records = records
        self.loss = loss
        self.radius_um = float(radius_um)
        self.R = self.radius_um * 1e-6
        self.straight = straight_index(loss)
        self.bend = bend_index(loss, radius_um)
        self.coupler = coupler_sax_model(records, self.straight)
        z_max = records[0][2]
        self.chord = 2.0 * z_max
        self.arc_inside = 2.0 * self.R * np.arcsin(min(z_max / self.R, 1.0))
        self.arc_length = 2.0 * np.pi * self.R - self.arc_inside
        self.loss_db_per_cm, self.loss_parts = arc_loss_db_per_cm(loss, radius_um, 1.55e-6, roughness, radiation)
        # the coupler models the ring's passage as the straight chord; the
        # arc it actually is exceeds the chord by (arc_inside - chord), with
        # the bend index - a 0.2 um correction at R = 10 um
        self.arc = arc_model(self.bend, self.R, angle=self.arc_length / self.R, loss_db_per_cm=self.loss_db_per_cm)
        self.chord_fix = straight_model(self.bend, self.arc_inside - self.chord, loss_db_per_cm=self.loss_db_per_cm)
        self.circuit, _ = sax.circuit(ring_netlist(), models={"coupler": self.coupler, "arc": self.arc,
                                                                 "chord": self.chord_fix})

    # ------------------------------------------------------------ spectrum
    def transmission(self, wl_um):
        return np.asarray(self.circuit(wl=jnp.asarray(wl_um))[("in", "out")])

    def coupler_at(self, wl_um=1.55):
        s = self.coupler(wl=wl_um)
        return {"t": complex(s[("o1", "o2")]), "kappa": complex(s[("o1", "o4")]),
                "t_ring": complex(s[("o3", "o4")])}

    def closed_form(self, wl_m=1.55e-6):
        """Metrics from |t|, the loss and n_g via `ring_metrics` (full circumference)."""
        c = self.coupler_at(wl_m * 1e6)
        L = 2.0 * np.pi * self.R
        a = 10 ** (-self.loss_db_per_cm * L * 100 / 20)
        ng = float(self.bend.ng(wl_m))
        r = abs(c["t"])
        m = ring_metrics(r, a, ng, L, wl_m)
        m.update({"r": r, "a": a, "n_g": ng, "kappa2": abs(c["kappa"]) ** 2})
        return m

    def spectrum_metrics(self, wl_um):
        """FSR, FWHM, Q and extinction measured on the computed spectrum."""
        T = np.abs(self.transmission(wl_um)) ** 2
        idx = np.flatnonzero((T[1:-1] < T[:-2]) & (T[1:-1] < T[2:])) + 1
        idx = [i for i in idx if T[i] < 0.9 * np.max(T)]
        dips = wl_um[idx]
        out = {"dips_um": dips.tolist(), "T_min": [float(T[i]) for i in idx], "T_max": float(np.max(T))}
        if len(dips) >= 2:
            out["FSR_nm"] = float(np.mean(np.diff(dips)) * 1e3)
        if idx:
            k = idx[np.argmin(np.abs(dips - 1.55))]
            half = 0.5 * (T[k] + np.max(T))
            lo, hi = k, k
            while lo > 0 and T[lo] < half:
                lo -= 1
            while hi < len(T) - 1 and T[hi] < half:
                hi += 1
            fwhm = wl_um[hi] - wl_um[lo]
            out.update({"lambda_res_um": float(wl_um[k]), "FWHM_pm": float(fwhm * 1e6),
                        "Q_loaded": float(wl_um[k] / fwhm) if fwhm > 0 else np.inf,
                        "extinction_dB": float(10 * np.log10(np.max(T) / T[k])) if T[k] > 0 else np.inf})
        return out


def gap_interpolated_coupler(table, wl_m=1.55e-6, straight=None):
    """A sax coupler model with a continuous ``gap_nm`` parameter, at one
    wavelength: ``|S|`` and phase PCHIP-interpolated across the solved gaps,
    for the Phase 5 gradient check.  Returns ``model(wl=..., gap_nm=...)``."""
    gaps = np.array(sorted(table))
    entries = {}
    for (i, j) in ((0, 1), (0, 3), (2, 1), (2, 3)):
        mags, phases = [], []
        for g in gaps:
            recs = table[g]
            rec = min(recs, key=lambda r: abs(r[0] - wl_m))
            S4 = rec[1]
            val = {(0, 1): S4[0, 0], (0, 3): S4[1, 0], (2, 1): S4[0, 1], (2, 3): S4[1, 1]}[(i, j)]
            mags.append(abs(val))
            phases.append(np.angle(val))
        phases = np.unwrap(np.array(phases))
        entries[(PORTS[i], PORTS[j])] = (PchipJax(gaps, mags), PchipJax(gaps, phases))

    def model(*, wl: float = 1.55, gap_nm: float = 200.0) -> sax.SDict:
        g = jnp.asarray(gap_nm, dtype=jnp.float64)
        out = {}
        for key, (mag, ph) in entries.items():
            out[key] = mag(g) * jnp.exp(1j * ph(g))
        return sax.reciprocal(out)

    model.info = {"gaps_nm": gaps.tolist(), "wavelength_m": wl_m}
    return model
