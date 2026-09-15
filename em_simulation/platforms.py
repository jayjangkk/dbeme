"""Reusable platform definitions, so a dataset file is thin.

A DBEME dataset is keyed to ``{stack, lambda, mesh + window, mode count,
cross-section topology, swept parameters}``.  Everything in that key except the
wavelength is a property of the *platform*, so building the same platform at
five wavelengths should not mean five copies of a 150-line ``dataset_info.py``.

Each factory here returns a ``DatasetInfo`` **class**, which a dataset file
instantiates:

.. code-block:: python

    from em_simulation.platforms import sacher_coupler_dataset_info

    WAVELENGTH = 1.31e-6
    DatasetInfo = sacher_coupler_dataset_info(WAVELENGTH)

The file stays the source of truth for that dataset - it still records exactly
one wavelength and one grid - and the physics is written down once.

Both platforms are the ones in Sacher et al., *Opt. Express* **22**, 3777
(2014): a 220 nm Si device layer with symmetric SiO2 cladding, fully etched for
the adiabatic coupler and partially etched to a 90 nm slab for the bi-level
taper.
"""

import numpy as np

from .fde import BiLevelStrip, CoupledStrips, EmepyFDE
from .fde.materials import air, as_material, silica, silicon

THICKNESS = 220e-9
SLAB_THICKNESS = 90e-9
COUPLER_GAP = 200e-9


def nm(start, stop, step):
    """Grid values in metres from a range given in nanometres."""
    return np.arange(start, stop, step) * 1e-9


def axis(*segments):
    """Join grid segments into one sorted, de-duplicated axis."""
    return np.round(np.unique(np.concatenate(segments)), 12)


def _materials(wavelength):
    """Si and SiO2 from refractiveindex.info, refusing to extrapolate.

    ``out_of_range="raise"`` matters here: these datasets are built across the
    O band as well as the C band, and silently extrapolating a dispersion
    formula past its data is how a wavelength sweep goes quietly wrong.
    """
    return silicon(out_of_range="raise"), silica(out_of_range="raise")


def _make_dataset_info(
    *,
    name,
    description,
    wavelength,
    parameters,
    parameter_names,
    cross_section_factory,
    window,
    mode_numbers,
    mesh,
    cladding=None,
    backend_factory=None,
):
    """Assemble a ``DatasetInfo`` class from a platform description.

    :param cladding: Override the default SiO2 cladding - pass
        ``materials.air()`` for an air-clad (undercut or unclad) device.  The
        cladding index feeds the guided-mode cutoff as well as the geometry, so
        it is part of the dataset identity.
    :param backend_factory: ``f(cross_section, parameter_names, wavelength,
        window, mode_numbers, mesh) -> FDEBackend``.  Defaults to the lossless
        ``EmepyFDE``; a plasmonic platform passes one that builds a
        ``PMLBackend``.  ``get_fde_backend()`` is per dataset, so lossless
        datasets keep the fast uniform-grid path (CLAUDE.md 5.9).
    """
    core_material, clad_material = _materials(wavelength)
    if cladding is not None:
        clad_material = as_material(cladding)

    class DatasetInfo:
        __doc__ = description

        def __init__(self):
            self.description = {
                "Description": description,
                "platform": name,
                "material": f"{core_material.name} / {clad_material.name}",
                "wavelength": f"{wavelength * 1e9:.0f} nm",
                "solver": "emepy MSEMpy (EMpy vectorial finite difference)",
                "mesh": f"{mesh} points across the window",
            }
            self.file_structure = {
                "dataset_info.py": "this file - parameter grid and cross section",
                "overlap.pkl": "point -> {adjacent point -> (2N,2N) overlap}",
                "neff.pkl": "point -> (2N,) complex effective indices",
                "TE_pol.pkl": "point -> (2N,) TE polarization fraction",
            }
            self.FDE_crosssection = {"crosssection_x": "x", "crosssection_y": "y"}

            self.mode_numbers = int(mode_numbers)
            self.wavelength = float(wavelength)
            # Ask the cross section, do not assume the cladding.  This value is
            # the *radiation cutoff* - the index above which a mode is bound -
            # and for a vertically asymmetric stack that is the larger of the
            # cladding and the substrate.  An air-clad guide on a buried oxide
            # box cuts off at 1.444, not at 1.0, and taking the cladding alone
            # counts three substrate-radiating box modes as guided.
            self.cladding_index = float(
                cross_section_factory(
                    core_material, clad_material, wavelength,
                    tuple(parameter_names),
                ).cladding_index_at(wavelength)
            )
            self.parameter_names = list(parameter_names)
            self.parameter_types = {p: "Length" for p in parameter_names}
            self.parameters = dict(parameters)
            self._window = tuple(window)
            self._backend = None

        def get_cross_section(self):
            return cross_section_factory(
                core_material, clad_material, wavelength, tuple(parameter_names)
            )

        def get_fde_backend(self):
            if self._backend is None:
                if backend_factory is not None:
                    self._backend = backend_factory(
                        self.get_cross_section(), self.parameter_names,
                        self.wavelength, self._window, self.mode_numbers, int(mesh),
                    )
                else:
                    self._backend = EmepyFDE(
                        cross_section=self.get_cross_section(),
                        parameter_names=self.parameter_names,
                        num_modes=self.mode_numbers,
                        wavelength=self.wavelength,
                        window=self._window,
                        mesh=int(mesh),
                    )
            return self._backend

        def get_description(self):
            return self.description

        def get_file_structure(self):
            return self.file_structure

        def get_parameter_names(self):
            return self.parameter_names

        def get_parameter_grid(self):
            return self.parameters

        def get_parameter_types(self):
            return self.parameter_types

        def get_mode_numbers(self):
            return self.mode_numbers

        def get_crosssection_x(self):
            return self.FDE_crosssection["crosssection_x"]

        def get_crosssection_y(self):
            return self.FDE_crosssection["crosssection_y"]

        def get_wavelength(self):
            return self.wavelength

        def get_cladding_index(self):
            return self.cladding_index

        def _is_variable_FDE(self):
            return False

    DatasetInfo.__name__ = "DatasetInfo"
    return DatasetInfo


# ------------------------------------------------------- adiabatic coupler


def sacher_coupler_dataset_info(wavelength=1.55e-6, mode_numbers=6, mesh=220):
    """Two fully etched strips, 200 nm gap - the TE0/TE1 splitter stage.

    Grid axes ``w1`` (broad, 850 -> 650 nm) and ``w2`` (narrow, 200 -> 500 nm),
    both non-uniform: 20 nm generally, **5 nm across the anti-crossing**
    (``w1`` 700-800, ``w2`` 280-440 nm).  That refinement is not optional - a
    20 nm step at the crossing scatters 29 % of the power between the two
    branches at a single interface, and a dataset built without it reports a
    *longer* adiabatic device as *worse*.  See
    ``reports/01_adiabatic_coupler.md`` §2.
    """

    def cross_section(core, cladding, wl, names):
        return CoupledStrips(
            thickness=THICKNESS,
            gap=COUPLER_GAP,
            core=core,
            cladding=cladding,
            swept_parameters=names,
            reference_wavelength=wl,
        )

    parameters = {
        "w1": axis(nm(610, 700, 20), nm(700, 805, 5), nm(810, 915, 20)),
        "w2": axis(nm(180, 280, 20), nm(280, 445, 5), nm(460, 545, 20)),
    }
    half_width = 0.5 * (0.91e-6 + COUPLER_GAP + 0.54e-6) + 1.3e-6
    return _make_dataset_info(
        name="Sacher adiabatic coupler: 2 x fully etched 220 nm Si strips, 200 nm gap",
        description="overlap, neff and TE_pol for a coupled pair of Si strips",
        wavelength=wavelength,
        parameters=parameters,
        parameter_names=("w1", "w2"),
        cross_section_factory=cross_section,
        window=(half_width, -0.8e-6, 0.8e-6),
        mode_numbers=mode_numbers,
        mesh=mesh,
    )


# --------------------------------------------------------- bi-level taper


def sacher_bilevel_dataset_info(
    wavelength=1.55e-6,
    mode_numbers=6,
    mesh=220,
    bottom_sidewall_angle=90.0,
    top_sidewall_angle=90.0,
):
    """A rib: full-height core on a wider 90 nm slab - the TM0 -> TE1 rotator.

    Grid axes ``w_core`` (450 -> 550 -> 850 nm) and ``w_slab`` (450 -> 1550 ->
    850 nm).  The slab is centred on the core, so the vertical mirror plane
    survives and only the *horizontal* one is broken - which is the one that
    separates TE from TM.

    The TM0/TE1 anti-crossing sits near ``w_slab`` = 900 nm and is far gentler
    than the coupler's: its branches stay ~0.17 apart in effective index, so a
    20 nm step mixes them by only 0.08 against the coupler's 0.54.  10 nm
    across the crossing is therefore ample.

    ``bottom_sidewall_angle`` and ``top_sidewall_angle`` are in degrees from
    horizontal, 90 being vertical.  The bottom wall is the full etch that
    defines the slab, the top wall the partial etch that defines the core above
    it.  Leaving both at 90 reproduces the rectangular geometry exactly, down to
    the dataset fingerprint.
    """

    def cross_section(core, cladding, wl, names):
        return BiLevelStrip(
            thickness=THICKNESS,
            slab_thickness=SLAB_THICKNESS,
            core=core,
            cladding=cladding,
            swept_parameters=names,
            reference_wavelength=wl,
            bottom_sidewall_angle=bottom_sidewall_angle,
            top_sidewall_angle=top_sidewall_angle,
        )

    parameters = {
        # 450 and 550 and 850 all land on the axis exactly.
        "w_core": axis(nm(450, 560, 10), nm(570, 900, 20)),
        # 450, 850 and 1550 all land on the axis exactly.
        "w_slab": axis(nm(450, 700, 20), nm(700, 1010, 10), nm(1010, 1610, 20)),
    }
    return _make_dataset_info(
        name="Sacher bi-level taper: 220 nm Si core on a 90 nm slab",
        description="overlap, neff and TE_pol for a bi-level (rib) Si waveguide",
        wavelength=wavelength,
        parameters=parameters,
        parameter_names=("w_core", "w_slab"),
        cross_section_factory=cross_section,
        window=(2.2e-6, -0.8e-6, 0.8e-6),
        mode_numbers=mode_numbers,
        mesh=mesh,
    )


# ------------------------------------------------- rapid adiabatic coupler


#: Fargas Cabanillas thesis section 3.5.3, the 220 nm SOI e-beam RAC.
RAC_GAP = 100e-9
RAC_W_BOT = 480e-9      # w_bot, the wider guide at the input of region III
RAC_W_TOP = 380e-9      # w_top, the narrower guide
RAC_W_EQUAL = 380e-9    # w_e, the symmetric output width
RAC_LC = 23e-6          # L_c = L_g + L_e, regions II and III together


def rac_dataset_info(
    wavelength=1.55e-6,
    mode_numbers=6,
    mesh=200,
    offsets_um=1.0,
    offset_step_nm=5,
    clad="oxide",
):
    """Region III of the RAC: two strips at a fixed 100 nm gap.

    The coupling region of the rapid adiabatic coupler in Fargas Cabanillas,
    *Rapid adiabatic devices enabling integrated electronic-photonic quantum
    systems on chip*, PhD thesis, Boston University, section 3.5.3 - the 220 nm
    SOI e-beam demonstration.  Widths run from ``w_bot`` = 480 nm beside
    ``w_top`` = 380 nm to a symmetric 380/380 nm pair at a constant 100 nm gap.

    Only ``w1`` varies along the device, so ``w2`` is held fixed rather than put
    on the grid.  The second axis is instead ``x_offset``: a rigid lateral
    translation of the pair, which is how the *tilt* that defines a RAC enters.
    A translation does nothing to a single cross section, but between one
    section and the next it displaces the modes relative to each other, and
    that displacement is the ``dx_p/dz`` term the thesis nulls to suppress
    inter-supermode coupling.

    :param offsets_um: Half-range of the lateral offset axis, in micrometres.
    :param offset_step_nm: Step of the ``x_offset`` axis, in nanometres.  This
        is not a free accuracy knob.  A RAC is defined by the *slope* of its
        outline, ``dx/dw1``, and a grid can only realise slopes that are ratios
        of its two axis steps.  The nominal design here runs 502 nm of lateral
        offset against 100 nm of width, i.e. a slope of 5, so a 5 nm offset step
        against the 2 nm width step near the symmetric point expresses it
        exactly.  The original 20 nm step could only produce slopes of 2, 4 and
        10, and the cached path zig-zagged around the intended outline - which
        destroys a mechanism that works by cancelling coupling to 1e-4.
    :param clad: ``"oxide"`` for a buried SiO2-clad guide, or ``"air"`` for the
        air-clad geometry of the fabricated device in section 3.5.3 (SiO2 box
        below, air above - vertically asymmetric, as the real chip is).
    """
    air_clad = str(clad).lower() == "air"

    def cross_section(core, cladding, wl, names):
        return CoupledStrips(
            thickness=THICKNESS,
            gap=RAC_GAP,
            w2=RAC_W_TOP,
            core=core,
            cladding=cladding,
            # Air above, box below.  Passing the box explicitly keeps the
            # buried case unchanged (substrate defaults to the cladding).
            substrate=silica(out_of_range="raise") if air_clad else None,
            swept_parameters=names,
            reference_wavelength=wl,
        )

    reach = int(round(offsets_um * 1000))
    step = int(offset_step_nm)
    parameters = {
        # Fine near the symmetric point, where the two branches are closest
        # and the device is hardest to keep adiabatic.
        "w1": axis(nm(370, 400, 2), nm(400, 440, 5), nm(440, 500, 10)),
        "x_offset": axis(nm(-reach, reach + step, step)),
    }
    half_width = 0.5 * (RAC_W_BOT + RAC_GAP + RAC_W_TOP) + offsets_um * 1e-6 + 1.2e-6
    return _make_dataset_info(
        name=(
            "Fargas RAC region III: 2 x 220 nm Si strips, 100 nm gap, "
            + ("air clad on a SiO2 box" if air_clad else "buried in SiO2")
        ),
        description="overlap, neff and TE_pol for the RAC coupling region",
        wavelength=wavelength,
        parameters=parameters,
        parameter_names=("w1", "x_offset"),
        cross_section_factory=cross_section,
        window=(half_width, -0.8e-6, 0.8e-6),
        mode_numbers=mode_numbers,
        mesh=mesh,
        cladding=air() if air_clad else None,
    )


# ---------------------------------------------------------------- plasmonics

PLASMONIC_SI_THICKNESS = 220e-9
# Wall height binds the lateral slot mode (PlasmonicSlotConverter): 400 nm
# walls, centred on the Si, hold the 40 nm slot at n = 1.15; 220 nm ones do not.
PLASMONIC_METAL_THICKNESS = 400e-9
PLASMONIC_W_SI_IN = 400e-9


def plasmonic_converter_dataset_info(
    wavelength=1.55e-6,
    gap=20e-9,
    metal_thickness=PLASMONIC_METAL_THICKNESS,
    mode_numbers=16,
    cell=5e-9,
    target_neff=2.0,
    pml_thickness=0.25e-6,
    substrate="air",
    corner_radius=20e-9,
    colocate=True,
):
    """Region of the Si-wire-to-plasmonic-slot converter: the lateral taper.

    Ono et al., *Optica* **3**, 999 (2016) / NTT Technical Review **16**(7)
    (2018): a 400 x 200 nm Si wire tapers laterally to a point over 600 nm
    inside a gold/air/gold slot, with an air gap ``gap`` between the Si edge
    and the metal (20 nm designed, 40 nm fabricated).  The cross section is
    :class:`~em_simulation.fde.slot_converter.PlasmonicSlotConverter`, the
    backend is the lossy :class:`~em_simulation.fde.pml.PMLBackend`, and the
    swept axis is the Si width.

    The axis is dense where the conversion happens.  The Si TE0 mode is cut
    off somewhere below ~150 nm and hands over to the gap plasmon; above that
    the wire is just a wire.

    :param cell: Grid pitch, metres, and the unit the geometry is snapped to.
        The window is rounded to whole cells with ``x = 0`` on a grid point,
        and the width axis runs in steps of ``2 * cell``, so the Si edge and
        the metal's inner edge fall **on grid points at every axis point**:
        every cross section is discretised identically and ``n_eff(w)`` is
        smooth.  It has to be.  A metal edge moving *inside* a cell swings the
        eps-averaged edge cell between air and gold, and the hybrid mode's
        index with it - at a 6 nm cell, 400 / 395 / 375 nm read 2.471 / 2.371
        / 2.429, and the spurious step at each interface showed up as a 1.5 %
        power *gain* per interface, independent of the mode count.  Widths
        off the axis (a direct run at arbitrary widths) do not get this
        protection.  The gold's skin depth is ~23 nm and the gap is 20 nm
        (4 cells at 5 nm).  Between the two pitches that put the gap on cell
        boundaries, 5 and 4 nm, the Si-end index moves 0.2 % and the slot's
        3 % (report 12 section 1); 5 nm is the shipped compromise (50 s a
        solve against 90 s).
    :param target_neff: Shift-invert target, between the Si-end mode (~2.4,
        a lateral hybrid plasmonic mode once the gold walls are 20 nm away)
        and the gap plasmon (~1.2-1.5) so one solve returns both.
    :param substrate: ``"air"`` (default, suspended) or ``"silica"``.  Over
        oxide the lateral slot's mode sits below the substrate index, leaks,
        and is not a port.  The device binds its plasmon vertically, which
        this lateral model cannot; suspending it is the honest stand-in.  Even
        then the slot binds only between tall walls - ``metal_thickness`` -
        and the 80 nm slot of the fabricated 40 nm gap is marginal
        (n = 1.008 between 400 nm walls).
    :param corner_radius: Rounding of the gold plates' corners, metres
        (20 nm = 4 cells at the shipped pitch).  Sharp metal wedges carry
        a singular, unresolvable field that every taper step sheds; the
        sharp variant is kept as ``Si_plasmonic_slot_1550_sharp`` for the
        comparison in report 12 section 8.
    :param colocate: Interpolate E onto H's node grid (``PMLModeSolver``).
        On, always, for a dataset built after 2026-09-07: the half-cell E/H
        offset of the zero-padded fields made every stored overlap
        non-symmetric (report 07 section 23), which is the ``O_ab != O_ba``
        that report 12 section 7.5 measured on this platform.  The key enters
        the dataset identity, so a cache built without it is refused.
    """
    from .fde.pml import PMLBackend
    from .fde.slot_converter import PlasmonicSlotConverter

    section = PlasmonicSlotConverter(
        si_thickness=PLASMONIC_SI_THICKNESS,
        metal_thickness=metal_thickness,
        metal_bottom=-0.5 * metal_thickness,      # centred on the Si
        corner_radius=corner_radius,
        gap=gap,
        core=silicon(out_of_range="raise"),
        metal=None,                      # Johnson & Christy gold
        cladding=air(),
        substrate=air() if substrate == "air" else silica(out_of_range="raise"),
        reference_wavelength=wavelength,
    )

    def cross_section(core, cladding, wl, names):
        # materials are fixed by the platform; the factory signature is shared
        return section

    def backend(cross_section, names, wl, window, modes, mesh_points):
        return PMLBackend(
            cross_section,
            target_neff=target_neff,
            parameter_names=names,
            wavelength=wl,
            window=window,
            mesh=mesh_points,
            mesh_y=mesh_y,
            pml_thickness=pml_thickness,
            # all four: the plates and the air above radiate laterally and
            # up, and a wide slot leaks into the oxide below
            pml_edges=("+x", "-x", "+y", "-y"),
            num_modes=modes,
            colocate=colocate,
        )

    step_nm = 2 * cell * 1e9
    parameters = {
        "w_si": axis(nm(0, PLASMONIC_W_SI_IN * 1e9 + 0.5 * step_nm, step_nm)),
    }
    half_width, y_min, y_max = section.default_window({"w_si": PLASMONIC_W_SI_IN})
    # whole cells, x = 0 and y = 0 on grid points: the pitch is then exactly
    # `cell` and the geometry's edges (all multiples of it) sit on grid points
    half_cells = int(np.ceil(half_width / cell))
    y_cells = int(np.ceil(max(-y_min, y_max) / cell))
    half_width = half_cells * cell
    y_min, y_max = -y_cells * cell, y_cells * cell
    mesh = 2 * half_cells + 1
    mesh_y = 2 * y_cells + 1
    return _make_dataset_info(
        name=(
            f"Ono plasmonic converter (lateral, {substrate}-suspended): Si wire in a "
            f"Au/air slot, gap {gap*1e9:.0f} nm, Au {metal_thickness*1e9:.0f} nm tall, "
            f"corners r = {corner_radius*1e9:.0f} nm"
        ),
        description="complex neff, TE_pol and overlaps for the lateral Si-to-slot taper",
        wavelength=wavelength,
        parameters=parameters,
        parameter_names=("w_si",),
        cross_section_factory=cross_section,
        window=(half_width, y_min, y_max),
        mode_numbers=mode_numbers,
        mesh=mesh,
        cladding=air(),
        backend_factory=backend,
    )


# --------------------------------------------------------------------------
# Kocabas: Si wire to plasmonic slot, fully embedded in SiO2
# --------------------------------------------------------------------------

#: Table II of S. E. Kocabas, arXiv:1801.00833 (2017): optimal converters for
#: 30 and 250 nm gold, everything in nm.  ``start`` is the untapered lead-in
#: over which the slot already exists, ``extra`` the slot after the tip.  The
#: Si is ``w_si -> w_end`` over ``l_taper`` while the slot goes independently
#: from ``w_si + 2 w_gap`` to ``w_slot``.
KOCABAS_SETS = {
    1: dict(h_au=30, h_si=300, w_si=400, w_slot=30, l_taper=600, w_end=0, w_gap=20,
            start=200, extra=200, transmission=0.88),
    2: dict(h_au=250, h_si=725, w_si=400, w_slot=250, l_taper=1700, w_end=0, w_gap=75,
            start=200, extra=200, transmission=0.95),
}
#: Table I of the same paper at 1550 nm, in this project's e^{i beta z}
#: convention (the paper uses e^{+i omega t}, so its Im(eps) is negative).
KOCABAS_EPS_SI = 12.085
KOCABAS_EPS_SIO2 = 2.0852
KOCABAS_EPS_AU = -126.80 + 5.3664j


def kocabas_materials():
    """The paper's constants as ``ConstantIndex`` materials."""
    from .fde.materials import ConstantIndex

    n_au = complex(np.sqrt(KOCABAS_EPS_AU))
    if n_au.imag < 0:
        n_au = -n_au
    return (
        ConstantIndex(float(np.sqrt(KOCABAS_EPS_SI)), name="Si_Kocabas"),
        ConstantIndex(float(np.sqrt(KOCABAS_EPS_SIO2)), name="SiO2_Kocabas"),
        ConstantIndex(n_au, name="Au_Kocabas"),
    )


def kocabas_path(set_number=2, w_gap=None, l_taper=None, w_slot=None, half_slot=False):
    """``{"w_si": f(z), "gap": f(z)}`` and the total length of one converter.

    Si width ``w_si -> w_end`` linearly over ``l_taper`` after a ``start``
    lead-in; the slot edges go linearly from ``w_si + 2 w_gap`` to ``w_slot``
    over the same length and stay at ``w_slot`` for ``extra``.  The gap the
    cross section sees is half the difference.  Any of ``w_gap``, ``l_taper``,
    ``w_slot`` may be overridden (metres) for a sweep.

    :param half_slot: Return ``{"w_si", "half_slot"}`` instead - the same
        device on the ``(w_si, half_slot)`` axes a tip-refined dataset uses
        (``kocabas_converter_dataset_info(tip_refine=...)``).
    """
    p = dict(KOCABAS_SETS[set_number])
    w_gap = p["w_gap"] * 1e-9 if w_gap is None else float(w_gap)
    l_taper = p["l_taper"] * 1e-9 if l_taper is None else float(l_taper)
    w_slot = p["w_slot"] * 1e-9 if w_slot is None else float(w_slot)
    w_in, w_end = p["w_si"] * 1e-9, p["w_end"] * 1e-9
    start, extra = p["start"] * 1e-9, p["extra"] * 1e-9
    slot_in = w_in + 2 * w_gap

    def frac(z):
        return np.clip((np.asarray(z, dtype=float) - start) / l_taper, 0.0, 1.0)

    def w_si(z):
        return w_in + (w_end - w_in) * frac(z)

    def gap(z):
        slot = slot_in + (w_slot - slot_in) * frac(z)
        return 0.5 * (slot - w_si(z))

    def half(z):
        return 0.5 * (slot_in + (w_slot - slot_in) * frac(z))

    if half_slot:
        return {"w_si": w_si, "half_slot": half}, start + l_taper + extra
    return {"w_si": w_si, "gap": gap}, start + l_taper + extra


def kocabas_converter_dataset_info(
    set_number=2,
    wavelength=1.55e-6,
    mode_numbers=20,
    cell=5e-9,
    target_neff=2.3,
    pml_thickness=0.25e-6,
    gap_range=(25e-9, 175e-9),
    corner_radius=0.0,
    plate_reach=0.3e-6,
    colocate=True,
    tip_refine=None,
    refine=None,
):
    """Kocabas's Si-wire-to-plasmonic-slot converter as a two-axis dataset.

    Everything is embedded in SiO2 and the Si and the gold are centred on
    each other vertically (the paper's Fig. 1b), which is what lets a
    plasmonic slot between films as thin as 30 nm bind: the slot is
    silica-filled and the environment symmetric, so its index sits above the
    surroundings - unlike an air slot on oxide (report 12 section 9).

    The shift-invert target sits **among the physical branches**, not below
    them.  Everything physical on the design path lies between the wire's
    2.44 and the slot's 1.45, while the Berenger band is at ``Re`` 1.2-1.45
    with ``Im`` 0.2-0.35 (``|n^2|`` about 1.5-2).  About a target of 1.9 the
    Berenger modes are *nearer* in ``n^2`` than the wire mode is, so once
    enough of them exist the fundamental drops out of the returned set - it
    did at 300 nm of Si - and the tracker links whatever is left: the
    launched power then rode a higher-order branch into cutoff and 99 % was
    lost.  About 2.3 the wire mode (distance 1.2) and the slot mode (2.7)
    both beat the Berenger band (about 3.2), at every width.

    Axes: ``w_si`` in steps of ``2 cell`` and ``gap`` in steps of ``cell``, so
    both the Si edge and the metal's inner edge fall on cell boundaries at
    every grid point (report 12 section 7.3).  The slot tapers independently
    of the Si in this layout, so the gap runs along the path and both axes
    are visited; lazy evaluation solves only the points a device touches.

    :param refine: Strips ``((lo, hi, cell_fine), ...)`` in metres of
        ``|x|`` (mirrored to ``-x``) inside which the grid is cut to
        ``cell_fine``; ``lo``, ``hi`` on the 5 nm base grid, ``cell`` an
        integer multiple of ``cell_fine``.  Any refinement switches the
        axes to ``("w_si", "half_slot")``: with ``(w_si, gap)`` the wall
        sits at ``w_si/2 + gap``, so a path detour ``(w_new, gap_old)``
        moves it by half a silicon step and off the coarse grid, and every
        "Si step" also moves the wall.  The axes follow the strips: the
        ``half_slot`` axis steps ``cell_fine`` where a wall position falls
        inside a strip and ``cell`` elsewhere; the ``w_si`` axis steps
        ``2 cell_fine`` only inside a strip that starts at ``0`` (a tip
        strip) and ``2 cell`` otherwise, since 10 nm width steps are cheap
        (report 13 section 7: fifty 2 nm steps at the tip cost 0.25 %).
        What the design path's staircase actually is, measured on clean
        axes, is the gold wall's 5 nm jumps once the mode is slot-like -
        6.1 % of the 7.2 % single-mode mismatch - so the strip to buy is
        the one the wall moves through, ``|x|`` = 125-275 nm.
    :param tip_refine: ``(half_extent, cell_fine)``, the same as
        ``refine=((0, half_extent, cell_fine),)``; kept for the
        ``_tip60_1`` dataset.
    """
    from .fde.pml import PMLBackend
    from .fde.slot_converter import PlasmonicSlotConverter

    p = KOCABAS_SETS[set_number]
    si, sio2, au = kocabas_materials()
    if tip_refine is not None:
        if refine is not None:
            raise ValueError("give tip_refine or refine, not both")
        refine = ((0.0, float(tip_refine[0]), float(tip_refine[1])),)
    strips = tuple((float(lo), float(hi), float(cf)) for lo, hi, cf in (refine or ()))
    for lo, hi, cf in strips:
        for v in (lo, hi):
            if abs(v / cell - round(v / cell)) > 1e-6:
                raise ValueError(f"refine: strip bound {v:g} m is not on the {cell:g} m base grid")
        if cf <= 0 or abs(cell / cf - round(cell / cf)) > 1e-6:
            raise ValueError(f"refine: base cell {cell:g} is not an integer multiple of {cf:g}")
        if hi <= lo or lo < 0:
            raise ValueError(f"refine: bad strip ({lo:g}, {hi:g})")
    refine_x = tuple(r for lo, hi, cf in strips
                     for r in (((-hi, hi, cf),) if lo == 0 else ((-hi, -lo, cf), (lo, hi, cf))))
    section = PlasmonicSlotConverter(
        si_thickness=p["h_si"] * 1e-9,
        metal_thickness=p["h_au"] * 1e-9,
        metal_bottom=-0.5 * p["h_au"] * 1e-9,          # centred on the Si
        gap=p["w_gap"] * 1e-9,
        plate_reach=plate_reach,
        corner_radius=corner_radius,
        core_mask_margin=0.0,
        sweep_gap=not strips,
        sweep_half_slot=bool(strips),
        core=si, metal=au, cladding=sio2, substrate=sio2,
        reference_wavelength=wavelength,
    )

    def cross_section(core, cladding, wl, names):
        return section

    step_nm = 2 * cell * 1e9
    g0, g1 = gap_range
    if not strips:
        names = ("w_si", "gap")
        parameters = {
            "w_si": axis(nm(0, p["w_si"] + 0.5 * step_nm, step_nm)),
            "gap": axis(nm(g0 * 1e9, g1 * 1e9 + 0.5 * cell * 1e9, cell * 1e9)),
        }
    else:
        names = ("w_si", "half_slot")
        hs0_nm = 0.5 * p["w_slot"]                       # the slot end
        hs1_nm = 0.5 * p["w_si"] + g1 * 1e9
        w_segments = [nm(0, p["w_si"] + 0.5 * step_nm, step_nm)]
        hs_segments = [nm(hs0_nm, hs1_nm + 0.5 * cell * 1e9, cell * 1e9)]
        for lo, hi, cf in strips:
            lo_nm, hi_nm, cf_nm = lo * 1e9, hi * 1e9, cf * 1e9
            if lo == 0:                                  # a tip strip: fine width steps
                w_segments.append(nm(0, 2 * hi_nm + cf_nm, 2 * cf_nm))
            # wall positions inside the strip step at the fine cell
            a, b = max(lo_nm, hs0_nm), min(hi_nm, hs1_nm)
            if b > a:
                hs_segments.append(nm(a, b + 0.5 * cf_nm, cf_nm))
        parameters = {"w_si": axis(*w_segments), "half_slot": axis(*hs_segments)}
        # every wall position must be a node: outside the strips that is the
        # base grid, so the slot end (hs0) has to be on it
        if abs(hs0_nm / (cell * 1e9) - round(hs0_nm / (cell * 1e9))) > 1e-6 and not any(
                lo * 1e9 <= hs0_nm <= hi * 1e9 for lo, hi, _ in strips):
            raise ValueError("refine: the slot end is neither on the base grid nor inside a strip")
    half_width = 0.5 * p["w_si"] * 1e-9 + g1 + plate_reach + 0.5e-6
    y_half = 0.5 * max(p["h_si"], p["h_au"]) * 1e-9 + 0.5e-6
    half_cells = int(np.ceil(half_width / cell))
    y_cells = int(np.ceil(y_half / cell))
    half_width, y_half = half_cells * cell, y_cells * cell
    mesh, mesh_y = 2 * half_cells + 1, 2 * y_cells + 1

    def backend(cross_section, names, wl, window, modes, mesh_points):
        return PMLBackend(
            cross_section, target_neff=target_neff, parameter_names=names,
            wavelength=wl, window=window, mesh=mesh_points, mesh_y=mesh_y,
            pml_thickness=pml_thickness, pml_edges=("+x", "-x", "+y", "-y"),
            num_modes=modes, colocate=colocate, refine_x=refine_x,
        )

    return _make_dataset_info(
        name=(f"Kocabas set {set_number}: Si wire {p['w_si']} x {p['h_si']} nm to a "
              f"{p['w_slot']} nm slot in {p['h_au']} nm Au, SiO2-embedded"),
        description=f"complex neff, TE_pol and overlaps of the {names} family, lossy PML basis",
        wavelength=wavelength,
        parameters=parameters,
        parameter_names=names,
        cross_section_factory=cross_section,
        window=(half_width, -y_half, y_half),
        mode_numbers=mode_numbers,
        mesh=mesh,
        cladding=sio2,
        backend_factory=backend,
    )
