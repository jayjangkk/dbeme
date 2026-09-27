r"""Refractive index as a function of wavelength.

Until now the cross sections carried bare floats - ``core_index = 3.4757``,
``clad_index = 1.444`` - which is exact at the one wavelength the dataset was
built for and wrong everywhere else.  Si moves by about 0.02 in index across
the O-to-L bands and SiO2 by about 0.004, so any bandwidth claim made with a
constant index is not meaningful.

This module puts a thin ``Material`` interface in front of the index so a cross
section can be handed a dispersion model instead of a number, and gets the
index at whatever wavelength it is being solved at.

Three sources, in decreasing order of preference:

* :class:`PyOptikMaterial` - the `refractiveindex.info
  <https://refractiveindex.info>`_ database via `PyOptik
  <https://github.com/MartinPdeS/PyOptik>`_.  Thousands of materials with
  their original references and validity ranges.  Needs the package and a
  one-off snapshot download.
* :class:`SellmeierMaterial` - a handful of dispersion formulas built in, so
  the project still runs with no PyOptik and no network.
* :class:`ConstantIndex` - a plain number, which is what a bare float becomes.

Use :func:`silicon`, :func:`silica` and friends unless you need a specific
source; they take the database entry when it is available and fall back to the
built-in formula when it is not.

Wavelengths are in **metres** throughout, matching the rest of the project.
PyOptik interprets a bare number as metres too, so the two agree.
"""

import abc
import hashlib
import functools
import pathlib
import warnings

import numpy as np

#: refractiveindex.info identifiers used by the convenience factories.
SI_PAGE = "main/Si/Li-293K"
SIO2_PAGE = "main/SiO2/Malitson"
AU_PAGE = "main/Au/Johnson"
SI3N4_PAGE = "main/Si3N4/Luke"


class Material(metaclass=abc.ABCMeta):
    """A refractive index that may depend on wavelength."""

    @abc.abstractmethod
    def index(self, wavelength):
        """Complex refractive index ``n + ik`` at ``wavelength`` (metres).

        Accepts a scalar or an array and returns the same shape.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short human-readable name, used in figures and messages."""

    @abc.abstractmethod
    def fingerprint(self) -> str:
        """Stable identity string.

        Two materials with the same fingerprint must give the same index at
        every wavelength.  A dataset records the fingerprints of its materials
        so that a cache built with one index model is never silently reused
        with another - see
        :func:`dbeme.data_updater.dataset_identity.fingerprint`.
        """

    def validity_range(self):
        """``(min, max)`` wavelength in metres, or ``None`` if unbounded."""
        return None

    def n(self, wavelength):
        """Real part of the index."""
        return np.real(self.index(wavelength))

    def k(self, wavelength):
        """Extinction coefficient."""
        return np.imag(self.index(wavelength))

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


class ConstantIndex(Material):
    """A wavelength-independent index.

    What a bare float becomes.  Correct at one wavelength by construction, and
    the right choice when you deliberately want to hold the index fixed - for
    instance when reproducing a published result that did the same.
    """

    def __init__(self, value, name=None):
        self.value = complex(value)
        self._name = name or f"n={self.value.real:.4f}"

    def index(self, wavelength):
        return np.full_like(np.asarray(wavelength, dtype=float), self.value, dtype=complex)

    @property
    def name(self):
        return self._name

    def fingerprint(self):
        return f"const:{self.value.real:.9g}{self.value.imag:+.9g}j"


class TabulatedIndex(Material):
    """A measured ``(wavelength, n)`` table, interpolated linearly.

    For index data that exists as *numbers from the process owner* rather than
    as a published fit: the platform document §5 gives Si (Palik, as
    sampled in the owner's Lumerical database) and SiO2 (SF_SIO2, measured
    in-house) exactly this way, and says of the silicon table that it "exists to
    match the reference solver, not to be smooth" - a one-term Sellmeier through
    those nine points misses one of them by 0.005, which is larger than the
    discrepancy the material was corrected to explain.

    Linear interpolation, not a spline, for the same reason: the table is the
    specification.

    :param out_of_range: ``"raise"`` (default) or ``"clip"``.  Raising is the
        right default - the platform doc's range guard says the interpolator
        "must still **raise** outside the measured span, never silently
        extrapolate", because a bandwidth plot built on extrapolated index data
        fails quietly.
    """

    def __init__(self, page, wavelengths, indices, name=None, out_of_range="raise"):
        self._page = str(page)
        self._name = name or self._page
        order = np.argsort(np.asarray(wavelengths, dtype=float))
        self.wavelengths = np.asarray(wavelengths, dtype=float)[order]
        self.indices = np.asarray(indices, dtype=float)[order]
        if out_of_range not in ("raise", "clip"):
            raise ValueError("out_of_range must be 'raise' or 'clip'")
        self.out_of_range = out_of_range

    def index(self, wavelength):
        value = np.asarray(wavelength, dtype=float)
        lo, hi = self.wavelengths[0], self.wavelengths[-1]
        if self.out_of_range == "raise" and (value.min() < lo or value.max() > hi):
            raise ValueError(
                f"{self._name}: {value.min()*1e9:.1f}-{value.max()*1e9:.1f} nm is "
                f"outside the measured span {lo*1e9:.2f}-{hi*1e9:.2f} nm"
            )
        return np.interp(value, self.wavelengths, self.indices).astype(complex)

    @property
    def name(self):
        return self._name

    def fingerprint(self):
        # The table itself, not just its name: two revisions of "Palik" that
        # disagree must not share a dataset or a cached path.
        digest = hashlib.sha256(
            np.concatenate([self.wavelengths, self.indices]).tobytes()
        ).hexdigest()[:12]
        return f"tab:{self._page}:{digest}"


class SellmeierMaterial(Material):
    r"""A Sellmeier dispersion formula, evaluated offline.

    .. math::

        n^2 - 1 = \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}

    with :math:`\lambda` in micrometres and :math:`C_i` in µm².  This is the
    fallback when PyOptik is unavailable; the coefficients are the published
    ones, cited in :data:`SELLMEIER_COEFFICIENTS`.
    """

    def __init__(self, terms, name, reference="", valid_range=None):
        """
        :param terms: Sequence of ``(B, C)`` pairs, ``C`` in µm².
        :param name: Short name.
        :param reference: Literature citation for the coefficients.
        :param valid_range: ``(min, max)`` wavelength in metres.
        """
        self.terms = tuple((float(b), float(c)) for b, c in terms)
        self._name = name
        self.reference = reference
        self._valid_range = valid_range

    def index(self, wavelength):
        lam_um = np.asarray(wavelength, dtype=float) * 1e6
        lam2 = lam_um**2
        n2 = np.ones_like(lam2)
        for b, c in self.terms:
            n2 = n2 + b * lam2 / (lam2 - c)
        return np.sqrt(n2.astype(complex))

    @property
    def name(self):
        return self._name

    def validity_range(self):
        return self._valid_range

    def fingerprint(self):
        terms = ",".join(f"{b:.9g}/{c:.9g}" for b, c in self.terms)
        return f"sellmeier:{self._name}:{terms}"


#: Built-in dispersion formulas, used when PyOptik is not available.
SELLMEIER_COEFFICIENTS = {
    "Si": dict(
        terms=[
            (10.6684293, 0.301516485**2),
            (0.003043475, 1.13475115**2),
            (1.54133408, 1104.0**2),
        ],
        name="Si (Salzberg 1957)",
        reference="C. D. Salzberg and J. J. Villa, J. Opt. Soc. Am. 47, 244 (1957)",
        valid_range=(1.36e-6, 11e-6),
    ),
    "SiO2": dict(
        terms=[
            (0.6961663, 0.0684043**2),
            (0.4079426, 0.1162414**2),
            (0.8974794, 9.896161**2),
        ],
        name="SiO2 (Malitson 1965)",
        reference="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965)",
        valid_range=(0.21e-6, 6.7e-6),
    ),
    "Si3N4": dict(
        terms=[(3.0249, 0.1353406**2), (40314.0, 1239.842**2)],
        name="Si3N4 (Luke 2015)",
        reference="K. Luke et al., Opt. Lett. 40, 4823 (2015)",
        valid_range=(0.31e-6, 5.5e-6),
    ),
}


# --------------------------------------------------------------- PyOptik glue


class PyOptikUnavailable(RuntimeError):
    """PyOptik is not installed, or its material snapshot is missing."""


def _patch_pathlib_utf8():
    """Make ``Path.open`` default to UTF-8.

    PyOptik reads its YAML and JSON with a bare ``path.open("r")``, so on
    Windows the cp1252 default codec hits the non-ASCII author names in the
    refractiveindex.info database and the whole catalog fails to load.  The
    files are UTF-8; this supplies that default when the caller did not ask for
    an encoding, and leaves binary modes alone.

    Applied once, at import of the catalog, and left in place - it only changes
    behaviour where an encoding was previously left to the platform, which is a
    bug in every case we care about.
    """
    original = pathlib.Path.open
    if getattr(original, "_dbeme_utf8", False):
        return

    @functools.wraps(original)
    def open_utf8(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return original(self, mode, buffering, encoding, errors, newline)

    open_utf8._dbeme_utf8 = True
    pathlib.Path.open = open_utf8


@functools.lru_cache(maxsize=1)
def get_catalog():
    """The local PyOptik catalog, built once.

    Uses the already-downloaded snapshot directly rather than
    ``MaterialCatalog.from_snapshot()``, which re-checks the upstream archive
    and is slow even when nothing needs fetching.

    :raises PyOptikUnavailable: if PyOptik or its snapshot is missing.
    """
    try:
        from PyOptik import MaterialCatalog
        from PyOptik.directories import user_data_path
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PyOptikUnavailable(
            "PyOptik is not installed. Install it with `pip install PyOptik`, "
            "then download the material snapshot once with `pyoptik setup`."
        ) from exc

    _patch_pathlib_utf8()

    root = pathlib.Path(user_data_path) / "rii"
    catalog_file = root / "catalog-nk.yml"
    if not catalog_file.exists():
        raise PyOptikUnavailable(
            f"PyOptik material snapshot not found at {root}. Download it once "
            "with `pyoptik setup`, or `python -c \"from PyOptik import "
            'download_snapshot; download_snapshot()"`.'
        )
    return MaterialCatalog(catalog_file=catalog_file, data_root=root)


class PyOptikMaterial(Material):
    """A material from the refractiveindex.info database, via PyOptik.

    :param page: ``shelf/book/page`` identifier, e.g. ``"main/Si/Li-293K"``.
    :param out_of_range: What to do outside the source's validity range -
        ``"warn"`` (default), ``"raise"`` or ``"clip"``.  ``"raise"`` is the
        right choice for a wavelength sweep, where silently extrapolating a
        fitted formula is how a bandwidth plot goes quietly wrong.
    """

    def __init__(self, page, out_of_range="warn"):
        self.page = str(page)
        self.out_of_range = out_of_range
        self._material = None

    @property
    def material(self):
        """The underlying PyOptik material, loaded on first use."""
        if self._material is None:
            catalog = get_catalog()
            try:
                self._material = catalog.get(self.page).load()
            except Exception as exc:
                raise PyOptikUnavailable(
                    f"could not load {self.page!r} from the PyOptik catalog: {exc}"
                ) from exc
        return self._material

    def index(self, wavelength):
        # PyOptik reads a bare number as metres, which is our convention too.
        value = self.material.compute_refractive_index(
            np.asarray(wavelength, dtype=float), out_of_range=self.out_of_range
        )
        return np.asarray(value, dtype=complex)

    @property
    def name(self):
        return self.page

    def validity_range(self):
        bound = getattr(self.material, "wavelength_bound", None)
        if bound is None:
            return None
        # A pint Quantity in micrometres; strip the units back to metres.
        try:
            magnitudes = np.asarray(bound.to("meter").magnitude, dtype=float)
        except AttributeError:
            magnitudes = np.asarray(bound, dtype=float) * 1e-6
        return float(magnitudes[0]), float(magnitudes[1])

    def fingerprint(self):
        return f"pyoptik:{self.page}"


# ------------------------------------------------------------------ factories


def _preferred(page, fallback_key, out_of_range="warn"):
    """A database material if the database is there, else the built-in formula."""
    try:
        material = PyOptikMaterial(page, out_of_range=out_of_range)
        material.material  # force the load now so the fallback can catch it
        return material
    except PyOptikUnavailable as exc:
        spec = SELLMEIER_COEFFICIENTS[fallback_key]
        warnings.warn(
            f"{exc} Falling back to the built-in {spec['name']} formula. "
            "Its index differs from the database entry by ~2e-3, so datasets "
            "built with the two are not interchangeable (the dataset "
            "fingerprint will differ, which is what stops them being mixed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return SellmeierMaterial(**spec)


def silicon(out_of_range="warn"):
    """Crystalline Si. Li 1980 at 293 K, valid 1.2-14 µm; n = 3.4757 at 1550 nm."""
    return _preferred(SI_PAGE, "Si", out_of_range)


def silica(out_of_range="warn"):
    """Fused SiO2. Malitson 1965, valid 0.21-6.7 µm; n = 1.4440 at 1550 nm."""
    return _preferred(SIO2_PAGE, "SiO2", out_of_range)


def silicon_nitride(out_of_range="warn"):
    """Stoichiometric Si3N4. Luke 2015; n = 1.9963 at 1550 nm."""
    return _preferred(SI3N4_PAGE, "Si3N4", out_of_range)


def gold(out_of_range="warn"):
    """Gold. Johnson & Christy 1972, valid 0.19-1.94 um.

    ``n = 0.524 + 10.74i`` at 1550 nm, i.e. ``eps = -115.1 + 11.26i``: an
    absorbing metal has ``Im(n) > 0`` in this project's ``e^{i beta z}``
    convention, so nothing needs sign-flipping downstream.  Thin-film data
    (``main/Au/Yakubovsky-25nm``, ``eps = -118.9 + 13.0i``) is the closer
    match for a 20 nm evaporated film; the difference is a few percent in the
    plasmon index and is the size of the metal-data uncertainty itself.
    """
    return _preferred(AU_PAGE, "Au", out_of_range)


def air():
    """Vacuum/air, index 1."""
    return ConstantIndex(1.0, name="air")


def as_material(spec, name=None):
    """Coerce a material specification into a :class:`Material`.

    Accepts a :class:`Material` (returned unchanged), a number (becomes
    :class:`ConstantIndex`), a ``"shelf/book/page"`` string (becomes
    :class:`PyOptikMaterial`), or any object with a
    ``compute_refractive_index`` method (a PyOptik material passed directly).

    :raises TypeError: for anything else.
    """
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        return PyOptikMaterial(spec)
    if hasattr(spec, "compute_refractive_index"):
        return _WrappedPyOptik(spec, name)
    if np.isscalar(spec) or isinstance(spec, complex):
        return ConstantIndex(spec, name=name)
    raise TypeError(
        f"cannot interpret {spec!r} as a material: expected a Material, a "
        "number, a 'shelf/book/page' string, or a PyOptik material object"
    )


class _WrappedPyOptik(Material):
    """A PyOptik material object handed to us directly."""

    def __init__(self, material, name=None):
        self._material = material
        self._name = name or getattr(material, "catalog_id", None) or repr(material)

    def index(self, wavelength):
        value = self._material.compute_refractive_index(
            np.asarray(wavelength, dtype=float)
        )
        return np.asarray(value, dtype=complex)

    @property
    def name(self):
        return str(self._name)

    def fingerprint(self):
        return f"pyoptik-object:{self._name}"


def check_lossless(materials, wavelength, tolerance=1e-6):
    """Warn if any material has significant absorption at ``wavelength``.

    The EME layer builds its backward-propagating basis by conjugation
    (``E- = conj(E+)``, ``n_eff -> -n_eff``), which is time reversal and only
    valid for a real index - conjugating a lossy mode turns loss into gain.
    ``correct_gain_modified`` then clips any negative ``Im(n_eff)`` to zero.
    Both are fine for Si and SiO2 in the telecom bands and both are wrong for a
    metal, so a material with real absorption should not pass silently.

    :param materials: Iterable of :class:`Material`.
    :param wavelength: Wavelength in metres.
    :param tolerance: Largest ``k`` treated as negligible.
    :returns: List of ``(material, k)`` that exceeded the tolerance.
    """
    offenders = []
    for material in materials:
        k = float(np.max(np.abs(material.k(wavelength))))
        if k > tolerance:
            offenders.append((material, k))
    if offenders:
        listed = ", ".join(f"{m.name} (k={k:.3g})" for m, k in offenders)
        warnings.warn(
            f"absorbing material(s) at {wavelength*1e9:.0f} nm: {listed}. "
            "This solver's backward-mode basis assumes a real index, and "
            "negative Im(n_eff) is clipped to zero, so loss will not be "
            "modelled correctly. Only the real part is used.",
            RuntimeWarning,
            stacklevel=2,
        )
    return offenders
