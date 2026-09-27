"""What a cached dataset is keyed to, and a check that it still matches.

A dataset's pickles are a cache of one specific mode problem.  Change the
material model, the wavelength, the mesh, the solve window, the mode count or
the parameter grid, and every stored effective index and overlap matrix belongs
to a different problem - but the pickles keep loading and the numbers keep
looking reasonable.

That failure is easy to trigger now that materials can be swapped: switching
``main/Si/Li-293K`` for ``main/Si/Salzberg`` moves the core index by 2e-3,
roughly a hundred times the accuracy the method otherwise delivers, and nothing
about the stored data would say so.

So the identity is written next to the pickles as ``fingerprint.json`` and
checked on open.  The rule follows CLAUDE.md: the pickles are a cache and
``dataset_info.py`` is the source of truth, so a mismatch is an error telling
you to regenerate, never a silent overwrite.
"""

import json
import os

import numpy as np

FINGERPRINT_FILE = "fingerprint.json"

#: Bumped whenever the *meaning* of a stored overlap changes, as opposed to the
#: problem it describes.  The rest of the fingerprint keys the mode problem
#: (materials, grid, wavelength); this keys the convention used to turn solved
#: modes into the stored matrices.
#:
#: 1 - original.
#: 2 - forward modes are Löwdin-biorthogonalised per cross section before the
#:     backward basis is built, so that ``O_aa = I`` holds exactly
#:     (:func:`dbeme.fde.assemble.biorthogonalise`, docs/validation_backlog.md §5.6).
#:     Pre-fix overlaps differ by up to ~7e-2 between near-degenerate modes and
#:     ~3e-3 otherwise, and - crucially - a stored pre-fix overlap cascaded
#:     against a freshly solved post-fix one mixes two conventions, which is the
#:     exact failure §5.6 describes.  Hence a fingerprint key rather than a
#:     silent upgrade.
BASIS_CONVENTION = 2


def _grid_signature(parameter_grid):
    """Summarise a parameter grid without storing every point."""
    signature = {}
    for name in sorted(parameter_grid):
        values = np.asarray(parameter_grid[name], dtype=float)
        signature[name] = {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            # Catches a grid that kept its endpoints but changed its step.
            "sum": float(np.round(values.sum(), 12)),
        }
    return signature


def fingerprint(data_info, backend=None):
    """Build the identity dict for a dataset.

    :param data_info: The dataset's ``DatasetInfo``.
    :param backend: The mode solver, if one is attached.  In test mode there is
        none and only the metadata half of the fingerprint is recorded.
    :rtype: dict
    """
    identity = {
        "basis_convention": BASIS_CONVENTION,
        "wavelength_m": float(data_info.get_wavelength()),
        "mode_numbers": int(data_info.get_mode_numbers()),
        "parameter_names": list(data_info.get_parameter_names()),
        "parameter_grid": _grid_signature(data_info.get_parameter_grid()),
    }

    if backend is not None:
        identity["solver"] = type(backend).__name__
        window = getattr(backend, "window", None)
        if window is not None:
            identity["window_m"] = [float(v) for v in window]
        grid = getattr(backend, "solve_grid", None)
        if grid is not None:
            x, y = grid()
            identity["mesh"] = [int(len(x)), int(len(y))]
        cross_section = getattr(backend, "cross_section", None)
        if cross_section is not None and hasattr(cross_section, "fingerprint"):
            identity["cross_section"] = cross_section.fingerprint()
            thickness = getattr(cross_section, "thickness", None)
            if thickness is not None:
                identity["thickness_m"] = float(thickness)
        # A backend whose *recipe* changes the meaning of its modes - a PML's
        # edges, depth and stretch, the shift-invert target - says so itself.
        # ``EmepyFDE`` has no recipe beyond the grid, so it contributes
        # nothing and every existing lossless dataset keeps its identity.
        describe = getattr(backend, "fingerprint", None)
        if callable(describe):
            identity["backend"] = describe()

    return identity


def _differences(stored, current):
    """Human-readable list of the keys that disagree."""
    lines = []
    for key in sorted(set(stored) | set(current)):
        was, now = stored.get(key, "<absent>"), current.get(key, "<absent>")
        if was != now:
            lines.append(f"  {key}:\n      cached: {was}\n      now:    {now}")
    return lines


class DatasetIdentityError(RuntimeError):
    """The cached dataset was built for a different problem."""


def verify(data_directory, identity, has_cached_data):
    """Compare ``identity`` with the fingerprint stored beside the pickles.

    Writes the fingerprint when there is none yet, or when the dataset is
    empty and can therefore be adopted safely.

    :param data_directory: Dataset directory.
    :param identity: Result of :func:`fingerprint`.
    :param has_cached_data: Whether any points are already cached.
    :raises DatasetIdentityError: on a mismatch against non-empty pickles.
    """
    path = os.path.join(data_directory, FINGERPRINT_FILE)

    if not os.path.exists(path):
        # Either a fresh dataset, or one from before fingerprinting existed.
        # Adopt the current identity; there is nothing to compare against.
        _write(path, identity)
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (ValueError, OSError):
        _write(path, identity)
        return

    if stored == identity:
        return

    if not has_cached_data:
        _write(path, identity)
        return

    differences = "\n".join(_differences(stored, identity))
    raise DatasetIdentityError(
        f"the cached dataset in {data_directory} was built for a different "
        f"problem:\n{differences}\n\n"
        "Stored effective indices and overlaps are only valid for the exact "
        "cross section, wavelength, mesh, window and grid they were solved "
        "on. Delete the .pkl files (and this fingerprint.json) to regenerate, "
        "or point DataUpdater at a different dataset directory."
    )


def _write(path, identity):
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
    except OSError:
        # A read-only dataset directory is not a reason to refuse to run.
        pass
