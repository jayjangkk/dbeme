"""Compatibility shims so that ``emepy`` / ``EMpy_gpu`` import and run on
modern NumPy / SciPy / Python 3.13.

Four problems are patched here:

0. ``numpy.trapz`` was renamed ``numpy.trapezoid`` in NumPy 2.0.
1. ``EMpy_gpu/__init__.py`` does ``from numpy.testing import Tester``.
   ``numpy.testing.Tester`` was removed in NumPy 1.25.
2. ``EMpy_gpu.modesolvers.FD`` calls ``scipy.sqrt`` / ``scipy.log``.  The NumPy
   aliases in the root ``scipy`` namespace were removed in SciPy 1.12.  The old
   behaviour is ``numpy.emath.*`` (returns complex for negative input), which is
   what the mode solver relies on for lossy/evanescent eigenvalues.
3. ``emepy/__init__.py`` eagerly imports ``emepy.eme`` and ``emepy.models``,
   which need ``simphony`` 0.6 (``from simphony import Model``).  Current
   simphony (0.7+) has a completely different API, so that import fails.  We
   only need the finite-difference mode solver, so we register an *empty*
   ``emepy`` package object whose ``__path__`` points at the installed source
   tree.  Submodule imports (``emepy.fd``, ``emepy.mode``, ``emepy.tools``,
   ``emepy.materials``) then work normally and nothing else is executed.

Import this module before anything that touches emepy.
"""

import importlib.util
import os
import sys
import types

_applied = False
_emepy_dir = None


def apply() -> str:
    """Install the shims. Returns the emepy package directory. Idempotent."""
    global _applied, _emepy_dir
    if _applied:
        return _emepy_dir

    # --- 1. numpy.testing.Tester ------------------------------------------
    import numpy.testing as _nt

    if not hasattr(_nt, "Tester"):
        class _Tester:  # pragma: no cover - EMpy imports it but never uses it
            def __init__(self, *args, **kwargs):
                pass

            def test(self, *args, **kwargs):
                pass

            def bench(self, *args, **kwargs):
                pass

        _nt.Tester = _Tester

    # --- 2. scipy.sqrt / scipy.log ----------------------------------------
    import numpy as np
    import scipy

    if not hasattr(scipy, "sqrt"):
        scipy.sqrt = np.emath.sqrt
    if not hasattr(scipy, "log"):
        scipy.log = np.emath.log

    # numpy.trapz was renamed numpy.trapezoid in NumPy 2.0; EMpy_gpu and
    # emepy.mode both still call numpy.trapz.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid

    # EMpy_gpu reaches `eigs` through the deprecated `scipy.sparse.linalg.eigen`
    # namespace.  It still works; silence the warning it raises on every solve
    # rather than have it drown out real output.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*scipy\.sparse\.linalg\.eigen.*",
        category=DeprecationWarning,
    )

    # --- 3. emepy package without its eager __init__ ----------------------
    existing = sys.modules.get("emepy")
    if getattr(existing, "_dbeme_stub", False):
        _emepy_dir = existing.__path__[0]
    else:
        spec = importlib.util.find_spec("emepy")
        if spec is None or spec.origin is None:
            raise ImportError(
                "emepy is not installed. Install it with:  pip install emepy --no-deps"
            )
        _emepy_dir = os.path.dirname(spec.origin)
        stub = types.ModuleType("emepy")
        stub.__path__ = [_emepy_dir]
        stub.__doc__ = (
            "Partially loaded emepy (mode solver only) "
            "- see dbeme.fde._compat"
        )
        stub._dbeme_stub = True
        sys.modules["emepy"] = stub

    _applied = True
    return _emepy_dir
