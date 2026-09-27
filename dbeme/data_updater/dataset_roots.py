"""Where a dataset directory lives when datasets are spread over several roots.

By default a dataset is the directory it is opened with, and nothing here
changes that.  Set ``DBEME_DATASET_ROOTS`` - directories separated by
``os.pathsep`` - and a dataset is looked up **by its directory name** in those
roots instead, so a project that keeps its own datasets next to a checkout of
this package (``<project>/datasets`` and ``<project>/dbeme/datasets``) opens
both kinds with the same ``<project>/datasets/<name>`` path.

A name present in more than one root raises: two datasets of one name are two
different caches, and picking one silently would mix them.  A name in no root
falls back to the path as given, which is how a new dataset is started.
"""

import os

__all__ = ["dataset_roots", "resolve_dataset_dir"]


def dataset_roots():
    """The roots named by ``DBEME_DATASET_ROOTS``, read on every call."""
    raw = os.environ.get("DBEME_DATASET_ROOTS", "")
    return [os.path.abspath(r) for r in raw.split(os.pathsep) if r.strip()]


def resolve_dataset_dir(path):
    """``path``, or the one root directory holding a dataset of its name."""
    roots = dataset_roots()
    if not roots:
        return path
    name = os.path.basename(os.path.normpath(path))
    found = []
    for root in roots:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate) and candidate not in found:
            found.append(candidate)
    if len(found) > 1:
        raise ValueError(f"dataset {name!r} is in more than one DBEME_DATASET_ROOTS "
                         f"entry: {found}")
    return found[0] if found else path
