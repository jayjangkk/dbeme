"""`DBEME_DATASET_ROOTS`: datasets looked up by name across several roots."""

import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dbeme.data_updater.dataset_roots import resolve_dataset_dir  # noqa: E402

SHIPPED = os.path.join(ROOT, "datasets", "Si_fulletch_220nm")


def _roots(monkeypatch, *roots):
    monkeypatch.setenv("DBEME_DATASET_ROOTS", os.pathsep.join(str(r) for r in roots))


def test_unset_leaves_the_path_alone(monkeypatch):
    monkeypatch.delenv("DBEME_DATASET_ROOTS", raising=False)
    assert resolve_dataset_dir("anywhere/Si_fulletch_220nm") == "anywhere/Si_fulletch_220nm"


def test_a_name_resolves_in_whichever_root_holds_it(tmp_path, monkeypatch):
    own, shared = tmp_path / "own", tmp_path / "shared"
    _dataset(own / "private_set")
    _dataset(shared / "public_set")
    (own / "public_set" / "__pycache__").mkdir(parents=True)   # a leftover shell
    _roots(monkeypatch, own, shared)
    asked = str(own / "public_set")            # the path a project script builds
    assert resolve_dataset_dir(asked) == str(shared / "public_set")
    assert resolve_dataset_dir(str(own / "private_set")) == str(own / "private_set")
    assert resolve_dataset_dir(str(own / "new_set")) == str(own / "new_set")


def _dataset(path):
    path.mkdir(parents=True)
    (path / "dataset_info.py").write_text("")


def test_a_name_in_two_roots_raises(tmp_path, monkeypatch):
    for root in ("a", "b"):
        _dataset(tmp_path / root / "twin")
    _roots(monkeypatch, tmp_path / "a", tmp_path / "b")
    with pytest.raises(ValueError, match="more than one"):
        resolve_dataset_dir(str(tmp_path / "a" / "twin"))


def test_data_updater_opens_a_dataset_through_the_roots(tmp_path, monkeypatch):
    from dbeme import DataUpdater

    shared = tmp_path / "shared"
    shutil.copytree(SHIPPED, shared / "Si_fulletch_220nm")
    _roots(monkeypatch, tmp_path / "own", shared)
    du = DataUpdater(str(tmp_path / "own" / "Si_fulletch_220nm"))   # opening solves nothing
    assert du.data_directory == str(shared / "Si_fulletch_220nm")
    assert len(du.neff) > 0
