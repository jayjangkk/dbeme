"""Every repo path a document cites exists (`scripts/check_links.py`, task 16 step 0.9).

The gate for any rename: a moved report, task or module that leaves a
citation behind fails here, and so does an allow-list entry that is no longer
needed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_links  # noqa: E402


def test_every_cited_path_exists():
    broken, seen, stale = check_links.check()
    assert seen > 1000, f"only {seen} references found - is the scan broken?"
    assert not broken, "\n".join(f"{rel}:{n}: {target}" for rel, n, target in broken)
    assert not stale, f"allow-list entries no longer needed: {sorted(stale)}"


def test_shorthand_and_module_references_resolve(tmp_path):
    doc = str(tmp_path / "doc.md")
    assert check_links.exists("reports/07", doc)
    assert check_links.exists("tasks/05b", doc)
    assert check_links.exists("dbeme/validation.lumped_smatrix", doc)
    assert not check_links.exists("reports/99", doc)
    assert check_links.clean("reports/07_sirac_optimization.md §20") == "reports/07_sirac_optimization.md"
    assert check_links.clean("studies/<slug>/tests/") is None
    assert check_links.clean("dbeme/fde/pml.py:64") == "dbeme/fde/pml.py"
