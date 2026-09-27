"""Every repo path a document cites exists (`scripts/check_links.py`, task 16).

The gate for any rename: a moved report, task or module that leaves a
citation behind fails here, and so does an excuse-list entry that is no
longer needed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_links  # noqa: E402


def test_every_cited_path_exists():
    broken, seen, stale = check_links.check()
    assert seen > 100, f"only {seen} references found - is the scan broken?"
    assert not broken, "\n".join(f"{rel}:{n}: {target}" for rel, n, target in broken)
    assert not stale, f"excuse-list entries no longer needed: {sorted(stale)}"


def test_shorthand_and_module_references_resolve():
    tree = check_links.Tree(keep=["reports/07_sirac_optimization.md", "tasks/05b_crosscheck.md",
                                  "dbeme/validation.py", "dbeme/fde/pml.py"])
    assert tree.exists("reports/07", "README.md")
    assert tree.exists("tasks/05b", "README.md")
    assert tree.exists("dbeme/validation.lumped_smatrix", "README.md")
    assert tree.exists("dbeme/fde/", "README.md")
    assert tree.exists("../dbeme/fde/pml.py", "docs/x.md")
    assert not tree.exists("reports/99", "README.md")
    assert check_links.clean("reports/07_sirac_optimization.md §20") == "reports/07_sirac_optimization.md"
    assert check_links.clean("studies/<slug>/tests/") is None
    assert check_links.clean("dbeme/fde/pml.py:64") == "dbeme/fde/pml.py"
