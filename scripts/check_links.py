"""Every repo path a document cites must exist (task 16, step 0.9).

    python scripts/check_links.py            # report broken references, exit 1 if any
    python scripts/check_links.py --list     # also print every reference checked

Scans every ``.md`` under ``tasks/ reports/ docs/`` plus ``CLAUDE.md`` and
``README.md``.  A reference is a backticked span or a markdown link target
that starts with one of ``PREFIXES`` (or is ``CLAUDE.md`` / ``README.md``).
It must exist relative to the repo root or to the citing file's directory.

Not checked, on purpose:

* anything inside a fenced block (``` or ~~~): code, or an archived brief
  kept verbatim;
* spans with glob or placeholder characters (``* ? < > { } [ ] … |``,
  ``...``), ``$``-variables, or whitespace inside - patterns, not paths;
* a ``:line``, ``#anchor``, `` §n`` or ``(...)`` suffix is stripped first;
* anything in ``scripts/check_links_allow.txt`` (one path per line, ``#``
  comments): references that are deliberately historical - a file that was
  superseded and removed, a planned path - each with its reason.

It is the gate for every rename in task 16 (`tests/test_docs_links.py`).
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ("tasks", "reports", "docs")
SCAN_FILES = ("CLAUDE.md", "README.md")
PREFIXES = ("reports/", "tasks/", "studies/", "examples/", "output/", "datasets/",
            "docs/", "scripts/", "dbeme/", "tests/", "tests_circuit/",
            "references/")
BARE = ("CLAUDE.md", "README.md")
ALLOW = os.path.join(ROOT, "scripts", "check_links_allow.txt")

BACKTICK = re.compile(r"`([^`\n]+)`")
MDLINK = re.compile(r"\]\(([^)\s]+)\)")
PATTERN_CHARS = re.compile(r"[*?<>{}\[\]|$…]|\.\.\.")


def documents():
    for name in SCAN_FILES:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            yield path
    for top in SCAN_DIRS:
        for folder, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name.endswith(".md"):
                    yield os.path.join(folder, name)


def clean(span):
    """The path inside a reference span, or None if it is not a path."""
    text = span.strip().strip("'\"")
    if text.startswith("./"):
        text = text[2:]
    text = re.split(r"\s+§|§|#|\s+\(", text)[0].strip()
    text = re.sub(r":\d+(-\d+)?(,\d+)*$", "", text)        # :line, :12-20
    text = re.sub(r"::[\w.]+$", "", text)                   # module.py::name
    text = text.rstrip(".,;:)")
    if not text or PATTERN_CHARS.search(text) or re.search(r"\s", text):
        return None
    if text.startswith(PREFIXES) or text in BARE:
        return text
    return None


def references(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        opener = re.match(r"\s*(`{3,}|~{3,})", line)
        if opener:
            mark = opener[1]
            if fence is None:
                fence = mark[0] * len(mark)
            elif mark.startswith(fence):
                fence = None
            continue
        if fence:                                           # code or archived text
            continue
        for regex in (BACKTICK, MDLINK):
            for match in regex.finditer(line):
                target = clean(match.group(1))
                if target:
                    yield number, target


def allowed():
    if not os.path.exists(ALLOW):
        return set()
    out = set()
    for line in open(ALLOW, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


SHORTHAND = re.compile(r"^(reports|tasks)/(\d+[a-z]?(?:_\d+)?)/?$")


def exists(target, doc):
    """A path, a ``reports/07``-style shorthand, or a dotted ``module.name``."""
    here = os.path.dirname(doc)
    for base in (ROOT, here):
        if os.path.exists(os.path.join(base, target)):
            return True
    short = SHORTHAND.match(target)
    if short:
        folder = os.path.join(ROOT, short[1])
        return any(name.startswith(short[2] + "_") and name.endswith(".md")
                   for name in os.listdir(folder))
    stem = target
    while "." in os.path.basename(stem):                    # ring_model.coupler
        stem = stem.rsplit(".", 1)[0]
        if any(os.path.exists(os.path.join(base, stem + ".py")) for base in (ROOT, here)):
            return True
    return False


def check(verbose=False):
    allow = allowed()
    broken, seen, used = [], 0, set()
    for doc in documents():
        rel = os.path.relpath(doc, ROOT).replace(os.sep, "/")
        for number, target in references(doc):
            seen += 1
            if verbose:
                print(f"{rel}:{number}: {target}")
            if exists(target, doc):
                continue
            if target in allow:
                used.add(target)
                continue
            broken.append((rel, number, target))
    return broken, seen, allow - used


def main():
    broken, seen, stale = check(verbose="--list" in sys.argv)
    for rel, number, target in broken:
        print(f"{rel}:{number}: missing {target}")
    for target in sorted(stale):
        print(f"allow-list entry no longer needed: {target}")
    print(f"{seen} references checked, {len(broken)} broken, "
          f"{len(allowed())} allow-listed, {len(stale)} stale allow entries")
    return 1 if broken or stale else 0


if __name__ == "__main__":
    sys.exit(main())
