"""Every repo path a document cites must exist (task 16, steps 0.9 and 1.1).

    python scripts/check_links.py                 # report broken references, exit 1 if any
    python scripts/check_links.py --list          # also print every reference checked
    python scripts/check_links.py --keep FILE     # check the tree FILE lists instead of the disk
    python scripts/check_links.py --keep FILE --write-companion
    python scripts/check_links.py --root DIR --also DIR2   # scan DIR, resolve in DIR and DIR2

Scans every ``.md`` under ``tasks/ reports/ docs/`` plus ``CLAUDE.md`` and
``README.md``.  A reference is a backticked span or a markdown link target
that starts with one of ``PREFIXES`` (or is ``CLAUDE.md`` / ``README.md``).
It must exist relative to a root or to the citing file's directory; the
house shorthand (`reports/07`, `tasks/05b`) and dotted ``module.name`` forms
resolve too.

Not checked, on purpose:

* anything inside a fenced block (``` or ~~~): code, or an archived brief
  kept verbatim;
* spans with glob or placeholder characters (``* ? < > { } [ ] … |``,
  ``...``), ``$``-variables, or whitespace inside - patterns, not paths;
* a ``:line``, ``#anchor``, `` §n`` or ``(...)`` suffix is stripped first.

Two lists under ``<root>/scripts/`` excuse a missing path:

* ``check_links_allow.txt`` - deliberately missing: a file superseded and
  removed, a path a plan names before it exists; each with its reason.  An
  entry whose path now exists is reported stale.
* ``check_links_companion.txt`` - cited here but outside the public tree:
  kept in the private companion repository (the tape-out work these documents
  draw evidence from), or generated locally (``examples/output/``).  It is
  written by ``--keep scripts/public_paths.txt --write-companion`` and holds
  exactly those cited paths, nothing else.  An entry no document cites any
  more is reported stale.

Both rules hold in the full private tree and in the public one, so one pair
of lists serves both.

It is the gate for every rename in task 16 (`tests/test_docs_links.py`).
"""

import os
import posixpath
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ("tasks", "reports", "docs")
SCAN_FILES = ("CLAUDE.md", "README.md")
PREFIXES = ("reports/", "tasks/", "studies/", "examples/", "output/", "datasets/",
            "docs/", "scripts/", "dbeme/", "tests/", "tests_circuit/",
            "references/")
BARE = ("CLAUDE.md", "README.md")

BACKTICK = re.compile(r"`([^`\n]+)`")
MDLINK = re.compile(r"\]\(([^)\s]+)\)")
PATTERN_CHARS = re.compile(r"[*?<>{}\[\]|$…]|\.\.\.")
SHORTHAND = re.compile(r"^(reports|tasks)/(\d+[a-z]?(?:_\d+)?)/?$")


class Tree:
    """Where a cited path may exist: directories on disk, or a keep-list."""

    def __init__(self, root=ROOT, also=(), keep=None):
        self.root = os.path.abspath(root)
        self.roots = [self.root] + [os.path.abspath(a) for a in also]
        self.keep = None
        if keep is not None:
            self.keep = set(keep)
            self.dirs = {posixpath.dirname(p) for p in self.keep}
            for d in list(self.dirs):
                while d:
                    d = posixpath.dirname(d)
                    self.dirs.add(d)

    def documents(self):
        if self.keep is not None:
            for rel in sorted(self.keep):
                if rel.endswith(".md") and (rel in SCAN_FILES or rel.startswith(SCAN_DIRS)):
                    yield rel
            return
        for name in SCAN_FILES:
            if os.path.exists(os.path.join(self.root, name)):
                yield name
        for top in SCAN_DIRS:
            for folder, dirs, files in os.walk(os.path.join(self.root, top)):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    if name.endswith(".md"):
                        rel = os.path.relpath(os.path.join(folder, name), self.root)
                        yield rel.replace(os.sep, "/")

    def _has(self, rel):
        rel = posixpath.normpath(rel)
        if self.keep is not None:
            return rel in self.keep or rel in self.dirs
        return any(os.path.exists(os.path.join(r, rel)) for r in self.roots)

    def _listing(self, folder):
        if self.keep is not None:
            return [posixpath.basename(p) for p in self.keep
                    if posixpath.dirname(p) == folder]
        names = []
        for r in self.roots:
            full = os.path.join(r, folder)
            if os.path.isdir(full):
                names += os.listdir(full)
        return names

    def exists(self, target, doc):
        """A path, a ``reports/07``-style shorthand, or a dotted ``module.name``."""
        here = posixpath.dirname(doc)
        candidates = [target, posixpath.join(here, target)]
        if any(self._has(c) for c in candidates):
            return True
        short = SHORTHAND.match(target)
        if short:
            return any(name.startswith(short[2] + "_") and name.endswith(".md")
                       for name in self._listing(short[1]))
        stem = target
        while "." in posixpath.basename(stem):              # ring_model.coupler
            stem = stem.rsplit(".", 1)[0]
            if any(self._has(c) for c in (stem + ".py", posixpath.join(here, stem + ".py"))):
                return True
        return False

    def read(self, rel):
        return open(os.path.join(self.root, rel), encoding="utf-8", errors="replace").read()


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
    bare = re.sub(r"^(\.\./)+", "", text)                   # ../reports/x, from docs/
    if bare.startswith(PREFIXES) or bare in BARE:
        return text
    return None


def references(text):
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


def load_list(root, name):
    path = os.path.join(root, "scripts", name)
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def check(verbose=False, tree=None, use_companion=True):
    """``(broken, seen, stale)``; ``stale`` covers both excuse lists."""
    tree = tree or Tree()
    allow = load_list(tree.root, "check_links_allow.txt")
    companion = load_list(tree.root, "check_links_companion.txt") if use_companion else set()
    broken, seen, cited = [], 0, set()
    for rel in tree.documents():
        for number, target in references(tree.read(rel)):
            seen += 1
            cited.add(target)
            if verbose:
                print(f"{rel}:{number}: {target}")
            if tree.exists(target, rel):
                continue
            if target in allow or target in companion:
                continue
            broken.append((rel, number, target))
    exists_now = {e for e in allow if tree.exists(e, "")}
    return broken, seen, exists_now | (companion - cited)


def main():
    args = sys.argv[1:]

    def value(flag):
        return args[args.index(flag) + 1] if flag in args else None

    root = value("--root") or ROOT
    also = [args[i + 1] for i, a in enumerate(args) if a == "--also"]
    keep = None
    if value("--keep"):
        keep = [l.strip() for l in open(value("--keep"), encoding="utf-8")
                if l.strip() and not l.startswith("#")]
    tree = Tree(root, also, keep)
    if "--write-companion" in args:
        if keep is None:
            raise SystemExit("--write-companion needs --keep")
        broken, _, _ = check(tree=tree, use_companion=False)
        targets = sorted({t for _, _, t in broken})
        path = os.path.join(tree.root, "scripts", "check_links_companion.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# Cited by the public documents but outside the public tree: kept in the\n"
                     "# private companion repository, or generated locally (examples/output/).\n"
                     "# Written by: scripts/check_links.py --keep scripts/public_paths.txt "
                     "--write-companion\n")
            fh.writelines(t + "\n" for t in targets)
        print(f"{len(targets)} companion paths -> {os.path.relpath(path, tree.root)}")
        return 0
    broken, seen, stale = check(verbose="--list" in args, tree=tree)
    for rel, number, target in broken:
        print(f"{rel}:{number}: missing {target}")
    for target in sorted(stale):
        print(f"excuse-list entry no longer needed: {target}")
    print(f"{seen} references checked, {len(broken)} broken, {len(stale)} stale excuse entries")
    return 1 if broken or stale else 0


if __name__ == "__main__":
    sys.exit(main())
