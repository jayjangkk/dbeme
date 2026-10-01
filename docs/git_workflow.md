# Git and GitHub workflow

Rules for anyone — human or agent — who commits to this repository. The
public remote is `https://github.com/jayjangkk/dbeme`; only `main` and the
release tags `vX.Y.Z` are published. Every version that has been released
is described in `changelog/`.

**Agents: the short version.** Work on a branch, never on `main`. Record
every behavioural change in `changelog/unreleased.md` in the same commit.
Never push, merge into `main`, tag a release, rewrite history or delete a
tag unless Jae has asked for that specific action in this session.

---

## 1. Versions

`X.Y.Z`, in the sense of Semantic Versioning, made concrete for a numerical
solver: what breaks for a user of this code is either their **code**, their
**datasets** (hours of mode solves), or the **numbers** they reported.

| bump | when — any one of these | examples |
|---|---|---|
| **X** major | a public import, class, function or argument is renamed or removed; **or** a stored dataset or path cache stops loading or is refused by the identity guard (`dataset_identity.BASIS_CONVENTION` bump, a fingerprint change, a pickle layout change) | `em_simulation` → `dbeme`; the Löwdin biorthogonalisation that set `BASIS_CONVENTION = 2` |
| **Y** minor | a new capability (backend, cross-section class, geometry, circuit model, solver option); **or** a change that moves a validated number with existing datasets still loading — then the changelog must carry a *Results changed* entry | `FemwellBackend`; `INTERFACE_COLUMN_CAP = "auto"` as default |
| **Z** patch | no API change and no validated number moves: a new example, report, study, dataset built from existing classes, test, document, or a bug fix whose effect is below every stated tolerance | `demo_ring_resonator.py`; `reports/16`; a new `Si_pair_*_1290` dataset |

*Validated number* means a value in README → *Validation at a glance*, in a
report's sanity gate, or asserted by a test. If a fix moves one of those
beyond its stated tolerance, it is at least **Y**, however small the diff.
When two rows apply, the highest wins. A bump resets the lower fields
(`1.4.2` → `1.5.0`, `1.5.0` → `2.0.0`).

**The direction of the rule is one-way.** A breaking change **must** be
major — never hidden in a minor or patch, because `dbeme>=1.2,<2` and every
downstream pin (the private repository first) take minors automatically. A
new capability is **at least** minor, and Jae **may** declare a major for a
milestone capability even without a break. Agents propose the bump; Jae
decides it.

The version lives in exactly two places, changed together in the release
commit: `pyproject.toml` (`version =`) and `CITATION.cff` (`version:`,
`date-released:`).

## 2. Releases are immutable

A released version is the commit its tag `vX.Y.Z` points at.

* A release tag is **never moved, re-created or deleted**, locally or on
  GitHub. A mistake in a release is fixed by a new release.
* A released `changelog/vX.Y.Z.md` is **never edited** after its tag is
  pushed, except to add a line under a final `## Errata` heading that
  points to the version which fixes the problem.
* The archive of a version is its tag, not a branch. A branch is mutable by
  nature; do not create `archive/*` branches.
* If a released line ever needs a fix that `main` cannot carry (e.g. `main`
  is already at `2.x`), branch `release/X.Y` from the tag, fix there, release
  `X.Y.(Z+1)`. Not before it is needed.

GitHub enforces this with a tag ruleset (§6); the rule holds without it.

## 3. Branches

Trunk-based. One long-lived branch, `main`, which is always green and
always releasable.

| branch | for | example |
|---|---|---|
| `main` | released and release-ready code only | |
| `feat/<slug>` | a capability, a device task, a backend | `feat/task18-edge-coupler` |
| `fix/<slug>` | a correction | `fix/interface-reflection` |
| `docs/<slug>` | documents, reports, examples only | `docs/readme-install` |
| `release/X.Y` | only per §2, last bullet | |

* Branch from the current `main`; keep a branch to one task; rebase or merge
  `main` into it before it is merged back.
* A branch reaches `main` with `git merge --no-ff <branch>` after the gate
  in §4 passes and Jae has said to merge. No direct commits on `main` other
  than the release commit (§5).
* **Restore points.** Before changing `dbeme/`, `CLAUDE.md` asks for a commit
  or a `pre_<label>` tag. On a branch, prefer a commit. `pre_*` tags are
  local checkpoints: **never push them**, and never run `git push --tags`
  (it pushes every local tag). Push tags by name only.

## 4. Commits

* One logical change per commit. The message's first line says what
  changed and where it came from: `Task 18 phase 2: fibre launch overlap`
  or `fix: interface reflection block uses O_ab^T` — the existing history's
  style.
* A commit that changes behaviour, adds a capability or moves a number
  updates `changelog/unreleased.md` **in the same commit**, under the
  heading that applies (template in `changelog/README.md`), and states the
  bump it implies (X, Y or Z) in that entry.
* Never commit: anything under `cache/`, `.venv*/`, run logs, credentials,
  or anything platform-specific (`CLAUDE.md` §0). New datasets: only those a
  test, example or report reads.
* **No file over 50 MB in git.** GitHub warns above 50 MB and rejects a push
  containing any file over 100 MB — and once committed, the file is in the
  history even if deleted later. A dataset that large keeps its
  `dataset_info.py` and `fingerprint.json` in git, its pickles in
  `.gitignore`, and ships zipped as an asset of the GitHub Release that
  first needs it (§5 step 8). Check before every commit that adds data:

  ```bash
  git diff --cached --name-only | xargs -r ls -l 2>/dev/null | awk '$5 > 50e6'
  ```
* Gate before a branch is offered for merge:

  ```bash
  .venv/Scripts/python -m pytest tests -q
  .venv/Scripts/python scripts/check_links.py
  ```

  plus `tests_circuit` in `.venv-circuit` if `dbeme/circuit/` changed, and
  `scripts/warm_check.py` on one example if the cascade or the dataset
  layer changed (a warm rerun must solve nothing). Report the results in
  the hand-off message; a red gate is not offered for merge.

## 5. Release procedure

Jae decides that a release happens and its number; an agent may prepare it.

1. On `main`, clean tree, gate of §4 green.
2. Decide the bump from `changelog/unreleased.md` with the table in §1.
3. `git mv changelog/unreleased.md changelog/vX.Y.Z.md`; fill its header
   (date, previous version, bump reason); create a fresh
   `changelog/unreleased.md` from the template; add the row to the index in
   `changelog/README.md`.
4. Set the version in `pyproject.toml` and `CITATION.cff`.
5. One commit, `Release X.Y.Z`, containing only steps 3–4.
6. `git tag -a vX.Y.Z -m "dbeme X.Y.Z"` on that commit.
7. `git push origin main` then `git push origin vX.Y.Z` — **Jae only, or an
   agent on his explicit instruction for this version**.
8. On GitHub: *Releases → Draft a new release*, choose the tag, paste
   `changelog/vX.Y.Z.md` as the body, attach any dataset kept out of git as
   `<dataset_name>.zip` (unzips to `datasets/<dataset_name>/`), publish.

A *GitHub Release* is the published page of a tag. It is not a branch:
`release/X.Y` branches exist only for §2's maintenance case.

## 6. GitHub settings (one-time, done by Jae in the web UI)

*Settings → Rules → Rulesets → New ruleset:*

| ruleset | target | rules |
|---|---|---|
| `protect-main` | branch `main` | Restrict deletions; Block force pushes |
| `immutable-releases` | tags matching `v*` | Restrict updates; Restrict deletions |

Leave *Require a pull request* off while there is one maintainer and no CI
— it would block his own release commit. Revisit when CI exists. If the
repository's *Settings → General* offers release immutability, enable it
too.

## 7. What an agent may and may not do

| action | agent |
|---|---|
| create a branch, commit on it, create local `pre_*` tags | yes |
| edit `changelog/unreleased.md` | yes, in the same commit as the change |
| prepare a release (§5 steps 2–5) | when asked |
| merge into `main`, commit on `main` | only when asked, for that merge |
| `git push` of anything; create or push a `v*` tag | only when asked, naming the ref |
| edit a released `changelog/vX.Y.Z.md` (other than *Errata*) | never |
| move or delete a `v*` tag; `push --force`; `rebase` or `filter-repo` of anything already pushed; `push --tags` | never |
