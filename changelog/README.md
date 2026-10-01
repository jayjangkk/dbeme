# Changelog

One file per released version, plus `unreleased.md` for work merged or in
progress since the last release. Versioning rules and the release procedure
are in [`docs/git_workflow.md`](../docs/git_workflow.md).

| version | date | bump | headline |
|---|---|---|---|
| [1.0.0](v1.0.0.md) | 2026-10-01 | first release | first public release: three backends, two cascade routes, circuit layer, edge coupler (report 22), 63 literature datasets |

**Released files are frozen.** Once its tag `vX.Y.Z` is pushed, a version's
file is never edited, except to append under a final `## Errata` heading a
pointer to the version that fixes a problem.

## Entry template

Copy into `unreleased.md`; keep only the headings that have entries. Each
entry is one or two lines and names the files, test or report that carry
the detail.

```markdown
# dbeme X.Y.Z — YYYY-MM-DD

Previous: vA.B.C · Bump: MAJOR | MINOR | PATCH — <one-line reason>

## API changes            <!-- MAJOR: what was renamed/removed, and the replacement -->
## Datasets invalidated   <!-- MAJOR: which caches stop loading, why, how to rebuild -->
## Results changed        <!-- validated number: before -> after, cause, where measured -->
## Added
## Changed
## Fixed
## Removed
## Examples, reports, datasets
## Upgrade notes
```

*Results changed* is the heading that matters most in a numerical code:
every entry gives the quantity, its value before and after, and the test or
report section where both are measured. A change with no number attached
does not belong there.
