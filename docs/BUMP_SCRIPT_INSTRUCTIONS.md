# GUI release and version bump instructions

**A request to bump this project's version means the GUI release version.**
It does not mean changing upstream Musubi's Python package version.

## Version sources

- `gui_version.py`: canonical GUI version, shared by Classic GUI and Musubi Studio.
- `README.md`: GUI version badge and title, updated by the script.
- `CHANGELOG.md`: user-facing GUI and downstream-feature release notes.
- `pyproject.toml`: bundled `musubi-tuner` package metadata; never changed by a GUI bump.
- `UPSTREAM_BASELINE.json`: imported upstream baseline and selective review history;
  updated only as part of actual upstream reviews/imports.

Independent GUI numbering starts at **0.1.0**. This is a release identity change,
not a restart of the project's functionality. Previous inherited package bumps
remain in Git history.

## Commands

The interface follows TagGUI Video and TTS Audio Suite: the first description is
for developers, and the second is user-facing Markdown for the changelog. Summarize
all relevant changes since the previous GUI release, not just the last commit.

```powershell
venv\Scripts\python.exe scripts\bump_version_enhanced.py patch "<commit description>" "<multiline changelog description>" --dry-run
venv\Scripts\python.exe scripts\bump_version_enhanced.py patch "<commit description>" "<multiline changelog description>" --no-commit
```

```bash
venv/bin/python scripts/bump_version_enhanced.py minor "<commit description>" "<multiline changelog description>" --no-commit
```

Use `patch` for fixes and small refinements, `minor` for features, `major` for
breaking changes, or an explicit newer `MAJOR.MINOR.PATCH` value. The script reads
the current GUI version automatically; do not pre-check or manually update it.

Example with real newlines and explicit changelog sections:

```powershell
$releaseCommit = @'
Refine sample comparison interactions

- Preserve locked divider position during dragging.
- Keep the image clear when the pointer leaves the handle.
'@
$releaseNotes = @'
### Fixed

- Fix repositioning a locked Wipe divider.

### Changed

- Show the central comparison handle only near the pointer or during dragging.
'@
venv\Scripts\python.exe scripts\bump_version_enhanced.py patch $releaseCommit $releaseNotes --no-commit
```

The script preserves Markdown sections and multiline text exactly; it does not
guess categories from words in the description. Use `--date YYYY-MM-DD` when an
explicit release date is needed.

## Review, commit, and publish

- The default is **no commit**, matching this repo's review policy. `--dry-run`
  changes nothing. After user confirmation, `--commit` stages and commits only
  the three release files; it refuses existing staged changes or dirty release
  files. Commit implementation changes separately first, or review the release
  changes and commit them together manually.
- The script never pushes, creates a tag, or publishes a GitHub release.
- When explicitly publishing a GUI tag/release, use **`gui-vX.Y.Z`**, for example
  `gui-v0.1.0`. Existing `vX.Y.Z` tags belong to inherited upstream history and
  must not be moved or reused for GUI releases.
- Keep credit to [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner).
- Never add generated-by or co-author footers.
