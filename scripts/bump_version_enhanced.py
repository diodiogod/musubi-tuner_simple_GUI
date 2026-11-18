#!/usr/bin/env python3
"""Bump this GUI's release metadata without changing the Musubi package or tags."""

import argparse
import ast
from datetime import date
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = ("gui_version.py", "README.md", "CHANGELOG.md")
SEMVER = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"


def version_values(source: str) -> tuple[str, str]:
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"__version__", "GUI_NAME"}:
                    values[target.id] = ast.literal_eval(node.value)
    current, name = values.get("__version__"), values.get("GUI_NAME")
    if not isinstance(current, str) or not re.fullmatch(SEMVER, current) or not isinstance(name, str):
        raise ValueError("gui_version.py must define a semantic __version__ and GUI_NAME.")
    return current, name


def next_version(current: str, requested: str) -> str:
    parts = [int(part) for part in current.split(".")]
    if requested in {"major", "minor", "patch"}:
        index = {"major": 0, "minor": 1, "patch": 2}[requested]
        parts[index] += 1
        parts[index + 1:] = [0] * (2 - index)
        return ".".join(map(str, parts))
    if not re.fullmatch(SEMVER, requested):
        raise ValueError("Choose patch, minor, major, or an explicit MAJOR.MINOR.PATCH version.")
    if tuple(map(int, requested.split("."))) <= tuple(map(int, current.split("."))):
        raise ValueError("The new GUI version must be newer than the current version.")
    return requested


def replace_once(pattern: str, replacement: str, source: str, label: str) -> str:
    updated, count = re.subn(pattern, lambda _: replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one {label}; found {count}. No release files were changed.")
    return updated


def release_updates(originals: dict[str, str], requested: str, notes: str, released_on: date):
    current, name = version_values(originals["gui_version.py"])
    new = next_version(current, requested)
    version_source = replace_once(
        rf'^__version__[ \t]*=[ \t]*[\"\']{re.escape(current)}[\"\'][ \t]*$',
        f'__version__ = "{new}"', originals["gui_version.py"], "GUI version assignment",
    )
    readme = replace_once(
        rf"^# {re.escape(name)} v{re.escape(current)}$",
        f"# {name} v{new}", originals["README.md"], "README GUI title",
    )
    readme = replace_once(
        rf"https://img\.shields\.io/badge/GUI_version-{re.escape(current)}-1f6feb\?style=for-the-badge",
        f"https://img.shields.io/badge/GUI_version-{new}-1f6feb?style=for-the-badge",
        readme, "README GUI version badge",
    )
    changelog = originals["CHANGELOG.md"]
    if re.search(rf"^## \[{re.escape(new)}\]", changelog, re.MULTILINE):
        raise ValueError("The changelog already contains that GUI version.")
    entry = f"## [{new}] - {released_on.isoformat()}\n\n{notes.strip()}\n\n"
    first_release = re.search(r"^## \[", changelog, re.MULTILINE)
    offset = first_release.start() if first_release else len(changelog.rstrip())
    changelog = changelog[:offset].rstrip() + "\n\n" + entry + changelog[offset:]
    return current, new, {"gui_version.py": version_source, "README.md": readme, "CHANGELOG.md": changelog}


def git(root: Path, *args: str):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8")


def require_clean_release_index(root: Path):
    if git(root, "diff", "--cached", "--name-only").stdout.strip():
        raise ValueError("Commit or unstage existing staged files before using --commit.")
    if git(root, "status", "--porcelain", "--", *RELEASE_FILES).stdout.strip():
        raise ValueError("Review and commit existing release-file edits before using --commit; or use --no-commit.")


def write_atomic(path: Path, contents: str):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="patch, minor, major, or an explicit semantic version")
    parser.add_argument("commit_description", help="Developer-facing description; newlines are preserved")
    parser.add_argument("changelog_description", help="User-facing Markdown release notes; use multiple lines")
    parser.add_argument("--dry-run", action="store_true", help="Preview the release without writing files")
    commit = parser.add_mutually_exclusive_group()
    commit.add_argument("--commit", action="store_true", help="Commit only the three release metadata files")
    commit.add_argument("--no-commit", action="store_true", help="Leave changes for review (the default)")
    parser.add_argument("--date", type=date.fromisoformat, help="Release date as YYYY-MM-DD; defaults to today")
    args = parser.parse_args(argv)
    try:
        if not args.commit_description.strip() or not args.changelog_description.strip():
            raise ValueError("Both descriptions must contain text.")
        original_bytes = {name: (root / name).read_bytes() for name in RELEASE_FILES}
        originals = {name: data.decode("utf-8").replace("\r\n", "\n") for name, data in original_bytes.items()}
        current, new, updates = release_updates(originals, args.version, args.changelog_description, args.date or date.today())
        print(f"GUI version: {current} -> {new}\nRelease files: {', '.join(RELEASE_FILES)}")
        if args.dry_run:
            print(f"Dry run; no files changed.\n\n{args.changelog_description.strip()}")
            return 0
        if args.commit:
            require_clean_release_index(root)
        written = []
        try:
            for name, contents in updates.items():
                write_atomic(root / name, contents)
                written.append(name)
        except OSError:
            for name in reversed(written):
                write_atomic(root / name, original_bytes[name].decode("utf-8"))
            raise
        if args.commit:
            git(root, "add", "--", *RELEASE_FILES)
            with tempfile.TemporaryDirectory(prefix="musubi-gui-release-") as directory:
                message = Path(directory) / "commit.txt"
                message.write_text(f"Bump GUI version to {new}\n\n{args.commit_description.strip()}\n", encoding="utf-8")
                print(git(root, "commit", "--file", str(message)).stdout.strip())
        else:
            print("Changes are ready for review; no commit was created.")
        print(f"GUI release tag, if publishing a release: gui-v{new}")
        return 0
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        parser.exit(1, f"Release failed: {detail}\n")


if __name__ == "__main__":
    raise SystemExit(main())
