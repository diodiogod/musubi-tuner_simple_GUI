"""Release isolation, consistent metadata, and opt-in commit behavior."""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from gui_version import __version__, application_info

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bump_gui_version", ROOT / "scripts/bump_version_enhanced.py")
bump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bump)
NOTES = "### Fixed\n\n- Fix the comparison divider.\n\n### Changed\n\n- Improve café caption previews."
DESCRIPTION = "Refine comparison controls\n\nPreserve the locked divider and caption text."


@pytest.fixture
def release_root(tmp_path):
    for name in bump.RELEASE_FILES:
        shutil.copyfile(ROOT / name, tmp_path / name)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "8.9.10"\n', encoding="utf-8")
    (tmp_path / "UPSTREAM_BASELINE.json").write_text('{"commit":"unchanged-upstream"}', encoding="utf-8")
    return tmp_path


def snapshot(root):
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


@pytest.mark.parametrize("kind,expected", [
    ("patch", "2.4.10"), ("minor", "2.5.0"), ("major", "3.0.0"), ("2.10.0", "2.10.0"),
])
def test_numeric_version_order_and_resets(kind, expected):
    assert bump.next_version("2.4.9", kind) == expected


def test_bump_updates_gui_badge_title_and_notes_without_touching_trainer(release_root):
    before = snapshot(release_root)
    expected = bump.next_version(__version__, "patch")
    assert bump.main(["patch", DESCRIPTION, NOTES, "--date", "2026-11-12"], root=release_root) == 0
    after = snapshot(release_root)
    assert {name for name in before if before[name] != after[name]} == set(bump.RELEASE_FILES)
    assert bump.version_values(after["gui_version.py"].decode())[0] == expected
    readme = after["README.md"].decode()
    assert f"# Musubi Tuner Simple GUI v{expected}" in readme
    assert f"/GUI_version-{expected}-" in readme
    changelog = after["CHANGELOG.md"].decode()
    assert f"## [{expected}] - 2026-11-12\n\n{NOTES}" in changelog
    assert changelog.index(f"## [{expected}]") < changelog.index(f"## [{__version__}]")
    assert (release_root / "pyproject.toml").read_bytes() == before["pyproject.toml"]
    assert (release_root / "UPSTREAM_BASELINE.json").read_bytes() == before["UPSTREAM_BASELINE.json"]


def test_dry_run_leaves_every_file_unchanged(release_root):
    before = snapshot(release_root)
    assert bump.main(["minor", DESCRIPTION, NOTES, "--dry-run"], root=release_root) == 0
    assert snapshot(release_root) == before


@pytest.mark.parametrize("target", ["0.0.0", "0.1", "v1.0.0", "01.2.3"])
def test_invalid_or_older_release_never_writes(release_root, target):
    before = snapshot(release_root)
    with pytest.raises(SystemExit) as failure:
        bump.main([target, DESCRIPTION, NOTES], root=release_root)
    assert failure.value.code == 1
    assert snapshot(release_root) == before


def test_missing_readme_marker_fails_before_changing_any_file(release_root):
    readme = release_root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("/GUI_version-", "/renamed-badge-"), encoding="utf-8")
    before = snapshot(release_root)
    with pytest.raises(SystemExit):
        bump.main(["patch", DESCRIPTION, NOTES], root=release_root)
    assert snapshot(release_root) == before


def test_write_failure_rolls_back_already_written_release_files(release_root, monkeypatch):
    version_path = release_root / "gui_version.py"
    version_path.write_bytes(version_path.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    before = snapshot(release_root)
    original_write = bump.write_atomic
    def fail_readme(path, text):
        if path.name == "README.md":
            raise OSError("Simulated write failure")
        original_write(path, text)
    monkeypatch.setattr(bump, "write_atomic", fail_readme)
    with pytest.raises(SystemExit):
        bump.main(["patch", DESCRIPTION, NOTES], root=release_root)
    assert snapshot(release_root) == before


def init_git(root):
    if not shutil.which("git"):
        pytest.skip("Git is required to verify opt-in release commits")
    bump.git(root, "init")
    bump.git(root, "config", "user.name", "Release tests")
    bump.git(root, "config", "user.email", "release-tests@example.invalid")
    bump.git(root, "config", "commit.gpgsign", "false")
    bump.git(root, "add", ".")
    bump.git(root, "commit", "-m", "Test release baseline")


def test_opt_in_commit_contains_only_release_files_and_multiline_message(release_root):
    init_git(release_root)
    (release_root / "unrelated.txt").write_text("Keep this user's work", encoding="utf-8")
    assert bump.main(["patch", DESCRIPTION, NOTES, "--commit"], root=release_root) == 0
    changed = bump.git(release_root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").stdout.splitlines()
    assert set(changed) == set(bump.RELEASE_FILES)
    message = bump.git(release_root, "log", "-1", "--format=%B").stdout
    assert f"Bump GUI version to {bump.next_version(__version__, 'patch')}\n\n{DESCRIPTION}" in message
    assert bump.git(release_root, "ls-files", "unrelated.txt").stdout == ""


@pytest.mark.parametrize("staged", [False, True])
def test_commit_refuses_existing_release_edits_or_unrelated_staged_files(release_root, staged):
    init_git(release_root)
    path = release_root / ("unrelated.txt" if staged else "README.md")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nKeep this user's work\n")
    if staged:
        bump.git(release_root, "add", "unrelated.txt")
    before = snapshot(release_root)
    head = bump.git(release_root, "rev-parse", "HEAD").stdout
    with pytest.raises(SystemExit):
        bump.main(["patch", DESCRIPTION, NOTES, "--commit"], root=release_root)
    assert snapshot(release_root) == before
    assert bump.git(release_root, "rev-parse", "HEAD").stdout == head


def test_release_identity_and_upstream_provenance_are_independent(tmp_path):
    (tmp_path / "UPSTREAM_BASELINE.json").write_text(json.dumps({
        "commit": "baseline", "latest_selective_review": {"main": "reviewed"},
    }), encoding="utf-8")
    info = application_info(tmp_path)
    assert info["version"] == __version__
    assert info["release_tag"] == f"gui-v{__version__}"
    assert info["upstream"]["baseline_commit"] == "baseline"
    assert info["upstream"]["latest_reviewed_commit"] == "reviewed"
    assert application_info(tmp_path / "missing")["upstream"]["baseline_commit"] == ""
