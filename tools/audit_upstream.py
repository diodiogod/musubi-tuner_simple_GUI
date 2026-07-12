"""Read-only report for reviewing a future Musubi upstream sync."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "UPSTREAM_BASELINE.json"


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def main() -> int:
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    remote_ref = f"{config['remote']}/main"
    try:
        remote_head = git("rev-parse", "--short", remote_ref)
        baseline = git("rev-parse", "--short", config["commit"])
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Run: git fetch {config['remote']} --prune", file=sys.stderr)
        return 2

    changed = git("diff", "--name-status", config["commit"], remote_ref, "--", config["vendored_path"])
    protected = tuple(config["protected_downstream_paths"])
    overlap = [line for line in changed.splitlines() if any(line.split("\t")[-1].startswith(path) for path in protected)]

    print(f"Recorded baseline: {config['tag']} ({baseline}), audited {config['audited_on']}")
    print(f"Fetched upstream:  {remote_ref} ({remote_head})")
    if baseline == remote_head:
        print("Upstream is still at the recorded baseline.")
        return 0
    print(f"\nVendored files changed upstream since baseline:\n{changed or '(none)'}")
    print("\nProtected downstream paths touched upstream:")
    print("\n".join(overlap) if overlap else "(none)")
    print("\nReview and import changes selectively; do not merge upstream wholesale.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
