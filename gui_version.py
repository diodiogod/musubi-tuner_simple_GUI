"""Independent GUI release identity; never use the bundled trainer version here."""

import json
from pathlib import Path

__version__ = "0.1.0"
GUI_NAME = "Musubi Tuner Simple GUI"
GUI_TITLE = f"{GUI_NAME} v{__version__}"


def application_info(root: Path | None = None) -> dict:
    """Expose release identity and the recorded upstream provenance without ML imports."""
    root = root or Path(__file__).resolve().parent
    try:
        baseline = json.loads((root / "UPSTREAM_BASELINE.json").read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            baseline = {}
    except (OSError, ValueError):
        baseline = {}
    review = baseline.get("latest_selective_review") or {}
    return {
        "name": GUI_NAME,
        "version": __version__,
        "release_tag": f"gui-v{__version__}",
        "upstream": {
            "name": "Musubi Tuner",
            "repository": baseline.get("repository", "https://github.com/kohya-ss/musubi-tuner.git"),
            "baseline_commit": baseline.get("commit", ""),
            "latest_reviewed_commit": review.get("main", "") if isinstance(review, dict) else "",
        },
    }
