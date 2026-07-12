"""Reference-face validation shared by the GUI and trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward, IMAGE_EXTS


def scan_reference_faces(reference_dir: str, model_dir: str) -> dict:
    root = Path(reference_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Reference folder does not exist: {root}")
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError("Reference folder contains no supported images")
    reward = FaceSimilarityReward(
        reference_images=images, model_dir=model_dir, reference_face_policy="largest",
        providers=["CPUExecutionProvider"], device="cpu",
    )
    valid = list(reward.valid_reference_images)
    skipped = [{"path": path, "faces": count} for path, count in reward.skipped_reference_images]
    similarities = (reward.reference_embeddings @ reward.reference_prototype.t()).squeeze(1).cpu().tolist()
    report = {
        "reference_dir": str(root),
        "images_scanned": len(images),
        "valid_faces": len(valid),
        "valid_images": valid,
        "skipped_images": skipped,
        "similarity_min": min(similarities),
        "similarity_mean": sum(similarities) / len(similarities),
        "similarity_max": max(similarities),
        "warnings": [],
    }
    if len(valid) < 3:
        report["warnings"].append("Fewer than three usable reference faces; identity refinement may overfit.")
    if report["similarity_min"] < 0.15:
        report["warnings"].append("Reference faces may contain mixed identities or severe outliers.")
    if skipped:
        report["warnings"].append(f"Skipped {len(skipped)} image(s) where no usable face was found.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate face-refinement reference images")
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--face_model_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = scan_reference_faces(args.reference_dir, args.face_model_dir)
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
