"""Reference-face validation shared by the GUI and trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward, IMAGE_EXTS
from musubi_tuner.face_refinement.pose import estimate_pose


def scan_reference_faces(reference_dir: str, model_dir: str, progress=None) -> dict:
    root = Path(reference_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Reference folder does not exist: {root}")
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError("Reference folder contains no supported images")
    reward = FaceSimilarityReward(
        reference_images=images, model_dir=model_dir, reference_face_policy="largest",
        providers=["CPUExecutionProvider"], device="cpu", reference_progress=progress,
    )
    valid = list(reward.valid_reference_images)
    skipped = [{"path": path, "faces": count} for path, count in reward.skipped_reference_images]
    similarities = (reward.reference_embeddings @ reward.reference_prototype.t()).squeeze(1).cpu().tolist()
    poses = [estimate_pose(face.get("kps"), face.get("bbox"), face.get("score", 0.0)) for face in reward.valid_reference_faces]
    scored = sorted(
        ({"path": path, "similarity": float(score), **pose} for path, score, pose in zip(valid, similarities, poses)),
        key=lambda item: item["similarity"],
    )
    similarity_mean = sum(similarities) / len(similarities)
    outlier_threshold = min(0.35, max(0.15, similarity_mean - 0.30))
    for item in scored:
        item["outlier"] = item["similarity"] < outlier_threshold
    bucket_counts = {}
    for item in scored:
        bucket_counts[item["bucket"]] = bucket_counts.get(item["bucket"], 0) + 1
    if skipped:
        bucket_counts["no_face"] = len(skipped)
    report = {
        "reference_dir": str(root),
        "images_scanned": len(images),
        "valid_faces": len(valid),
        "valid_images": valid,
        "scored_images": scored,
        "skipped_images": skipped,
        "similarity_min": min(similarities),
        "similarity_mean": similarity_mean,
        "similarity_max": max(similarities),
        "outlier_threshold": outlier_threshold,
        "pose_bucket_counts": bucket_counts,
        "warnings": [],
    }
    if len(valid) < 3:
        report["warnings"].append("Fewer than three usable reference faces; identity refinement may overfit.")
    if report["similarity_min"] < 0.15:
        report["warnings"].append("Reference faces may contain mixed identities or severe outliers.")
    if skipped:
        report["warnings"].append(
            f"Skipped {len(skipped)} image(s): no reliable full-face identity embedding could be created. "
            "They will not be used by face refinement, but may still be useful for normal LoRA training."
        )
    uncertain = bucket_counts.get("uncertain", 0)
    if uncertain:
        report["warnings"].append(f"Pose estimate was uncertain for {uncertain} detected face(s); review them manually before pose-aware training.")
    usable_pose_counts = {key: value for key, value in bucket_counts.items() if key not in ("uncertain", "no_face")}
    if usable_pose_counts and max(usable_pose_counts.values()) / max(1, sum(usable_pose_counts.values())) > 0.75:
        report["warnings"].append("One pose bucket contains over 75% of usable references; pose-aware rewards may be imbalanced.")
    for bucket, count in usable_pose_counts.items():
        if count < 2:
            report["warnings"].append(f"Pose bucket '{bucket}' has only {count} reference and will fall back to overall identity unless corrected.")
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
