"""Read-only Turbo evaluation of a face/pose LoRA generation suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from musubi_tuner.face_refinement.face_reward import FaceSimilarityReward, bgr_to_rgb_tensor
from musubi_tuner.face_refinement.pose import estimate_pose


def evaluate_suite(suite_path, images_dir, reference_manifest, face_model_dir, output_path, baseline_path=None) -> dict:
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    reference_payload = json.loads(Path(reference_manifest).read_text(encoding="utf-8"))
    entries = [item for item in reference_payload.get("reference_images", []) if item.get("enabled", True)]
    paths = [str(item["path"]) for item in entries]
    buckets = {str(item["path"]): str(item.get("pose", "uncertain")) for item in entries}
    reward = FaceSimilarityReward(
        reference_images=paths, model_dir=face_model_dir, providers=["CPUExecutionProvider"],
        device="cpu", pose_buckets=buckets, pose_reward_weight=0.0,
        pose_min_references=int(suite.get("pose_min_references", 2)), expression_diversity_weight=0.0,
    )
    image_root = Path(images_dir)
    cases = []
    for case in suite["cases"]:
        explicit_image = Path(case.get("image", "")) if case.get("image") else None
        matches = [explicit_image] if explicit_image and explicit_image.is_file() else sorted(image_root.glob(f"*_{int(case['seed'])}.png"), key=lambda path: path.stat().st_mtime)
        if not matches:
            cases.append({**case, "detected": False, "error": "generated image not found"}); continue
        path = matches[-1]
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        faces = reward.detect_faces(bgr)
        if not faces:
            cases.append({**case, "image": str(path), "detected": False, "pose_success": False}); continue
        face = max(faces, key=reward._face_area)
        image = bgr_to_rgb_tensor(bgr)
        with torch.no_grad():
            embedding = reward.encode_faces(reward.crop_tensor(image, face))
            overall = float(reward.identity_scores(embedding).item())
            prototype = reward.pose_prototypes.get(case["pose"])
            pose_similarity = float((embedding @ prototype.t()).item()) if prototype is not None else None
        actual = estimate_pose(face.get("kps"), face.get("bbox"), face.get("score", 0.0))
        requested = case["pose"]
        cases.append({**case, "image": str(path), "detected": True, "overall_similarity": overall,
                      "pose_similarity": pose_similarity, "actual_pose": actual["bucket"],
                      "pose_confidence": actual["confidence"], "pose_success": actual["bucket"] == requested})
    summaries = {}
    for pose in sorted({case["pose"] for case in cases}):
        group = [case for case in cases if case["pose"] == pose]
        detected = [case for case in group if case.get("detected")]
        pose_values = [case["pose_similarity"] for case in detected if case.get("pose_similarity") is not None]
        summaries[pose] = {
            "samples": len(group), "detection_rate": len(detected) / len(group),
            "pose_success_rate": sum(bool(case.get("pose_success")) for case in group) / len(group),
            "overall_similarity": sum(case["overall_similarity"] for case in detected) / len(detected) if detected else None,
            "pose_similarity": sum(pose_values) / len(pose_values) if pose_values else None,
            "worst_overall_similarity": min((case["overall_similarity"] for case in detected), default=None),
        }
    result = {"renderer": "krea2_turbo", "suite": str(Path(suite_path)), "cases": cases, "poses": summaries}
    if baseline_path and Path(baseline_path).is_file():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8")); deltas = {}
        for pose, current in summaries.items():
            before = baseline.get("poses", {}).get(pose, {})
            deltas[pose] = {key: (current[key] - before[key] if current.get(key) is not None and before.get(key) is not None else None)
                            for key in ("overall_similarity", "pose_similarity", "pose_success_rate", "detection_rate")}
        result["baseline"] = str(Path(baseline_path)); result["deltas"] = deltas
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
