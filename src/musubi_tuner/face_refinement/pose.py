"""Landmark-only head-pose bucketing for optional pose-aware refinement.

This intentionally estimates broad buckets, not biometric-grade 3D pose. Low
confidence results are routed to manual review.
"""

from __future__ import annotations

import math
import re

import numpy as np

POSE_BUCKETS = (
    "frontal", "three_quarter_left", "three_quarter_right",
    "profile_left", "profile_right", "looking_up", "looking_down", "uncertain",
)
POSE_LABELS = {
    "frontal": "Frontal", "three_quarter_left": "Three-quarter left",
    "three_quarter_right": "Three-quarter right", "profile_left": "Profile left",
    "profile_right": "Profile right", "looking_up": "Looking up",
    "looking_down": "Looking down", "uncertain": "Uncertain",
}
TAG_PATTERN = re.compile(r"^\s*\[([a-z_]+)\]\s*", re.IGNORECASE)


def estimate_pose(kps, bbox=None, detection_score: float = 1.0) -> dict:
    points = np.asarray(kps, dtype=np.float32) if kps is not None else np.empty((0, 2), dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return {"bucket": "uncertain", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    face_height = float(np.linalg.norm(mouth_mid - eye_mid))
    if eye_distance < 2.0 or face_height < 2.0:
        return {"bucket": "uncertain", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}

    symmetry_x = float((eye_mid[0] + mouth_mid[0]) * 0.5)
    yaw = float(np.clip((nose[0] - symmetry_x) / eye_distance * 105.0, -90.0, 90.0))
    vertical_ratio = float((nose[1] - eye_mid[1]) / max(face_height, 1e-6))
    pitch = float(np.clip((0.50 - vertical_ratio) * 90.0, -45.0, 45.0))
    roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))

    mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
    geometry = min(1.0, eye_distance / max(face_height * 0.55, 1e-6))
    geometry *= min(1.0, mouth_width / max(eye_distance * 0.45, 1e-6))
    confidence = float(np.clip(float(detection_score) * (0.45 + 0.55 * geometry), 0.0, 1.0))
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float32)
        if box.shape[0] >= 4 and min(box[2] - box[0], box[3] - box[1]) < 48:
            confidence *= 0.75

    if confidence < 0.45:
        bucket = "uncertain"
    elif pitch >= 20.0:
        bucket = "looking_up"
    elif pitch <= -20.0:
        bucket = "looking_down"
    elif abs(yaw) < 15.0:
        bucket = "frontal"
    elif abs(yaw) < 52.0:
        bucket = "three_quarter_right" if yaw > 0 else "three_quarter_left"
    else:
        bucket = "profile_right" if yaw > 0 else "profile_left"
    return {"bucket": bucket, "yaw": yaw, "pitch": pitch, "roll": roll, "confidence": confidence}


def parse_pose_prompt(prompt: str) -> tuple[str | None, str]:
    match = TAG_PATTERN.match(prompt)
    if not match:
        return None, prompt.strip()
    tag = match.group(1).lower()
    if tag == "auto":
        return "auto", prompt[match.end():].strip()
    if tag not in POSE_BUCKETS:
        raise ValueError(f"Unknown pose tag [{tag}]")
    return tag, prompt[match.end():].strip()
