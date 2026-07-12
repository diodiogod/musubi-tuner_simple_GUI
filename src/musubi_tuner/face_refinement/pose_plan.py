"""Pose training-plan presets, prompt suggestions, and progress tracking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

TRAINABLE_POSES = (
    "frontal", "three_quarter_left", "three_quarter_right",
    "profile_left", "profile_right", "looking_up", "looking_down",
)

POSE_PHRASES = {
    "frontal": "front-facing portrait",
    "three_quarter_left": "three-quarter portrait, turned slightly left",
    "three_quarter_right": "three-quarter portrait, turned slightly right",
    "profile_left": "clear left side-profile portrait",
    "profile_right": "clear right side-profile portrait",
    "looking_up": "portrait looking slightly upward",
    "looking_down": "portrait looking slightly downward",
}

PRESET_SHARES = {
    "balanced_identity": {pose: 1 for pose in TRAINABLE_POSES},
    "improve_profiles": {"frontal": 5, "three_quarter_left": 15, "three_quarter_right": 15, "profile_left": 30, "profile_right": 30, "looking_up": 2.5, "looking_down": 2.5},
    "improve_three_quarter": {"frontal": 10, "three_quarter_left": 35, "three_quarter_right": 35, "profile_left": 7.5, "profile_right": 7.5, "looking_up": 2.5, "looking_down": 2.5},
}


def suggest_prompts(pose: str, trigger: str = "{trigger}", variations=None) -> list[str]:
    if pose not in TRAINABLE_POSES:
        return []
    choices = set(variations or ("natural", "studio", "cinematic", "expression"))
    base = f"[{pose}] {POSE_PHRASES[pose]} of {trigger}"
    suffixes = []
    if "natural" in choices: suffixes.append("natural daylight, realistic skin texture")
    if "studio" in choices: suffixes.append("neutral studio background, soft balanced lighting")
    if "cinematic" in choices: suffixes.append("cinematic lighting, detailed photograph")
    if "expression" in choices: suffixes.append("natural expression, candid photograph")
    return [f"{base}, {suffix}" for suffix in suffixes] or [base]


def default_pose_plan(preset: str = "balanced_identity") -> dict:
    plan = {
        "enabled": False, "preset": preset, "overall_anchor_weight": 0.80,
        "variations": ["natural", "studio", "cinematic", "expression"], "buckets": {},
    }
    apply_preset(plan, preset)
    return plan


def apply_preset(plan: dict, preset: str) -> dict:
    shares = PRESET_SHARES.get(preset, PRESET_SHARES["balanced_identity"])
    buckets = plan.setdefault("buckets", {})
    total = sum(shares.values())
    for pose in TRAINABLE_POSES:
        current = buckets.setdefault(pose, {})
        current.update({
            "enabled": shares.get(pose, 0) > 0,
            "share": round(100.0 * shares.get(pose, 0) / total, 3),
            "target": current.get("target", 0.55), "patience": current.get("patience", 2),
            "plateau_patience": current.get("plateau_patience", 4),
            "min_evaluations": current.get("min_evaluations", 2),
            "prompts": current.get("prompts", suggest_prompts(pose)),
        })
    plan["preset"] = preset
    return plan


def normalize_pose_plan(plan: dict, reference_counts: dict | None = None, min_references: int = 2) -> tuple[dict, list[str]]:
    normalized = deepcopy(plan)
    warnings = []
    buckets = normalized.setdefault("buckets", {})
    active = []
    for pose in TRAINABLE_POSES:
        cfg = buckets.setdefault(pose, {})
        count = int((reference_counts or {}).get(pose, min_references))
        prompts = [str(item).strip() for item in cfg.get("prompts", []) if str(item).strip()]
        enabled = bool(cfg.get("enabled", False)) and bool(prompts)
        if enabled and count < min_references:
            enabled = False
            warnings.append(f"{pose} has {count} usable reference(s); its sampling share was disabled.")
        cfg["enabled"] = enabled
        cfg["prompts"] = prompts
        cfg["share"] = max(0.0, float(cfg.get("share", 0.0)))
        if enabled and cfg["share"] > 0: active.append((pose, cfg))
    total = sum(cfg["share"] for _, cfg in active)
    if normalized.get("enabled") and total <= 0:
        raise ValueError("The Pose Training Plan needs at least one enabled pose with prompts and a positive share.")
    if total > 0:
        for _, cfg in active: cfg["share"] = 100.0 * cfg["share"] / total
    return normalized, warnings


def weighted_prompt_records(plan: dict) -> list[dict]:
    records = []
    for pose, cfg in plan.get("buckets", {}).items():
        if not cfg.get("enabled") or float(cfg.get("share", 0)) <= 0: continue
        prompts = cfg.get("prompts") or []
        per_prompt = float(cfg["share"]) / len(prompts)
        records.extend({"pose": pose, "prompt": prompt, "weight": per_prompt} for prompt in prompts)
    total = sum(record["weight"] for record in records)
    if total > 0:
        for record in records: record["weight"] = 100.0 * record["weight"] / total
    return records


@dataclass
class PoseProgressTracker:
    plan: dict
    histories: dict[str, list[float]] = field(default_factory=dict)
    target_streaks: dict[str, int] = field(default_factory=dict)

    def update(self, pose: str | None, similarity: float) -> None:
        if not pose or pose not in self.plan.get("buckets", {}): return
        cfg = self.plan["buckets"][pose]
        if not cfg.get("enabled"): return
        history = self.histories.setdefault(pose, []); history.append(float(similarity))
        target = float(cfg.get("target", 0.55))
        self.target_streaks[pose] = self.target_streaks.get(pose, 0) + 1 if similarity >= target else 0

    def pose_status(self, pose: str) -> dict:
        cfg = self.plan["buckets"][pose]; history = self.histories.get(pose, [])
        minimum = int(cfg.get("min_evaluations", 3)); patience = int(cfg.get("patience", 5))
        plateau_patience = int(cfg.get("plateau_patience", 8))
        reached = len(history) >= minimum and self.target_streaks.get(pose, 0) >= patience
        plateau = False
        if len(history) >= max(minimum, plateau_patience + 1):
            recent = history[-plateau_patience:]
            previous_best = max(history[:-plateau_patience])
            plateau = max(recent) <= previous_best + 0.005
        return {"evaluations": len(history), "latest": history[-1] if history else None, "reached": reached, "plateau": plateau}

    def stop_reason(self) -> str | None:
        enabled = [pose for pose, cfg in self.plan.get("buckets", {}).items() if cfg.get("enabled")]
        if not enabled: return None
        statuses = [self.pose_status(pose) for pose in enabled]
        if all(item["reached"] for item in statuses): return "all_pose_targets_reached"
        if all(item["reached"] or item["plateau"] for item in statuses): return "pose_targets_reached_or_plateaued"
        return None
