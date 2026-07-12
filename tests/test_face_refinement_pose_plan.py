import unittest

from musubi_tuner.face_refinement.pose_plan import (
    PoseProgressTracker, apply_preset, default_pose_plan, normalize_pose_plan,
    suggest_prompts, weighted_prompt_records,
)


class FaceRefinementPosePlanTests(unittest.TestCase):
    def test_profile_preset_weights_profiles_most(self):
        plan = apply_preset(default_pose_plan(), "improve_profiles")
        self.assertGreater(plan["buckets"]["profile_left"]["share"], plan["buckets"]["frontal"]["share"])

    def test_sparse_buckets_are_disabled_and_shares_normalized(self):
        plan = apply_preset(default_pose_plan(), "improve_profiles"); plan["enabled"] = True
        normalized, warnings = normalize_pose_plan(plan, {pose: (1 if pose == "profile_left" else 3) for pose in plan["buckets"]})
        self.assertFalse(normalized["buckets"]["profile_left"]["enabled"])
        self.assertAlmostEqual(sum(cfg["share"] for cfg in normalized["buckets"].values() if cfg["enabled"]), 100.0)
        self.assertTrue(warnings)

    def test_suggestions_and_weighted_records_include_internal_tags(self):
        self.assertTrue(all(prompt.startswith("[profile_right]") for prompt in suggest_prompts("profile_right")))
        plan = default_pose_plan(); plan["enabled"] = True
        records = weighted_prompt_records(plan)
        self.assertTrue(records)
        self.assertAlmostEqual(sum(record["weight"] for record in records), 100.0)

    def test_per_pose_targets_and_plateau_stop(self):
        plan = default_pose_plan(); plan["buckets"] = {"profile_left": {"enabled": True, "target": 0.5, "patience": 2, "plateau_patience": 3, "min_evaluations": 2}}
        tracker = PoseProgressTracker(plan)
        tracker.update("profile_left", 0.51); tracker.update("profile_left", 0.52)
        self.assertEqual(tracker.stop_reason(), "all_pose_targets_reached")


if __name__ == "__main__":
    unittest.main()
