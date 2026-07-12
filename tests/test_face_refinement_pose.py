import unittest

import numpy as np

from musubi_tuner.face_refinement.pose import estimate_pose, parse_pose_prompt


class FaceRefinementPoseTests(unittest.TestCase):
    def setUp(self):
        self.frontal = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)

    def test_broad_pose_buckets_from_landmarks(self):
        self.assertEqual(estimate_pose(self.frontal)["bucket"], "frontal")
        three_quarter = self.frontal.copy(); three_quarter[2, 0] = 65
        self.assertEqual(estimate_pose(three_quarter)["bucket"], "three_quarter_right")
        profile = self.frontal.copy(); profile[2, 0] = 80
        self.assertEqual(estimate_pose(profile)["bucket"], "profile_right")
        looking_up = self.frontal.copy(); looking_up[2, 1] = 50
        self.assertEqual(estimate_pose(looking_up)["bucket"], "looking_up")

    def test_invalid_landmarks_are_uncertain(self):
        result = estimate_pose(np.zeros((3, 2)))
        self.assertEqual(result["bucket"], "uncertain")
        self.assertEqual(result["confidence"], 0.0)

    def test_pose_prompt_tags_are_removed_for_text_encoder(self):
        self.assertEqual(parse_pose_prompt("[profile_left] portrait of subject"), ("profile_left", "portrait of subject"))
        self.assertEqual(parse_pose_prompt("[auto] candid photo"), ("auto", "candid photo"))
        self.assertEqual(parse_pose_prompt("ordinary portrait"), (None, "ordinary portrait"))
        with self.assertRaisesRegex(ValueError, "Unknown pose tag"):
            parse_pose_prompt("[diagonal] portrait")


if __name__ == "__main__":
    unittest.main()
