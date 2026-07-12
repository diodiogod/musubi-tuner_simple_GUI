import tempfile
import unittest
import json
from pathlib import Path

from backends.krea2_face_eval import prepare
from musubi_tuner.face_refinement.pose_plan import default_pose_plan


class Krea2FaceEvaluationBackendTests(unittest.TestCase):
    def test_prepares_fixed_turbo_suite_without_raw_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("turbo.safetensors", "input.safetensors"):
                (root / name).touch()
            plan = default_pose_plan("improve_profiles"); plan["enabled"] = True
            for pose, cfg in plan["buckets"].items():
                cfg["enabled"] = pose in ("profile_left", "profile_right")
            settings = {
                "python_executable": "python", "krea2_turbo_dit": str(root / "turbo.safetensors"),
                "krea2_dit_model": "raw.safetensors", "vae_model": "vae.safetensors",
                "krea2_text_encoder": "text.safetensors", "output_dir": str(root),
                "output_name": "generic-run", "attention_mechanism": "sdpa",
            }
            config = {
                "pose_aware": True, "pose_plan": plan, "trigger_word": "subject_token",
                "face_model_dir": "face-models", "pose_min_references": 2,
                "evaluation_prompts_per_pose": 1, "evaluation_seeds_per_prompt": 2,
                "evaluation_seed": 1000, "evaluation_resolution": 512, "evaluation_steps": 8,
                "preflight_report": {"scored_images": [
                    {"path": "left.jpg", "bucket": "profile_left"},
                    {"path": "right.jpg", "bucket": "profile_right"},
                ]},
            }
            result = prepare(settings, config, root / "input.safetensors")
            generate, evaluate = result["commands"]
            self.assertIn("--turbo", generate)
            self.assertEqual(generate[generate.index("--dit") + 1], str(root / "turbo.safetensors"))
            self.assertEqual(result["cases"], 4)
            self.assertIn("krea2_face_evaluate.py", evaluate[1])
            self.assertNotIn("krea2_face_refinement.py", " ".join(generate + evaluate))

            baseline_suite = root / "baseline-suite.json"
            baseline_suite_payload = json.loads((result["run_dir"] / "suite.json").read_text(encoding="utf-8"))
            baseline_suite_payload["renderer_settings"]["resolution"] = 640
            baseline_suite_payload["cases"] = [{"pose": "profile_left", "prompt": "fixed prompt", "seed": 77}]
            baseline_suite.write_text(json.dumps(baseline_suite_payload), encoding="utf-8")
            baseline_result = root / "baseline-result.json"
            baseline_result.write_text(json.dumps({"suite": str(baseline_suite), "poses": {}}), encoding="utf-8")
            compared = prepare(settings, config, root / "input.safetensors", baseline_result=baseline_result, label="compare")
            compared_suite = json.loads((compared["run_dir"] / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(compared_suite["cases"][0]["seed"], 77)
            self.assertEqual(compared_suite["renderer_settings"]["resolution"], 640)


if __name__ == "__main__":
    unittest.main()
