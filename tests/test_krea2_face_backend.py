import unittest

from backends.krea2_face import build_command


class Krea2FaceBackendTests(unittest.TestCase):
    def test_command_uses_job_state_not_dataset_toml(self):
        settings = {
            "python_executable": "python", "krea2_dit_model": "raw.safetensors",
            "vae_model": "vae.safetensors", "krea2_text_encoder": "text.safetensors",
            "attention_mechanism": "sdpa", "fp8_scaled": True,
        }
        config = {
            "reference_dir": "refs", "face_model_dir": "faces", "steps": 30,
            "reference_manifest": "enabled_refs.json",
            "resolution": 512, "denoise_steps": 12, "draft_k": 1,
        }
        command = build_command(settings, config, "input.safetensors", "output.safetensors", "prompts.json")
        self.assertEqual(command[:2], ["python", "src/musubi_tuner/krea2_face_refinement.py"])
        self.assertNotIn("--dataset_config", command)
        self.assertEqual(command[command.index("--network_weights") + 1], "input.safetensors")
        self.assertEqual(command[command.index("--train_steps") + 1], "30")
        self.assertEqual(command[command.index("--reference_manifest") + 1], "enabled_refs.json")
        self.assertIn("--fp8_scaled", command)


if __name__ == "__main__":
    unittest.main()
