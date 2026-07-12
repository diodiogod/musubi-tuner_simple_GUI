import unittest

from backends.krea2 import build_commands


class Krea2BackendRegularizationTests(unittest.TestCase):
    def _settings(self):
        return {
            "mixed_precision": "bf16", "output_dir": ".", "output_name": "test",
            "network_dim_low": "32", "network_alpha_low": "32", "network_type": "LoRA",
        }

    def test_disabled_defaults_remain_explicit_noops(self):
        settings = self._settings() | {
            "krea2_weight_noise_sigma": "0", "krea2_weight_noise_mode": "relative",
            "krea2_depth_anchor_weight": "0", "krea2_depth_anchor_grad_checkpoint": True,
        }
        command = build_commands(settings)[0]
        self.assertEqual(command[command.index("--weight_noise_sigma") + 1], "0")
        self.assertEqual(command[command.index("--depth_anchor_weight") + 1], "0")
        self.assertNotIn("--no-depth_anchor_grad_checkpoint", command)

    def test_enabled_options_are_forwarded(self):
        settings = self._settings() | {
            "krea2_weight_noise_sigma": "0.0125", "krea2_weight_noise_mode": "relative",
            "krea2_weight_noise_bound_norm": True, "krea2_depth_anchor_weight": "0.01",
            "krea2_depth_anchor_model": "depth-model", "krea2_depth_anchor_input_size": "518",
            "krea2_depth_anchor_gradient_weight": "0.5", "krea2_depth_anchor_grad_checkpoint": False,
        }
        command = build_commands(settings)[0]
        self.assertIn("--weight_noise_bound_norm", command)
        self.assertIn("--no-depth_anchor_grad_checkpoint", command)
        self.assertEqual(command[command.index("--depth_anchor_model") + 1], "depth-model")


if __name__ == "__main__":
    unittest.main()
