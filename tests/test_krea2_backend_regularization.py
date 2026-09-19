import unittest

from backends.krea2 import build_commands
from musubi_tuner.krea2_train_network import krea2_setup_parser


class Krea2BackendRegularizationTests(unittest.TestCase):
    def test_turbo_lora_and_target_preset_are_forwarded(self):
        settings = {
            "mixed_precision": "bf16", "krea2_dit_model": "raw.safetensors", "vae_model": "vae.safetensors",
            "dataset_config": "dataset.toml", "krea2_text_encoder": "text.safetensors",
            "krea2_turbo_lora": "turbo.safetensors", "krea2_turbo_lora_multiplier": "0.9",
            "krea2_lora_target_preset": "Skip text fusion (experimental)",
            "network_dim_low": "32", "network_alpha_low": "32", "network_type": "LoRA",
            "output_dir": "out", "output_name": "run",
        }
        command = build_commands(settings | {"_preview_only": True})[0]
        self.assertIn("--turbo_lora", command)
        self.assertIn("turbo.safetensors", command)
        self.assertIn("--network_args", command)
        network_arg = command[command.index("--network_args") + 1]
        self.assertIn("txtfusion", network_arg)
        self.assertIn("txtmlp", network_arg)

    def test_attention_only_preset_excludes_main_mlp(self):
        settings = {
            "krea2_dit_model": "raw.safetensors", "vae_model": "vae.safetensors", "dataset_config": "dataset.toml",
            "krea2_lora_target_preset": "Attention only (long-run safe)",
            "network_dim_low": "32", "network_alpha_low": "32", "network_type": "LoRA",
            "output_dir": "out", "output_name": "run", "_preview_only": True,
        }
        command = build_commands(settings)[0]
        network_arg = command[command.index("--network_args") + 1]
        self.assertIn("mlp", network_arg)
        self.assertIn("txtfusion", network_arg)
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
            "krea2_keep_depth_helpers_on_gpu": True,
            "krea2_depth_vae_device": "secondary",
        }
        command = build_commands(settings)[0]
        self.assertIn("--weight_noise_bound_norm", command)
        self.assertIn("--no-depth_anchor_grad_checkpoint", command)
        self.assertIn("--keep_depth_helpers_on_gpu", command)
        self.assertEqual(command[command.index("--depth_anchor_vae_device") + 1], "secondary")
        self.assertEqual(command[command.index("--depth_anchor_model") + 1], "depth-model")

    def test_safe_default_offloads_depth_helpers(self):
        command = build_commands(self._settings() | {"krea2_depth_anchor_weight": "0.01"})[0]
        self.assertNotIn("--keep_depth_helpers_on_gpu", command)
        self.assertEqual(command[command.index("--depth_anchor_vae_device") + 1], "training")

    def test_krea_depth_parser_defaults_to_training_device(self):
        import argparse

        parser = krea2_setup_parser(argparse.ArgumentParser())
        self.assertEqual(parser.get_default("depth_anchor_vae_device"), "training")


if __name__ == "__main__":
    unittest.main()
