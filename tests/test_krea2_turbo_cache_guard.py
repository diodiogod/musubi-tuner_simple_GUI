import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


class Krea2TurboCacheGuardTests(unittest.TestCase):
    def test_rejects_ram_cache_when_turbo_copy_does_not_fit(self):
        trainer = Krea2NetworkTrainer()
        model = torch.nn.Linear(16, 16, bias=False)
        memory = type("Memory", (), {"available": 1})()
        with patch("psutil.virtual_memory", return_value=memory):
            self.assertFalse(trainer._turbo_cache_has_headroom(model))

    def test_cached_turbo_mode_restores_raw_from_checkpoint(self):
        trainer = Krea2NetworkTrainer()
        model = torch.nn.Linear(2, 2, bias=False)
        accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda value: value)
        args = SimpleNamespace(
            turbo_dit="turbo.safetensors",
            turbo_dit_cache=True,
            dit="raw.safetensors",
            fp8_scaled=True,
            projector_diff=None,
            projector_diff_strength=1.0,
        )
        raw_state = {"weight": torch.ones_like(model.weight)}
        trainer._turbo_stash = {"weight": torch.zeros_like(model.weight)}
        trainer._free_base_weights = MagicMock()
        trainer._assign_weights = MagicMock()

        with patch(
            "musubi_tuner.krea2_train_network.krea2_utils.load_krea2_dit_state_dict",
            return_value=raw_state,
        ) as load_weights:
            trainer.on_after_sample_images(accelerator, args, 1, 1, None, model, None, None, None)

        load_weights.assert_called_once()
        self.assertEqual(load_weights.call_args.args[0], "raw.safetensors")
        trainer._free_base_weights.assert_called_once_with(model)
        trainer._assign_weights.assert_called_once_with(model, raw_state)


if __name__ == "__main__":
    unittest.main()
