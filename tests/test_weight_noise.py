import unittest

import torch

from musubi_tuner.training.weight_noise import AdapterWeightNoise, WeightNoiseConfig


class WeightNoiseTests(unittest.TestCase):
    def test_relative_noise_skips_zero_initialized_tensor(self):
        parameter = torch.nn.Parameter(torch.zeros(8))
        metrics = AdapterWeightNoise(WeightNoiseConfig(sigma=0.0125)).apply([parameter])
        self.assertEqual(torch.count_nonzero(parameter), 0)
        self.assertEqual(metrics["regularization/weight_noise_norm"], 0.0)

    def test_absolute_noise_changes_trainable_weights_only(self):
        torch.manual_seed(1)
        trainable = torch.nn.Parameter(torch.ones(32))
        frozen = torch.nn.Parameter(torch.ones(32), requires_grad=False)
        AdapterWeightNoise(WeightNoiseConfig(sigma=0.1, mode="absolute")).apply([trainable, frozen])
        self.assertFalse(torch.equal(trainable, torch.ones_like(trainable)))
        self.assertTrue(torch.equal(frozen, torch.ones_like(frozen)))

    def test_bound_norm_preserves_tensor_norm(self):
        torch.manual_seed(2)
        parameter = torch.nn.Parameter(torch.randn(128))
        before = torch.linalg.vector_norm(parameter).item()
        AdapterWeightNoise(WeightNoiseConfig(sigma=0.1, bound_norm=True)).apply([parameter])
        after = torch.linalg.vector_norm(parameter).item()
        self.assertAlmostEqual(after, before, places=5)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            WeightNoiseConfig(sigma=-0.1)
        with self.assertRaises(ValueError):
            WeightNoiseConfig(sigma=0.1, mode="unknown")


if __name__ == "__main__":
    unittest.main()
