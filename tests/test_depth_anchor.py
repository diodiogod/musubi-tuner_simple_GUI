import unittest
from collections import OrderedDict

import torch

from musubi_tuner.perceptual.depth_anchor import DepthAnchor, reconstruct_clean_latents, resize_latents_for_depth_decode


class DepthAnchorTests(unittest.TestCase):
    def test_clean_latent_reconstruction_matches_flow_equation(self):
        clean = torch.randn(2, 4, 1, 3, 3)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([250.0, 750.0])
        t = (timesteps / 1000).view(2, 1, 1, 1, 1)
        noisy = (1 - t) * clean + t * noise
        velocity = noise - clean
        recovered = reconstruct_clean_latents(noisy, velocity, timesteps)
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_depth_loss_backpropagates_only_through_prediction(self):
        anchor = object.__new__(DepthAnchor)
        anchor._predict = lambda pixels: pixels.mean(dim=1, keepdim=True)
        predicted = torch.rand(1, 3, 8, 8, requires_grad=True)
        target = torch.rand(1, 3, 8, 8, requires_grad=True)
        loss = anchor.loss(predicted, target)
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertGreater(predicted.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_depth_decode_resize_limits_long_side_and_preserves_gradient(self):
        latents = torch.randn(1, 4, 1, 360, 182, requires_grad=True)
        resized = resize_latents_for_depth_decode(latents, 518, 8)
        self.assertEqual(resized.shape, (1, 4, 1, 65, 33))
        resized.sum().backward()
        self.assertIsNotNone(latents.grad)

    def test_depth_decode_resize_does_not_upscale_small_latents(self):
        latents = torch.randn(1, 4, 1, 32, 24)
        self.assertIs(resize_latents_for_depth_decode(latents, 518, 8), latents)

    def test_target_depth_cache_avoids_duplicate_perceptor_work(self):
        anchor = object.__new__(DepthAnchor)
        anchor.device = torch.device("cpu")
        anchor.target_cache = OrderedDict()
        anchor.target_cache_limit = 2
        calls = []
        anchor._predict = lambda pixels: calls.append(True) or pixels.mean(dim=1, keepdim=True)
        pixels = torch.rand(1, 3, 8, 8)
        first = anchor.target_depth(pixels, "sample")
        second = anchor.target_depth(None, "sample")
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
