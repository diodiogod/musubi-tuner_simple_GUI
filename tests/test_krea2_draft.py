import unittest

import torch

from musubi_tuner.face_refinement.draft import generate_differentiable


class _Config:
    patch = 2


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, img, context, t, pos, mask):
        return img * self.weight


class _ToyVAE(torch.nn.Module):
    z_dim = 4
    temperal_downsample = (True, True, True)

    @property
    def dtype(self):
        return torch.float32

    def decode_to_pixels(self, latent):
        return torch.sigmoid(latent[:, :3, 0])


class DraftTests(unittest.TestCase):
    def test_truncated_sampling_backpropagates_through_final_step(self):
        model = _ToyModel()
        vae = _ToyVAE()
        text = torch.zeros(1, 2, 1, 1)
        mask = torch.ones(1, 2, dtype=torch.bool)
        pixels = generate_differentiable(
            model, vae, text, mask, None, None,
            resolution=32, denoise_steps=4, draft_k=1,
            cfg_scale=1.0, seed=3, checkpoint_vae=True,
        )
        pixels.mean().backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(abs(model.weight.grad.item()), 0)


if __name__ == "__main__":
    unittest.main()
