from types import SimpleNamespace
import torch

from musubi_tuner.moss_image.regularization import MossDepthRegularizer


def test_packed_depth_decode_preserves_gradient_and_range():
    helper = object.__new__(MossDepthRegularizer)
    helper.args = SimpleNamespace(depth_anchor_input_size=56)
    helper.vae_device = helper.training_device = torch.device("cpu")

    class FakeVAE:
        def decode(self, latents):
            assert latents.shape == (1, 128, 4, 2)
            assert latents.dtype == torch.float32
            return latents[:, :3]

    latents = torch.zeros(1, 128, 1, 8, 4, requires_grad=True)
    pixels = helper.decode(FakeVAE(), latents)
    torch.testing.assert_close(pixels, torch.full_like(pixels, .5))
    pixels.sum().backward()
    assert latents.grad is not None and latents.grad.abs().sum() > 0
