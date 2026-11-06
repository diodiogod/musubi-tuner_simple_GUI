import torch
from musubi_tuner.face_refinement.draft_moss_image import generate_differentiable_moss


def test_moss_draft_backpropagates_through_the_last_step():
    class Model(torch.nn.Module):
        latent_channels, num_layers = 4, 42
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            self.calls = []
        @property
        def device(self):
            return self.weight.device
        @property
        def dtype(self):
            return self.weight.dtype
        def prepare_block_swap_before_forward(self):
            pass
        def forward(self, x, time, text, mask, skip_layers=()):
            self.calls.append((torch.is_grad_enabled(), tuple(skip_layers)))
            return x * self.weight + text.mean() * self.weight

    class VAE:
        def decode(self, latent):
            return latent[:, :3]

    model = Model()
    text, mask = torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.int32)
    pixels = generate_differentiable_moss(model, VAE(), text, mask, text * 0, mask,
                                          resolution=32, denoise_steps=4, draft_k=1,
                                          cfg_scale=3, seed=4, checkpoint_vae=True)
    pixels.sum().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)
    assert model.weight.grad.abs() > 0
    assert sum(enabled for enabled, _ in model.calls) == 1
    assert all(not enabled for enabled, skip in model.calls if skip == (22,))
