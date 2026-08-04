import pytest
import torch
from torch import nn

from musubi_tuner.minimax_h3.image_sampling import (
    build_image_sigma_schedule,
    decode_image_latent,
    initialize_image_noise,
    sample_image_latent,
)


class _ConstantVelocity:
    def __init__(self):
        self.timesteps = []

    def forward_image(self, latent, model_t, text):
        assert text.shape == (1, 2, 5120)
        self.timesteps.append(float(model_t))
        return torch.ones_like(latent)


class _RecordingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.seen = None

    def decode(self, latent):
        self.seen = latent
        return latent[:, :3]


def test_shifted_sigma_schedule_has_exact_endpoints_and_descends():
    sigmas = build_image_sigma_schedule(4, shift=12.0)
    assert sigmas.shape == (5,)
    assert sigmas[0] == 1
    assert sigmas[-1] == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_noise_is_seeded_on_cpu_and_has_image_latent_geometry():
    first = initialize_image_noise(768, 512, seed=123, device="cpu", dtype=torch.float32)
    second = initialize_image_noise(768, 512, seed=123, device="cpu", dtype=torch.float32)
    assert first.shape == (1, 24, 1, 32, 48)
    torch.testing.assert_close(first, second)


def test_sampler_integrates_native_clean_minus_noise_velocity():
    model = _ConstantVelocity()
    text = torch.zeros(2, 5120)
    initial = initialize_image_noise(32, 32, seed=9, device="cpu", dtype=torch.float32)
    sampled = sample_image_latent(
        model,
        text,
        width=32,
        height=32,
        steps=3,
        seed=9,
        shift=12.0,
        device="cpu",
        dtype=torch.float32,
    )
    torch.testing.assert_close(sampled, initial + 1.0)
    assert model.timesteps[0] == pytest.approx(0.0)
    assert model.timesteps[-1] < 1.0


def test_decode_duplicates_single_latent_and_returns_only_first_frame():
    vae = _RecordingVAE()
    latent = torch.zeros(1, 24, 1, 2, 3)
    pixels = decode_image_latent(vae, latent)
    assert vae.seen.shape == (1, 24, 2, 2, 3)
    assert pixels.shape == (1, 3, 1, 2, 3)
    torch.testing.assert_close(pixels, torch.full_like(pixels, 0.5))


def test_sampler_rejects_unaligned_image_size():
    with pytest.raises(ValueError, match="at least 32"):
        initialize_image_noise(16, 32, seed=1, device="cpu")
