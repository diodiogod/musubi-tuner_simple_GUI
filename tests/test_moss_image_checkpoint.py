import torch
from safetensors.torch import save_file

from musubi_tuner.moss_image.checkpoint import load_model
from musubi_tuner.moss_image.models import MossImage
from musubi_tuner.moss_image.sampling import get_sampling_sigmas, generate_latents
from musubi_tuner.moss_image.text_encoder import pad_text_embeds


def test_comfy_fused_checkpoint_matches_unfused_model(tmp_path):
    config = dict(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                  text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
    torch.manual_seed(11)
    original = MossImage(**config).eval()
    state = dict(original.state_dict())
    for index in range(2):
        prefix = f"layers.{index}.proj."
        state[prefix + "qkv.weight"] = torch.cat([state.pop(prefix + p + ".weight") for p in ("q", "k", "v")])
    state = {"model.diffusion_model." + key: value for key, value in state.items()}
    state["model.diffusion_model.layers.0.comfy_attention.config"] = torch.tensor([1], dtype=torch.uint8)
    path = tmp_path / "comfy.safetensors"
    save_file(state, str(path))
    loaded = load_model(path, config=config, dtype=torch.float32)
    x = torch.randn(1, 8, 2, 2)
    context = torch.randn(1, 3, 24)
    time = torch.tensor([450.0])
    with torch.no_grad():
        torch.testing.assert_close(loaded(x, time, context), original(x, time, context))
    assert not any(p.requires_grad for p in loaded.parameters())


def test_sampling_uses_training_grid():
    sigmas = get_sampling_sigmas(50, 5.0)
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])
    grid = torch.linspace(1, .001, 1000)
    assert all(torch.isclose(grid, sigma).any() for sigma in sigmas[:-1])


def test_text_mask_excludes_padding():
    a, b = torch.randn(3, 4), torch.randn(5, 4)
    values, mask = pad_text_embeds([a, b])
    torch.testing.assert_close(values[0, :3], a)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert not values[0, 3:].any()


def test_preview_sampler_is_seeded_and_uses_negative_skip_layer():
    config = dict(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                  text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
    torch.manual_seed(21)
    model = MossImage(**config).eval()
    positive, mask = pad_text_embeds([torch.randn(3, 24)])
    negative, neg_mask = pad_text_embeds([torch.randn(2, 24)])
    arguments = (model, positive, mask, negative, neg_mask, 32, 32, 3, 3.0, 5.0, (1,), 123, torch.device("cpu"))
    first = generate_latents(*arguments, show_progress=False)
    second = generate_latents(*arguments, show_progress=False)
    assert first.shape == (1, 8, 2, 2) and first.isfinite().all()
    torch.testing.assert_close(first, second)
