import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.moss_image.models import MossImage
from musubi_tuner.moss_image.sampling import generate_latents, get_sampling_sigmas
from musubi_tuner.moss_image.sampling_adapter import load_sampling_adapter, sampling_adapter_enabled
from musubi_tuner.networks import lora_moss_image


def tiny_model():
    return MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=24, freq_dim=16, latent_channels=8)


def test_sampling_adapter_never_changes_training_weights_or_gradients(tmp_path):
    model = tiny_model().requires_grad_(False)
    trained = lora_moss_image.create_arch_network(1, 2, 2, None, None, model)
    trained.apply_to(None, model, apply_text_encoder=False)
    for module in trained.unet_loras:
        torch.nn.init.normal_(module.lora_up.weight)
    before = {key: value.clone() for key, value in trained.state_dict().items()}
    # Test the actual ComfyUI Turbo adapter naming, not only our saved format.
    extra_state = {key.replace("lora_unet_layers_", "diffusion_model.layers.", 1).replace("_proj_", ".proj.", 1): value.clone()
                   for key, value in before.items()}
    path = tmp_path / "turbo.safetensors"
    save_file(extra_state, path)
    x = torch.randn(1, 2, 32)
    original = model.layers[0].proj.q(x).detach()
    extra = load_sampling_adapter(path, 0.6, model, "cpu", torch.float32)
    torch.testing.assert_close(model.layers[0].proj.q(x), original)
    with pytest.raises(RuntimeError, match="sample failure"):
        with sampling_adapter_enabled(extra):
            assert not torch.allclose(model.layers[0].proj.q(x), original)
            raise RuntimeError("sample failure")
    result = model.layers[0].proj.q(x)
    torch.testing.assert_close(result, original)
    result.sum().backward()
    assert any(param.grad is not None for param in trained.parameters())
    assert all(param.grad is None and not param.requires_grad for param in extra.parameters())
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(value, before[key])
    frozen = {key: value.clone() for key, value in extra.state_dict().items()}
    torch.optim.SGD(trained.parameters(), lr=0.01).step()
    assert any(not torch.equal(value, before[key]) for key, value in trained.state_dict().items())
    for key, value in extra.state_dict().items():
        torch.testing.assert_close(value, frozen[key])


def test_bad_adapter_is_rejected_before_attaching(tmp_path):
    model = tiny_model()
    original = model.layers[0].proj.q.forward
    path = tmp_path / "wrong.safetensors"
    save_file({"lora_unet_not_a_moss_layer.lora_down.weight": torch.ones(2, 3)}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_sampling_adapter(path, 0.6, model, "cpu")
    assert model.layers[0].proj.q.forward == original


def test_simple_schedule_uses_comfy_noise_levels_and_ddim_updates():
    sigmas = get_sampling_sigmas(25, 5, scheduler="simple")
    # Comfy's 1000-entry grid at 25 steps visits 1.00, .96, ... .04.
    times = torch.arange(1000, 0, -40, dtype=torch.float32) / 1000
    expected = torch.cat([5 * times / (1 + 4 * times), torch.zeros(1)])
    torch.testing.assert_close(sigmas, expected)
    model = tiny_model().eval()
    embeds, mask = torch.randn(1, 2, 24), torch.ones(1, 2)
    kwargs = dict(dit=model, prompt_embeds=embeds, prompt_mask=mask, negative_embeds=None, negative_mask=None,
                  height=32, width=32, steps=2, guidance_scale=1, flow_shift=5, uncond_skip_layers=(),
                  seed=42, device=torch.device("cpu"), show_progress=False, scheduler="simple")
    torch.testing.assert_close(generate_latents(**kwargs, sampler="ddim"), generate_latents(**kwargs, sampler="euler"))


def test_standalone_preview_falls_back_from_missing_optional_attention(monkeypatch):
    from musubi_tuner.modules import attention
    from musubi_tuner.moss_image_generate import preview_attention_mode
    monkeypatch.setattr(attention, "flash_attn_varlen_func", None)
    monkeypatch.setattr(attention, "sageattn_varlen", None)
    monkeypatch.setattr(attention, "xops", None)
    assert preview_attention_mode("flash") == "torch"
    assert preview_attention_mode("sageattn") == "torch"
    assert preview_attention_mode("xformers") == "torch"
