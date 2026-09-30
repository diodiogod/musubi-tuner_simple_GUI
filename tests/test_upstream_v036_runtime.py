import importlib
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize("family", ["minimax_h3", "minimax_h3_native"])
def test_encoder_only_loader_skips_decoder_reads_and_keeps_moments(tmp_path, monkeypatch, family):
    module = importlib.import_module(f"musubi_tuner.{family}.video_vae")
    original = module.MiniMaxH3VideoVAE
    monkeypatch.setattr(module, "ViT3DDecoder", lambda **kwargs: torch.nn.Conv3d(24, 3, 1))
    monkeypatch.setattr(module, "MiniMaxH3VideoVAE", lambda: original(
        ch=32, ch_mult=(1,), space_down=(1,), time_down=(1,), num_res_blocks=1,
        clip_length=1, token_drop=0, tiling=False,
    ))
    model = module.MiniMaxH3VideoVAE().eval()
    checkpoint = tmp_path / "vae.safetensors"
    save_file({"vae." + key: value for key, value in model.state_dict().items()}, checkpoint)
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
    reads = []
    get_tensor = MemoryEfficientSafeOpen.get_tensor
    def tracked(self, key, **kwargs):
        reads.append(key)
        return get_tensor(self, key, **kwargs)
    monkeypatch.setattr(MemoryEfficientSafeOpen, "get_tensor", tracked)
    encoder = module.load_video_vae(checkpoint, dtype=torch.float32, load_decoder=False)
    assert not any(key.startswith(("vae.decoder.", "vae.post_quant_conv.")) for key in reads)
    assert encoder.decoder is None
    assert all(parameter.device.type != "meta" for parameter in encoder.parameters())
    pixels = torch.randn(1, 3, 1, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(model.encode_moments(pixels), encoder.encode_moments(pixels), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="encoder-only"):
        encoder.decode(torch.zeros(1, 24, 1, 4, 4))


@pytest.mark.parametrize("shape", [(2, 3, 4, 8), (2, 3, 8), (2, 8)])
def test_rank_dropout_on_linear_supports_leading_dimensions(shape):
    from musubi_tuner.networks.lora import LoRAModule
    linear = torch.nn.Linear(8, 12)
    adapter = LoRAModule("test", linear, lora_dim=4, rank_dropout=0.5)
    adapter.apply_to()
    result = linear(torch.randn(shape))
    assert result.shape == (*shape[:-1], 12)
    result.sum().backward()
    assert adapter.lora_up.weight.grad is not None


def test_helper_presets_preserve_explicit_old_path():
    from backends.minimax_h3 import assistant_source
    old = "my/old/v1.safetensors"
    assert assistant_source({"minimax_h3_training_assistant": old}) == old
    assert assistant_source({"minimax_h3_training_assistant_preset": "Ref2VA v2"}).endswith("ref2va_training_adapter_v2.safetensors")


def test_native_cache_commands_forward_media_tolerances():
    from backends.minimax_h3 import build_cache_commands
    settings = {
        "minimax_h3_training_workflow": "Video + audio · official multimodal",
        "recache_latents": True, "dataset_config": "data.toml",
        "minimax_h3_video_vae": "video.safetensors", "minimax_h3_audio_vae": "audio.safetensors",
        "minimax_h3_cache_skip_broken": True,
        "minimax_h3_audio_max_discontinuity": "0.08", "minimax_h3_audio_max_missing": "0.3",
    }
    command = build_cache_commands(settings, "python")[0]
    assert "--skip_broken" in command
    assert command[command.index("--audio_max_discontinuity") + 1] == "0.08"
    assert command[command.index("--audio_max_missing") + 1] == "0.3"
    settings["minimax_h3_audio_max_missing"] = "nan"
    with pytest.raises(ValueError, match="nonnegative"):
        build_cache_commands(settings, "python")
