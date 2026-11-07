from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import Dinov2Config, Dinov2Model

from musubi_tuner.moss_image.repa import MossRepa
from musubi_tuner.moss_image.repa_cache import MossRepaTargets, latent_key
from musubi_tuner.moss_image_train_network import moss_setup_parser, setup_parser_common


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decodes = self.moves = 0

    def to(self, *args, **kwargs):
        self.moves += 1
        return super().to(*args, **kwargs)

    def decode(self, latents):
        self.decodes += 1
        return torch.nn.functional.interpolate(latents[:, :3], size=(28, 28))


def setup(tmp_path, cache_size=1):
    vae_path = tmp_path / "vae.safetensors"
    vae_path.touch()
    encoder = Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                      intermediate_size=32, image_size=28, patch_size=14))
    encoder_path = tmp_path / "teacher"
    encoder.save_pretrained(encoder_path)
    args = moss_setup_parser(setup_parser_common()).parse_args([
        "--moss_repa_enabled", "--moss_repa_image_size", "28", "--moss_repa_block", "0",
        "--moss_repa_cache_size", str(cache_size), "--moss_repa_encoder", str(encoder_path),
        "--vae", str(vae_path),
    ])
    latents = [torch.randn(128, 1, 2, 3), torch.randn(128, 1, 2, 3)]
    paths = []
    for i, latent in enumerate(latents):
        path = tmp_path / f"latent-{i}.safetensors"
        save_file({"latents_1x2x3_float32": latent}, path)
        paths.append(path)
    items = [SimpleNamespace(latent_cache_path=path) for path in paths]
    dataset = SimpleNamespace(cache_directory=str(tmp_path),
                              batch_manager=SimpleNamespace(buckets={"rectangular": items * 3}))
    return args, SimpleNamespace(datasets=[dataset]), latents


def test_disk_cache_survives_ram_eviction_and_restart_without_decoder_or_teacher(tmp_path):
    args, group, latents = setup(tmp_path)
    vae = TinyVAE()
    online = MossRepaTargets(args, "cpu")
    expected = [online.targets(vae, latent.unsqueeze(0)) for latent in latents]
    first = MossRepaTargets(args, "cpu")
    rng = torch.random.get_rng_state().clone()
    result = first.prepare(group, vae)
    assert result == {"generated": 2, "reused": 0}  # six repeated items
    assert torch.equal(rng, torch.random.get_rng_state())
    assert first.encoder is None and len(first.cache) == 1
    assert len(list(tmp_path.rglob("moss_repa_features/**/*.safetensors"))) == 2
    args.moss_repa_cache_size = 0  # disk reuse is independent of RAM capacity
    restart = MossRepaTargets(args, "cpu")
    with patch.object(vae, "decode", side_effect=AssertionError("warm preparation decoded an image")), \
         patch("musubi_tuner.moss_image.repa_cache.load_encoder", side_effect=AssertionError("loaded DINO")):
        assert restart.prepare(group, vae) == {"generated": 0, "reused": 2}
    vae.moves = 0
    with patch.object(vae, "decode", side_effect=AssertionError("decoded during training")), \
         patch("musubi_tuner.moss_image.repa_cache.load_encoder", side_effect=AssertionError("loaded DINO")):
        for latent, target in zip(latents, expected):
            actual = restart.targets(vae, latent.unsqueeze(0))
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert not restart.cache and vae.moves == 0  # preserves depth residency


def test_changed_latent_and_vae_invalidate_saved_targets(tmp_path):
    args, group, latents = setup(tmp_path)
    first = MossRepaTargets(args, "cpu")
    first.prepare(group, TinyVAE())
    path = group.datasets[0].batch_manager.buckets["rectangular"][0].latent_cache_path
    save_file({"latents_1x2x3_float32": latents[0] + 1}, path)
    changed = MossRepaTargets(args, "cpu")
    assert changed.prepare(group, TinyVAE()) == {"generated": 1, "reused": 1}
    first_signature = changed.signature
    (tmp_path / "vae.safetensors").write_bytes(b"different VAE checkpoint")
    changed_vae = MossRepaTargets(args, "cpu")
    assert changed_vae.prepare(group, TinyVAE()) == {"generated": 2, "reused": 0}
    assert changed_vae.signature != first_signature


def test_changed_teacher_weights_invalidate_saved_targets(tmp_path):
    args, group, _ = setup(tmp_path)
    before = MossRepaTargets(args, "cpu")
    before.prepare(group, TinyVAE())
    teacher_path = tmp_path / "teacher"
    teacher = Dinov2Model.from_pretrained(teacher_path)
    with torch.no_grad():
        next(teacher.parameters()).add_(1)
    teacher.save_pretrained(teacher_path)
    after = MossRepaTargets(args, "cpu")
    assert after.prepare(group, TinyVAE()) == {"generated": 2, "reused": 0}
    assert before.signature != after.signature


def test_damaged_features_rebuild_before_training_and_fail_clearly_mid_run(tmp_path):
    args, group, latents = setup(tmp_path, cache_size=0)
    first = MossRepaTargets(args, "cpu")
    first.prepare(group, TinyVAE())
    path = first.paths[latent_key(latents[0])]
    path.write_bytes(b"interrupted write")
    with pytest.raises(RuntimeError, match="Restart the run"):
        first.targets(TinyVAE(), latents[0].unsqueeze(0))
    repaired = MossRepaTargets(args, "cpu")
    assert repaired.prepare(group, TinyVAE()) == {"generated": 1, "reused": 1}
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_location_is_optional_and_projector_uses_prepared_features(tmp_path):
    args, group, latents = setup(tmp_path)
    args.moss_repa_cache_dir = str(tmp_path / "shared features")
    targets = MossRepaTargets(args, "cpu")
    targets.prepare(group, TinyVAE())
    assert all(path.is_relative_to(tmp_path / "shared features") for path in targets.paths.values())
    repa = MossRepa(args, 32, "cpu", feature_cache=targets)
    features = repa.targets(None, latents[0].unsqueeze(0))
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    loss, _ = repa.loss(hidden, features, 2, 3)
    loss.backward()
    assert hidden.grad.abs().sum() > 0 and repa.projector[1].weight.grad.abs().sum() > 0


def test_cache_in_local_teacher_folder_does_not_invalidate_itself(tmp_path):
    args, group, _ = setup(tmp_path)
    args.moss_repa_cache_dir = args.moss_repa_encoder
    first = MossRepaTargets(args, "cpu")
    assert first.prepare(group, TinyVAE())["generated"] == 2
    second = MossRepaTargets(args, "cpu")
    assert second.prepare(group, TinyVAE()) == {"generated": 0, "reused": 2}
