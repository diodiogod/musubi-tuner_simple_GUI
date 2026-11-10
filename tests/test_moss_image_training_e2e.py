"""One real Musubi training step on CPU with a tiny MOSS transformer and real cache files."""
import os
import subprocess
import sys
from pathlib import Path

import torch
import pytest
from PIL import Image
from safetensors import safe_open

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.moss_image.cache import save_embeddings, save_latents
from musubi_tuner.training.dop import dop_signature


@pytest.mark.parametrize("repa,ema,sigmoid_high,optimizer", [(False, False, False, None), (True, False, False, None),
    (False, True, False, None), (True, True, False, None), (True, True, True, None),
    (True, True, False, "Muon"), (False, True, False, "Automagic4")])
def test_one_cpu_training_step_consumes_real_moss_cache_and_saves_adapter(tmp_path, repa, ema, sigmoid_high, optimizer):
    Image.new("RGB", (32, 32)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("a person_token portrait", encoding="utf-8")
    dataset = ImageDataset(resolution=(32, 32), caption_extension=".txt", batch_size=1,
                           num_repeats=1, enable_bucket=False, bucket_no_upscale=False,
                           image_directory=str(tmp_path), cache_directory=str(tmp_path),
                           architecture=ARCHITECTURE_MOSS_IMAGE)
    item = ItemInfo(str(tmp_path / "a.png"), "a person_token portrait", (32, 32))
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
    save_latents(item, torch.randn(128, 1, 2, 2))
    save_embeddings(item, torch.randn(3, 4096), torch.randn(2, 4096), dop_signature("person_token", "a person"))
    dataset_file = tmp_path / "dataset.toml"
    folder = str(tmp_path).replace("\\", "/")
    dataset_file.write_text(f'[[datasets]]\nresolution = [32, 32]\nbatch_size = 1\nnum_repeats = {2 if ema else 1}\nimage_directory = "{folder}"\n', encoding="utf-8")
    dit = tmp_path / "tiny-dummy.safetensors"
    dit.touch()
    script = """
import sys, torch
from musubi_tuner.moss_image.models import MossImage
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common
torch.set_num_threads(2)
trainer=MossImageNetworkTrainer()
def load_transformer(*a, **k):
    if args.moss_repa_enabled:
        assert trainer._repa_targets.paths, 'targets must be ready before loading the DiT'
        assert trainer._repa_targets.encoder is None, 'teacher must be released before loading the DiT'
        def forbidden_decode(*a, **k):
            raise AssertionError('REPA must not encode a target during training')
        trainer._repa_targets._encode = forbidden_decode
    return MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=4096, freq_dim=16, latent_channels=128)
trainer.load_transformer=load_transformer
class TinyVAE(torch.nn.Module):
    def decode(self, x):
        return torch.nn.functional.interpolate(x[:, :3], size=(28, 28))
trainer.load_vae=lambda *a, **k: TinyVAE()
args=moss_setup_parser(setup_parser_common()).parse_args(sys.argv[1:])
args.dit_dtype='float32'
args.vae_dtype='float32'
trainer.train(args)
"""
    cmd = [sys.executable, "-c", script, "--dataset_config", str(dataset_file), "--dit", str(dit),
           "--output_dir", str(tmp_path / "out"), "--output_name", "tiny", "--network_module", "networks.lora_moss_image",
           "--network_dim", "2", "--network_alpha", "2", "--max_train_steps", "1", "--learning_rate", "0.0001",
           "--optimizer_type", "AdamW", "--max_data_loader_n_workers", "0", "--sdpa", "--mixed_precision", "no",
           "--moss_pretraining_schedule", "--gradient_checkpointing",
           "--dop_trigger_word", "person_token", "--dop_class_word", "a person", "--dop_loss_weight", "0.1",
           "--save_state"]
    if sigmoid_high:
        cmd[cmd.index("--moss_pretraining_schedule")] = "--moss_sigmoid_high"
    if ema:
        cmd[cmd.index("--optimizer_type") + 1] = "Automagic3"
        cmd[cmd.index("--max_train_steps") + 1] = "2"
        cmd += ["--moss_ema_enabled", "--moss_ema_decay", "0.98", "--gradient_accumulation_steps", "2",
                "--optimizer_args", "min_lr=1e-6", "max_lr=0.001", "fused=False"]
    if repa:
        from transformers import Dinov2Config, Dinov2Model
        encoder_path = tmp_path / "tiny-dino"
        Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32, image_size=28, patch_size=14)).save_pretrained(encoder_path)
        cmd += ["--moss_repa_enabled", "--moss_repa_block", "0", "--moss_repa_image_size", "28",
                "--moss_repa_encoder", str(encoder_path), "--vae", str(dit)]
    if optimizer:
        cmd[cmd.index("--optimizer_type") + 1] = optimizer
        position = cmd.index("--optimizer_args")
        del cmd[position:position + 4]  # remove the Automagic3-specific bounds and fused flag
        cmd += ["--moss_unpadded_batches", "--moss_clean_end_share", "40",
                "--network_args", "moss_lowp_lora=True"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", ACCELERATE_USE_CPU="true", PYTHONIOENCODING="utf-8")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    artifact = tmp_path / "out" / "tiny.safetensors"
    assert artifact.is_file(), result.stdout + "\n" + result.stderr
    with safe_open(str(artifact), framework="pt", device="cpu") as handle:
        assert any(key.startswith("lora_unet_layers_0_proj_q.") for key in handle.keys())
        assert not any("projector" in key or "repa" in key for key in handle.keys())
        if repa:
            assert handle.metadata()["ss_moss_repa_enabled"] == "True"
        assert handle.metadata()["ss_moss_sigmoid_high"] == str(sigmoid_high)
    state = tmp_path / "out" / "tiny-state"
    assert state.is_dir(), result.stdout + "\n" + result.stderr
    if ema:
        from safetensors.torch import load_file
        snapshots = [torch.load(path, map_location="cpu", weights_only=True)
                     for path in state.glob("custom_checkpoint_*.pkl")]
        saved_ema = next(value for value in snapshots if value.get("kind") == "moss-adapter-ema-v1")
        assert saved_ema["num_updates"] == 2  # Four microbatches, two real optimizer steps.
        assert (state / "moss_ema.json").is_file()
        averaged = load_file(str(state / "ema.safetensors"))
        live = load_file(str(state / "model.safetensors"))
        exported = load_file(str(artifact))
        for name, value in saved_ema["shadow"].items():
            torch.testing.assert_close(averaged[name], value, rtol=0, atol=0)
            torch.testing.assert_close(exported[name].float(), value, rtol=0.01, atol=1e-7)
        assert any(not torch.equal(live[name], value) for name, value in saved_ema["shadow"].items())
    if repa:
        assert (state / "custom_checkpoint_0.pkl").is_file()
        feature_files = list(tmp_path.glob("moss_repa_features/**/*.safetensors"))
        assert len(feature_files) == 1
        feature_mtime = feature_files[0].stat().st_mtime_ns
    resumed = subprocess.run(cmd + ["--resume", str(state), "--output_name", "resumed"], env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert resumed.returncode == 0, resumed.stdout + "\n" + resumed.stderr
    assert (tmp_path / "out" / "resumed.safetensors").is_file()
    if ema:
        snapshots = [torch.load(path, map_location="cpu", weights_only=True)
                     for path in (tmp_path / "out" / "resumed-state").glob("custom_checkpoint_*.pkl")]
        assert next(value for value in snapshots if value.get("kind") == "moss-adapter-ema-v1")["num_updates"] == 4
    if repa:
        assert feature_files[0].stat().st_mtime_ns == feature_mtime
        assert "0 generated, 1 reused" in resumed.stderr
