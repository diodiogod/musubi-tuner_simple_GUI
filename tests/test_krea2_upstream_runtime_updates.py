from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.networks import lora_krea2


def _tiny_model():
    return SingleStreamDiT(
        SingleMMDiTConfig(
            features=32, tdim=32, txtdim=32, heads=2, multiplier=1, layers=2,
            patch=2, channels=4, bias=False, theta=1e3, kvheads=None,
            txtlayers=1, txtheads=2, txtkvheads=2,
        ),
        attn_mode="torch",
    )


def test_krea2_activation_cpu_offload_flag_is_not_silently_ignored():
    model = _tiny_model()
    model.enable_gradient_checkpointing(cpu_offload=True)
    assert model.gradient_checkpointing is True
    assert model.activation_cpu_offloading is True
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    assert model.activation_cpu_offloading is False


def test_krea2_parser_exposes_turbo_lora():
    import argparse

    parser = krea2_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args([])
    assert args.turbo_lora is None
    assert args.turbo_lora_multiplier == 1.0


class _CpuAccelerator:
    device = torch.device("cpu")


def test_turbo_lora_network_starts_frozen_and_disabled(tmp_path):
    model = _tiny_model()
    source = lora_krea2.create_arch_network(1.0, 4, 4, None, None, model)
    source.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    path = tmp_path / "turbo.safetensors"
    save_file(source.state_dict(), str(path))

    trainer = Krea2NetworkTrainer()
    args = SimpleNamespace(turbo_lora=str(path), turbo_lora_multiplier=1.0)
    network = trainer._build_turbo_lora_network(args, _CpuAccelerator(), model)

    assert network is trainer._turbo_lora_network
    assert all(not layer.enabled for layer in network.unet_loras)
    assert all(not parameter.requires_grad for parameter in network.parameters())
