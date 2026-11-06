"""MOSS Flux.2 VAE loading, including diffusers and small-decoder weights.

The key conversion follows kohya-ss/sd-scripts (Apache-2.0). The returned
model's encode/decode operate on MOSS's normalized, packed latent format.
"""
import torch
from accelerate import init_empty_weights
from torch import Tensor

from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams, Decoder
from musubi_tuner.utils.safetensors_utils import load_split_weights

def _convert_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a diffusers AutoencoderKLFlux2 state dict to the original (BFL) key names."""
    new_sd = {}
    num_levels = len({k.split(".")[2] for k in sd if k.startswith("decoder.up_blocks.")})

    def convert_resnet(key: str) -> str:
        return key.replace("conv_shortcut.", "nin_shortcut.")

    for key, value in sd.items():
        new_key = key
        if key.startswith("quant_conv."):
            new_key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            new_key = "decoder." + key
        elif ".mid_block." in key:
            prefix, rest = key.split(".mid_block.", 1)
            if rest.startswith("resnets."):
                _, idx, rest2 = rest.split(".", 2)
                new_key = f"{prefix}.mid.block_{int(idx) + 1}.{convert_resnet(rest2)}"
            elif rest.startswith("attentions."):
                _, idx, rest2 = rest.split(".", 2)
                mapping = {
                    "group_norm": "norm",
                    "to_q": "q",
                    "to_k": "k",
                    "to_v": "v",
                    "to_out.0": "proj_out",
                }
                for src, dst in mapping.items():
                    if rest2.startswith(src + "."):
                        rest2 = dst + rest2[len(src) :]
                        break
                new_key = f"{prefix}.mid.attn_1.{rest2}"
                if rest2.endswith(".weight") and value.ndim == 2:
                    value = value[:, :, None, None]  # Linear -> 1x1 Conv2d
        elif key.startswith("encoder.down_blocks."):
            _, _, idx, rest = key.split(".", 3)
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("downsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{idx}.downsample.{rest2}"
        elif key.startswith("decoder.up_blocks."):
            _, _, idx, rest = key.split(".", 3)
            level = num_levels - 1 - int(idx)  # diffusers up_blocks.0 is the deepest level (BFL up.{n-1})
            if rest.startswith("resnets."):
                _, bidx, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.block.{bidx}.{convert_resnet(rest2)}"
            elif rest.startswith("upsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.upsample.{rest2}"
        elif key.startswith("encoder.conv_norm_out.") or key.startswith("decoder.conv_norm_out."):
            new_key = key.replace(".conv_norm_out.", ".norm_out.")
        new_sd[new_key] = value
    return new_sd


def is_diffusers_vae_state_dict(sd: dict[str, Tensor]) -> bool:
    return any(k.startswith("encoder.down_blocks.") for k in sd)


def load_vae(path, dtype=torch.float32, device="cpu", disable_mmap=False, params=None):
    sd = load_split_weights(str(path), device=str(device), disable_mmap=disable_mmap)
    for prefix in ("vae.", "first_stage_model."):
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    if is_diffusers_vae_state_dict(sd):
        sd = _convert_diffusers_vae_state_dict(sd)
    params = params or AutoEncoderParams()
    if "decoder.conv_out.weight" not in sd:
        raise ValueError("The selected file is not a complete Flux.2 VAE.")
    decoder_ch = sd["decoder.conv_out.weight"].shape[1]
    with init_empty_weights():
        model = AutoEncoder(params)
        if decoder_ch != params.ch:
            model.decoder = Decoder(
                resolution=params.resolution, in_channels=params.in_channels, ch=decoder_ch,
                out_ch=params.out_ch, ch_mult=params.ch_mult, num_res_blocks=params.num_res_blocks,
                z_channels=params.z_channels,
            )
    for k, v in sd.items():
        if v.is_floating_point():
            sd[k] = v.to(dtype=dtype)
    model.load_state_dict(sd, strict=True, assign=True)
    model.to(device).requires_grad_(False).eval()
    return model
