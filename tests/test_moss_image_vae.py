import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.flux_2.flux2_models import AutoEncoderParams
from musubi_tuner.moss_image.vae import load_vae


def test_diffusers_small_decoder_conversion_matches_reference(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(42)
    reference = diffusers.AutoencoderKLFlux2(
        in_channels=3, out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[64, 128], decoder_block_out_channels=[32, 64],
        layers_per_block=2, latent_channels=4, norm_num_groups=32,
        sample_size=32, mid_block_add_attention=True, use_quant_conv=True,
        use_post_quant_conv=True, batch_norm_eps=1e-4, patch_size=[2, 2], force_upcast=True,
    ).eval()
    path = tmp_path / "diffusers.safetensors"
    save_file(reference.state_dict(), str(path))
    params = AutoEncoderParams(ch=64, ch_mult=[1, 2], z_channels=4)
    vae = load_vae(path, params=params)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        raw = reference.encode(pixels, return_dict=False)[0].mode()
        packed = vae.encode(pixels)
        expected_packed = torch.nn.functional.pixel_unshuffle(raw, 2)
        expected_packed = (expected_packed - reference.bn.running_mean[None, :, None, None]) / torch.sqrt(
            reference.bn.running_var[None, :, None, None] + reference.bn.eps
        )
        torch.testing.assert_close(packed, expected_packed, atol=1e-5, rtol=1e-5)
        expected_pixels = reference.decode(raw, return_dict=False)[0]
        torch.testing.assert_close(vae.decode(packed), expected_pixels, atol=1e-5, rtol=1e-5)
