"""Experimental still-image LoRA training for the pruned MiniMax-H3 ConvRot INT8 base."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


logger = logging.getLogger(__name__)


class MiniMaxH3ImageNetworkTrainer(NetworkTrainer):
    """Training-only H3 path: one still image, no audio, no previews, frozen INT8 base."""

    def __init__(self) -> None:
        super().__init__()
        self.vae_frame_stride = 1

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        if args.mixed_precision != "bf16":
            raise ValueError("Experimental MiniMax-H3 image training requires --mixed_precision bf16")
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MiniMax-H3 image training loads the pre-quantized ConvRot INT8 base directly; do not use FP8 flags")
        if args.sample_prompts:
            raise ValueError("Experimental MiniMax-H3 image training does not support in-training samples yet")
        if args.compile:
            raise ValueError("torch.compile is not enabled for the experimental ConvRot INT8 MiniMax-H3 path")
        if args.blocks_to_swap and not args.block_swap_h2d_only:
            raise ValueError("MiniMax-H3 ConvRot INT8 block swap requires --block_swap_h2d_only")
        if args.blocks_to_swap and not args.gradient_checkpointing:
            raise ValueError("MiniMax-H3 H2D-only block swap requires --gradient_checkpointing")
        if args.blocks_to_swap is None or args.blocks_to_swap <= 0:
            logger.warning(
                "MiniMax-H3 block swap is disabled. The ~21 GB pruned base leaves too little training headroom on a 24 GB GPU."
            )
        elif args.blocks_to_swap > 48:
            raise ValueError("MiniMax-H3 supports at most 48 swapped blocks out of 50")

    def _build_dataset(self, args):
        group, collator, current_epoch = super()._build_dataset(args)
        for index, dataset in enumerate(group.datasets):
            if int(dataset.batch_manager.batch_size) != 1:
                raise ValueError(
                    f"MiniMax-H3 image dataset {index} requires batch_size=1; use gradient accumulation for larger batches"
                )
        return group, collator, current_epoch

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        raise NotImplementedError("MiniMax-H3 experimental training has no sample generation")

    def do_inference(self, *args, **kwargs):
        raise NotImplementedError("MiniMax-H3 experimental training has no sample generation")

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        raise NotImplementedError("The VAE is used only by the MiniMax-H3 latent-cache command")

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = load_h3_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_mmap=args.disable_numpy_memmap,
            convrot_bwd_mode=args.convrot_bwd_mode,
        )
        if not model._has_convrot_int8:
            raise ValueError(
                "This experimental trainer accepts only the pruned ConvRot INT8 MiniMax-H3 checkpoint; "
                "it intentionally does not require or convert the ~66 GB BF16 transformer"
            )
        return model

    def compile_transformer(self, args, transformer):
        return transformer

    def scale_shift_latents(self, latents):
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if latents.ndim != 5 or tuple(latents.shape[1:3]) != (24, 1):
            raise ValueError(f"MiniMax-H3 image latents must be [B,24,1,H,W], got {tuple(latents.shape)}")
        if latents.shape[-2] % 2 or latents.shape[-1] % 2:
            raise ValueError("MiniMax-H3 cached latent height and width must be divisible by 2 (image buckets by 32 pixels)")
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        if latents.shape[0] != 1:
            raise ValueError("MiniMax-H3 image training requires batch_size=1")
        hidden_rows = batch.get("mmh3_hidden_states")
        if not isinstance(hidden_rows, list) or len(hidden_rows) != 1:
            raise ValueError("MiniMax-H3 text cache must provide one varlen_mmh3_hidden_states tensor")
        text_hidden_states = hidden_rows[0].unsqueeze(0).to(
            device=accelerator.device,
            dtype=network_dtype,
        )
        model_t = 1.0 - (timesteps.to(accelerator.device, dtype=torch.float32) - 1.0) / 1000.0
        with accelerator.autocast():
            prediction = transformer.forward_image(
                noisy_model_input.to(accelerator.device, dtype=network_dtype),
                model_t,
                text_hidden_states,
            )
        target = latents.to(accelerator.device, dtype=prediction.dtype) - noise.to(
            accelerator.device,
            dtype=prediction.dtype,
        )
        return DiTOutput(pred=prediction, target=target)

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_minimax_h3_training_mode": "experimental_image_only",
            "ss_minimax_h3_base_format": "pruned_int8_convrot",
            "ss_minimax_h3_convrot_bwd_mode": args.convrot_bwd_mode,
        }


def minimax_h3_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--convrot_bwd_mode",
        choices=("bf16", "int8"),
        default="bf16",
        help="ConvRot base-weight backward: bf16 is the safe default; int8 requires Triton and is more experimental",
    )
    parser.set_defaults(
        network_module="musubi_tuner.networks.lora_minimax_h3",
        timestep_sampling="krea2_shift",
        blocks_to_swap=30,
        block_swap_h2d_only=True,
        gradient_checkpointing=True,
        sdpa=True,
    )
    return parser


def main() -> None:
    parser = minimax_h3_image_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    MiniMaxH3ImageNetworkTrainer().train(args)


if __name__ == "__main__":
    main()
